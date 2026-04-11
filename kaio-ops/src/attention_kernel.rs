//! Single-head attention: standard (materialized) and FlashAttention.
//!
//! Computes: out = softmax(Q * K^T / sqrt(d_k)) * V
//!
//! Three-kernel decomposition:
//! 1. `qk_scaled_matmul`: S = Q * K^T / sqrt(d_k) — naive 16×16 tiled matmul
//!    with transposed K indexing
//! 2. `row_softmax`: P = softmax(S) — row-wise, one block per row
//! 3. `matmul()`: out = P * V — reuses existing kaio-ops matmul
//!
//! This is the correctness baseline for FlashAttention (Sprint 5.4).
//! Materializes O(seq_len^2) intermediate buffers (scores + probs).
//! Intended for modest sequence lengths (seq_len <= 512).
//!
//! # Layout
//!
//! All matrices are f32, row-major, contiguous:
//! - Q: (seq_len, d_k)
//! - K: (seq_len, d_k) — NOT pre-transposed
//! - V: (seq_len, d_k) — d_v == d_k for now
//! - out: (seq_len, d_k)

use kaio::prelude::*;

use crate::matmul;

// ---------------------------------------------------------------------------
// Kernel 1: Q * K^T / sqrt(d_k)
// ---------------------------------------------------------------------------
// Adapted from naive 16×16 tiled matmul. K is stored as (seq_len, d_k)
// row-major but accessed as K^T via transposed indexing:
//   K^T[d, j] = K[j, d] = K_data[j * d_k + d]
// In standard matmul: tile_b loads B[inner_row * N + col].
// For K^T: tile_b loads K[col_global * d_k + inner + ty].

#[allow(clippy::too_many_arguments)]
#[gpu_kernel(block_size = (16, 16))]
fn qk_scaled_matmul(q: &[f32], k: &[f32], s: &mut [f32], seq_len: u32, d_k: u32, inv_sqrt_dk: f32) {
    let tx = thread_idx_x();
    let ty = thread_idx_y();
    let row = block_idx_y() * 16 + ty; // query position
    let col = block_idx_x() * 16 + tx; // key position

    let tile_q = shared_mem![f32; 256]; // 16×16
    let tile_k = shared_mem![f32; 256]; // 16×16

    let mut acc = 0.0f32;
    let num_tiles = (d_k + 15) / 16;

    let mut t = 0u32;
    while t < num_tiles {
        // Load Q tile: Q[row, t*16 + tx]
        let q_col = t * 16 + tx;
        tile_q[ty * 16 + tx] = 0.0f32;
        if row < seq_len {
            if q_col < d_k {
                tile_q[ty * 16 + tx] = q[row * d_k + q_col];
            }
        }

        // Load K^T tile: K^T[t*16 + ty, col] = K[col, t*16 + ty]
        // K is (seq_len, d_k) row-major, so K[col, d] = k[col * d_k + d]
        let k_d = t * 16 + ty;
        tile_k[ty * 16 + tx] = 0.0f32;
        if col < seq_len {
            if k_d < d_k {
                tile_k[ty * 16 + tx] = k[col * d_k + k_d];
            }
        }

        bar_sync();

        // Accumulate: S[row, col] += Q_tile[ty, i] * K^T_tile[i, tx]
        let mut i = 0u32;
        while i < 16 {
            acc = fma(tile_q[ty * 16 + i], tile_k[i * 16 + tx], acc);
            i += 1;
        }

        bar_sync();
        t += 1;
    }

    // Write scaled result
    if row < seq_len {
        if col < seq_len {
            s[row * seq_len + col] = acc * inv_sqrt_dk;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: Row-wise softmax
// ---------------------------------------------------------------------------
// Adapted from Phase 3 softmax_row kernel. One block per row.
// Block uses strided loops to handle rows longer than block_size.
// Uses block_size = (256, 1) so it takes an explicit grid tuple —
// 1D auto-grid would infer ceil(row_len/256) blocks, but we need
// num_rows blocks (one per row).

#[gpu_kernel(block_size = (256, 1))]
fn row_softmax(input: &[f32], output: &mut [f32], row_len: u32) {
    let tid = thread_idx_x();
    let bsize = 256u32;
    let row_offset = block_idx_x() * row_len;

    // Pass 1: find row max
    let mut local_max = -3.402823e+38f32;
    let mut i1 = tid;
    while i1 < row_len {
        let val = input[row_offset + i1];
        if val > local_max {
            local_max = val;
        }
        i1 += bsize;
    }
    let row_max = block_reduce_max(local_max);

    // Pass 2: compute exp(x - max) and sum
    let mut local_sum = 0.0f32;
    let mut i2 = tid;
    while i2 < row_len {
        local_sum = local_sum + exp(input[row_offset + i2] - row_max);
        i2 += bsize;
    }
    let row_sum = block_reduce_sum(local_sum);

    // Pass 3: normalize
    let mut i3 = tid;
    while i3 < row_len {
        output[row_offset + i3] = exp(input[row_offset + i3] - row_max) / row_sum;
        i3 += bsize;
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: Causal mask
// ---------------------------------------------------------------------------
// Sets S[i,j] = -FLT_MAX where j > i (future positions).
// Softmax then zeros these positions: exp(-3.4e38 - max) ≈ 0.
// Separate kernel for composability — unmasked attention unchanged.

#[gpu_kernel(block_size = (16, 16))]
fn apply_causal_mask(s: &mut [f32], seq_len: u32) {
    let row = block_idx_y() * 16 + thread_idx_y();
    let col = block_idx_x() * 16 + thread_idx_x();
    if row < seq_len {
        if col < seq_len {
            if col > row {
                // -FLT_MAX, not -inf: DSL has no f32::NEG_INFINITY.
                // exp(-3.4e38 - max) ≈ 0 regardless of max, so softmax
                // zeros these positions. Do not "fix" to -inf.
                s[row * seq_len + col] = -3.402823e+38f32;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute single-head scaled dot-product attention.
///
/// out = softmax(Q * K^T / sqrt(d_k)) * V
///
/// Standard (materialized) implementation — allocates O(seq_len^2)
/// intermediate buffers. For correctness validation, not production
/// efficiency. Intended for seq_len <= 512.
///
/// All f32, row-major, contiguous. d_v == d_k.
///
/// # Errors
///
/// Returns `KaioError::InvalidConfig` if any dimension is zero or
/// buffers are too small.
///
/// # Example
///
/// ```ignore
/// use kaio::prelude::*;
/// use kaio_ops::attention;
///
/// let device = KaioDevice::new(0)?;
/// let q = device.alloc_from(&q_data)?;
/// let k = device.alloc_from(&k_data)?;
/// let v = device.alloc_from(&v_data)?;
/// let mut out = device.alloc_zeros::<f32>(seq_len * d_k)?;
/// attention(&device, &q, &k, &v, &mut out, seq_len as u32, d_k as u32)?;
/// ```
pub fn attention(
    device: &KaioDevice,
    q: &GpuBuffer<f32>,
    k: &GpuBuffer<f32>,
    v: &GpuBuffer<f32>,
    out: &mut GpuBuffer<f32>,
    seq_len: u32,
    d_k: u32,
) -> Result<()> {
    validate_attention_dims(q, k, v, out, seq_len, d_k)?;

    let inv_sqrt_dk = 1.0f32 / (d_k as f32).sqrt();

    // Step 1: S = Q * K^T / sqrt(d_k)
    let mut scores = device.alloc_zeros::<f32>((seq_len as usize) * (seq_len as usize))?;
    let grid_qk = (seq_len.div_ceil(16), seq_len.div_ceil(16), 1);
    qk_scaled_matmul::launch(
        device,
        q,
        k,
        &mut scores,
        seq_len,
        d_k,
        inv_sqrt_dk,
        grid_qk,
    )?;

    // Step 2: P = softmax(S) — row-wise, one block per row
    let mut probs = device.alloc_zeros::<f32>((seq_len as usize) * (seq_len as usize))?;
    let grid_sm = (seq_len, 1, 1);
    row_softmax::launch(device, &scores, &mut probs, seq_len, grid_sm)?;

    // Step 3: out = P * V — standard matmul
    // P is (seq_len × seq_len), V is (seq_len × d_k), out is (seq_len × d_k)
    matmul(device, &probs, v, out, seq_len, d_k, seq_len)?;

    Ok(())
}

/// Compute single-head scaled dot-product attention with causal mask.
///
/// out = softmax(causal_mask(Q * K^T / sqrt(d_k))) * V
///
/// Causal mask sets S[i,j] = -FLT_MAX where j > i, preventing
/// attention to future positions. Standard for autoregressive models.
///
/// Same constraints as [`attention()`]: f32, row-major, d_v == d_k,
/// O(seq_len^2) intermediate buffers.
pub fn attention_causal(
    device: &KaioDevice,
    q: &GpuBuffer<f32>,
    k: &GpuBuffer<f32>,
    v: &GpuBuffer<f32>,
    out: &mut GpuBuffer<f32>,
    seq_len: u32,
    d_k: u32,
) -> Result<()> {
    validate_attention_dims(q, k, v, out, seq_len, d_k)?;

    let inv_sqrt_dk = 1.0f32 / (d_k as f32).sqrt();

    // Step 1: S = Q * K^T / sqrt(d_k)
    let mut scores = device.alloc_zeros::<f32>((seq_len as usize) * (seq_len as usize))?;
    let grid_qk = (seq_len.div_ceil(16), seq_len.div_ceil(16), 1);
    qk_scaled_matmul::launch(
        device,
        q,
        k,
        &mut scores,
        seq_len,
        d_k,
        inv_sqrt_dk,
        grid_qk,
    )?;

    // Step 1.5: Apply causal mask — S[i,j] = -FLT_MAX where j > i
    apply_causal_mask::launch(device, &mut scores, seq_len, grid_qk)?;

    // Step 2: P = softmax(S) — row-wise, one block per row
    let mut probs = device.alloc_zeros::<f32>((seq_len as usize) * (seq_len as usize))?;
    let grid_sm = (seq_len, 1, 1);
    row_softmax::launch(device, &scores, &mut probs, seq_len, grid_sm)?;

    // Step 3: out = P * V — standard matmul
    matmul(device, &probs, v, out, seq_len, d_k, seq_len)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// FlashAttention kernels (Sprint 5.4)
// ---------------------------------------------------------------------------
// BLOCK_M = 1: one query position per block, 256 threads.
// No materialized attention matrix — O(d_k + 256) memory per block.
// Online softmax: running (m, l, O) updated per K/V tile.
//
// Assumption: every query row has at least one valid key (causal
// self-attention guarantees the diagonal). If all keys are masked,
// l = 0 and output is undefined (divide by zero).

#[allow(clippy::too_many_arguments)]
#[gpu_kernel(block_size = (256, 1))]
fn flash_attn_kernel(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    seq_len: u32,
    d_k: u32,
    inv_sqrt_dk: f32,
) {
    let tid = thread_idx_x();
    let q_row = block_idx_x();
    let q_base = q_row * d_k;

    let tile = shared_mem![f32; 256];

    // Per-thread output accumulator (tid < d_k handles output dim tid)
    let mut o_acc = 0.0f32;

    // Running softmax state (block-wide, per-thread register copies)
    let mut m = -3.402823e+38f32;
    let mut l = 0.0f32;

    let mut kv_start = 0u32;
    while kv_start < seq_len {
        // Phase 1: Each thread computes one attention score
        let j = kv_start + tid;
        let mut score = -3.402823e+38f32;
        if j < seq_len {
            score = 0.0f32;
            let mut d = 0u32;
            while d < d_k {
                score = fma(q[q_base + d], k[j * d_k + d], score);
                d += 1;
            }
            score = score * inv_sqrt_dk;
        }
        tile[tid] = score;
        bar_sync(); // all scores written before reduction

        // Phase 2: Online softmax update
        let tile_max = block_reduce_max(tile[tid]);
        let mut m_new = m;
        if tile_max > m {
            m_new = tile_max;
        }
        let old_scale = exp(m - m_new);
        tile[tid] = exp(tile[tid] - m_new);
        bar_sync(); // exp_scores written before sum + V phase
        let tile_sum = block_reduce_sum(tile[tid]);

        // Rescale existing accumulator
        // INVARIANT: m, l, o_acc stay in the same scaling frame.
        // After tile t: m_t = max(scores 0..t), l_t = sum exp(s - m_t),
        // o_acc = sum exp(s - m_t) * V[s, tid] for tid < d_k.
        if tid < d_k {
            o_acc = o_acc * old_scale;
        }
        l = old_scale * l + tile_sum;
        m = m_new;

        // Phase 3: Accumulate P * V (first d_k threads)
        if tid < d_k {
            let mut jj = 0u32;
            while jj < 256 {
                if kv_start + jj < seq_len {
                    let v_val = v[(kv_start + jj) * d_k + tid];
                    o_acc = fma(tile[jj], v_val, o_acc);
                }
                jj += 1;
            }
        }
        bar_sync(); // sync before next tile overwrites shared

        kv_start += 256;
    }

    // Final normalization
    if tid < d_k {
        out[q_row * d_k + tid] = o_acc / l;
    }
}

// Causal variant: masks future positions (j > q_row) with -FLT_MAX.
#[allow(clippy::too_many_arguments)]
#[gpu_kernel(block_size = (256, 1))]
fn flash_attn_causal_kernel(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    seq_len: u32,
    d_k: u32,
    inv_sqrt_dk: f32,
) {
    let tid = thread_idx_x();
    let q_row = block_idx_x();
    let q_base = q_row * d_k;

    let tile = shared_mem![f32; 256];

    let mut o_acc = 0.0f32;
    let mut m = -3.402823e+38f32;
    let mut l = 0.0f32;

    let mut kv_start = 0u32;
    while kv_start < seq_len {
        let j = kv_start + tid;
        let mut score = -3.402823e+38f32;
        if j < seq_len {
            if j <= q_row {
                // Only attend to positions <= q_row (causal)
                score = 0.0f32;
                let mut d = 0u32;
                while d < d_k {
                    score = fma(q[q_base + d], k[j * d_k + d], score);
                    d += 1;
                }
                score = score * inv_sqrt_dk;
            }
        }
        tile[tid] = score;
        bar_sync();

        let tile_max = block_reduce_max(tile[tid]);
        let mut m_new = m;
        if tile_max > m {
            m_new = tile_max;
        }
        let old_scale = exp(m - m_new);
        tile[tid] = exp(tile[tid] - m_new);
        bar_sync();
        let tile_sum = block_reduce_sum(tile[tid]);

        if tid < d_k {
            o_acc = o_acc * old_scale;
        }
        l = old_scale * l + tile_sum;
        m = m_new;

        if tid < d_k {
            let mut jj = 0u32;
            while jj < 256 {
                if kv_start + jj < seq_len {
                    if kv_start + jj <= q_row {
                        let v_val = v[(kv_start + jj) * d_k + tid];
                        o_acc = fma(tile[jj], v_val, o_acc);
                    }
                }
                jj += 1;
            }
        }
        bar_sync();

        kv_start += 256;
    }

    if tid < d_k {
        out[q_row * d_k + tid] = o_acc / l;
    }
}

/// FlashAttention: single-head attention without materializing the
/// O(seq_len^2) attention matrix. O(d_k) memory per query position.
///
/// Same output as [`attention()`] within floating-point tolerance.
/// Different reduction order means results are numerically close,
/// not bitwise identical.
///
/// # Constraints
///
/// - d_k must be <= 256 (one thread per output dimension)
/// - d_v == d_k
/// - f32 only, row-major, contiguous
pub fn attention_flash(
    device: &KaioDevice,
    q: &GpuBuffer<f32>,
    k: &GpuBuffer<f32>,
    v: &GpuBuffer<f32>,
    out: &mut GpuBuffer<f32>,
    seq_len: u32,
    d_k: u32,
) -> Result<()> {
    validate_attention_dims(q, k, v, out, seq_len, d_k)?;
    validate_flash_dk(d_k)?;

    let inv_sqrt_dk = 1.0f32 / (d_k as f32).sqrt();
    let grid = (seq_len, 1, 1); // one block per query position
    flash_attn_kernel::launch(device, q, k, v, out, seq_len, d_k, inv_sqrt_dk, grid)?;
    Ok(())
}

/// FlashAttention with causal mask. See [`attention_flash()`] and
/// [`attention_causal()`] for details.
pub fn attention_flash_causal(
    device: &KaioDevice,
    q: &GpuBuffer<f32>,
    k: &GpuBuffer<f32>,
    v: &GpuBuffer<f32>,
    out: &mut GpuBuffer<f32>,
    seq_len: u32,
    d_k: u32,
) -> Result<()> {
    validate_attention_dims(q, k, v, out, seq_len, d_k)?;
    validate_flash_dk(d_k)?;

    let inv_sqrt_dk = 1.0f32 / (d_k as f32).sqrt();
    let grid = (seq_len, 1, 1);
    flash_attn_causal_kernel::launch(device, q, k, v, out, seq_len, d_k, inv_sqrt_dk, grid)?;
    Ok(())
}

fn validate_flash_dk(d_k: u32) -> Result<()> {
    if d_k > 256 {
        return Err(KaioError::InvalidConfig(format!(
            "FlashAttention requires d_k <= 256 (one thread per output dim), got {d_k}"
        )));
    }
    Ok(())
}

fn validate_attention_dims(
    q: &GpuBuffer<f32>,
    k: &GpuBuffer<f32>,
    v: &GpuBuffer<f32>,
    out: &GpuBuffer<f32>,
    seq_len: u32,
    d_k: u32,
) -> Result<()> {
    if seq_len == 0 || d_k == 0 {
        return Err(KaioError::InvalidConfig(
            "attention dimensions must be non-zero".to_string(),
        ));
    }
    let sd = (seq_len as usize) * (d_k as usize);
    if q.len() < sd {
        return Err(KaioError::InvalidConfig(format!(
            "Q buffer too small: need {sd} elements ({seq_len}×{d_k}), got {}",
            q.len()
        )));
    }
    if k.len() < sd {
        return Err(KaioError::InvalidConfig(format!(
            "K buffer too small: need {sd} elements ({seq_len}×{d_k}), got {}",
            k.len()
        )));
    }
    if v.len() < sd {
        return Err(KaioError::InvalidConfig(format!(
            "V buffer too small: need {sd} elements ({seq_len}×{d_k}), got {}",
            v.len()
        )));
    }
    if out.len() < sd {
        return Err(KaioError::InvalidConfig(format!(
            "output buffer too small: need {sd} elements ({seq_len}×{d_k}), got {}",
            out.len()
        )));
    }
    Ok(())
}
