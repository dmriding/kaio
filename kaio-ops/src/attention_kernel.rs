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
/// Causal mask sets S\[i,j\] = -FLT_MAX where j > i, preventing
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

// ---------------------------------------------------------------------------
// FlashAttention `_with_stats` forward variants (Sprint 9.2)
// ---------------------------------------------------------------------------
// Identical computation to the kernels above plus one extra store: the
// per-row softmax logsumexp L = m + log(l), written once per block by
// thread 0. The backward kernels rebuild P_ij = exp(S_ij - L_i) from L
// instead of re-tracking the online-softmax max/sum.
//
// Separate kernel functions (not a flag) so the shipped forward kernels
// stay untouched; copy-per-variant matches the causal/non-causal split
// above. l >= 1 whenever the row has at least one valid key (the max
// entry contributes exp(0) = 1), so log(l) >= 0 is well-defined — same
// at-least-one-valid-key assumption documented at the top of this
// section.

#[allow(clippy::too_many_arguments)]
#[gpu_kernel(block_size = (256, 1))]
fn flash_attn_with_stats_kernel(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    stats: &mut [f32],
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
            score = 0.0f32;
            let mut d = 0u32;
            while d < d_k {
                score = fma(q[q_base + d], k[j * d_k + d], score);
                d += 1;
            }
            score = score * inv_sqrt_dk;
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
                    let v_val = v[(kv_start + jj) * d_k + tid];
                    o_acc = fma(tile[jj], v_val, o_acc);
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

    // m and l are block-uniform (both reduction-derived), so one thread
    // writes the row's logsumexp.
    if tid == 0 {
        stats[q_row] = m + log(l);
    }
}

// Causal `_with_stats` variant: same mask predicate as
// flash_attn_causal_kernel, same stats tail as above.
#[allow(clippy::too_many_arguments)]
#[gpu_kernel(block_size = (256, 1))]
fn flash_attn_causal_with_stats_kernel(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    stats: &mut [f32],
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

    if tid == 0 {
        stats[q_row] = m + log(l);
    }
}

// ---------------------------------------------------------------------------
// FlashAttention backward kernels (Sprint 9.2)
// ---------------------------------------------------------------------------
// Standard attention-backward identities with P rebuilt from the saved
// logsumexp L (no online max/sum re-tracking):
//
//   P_ij  = exp(S_ij - L_i)            S = scale * Q K^T
//   dV_j  = sum_i P_ij * dO_i
//   dP_ij = dO_i . V_j
//   D_i   = sum_d dO[i,d] * O[i,d]     (preprocess kernel, one pass)
//   dS_ij = P_ij * (dP_ij - D_i)
//   dQ_i  = scale * sum_j dS_ij * K_j
//   dK_j  = scale * sum_i dS_ij * Q_i
//
// Three kernels, all block-per-output-row with 256 threads (the same
// proven structure as the forward): a tiny preprocess computing D, a
// dK/dV kernel (one block per key row, streaming query rows), and a dQ
// kernel (one block per query row, streaming key rows). The loop-nest
// swap between the dK/dV and dQ kernels gives every output row exactly
// one owning block — no atomics, no cross-block reductions.

// D_i = sum_d dO[i,d] * O[i,d]. One block per row; threads cover the
// d_k dims (d_k <= 256), block-reduce, thread 0 stores. Mask-independent
// (only dO and O are involved), so one kernel serves both variants.
#[gpu_kernel(block_size = (256, 1))]
fn flash_attn_bwd_preprocess_kernel(d_out: &[f32], o: &[f32], d_buf: &mut [f32], d_k: u32) {
    let tid = thread_idx_x();
    let row = block_idx_x();

    let mut prod = 0.0f32;
    if tid < d_k {
        prod = d_out[row * d_k + tid] * o[row * d_k + tid];
    }
    let total = block_reduce_sum(prod);
    if tid == 0 {
        d_buf[row] = total;
    }
}

// dK_j / dV_j: one block per key row j, threads stream query rows i in
// 256-tiles. Phase 1: thread tid owns query row i = q_start + tid and
// computes P_ij and dS_ij into two shared tiles. Phase 2: the first d_k
// threads serially accumulate the tile into per-dim dK/dV registers
// (same accumulation shape as the forward's P*V phase).
#[allow(clippy::too_many_arguments)]
#[gpu_kernel(block_size = (256, 1))]
fn flash_attn_bwd_dkdv_kernel(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d_out: &[f32],
    stats: &[f32],
    d_buf: &[f32],
    dk: &mut [f32],
    dv: &mut [f32],
    seq_len: u32,
    d_k: u32,
    inv_sqrt_dk: f32,
) {
    let tid = thread_idx_x();
    let key_row = block_idx_x();

    let tile_p = shared_mem![f32; 256];
    let tile_ds = shared_mem![f32; 256];

    let mut dk_acc = 0.0f32;
    let mut dv_acc = 0.0f32;

    let mut q_start = 0u32;
    while q_start < seq_len {
        let i = q_start + tid;
        let mut p_val = 0.0f32;
        let mut ds_val = 0.0f32;
        if i < seq_len {
            let mut s = 0.0f32;
            let mut dp = 0.0f32;
            let mut d = 0u32;
            while d < d_k {
                s = fma(q[i * d_k + d], k[key_row * d_k + d], s);
                dp = fma(d_out[i * d_k + d], v[key_row * d_k + d], dp);
                d += 1;
            }
            s = s * inv_sqrt_dk;
            p_val = exp(s - stats[i]);
            ds_val = p_val * (dp - d_buf[i]);
        }
        tile_p[tid] = p_val;
        tile_ds[tid] = ds_val;
        bar_sync();

        if tid < d_k {
            let mut ii = 0u32;
            while ii < 256 {
                if q_start + ii < seq_len {
                    let row = q_start + ii;
                    dv_acc = fma(tile_p[ii], d_out[row * d_k + tid], dv_acc);
                    dk_acc = fma(tile_ds[ii], q[row * d_k + tid], dk_acc);
                }
                ii += 1;
            }
        }
        bar_sync();

        q_start += 256;
    }

    if tid < d_k {
        dk[key_row * d_k + tid] = dk_acc * inv_sqrt_dk;
        dv[key_row * d_k + tid] = dv_acc;
    }
}

// Causal dK/dV: key row j only receives gradient from query rows
// i >= j (the forward masked j > i). Invalid rows park zeros in the
// shared tiles, mirroring the forward's mask-then-accumulate shape.
#[allow(clippy::too_many_arguments)]
#[gpu_kernel(block_size = (256, 1))]
fn flash_attn_bwd_dkdv_causal_kernel(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d_out: &[f32],
    stats: &[f32],
    d_buf: &[f32],
    dk: &mut [f32],
    dv: &mut [f32],
    seq_len: u32,
    d_k: u32,
    inv_sqrt_dk: f32,
) {
    let tid = thread_idx_x();
    let key_row = block_idx_x();

    let tile_p = shared_mem![f32; 256];
    let tile_ds = shared_mem![f32; 256];

    let mut dk_acc = 0.0f32;
    let mut dv_acc = 0.0f32;

    let mut q_start = 0u32;
    while q_start < seq_len {
        let i = q_start + tid;
        let mut p_val = 0.0f32;
        let mut ds_val = 0.0f32;
        if i < seq_len {
            if i >= key_row {
                let mut s = 0.0f32;
                let mut dp = 0.0f32;
                let mut d = 0u32;
                while d < d_k {
                    s = fma(q[i * d_k + d], k[key_row * d_k + d], s);
                    dp = fma(d_out[i * d_k + d], v[key_row * d_k + d], dp);
                    d += 1;
                }
                s = s * inv_sqrt_dk;
                p_val = exp(s - stats[i]);
                ds_val = p_val * (dp - d_buf[i]);
            }
        }
        tile_p[tid] = p_val;
        tile_ds[tid] = ds_val;
        bar_sync();

        if tid < d_k {
            let mut ii = 0u32;
            while ii < 256 {
                if q_start + ii < seq_len {
                    let row = q_start + ii;
                    dv_acc = fma(tile_p[ii], d_out[row * d_k + tid], dv_acc);
                    dk_acc = fma(tile_ds[ii], q[row * d_k + tid], dk_acc);
                }
                ii += 1;
            }
        }
        bar_sync();

        q_start += 256;
    }

    if tid < d_k {
        dk[key_row * d_k + tid] = dk_acc * inv_sqrt_dk;
        dv[key_row * d_k + tid] = dv_acc;
    }
}

// dQ_i: one block per query row i, threads stream key rows j in
// 256-tiles (the forward's loop nest). Phase 1: thread tid owns key
// row j = kv_start + tid and computes dS_ij into a shared tile.
// Phase 2: the first d_k threads serially accumulate dQ_i from the
// tile. L_i and D_i are block-constant scalars.
#[allow(clippy::too_many_arguments)]
#[gpu_kernel(block_size = (256, 1))]
fn flash_attn_bwd_dq_kernel(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d_out: &[f32],
    stats: &[f32],
    d_buf: &[f32],
    dq: &mut [f32],
    seq_len: u32,
    d_k: u32,
    inv_sqrt_dk: f32,
) {
    let tid = thread_idx_x();
    let q_row = block_idx_x();
    let q_base = q_row * d_k;

    let tile_ds = shared_mem![f32; 256];

    let l_i = stats[q_row];
    let d_i = d_buf[q_row];

    let mut dq_acc = 0.0f32;

    let mut kv_start = 0u32;
    while kv_start < seq_len {
        let j = kv_start + tid;
        let mut ds_val = 0.0f32;
        if j < seq_len {
            let mut s = 0.0f32;
            let mut dp = 0.0f32;
            let mut d = 0u32;
            while d < d_k {
                s = fma(q[q_base + d], k[j * d_k + d], s);
                dp = fma(d_out[q_base + d], v[j * d_k + d], dp);
                d += 1;
            }
            s = s * inv_sqrt_dk;
            let p_val = exp(s - l_i);
            ds_val = p_val * (dp - d_i);
        }
        tile_ds[tid] = ds_val;
        bar_sync();

        if tid < d_k {
            let mut jj = 0u32;
            while jj < 256 {
                if kv_start + jj < seq_len {
                    dq_acc = fma(tile_ds[jj], k[(kv_start + jj) * d_k + tid], dq_acc);
                }
                jj += 1;
            }
        }
        bar_sync();

        kv_start += 256;
    }

    if tid < d_k {
        dq[q_base + tid] = dq_acc * inv_sqrt_dk;
    }
}

// Causal dQ: query row i only attends to keys j <= i, mirroring the
// forward's mask predicate. Masked keys park zeros in the shared tile.
#[allow(clippy::too_many_arguments)]
#[gpu_kernel(block_size = (256, 1))]
fn flash_attn_bwd_dq_causal_kernel(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d_out: &[f32],
    stats: &[f32],
    d_buf: &[f32],
    dq: &mut [f32],
    seq_len: u32,
    d_k: u32,
    inv_sqrt_dk: f32,
) {
    let tid = thread_idx_x();
    let q_row = block_idx_x();
    let q_base = q_row * d_k;

    let tile_ds = shared_mem![f32; 256];

    let l_i = stats[q_row];
    let d_i = d_buf[q_row];

    let mut dq_acc = 0.0f32;

    let mut kv_start = 0u32;
    while kv_start < seq_len {
        let j = kv_start + tid;
        let mut ds_val = 0.0f32;
        if j < seq_len {
            if j <= q_row {
                let mut s = 0.0f32;
                let mut dp = 0.0f32;
                let mut d = 0u32;
                while d < d_k {
                    s = fma(q[q_base + d], k[j * d_k + d], s);
                    dp = fma(d_out[q_base + d], v[j * d_k + d], dp);
                    d += 1;
                }
                s = s * inv_sqrt_dk;
                let p_val = exp(s - l_i);
                ds_val = p_val * (dp - d_i);
            }
        }
        tile_ds[tid] = ds_val;
        bar_sync();

        if tid < d_k {
            let mut jj = 0u32;
            while jj < 256 {
                if kv_start + jj < seq_len {
                    dq_acc = fma(tile_ds[jj], k[(kv_start + jj) * d_k + tid], dq_acc);
                }
                jj += 1;
            }
        }
        bar_sync();

        kv_start += 256;
    }

    if tid < d_k {
        dq[q_base + tid] = dq_acc * inv_sqrt_dk;
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

/// FlashAttention forward that additionally saves per-row softmax
/// statistics for a subsequent backward pass.
///
/// Identical computation and output to [`attention_flash()`], plus one
/// extra write per query row: `stats[i] = L_i = m_i + log(l_i)` — the
/// row-wise logsumexp of the scaled attention scores. A backward pass
/// rebuilds `P_ij = exp(S_ij − L_i)` from `L` instead of re-tracking
/// the online-softmax max/sum, at the cost of one extra
/// `seq_len × f32` buffer.
///
/// The stats contract is one `f32` per (batch item, head, query row).
/// With this op's single-head self-attention scope that collapses to a
/// flat `[seq_len]` buffer, which is all the validation requires.
///
/// # Constraints
///
/// Same as [`attention_flash()`], plus `stats` must hold at least
/// `seq_len` elements. Every query row attends to at least one key
/// (nothing is masked in this variant), so `l ≥ 1` and `log(l)` is
/// well-defined.
pub fn attention_flash_with_stats(
    device: &KaioDevice,
    q: &GpuBuffer<f32>,
    k: &GpuBuffer<f32>,
    v: &GpuBuffer<f32>,
    out: &mut GpuBuffer<f32>,
    stats: &mut GpuBuffer<f32>,
    seq_len: u32,
    d_k: u32,
) -> Result<()> {
    validate_attention_dims(q, k, v, out, seq_len, d_k)?;
    validate_flash_dk(d_k)?;
    validate_flash_stats(stats, seq_len)?;

    let inv_sqrt_dk = 1.0f32 / (d_k as f32).sqrt();
    let grid = (seq_len, 1, 1);
    flash_attn_with_stats_kernel::launch(
        device,
        q,
        k,
        v,
        out,
        stats,
        seq_len,
        d_k,
        inv_sqrt_dk,
        grid,
    )?;
    Ok(())
}

/// Causal-mask sibling of [`attention_flash_with_stats()`]. See
/// [`attention_flash_causal()`] for the mask semantics.
///
/// The causal diagonal guarantees every query row attends to at least
/// its own position, so `l ≥ 1` and `log(l)` is well-defined here too.
pub fn attention_flash_causal_with_stats(
    device: &KaioDevice,
    q: &GpuBuffer<f32>,
    k: &GpuBuffer<f32>,
    v: &GpuBuffer<f32>,
    out: &mut GpuBuffer<f32>,
    stats: &mut GpuBuffer<f32>,
    seq_len: u32,
    d_k: u32,
) -> Result<()> {
    validate_attention_dims(q, k, v, out, seq_len, d_k)?;
    validate_flash_dk(d_k)?;
    validate_flash_stats(stats, seq_len)?;

    let inv_sqrt_dk = 1.0f32 / (d_k as f32).sqrt();
    let grid = (seq_len, 1, 1);
    flash_attn_causal_with_stats_kernel::launch(
        device,
        q,
        k,
        v,
        out,
        stats,
        seq_len,
        d_k,
        inv_sqrt_dk,
        grid,
    )?;
    Ok(())
}

/// FlashAttention backward: computes `dQ`, `dK`, `dV` from the forward
/// inputs, the forward output, the saved logsumexp stats, and the
/// upstream gradient.
///
/// Launches three kernels on the device stream in order: a preprocess
/// computing `D_i = Σ_d dO[i,d]·O[i,d]` (into an internal `seq_len × f32`
/// scratch allocation), then dK/dV (one block per key row), then dQ
/// (one block per query row). No atomics — every output row is owned by
/// exactly one block.
///
/// # Provenance contract
///
/// `out` and `stats` must come from a single
/// [`attention_flash_with_stats()`] call on the **same** `q`, `k`, `v`
/// with the same `seq_len` / `d_k`. Validation can only check buffer
/// lengths — it cannot detect stats produced from different inputs or
/// from the causal sibling. Mixing produces silently wrong gradients.
///
/// # Constraints
///
/// Same as the forward: `d_k ≤ 256`, `d_v == d_k`, f32 row-major
/// contiguous; every query row attends to at least one key. `grad_out`,
/// `dq`, `dk`, `dv` hold at least `seq_len × d_k` elements; `stats` at
/// least `seq_len`.
#[allow(clippy::too_many_arguments)]
pub fn attention_flash_bwd(
    device: &KaioDevice,
    grad_out: &GpuBuffer<f32>,
    q: &GpuBuffer<f32>,
    k: &GpuBuffer<f32>,
    v: &GpuBuffer<f32>,
    out: &GpuBuffer<f32>,
    stats: &GpuBuffer<f32>,
    dq: &mut GpuBuffer<f32>,
    dk: &mut GpuBuffer<f32>,
    dv: &mut GpuBuffer<f32>,
    seq_len: u32,
    d_k: u32,
) -> Result<()> {
    validate_flash_bwd_dims(grad_out, q, k, v, out, stats, dq, dk, dv, seq_len, d_k)?;

    let mut d_buf = device.alloc_zeros::<f32>(seq_len as usize)?;
    attention_flash_bwd_preprocess(device, grad_out, out, &mut d_buf, seq_len, d_k)?;
    attention_flash_bwd_dkdv(
        device, grad_out, q, k, v, stats, &d_buf, dk, dv, seq_len, d_k, false,
    )?;
    attention_flash_bwd_dq(
        device, grad_out, q, k, v, stats, &d_buf, dq, seq_len, d_k, false,
    )?;
    Ok(())
}

/// Causal-mask sibling of [`attention_flash_bwd()`]. `out` and `stats`
/// must come from [`attention_flash_causal_with_stats()`] on the same
/// inputs — the same provenance contract applies, including the mask
/// mode (causal stats with the non-causal backward, or vice versa,
/// produce silently wrong gradients).
#[allow(clippy::too_many_arguments)]
pub fn attention_flash_bwd_causal(
    device: &KaioDevice,
    grad_out: &GpuBuffer<f32>,
    q: &GpuBuffer<f32>,
    k: &GpuBuffer<f32>,
    v: &GpuBuffer<f32>,
    out: &GpuBuffer<f32>,
    stats: &GpuBuffer<f32>,
    dq: &mut GpuBuffer<f32>,
    dk: &mut GpuBuffer<f32>,
    dv: &mut GpuBuffer<f32>,
    seq_len: u32,
    d_k: u32,
) -> Result<()> {
    validate_flash_bwd_dims(grad_out, q, k, v, out, stats, dq, dk, dv, seq_len, d_k)?;

    let mut d_buf = device.alloc_zeros::<f32>(seq_len as usize)?;
    attention_flash_bwd_preprocess(device, grad_out, out, &mut d_buf, seq_len, d_k)?;
    attention_flash_bwd_dkdv(
        device, grad_out, q, k, v, stats, &d_buf, dk, dv, seq_len, d_k, true,
    )?;
    attention_flash_bwd_dq(
        device, grad_out, q, k, v, stats, &d_buf, dq, seq_len, d_k, true,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_flash_bwd_dims(
    grad_out: &GpuBuffer<f32>,
    q: &GpuBuffer<f32>,
    k: &GpuBuffer<f32>,
    v: &GpuBuffer<f32>,
    out: &GpuBuffer<f32>,
    stats: &GpuBuffer<f32>,
    dq: &GpuBuffer<f32>,
    dk: &GpuBuffer<f32>,
    dv: &GpuBuffer<f32>,
    seq_len: u32,
    d_k: u32,
) -> Result<()> {
    if seq_len == 0 || d_k == 0 {
        return Err(KaioError::InvalidConfig(
            "attention dimensions must be non-zero".to_string(),
        ));
    }
    validate_flash_dk(d_k)?;
    let sd = (seq_len as usize) * (d_k as usize);
    let buffers: [(&str, usize); 8] = [
        ("grad_out", grad_out.len()),
        ("Q", q.len()),
        ("K", k.len()),
        ("V", v.len()),
        ("out", out.len()),
        ("dQ", dq.len()),
        ("dK", dk.len()),
        ("dV", dv.len()),
    ];
    for (name, len) in buffers {
        if len < sd {
            return Err(KaioError::InvalidConfig(format!(
                "{name} buffer too small: need {sd} elements ({seq_len}×{d_k}), got {len}"
            )));
        }
    }
    validate_flash_stats(stats, seq_len)
}

/// Backward preprocess: `D[i] = Σ_d dO[i,d] · O[i,d]`, one f32 per
/// query row. Internal building block of `attention_flash_bwd`;
/// exposed for the per-kernel correctness tests only.
#[doc(hidden)]
pub fn attention_flash_bwd_preprocess(
    device: &KaioDevice,
    grad_out: &GpuBuffer<f32>,
    out: &GpuBuffer<f32>,
    d_buf: &mut GpuBuffer<f32>,
    seq_len: u32,
    d_k: u32,
) -> Result<()> {
    let grid = (seq_len, 1, 1);
    flash_attn_bwd_preprocess_kernel::launch(device, grad_out, out, d_buf, d_k, grid)?;
    Ok(())
}

/// Backward dK/dV accumulation. Internal building block of
/// `attention_flash_bwd`; exposed for the per-kernel correctness tests
/// only. Callers are responsible for `stats` and `d_buf` provenance
/// (same q/k/v, same mask mode).
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn attention_flash_bwd_dkdv(
    device: &KaioDevice,
    grad_out: &GpuBuffer<f32>,
    q: &GpuBuffer<f32>,
    k: &GpuBuffer<f32>,
    v: &GpuBuffer<f32>,
    stats: &GpuBuffer<f32>,
    d_buf: &GpuBuffer<f32>,
    dk: &mut GpuBuffer<f32>,
    dv: &mut GpuBuffer<f32>,
    seq_len: u32,
    d_k: u32,
    causal: bool,
) -> Result<()> {
    let inv_sqrt_dk = 1.0f32 / (d_k as f32).sqrt();
    let grid = (seq_len, 1, 1); // one block per key row
    if causal {
        flash_attn_bwd_dkdv_causal_kernel::launch(
            device,
            q,
            k,
            v,
            grad_out,
            stats,
            d_buf,
            dk,
            dv,
            seq_len,
            d_k,
            inv_sqrt_dk,
            grid,
        )?;
    } else {
        flash_attn_bwd_dkdv_kernel::launch(
            device,
            q,
            k,
            v,
            grad_out,
            stats,
            d_buf,
            dk,
            dv,
            seq_len,
            d_k,
            inv_sqrt_dk,
            grid,
        )?;
    }
    Ok(())
}

/// Backward dQ accumulation. Internal building block of
/// `attention_flash_bwd`; exposed for the per-kernel correctness tests
/// only. Callers are responsible for `stats` and `d_buf` provenance
/// (same q/k/v, same mask mode).
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn attention_flash_bwd_dq(
    device: &KaioDevice,
    grad_out: &GpuBuffer<f32>,
    q: &GpuBuffer<f32>,
    k: &GpuBuffer<f32>,
    v: &GpuBuffer<f32>,
    stats: &GpuBuffer<f32>,
    d_buf: &GpuBuffer<f32>,
    dq: &mut GpuBuffer<f32>,
    seq_len: u32,
    d_k: u32,
    causal: bool,
) -> Result<()> {
    let inv_sqrt_dk = 1.0f32 / (d_k as f32).sqrt();
    let grid = (seq_len, 1, 1); // one block per query row
    if causal {
        flash_attn_bwd_dq_causal_kernel::launch(
            device,
            q,
            k,
            v,
            grad_out,
            stats,
            d_buf,
            dq,
            seq_len,
            d_k,
            inv_sqrt_dk,
            grid,
        )?;
    } else {
        flash_attn_bwd_dq_kernel::launch(
            device,
            q,
            k,
            v,
            grad_out,
            stats,
            d_buf,
            dq,
            seq_len,
            d_k,
            inv_sqrt_dk,
            grid,
        )?;
    }
    Ok(())
}

fn validate_flash_stats(stats: &GpuBuffer<f32>, seq_len: u32) -> Result<()> {
    if stats.len() < seq_len as usize {
        return Err(KaioError::InvalidConfig(format!(
            "stats buffer too small: need seq_len = {seq_len} elements (one logsumexp per query row), got {}",
            stats.len()
        )));
    }
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
