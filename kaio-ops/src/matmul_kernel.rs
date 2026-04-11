//! Tiled matrix multiplication: C = A × B.
//!
//! All f32, row-major, contiguous. A is M×K, B is K×N, C is M×N.
//!
//! # Kernel variants
//!
//! - **Naive** (`tiled_matmul_kernel`): 16×16 tiles, 1 output per thread.
//!   Used as correctness reference. ~8% of cuBLAS.
//! - **Optimized** (`reg_tiled_matmul_kernel`): 64×64 tiles, 4×4 outputs
//!   per thread via register tiling. Default for `matmul()`.
//!
//! # Design parameters (optimized kernel)
//!
//! - BM = 64 (block output rows)
//! - BN = 64 (block output cols)
//! - BK = 16 (K tile size)
//! - TM = 4 (per-thread output rows)
//! - TN = 4 (per-thread output cols)
//! - Block = (16, 16) = 256 threads
//! - Shared tile_a: 64×17 (stride 17 for bank conflict avoidance)
//! - Shared tile_b: 16×65 (stride 65 for bank conflict avoidance)
//!
//! # Launch derivation
//!
//! `matmul()` owns the launch configuration policy:
//! - Optimized: grid = `(ceil(N/64), ceil(M/64), 1)`
//! - Naive (internal): grid = `(ceil(N/16), ceil(M/16), 1)`
//!
//! # Buffer sizing
//!
//! Buffers may be larger than the logical matrix region. Only the first
//! M×K, K×N, and M×N elements are used from A, B, and C respectively.

use kaio::prelude::*;

// ---------------------------------------------------------------------------
// Naive kernel (correctness reference, ~8% of cuBLAS)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
#[gpu_kernel(block_size = (16, 16))]
fn tiled_matmul_kernel(a: &[f32], b: &[f32], c: &mut [f32], m: u32, n: u32, k: u32) {
    let tx = thread_idx_x();
    let ty = thread_idx_y();
    let row = block_idx_y() * 16 + ty;
    let col = block_idx_x() * 16 + tx;

    let tile_a = shared_mem![f32; 256];
    let tile_b = shared_mem![f32; 256];

    let mut acc = 0.0f32;
    let num_tiles = (k + 15) / 16;

    let mut t = 0u32;
    while t < num_tiles {
        let a_col = t * 16 + tx;
        tile_a[ty * 16 + tx] = 0.0f32;
        if row < m {
            if a_col < k {
                tile_a[ty * 16 + tx] = a[row * k + a_col];
            }
        }

        let b_row = t * 16 + ty;
        tile_b[ty * 16 + tx] = 0.0f32;
        if b_row < k {
            if col < n {
                tile_b[ty * 16 + tx] = b[b_row * n + col];
            }
        }

        bar_sync();

        let mut i = 0u32;
        while i < 16 {
            acc = fma(tile_a[ty * 16 + i], tile_b[i * 16 + tx], acc);
            i += 1;
        }

        bar_sync();
        t += 1;
    }

    if row < m {
        if col < n {
            c[row * n + col] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Register-tiled kernel (optimized, 4×4 per thread)
// BM=64, BN=64, BK=16, TM=4, TN=4
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
#[gpu_kernel(block_size = (16, 16))]
fn reg_tiled_matmul_kernel(a: &[f32], b: &[f32], c: &mut [f32], m: u32, n: u32, k: u32) {
    let tx = thread_idx_x();
    let ty = thread_idx_y();
    let block_row = block_idx_y() * 64;
    let block_col = block_idx_x() * 64;

    // Shared tiles with bank-conflict-avoiding padding
    // tile_a: BM×BK = 64×16, padded stride = 17
    // tile_b: BK×BN = 16×64, padded stride = 65
    let tile_a = shared_mem![f32; 1088]; // 64 * 17
    let tile_b = shared_mem![f32; 1040]; // 16 * 65

    // 16 accumulators: TM×TN = 4×4 output elements per thread
    let mut acc_00 = 0.0f32;
    let mut acc_01 = 0.0f32;
    let mut acc_02 = 0.0f32;
    let mut acc_03 = 0.0f32;
    let mut acc_10 = 0.0f32;
    let mut acc_11 = 0.0f32;
    let mut acc_12 = 0.0f32;
    let mut acc_13 = 0.0f32;
    let mut acc_20 = 0.0f32;
    let mut acc_21 = 0.0f32;
    let mut acc_22 = 0.0f32;
    let mut acc_23 = 0.0f32;
    let mut acc_30 = 0.0f32;
    let mut acc_31 = 0.0f32;
    let mut acc_32 = 0.0f32;
    let mut acc_33 = 0.0f32;

    let num_tiles = (k + 15) / 16;
    let mut t = 0u32;

    while t < num_tiles {
        // --- Load phase ---
        // tile_a: 64×16, 256 threads load 4 elements each
        // Thread (ty,tx) loads rows {ty, ty+16, ty+32, ty+48} at column tx
        let mut li = 0u32;
        while li < 4 {
            let load_row = ty + li * 16;
            let global_row = block_row + load_row;
            let global_col = t * 16 + tx;
            tile_a[load_row * 17 + tx] = 0.0f32;
            if global_row < m {
                if global_col < k {
                    tile_a[load_row * 17 + tx] = a[global_row * k + global_col];
                }
            }
            li += 1;
        }

        // tile_b: 16×64, 256 threads load 4 elements each
        // Thread (ty,tx) loads row ty at columns {tx, tx+16, tx+32, tx+48}
        let mut lj = 0u32;
        while lj < 4 {
            let load_col = tx + lj * 16;
            let global_row = t * 16 + ty;
            let global_col = block_col + load_col;
            tile_b[ty * 65 + load_col] = 0.0f32;
            if global_row < k {
                if global_col < n {
                    tile_b[ty * 65 + load_col] = b[global_row * n + global_col];
                }
            }
            lj += 1;
        }

        bar_sync();

        // --- Compute phase ---
        // Inner loop over BK=16: each step reads TM=4 from tile_a, TN=4
        // from tile_b, does TM×TN=16 FMAs.
        let mut ki = 0u32;
        while ki < 16 {
            // Load TM=4 values from tile_a (thread's output rows, column ki)
            let a0 = tile_a[(ty * 4) * 17 + ki];
            let a1 = tile_a[(ty * 4 + 1) * 17 + ki];
            let a2 = tile_a[(ty * 4 + 2) * 17 + ki];
            let a3 = tile_a[(ty * 4 + 3) * 17 + ki];

            // Load TN=4 values from tile_b (row ki, thread's output cols)
            let b0 = tile_b[ki * 65 + tx * 4];
            let b1 = tile_b[ki * 65 + tx * 4 + 1];
            let b2 = tile_b[ki * 65 + tx * 4 + 2];
            let b3 = tile_b[ki * 65 + tx * 4 + 3];

            // 16 FMAs: acc[i][j] = fma(a[i], b[j], acc[i][j])
            // Row 0
            acc_00 = fma(a0, b0, acc_00);
            acc_01 = fma(a0, b1, acc_01);
            acc_02 = fma(a0, b2, acc_02);
            acc_03 = fma(a0, b3, acc_03);
            // Row 1
            acc_10 = fma(a1, b0, acc_10);
            acc_11 = fma(a1, b1, acc_11);
            acc_12 = fma(a1, b2, acc_12);
            acc_13 = fma(a1, b3, acc_13);
            // Row 2
            acc_20 = fma(a2, b0, acc_20);
            acc_21 = fma(a2, b1, acc_21);
            acc_22 = fma(a2, b2, acc_22);
            acc_23 = fma(a2, b3, acc_23);
            // Row 3
            acc_30 = fma(a3, b0, acc_30);
            acc_31 = fma(a3, b1, acc_31);
            acc_32 = fma(a3, b2, acc_32);
            acc_33 = fma(a3, b3, acc_33);

            ki += 1;
        }

        bar_sync();
        t += 1;
    }

    // --- Output write phase ---
    // Write 4×4 output tile with bounds checks, grouped by row.
    let out_row_0 = block_row + ty * 4;
    let out_col_0 = block_col + tx * 4;

    // Row 0 of per-thread tile
    if out_row_0 < m {
        if out_col_0 < n {
            c[out_row_0 * n + out_col_0] = acc_00;
        }
        if out_col_0 + 1 < n {
            c[out_row_0 * n + out_col_0 + 1] = acc_01;
        }
        if out_col_0 + 2 < n {
            c[out_row_0 * n + out_col_0 + 2] = acc_02;
        }
        if out_col_0 + 3 < n {
            c[out_row_0 * n + out_col_0 + 3] = acc_03;
        }
    }
    // Row 1 of per-thread tile
    if out_row_0 + 1 < m {
        if out_col_0 < n {
            c[(out_row_0 + 1) * n + out_col_0] = acc_10;
        }
        if out_col_0 + 1 < n {
            c[(out_row_0 + 1) * n + out_col_0 + 1] = acc_11;
        }
        if out_col_0 + 2 < n {
            c[(out_row_0 + 1) * n + out_col_0 + 2] = acc_12;
        }
        if out_col_0 + 3 < n {
            c[(out_row_0 + 1) * n + out_col_0 + 3] = acc_13;
        }
    }
    // Row 2 of per-thread tile
    if out_row_0 + 2 < m {
        if out_col_0 < n {
            c[(out_row_0 + 2) * n + out_col_0] = acc_20;
        }
        if out_col_0 + 1 < n {
            c[(out_row_0 + 2) * n + out_col_0 + 1] = acc_21;
        }
        if out_col_0 + 2 < n {
            c[(out_row_0 + 2) * n + out_col_0 + 2] = acc_22;
        }
        if out_col_0 + 3 < n {
            c[(out_row_0 + 2) * n + out_col_0 + 3] = acc_23;
        }
    }
    // Row 3 of per-thread tile
    if out_row_0 + 3 < m {
        if out_col_0 < n {
            c[(out_row_0 + 3) * n + out_col_0] = acc_30;
        }
        if out_col_0 + 1 < n {
            c[(out_row_0 + 3) * n + out_col_0 + 1] = acc_31;
        }
        if out_col_0 + 2 < n {
            c[(out_row_0 + 3) * n + out_col_0 + 2] = acc_32;
        }
        if out_col_0 + 3 < n {
            c[(out_row_0 + 3) * n + out_col_0 + 3] = acc_33;
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute C = A × B where A is M×K, B is K×N, C is M×N.
///
/// All f32, row-major, contiguous. Buffers may be larger than the logical
/// matrix region — only the first M×K / K×N / M×N elements are used.
///
/// Uses the register-tiled kernel (4×4 per thread, 64×64 block tile) by
/// default. The naive 16×16 kernel is kept internally as a correctness
/// reference.
///
/// # Errors
///
/// Returns `KaioError::InvalidConfig` if:
/// - Any dimension is zero
/// - A buffer is too small for the declared dimensions
///
/// # Example
///
/// ```ignore
/// use kaio::prelude::*;
/// use kaio_ops::matmul;
///
/// let device = KaioDevice::new(0)?;
/// let a = device.alloc_from(&vec![1.0f32; 64 * 128])?;
/// let b = device.alloc_from(&vec![1.0f32; 128 * 32])?;
/// let mut c = device.alloc_zeros::<f32>(64 * 32)?;
/// matmul(&device, &a, &b, &mut c, 64, 32, 128)?;
/// ```
pub fn matmul(
    device: &KaioDevice,
    a: &GpuBuffer<f32>,
    b: &GpuBuffer<f32>,
    c: &mut GpuBuffer<f32>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    validate_dims(a, b, c, m, n, k)?;

    // Optimized: BM=BN=64 block output tile
    let grid = (n.div_ceil(64), m.div_ceil(64), 1);
    reg_tiled_matmul_kernel::launch(device, a, b, c, m, n, k, grid)
}

/// Run the naive 16×16 kernel (for benchmarking comparison).
#[doc(hidden)]
pub fn matmul_naive(
    device: &KaioDevice,
    a: &GpuBuffer<f32>,
    b: &GpuBuffer<f32>,
    c: &mut GpuBuffer<f32>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    validate_dims(a, b, c, m, n, k)?;
    let grid = (n.div_ceil(16), m.div_ceil(16), 1);
    tiled_matmul_kernel::launch(device, a, b, c, m, n, k, grid)
}

fn validate_dims(
    a: &GpuBuffer<f32>,
    b: &GpuBuffer<f32>,
    c: &GpuBuffer<f32>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    if m == 0 || n == 0 || k == 0 {
        return Err(KaioError::InvalidConfig(
            "matmul dimensions must be non-zero".to_string(),
        ));
    }
    let mk = (m as usize) * (k as usize);
    let kn = (k as usize) * (n as usize);
    let mn = (m as usize) * (n as usize);
    if a.len() < mk {
        return Err(KaioError::InvalidConfig(format!(
            "A buffer too small: need {} elements ({}×{}), got {}",
            mk,
            m,
            k,
            a.len()
        )));
    }
    if b.len() < kn {
        return Err(KaioError::InvalidConfig(format!(
            "B buffer too small: need {} elements ({}×{}), got {}",
            kn,
            k,
            n,
            b.len()
        )));
    }
    if c.len() < mn {
        return Err(KaioError::InvalidConfig(format!(
            "C buffer too small: need {} elements ({}×{}), got {}",
            mn,
            m,
            n,
            c.len()
        )));
    }
    Ok(())
}
