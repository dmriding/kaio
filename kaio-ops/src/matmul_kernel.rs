//! Tiled matrix multiplication: C = A × B.
//!
//! All f32, row-major, contiguous. A is M×K, B is K×N, C is M×N.
//!
//! # Launch derivation
//!
//! `matmul()` owns the launch configuration policy:
//! - Tile size: 16×16
//! - Block size: (16, 16) = 256 threads
//! - Grid: `(ceil(N/16), ceil(M/16), 1)`
//!
//! # Buffer sizing
//!
//! Buffers may be larger than the logical matrix region. Only the first
//! M×K, K×N, and M×N elements are used from A, B, and C respectively.

use kaio::prelude::*;

// --- Internal kernel (not public) ---

#[allow(clippy::too_many_arguments)]
#[gpu_kernel(block_size = (16, 16))]
fn tiled_matmul_kernel(a: &[f32], b: &[f32], c: &mut [f32], m: u32, n: u32, k: u32) {
    let tx = thread_idx_x();
    let ty = thread_idx_y();
    let row = block_idx_y() * 16 + ty;
    let col = block_idx_x() * 16 + tx;

    let tile_a = shared_mem![f32; 256]; // 16×16
    let tile_b = shared_mem![f32; 256]; // 16×16

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

// --- Public API ---

/// Compute C = A × B where A is M×K, B is K×N, C is M×N.
///
/// All f32, row-major, contiguous. Buffers may be larger than the logical
/// matrix region — only the first M×K / K×N / M×N elements are used.
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
    // Validate dimensions
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

    let grid = (n.div_ceil(16), m.div_ceil(16), 1);
    tiled_matmul_kernel::launch(device, a, b, c, m, n, k, grid)
}
