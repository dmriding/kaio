//! GPU tests for 2D block_size kernels.
//!
//! Validates that `#[gpu_kernel(block_size = (X, Y))]` with explicit
//! `grid: (u32, u32, u32)` works end-to-end on real GPU hardware.
//! Block dims are hardcoded from the attribute — callers only supply grid.

use kaio::prelude::*;

/// Simple 2D kernel: each thread writes `row * cols + col` to output.
/// Tests 2D thread indexing + grid-tuple launch.
#[gpu_kernel(block_size = (16, 16))]
fn write_2d_indices(out: &mut [f32], rows: u32, cols: u32) {
    let tx = thread_idx_x();
    let ty = thread_idx_y();
    let row = block_idx_y() * 16 + ty;
    let col = block_idx_x() * 16 + tx;
    if row < rows {
        if col < cols {
            out[row * cols + col] = row as f32 * 1000.0 + col as f32;
        }
    }
}

/// 2D kernel using FMA: c[i] = a[i] * b[i] + c[i] via fma().
#[gpu_kernel(block_size = 256)]
fn fma_elementwise(a: &[f32], b: &[f32], c: &mut [f32], n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        c[idx] = fma(a[idx], b[idx], c[idx]);
    }
}

#[test]
#[ignore] // requires NVIDIA GPU
fn kernel_2d_write_indices() {
    let device = KaioDevice::new(0).expect("GPU required");
    let rows = 32u32;
    let cols = 48u32;
    let n = (rows * cols) as usize;

    let mut out = device.alloc_zeros::<f32>(n).unwrap();

    let grid = (cols.div_ceil(16), rows.div_ceil(16), 1);
    write_2d_indices::launch(&device, &mut out, rows, cols, grid).unwrap();

    let result = out.to_host(&device).unwrap();
    for r in 0..rows {
        for c in 0..cols {
            let idx = (r * cols + c) as usize;
            let expected = r as f32 * 1000.0 + c as f32;
            assert_eq!(
                result[idx], expected,
                "mismatch at ({r}, {c}): got {}, expected {expected}",
                result[idx]
            );
        }
    }
}

#[test]
#[ignore] // requires NVIDIA GPU
fn kernel_2d_non_aligned_dims() {
    // Non-tile-aligned: 17×33 (not divisible by 16)
    let device = KaioDevice::new(0).expect("GPU required");
    let rows = 17u32;
    let cols = 33u32;
    let n = (rows * cols) as usize;

    let mut out = device.alloc_zeros::<f32>(n).unwrap();

    let grid = (cols.div_ceil(16), rows.div_ceil(16), 1);
    write_2d_indices::launch(&device, &mut out, rows, cols, grid).unwrap();

    let result = out.to_host(&device).unwrap();
    for r in 0..rows {
        for c in 0..cols {
            let idx = (r * cols + c) as usize;
            let expected = r as f32 * 1000.0 + c as f32;
            assert_eq!(result[idx], expected, "mismatch at ({r}, {c})");
        }
    }
}

#[test]
#[ignore] // requires NVIDIA GPU
fn kernel_fma_correctness() {
    let device = KaioDevice::new(0).expect("GPU required");
    let n = 1024usize;

    let a_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let b_data: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.01).collect();
    let c_data: Vec<f32> = vec![1.0f32; n];

    let a = device.alloc_from(&a_data).unwrap();
    let b = device.alloc_from(&b_data).unwrap();
    let mut c = device.alloc_from(&c_data).unwrap();

    fma_elementwise::launch(&device, &a, &b, &mut c, n as u32).unwrap();

    let result = c.to_host(&device).unwrap();
    for i in 0..n {
        let expected = a_data[i] * b_data[i] + c_data[i];
        let diff = (result[i] - expected).abs();
        assert!(
            diff < 1e-4,
            "fma mismatch at {i}: got {}, expected {expected}, diff {diff}",
            result[i]
        );
    }
}
