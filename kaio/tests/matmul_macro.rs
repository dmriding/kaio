//! GPU tests for tiled matrix multiplication.
//!
//! Sprint 4.3: naive 16×16 tiled matmul, correctness-first.
//! C = A × B where A is M×K, B is K×N, C is M×N. All f32 row-major.

// The generated launch() function for matmul has 8 params (device + 6 kernel + grid).
#![allow(clippy::too_many_arguments)]

use kaio::prelude::*;

// --- Kernel ---

/// Naive tiled matmul: 16×16 tiles, FMA inner loop, zero-fill edges.
#[gpu_kernel(block_size = (16, 16))]
fn tiled_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: u32, n: u32, k: u32) {
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
        // Zero-fill, then conditionally load from A
        tile_a[ty * 16 + tx] = 0.0f32;
        if row < m {
            if a_col < k {
                tile_a[ty * 16 + tx] = a[row * k + a_col];
            }
        }

        let b_row = t * 16 + ty;
        // Zero-fill, then conditionally load from B
        tile_b[ty * 16 + tx] = 0.0f32;
        if b_row < k {
            if col < n {
                tile_b[ty * 16 + tx] = b[b_row * n + col];
            }
        }

        bar_sync();

        // Accumulate dot product for this tile
        let mut i = 0u32;
        while i < 16 {
            acc = fma(tile_a[ty * 16 + i], tile_b[i * 16 + tx], acc);
            i += 1;
        }

        bar_sync();
        t += 1;
    }

    // Write result
    if row < m {
        if col < n {
            c[row * n + col] = acc;
        }
    }
}

// --- CPU reference ---

fn cpu_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

// --- Test helper ---

fn run_matmul_test(m: usize, n: usize, k: usize, label: &str) {
    assert!(m > 0 && n > 0 && k > 0, "dimensions must be > 0");

    let device = KaioDevice::new(0).expect("GPU required");

    // Generate input data
    let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();

    let a = device.alloc_from(&a_data).unwrap();
    let b = device.alloc_from(&b_data).unwrap();
    let mut c = device.alloc_zeros::<f32>(m * n).unwrap();

    let grid = ((n as u32).div_ceil(16), (m as u32).div_ceil(16), 1);
    tiled_matmul::launch(&device, &a, &b, &mut c, m as u32, n as u32, k as u32, grid).unwrap();

    let result = c.to_host(&device).unwrap();
    let expected = cpu_matmul(&a_data, &b_data, m, n, k);

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;

    for idx in 0..m * n {
        let got = result[idx];
        let exp = expected[idx];
        let abs_err = (got - exp).abs();
        let rel_err = if exp.abs() > 1e-6 {
            abs_err / exp.abs()
        } else {
            abs_err
        };
        if abs_err > max_abs {
            max_abs = abs_err;
        }
        if rel_err > max_rel {
            max_rel = rel_err;
        }
        assert!(
            abs_err < 1e-3,
            "{label}: abs error {abs_err} at [{}, {}] (got {got}, expected {exp})",
            idx / n,
            idx % n,
        );
    }

    eprintln!("{label} ({m}x{k} × {k}x{n}): max_abs={max_abs:.2e}, max_rel={max_rel:.2e}");
}

// --- GPU tests ---

#[test]
#[ignore] // requires NVIDIA GPU
fn matmul_tiny() {
    // Hand-checkable: A(2×4) × B(4×3) = C(2×3)
    let device = KaioDevice::new(0).expect("GPU required");

    #[rustfmt::skip]
    let a_data = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
    ];
    #[rustfmt::skip]
    let b_data = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0,
    ];
    // Expected C:
    // [1*1+2*4+3*7+4*10, 1*2+2*5+3*8+4*11, 1*3+2*6+3*9+4*12] = [70, 80, 90]
    // [5*1+6*4+7*7+8*10, 5*2+6*5+7*8+8*11, 5*3+6*6+7*9+8*12] = [158, 184, 210]
    let expected = [70.0f32, 80.0, 90.0, 158.0, 184.0, 210.0];

    let (m, n, k) = (2u32, 3u32, 4u32);
    let a = device.alloc_from(&a_data).unwrap();
    let b = device.alloc_from(&b_data).unwrap();
    let mut c = device.alloc_zeros::<f32>((m * n) as usize).unwrap();

    let grid = (n.div_ceil(16), m.div_ceil(16), 1);
    tiled_matmul::launch(&device, &a, &b, &mut c, m, n, k, grid).unwrap();

    let result = c.to_host(&device).unwrap();
    for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-4,
            "tiny matmul mismatch at {i}: got {got}, expected {exp}"
        );
    }
}

#[test]
#[ignore] // requires NVIDIA GPU
fn matmul_16x16() {
    run_matmul_test(16, 16, 16, "16x16");
}

#[test]
#[ignore] // requires NVIDIA GPU
fn matmul_64x64() {
    run_matmul_test(64, 64, 64, "64x64");
}

#[test]
#[ignore] // requires NVIDIA GPU
fn matmul_non_square() {
    run_matmul_test(100, 200, 150, "non_square");
}

#[test]
#[ignore] // requires NVIDIA GPU
fn matmul_non_aligned() {
    run_matmul_test(17, 33, 19, "non_aligned");
}

#[test]
#[ignore] // requires NVIDIA GPU
fn matmul_1024x1024() {
    run_matmul_test(1024, 1024, 1024, "1024x1024");
}
