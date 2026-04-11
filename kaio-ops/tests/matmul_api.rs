//! Tests for kaio_ops::matmul() through the public API.

use kaio::prelude::*;
use kaio_ops::matmul;

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

fn run_matmul_test(m: usize, n: usize, k: usize, label: &str) {
    let device = KaioDevice::new(0).expect("GPU required");

    let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();

    let a = device.alloc_from(&a_data).unwrap();
    let b = device.alloc_from(&b_data).unwrap();
    let mut c = device.alloc_zeros::<f32>(m * n).unwrap();

    matmul(&device, &a, &b, &mut c, m as u32, n as u32, k as u32).unwrap();

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
            "{label}: abs error {abs_err} at index {idx}"
        );
        assert!(
            rel_err < 1e-4,
            "{label}: rel error {rel_err} at index {idx}"
        );
    }

    eprintln!("{label} ({m}×{k} × {k}×{n}): max_abs={max_abs:.2e}, max_rel={max_rel:.2e}");
}

// --- Correctness tests ---

#[test]
#[ignore] // requires NVIDIA GPU
fn api_matmul_tiny() {
    let device = KaioDevice::new(0).expect("GPU required");

    #[rustfmt::skip]
    let a_data = vec![
        1.0f32, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
    ];
    #[rustfmt::skip]
    let b_data = vec![
        1.0f32, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0,
    ];
    let expected = [70.0f32, 80.0, 90.0, 158.0, 184.0, 210.0];

    let a = device.alloc_from(&a_data).unwrap();
    let b = device.alloc_from(&b_data).unwrap();
    let mut c = device.alloc_zeros::<f32>(6).unwrap();

    matmul(&device, &a, &b, &mut c, 2, 3, 4).unwrap();

    let result = c.to_host(&device).unwrap();
    for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-4,
            "mismatch at {i}: got {got}, expected {exp}"
        );
    }
}

#[test]
#[ignore] // requires NVIDIA GPU
fn api_matmul_64x64() {
    run_matmul_test(64, 64, 64, "64x64");
}

#[test]
#[ignore] // requires NVIDIA GPU
fn api_matmul_non_square() {
    run_matmul_test(100, 200, 150, "non_square");
}

#[test]
#[ignore] // requires NVIDIA GPU
fn api_matmul_non_aligned() {
    run_matmul_test(17, 33, 19, "non_aligned");
}

// --- Validation tests (need GPU for buffer allocation, not for the check itself) ---

#[test]
#[ignore] // requires NVIDIA GPU (buffer allocation needs a device)
fn api_matmul_rejects_zero_m() {
    let device = KaioDevice::new(0).expect("GPU required");
    let a = device.alloc_zeros::<f32>(1).unwrap();
    let b = device.alloc_zeros::<f32>(1).unwrap();
    let mut c = device.alloc_zeros::<f32>(1).unwrap();
    let err = matmul(&device, &a, &b, &mut c, 0, 1, 1).unwrap_err();
    assert!(
        err.to_string().contains("non-zero"),
        "expected zero-dim error, got: {err}"
    );
}

#[test]
#[ignore] // requires NVIDIA GPU (buffer allocation needs a device)
fn api_matmul_rejects_small_buffer() {
    let device = KaioDevice::new(0).expect("GPU required");
    let a = device.alloc_zeros::<f32>(4).unwrap(); // too small for 3×3
    let b = device.alloc_zeros::<f32>(9).unwrap();
    let mut c = device.alloc_zeros::<f32>(9).unwrap();
    let err = matmul(&device, &a, &b, &mut c, 3, 3, 3).unwrap_err();
    assert!(
        err.to_string().contains("A buffer too small"),
        "expected buffer error, got: {err}"
    );
}

#[test]
#[ignore] // requires NVIDIA GPU
fn api_matmul_oversized_buffers() {
    // Buffers larger than logical region — only M×K/K×N/M×N elements used
    let device = KaioDevice::new(0).expect("GPU required");
    let a = device.alloc_from(&vec![1.0f32; 100]).unwrap(); // 100 > 2*4=8
    let b = device.alloc_from(&vec![1.0f32; 100]).unwrap(); // 100 > 4*3=12
    let mut c = device.alloc_zeros::<f32>(100).unwrap(); // 100 > 2*3=6

    matmul(&device, &a, &b, &mut c, 2, 3, 4).unwrap();

    let result = c.to_host(&device).unwrap();
    // Each element of C should be sum of K=4 ones = 4.0
    for (i, &val) in result.iter().enumerate().take(6) {
        assert!(
            (val - 4.0).abs() < 1e-4,
            "oversized buffer mismatch at {i}: got {}",
            val
        );
    }
}
