//! E2E test: vector_add through the #[gpu_kernel] macro.
//!
//! This is the Phase 2 equivalent of kaio-runtime/tests/vector_add_e2e.rs.
//! The kernel is defined via the macro instead of hand-built IR.

use kaio::prelude::*;

#[gpu_kernel(block_size = 256)]
fn vector_add(a: &[f32], b: &[f32], out: &mut [f32], n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        out[idx] = a[idx] + b[idx];
    }
}

#[test]
#[ignore] // requires NVIDIA GPU
fn vector_add_small() {
    let device = KaioDevice::new(0).expect("GPU required");

    let a_host = [1.0f32, 2.0, 3.0];
    let b_host = [4.0f32, 5.0, 6.0];
    let n: u32 = 3;

    let a = device.alloc_from(&a_host).expect("alloc a");
    let b = device.alloc_from(&b_host).expect("alloc b");
    let mut out = device.alloc_zeros::<f32>(n as usize).expect("alloc out");

    vector_add::launch(&device, &a, &b, &mut out, n).expect("launch failed");

    let result = out.to_host(&device).expect("to_host");
    assert_eq!(
        result,
        vec![5.0f32, 7.0, 9.0],
        "vector_add produced wrong results"
    );
}

#[test]
#[ignore] // requires NVIDIA GPU
fn vector_add_large() {
    let device = KaioDevice::new(0).expect("GPU required");

    let n: u32 = 10_000;
    let a_host: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
    let expected: Vec<f32> = a_host.iter().zip(&b_host).map(|(a, b)| a + b).collect();

    let a = device.alloc_from(&a_host).expect("alloc a");
    let b = device.alloc_from(&b_host).expect("alloc b");
    let mut out = device.alloc_zeros::<f32>(n as usize).expect("alloc out");

    vector_add::launch(&device, &a, &b, &mut out, n).expect("launch failed");

    let result = out.to_host(&device).expect("to_host");
    assert_eq!(
        result, expected,
        "vector_add (10k elements) produced wrong results"
    );
}
