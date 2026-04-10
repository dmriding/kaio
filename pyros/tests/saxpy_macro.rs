//! E2E test: saxpy (y = a*x + y) through the #[gpu_kernel] macro.

use pyros::prelude::*;

#[gpu_kernel(block_size = 256)]
fn saxpy(x: &[f32], y: &mut [f32], alpha: f32, n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        y[idx] = alpha * x[idx] + y[idx];
    }
}

#[test]
#[ignore] // requires NVIDIA GPU
fn saxpy_correctness() {
    let device = PyrosDevice::new(0).expect("GPU required");

    let n: u32 = 1024;
    let alpha: f32 = 2.5;
    let x_host: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let y_host: Vec<f32> = (0..n).map(|i| (i * 3) as f32).collect();
    let expected: Vec<f32> = x_host
        .iter()
        .zip(&y_host)
        .map(|(x, y)| alpha * x + y)
        .collect();

    let x = device.alloc_from(&x_host).expect("alloc x");
    let mut y = device.alloc_from(&y_host).expect("alloc y");

    saxpy::launch(&device, &x, &mut y, alpha, n).expect("launch failed");

    let result = y.to_host(&device).expect("to_host");
    assert_eq!(result, expected, "saxpy produced wrong results");
}
