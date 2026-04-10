//! E2E test: fused_relu through the #[gpu_kernel] macro.
//!
//! relu(x) = max(x, 0) — tests if/else with float comparison.

use kaio::prelude::*;

#[gpu_kernel(block_size = 256)]
fn fused_relu(x: &[f32], out: &mut [f32], n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        if x[idx] > 0.0 {
            out[idx] = x[idx];
        } else {
            out[idx] = 0.0;
        }
    }
}

#[test]
#[ignore] // requires NVIDIA GPU
fn fused_relu_correctness() {
    let device = KaioDevice::new(0).expect("GPU required");

    let n: u32 = 1024;
    let x_host: Vec<f32> = (0..n).map(|i| (i as f32) - 512.0).collect(); // -512 to +511
    let expected: Vec<f32> = x_host
        .iter()
        .map(|&x| if x > 0.0 { x } else { 0.0 })
        .collect();

    let x = device.alloc_from(&x_host).expect("alloc x");
    let mut out = device.alloc_zeros::<f32>(n as usize).expect("alloc out");

    fused_relu::launch(&device, &x, &mut out, n).expect("launch failed");

    let result = out.to_host(&device).expect("to_host");
    assert_eq!(result, expected, "fused_relu produced wrong results");
}
