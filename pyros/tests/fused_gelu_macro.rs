//! E2E test: fused_gelu through the #[gpu_kernel] macro.
//!
//! GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//!
//! Tests complex arithmetic chains + math builtins (tanh).
//! Numerical accuracy validated within tolerance from success-criteria.md.

use pyros::prelude::*;

#[gpu_kernel(block_size = 256)]
fn fused_gelu(x: &[f32], out: &mut [f32], n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        let val = x[idx];
        let cube = val * val * val;
        let inner = 0.7978845 * (val + 0.044715 * cube);
        let t = tanh(inner);
        out[idx] = 0.5 * val * (1.0 + t);
    }
}

/// CPU reference GELU for validation.
fn cpu_gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0_f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

#[test]
#[ignore] // requires NVIDIA GPU
fn fused_gelu_correctness() {
    let device = PyrosDevice::new(0).expect("GPU required");

    let n: u32 = 1024;
    let x_host: Vec<f32> = (0..n).map(|i| (i as f32 - 512.0) * 0.01).collect(); // -5.12 to +5.11
    let expected: Vec<f32> = x_host.iter().map(|&x| cpu_gelu(x)).collect();

    let x = device.alloc_from(&x_host).expect("alloc x");
    let mut out = device.alloc_zeros::<f32>(n as usize).expect("alloc out");

    fused_gelu::launch(&device, &x, &mut out, n).expect("launch failed");

    let result = out.to_host(&device).expect("to_host");

    // Validate within tolerance (success-criteria.md: gelu max abs error 1e-4)
    let max_abs_error = result
        .iter()
        .zip(&expected)
        .map(|(got, exp)| (got - exp).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_abs_error < 1e-4,
        "fused_gelu max absolute error {max_abs_error} exceeds tolerance 1e-4"
    );
}
