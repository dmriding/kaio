//! GELU activation — exact (tanh) vs fast (sigmoid) side-by-side.
//!
//! GELU is the activation in BERT / GPT / most modern encoder-decoder
//! transformers. The exact form uses a tanh-based approximation that
//! matches the Gaussian CDF to within ~1e-4; the fast form trades
//! precision (~5e-3) for fewer ops.
//!
//! This example is the teaching moment for kernel-variant workflows:
//! write two, measure both, pick the winner. Triton sells itself on
//! exactly this pattern — KAIO does it in ordinary Rust.
//!
//! Run: `cargo run --release` from this directory.

use std::time::Instant;

use kaio::prelude::*;

// Exact GELU (tanh approximation):
//   0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
#[gpu_kernel(block_size = 256)]
fn gelu_exact(x: *const [f32], out: *mut [f32], n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        let xi = x[idx];
        let c: f32 = 0.7978845608028654f32; // sqrt(2/π)
        let inner = c * (xi + 0.044715f32 * xi * xi * xi);
        out[idx] = 0.5f32 * xi * (1.0f32 + tanh(inner));
    }
}

// Fast GELU (sigmoid approximation):
//   x * sigmoid(1.702 * x) = x / (1 + exp(-1.702 * x))
#[gpu_kernel(block_size = 256)]
fn gelu_fast(x: *const [f32], out: *mut [f32], n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        let xi = x[idx];
        out[idx] = xi / (1.0f32 + exp(-1.702f32 * xi));
    }
}

fn cpu_gelu_exact(x: &[f32]) -> Vec<f32> {
    let c = (2.0_f64 / std::f64::consts::PI).sqrt();
    x.iter()
        .map(|&xi| {
            let xi = xi as f64;
            let inner = c * (xi + 0.044715 * xi * xi * xi);
            (0.5 * xi * (1.0 + inner.tanh())) as f32
        })
        .collect()
}

fn cpu_gelu_fast(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|&xi| {
            let xi = xi as f64;
            (xi / (1.0 + (-1.702_f64 * xi).exp())) as f32
        })
        .collect()
}

fn max_abs_err(got: &[f32], expected: &[f32]) -> f32 {
    got.iter()
        .zip(expected.iter())
        .map(|(g, e)| (g - e).abs())
        .fold(0.0f32, f32::max)
}

fn bench<F: FnMut() -> Result<()>>(mut launch: F) -> Result<f64> {
    let mut times_us = Vec::with_capacity(100);
    for _ in 0..100 {
        let start = Instant::now();
        launch()?;
        times_us.push(start.elapsed().as_secs_f64() * 1e6);
    }
    times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(times_us[times_us.len() / 2])
}

fn main() -> Result<()> {
    let device = KaioDevice::new(0)?;
    let info = device.info()?;
    println!("GPU: {:?}", info.name);
    println!(
        "Compute capability: sm_{}{}",
        info.compute_capability.0, info.compute_capability.1
    );
    println!();

    let n: u32 = 1 << 20; // 1,048,576
    let x_host: Vec<f32> = (0..n)
        .map(|i| ((i % 197) as f32 / 197.0) * 6.0 - 3.0)
        .collect();

    let x = device.alloc_from(&x_host)?;
    let mut out_exact = device.alloc_zeros::<f32>(n as usize)?;
    let mut out_fast = device.alloc_zeros::<f32>(n as usize)?;

    // Warm-up both kernels.
    for _ in 0..5 {
        gelu_exact::launch(&device, &x, &mut out_exact, n)?;
        gelu_fast::launch(&device, &x, &mut out_fast, n)?;
    }
    device.stream().synchronize()?;

    // Correctness: compare each kernel to its own f64 reference.
    let got_exact = out_exact.to_host(&device)?;
    let got_fast = out_fast.to_host(&device)?;
    let ref_exact = cpu_gelu_exact(&x_host);
    let ref_fast = cpu_gelu_fast(&x_host);
    let err_exact = max_abs_err(&got_exact, &ref_exact);
    let err_fast = max_abs_err(&got_fast, &ref_fast);
    let pass_exact = if err_exact < 1e-4 { "PASS" } else { "FAIL" };
    let pass_fast = if err_fast < 5e-3 { "PASS" } else { "FAIL" };

    // Timing each variant.
    let median_exact_us = bench(|| {
        device.stream().synchronize()?;
        gelu_exact::launch(&device, &x, &mut out_exact, n)?;
        device.stream().synchronize()?;
        Ok(())
    })?;
    let median_fast_us = bench(|| {
        device.stream().synchronize()?;
        gelu_fast::launch(&device, &x, &mut out_fast, n)?;
        device.stream().synchronize()?;
        Ok(())
    })?;
    let ratio_pct = median_fast_us / median_exact_us * 100.0;

    println!("=== gelu_comparison ===");
    println!("Input size:        {n} elements");
    println!();
    println!(
        "Exact (tanh):      {pass_exact}  (max_abs_err = {err_exact:.2e})  — {median_exact_us:.1} μs"
    );
    println!(
        "Fast (sigmoid):    {pass_fast}  (max_abs_err = {err_fast:.2e})  — {median_fast_us:.1} μs"
    );
    println!();
    println!("Fast is {ratio_pct:.1}% of exact's time.");

    if pass_exact == "FAIL" || pass_fast == "FAIL" {
        std::process::exit(1);
    }
    Ok(())
}
