//! RMSNorm — `out[i] = (x[i] / rms) * weight[i]` where
//! `rms = sqrt(mean(x²) + eps)`.
//!
//! RMSNorm is the normalization LLaMA adopted in place of LayerNorm —
//! cheaper (no mean subtraction, no bias) with comparable training
//! stability. The entire LLaMA / Mistral / Qwen family uses it.
//!
//! # Single-block caveat
//!
//! This example does the single-block case: `hidden_dim = 256`,
//! `block_size = 256`. Real LLaMA RMSNorm operates over
//! `hidden_dim = 4096`, which is 16 blocks at `block_size = 256` and
//! requires cross-block reduction (either atomic accumulation into a
//! scratch buffer or a two-kernel split). Multi-block RMSNorm lands
//! when KAIO's builtin set gains cross-block primitives, or ships as
//! a pre-built `kaio_ops::rms_norm`. Post-v0.2.0 work.
//!
//! Run: `cargo run --release` from this directory.

use std::time::Instant;

use kaio::prelude::*;

const HIDDEN_DIM: u32 = 256;
const EPS: f32 = 1e-6;

#[gpu_kernel(block_size = 256)]
fn rms_norm(x: *const [f32], weight: *const [f32], out: *mut [f32], n: u32, eps: f32) {
    let tid = thread_idx_x();
    // Single-block kernel: one block covers the entire hidden dim.
    // Each thread handles exactly one element.
    let mut val = 0.0f32;
    if tid < n {
        val = x[tid];
    }
    let sq = val * val;

    // Block-wide sum of squares.
    let sum_sq = block_reduce_sum(sq);

    // Every thread now has the same sum_sq (block_reduce_sum broadcasts).
    let inv_rms = 1.0f32 / sqrt(sum_sq / (n as f32) + eps);

    if tid < n {
        out[tid] = val * inv_rms * weight[tid];
    }
}

fn cpu_reference(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f64;
    let sum_sq: f64 = x.iter().map(|&v| (v as f64).powi(2)).sum();
    let inv_rms = 1.0 / ((sum_sq / n) + eps as f64).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(&xi, &wi)| ((xi as f64) * inv_rms * (wi as f64)) as f32)
        .collect()
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

    let n = HIDDEN_DIM;
    let x_host: Vec<f32> = (0..n)
        .map(|i| ((i % 131) as f32 / 131.0) * 2.0 - 1.0)
        .collect();
    let weight_host: Vec<f32> = (0..n)
        .map(|i| 0.5f32 + ((i % 17) as f32 / 17.0))
        .collect();

    let x = device.alloc_from(&x_host)?;
    let weight = device.alloc_from(&weight_host)?;
    let mut out = device.alloc_zeros::<f32>(n as usize)?;

    // Warm-up.
    for _ in 0..5 {
        rms_norm::launch(&device, &x, &weight, &mut out, n, EPS)?;
    }
    device.stream().synchronize()?;

    // Correctness.
    let got = out.to_host(&device)?;
    let expected = cpu_reference(&x_host, &weight_host, EPS);
    let max_abs_err = got
        .iter()
        .zip(expected.iter())
        .map(|(g, e)| (g - e).abs())
        .fold(0.0f32, f32::max);
    let correctness = if max_abs_err < 1e-4 { "PASS" } else { "FAIL" };

    // Timing.
    let mut times_us = Vec::with_capacity(100);
    for _ in 0..100 {
        device.stream().synchronize()?;
        let start = Instant::now();
        rms_norm::launch(&device, &x, &weight, &mut out, n, EPS)?;
        device.stream().synchronize()?;
        times_us.push(start.elapsed().as_secs_f64() * 1e6);
    }
    times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_us = times_us[times_us.len() / 2];

    println!("=== rms_norm ===");
    println!("Input size:        {n} elements  (single-block — see README)");
    println!("Correctness:       {correctness}  (max_abs_err = {max_abs_err:.2e})");
    println!("Median latency:    {median_us:.1} μs  (of 100 timed runs, 5 warm-ups skipped)");

    if correctness == "FAIL" {
        std::process::exit(1);
    }
    Ok(())
}
