//! LayerNorm — `out[i] = ((x[i] - mean) * inv_std) * gamma[i] + beta[i]`
//! where `inv_std = 1 / sqrt(var + eps)`.
//!
//! LayerNorm is the classic transformer normalization used by BERT,
//! GPT-2, T5, and many encoder-decoder stacks. RMSNorm replaced it in
//! the LLaMA family, but LayerNorm remains one of the most recognizable
//! ML primitives and a good demonstration of block-wide reductions.
//!
//! # Single-block caveat
//!
//! This example intentionally does the single-block case:
//! `hidden_dim = 256`, `block_size = 256`. Real model LayerNorm often
//! runs over hidden sizes like 768, 1024, 4096, or larger, which
//! requires cross-block coordination or a multi-kernel split. This
//! example is about expressing the core math cleanly with KAIO's current
//! DSL, not claiming full production LayerNorm coverage yet.
//!
//! Run: `cargo run --release` from this directory.

use std::time::Instant;

use kaio::prelude::*;

const HIDDEN_DIM: u32 = 256;
const BLOCK_SIZE: u32 = 256;
const EPS: f32 = 1e-5;
const WARMUP_RUNS: usize = 5;
const TIMED_RUNS: usize = 100;

#[gpu_kernel(block_size = 256)]
fn layer_norm(x: *const [f32], gamma: *const [f32], beta: *const [f32], out: *mut [f32], n: u32, eps: f32) {
    let tid = thread_idx_x();

    // Single-block kernel: one block covers the entire vector. Threads
    // beyond n contribute neutral elements into the reductions.
    let mut val = 0.0f32;
    if tid < n {
        val = x[tid];
    }

    let sum = block_reduce_sum(val);
    let mean = sum / (n as f32);

    let mut centered = 0.0f32;
    if tid < n {
        centered = val - mean;
    }
    let var_sum = block_reduce_sum(centered * centered);
    let inv_std = 1.0f32 / sqrt(var_sum / (n as f32) + eps);

    if tid < n {
        out[tid] = centered * inv_std * gamma[tid] + beta[tid];
    }
}

fn cpu_reference(x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f64;
    let mean: f64 = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var: f64 = x
        .iter()
        .map(|&v| {
            let centered = v as f64 - mean;
            centered * centered
        })
        .sum::<f64>()
        / n;
    let inv_std = 1.0 / (var + eps as f64).sqrt();

    x.iter()
        .zip(gamma.iter())
        .zip(beta.iter())
        .map(|((&xi, &gi), &bi)| (((xi as f64 - mean) * inv_std) * gi as f64 + bi as f64) as f32)
        .collect()
}

fn max_abs_err(got: &[f32], expected: &[f32]) -> f32 {
    got.iter()
        .zip(expected.iter())
        .map(|(g, e)| (g - e).abs())
        .fold(0.0f32, f32::max)
}

fn median_latency_us<F: FnMut() -> Result<()>>(mut launch: F) -> Result<f64> {
    let mut times_us = Vec::with_capacity(TIMED_RUNS);
    for _ in 0..TIMED_RUNS {
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

    let n = HIDDEN_DIM;
    assert!(
        n <= BLOCK_SIZE,
        "single-block example requires n <= block_size"
    );

    let x_host: Vec<f32> = (0..n)
        .map(|i| ((i % 73) as f32 / 73.0) * 3.0 - 1.5)
        .collect();
    let gamma_host: Vec<f32> = (0..n)
        .map(|i| 0.8f32 + ((i % 29) as f32 / 29.0) * 0.4)
        .collect();
    let beta_host: Vec<f32> = (0..n)
        .map(|i| ((i % 31) as f32 / 31.0) * 0.2 - 0.1)
        .collect();

    let x = device.alloc_from(&x_host)?;
    let gamma = device.alloc_from(&gamma_host)?;
    let beta = device.alloc_from(&beta_host)?;
    let mut out = device.alloc_zeros::<f32>(n as usize)?;

    for _ in 0..WARMUP_RUNS {
        layer_norm::launch(&device, &x, &gamma, &beta, &mut out, n, EPS)?;
    }
    device.stream().synchronize()?;

    let got = out.to_host(&device)?;
    let expected = cpu_reference(&x_host, &gamma_host, &beta_host, EPS);
    let max_abs_err = max_abs_err(&got, &expected);
    let correctness = if max_abs_err < 2e-4 { "PASS" } else { "FAIL" };

    let median_us = median_latency_us(|| {
        device.stream().synchronize()?;
        layer_norm::launch(&device, &x, &gamma, &beta, &mut out, n, EPS)?;
        device.stream().synchronize()?;
        Ok(())
    })?;

    println!("=== layer_norm ===");
    println!("Input size:        {n} elements  (single-block — see README)");
    println!("Correctness:       {correctness}  (max_abs_err = {max_abs_err:.2e})");
    println!(
        "Median latency:    {median_us:.1} μs  (of {TIMED_RUNS} timed runs, {WARMUP_RUNS} warm-ups skipped)"
    );

    if correctness == "FAIL" {
        std::process::exit(1);
    }
    Ok(())
}
