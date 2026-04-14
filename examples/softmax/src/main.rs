//! Softmax — `out[i] = exp(x[i] - max(x)) / sum_j exp(x[j] - max(x))`.
//!
//! Softmax is the normalization at the heart of attention: scores come
//! in, probabilities come out. It is also the canonical example of a
//! numerically stable reduction pattern on GPU: subtract the row max,
//! exponentiate, sum, normalize.
//!
//! # Single-block caveat
//!
//! This example intentionally handles only the single-block case
//! `n <= 256`. Real attention workloads often normalize rows much
//! larger than one block and require striding or tiled multi-block
//! reduction. This example stays inside KAIO's current supported
//! feature set and demonstrates the core pattern cleanly.
//!
//! Run: `cargo run --release` from this directory.

use std::time::Instant;

use kaio::prelude::*;

const ROW_LEN: u32 = 256;
const BLOCK_SIZE: u32 = 256;
const WARMUP_RUNS: usize = 5;
const TIMED_RUNS: usize = 100;

#[gpu_kernel(block_size = 256)]
fn softmax(input: &[f32], output: &mut [f32], n: u32) {
    let tid = thread_idx_x();

    let mut local_max = -3.402823e38f32;
    if tid < n {
        local_max = input[tid];
    }
    let row_max = block_reduce_max(local_max);

    let mut exp_val = 0.0f32;
    if tid < n {
        exp_val = exp(local_max - row_max);
    }
    let row_sum = block_reduce_sum(exp_val);

    if tid < n {
        output[tid] = exp_val / row_sum;
    }
}

fn cpu_reference(input: &[f32]) -> Vec<f32> {
    let max = input
        .iter()
        .copied()
        .map(|v| v as f64)
        .fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = input.iter().map(|&x| (x as f64 - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| (e / sum) as f32).collect()
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

    let n = ROW_LEN;
    assert!(
        n <= BLOCK_SIZE,
        "single-block example requires n <= block_size"
    );

    let input_host: Vec<f32> = (0..n)
        .map(|i| ((i % 101) as f32 / 101.0) * 12.0 - 6.0)
        .collect();

    let input = device.alloc_from(&input_host)?;
    let mut output = device.alloc_zeros::<f32>(n as usize)?;

    for _ in 0..WARMUP_RUNS {
        softmax::launch(&device, &input, &mut output, n)?;
    }
    device.stream().synchronize()?;

    let got = output.to_host(&device)?;
    let expected = cpu_reference(&input_host);
    let max_abs_err = max_abs_err(&got, &expected);
    let correctness = if max_abs_err < 1e-5 { "PASS" } else { "FAIL" };

    let median_us = median_latency_us(|| {
        device.stream().synchronize()?;
        softmax::launch(&device, &input, &mut output, n)?;
        device.stream().synchronize()?;
        Ok(())
    })?;

    println!("=== softmax ===");
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
