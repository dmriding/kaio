//! Fused SiLU-gate — `out[i] = x[i] * sigmoid(x[i]) * gate[i]`.
//!
//! This is the gated activation inside every LLaMA / Mistral / Qwen
//! feedforward block. Production inference stacks ship hand-written
//! CUDA C++ for it; KAIO lets you write it as a 7-line Rust kernel.
//!
//! Run: `cargo run --release` from this directory.

use std::time::Instant;

use kaio::prelude::*;

#[gpu_kernel(block_size = 256)]
fn fused_silu_gate(x: &[f32], gate: &[f32], out: &mut [f32], n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        let xi = x[idx];
        let sig = 1.0f32 / (1.0f32 + exp(-xi));
        out[idx] = xi * sig * gate[idx];
    }
}

fn cpu_reference(x: &[f32], gate: &[f32]) -> Vec<f32> {
    x.iter()
        .zip(gate.iter())
        .map(|(&xi, &gi)| {
            let xi = xi as f64;
            let gi = gi as f64;
            let sig = 1.0_f64 / (1.0_f64 + (-xi).exp());
            (xi * sig * gi) as f32
        })
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

    let n: u32 = 1 << 20; // 1,048,576
    let x_host: Vec<f32> = (0..n)
        .map(|i| ((i % 97) as f32 / 97.0) - 0.5)
        .collect();
    let gate_host: Vec<f32> = (0..n)
        .map(|i| ((i % 113) as f32 / 113.0) - 0.5)
        .collect();

    let x = device.alloc_from(&x_host)?;
    let gate = device.alloc_from(&gate_host)?;
    let mut out = device.alloc_zeros::<f32>(n as usize)?;

    // Warm-up: first call compiles + loads the PTX module.
    for _ in 0..5 {
        fused_silu_gate::launch(&device, &x, &gate, &mut out, n)?;
    }
    device.stream().synchronize()?;

    // Correctness against an f64 CPU reference.
    let got = out.to_host(&device)?;
    let expected = cpu_reference(&x_host, &gate_host);
    let max_abs_err = got
        .iter()
        .zip(expected.iter())
        .map(|(g, e)| (g - e).abs())
        .fold(0.0f32, f32::max);
    let correctness = if max_abs_err < 1e-5 { "PASS" } else { "FAIL" };

    // Timing: 100 runs, median. Force device sync after each launch
    // so we're measuring kernel time, not an unsynced pipeline.
    let mut times_us = Vec::with_capacity(100);
    for _ in 0..100 {
        device.stream().synchronize()?;
        let start = Instant::now();
        fused_silu_gate::launch(&device, &x, &gate, &mut out, n)?;
        device.stream().synchronize()?;
        times_us.push(start.elapsed().as_secs_f64() * 1e6);
    }
    times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_us = times_us[times_us.len() / 2];

    println!("=== fused_silu_gate ===");
    println!("Input size:        {n} elements");
    println!("Correctness:       {correctness}  (max_abs_err = {max_abs_err:.2e})");
    println!("Median latency:    {median_us:.1} μs  (of 100 timed runs, 5 warm-ups skipped)");

    if correctness == "FAIL" {
        std::process::exit(1);
    }
    Ok(())
}
