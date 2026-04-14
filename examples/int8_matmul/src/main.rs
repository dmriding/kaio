//! INT8 symmetric dequantize-matmul — the shipped `matmul_int8` op.
//!
//! This example demonstrates the **full pipeline**: quantize two f32
//! matrices to i8, run tensor-core `matmul_int8` on-device, and compare
//! against a naive f32 CPU matmul reference. It complements
//! `examples/int8_dequant/`, which demonstrates the DSL shift-and-mask
//! dequant primitive in isolation.
//!
//! # What `matmul_int8` is (Sprint 7.1 / v0.3.0 reference quant op)
//!
//! - **W8A8** — both operands are quantized to i8. Mixed-precision
//!   W8A16 (i8 weights × f16 activations) is NOT supported; that
//!   would be a distinct future op.
//! - **Symmetric** (zero-point = 0). Asymmetric quant is a future
//!   additive refinement.
//! - **Single global scalar scale** applied post-accumulation —
//!   one `f32` for the full output.
//! - **`K % 32 == 0` required** (mma K-tile structural).
//! - **Sync-only** — async INT8 is a known follow-up.
//!
//! This is the reference quant op, not the final general-quant API.
//! GPTQ / AWQ / per-channel / per-group / INT4 all land as future
//! additive refinements.
//!
//! Run: `cargo run --release` from this directory.

use std::time::Instant;

use kaio::prelude::*;
use kaio_ops::matmul_int8;

const M: u32 = 256;
const N: u32 = 256;
const K: u32 = 256; // must be a multiple of 32

const WARMUP_RUNS: usize = 5;
const TIMED_RUNS: usize = 50;

/// Symmetric quantizer: `i8 = round(f32 / scale)` clamped to [-127, 127].
/// (Avoids i8::MIN=-128 to keep the scale symmetric round-trip clean.)
fn quantize(values: &[f32], scale: f32) -> Vec<i8> {
    values
        .iter()
        .map(|&x| {
            let q = (x / scale).round().clamp(-127.0, 127.0);
            q as i8
        })
        .collect()
}

/// Derive a per-tensor scale so the tensor's max abs value maps to ~127.
/// Real-world quantizers use calibration; this is the simplest choice
/// for a demo.
fn derive_scale(values: &[f32]) -> f32 {
    let max_abs = values.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    if max_abs == 0.0 {
        1.0
    } else {
        max_abs / 127.0
    }
}

fn cpu_matmul_f32(a: &[f32], b: &[f32], m: u32, n: u32, k: u32) -> Vec<f32> {
    let mut out = vec![0.0f32; (m * n) as usize];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for kk in 0..k {
                acc += a[(i * k + kk) as usize] as f64 * b[(kk * n + j) as usize] as f64;
            }
            out[(i * n + j) as usize] = acc as f32;
        }
    }
    out
}

fn max_abs_err(got: &[f32], want: &[f32]) -> f32 {
    got.iter()
        .zip(want.iter())
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f32, f32::max)
}

fn max_rel_err(got: &[f32], want: &[f32]) -> f32 {
    got.iter()
        .zip(want.iter())
        .map(|(g, w)| {
            let denom = w.abs().max(1.0);
            (g - w).abs() / denom
        })
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

    assert!(
        K.is_multiple_of(32),
        "matmul_int8 requires K % 32 == 0 (got K={K})"
    );
    let (major, _) = info.compute_capability;
    assert!(
        major >= 8,
        "matmul_int8 requires SM 8.0+ (Ampere or newer)"
    );

    // --- Generate deterministic f32 inputs in [-1.0, 1.0] ---
    let mk = (M * K) as usize;
    let kn = (K * N) as usize;
    let a_f32: Vec<f32> = (0..mk)
        .map(|i| ((i * 2654435761usize) % 10000) as f32 / 5000.0 - 1.0)
        .collect();
    let b_f32: Vec<f32> = (0..kn)
        .map(|i| (((i + 12345) * 40503usize) % 10000) as f32 / 5000.0 - 1.0)
        .collect();

    // --- Symmetric per-tensor quantization ---
    // For the output of matmul(a, b), the combined scale is scale_a * scale_b:
    //   a_q[i] ≈ a[i] / scale_a     (i8)
    //   b_q[i] ≈ b[i] / scale_b     (i8)
    //   c[i] = sum(a_q * b_q) * (scale_a * scale_b)
    let scale_a = derive_scale(&a_f32);
    let scale_b = derive_scale(&b_f32);
    let scale = scale_a * scale_b;
    let a_i8 = quantize(&a_f32, scale_a);
    let b_i8 = quantize(&b_f32, scale_b);

    // --- GPU run ---
    let a_gpu = device.alloc_from(&a_i8)?;
    let b_gpu = device.alloc_from(&b_i8)?;
    let mut c_gpu = device.alloc_zeros::<f32>((M * N) as usize)?;

    for _ in 0..WARMUP_RUNS {
        matmul_int8(&device, &a_gpu, &b_gpu, &mut c_gpu, scale, M, N, K)?;
    }
    device.stream().synchronize()?;

    let got: Vec<f32> = c_gpu.to_host(&device)?;

    // --- CPU reference: naive f32 matmul on the ORIGINAL f32 inputs ---
    // Comparison tolerance reflects both (a) fp32 accumulation noise and
    // (b) the quantization error introduced by the i8 round-trip. Real
    // quantized inference tolerates this — the example uses ~1% relative
    // tolerance, which is loose enough to pass with K=256 but tight
    // enough to catch a broken kernel.
    let want = cpu_matmul_f32(&a_f32, &b_f32, M, N, K);
    let abs_err = max_abs_err(&got, &want);
    let rel_err = max_rel_err(&got, &want);
    // Tolerance reflects *quantization* error, not kernel error. For
    // random-ish inputs at K=256, per-tensor i8 round-trip accumulates
    // to roughly a few percent max-rel. A broken kernel produces radically
    // worse error (>> 1x or structural NaN/zero patterns), so 10% comfortably
    // catches kernel regressions without flaking on legit quant noise.
    let correctness = if rel_err < 0.10 { "PASS" } else { "FAIL" };

    // --- Timing ---
    let median_us = median_latency_us(|| {
        device.stream().synchronize()?;
        matmul_int8(&device, &a_gpu, &b_gpu, &mut c_gpu, scale, M, N, K)?;
        device.stream().synchronize()?;
        Ok(())
    })?;

    let ops = 2.0 * (M as f64) * (N as f64) * (K as f64);
    let gflops = ops / (median_us * 1e3);

    println!("=== int8_matmul (W8A8 tensor-core) ===");
    println!("Shape:             {M}x{N}x{K}  (M x N x K)");
    println!("scale_a:           {scale_a:.6}");
    println!("scale_b:           {scale_b:.6}");
    println!("combined scale:    {scale:.6}");
    println!(
        "Correctness vs f32 naive: {correctness}  (max_abs_err = {abs_err:.2e}, max_rel_err = {rel_err:.2e})"
    );
    println!(
        "Median latency:    {median_us:.1} μs   ({gflops:.1} GFLOPS i8 ops)  ({TIMED_RUNS} timed / {WARMUP_RUNS} warmup)"
    );
    println!();
    println!("Note: accuracy loss here reflects i8 round-trip quantization,");
    println!("not kernel error. Run `cargo test -p kaio-ops --test matmul_int8_e2e`");
    println!("with --ignored for bit-exact i8 x i8 -> i32 round-trip validation.");

    if correctness == "FAIL" {
        std::process::exit(1);
    }
    Ok(())
}
