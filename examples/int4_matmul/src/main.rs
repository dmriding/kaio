//! INT4 GPTQ-style dequantize-matmul — the shipped `matmul_int4` op.
//!
//! This example demonstrates the **full pipeline**: quantize an f32
//! weight matrix using symmetric GPTQ-lite per-column group
//! quantization, pack nibbles into the KAIO `[K/8, N]` col-major u32
//! layout, convert f32 activations to f16, run `matmul_int4` on-device,
//! and compare against a naive f32 CPU matmul reference.
//!
//! # What `matmul_int4` is (Sprint 7.2 reference quant op)
//!
//! - **W4A16** — signed INT4 packed weights × f16 activations → f32 output.
//! - **Symmetric, group-quantized** — one f16 scale per (output column, group).
//! - **Group size fixed at 128** — K % 128 == 0 required.
//! - **DEQUANT-F16 path** — per-lane unpack + sign-extend + cvt + scale fold,
//!   feeding `mma.sync.m16n8k16.f16.f16.f32`. No native INT4 tensor core
//!   on sm_80+, so dequant-then-mma is the only viable path.
//!
//! # Convention — NOT compatible with external GPTQ/GGUF weight formats
//!
//! `matmul_int4` defines its own packed weight + scale layout (see the
//! [`pack_s4_weights`] helper below for the reference CPU packer).
//! Users with pre-quantized AutoGPTQ / exllama / GGUF models must
//! repack to the KAIO convention before calling `matmul_int4`. See
//! the rustdoc on `kaio_ops::matmul_int4` for the exact indexing
//! formula.
//!
//! Run: `cargo run --release` from this directory.

use std::time::Instant;

use half::f16;
use kaio::prelude::*;
use kaio_ops::matmul_int4;

const M: u32 = 256;
const N: u32 = 256;
const K: u32 = 256; // must be a multiple of GROUP_SIZE = 128
const GROUP_SIZE: u32 = 128;

const WARMUP_RUNS: usize = 5;
const TIMED_RUNS: usize = 50;

/// GPTQ-lite symmetric per-column group quantizer. For each (column,
/// group) pair, pick `scale = max(|w|) / 7` and quantize as
/// `q = clamp(round(w / scale), -8, +7)`.
///
/// Returns `(quant_weights: [K, N] i8, scales: [K/group_size, N] f16)`.
fn quantize_gptq_lite(w: &[f32], k: usize, n: usize, group_size: usize) -> (Vec<i8>, Vec<f16>) {
    assert!(
        k.is_multiple_of(group_size),
        "K must be a multiple of group_size"
    );
    let num_groups = k / group_size;
    let mut q = vec![0i8; k * n];
    let mut scales = vec![f16::from_f32(0.0); num_groups * n];
    for col in 0..n {
        for g in 0..num_groups {
            let k_lo = g * group_size;
            let k_hi = k_lo + group_size;
            let mut max_abs = 0.0f32;
            for kk in k_lo..k_hi {
                let v = w[kk * n + col].abs();
                if v > max_abs {
                    max_abs = v;
                }
            }
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 7.0 };
            scales[g * n + col] = f16::from_f32(scale);
            for kk in k_lo..k_hi {
                let v = w[kk * n + col];
                let qi = (v / scale).round().clamp(-8.0, 7.0) as i8;
                q[kk * n + col] = qi;
            }
        }
    }
    (q, scales)
}

/// Pack `[K, N]` signed-INT4 weights into the KAIO `[K/8, N]`
/// col-major u32 layout (8 nibbles per u32, K-contiguous).
/// Matches the formula documented at `kaio_ops::matmul_int4`.
fn pack_s4_weights(w: &[i8], k: usize, n: usize) -> Vec<u32> {
    assert!(k.is_multiple_of(8), "K must be a multiple of 8");
    let k_words = k / 8;
    let mut out = vec![0u32; k_words * n];
    for col in 0..n {
        for word_idx in 0..k_words {
            let mut word = 0u32;
            for nibble_idx in 0..8 {
                let k_pos = word_idx * 8 + nibble_idx;
                let val = w[k_pos * n + col];
                let nibble = (val as u32) & 0xF;
                word |= nibble << (4 * nibble_idx);
            }
            out[word_idx + col * k_words] = word;
        }
    }
    out
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
        K.is_multiple_of(GROUP_SIZE),
        "matmul_int4 requires K % {GROUP_SIZE} == 0 (got K={K})"
    );
    let (major, _) = info.compute_capability;
    assert!(
        major >= 8,
        "matmul_int4 requires SM 8.0+ (Ampere or newer)"
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

    // --- GPTQ-lite quantize + pack weights ---
    let (b_s4, b_scales) = quantize_gptq_lite(&b_f32, K as usize, N as usize, GROUP_SIZE as usize);
    let b_packed = pack_s4_weights(&b_s4, K as usize, N as usize);

    // --- Convert activations to f16 ---
    let a_f16: Vec<f16> = a_f32.iter().map(|&x| f16::from_f32(x)).collect();

    // --- GPU run ---
    let a_gpu = device.alloc_from(&a_f16)?;
    let b_gpu = device.alloc_from(&b_packed)?;
    let s_gpu = device.alloc_from(&b_scales)?;
    let mut c_gpu = device.alloc_zeros::<f32>((M * N) as usize)?;

    for _ in 0..WARMUP_RUNS {
        matmul_int4(&device, &a_gpu, &b_gpu, &s_gpu, &mut c_gpu, M, N, K, GROUP_SIZE)?;
    }
    device.stream().synchronize()?;
    let got: Vec<f32> = c_gpu.to_host(&device)?;

    // --- CPU reference on the ORIGINAL f32 inputs ---
    // Comparison tolerance reflects **INT4 quantization error** (the
    // f32 → s4 + group-scale round-trip), NOT kernel error. Signed
    // INT4 has only 16 representable values — 16× fewer than INT8 —
    // so per-element quant noise is an order of magnitude larger than
    // the INT8 showcase's 10% threshold. For random-ish uniform inputs
    // at K=256 with group_size=128 and a naive max-abs group scale,
    // per-output-cell max-rel error commonly lands in the 50-80%
    // range. Real GPTQ uses activation-aware scaling + error
    // compensation to tighten this; the showcase uses the simplest
    // possible scheme to keep the example focused on the kernel path.
    //
    // A broken kernel produces radically worse error (>> 1× or
    // structural NaN / zero patterns), so an 80% max-rel threshold
    // comfortably catches kernel regressions without flaking on
    // legitimate INT4 quant noise.
    //
    // For bit-exact kernel-correctness tests (GPU output matches the
    // exact f16 dequant-chain CPU reference), see
    // `cargo test -p kaio-ops --test matmul_int4_e2e -- --ignored`.
    let want = cpu_matmul_f32(&a_f32, &b_f32, M, N, K);
    let abs_err = max_abs_err(&got, &want);
    let rel_err = max_rel_err(&got, &want);
    let correctness = if rel_err < 0.80 { "PASS" } else { "FAIL" };

    // --- Timing ---
    let median_us = median_latency_us(|| {
        device.stream().synchronize()?;
        matmul_int4(&device, &a_gpu, &b_gpu, &s_gpu, &mut c_gpu, M, N, K, GROUP_SIZE)?;
        device.stream().synchronize()?;
        Ok(())
    })?;

    let ops = 2.0 * (M as f64) * (N as f64) * (K as f64);
    let gflops = ops / (median_us * 1e3);

    println!("=== int4_matmul (W4A16 GPTQ-style tensor-core) ===");
    println!("Shape:                   {M}x{N}x{K}  (M x N x K, group_size={GROUP_SIZE})");
    println!("Num groups per column:   {}", K / GROUP_SIZE);
    println!(
        "Correctness vs f32 naive: {correctness}  (max_abs_err = {abs_err:.2e}, max_rel_err = {rel_err:.2e})"
    );
    println!(
        "Median latency:          {median_us:.1} μs   ({gflops:.1} GFLOPS int4 ops)  ({TIMED_RUNS} timed / {WARMUP_RUNS} warmup)"
    );
    println!();
    println!("Note: accuracy loss here reflects the GPTQ-lite quantization");
    println!("round-trip, not kernel error. See kaio-ops/tests/matmul_int4_e2e.rs");
    println!("for bit-exact GPU vs CPU-dequant-chain round-trip correctness tests.");

    if correctness == "FAIL" {
        std::process::exit(1);
    }
    Ok(())
}
