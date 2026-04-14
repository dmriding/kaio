//! INT8 symmetric dequantization — `out[i] = (i8_value) * scale`.
//!
//! Weights are stored packed four-per-`u32`. Each thread unpacks one
//! `u32`, sign-extends each byte to `i32`, casts to `f32`, and applies
//! the scale. The sign-extension uses the shift-then-arithmetic-shift
//! trick, which requires the DSL to emit `shr.s32` on signed right
//! shifts (arithmetic shift) and `shr.u32` on unsigned ones (logical
//! shift). That distinction landed in KAIO v0.2.1.
//!
//! This example does NOT include the matmul half — it demonstrates
//! only the dequantize primitive. The full INT8 dequantize-matmul
//! fusion is the planned Phase 7.1 milestone (uses tensor-core INT8
//! `mma.sync` which isn't shipped yet).
//!
//! # Scope / limitations
//!
//! - Symmetric INT8 only: values in [-128, 127], `zero_point = 0`.
//!   Asymmetric quant would add `(byte - zero_point) * scale` instead.
//! - Single scalar `scale` across the whole tensor. Real quantized
//!   formats often use per-channel or per-group scales.
//! - Single-block: `n_words <= block_size = 256` (→ 1024 output
//!   elements). A multi-block grid launch is a one-line change but
//!   this example keeps the pattern visible.
//!
//! Run: `cargo run --release` from this directory.

use std::time::Instant;

use kaio::prelude::*;

const N_WORDS: u32 = 256;
const BLOCK_SIZE: u32 = 256;
const WARMUP_RUNS: usize = 5;
const TIMED_RUNS: usize = 100;

#[gpu_kernel(block_size = 256)]
fn dequant_i8(packed: &[u32], out: &mut [f32], scale: f32, n_words: u32) {
    let tid = thread_idx_x();

    if tid < n_words {
        let word = packed[tid];

        // Extract each byte and sign-extend to i32.
        // Pattern: (byte_u32 as i32) << 24 >> 24.
        //   - << 24 puts byte-7 into bit 31 (the i32 sign bit).
        //   - >> 24 on i32 is arithmetic shift — sign-extends, giving
        //     the correct signed value back. Requires `shr.s32` on
        //     signed operands (KAIO v0.2.1 AD2).
        let b0 = (((word & 0xFF) as i32) << 24) >> 24;
        let b1 = ((((word >> 8) & 0xFF) as i32) << 24) >> 24;
        let b2 = ((((word >> 16) & 0xFF) as i32) << 24) >> 24;
        let b3 = ((((word >> 24) & 0xFF) as i32) << 24) >> 24;

        let base = tid * 4;
        out[base] = (b0 as f32) * scale;
        out[base + 1] = (b1 as f32) * scale;
        out[base + 2] = (b2 as f32) * scale;
        out[base + 3] = (b3 as f32) * scale;
    }
}

/// Pack four signed i8 values into a u32, low byte = first.
fn pack_i8x4(a: i8, b: i8, c: i8, d: i8) -> u32 {
    (a as u8 as u32) | ((b as u8 as u32) << 8) | ((c as u8 as u32) << 16) | ((d as u8 as u32) << 24)
}

fn cpu_reference(packed: &[u32], scale: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(packed.len() * 4);
    for &word in packed {
        let bytes = [
            (word & 0xFF) as u8 as i8,
            ((word >> 8) & 0xFF) as u8 as i8,
            ((word >> 16) & 0xFF) as u8 as i8,
            ((word >> 24) & 0xFF) as u8 as i8,
        ];
        for &b in &bytes {
            out.push(b as f32 * scale);
        }
    }
    out
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

    let n_words = N_WORDS;
    assert!(
        n_words <= BLOCK_SIZE,
        "single-block example requires n_words <= block_size"
    );

    // Deterministic input covering the full i8 range, including negatives
    // (to exercise the sign-extension path) and edge values (-128, 127).
    let packed_host: Vec<u32> = (0..n_words)
        .map(|i| {
            let base = (i as i32) * 4;
            let b0 = ((base - 512).clamp(-128, 127)) as i8;
            let b1 = ((base + 1 - 400).clamp(-128, 127)) as i8;
            let b2 = ((base + 2 - 300).clamp(-128, 127)) as i8;
            let b3 = ((base + 3 - 200).clamp(-128, 127)) as i8;
            pack_i8x4(b0, b1, b2, b3)
        })
        .collect();
    let scale: f32 = 0.0125; // i8 range [-128, 127] maps roughly to [-1.6, 1.59].

    let packed = device.alloc_from(&packed_host)?;
    let mut out = device.alloc_zeros::<f32>((n_words * 4) as usize)?;

    for _ in 0..WARMUP_RUNS {
        dequant_i8::launch(&device, &packed, &mut out, scale, n_words)?;
    }
    device.stream().synchronize()?;

    let got = out.to_host(&device)?;
    let expected = cpu_reference(&packed_host, scale);
    let err = max_abs_err(&got, &expected);
    let correctness = if err < 1e-6 { "PASS" } else { "FAIL" };

    let median_us = median_latency_us(|| {
        device.stream().synchronize()?;
        dequant_i8::launch(&device, &packed, &mut out, scale, n_words)?;
        device.stream().synchronize()?;
        Ok(())
    })?;

    println!("=== int8_dequant ===");
    println!(
        "Packed input:      {n_words} u32 words  ({} i8 values, single-block)",
        n_words * 4
    );
    println!("Scale:             {scale}");
    println!("Correctness:       {correctness}  (max_abs_err = {err:.2e})");
    println!(
        "Median latency:    {median_us:.1} μs  (of {TIMED_RUNS} timed runs, {WARMUP_RUNS} warm-ups skipped)"
    );

    if correctness == "FAIL" {
        std::process::exit(1);
    }
    Ok(())
}
