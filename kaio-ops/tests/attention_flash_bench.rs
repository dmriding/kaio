#![allow(clippy::too_many_arguments)]

//! Benchmark: KAIO FlashAttention — `attention_flash` +
//! `attention_flash_causal` at canonical self-attention shapes.
//!
//! Run with:
//! ```sh
//! cargo test -p kaio-ops --test attention_flash_bench -- --ignored --nocapture
//! ```
//!
//! # What is measured
//!
//! Single-head self-attention latency and derived throughput at four
//! sequence lengths (128, 512, 1024, 2048), head dim 128. Each row
//! reports median-of-20 wall-clock after 5 warm-up launches. No
//! cuDNN / cuBLAS baseline — KAIO's FlashAttention surface doesn't
//! have a directly-comparable reference in `cudarc` 0.19.
//!
//! # Shape regime
//!
//! The public `attention_flash` signature is
//! `(q, k, v, out, seq_len, d_k)`. The kernel is strictly
//! self-attention (`seq_q == seq_k == seq_len`) with `d_v == d_k`.
//! Because the online-softmax formulation materializes no score
//! matrix, seq_len is not capped by shared memory — this is the
//! long-sequence complement to `attention_tc` (which caps at
//! `seq_k <= 384`).
//!
//! # Inputs
//!
//! f32 Q/K/V, f32 output (distinct from `attention_tc`'s f16 inputs).
//! Apples-to-apples across the two attention benches is time at the
//! same `(seq_len, d_k)`, not input-dtype-matched. Deterministic
//! pseudo-random in `[-1, 1]` with fixed seeds (42 Q, 137 K, 7 V).

use std::time::Instant;

use kaio::prelude::*;
use kaio_ops::{attention_flash, attention_flash_causal};

fn deterministic_f32(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let f = ((state >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0;
            f * 0.25 // dampen to keep post-softmax scale sane
        })
        .collect()
}

fn median_seconds<F: FnMut()>(mut launch: F, warmup: usize, iters: usize) -> f64 {
    for _ in 0..warmup {
        launch();
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        launch();
        times.push(start.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[iters / 2]
}

fn print_header() {
    eprintln!(
        "{:<10} {:<10} {:>12} {:>14} {:>18}",
        "shape", "variant", "median ms", "seq/s", "attn_scores/s"
    );
    eprintln!("{}", "-".repeat(70));
}

fn print_row(shape: &str, variant: &str, seq_len: u32, median_s: f64) {
    let scores_per_s = (seq_len as f64) * (seq_len as f64) / median_s;
    let seq_per_s = (seq_len as f64) / median_s;
    eprintln!(
        "{:<10} {:<10} {:>12.3} {:>14.1} {:>18.2e}",
        shape,
        variant,
        median_s * 1e3,
        seq_per_s,
        scores_per_s,
    );
}

const SHAPES: &[(&str, u32)] = &[
    ("n128", 128),
    ("n512", 512),
    ("n1024", 1024),
    ("n2048", 2048),
];

const D_HEAD: u32 = 128;

#[test]
#[ignore]
fn benchmark_attention_flash_self_attention() {
    let device = KaioDevice::new(0).expect("GPU required");
    let info = device.info().expect("device info");

    eprintln!();
    eprintln!(
        "=== KAIO attention_flash benchmark (single-head self-attention, f32 Q/K/V -> f32 out) ==="
    );
    eprintln!(
        "GPU: {:?}  sm_{}{}",
        info.name, info.compute_capability.0, info.compute_capability.1
    );
    eprintln!(
        "Warmup 5 | Iters 20 | Median | Shapes: seq = {{128, 512, 1024, 2048}}, d_k = {D_HEAD}"
    );
    eprintln!();
    print_header();

    for &(label, n) in SHAPES {
        let q = deterministic_f32((n * D_HEAD) as usize, 42);
        let k = deterministic_f32((n * D_HEAD) as usize, 137);
        let v = deterministic_f32((n * D_HEAD) as usize, 7);

        let q_gpu = device.alloc_from(&q).unwrap();
        let k_gpu = device.alloc_from(&k).unwrap();
        let v_gpu = device.alloc_from(&v).unwrap();
        let mut out_plain = device.alloc_zeros::<f32>((n * D_HEAD) as usize).unwrap();
        let mut out_causal = device.alloc_zeros::<f32>((n * D_HEAD) as usize).unwrap();

        let plain_s = median_seconds(
            || {
                attention_flash(&device, &q_gpu, &k_gpu, &v_gpu, &mut out_plain, n, D_HEAD)
                    .unwrap();
                device.stream().synchronize().unwrap();
            },
            5,
            20,
        );
        print_row(label, "plain", n, plain_s);

        let causal_s = median_seconds(
            || {
                attention_flash_causal(&device, &q_gpu, &k_gpu, &v_gpu, &mut out_causal, n, D_HEAD)
                    .unwrap();
                device.stream().synchronize().unwrap();
            },
            5,
            20,
        );
        print_row(label, "causal", n, causal_s);
    }

    eprintln!();
    eprintln!("Throughput definitions:");
    eprintln!("  seq/s             = seq_len / median_s");
    eprintln!("  attn_scores/s     = seq_len^2 / median_s  (single-head, self-attention)");
}
