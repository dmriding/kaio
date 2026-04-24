#![allow(clippy::too_many_arguments)]

//! Benchmark: KAIO tensor-core attention — `attention_tc` +
//! `attention_tc_causal` at canonical self-attention shapes.
//!
//! Run with:
//! ```sh
//! cargo test -p kaio-ops --test attention_tc_bench -- --ignored --nocapture
//! ```
//!
//! # What is measured
//!
//! Single-head self-attention latency and derived throughput at four
//! short-sequence shapes (64, 128, 256, 384), head dim 128. Each row
//! reports median-of-20 wall-clock after 5 warm-up launches. No
//! cuDNN / cuBLAS baseline — KAIO's attention surface doesn't have a
//! directly-comparable reference in `cudarc` 0.19. For long-sequence
//! coverage see `attention_flash_bench`.
//!
//! # Shape regime
//!
//! The public `attention_tc` signature is
//! `(q, k, v, out, seq_q, seq_k, d_k, d_v)`. The kernel has two hard
//! constraints:
//!
//! - `seq_q % 16 == 0` (BM block-tile constraint)
//! - `seq_k <= 384` (shared-memory scores buffer cap)
//!
//! The seq_k cap makes this a **short-sequence fused-TC attention**
//! kernel by design — long-sequence paths use `attention_flash`
//! (online softmax, no score materialization, supports seq_len up to
//! typical prefill sizes). The bench therefore uses a short-sequence
//! sweep matched to what the kernel is actually built for:
//! `n64, n128, n256, n384`. For longer-sequence coverage see the
//! FlashAttention bench (`attention_flash_bench`), which uses the
//! same column schema and extends to `n2048`.
//!
//! # Inputs
//!
//! f16 Q/K/V, f32 output. Deterministic pseudo-random in `[-1, 1]`
//! with fixed seeds (42 Q, 137 K, 7 V).

use std::time::Instant;

use half::f16;
use kaio::prelude::*;
use kaio_ops::{attention_tc, attention_tc_causal};

fn deterministic_f16(len: usize, seed: u64) -> Vec<f16> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let f = ((state >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0;
            f16::from_f32(f * 0.25) // dampen to keep post-softmax scale sane
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

const SHAPES: &[(&str, u32)] = &[("n64", 64), ("n128", 128), ("n256", 256), ("n384", 384)];

const D_HEAD: u32 = 128;

#[test]
#[ignore]
fn benchmark_attention_tc_self_attention() {
    let device = KaioDevice::new(0).expect("GPU required");
    let info = device.info().expect("device info");
    assert!(
        info.compute_capability.0 >= 8,
        "attention_tc requires SM 8.0+; got sm_{}{}",
        info.compute_capability.0,
        info.compute_capability.1
    );

    eprintln!();
    eprintln!(
        "=== KAIO attention_tc benchmark (single-head self-attention, f16 Q/K/V -> f32 out) ==="
    );
    eprintln!(
        "GPU: {:?}  sm_{}{}",
        info.name, info.compute_capability.0, info.compute_capability.1
    );
    eprintln!(
        "Warmup 5 | Iters 20 | Median | Shapes: seq = {{64, 128, 256, 384}}, d_k = d_v = {D_HEAD}"
    );
    eprintln!();
    print_header();

    for &(label, n) in SHAPES {
        let q = deterministic_f16((n * D_HEAD) as usize, 42);
        let k = deterministic_f16((n * D_HEAD) as usize, 137);
        let v = deterministic_f16((n * D_HEAD) as usize, 7);

        let q_gpu = device.alloc_from(&q).unwrap();
        let k_gpu = device.alloc_from(&k).unwrap();
        let v_gpu = device.alloc_from(&v).unwrap();
        let mut out_plain = device.alloc_zeros::<f32>((n * D_HEAD) as usize).unwrap();
        let mut out_causal = device.alloc_zeros::<f32>((n * D_HEAD) as usize).unwrap();

        let plain_s = median_seconds(
            || {
                attention_tc(
                    &device,
                    &q_gpu,
                    &k_gpu,
                    &v_gpu,
                    &mut out_plain,
                    n,
                    n,
                    D_HEAD,
                    D_HEAD,
                )
                .unwrap();
                device.stream().synchronize().unwrap();
            },
            5,
            20,
        );
        print_row(label, "plain", n, plain_s);

        let causal_s = median_seconds(
            || {
                attention_tc_causal(
                    &device,
                    &q_gpu,
                    &k_gpu,
                    &v_gpu,
                    &mut out_causal,
                    n,
                    n,
                    D_HEAD,
                    D_HEAD,
                )
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
