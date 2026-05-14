#![allow(clippy::too_many_arguments)]

//! Benchmark: KAIO tensor-core matmul **bf16 async** vs **f16 async**
//! (Sprint 9.1.1) plus a cuBLAS sgemm reference column. Includes the
//! **SC-2 perf-parity gate** at 4096³: 10 interleaved alternating-order
//! runs, with two independent bounds applied to the per-iter
//! bf16_async/f16_async TFLOPS ratios:
//!
//! - **Median ratio ±3%** — structural-kernel gate. Tight, noise-robust,
//!   measures whether the bf16 async kernel has a real perf delta from
//!   f16 async (which it shouldn't — same staging structure, only the
//!   mma operand dtype tag differs).
//! - **Worst ratio ±15%** — catastrophic-tail gate. Generous to OS noise,
//!   catches genuinely pathological tail behavior.
//!
//! Both bounds must hold.
//!
//! # Why bf16_async vs f16_async (not bf16_async vs bf16_sync)
//!
//! Same precision-isolation logic Sprint 9.1's SC-2 applied between
//! sync variants: cancel the staging variable, measure the precision
//! variable. The matching question for 9.1.1 is "did we pay for bf16
//! in the async staging path?" — answered by bf16_async vs f16_async.
//! The "is async-pipelining worth it on bf16?" question (bf16_async vs
//! bf16_sync) is per-shape and is answered by the future
//! `matmul_auto_tc_bf16` auto-tuner cache (Sprint 9.1.2), not by a
//! single binary gate here.
//!
//! # Same-run interleaving (no historical baseline gate)
//!
//! SC-2 interleaves bf16_async and f16_async in the **same** bench run
//! with alternating order. The same-run f16_async value is the
//! baseline for the precision-isolation ratio — no separate historical
//! anchor needed. Same-run interleaving cancels thermal drift and
//! toolchain/driver variations on its own.
//!
//! Historical f16_async numbers (from prior runs of this bench) are
//! informational only — the same-run f16_async median is printed
//! after the SC-2 verdict as a reference value for manual comparison.
//!
//! Run with:
//! ```sh
//! cargo xtask bench matmul_tc_bf16_async_bench
//! ```
//! or equivalently:
//! ```sh
//! cargo test --release -p kaio-ops --test matmul_tc_bf16_async_bench -- --ignored --nocapture
//! ```
//!
//! # Apples-to-apples disclaimer
//!
//! Same cuBLAS-sgemm-as-baseline asymmetry as `matmul_tc_bf16_bench`:
//! KAIO TC matmul (both f16 async and bf16 async) consumes 16-bit
//! inputs with fp32 accumulation, cuBLAS sgemm consumes f32 inputs.
//! The cuBLAS column is a project-local reference, not an apples-to-
//! apples claim — see `tech_debt.md` for the tracked future
//! `cublasGemmEx`-bf16 reference (Sprint 9.1 C8).
//!
//! The bf16_async-vs-f16_async column **is** apples-to-apples: same
//! kernel structure, same tile, same warp layout, same memory
//! bandwidth, same cp.async pipeline, same accumulator type. Only the
//! mma operand dtype tag differs. The SC-2 gate is what makes that
//! comparison load-bearing.

use std::time::Instant;

use half::{bf16, f16};
use kaio::prelude::*;
use kaio_ops::{matmul_tc_async, matmul_tc_bf16_async};

// --- Deterministic random data (LCG, matches matmul_tc_bf16_bench.rs pattern) ---

fn deterministic_data_f32(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0
        })
        .collect()
}

fn deterministic_data_f16(len: usize, seed: u64) -> Vec<f16> {
    deterministic_data_f32(len, seed)
        .into_iter()
        .map(f16::from_f32)
        .collect()
}

fn deterministic_data_bf16(len: usize, seed: u64) -> Vec<bf16> {
    deterministic_data_f32(len, seed)
        .into_iter()
        .map(bf16::from_f32)
        .collect()
}

// --- Bench harnesses ---

/// One bench "run" for the f16 async path: 5 warm-ups + 20 timed
/// iterations of `matmul_tc_async`, returns the median elapsed seconds.
fn bench_run_f16_async(
    device: &KaioDevice,
    a: &GpuBuffer<f16>,
    b: &GpuBuffer<f16>,
    c: &mut GpuBuffer<f32>,
    m: u32,
    n: u32,
    k: u32,
) -> f64 {
    for _ in 0..5 {
        matmul_tc_async(device, a, b, c, m, n, k).unwrap();
    }
    device.stream().synchronize().unwrap();
    let mut times = Vec::with_capacity(20);
    for _ in 0..20 {
        device.stream().synchronize().unwrap();
        let start = Instant::now();
        matmul_tc_async(device, a, b, c, m, n, k).unwrap();
        device.stream().synchronize().unwrap();
        times.push(start.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[10]
}

/// One bench "run" for the bf16 async path; same methodology as
/// `bench_run_f16_async`.
fn bench_run_bf16_async(
    device: &KaioDevice,
    a: &GpuBuffer<bf16>,
    b: &GpuBuffer<bf16>,
    c: &mut GpuBuffer<f32>,
    m: u32,
    n: u32,
    k: u32,
) -> f64 {
    for _ in 0..5 {
        matmul_tc_bf16_async(device, a, b, c, m, n, k).unwrap();
    }
    device.stream().synchronize().unwrap();
    let mut times = Vec::with_capacity(20);
    for _ in 0..20 {
        device.stream().synchronize().unwrap();
        let start = Instant::now();
        matmul_tc_bf16_async(device, a, b, c, m, n, k).unwrap();
        device.stream().synchronize().unwrap();
        times.push(start.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[10]
}

fn bench_cublas_sgemm(
    device: &KaioDevice,
    a: &GpuBuffer<f32>,
    b: &GpuBuffer<f32>,
    c: &mut GpuBuffer<f32>,
    m: u32,
    n: u32,
    k: u32,
) -> f64 {
    use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
    let blas = CudaBlas::new(device.stream().clone()).unwrap();
    let cfg = GemmConfig {
        transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        alpha: 1.0f32,
        lda: n as i32,
        ldb: k as i32,
        beta: 0.0f32,
        ldc: n as i32,
    };
    for _ in 0..5 {
        unsafe {
            blas.gemm(cfg, b.inner(), a.inner(), c.inner_mut()).unwrap();
        }
    }
    device.stream().synchronize().unwrap();
    let mut times = Vec::with_capacity(20);
    for _ in 0..20 {
        device.stream().synchronize().unwrap();
        let start = Instant::now();
        unsafe {
            blas.gemm(cfg, b.inner(), a.inner(), c.inner_mut()).unwrap();
        }
        device.stream().synchronize().unwrap();
        times.push(start.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[10]
}

fn tflops(m: usize, n: usize, k: usize, seconds: f64) -> f64 {
    2.0 * m as f64 * n as f64 * k as f64 / seconds / 1e12
}

// --- Benchmark test ---

// Sprint 9.1.1 SC-2 split bounds — same methodology as 9.1's SC-2.
// See `internal_9_1_1_async_bf16_plan.md` § D3 for the bf16_async vs
// f16_async precision-isolation framing.
//   - Median ratio bound: tight, structural-kernel-quality gate.
//     ±3% measures whether the bf16 async kernel has a real perf
//     delta from f16 async at the structural level.
//   - Worst ratio bound: generous, catastrophic-tail canary. ±15%
//     tolerates single OS-noise outliers while still catching
//     pathological tail behavior.
const SC2_MEDIAN_BOUND_PCT: f64 = 3.0;
const SC2_WORST_BOUND_PCT: f64 = 15.0;
const SC2_WORST_OF: usize = 10;

#[test]
#[ignore] // requires NVIDIA GPU (SM 8.0+) + CUDA toolkit (cuBLAS)
fn benchmark_matmul_tc_bf16_async() {
    let device = KaioDevice::new(0).expect("GPU required");
    let info = device.info().expect("device info");
    assert!(
        info.compute_capability.0 >= 8,
        "matmul_tc_bf16_async benchmark requires SM 8.0+ (Ampere). Have sm_{}{}",
        info.compute_capability.0,
        info.compute_capability.1
    );

    eprintln!();
    eprintln!("=== KAIO bf16 async vs f16 async Tensor-Core Matmul Benchmark (Sprint 9.1.1) ===");
    eprintln!("GPU: {:?}", info.name);
    eprintln!(
        "Compute capability: sm_{}{}",
        info.compute_capability.0, info.compute_capability.1
    );
    eprintln!("Per-run methodology: 5 warm-ups + 20 timed iterations (median)");
    eprintln!(
        "SC-2 gate (4096³ only): {SC2_WORST_OF} interleaved alternating-order runs; \
         median ratio ±{SC2_MEDIAN_BOUND_PCT:.0}% (structural) AND worst ratio ±{SC2_WORST_BOUND_PCT:.0}% (catastrophic-tail)"
    );
    eprintln!();
    eprintln!("APPLES-TO-APPLES NOTE:");
    eprintln!("  cuBLAS column is sgemm (f32 inputs) — project-local reference baseline,");
    eprintln!(
        "  not an apples-to-apples claim. bf16_async-vs-f16_async column IS apples-to-apples"
    );
    eprintln!(
        "  (same tile, warp layout, bandwidth, cp.async pipeline, accumulator — only mma operand dtype differs)."
    );
    eprintln!();

    let sizes: Vec<(usize, usize, usize)> = vec![
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ];

    eprintln!(
        "{:<14} {:>13} {:>9} {:>13} {:>9} {:>10} {:>9} {:>14} {:>10}",
        "Size",
        "f16_async ms",
        "f16_a TF",
        "bf16_async ms",
        "bf16_a TF",
        "cuBLAS ms",
        "cuBLAS TF",
        "bf16_a/f16_a",
        "bf16_a/cuB"
    );
    eprintln!("{}", "-".repeat(120));

    for &(m, n, k) in &sizes {
        let a_f16 = deterministic_data_f16(m * k, 42);
        let b_f16 = deterministic_data_f16(k * n, 137);
        let a_bf16 = deterministic_data_bf16(m * k, 42);
        let b_bf16 = deterministic_data_bf16(k * n, 137);
        let a_f32 = deterministic_data_f32(m * k, 42);
        let b_f32 = deterministic_data_f32(k * n, 137);

        let a_f16_dev = device.alloc_from(&a_f16).unwrap();
        let b_f16_dev = device.alloc_from(&b_f16).unwrap();
        let mut c_f16 = device.alloc_zeros::<f32>(m * n).unwrap();

        let a_bf16_dev = device.alloc_from(&a_bf16).unwrap();
        let b_bf16_dev = device.alloc_from(&b_bf16).unwrap();
        let mut c_bf16 = device.alloc_zeros::<f32>(m * n).unwrap();

        let a_f32_dev = device.alloc_from(&a_f32).unwrap();
        let b_f32_dev = device.alloc_from(&b_f32).unwrap();
        let mut c_f32 = device.alloc_zeros::<f32>(m * n).unwrap();

        let f16_s = bench_run_f16_async(
            &device, &a_f16_dev, &b_f16_dev, &mut c_f16, m as u32, n as u32, k as u32,
        );
        let bf16_s = bench_run_bf16_async(
            &device,
            &a_bf16_dev,
            &b_bf16_dev,
            &mut c_bf16,
            m as u32,
            n as u32,
            k as u32,
        );
        let cublas_s = bench_cublas_sgemm(
            &device, &a_f32_dev, &b_f32_dev, &mut c_f32, m as u32, n as u32, k as u32,
        );

        let f16_tf = tflops(m, n, k, f16_s);
        let bf16_tf = tflops(m, n, k, bf16_s);
        let cublas_tf = tflops(m, n, k, cublas_s);
        let bf16_vs_f16 = bf16_tf / f16_tf * 100.0;
        let bf16_vs_cu = bf16_tf / cublas_tf * 100.0;

        let label = format!("{m}x{n}x{k}");
        eprintln!(
            "{:<14} {:>11.2}ms {:>8.2} {:>11.2}ms {:>8.2} {:>8.2}ms {:>8.2} {:>12.1}% {:>9.1}%",
            label,
            f16_s * 1000.0,
            f16_tf,
            bf16_s * 1000.0,
            bf16_tf,
            cublas_s * 1000.0,
            cublas_tf,
            bf16_vs_f16,
            bf16_vs_cu,
        );
    }

    // ------------------------------------------------------------------------
    // SC-2 perf-parity gate at 4096³ — interleaved alternating-order runs,
    // bf16_async / f16_async per-iter ratio with split-bound thresholds.
    //
    // Methodology cloned from Sprint 9.1's SC-2 (matmul_tc_bf16_bench.rs) with
    // the staging variable changed from "sync vs sync" to "async vs async":
    //   1. Run 10 OUTER iters. Each iter does one f16_async bench-run AND
    //      one bf16_async bench-run, separated by milliseconds (same
    //      thermal sample).
    //   2. Alternate WHICH KERNEL GOES FIRST per outer iter. Cancels the
    //      intra-iter "second kernel sees hotter GPU" bias.
    //   3. Compute the bf16_async/f16_async TFLOPS ratio per iter.
    //   4. Apply the split-bound gate (median ±3% + worst ±15%).
    // ------------------------------------------------------------------------
    eprintln!();
    eprintln!(
        "Running SC-2 gate at 4096³ ({SC2_WORST_OF} interleaved iters, alternating order)..."
    );
    let (m, n, k) = (4096usize, 4096usize, 4096usize);

    let a_f16 = deterministic_data_f16(m * k, 42);
    let b_f16 = deterministic_data_f16(k * n, 137);
    let a_bf16 = deterministic_data_bf16(m * k, 42);
    let b_bf16 = deterministic_data_bf16(k * n, 137);

    let a_f16_dev = device.alloc_from(&a_f16).unwrap();
    let b_f16_dev = device.alloc_from(&b_f16).unwrap();
    let mut c_f16 = device.alloc_zeros::<f32>(m * n).unwrap();
    let a_bf16_dev = device.alloc_from(&a_bf16).unwrap();
    let b_bf16_dev = device.alloc_from(&b_bf16).unwrap();
    let mut c_bf16 = device.alloc_zeros::<f32>(m * n).unwrap();

    let mut iter_samples: Vec<(f64, f64, bool)> = Vec::with_capacity(SC2_WORST_OF);
    for i in 0..SC2_WORST_OF {
        let f16_first = i.is_multiple_of(2);
        let (f16_t, bf16_t) = if f16_first {
            let f = bench_run_f16_async(
                &device, &a_f16_dev, &b_f16_dev, &mut c_f16, m as u32, n as u32, k as u32,
            );
            let b = bench_run_bf16_async(
                &device,
                &a_bf16_dev,
                &b_bf16_dev,
                &mut c_bf16,
                m as u32,
                n as u32,
                k as u32,
            );
            (f, b)
        } else {
            let b = bench_run_bf16_async(
                &device,
                &a_bf16_dev,
                &b_bf16_dev,
                &mut c_bf16,
                m as u32,
                n as u32,
                k as u32,
            );
            let f = bench_run_f16_async(
                &device, &a_f16_dev, &b_f16_dev, &mut c_f16, m as u32, n as u32, k as u32,
            );
            (f, b)
        };
        iter_samples.push((f16_t, bf16_t, f16_first));
    }

    let ratios: Vec<f64> = iter_samples
        .iter()
        .map(|&(f16_t, bf16_t, _)| {
            let f16_tf = tflops(m, n, k, f16_t);
            let bf16_tf = tflops(m, n, k, bf16_t);
            bf16_tf / f16_tf * 100.0
        })
        .collect();

    eprintln!();
    eprintln!("Per-iter SC-2 samples at 4096³:");
    eprintln!(
        "  {:>4} {:>17} {:>13} {:>13} {:>11} {:>11} {:>14}",
        "iter", "order", "f16_async ms", "bf16_async ms", "f16_a TF", "bf16_a TF", "bf16_a/f16_a",
    );
    for (i, (&(f16_t, bf16_t, f16_first), &ratio_pct)) in
        iter_samples.iter().zip(ratios.iter()).enumerate()
    {
        let order = if f16_first {
            "f16_a→bf16_a"
        } else {
            "bf16_a→f16_a"
        };
        let f16_tf = tflops(m, n, k, f16_t);
        let bf16_tf = tflops(m, n, k, bf16_t);
        eprintln!(
            "  {:>4} {:>17} {:>11.2}ms {:>11.2}ms {:>10.2} {:>10.2} {:>13.2}%",
            i,
            order,
            f16_t * 1000.0,
            bf16_t * 1000.0,
            f16_tf,
            bf16_tf,
            ratio_pct,
        );
    }

    let worst_ratio_pct = ratios.iter().cloned().fold(100.0f64, |acc, r| {
        if (r - 100.0).abs() > (acc - 100.0).abs() {
            r
        } else {
            acc
        }
    });
    let median_ratio_pct = {
        let mut sorted = ratios.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[SC2_WORST_OF / 2]
    };
    let median_delta_pct = median_ratio_pct - 100.0;
    let worst_delta_pct = worst_ratio_pct - 100.0;

    // Same-run f16_async TFLOPS median, useful as a forward-looking
    // local-reference data point for future runs to compare against.
    let f16_async_median_tflops = {
        let mut f16_tfs: Vec<f64> = iter_samples
            .iter()
            .map(|&(f16_t, _, _)| tflops(m, n, k, f16_t))
            .collect();
        f16_tfs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        f16_tfs[SC2_WORST_OF / 2]
    };

    eprintln!();
    eprintln!("SC-2 results at 4096³ (per-iter ratios across {SC2_WORST_OF} interleaved runs):");
    eprintln!(
        "  median ratio:  {median_ratio_pct:.2}% (delta {median_delta_pct:+.2}%) \
         — structural-kernel gate, bound ±{SC2_MEDIAN_BOUND_PCT:.0}%"
    );
    eprintln!(
        "  worst  ratio:  {worst_ratio_pct:.2}% (delta {worst_delta_pct:+.2}%) \
         — catastrophic-tail gate,  bound ±{SC2_WORST_BOUND_PCT:.0}%"
    );
    eprintln!();
    eprintln!(
        "Same-run f16_async median at 4096³: {f16_async_median_tflops:.2} TFLOPS \
         (record for future drift comparison — informational only, not a gate)"
    );
    eprintln!();

    if cfg!(debug_assertions) {
        eprintln!(
            "DEBUG BUILD detected (debug_assertions=on): SC-2 hard assertions skipped.\n\
             The per-iter TFLOPS ratios above are not representative — debug-mode\n\
             launch overhead inflates per-iter variance asymmetrically between the\n\
             two kernels. For the canonical SC-2 verdict run:\n\
                 cargo xtask bench matmul_tc_bf16_async_bench\n\
             or equivalently:\n\
                 cargo test --release -p kaio-ops --test matmul_tc_bf16_async_bench -- --ignored --nocapture"
        );
        return;
    }

    assert!(
        median_delta_pct.abs() <= SC2_MEDIAN_BOUND_PCT,
        "Sprint 9.1.1 SC-2 STRUCTURAL-KERNEL GATE FAILED (median ratio):\n\
         \n\
         bf16_async at 4096³ is {median_ratio_pct:.2}% of f16_async at the median \
         per-iter ratio (delta {median_delta_pct:+.2}%);\n\
         the structural-kernel bound ±{SC2_MEDIAN_BOUND_PCT:.0}% was exceeded.\n\
         \n\
         The median is the noise-robust signal of kernel-level perf difference.\n\
         A miss on this bound means a real structural regression on the bf16 async path,\n\
         not OS noise. The bf16_async kernel structure is byte-identical to f16_async's\n\
         (same tile, warp layout, cp.async pipeline, accumulator) — only the mma\n\
         operand dtype tag differs.\n\
         \n\
         Investigate:\n\
         - PTX diff: KAIO_DUMP_PTX=1 cargo test --test matmul_tc_bf16_async_bench -- --ignored\n\
           and `diff` the K_LOOP body against the f16 async emit.\n\
         - Register pressure: ptxas --verbose to compare `.reg` counts.\n\
         - Helper visibility: confirm `emit_warp_quadrant_mma_bf16` and\n\
           `emit_mw_load_tile_a_64x16_async` are the C0-promoted versions, not\n\
           accidentally duplicated."
    );
    assert!(
        worst_delta_pct.abs() <= SC2_WORST_BOUND_PCT,
        "Sprint 9.1.1 SC-2 CATASTROPHIC-TAIL GATE FAILED (worst-of-{SC2_WORST_OF} ratio):\n\
         \n\
         bf16_async at 4096³ hit a worst per-iter ratio of {worst_ratio_pct:.2}% \
         (delta {worst_delta_pct:+.2}%);\n\
         the catastrophic-tail bound ±{SC2_WORST_BOUND_PCT:.0}% was exceeded.\n\
         \n\
         The worst bound is generous to OS noise — exceeding it indicates a\n\
         pathological tail rather than launch-overhead variance. Inspect the\n\
         per-iter table above: a single outlier iter is OS noise; sustained\n\
         skew is a kernel issue. If the median also failed, structural\n\
         regression; if only worst failed, kernel is fine but the run caught\n\
         a genuinely pathological tail event worth a re-run + investigate."
    );

    eprintln!(
        "SC-2 GATE PASSED: structural ±{SC2_MEDIAN_BOUND_PCT:.0}% (median) AND \
         catastrophic-tail ±{SC2_WORST_BOUND_PCT:.0}% (worst) bounds held at 4096³."
    );
}
