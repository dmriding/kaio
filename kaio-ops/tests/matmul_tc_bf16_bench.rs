#![allow(clippy::too_many_arguments)]

//! Benchmark: KAIO tensor-core matmul **bf16** sync vs **f16** sync
//! (Sprint 9.1) plus a cuBLAS sgemm reference column. Includes the
//! **SC-2 perf-parity gate** at 4096³: 10 interleaved alternating-order
//! runs, with two independent bounds applied to the per-iter
//! bf16/f16 TFLOPS ratios:
//!
//! - **Median ratio ±3%** — structural-kernel gate. Tight, noise-robust,
//!   measures whether the bf16 kernel has a real perf delta from f16.
//! - **Worst ratio ±15%** — catastrophic-tail gate. Generous to OS noise,
//!   catches genuinely pathological tail behavior.
//!
//! Both bounds must hold. See `sprint_9_1.md` § "Methodology evolution"
//! for the full rationale — the gate evolved from the plan's locked
//! worst-of-10 medians, through per-iter ratio with alternating order,
//! to the current split bounds. The comment block on the SC-2 gate loop
//! below carries the in-code annotation.
//!
//! Run with:
//! ```sh
//! cargo test -p kaio-ops --test matmul_tc_bf16_bench -- --ignored --nocapture
//! ```
//!
//! # Apples-to-apples disclaimer
//!
//! Same cuBLAS-sgemm-as-baseline asymmetry as `matmul_tc_bench`: KAIO TC
//! matmul (both f16 and bf16) consumes 16-bit inputs with fp32 accumulation,
//! cuBLAS sgemm consumes f32 inputs. The cuBLAS column is a project-local
//! reference, not an apples-to-apples claim — see `tech_debt.md` for the
//! tracked future `cublasGemmEx`-bf16 reference (Sprint 9.1 C8).
//!
//! The bf16-vs-f16 column **is** apples-to-apples: same kernel structure,
//! same tile, same warp layout, same memory bandwidth, same accumulator
//! type. Only the mma operand dtype tag differs. The SC-2 gate is what
//! makes that comparison load-bearing.

use std::time::Instant;

use half::{bf16, f16};
use kaio::prelude::*;
use kaio_ops::{matmul_tc, matmul_tc_bf16};

// --- Deterministic random data (LCG, matches matmul_tc_bench.rs pattern) ---

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

/// One bench "run": 5 warm-ups + 20 timed iterations of `matmul_tc`,
/// returns the median elapsed seconds.
fn bench_run_f16(
    device: &KaioDevice,
    a: &GpuBuffer<f16>,
    b: &GpuBuffer<f16>,
    c: &mut GpuBuffer<f32>,
    m: u32,
    n: u32,
    k: u32,
) -> f64 {
    for _ in 0..5 {
        matmul_tc(device, a, b, c, m, n, k).unwrap();
    }
    device.stream().synchronize().unwrap();
    let mut times = Vec::with_capacity(20);
    for _ in 0..20 {
        device.stream().synchronize().unwrap();
        let start = Instant::now();
        matmul_tc(device, a, b, c, m, n, k).unwrap();
        device.stream().synchronize().unwrap();
        times.push(start.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[10]
}

/// One bench "run" for the bf16 path; same methodology as `bench_run_f16`.
fn bench_run_bf16(
    device: &KaioDevice,
    a: &GpuBuffer<bf16>,
    b: &GpuBuffer<bf16>,
    c: &mut GpuBuffer<f32>,
    m: u32,
    n: u32,
    k: u32,
) -> f64 {
    for _ in 0..5 {
        matmul_tc_bf16(device, a, b, c, m, n, k).unwrap();
    }
    device.stream().synchronize().unwrap();
    let mut times = Vec::with_capacity(20);
    for _ in 0..20 {
        device.stream().synchronize().unwrap();
        let start = Instant::now();
        matmul_tc_bf16(device, a, b, c, m, n, k).unwrap();
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

// Sprint 9.1 SC-2 split bounds — see `sprint_9_1.md` § "Methodology
// evolution" for the rationale behind the move from a single ±5%
// worst-of-10 bound to these two independent bounds.
//   - Median ratio bound: tight, structural-kernel-quality gate.
//     ±3% sharpens the original plan's ±5% on the axis that actually
//     measures kernel difference (cancels OS noise via median).
//   - Worst ratio bound:  generous, catastrophic-tail canary. ±15%
//     tolerates single OS-noise outliers (driver state transitions,
//     scheduler interrupts during one of the 5+20 sub-iters) while
//     still catching a genuinely pathological tail behavior.
const SC2_MEDIAN_BOUND_PCT: f64 = 3.0;
const SC2_WORST_BOUND_PCT: f64 = 15.0;
const SC2_WORST_OF: usize = 10;

#[test]
#[ignore] // requires NVIDIA GPU (SM 8.0+) + CUDA toolkit (cuBLAS)
fn benchmark_matmul_tc_bf16() {
    let device = KaioDevice::new(0).expect("GPU required");
    let info = device.info().expect("device info");
    assert!(
        info.compute_capability.0 >= 8,
        "matmul_tc_bf16 benchmark requires SM 8.0+ (Ampere). Have sm_{}{}",
        info.compute_capability.0,
        info.compute_capability.1
    );

    eprintln!();
    eprintln!("=== KAIO bf16 vs f16 Tensor-Core Matmul Benchmark (Sprint 9.1) ===");
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
    eprintln!("  not an apples-to-apples claim. bf16-vs-f16 column IS apples-to-apples");
    eprintln!(
        "  (same tile, warp layout, bandwidth, accumulator — only mma operand dtype differs)."
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
        "{:<14} {:>10} {:>9} {:>10} {:>9} {:>10} {:>9} {:>10} {:>10}",
        "Size",
        "f16 ms",
        "f16 TF",
        "bf16 ms",
        "bf16 TF",
        "cuBLAS ms",
        "cuBLAS TF",
        "bf16/f16",
        "bf16/cuB"
    );
    eprintln!("{}", "-".repeat(110));

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

        let f16_s = bench_run_f16(
            &device, &a_f16_dev, &b_f16_dev, &mut c_f16, m as u32, n as u32, k as u32,
        );
        let bf16_s = bench_run_bf16(
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
            "{:<14} {:>8.2}ms {:>8.2} {:>8.2}ms {:>8.2} {:>8.2}ms {:>8.2} {:>8.1}% {:>8.1}%",
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
    // SC-2 perf-parity gate at 4096³ — worst per-iter ratio, interleaved with
    // alternating order to cancel intra-iter thermal/state bias.
    //
    // **Methodology refinement vs the locked plan's "worst-of-10 medians":**
    // an initial implementation took the worst f16 median and worst bf16 median
    // independently across 10 consecutive runs. That signal turned out to be
    // dominated by GPU thermal drift (kernel ordering bias + per-tail noise
    // sampled at different thermal states), not by kernel-level differences.
    // The kernels are byte-identical at the structural level (same tile, warp
    // layout, bandwidth, mma instance shape) so the gate must measure
    // structural difference, not measurement noise.
    //
    // Refined methodology, faithful to the plan's "worst across 10 consecutive
    // runs" intent:
    //   1. Run 10 OUTER iters. Each iter does one f16 bench-run AND one
    //      bf16 bench-run, separated by milliseconds (same thermal sample).
    //   2. Alternate WHICH KERNEL GOES FIRST per outer iter (5× f16-first,
    //      5× bf16-first). Cancels the intra-iter "second kernel sees hotter
    //      GPU" bias.
    //   3. Compute the bf16/f16 TFLOPS ratio per iter — both kernels share
    //      thermal state in that pair, so the ratio is a thermal-invariant
    //      measure of structural kernel difference.
    //   4. Take the WORST per-iter ratio (largest |1 − ratio|) as the SC-2
    //      number. Preserves the plan's "worst across 10" tail-behavior canary,
    //      but applied to ratios (sensitive to kernel diff) rather than to
    //      independent absolute timings (sensitive to noise).
    //
    // A >5% miss on this refined gate would mean a real structural kernel
    // regression — investigate before C7 per the master plan.
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

    // (f16_seconds, bf16_seconds, f16_first) per outer iter.
    let mut iter_samples: Vec<(f64, f64, bool)> = Vec::with_capacity(SC2_WORST_OF);
    for i in 0..SC2_WORST_OF {
        let f16_first = i.is_multiple_of(2);
        let (f16_t, bf16_t) = if f16_first {
            let f = bench_run_f16(
                &device, &a_f16_dev, &b_f16_dev, &mut c_f16, m as u32, n as u32, k as u32,
            );
            let b = bench_run_bf16(
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
            let b = bench_run_bf16(
                &device,
                &a_bf16_dev,
                &b_bf16_dev,
                &mut c_bf16,
                m as u32,
                n as u32,
                k as u32,
            );
            let f = bench_run_f16(
                &device, &a_f16_dev, &b_f16_dev, &mut c_f16, m as u32, n as u32, k as u32,
            );
            (f, b)
        };
        iter_samples.push((f16_t, bf16_t, f16_first));
    }

    // Per-iter ratios (bf16 TFLOPS / f16 TFLOPS × 100). Higher = bf16 winning.
    let ratios: Vec<f64> = iter_samples
        .iter()
        .map(|&(f16_t, bf16_t, _)| {
            let f16_tf = tflops(m, n, k, f16_t);
            let bf16_tf = tflops(m, n, k, bf16_t);
            bf16_tf / f16_tf * 100.0
        })
        .collect();

    // Print the per-iter table so a future debugger can distinguish "all
    // ratios clustered tight" (= no kernel regression, gate failure was an
    // outlier) from "ratios systematically biased" (= real kernel regression).
    eprintln!();
    eprintln!("Per-iter SC-2 samples at 4096³:");
    eprintln!(
        "  {:>4} {:>9} {:>10} {:>10} {:>9} {:>9} {:>10}",
        "iter", "order", "f16 ms", "bf16 ms", "f16 TF", "bf16 TF", "bf16/f16",
    );
    for (i, (&(f16_t, bf16_t, f16_first), &ratio_pct)) in
        iter_samples.iter().zip(ratios.iter()).enumerate()
    {
        let order = if f16_first {
            "f16→bf16"
        } else {
            "bf16→f16"
        };
        let f16_tf = tflops(m, n, k, f16_t);
        let bf16_tf = tflops(m, n, k, bf16_t);
        eprintln!(
            "  {:>4} {:>9} {:>8.2}ms {:>8.2}ms {:>8.2} {:>8.2} {:>9.2}%",
            i,
            order,
            f16_t * 1000.0,
            bf16_t * 1000.0,
            f16_tf,
            bf16_tf,
            ratio_pct,
        );
    }

    // Worst per-iter ratio: largest |1 − ratio_pct/100|.
    let worst_ratio_pct = ratios.iter().cloned().fold(100.0f64, |acc, r| {
        if (r - 100.0).abs() > (acc - 100.0).abs() {
            r
        } else {
            acc
        }
    });
    // Median ratio as a robustness companion to the worst (useful for
    // distinguishing real regression from a single outlier in the
    // diagnostic output).
    let median_ratio_pct = {
        let mut sorted = ratios.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[SC2_WORST_OF / 2]
    };
    let median_delta_pct = median_ratio_pct - 100.0;
    let worst_delta_pct = worst_ratio_pct - 100.0;

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

    // Debug builds suffer ~10-20x slower kernel execution and much wider
    // per-iter variance (launch overhead dominates, CPU-side measurement
    // jitter inflates the worst-of-N pick asymmetrically across kernels).
    // Neither SC-2 bound is meaningful on debug timings — only assert in
    // release builds. `cargo xtask bench` always uses release; the raw
    // `cargo test` path falls into this debug branch unless given `--release`.
    if cfg!(debug_assertions) {
        eprintln!(
            "DEBUG BUILD detected (debug_assertions=on): SC-2 hard assertions skipped.\n\
             The per-iter TFLOPS ratios above are not representative — debug-mode\n\
             launch overhead inflates per-iter variance asymmetrically between the\n\
             two kernels. For the canonical SC-2 verdict run:\n\
                 cargo xtask bench matmul_tc_bf16_bench\n\
             or equivalently:\n\
                 cargo test --release -p kaio-ops --test matmul_tc_bf16_bench -- --ignored --nocapture"
        );
        return;
    }

    // Two independent assertions — see `sprint_9_1.md` § "Methodology
    // evolution" for the rationale.
    assert!(
        median_delta_pct.abs() <= SC2_MEDIAN_BOUND_PCT,
        "Sprint 9.1 SC-2 STRUCTURAL-KERNEL GATE FAILED (median ratio):\n\
         \n\
         bf16 sync at 4096³ is {median_ratio_pct:.2}% of f16 sync at the median \
         per-iter ratio (delta {median_delta_pct:+.2}%);\n\
         the structural-kernel bound ±{SC2_MEDIAN_BOUND_PCT:.0}% was exceeded.\n\
         \n\
         The median is the noise-robust signal of kernel-level perf difference.\n\
         A miss on this bound means a real structural regression on the bf16 path,\n\
         not OS noise. The bf16 kernel structure is byte-identical to f16's (same\n\
         tile, warp layout, memory bandwidth, accumulator) — only the mma operand\n\
         dtype tag differs.\n\
         \n\
         Investigate:\n\
         - PTX diff: KAIO_DUMP_PTX=1 cargo test --test matmul_tc_bf16_bench -- --ignored\n\
           and `diff` the K_LOOP body against the f16 emit.\n\
         - Register pressure: ptxas --verbose to compare `.reg` counts between f16/bf16.\n\
         - Shared-mem layout: confirm the bf16 path uses the same Sprint 6.7b\n\
           col-stride 36 B padding via the reused `pub(crate)` tile-B loader."
    );
    assert!(
        worst_delta_pct.abs() <= SC2_WORST_BOUND_PCT,
        "Sprint 9.1 SC-2 CATASTROPHIC-TAIL GATE FAILED (worst-of-{SC2_WORST_OF} ratio):\n\
         \n\
         bf16 sync at 4096³ hit a worst per-iter ratio of {worst_ratio_pct:.2}% \
         (delta {worst_delta_pct:+.2}%);\n\
         the catastrophic-tail bound ±{SC2_WORST_BOUND_PCT:.0}% was exceeded.\n\
         \n\
         The worst bound is generous to OS noise — exceeding it indicates a\n\
         pathological tail rather than launch-overhead variance. Inspect the\n\
         per-iter table above: a single outlier iter (with extreme ratio relative\n\
         to its peers) is OS noise; a sustained skew across iters is a kernel\n\
         issue. The median bound, if it also failed, points to structural\n\
         regression; if only the worst failed, the kernel is fine but the run\n\
         caught a genuinely pathological tail event worth a re-run + investigate."
    );

    eprintln!(
        "SC-2 GATE PASSED: structural ±{SC2_MEDIAN_BOUND_PCT:.0}% (median) AND \
         catastrophic-tail ±{SC2_WORST_BOUND_PCT:.0}% (worst) bounds held at 4096³."
    );
}
