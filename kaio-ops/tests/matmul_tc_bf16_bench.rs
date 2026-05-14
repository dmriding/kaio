#![allow(clippy::too_many_arguments)]

//! Benchmark: KAIO tensor-core matmul **bf16** sync vs **f16** sync
//! (Sprint 9.1) plus a cuBLAS sgemm reference column. Includes the
//! **SC-2 perf-parity gate** at 4096³: worst-of-10 bf16 within ±5% of
//! worst-of-10 f16 in the same `cargo xtask bench` run. A miss is a
//! 9.1 ship blocker — kernel structure is identical to f16's so a >5%
//! gap means something is wrong on the bf16 path (investigate via PTX
//! diff vs f16, register pressure, shared-mem layout) and either
//! resolve or escalate to a separate debug sprint before C7.
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

const SC2_BOUND_PCT: f64 = 5.0;
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
        "SC-2 gate (4096³ only): worst across {SC2_WORST_OF} consecutive runs, ±{SC2_BOUND_PCT:.0}% bf16 vs f16"
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
    // SC-2 perf-parity gate at 4096³ — worst-of-10 bf16 vs worst-of-10 f16.
    //
    // Run 10 consecutive (warm + iter) "runs" per kernel, track the WORST
    // median across them, then compare bf16 vs f16. A >5% gap is a 9.1
    // ship blocker — investigate before C7 per the master plan.
    // ------------------------------------------------------------------------
    eprintln!();
    eprintln!(
        "Running SC-2 gate at 4096³ ({SC2_WORST_OF} runs each, worst-of-{SC2_WORST_OF} median)..."
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

    let mut f16_runs = Vec::with_capacity(SC2_WORST_OF);
    let mut bf16_runs = Vec::with_capacity(SC2_WORST_OF);
    for _ in 0..SC2_WORST_OF {
        f16_runs.push(bench_run_f16(
            &device, &a_f16_dev, &b_f16_dev, &mut c_f16, m as u32, n as u32, k as u32,
        ));
        bf16_runs.push(bench_run_bf16(
            &device,
            &a_bf16_dev,
            &b_bf16_dev,
            &mut c_bf16,
            m as u32,
            n as u32,
            k as u32,
        ));
    }
    let f16_worst = f16_runs.iter().cloned().fold(0.0f64, f64::max);
    let bf16_worst = bf16_runs.iter().cloned().fold(0.0f64, f64::max);
    let f16_worst_tf = tflops(m, n, k, f16_worst);
    let bf16_worst_tf = tflops(m, n, k, bf16_worst);
    // Compare TFLOPS (higher = better): bf16 / f16 ratio in percent.
    // ±5% means bf16_tf ≥ 0.95 × f16_tf (slower-than-f16 is the concerning direction).
    let bf16_vs_f16_pct = bf16_worst_tf / f16_worst_tf * 100.0;
    let delta_pct = bf16_vs_f16_pct - 100.0;

    eprintln!();
    eprintln!("SC-2 results at 4096³:");
    eprintln!(
        "  f16  worst-of-{SC2_WORST_OF}: {:.2} ms / {:.2} TF",
        f16_worst * 1000.0,
        f16_worst_tf
    );
    eprintln!(
        "  bf16 worst-of-{SC2_WORST_OF}: {:.2} ms / {:.2} TF",
        bf16_worst * 1000.0,
        bf16_worst_tf
    );
    eprintln!("  bf16 / f16 (TFLOPS): {bf16_vs_f16_pct:.2}% (delta: {delta_pct:+.2}%)");
    eprintln!();

    // Debug builds suffer ~10-20x slower kernel execution and much wider
    // per-iter variance (launch overhead dominates, CPU-side measurement
    // jitter inflates the worst-of-N pick asymmetrically across kernels).
    // The SC-2 gate compares kernel throughput, not host-side noise — only
    // assert in release builds. `cargo xtask bench` always uses release;
    // the raw `cargo test` path falls into this debug branch unless given
    // `--release`.
    if cfg!(debug_assertions) {
        eprintln!(
            "DEBUG BUILD detected (debug_assertions=on): SC-2 hard assertion skipped.\n\
             The worst-of-{SC2_WORST_OF} TFLOPS comparison above is not representative —\n\
             debug-mode launch overhead inflates per-iter variance asymmetrically\n\
             between the two kernels. For the canonical SC-2 verdict run:\n\
                 cargo xtask bench matmul_tc_bf16_bench\n\
             or equivalently:\n\
                 cargo test --release -p kaio-ops --test matmul_tc_bf16_bench -- --ignored --nocapture"
        );
        return;
    }

    assert!(
        delta_pct.abs() <= SC2_BOUND_PCT,
        "Sprint 9.1 SC-2 PERF-PARITY GATE FAILED:\n\
         \n\
         bf16 sync at 4096³ is {bf16_vs_f16_pct:.2}% of f16 sync (delta {delta_pct:+.2}%);\n\
         the ±{SC2_BOUND_PCT:.0}% bound was exceeded.\n\
         \n\
         Per the master plan's R-9.1-PERF-PARITY risk, a >5% gap is a ship blocker.\n\
         The bf16 kernel structure is byte-identical to f16's (same tile, warp layout,\n\
         memory bandwidth, accumulator) — only the mma operand dtype tag differs.\n\
         A miss means something is wrong on the bf16 path.\n\
         \n\
         Investigate before C7:\n\
         - PTX diff: KAIO_DUMP_PTX=1 cargo test --test matmul_tc_bf16_bench -- --ignored\n\
           and `diff` the K_LOOP body against the f16 emit.\n\
         - Register pressure: ptxas --verbose to compare `.reg` counts.\n\
         - Shared-mem layout: confirm the bf16 path uses the same Sprint 6.7b\n\
           col-stride 36 B padding via the reused `pub(crate)` tile-B loader."
    );

    eprintln!("SC-2 GATE PASSED: bf16 sync within ±{SC2_BOUND_PCT:.0}% of f16 sync at 4096³.");
}
