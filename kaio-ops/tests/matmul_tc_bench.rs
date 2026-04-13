#![allow(clippy::too_many_arguments)]

//! Benchmark: KAIO tensor-core matmul (sync + async cp.async) vs
//! cuBLAS sgemm. Sprint 6.7.
//!
//! Run with:
//! ```sh
//! cargo test -p kaio-ops --test matmul_tc_bench -- --ignored --nocapture
//! ```
//!
//! # Apples-to-apples disclaimer
//!
//! KAIO TC matmul uses **fp16 inputs with fp32 accumulation**. Comparison
//! is against **cuBLAS sgemm** (f32 inputs, f32 output) because that is
//! the existing supported benchmark path in this repo (cudarc 0.19's
//! `Gemm::gemm` exposes sgemm cleanly; gemmEx with fp16 inputs would
//! require dropping into raw FFI, which is orthogonal to the multi-warp
//! restructure that defines Sprint 6.7).
//!
//! Results should be read as a **project-local performance baseline,
//! not a claim of apples-to-apples precision identity.** The fp16-input
//! / fp32-input asymmetry halves global memory bandwidth on the TC
//! side and unlocks tensor-core throughput; that gap is part of the
//! value proposition, not a flaw in the comparison.
//!
//! For a true f16-vs-f16 comparison against cuBLAS HGEMM/GemmEx, see
//! tracked tech debt — Sprint 6.7+ scope decision.

use std::time::Instant;

use half::f16;
use kaio::prelude::*;
use kaio_ops::{matmul_tc, matmul_tc_async};

// --- Deterministic random data (LCG, matches matmul_bench.rs pattern) ---

fn deterministic_data_f32(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Map to [-1.0, 1.0] — keeps fp16 representable with no overflow.
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

// --- Bench harness ---

fn bench_matmul_tc_sync(
    device: &KaioDevice,
    a: &GpuBuffer<f16>,
    b: &GpuBuffer<f16>,
    c: &mut GpuBuffer<f32>,
    m: u32,
    n: u32,
    k: u32,
    warmup: usize,
    iters: usize,
) -> f64 {
    for _ in 0..warmup {
        matmul_tc(device, a, b, c, m, n, k).unwrap();
    }
    device.stream().synchronize().unwrap();

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        device.stream().synchronize().unwrap();
        let start = Instant::now();
        matmul_tc(device, a, b, c, m, n, k).unwrap();
        device.stream().synchronize().unwrap();
        times.push(start.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[iters / 2]
}

fn bench_matmul_tc_async(
    device: &KaioDevice,
    a: &GpuBuffer<f16>,
    b: &GpuBuffer<f16>,
    c: &mut GpuBuffer<f32>,
    m: u32,
    n: u32,
    k: u32,
    warmup: usize,
    iters: usize,
) -> f64 {
    for _ in 0..warmup {
        matmul_tc_async(device, a, b, c, m, n, k).unwrap();
    }
    device.stream().synchronize().unwrap();

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        device.stream().synchronize().unwrap();
        let start = Instant::now();
        matmul_tc_async(device, a, b, c, m, n, k).unwrap();
        device.stream().synchronize().unwrap();
        times.push(start.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[iters / 2]
}

fn bench_cublas_sgemm(
    device: &KaioDevice,
    a: &GpuBuffer<f32>,
    b: &GpuBuffer<f32>,
    c: &mut GpuBuffer<f32>,
    m: u32,
    n: u32,
    k: u32,
    warmup: usize,
    iters: usize,
) -> f64 {
    use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};

    let blas = CudaBlas::new(device.stream().clone()).unwrap();

    // cuBLAS row-major trick: C = A×B (row-major) ↔ C^T = B^T × A^T (col-major).
    // GEMM params: m=N, n=M, k=K, lda=N, ldb=K, ldc=N. Same as matmul_bench.rs.
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

    for _ in 0..warmup {
        unsafe {
            blas.gemm(cfg, b.inner(), a.inner(), c.inner_mut()).unwrap();
        }
    }
    device.stream().synchronize().unwrap();

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        device.stream().synchronize().unwrap();
        let start = Instant::now();
        unsafe {
            blas.gemm(cfg, b.inner(), a.inner(), c.inner_mut()).unwrap();
        }
        device.stream().synchronize().unwrap();
        times.push(start.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[iters / 2]
}

fn tflops(m: usize, n: usize, k: usize, seconds: f64) -> f64 {
    2.0 * m as f64 * n as f64 * k as f64 / seconds / 1e12
}

// --- Benchmark test ---

#[test]
#[ignore] // requires NVIDIA GPU + CUDA toolkit + sm_80+
fn benchmark_matmul_tc() {
    let device = KaioDevice::new(0).expect("GPU required");
    let info = device.info().expect("device info");
    assert!(
        info.compute_capability.0 >= 8,
        "matmul_tc benchmark requires SM 8.0+ (Ampere). Have sm_{}{}",
        info.compute_capability.0,
        info.compute_capability.1
    );

    eprintln!();
    eprintln!("=== KAIO Tensor-Core Matmul Benchmark (Sprint 6.7) ===");
    eprintln!("GPU: {:?}", info.name);
    eprintln!(
        "Compute capability: sm_{}{}",
        info.compute_capability.0, info.compute_capability.1
    );
    eprintln!("Warm-up: 5 | Iterations: 20 | Metric: median TFLOPS");
    eprintln!();
    eprintln!("APPLES-TO-APPLES DISCLAIMER:");
    eprintln!("  KAIO TC matmul uses fp16 inputs with fp32 accumulation.");
    eprintln!("  Comparison is against cuBLAS sgemm because that is the existing");
    eprintln!("  supported benchmark path in this repo. Results should be read as");
    eprintln!("  a project-local performance baseline, not a claim of apples-to-");
    eprintln!("  apples precision identity.");
    eprintln!();

    let sizes: Vec<(usize, usize, usize)> = vec![
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ];

    eprintln!(
        "{:<14} {:>10} {:>9} {:>10} {:>9} {:>10} {:>9} {:>9} {:>9}",
        "Size",
        "TC sync ms",
        "TC sync TF",
        "TC async ms",
        "TC async TF",
        "cuBLAS ms",
        "cuBLAS TF",
        "sync vs cuB",
        "async vs cuB"
    );
    eprintln!("{}", "-".repeat(108));

    let mut sync_ms_at_4096 = 0.0;
    let mut async_ms_at_4096 = 0.0;
    let mut cublas_ms_at_4096 = 0.0;

    for &(m, n, k) in &sizes {
        // f16 inputs for TC kernels.
        let a_f16 = deterministic_data_f16(m * k, 42);
        let b_f16 = deterministic_data_f16(k * n, 137);
        // f32 inputs for cuBLAS sgemm (same numeric values, different precision).
        let a_f32 = deterministic_data_f32(m * k, 42);
        let b_f32 = deterministic_data_f32(k * n, 137);

        let a_tc = device.alloc_from(&a_f16).unwrap();
        let b_tc = device.alloc_from(&b_f16).unwrap();
        let mut c_sync = device.alloc_zeros::<f32>(m * n).unwrap();
        let mut c_async = device.alloc_zeros::<f32>(m * n).unwrap();

        let a_cu = device.alloc_from(&a_f32).unwrap();
        let b_cu = device.alloc_from(&b_f32).unwrap();
        let mut c_cublas = device.alloc_zeros::<f32>(m * n).unwrap();

        let sync_s = bench_matmul_tc_sync(
            &device,
            &a_tc,
            &b_tc,
            &mut c_sync,
            m as u32,
            n as u32,
            k as u32,
            5,
            20,
        );
        let async_s = bench_matmul_tc_async(
            &device,
            &a_tc,
            &b_tc,
            &mut c_async,
            m as u32,
            n as u32,
            k as u32,
            5,
            20,
        );
        let cublas_s = bench_cublas_sgemm(
            &device,
            &a_cu,
            &b_cu,
            &mut c_cublas,
            m as u32,
            n as u32,
            k as u32,
            5,
            20,
        );

        let sync_tf = tflops(m, n, k, sync_s);
        let async_tf = tflops(m, n, k, async_s);
        let cublas_tf = tflops(m, n, k, cublas_s);
        let sync_vs_cu = sync_tf / cublas_tf * 100.0;
        let async_vs_cu = async_tf / cublas_tf * 100.0;

        let label = format!("{m}x{n}x{k}");
        eprintln!(
            "{:<14} {:>8.2}ms {:>8.2} {:>8.2}ms {:>8.2} {:>8.2}ms {:>8.2} {:>7.1}% {:>7.1}%",
            label,
            sync_s * 1000.0,
            sync_tf,
            async_s * 1000.0,
            async_tf,
            cublas_s * 1000.0,
            cublas_tf,
            sync_vs_cu,
            async_vs_cu,
        );

        if m == 4096 {
            sync_ms_at_4096 = sync_s * 1000.0;
            async_ms_at_4096 = async_s * 1000.0;
            cublas_ms_at_4096 = cublas_s * 1000.0;
        }
    }

    eprintln!();
    eprintln!("Summary at 4096-squared vs cuBLAS sgemm:");
    let sync_4096_pct = tflops(4096, 4096, 4096, sync_ms_at_4096 / 1000.0)
        / tflops(4096, 4096, 4096, cublas_ms_at_4096 / 1000.0)
        * 100.0;
    let async_4096_pct = tflops(4096, 4096, 4096, async_ms_at_4096 / 1000.0)
        / tflops(4096, 4096, 4096, cublas_ms_at_4096 / 1000.0)
        * 100.0;
    eprintln!("  TC sync: {sync_4096_pct:.1}% | TC async: {async_4096_pct:.1}%");
}
