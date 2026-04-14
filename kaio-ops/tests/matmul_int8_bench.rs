#![allow(clippy::too_many_arguments)]

//! Benchmark: KAIO `matmul_int8` — W8A8 symmetric tensor-core matmul
//! (Sprint 7.1, v0.3.0 reference quant op).
//!
//! Run with:
//! ```sh
//! cargo test -p kaio-ops --test matmul_int8_bench -- --ignored --nocapture
//! ```
//!
//! # Apples-to-apples disclaimer (per Sprint 7.1 plan R5, conservative)
//!
//! Comparison is against **cuBLAS sgemm** (f32 × f32 → f32) because that
//! is the cleanly-exposed cuBLAS path in cudarc 0.19. `cublasGemmEx` with
//! `CUDA_R_8I` inputs would be the true apples-to-apples baseline but
//! requires dropping to raw FFI — out of scope for 7.1 first-ship.
//!
//! The numbers reported here are a **project-local performance baseline**
//! for regression tracking across sprints, NOT a claim of apples-to-
//! apples compute-density identity with cuBLAS INT8 gemm. KAIO matmul_int8
//! moves 1 byte per operand vs 4 bytes for sgemm → the memory-bandwidth
//! profile is different. Compare sprints-over-sprints on the KAIO column,
//! not the "vs cuBLAS" column.

use std::time::Instant;

use kaio::prelude::*;
use kaio_ops::matmul_int8;

fn deterministic_data_f32(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0
        })
        .collect()
}

fn deterministic_data_i8(len: usize, seed: u64) -> Vec<i8> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 56) as i8
        })
        .collect()
}

fn bench_matmul_int8(
    device: &KaioDevice,
    a: &GpuBuffer<i8>,
    b: &GpuBuffer<i8>,
    c: &mut GpuBuffer<f32>,
    scale: f32,
    m: u32,
    n: u32,
    k: u32,
    warmup: usize,
    iters: usize,
) -> f64 {
    for _ in 0..warmup {
        matmul_int8(device, a, b, c, scale, m, n, k).unwrap();
    }
    device.stream().synchronize().unwrap();

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        device.stream().synchronize().unwrap();
        let start = Instant::now();
        matmul_int8(device, a, b, c, scale, m, n, k).unwrap();
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

fn tops(m: usize, n: usize, k: usize, seconds: f64) -> f64 {
    2.0 * m as f64 * n as f64 * k as f64 / seconds / 1e12
}

#[test]
#[ignore] // requires NVIDIA Ampere+ GPU
fn benchmark_matmul_int8() {
    let device = KaioDevice::new(0).expect("GPU required");
    let info = device.info().expect("device info");
    assert!(
        info.compute_capability.0 >= 8,
        "matmul_int8 benchmark requires SM 8.0+ (Ampere). Have sm_{}{}",
        info.compute_capability.0,
        info.compute_capability.1
    );

    eprintln!();
    eprintln!("=== KAIO matmul_int8 Benchmark (Sprint 7.1, v0.3.0) ===");
    eprintln!("GPU: {:?}", info.name);
    eprintln!(
        "Compute capability: sm_{}{}",
        info.compute_capability.0, info.compute_capability.1
    );
    eprintln!("Warm-up: 5 | Iterations: 20 | Metric: median (i8 ops × 2) / second");
    eprintln!();
    eprintln!("APPLES-TO-APPLES DISCLAIMER (per Sprint 7.1 plan R5):");
    eprintln!("  KAIO matmul_int8 is W8A8 (i8 × i8 → s32 → scale → f32).");
    eprintln!("  cuBLAS column uses sgemm (f32 × f32 → f32) as a rough compute-density");
    eprintln!("  reference; true apples-to-apples vs cublasGemmEx INT8 is out of scope");
    eprintln!("  for v0.3.0 (requires raw FFI; cudarc 0.19 does not expose it cleanly).");
    eprintln!("  Compare sprint-over-sprint on the KAIO column for regression detection;");
    eprintln!("  the 'vs cuBLAS' column is indicative, not definitive.");
    eprintln!();

    let sizes: Vec<(usize, usize, usize)> = vec![
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ];

    eprintln!(
        "{:<16} {:>12} {:>10} {:>12} {:>10} {:>12}",
        "Size", "KAIO i8 ms", "KAIO TOPS", "cuBLAS ms", "cuBLAS TF", "i8 vs sgemm"
    );
    eprintln!("{}", "-".repeat(80));

    let scale = 1.0f32 / (127.0 * 127.0); // placeholder; bench is throughput-only
    for &(m, n, k) in &sizes {
        let a_i8 = deterministic_data_i8(m * k, 42);
        let b_i8 = deterministic_data_i8(k * n, 137);
        let a_f32 = deterministic_data_f32(m * k, 42);
        let b_f32 = deterministic_data_f32(k * n, 137);

        let a_int8 = device.alloc_from(&a_i8).unwrap();
        let b_int8 = device.alloc_from(&b_i8).unwrap();
        let mut c_int8 = device.alloc_zeros::<f32>(m * n).unwrap();

        let a_cu = device.alloc_from(&a_f32).unwrap();
        let b_cu = device.alloc_from(&b_f32).unwrap();
        let mut c_cublas = device.alloc_zeros::<f32>(m * n).unwrap();

        let int8_s = bench_matmul_int8(
            &device,
            &a_int8,
            &b_int8,
            &mut c_int8,
            scale,
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

        let int8_tops = tops(m, n, k, int8_s);
        let cublas_tf = tops(m, n, k, cublas_s);
        let ratio = int8_tops / cublas_tf * 100.0;

        let label = format!("{m}x{n}x{k}");
        eprintln!(
            "{:<16} {:>10.2}ms {:>10.2} {:>10.2}ms {:>10.2} {:>10.1}%",
            label,
            int8_s * 1000.0,
            int8_tops,
            cublas_s * 1000.0,
            cublas_tf,
            ratio
        );
    }

    eprintln!();
    eprintln!("Internal-regression use: compare the 'KAIO TOPS' column against");
    eprintln!("prior sprint logs (docs/development/sprints/phase7/sprint_7_1.md).");
}
