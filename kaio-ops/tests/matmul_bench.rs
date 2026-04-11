#![allow(clippy::too_many_arguments)]

//! Benchmark: KAIO tiled matmul vs cuBLAS sgemm.
//!
//! Run with:
//! ```sh
//! cargo test -p kaio-ops --test matmul_bench -- --ignored --nocapture
//! ```
//!
//! For cuBLAS comparison (requires CUDA toolkit):
//! ```sh
//! cargo test -p kaio-ops --test matmul_bench -- --ignored --nocapture
//! ```
//! cuBLAS is available as a dev-dependency — no feature flag needed.

use std::time::Instant;

use kaio::prelude::*;
use kaio_ops::matmul;

// --- Deterministic random data ---

fn deterministic_data(len: usize, seed: u64) -> Vec<f32> {
    // Simple LCG — deterministic, good enough for benchmark inputs
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Map to [-1.0, 1.0]
            ((state >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0
        })
        .collect()
}

// --- CPU reference ---

fn cpu_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

// --- KAIO benchmark ---

fn bench_kaio(
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
    // Warm-up
    for _ in 0..warmup {
        matmul(device, a, b, c, m, n, k).unwrap();
    }
    device.stream().synchronize().unwrap();

    // Timed iterations
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        device.stream().synchronize().unwrap();
        let start = Instant::now();
        matmul(device, a, b, c, m, n, k).unwrap();
        device.stream().synchronize().unwrap();
        times.push(start.elapsed().as_secs_f64());
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[iters / 2] // median
}

// --- cuBLAS benchmark ---

fn bench_cublas(
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

    // cuBLAS row-major trick: C = A×B in row-major is C^T = B^T × A^T in column-major
    // GEMM params: m=N, n=M, k=K, lda=N, ldb=K, ldc=N
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

    // Warm-up
    for _ in 0..warmup {
        unsafe {
            blas.gemm(cfg, b.inner(), a.inner(), c.inner_mut()).unwrap();
        }
    }
    device.stream().synchronize().unwrap();

    // Timed iterations
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
    times[iters / 2] // median
}

fn tflops(m: usize, n: usize, k: usize, seconds: f64) -> f64 {
    2.0 * m as f64 * n as f64 * k as f64 / seconds / 1e12
}

// --- Benchmark test ---

#[test]
#[ignore] // requires NVIDIA GPU + CUDA toolkit (for cuBLAS)
fn benchmark_matmul() {
    let device = KaioDevice::new(0).expect("GPU required");

    // Print header
    eprintln!();
    eprintln!("=== KAIO Matmul Benchmark ===");
    eprintln!(
        "GPU: {:?}",
        device.info().map(|i| i.name).unwrap_or_default()
    );
    eprintln!("Warm-up: 5 | Iterations: 20 | Metric: median TFLOPS");
    eprintln!();

    // Validate cuBLAS correctness before trusting it as baseline
    eprintln!("--- cuBLAS correctness validation ---");
    {
        let (m, n, k) = (64usize, 64, 64);
        let a_data = deterministic_data(m * k, 42);
        let b_data = deterministic_data(k * n, 137);
        let expected = cpu_matmul(&a_data, &b_data, m, n, k);

        let a = device.alloc_from(&a_data).unwrap();
        let b = device.alloc_from(&b_data).unwrap();
        let mut c = device.alloc_zeros::<f32>(m * n).unwrap();

        // Run cuBLAS
        {
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
            unsafe {
                blas.gemm(cfg, b.inner(), a.inner(), c.inner_mut()).unwrap();
            }
        }

        let result = c.to_host(&device).unwrap();
        let mut max_abs = 0.0f32;
        for i in 0..m * n {
            let err = (result[i] - expected[i]).abs();
            if err > max_abs {
                max_abs = err;
            }
        }
        assert!(
            max_abs < 1e-3,
            "cuBLAS validation FAILED: max_abs={max_abs:.2e}. Row-major trick may be wrong."
        );
        eprintln!("cuBLAS 64x64 validated: max_abs={max_abs:.2e} ✓");
    }

    // Benchmark sizes
    let sizes: Vec<(usize, usize, usize)> = vec![
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (1024, 2048, 512), // non-square
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ];

    eprintln!();
    eprintln!(
        "{:<20} {:>12} {:>12} {:>12} {:>12} {:>8}",
        "Size", "KAIO ms", "KAIO TFLOPS", "cuBLAS ms", "cuBLAS TFLOPS", "Ratio"
    );
    eprintln!("{}", "-".repeat(80));

    for &(m, n, k) in &sizes {
        let a_data = deterministic_data(m * k, 42);
        let b_data = deterministic_data(k * n, 137);

        let a = device.alloc_from(&a_data).unwrap();
        let b = device.alloc_from(&b_data).unwrap();
        let mut c_kaio = device.alloc_zeros::<f32>(m * n).unwrap();
        let mut c_cublas = device.alloc_zeros::<f32>(m * n).unwrap();

        let kaio_s = bench_kaio(
            &device,
            &a,
            &b,
            &mut c_kaio,
            m as u32,
            n as u32,
            k as u32,
            5,
            20,
        );
        let cublas_s = bench_cublas(
            &device,
            &a,
            &b,
            &mut c_cublas,
            m as u32,
            n as u32,
            k as u32,
            5,
            20,
        );

        let kaio_tflops = tflops(m, n, k, kaio_s);
        let cublas_tflops = tflops(m, n, k, cublas_s);
        let ratio = kaio_tflops / cublas_tflops * 100.0;

        let label = format!("{m}×{n}×{k}");

        eprintln!(
            "{:<20} {:>10.2}ms {:>10.2} {:>10.2}ms {:>10.2} {:>7.1}%",
            label,
            kaio_s * 1000.0,
            kaio_tflops,
            cublas_s * 1000.0,
            cublas_tflops,
            ratio,
        );
    }

    eprintln!();
    eprintln!("Note: KAIO uses naive 16×16 tiled matmul. Optimization is Sprint 4.6.");
}
