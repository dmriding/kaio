#![allow(clippy::too_many_arguments)]

//! Benchmark: KAIO `matmul_int4` — W4A16 GPTQ-style tensor-core
//! dequantize-matmul (Sprint 7.2).
//!
//! Run with:
//! ```sh
//! cargo test -p kaio-ops --test matmul_int4_bench -- --ignored --nocapture
//! ```
//!
//! # Apples-to-apples disclaimer
//!
//! Comparison is against **cuBLAS sgemm** (f32 × f32 → f32). `matmul_int4`
//! moves 0.5 B per weight and 2 B per activation vs 4 B each for sgemm —
//! the memory-bandwidth profile is fundamentally different. KAIO's
//! headline win on realistic LLM inference shapes is *bandwidth
//! reduction* (8× weight compression vs f16, 4× vs INT8), not
//! compute-bound TOPS parity. The numbers here are a project-local
//! performance baseline for regression tracking across sprints; the
//! "vs sgemm" ratio is indicative, not a definitive TOPS comparison.

use std::time::Instant;

use half::f16;
use kaio::prelude::*;
use kaio_ops::matmul_int4;

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

/// Generate deterministic packed INT4 weights: K/8 × N u32, col-major.
fn deterministic_packed_weights(k: usize, n: usize, seed: u64) -> Vec<u32> {
    assert!(k.is_multiple_of(8));
    let mut state = seed;
    (0..(k / 8) * n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 32) as u32
        })
        .collect()
}

/// Generate deterministic f16 group scales: num_groups × N row-major.
fn deterministic_scales(num_groups: usize, n: usize, seed: u64) -> Vec<f16> {
    let mut state = seed;
    (0..num_groups * n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Positive-ish scale in [0.005, 0.02] (typical quant scale magnitudes).
            let u = (state >> 40) as u32;
            let t = (u as f32 / u32::MAX as f32) * 0.015 + 0.005;
            f16::from_f32(t)
        })
        .collect()
}

fn bench_matmul_int4(
    device: &KaioDevice,
    a: &GpuBuffer<f16>,
    b: &GpuBuffer<u32>,
    scales: &GpuBuffer<f16>,
    c: &mut GpuBuffer<f32>,
    m: u32,
    n: u32,
    k: u32,
    warmup: usize,
    iters: usize,
) -> f64 {
    for _ in 0..warmup {
        matmul_int4(device, a, b, scales, c, m, n, k, 128).unwrap();
    }
    device.stream().synchronize().unwrap();

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        device.stream().synchronize().unwrap();
        let start = Instant::now();
        matmul_int4(device, a, b, scales, c, m, n, k, 128).unwrap();
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
fn benchmark_matmul_int4() {
    let device = KaioDevice::new(0).expect("GPU required");
    let info = device.info().expect("device info");
    assert!(
        info.compute_capability.0 >= 8,
        "matmul_int4 benchmark requires SM 8.0+ (Ampere). Have sm_{}{}",
        info.compute_capability.0,
        info.compute_capability.1
    );

    eprintln!();
    eprintln!("=== KAIO matmul_int4 Benchmark (Sprint 7.2) ===");
    eprintln!("GPU: {:?}", info.name);
    eprintln!(
        "Compute capability: sm_{}{}",
        info.compute_capability.0, info.compute_capability.1
    );
    eprintln!("Warm-up: 5 | Iterations: 20 | Metric: median (int4 ops × 2) / second");
    eprintln!();
    eprintln!("APPLES-TO-APPLES DISCLAIMER:");
    eprintln!("  matmul_int4 is W4A16 (packed s4 × f16 → f32 via dequant-to-f16 + mma).");
    eprintln!("  cuBLAS column uses sgemm (f32 × f32 → f32). Weight bandwidth in KAIO is");
    eprintln!("  0.5 B/weight vs 4 B for sgemm — the memory-bandwidth profile is");
    eprintln!("  fundamentally different. This column is a project-local baseline for");
    eprintln!("  regression detection, not a definitive TOPS comparison.");
    eprintln!();

    let sizes: Vec<(usize, usize, usize)> = vec![
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ];

    eprintln!(
        "{:<16} {:>12} {:>10} {:>12} {:>10} {:>12}",
        "Size", "KAIO i4 ms", "KAIO TOPS", "cuBLAS ms", "cuBLAS TF", "i4 vs sgemm"
    );
    eprintln!("{}", "-".repeat(80));

    for &(m, n, k) in &sizes {
        assert!(
            k.is_multiple_of(128),
            "bench K={k} must be multiple of 128 (group_size)"
        );
        let num_groups = k / 128;

        let a_f16 = deterministic_data_f16(m * k, 42);
        let b_packed = deterministic_packed_weights(k, n, 137);
        let scales = deterministic_scales(num_groups, n, 4242);
        let a_f32 = deterministic_data_f32(m * k, 42);
        let b_f32 = deterministic_data_f32(k * n, 137);

        let a_gpu = device.alloc_from(&a_f16).unwrap();
        let b_gpu = device.alloc_from(&b_packed).unwrap();
        let s_gpu = device.alloc_from(&scales).unwrap();
        let mut c_gpu = device.alloc_zeros::<f32>(m * n).unwrap();

        let a_cu = device.alloc_from(&a_f32).unwrap();
        let b_cu = device.alloc_from(&b_f32).unwrap();
        let mut c_cublas = device.alloc_zeros::<f32>(m * n).unwrap();

        let int4_s = bench_matmul_int4(
            &device, &a_gpu, &b_gpu, &s_gpu, &mut c_gpu, m as u32, n as u32, k as u32, 5, 20,
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

        let int4_tops = tops(m, n, k, int4_s);
        let cublas_tf = tops(m, n, k, cublas_s);
        let ratio = int4_tops / cublas_tf * 100.0;

        let label = format!("{m}x{n}x{k}");
        eprintln!(
            "{:<16} {:>10.2}ms {:>10.2} {:>10.2}ms {:>10.2} {:>10.1}%",
            label,
            int4_s * 1000.0,
            int4_tops,
            cublas_s * 1000.0,
            cublas_tf,
            ratio
        );
    }

    eprintln!();
    eprintln!("Internal-regression use: compare the 'KAIO TOPS' column against");
    eprintln!("prior sprint logs (docs/development/sprints/phase7/sprint_7_2.md).");
}
