#![allow(clippy::too_many_arguments)]

//! Benchmark: fused tri-output QKV projection vs 3× standalone projection.
//!
//! Sprint 7.3 D8. Measures:
//! - **Relative ratio**: fused `qkv_project_int4` vs three sequential
//!   [`matmul_int4`] calls (clean W4A16 apples-to-apples — same mma path,
//!   same weight bandwidth, same group-scale cadence).
//! - **Absolute TOPS**: fused `qkv_project_int{4,8}` throughput in
//!   isolation, versus the standalone `matmul_int{4,8}` ceilings from
//!   Sprint 7.1/7.2 headline numbers.
//!
//! # Tiered ship criteria (per Sprint 7.3 plan D8)
//!
//! - ≥ 1.3× baseline at prefill → celebrate
//! - ≥ 1.15× baseline at prefill → ship
//! - 1.0–1.15× at prefill → ship but frame narrowly
//! - < 1.0× at prefill OR absolute envelope off by > 50% → halt, activate D4.5
//!
//! Run with:
//! ```sh
//! cargo test -p kaio-ops --test qkv_project_bench -- --ignored --nocapture
//! ```

use std::time::Instant;

use half::f16;
use kaio::prelude::*;
use kaio_ops::{matmul_int4, qkv_project_int4, qkv_project_int8};

fn deterministic_f16(len: usize, seed: u64) -> Vec<f16> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let f = ((state >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0;
            f16::from_f32(f * 0.25)
        })
        .collect()
}

fn deterministic_i8(len: usize, seed: u64) -> Vec<i8> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 56) as i8) / 4
        })
        .collect()
}

fn deterministic_packed(k: usize, n: usize, seed: u64) -> Vec<u32> {
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

fn deterministic_scales(num_groups: usize, n: usize, seed: u64) -> Vec<f16> {
    let mut state = seed;
    (0..num_groups * n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (state >> 40) as u32;
            let t = (u as f32 / u32::MAX as f32) * 0.015 + 0.005;
            f16::from_f32(t)
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

/// One fused QKV projection = `3 × M × N × K` multiply-adds × 2 FLOPs = `6 MNK` FLOPs.
fn qkv_tops(m: usize, n: usize, k: usize, seconds: f64) -> f64 {
    6.0 * m as f64 * n as f64 * k as f64 / seconds / 1e12
}

#[test]
#[ignore]
fn benchmark_qkv_project_int4_fused_vs_three_matmul_int4() {
    let device = KaioDevice::new(0).expect("GPU required");
    let info = device.info().expect("device info");
    assert!(
        info.compute_capability.0 >= 8,
        "qkv_project_int4 requires SM 8.0+; got sm_{}{}",
        info.compute_capability.0,
        info.compute_capability.1
    );

    eprintln!();
    eprintln!("=== Sprint 7.3 D8: qkv_project_int4 (fused) vs 3× matmul_int4 ===");
    eprintln!(
        "GPU: {:?}  sm_{}{}",
        info.name, info.compute_capability.0, info.compute_capability.1
    );
    eprintln!("Warmup 5 | Iters 20 | Median");
    eprintln!();

    // Decode + prefill tiers per plan. K/N must be multiples of 128 (GROUP_SIZE).
    // M + N in {BM_BLOCK, BN_BLOCK} grid terms (BM_BLOCK=64, BN_BLOCK=16 post-rollback).
    let shapes = [
        // Decode tier (small M).
        ("decode_m1", 1usize, 2048usize, 2048usize),
        ("decode_m64", 64, 2048, 2048),
        ("decode_m64_large", 64, 4096, 4096),
        // Outlier diagnostic sweep around decode_m64 K/N=2048.
        ("diag_m64_n1024_k2048", 64, 1024, 2048),
        ("diag_m64_n2048_k1024", 64, 2048, 1024),
        ("diag_m64_n2048_k4096", 64, 2048, 4096),
        ("diag_m128_n2048_k2048", 128, 2048, 2048),
        // Prefill tier (large M).
        ("prefill_m512", 512, 4096, 4096),
        ("prefill_m2048", 2048, 4096, 4096),
    ];

    eprintln!(
        "{:<22} {:>12} {:>12} {:>8} {:>10} {:>10}",
        "shape", "fused ms", "3xmm ms", "ratio", "fused TOPS", "3xmm TOPS"
    );
    eprintln!("{}", "-".repeat(80));

    for (label, m_raw, n_raw, k_raw) in shapes {
        // Round M up to BM_BLOCK = 64 for the launch geometry (M=1 decode
        // still launches one block's worth of threads; the fused kernel has
        // no M-edge predication so we benchmark the full block).
        let m = m_raw.max(64);
        let n = n_raw;
        let k = k_raw;
        let num_groups = k / 128;

        let x = deterministic_f16(m * k, 1);
        let w_q = deterministic_packed(k, n, 2);
        let w_k = deterministic_packed(k, n, 3);
        let w_v = deterministic_packed(k, n, 4);
        let s_q = deterministic_scales(num_groups, n, 5);
        let s_k = deterministic_scales(num_groups, n, 6);
        let s_v = deterministic_scales(num_groups, n, 7);

        let x_gpu = device.alloc_from(&x).unwrap();
        let w_q_gpu = device.alloc_from(&w_q).unwrap();
        let w_k_gpu = device.alloc_from(&w_k).unwrap();
        let w_v_gpu = device.alloc_from(&w_v).unwrap();
        let s_q_gpu = device.alloc_from(&s_q).unwrap();
        let s_k_gpu = device.alloc_from(&s_k).unwrap();
        let s_v_gpu = device.alloc_from(&s_v).unwrap();
        let mut q_out = device.alloc_zeros::<f16>(m * n).unwrap();
        let mut k_out = device.alloc_zeros::<f16>(m * n).unwrap();
        let mut v_out = device.alloc_zeros::<f16>(m * n).unwrap();
        let mut q_f32 = device.alloc_zeros::<f32>(m * n).unwrap();
        let mut k_f32 = device.alloc_zeros::<f32>(m * n).unwrap();
        let mut v_f32 = device.alloc_zeros::<f32>(m * n).unwrap();

        let fused_s = median_seconds(
            || {
                qkv_project_int4(
                    &device, &x_gpu, &w_q_gpu, &w_k_gpu, &w_v_gpu, &s_q_gpu, &s_k_gpu, &s_v_gpu,
                    &mut q_out, &mut k_out, &mut v_out, m as u32, n as u32, k as u32, 128,
                )
                .unwrap();
                device.stream().synchronize().unwrap();
            },
            5,
            20,
        );

        let baseline_s = median_seconds(
            || {
                matmul_int4(
                    &device, &x_gpu, &w_q_gpu, &s_q_gpu, &mut q_f32, m as u32, n as u32, k as u32,
                    128,
                )
                .unwrap();
                matmul_int4(
                    &device, &x_gpu, &w_k_gpu, &s_k_gpu, &mut k_f32, m as u32, n as u32, k as u32,
                    128,
                )
                .unwrap();
                matmul_int4(
                    &device, &x_gpu, &w_v_gpu, &s_v_gpu, &mut v_f32, m as u32, n as u32, k as u32,
                    128,
                )
                .unwrap();
                device.stream().synchronize().unwrap();
            },
            5,
            20,
        );

        let ratio = baseline_s / fused_s;
        let f_tops = qkv_tops(m, n, k, fused_s);
        let b_tops = qkv_tops(m, n, k, baseline_s);
        eprintln!(
            "{label:<22} {:>12.3} {:>12.3} {:>8.2}x {:>10.1} {:>10.1}",
            fused_s * 1e3,
            baseline_s * 1e3,
            ratio,
            f_tops,
            b_tops,
        );
    }

    eprintln!();
    eprintln!("Ratio > 1.0 means fused wins. Plan ship thresholds:");
    eprintln!("  >=1.30x at prefill → celebrate | >=1.15x → ship | >=1.00x → ship narrow");
}

#[test]
#[ignore]
fn benchmark_qkv_project_int8_absolute_tops() {
    let device = KaioDevice::new(0).expect("GPU required");
    let info = device.info().expect("device info");
    assert!(
        info.compute_capability.0 >= 8,
        "qkv_project_int8 requires SM 8.0+; got sm_{}{}",
        info.compute_capability.0,
        info.compute_capability.1
    );

    eprintln!();
    eprintln!("=== Sprint 7.3 D8: qkv_project_int8 (fused W8A16) absolute TOPS ===");
    eprintln!(
        "GPU: {:?}  sm_{}{}",
        info.name, info.compute_capability.0, info.compute_capability.1
    );
    eprintln!();
    eprintln!("Note: INT8 uses W8A16 (i8 weights × f16 activations). No apples-to-apples");
    eprintln!("      W8A16 standalone op exists - reporting absolute TOPS only. matmul_int8");
    eprintln!("      (Sprint 7.1) is W8A8 and cannot serve as a fair 3x baseline here.");
    eprintln!();

    let shapes = [
        ("decode_m1", 1usize, 2048usize, 2048usize),
        ("decode_m64", 64, 2048, 2048),
        ("decode_m64_large", 64, 4096, 4096),
        ("prefill_m512", 512, 4096, 4096),
        ("prefill_m2048", 2048, 4096, 4096),
    ];

    eprintln!("{:<22} {:>12} {:>10}", "shape", "fused ms", "TOPS");
    eprintln!("{}", "-".repeat(50));

    for (label, m_raw, n, k) in shapes {
        let m = m_raw.max(64);

        let x = deterministic_f16(m * k, 1);
        let w_q = deterministic_i8(k * n, 2);
        let w_k = deterministic_i8(k * n, 3);
        let w_v = deterministic_i8(k * n, 4);

        let x_gpu = device.alloc_from(&x).unwrap();
        let w_q_gpu = device.alloc_from(&w_q).unwrap();
        let w_k_gpu = device.alloc_from(&w_k).unwrap();
        let w_v_gpu = device.alloc_from(&w_v).unwrap();
        let mut q_out = device.alloc_zeros::<f16>(m * n).unwrap();
        let mut k_out = device.alloc_zeros::<f16>(m * n).unwrap();
        let mut v_out = device.alloc_zeros::<f16>(m * n).unwrap();

        let scale = 1.0 / (k as f32);
        let fused_s = median_seconds(
            || {
                qkv_project_int8(
                    &device, &x_gpu, &w_q_gpu, &w_k_gpu, &w_v_gpu, scale, scale, scale, &mut q_out,
                    &mut k_out, &mut v_out, m as u32, n as u32, k as u32,
                )
                .unwrap();
                device.stream().synchronize().unwrap();
            },
            5,
            20,
        );
        let tops = qkv_tops(m, n, k, fused_s);
        eprintln!("{label:<22} {:>12.3} {:>10.1}", fused_s * 1e3, tops);
    }
}
