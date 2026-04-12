//! GPU correctness tests for `matmul_tc_async` (Sprint 6.4 gate).
//!
//! Mirrors the four size configurations from `matmul_tc_api.rs` so the
//! async kernel is held to the same correctness bar as the sync kernel.
//! Same K-scaled tolerance (`K * 2^-10 * max_abs_input_product`),
//! same patterned inputs scaled to `|x| ≤ 1.0`, same full-diagnostic
//! panic on tolerance miss — now via the shared `common` module
//! (extracted in Sprint 6.5).
//!
//! Sprint 6.4 is **not** a performance sprint. The optional `medium`
//! test logs an elapsed-ms line when `KAIO_SPRINT_6_4_TIMING=1` is set,
//! which gives situational awareness without adding a CI benchmark.

use kaio::prelude::*;
use kaio_ops::{matmul_tc, matmul_tc_async};

mod common;
use common::{assert_close_with_k_scaled_tol, cpu_matmul_f16xf16_f32, patterned_f16_data};

fn run_matmul_tc_async_test(m: usize, n: usize, k: usize, label: &str) {
    let device = KaioDevice::new(0).expect("GPU required");
    let info = device.info().expect("device info");
    assert!(
        info.compute_capability.0 >= 8,
        "{label}: matmul_tc_async requires SM 8.0+ (have sm_{}{})",
        info.compute_capability.0,
        info.compute_capability.1
    );

    let a_host = patterned_f16_data(m * k);
    let b_host = patterned_f16_data(k * n);

    let a = device.alloc_from(&a_host).expect("alloc A");
    let b = device.alloc_from(&b_host).expect("alloc B");
    let mut c = device.alloc_zeros::<f32>(m * n).expect("alloc C");

    matmul_tc_async(&device, &a, &b, &mut c, m as u32, n as u32, k as u32)
        .unwrap_or_else(|e| panic!("{label}: matmul_tc_async failed: {e}"));

    let got = c.to_host(&device).expect("C to host");
    let expected = cpu_matmul_f16xf16_f32(&a_host, &b_host, m, n, k);
    assert_close_with_k_scaled_tol(&got, &expected, &a_host, &b_host, m, n, k, label);
}

#[test]
#[ignore] // requires NVIDIA GPU (SM 8.0+)
fn tc_async_matmul_tiny_16_8_16() {
    run_matmul_tc_async_test(16, 8, 16, "async tiny");
}

#[test]
#[ignore]
fn tc_async_matmul_small_32_16_32() {
    run_matmul_tc_async_test(32, 16, 32, "async small");
}

/// Non-square: many M blocks, single N block. Catches block-idx x↔y
/// swaps that square dimensions would mask.
#[test]
#[ignore]
fn tc_async_matmul_rect_128_8_16() {
    run_matmul_tc_async_test(128, 8, 16, "async rect");
}

#[test]
#[ignore]
fn tc_async_matmul_medium_64_64_64() {
    run_matmul_tc_async_test(64, 64, 64, "async medium");

    // Optional sanity timing — sync vs async on the same workload.
    // Gated behind an env var so CI logs stay clean and no one mistakes
    // this for a benchmark. Expected: async ≈ sync (or slightly slower)
    // at 1 warp / block; 6.7's multi-warp raises async's ceiling.
    if std::env::var("KAIO_SPRINT_6_4_TIMING").is_ok() {
        let device = KaioDevice::new(0).expect("GPU required");
        let m = 64usize;
        let n = 64usize;
        let k = 64usize;
        let a_host = patterned_f16_data(m * k);
        let b_host = patterned_f16_data(k * n);
        let a = device.alloc_from(&a_host).expect("alloc A");
        let b = device.alloc_from(&b_host).expect("alloc B");

        // Warm up both kernels (PTX compile + module load) so timings
        // reflect steady-state launch cost, not cold-start.
        let mut warm = device.alloc_zeros::<f32>(m * n).unwrap();
        matmul_tc(&device, &a, &b, &mut warm, m as u32, n as u32, k as u32).unwrap();
        matmul_tc_async(&device, &a, &b, &mut warm, m as u32, n as u32, k as u32).unwrap();
        device.stream().synchronize().ok();

        let iters = 50u32;

        let mut c_sync = device.alloc_zeros::<f32>(m * n).unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            matmul_tc(&device, &a, &b, &mut c_sync, m as u32, n as u32, k as u32).unwrap();
        }
        device.stream().synchronize().ok();
        let sync_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        let mut c_async = device.alloc_zeros::<f32>(m * n).unwrap();
        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            matmul_tc_async(&device, &a, &b, &mut c_async, m as u32, n as u32, k as u32).unwrap();
        }
        device.stream().synchronize().ok();
        let async_ms = t1.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        eprintln!(
            "TIMING (64×64×64, {iters} iters, env-gated): sync = {sync_ms:.3} ms/iter, \
             async = {async_ms:.3} ms/iter, async/sync = {:.2}×",
            async_ms / sync_ms
        );
    }
}
