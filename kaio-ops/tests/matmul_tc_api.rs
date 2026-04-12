//! GPU correctness tests for `matmul_tc` (Sprint 6.3 gate).
//!
//! Four size configurations, each compared element-wise against a fp32
//! CPU reference. Tolerance scales with K using `K * 2^-10 *
//! max_abs_input_product` — tighter than a fixed constant, catches
//! regressions a loose tolerance would hide.
//!
//! Inputs are patterned (not random) and scaled to `|x| ≤ 1.0` to keep
//! the tolerance bound interpretable.
//!
//! Shared helpers (generator, CPU reference, tolerance check) live in
//! `tests/common/mod.rs` as of Sprint 6.5. Three test files were
//! already duplicating them before the tuner test was added.

use kaio::prelude::*;
use kaio_ops::matmul_tc;

mod common;
use common::{assert_close_with_k_scaled_tol, cpu_matmul_f16xf16_f32, patterned_f16_data};

fn run_matmul_tc_test(m: usize, n: usize, k: usize, label: &str) {
    let device = KaioDevice::new(0).expect("GPU required");
    let info = device.info().expect("device info");
    assert!(
        info.compute_capability.0 >= 8,
        "{label}: matmul_tc requires SM 8.0+ (have sm_{}{})",
        info.compute_capability.0,
        info.compute_capability.1
    );

    let a_host = patterned_f16_data(m * k);
    let b_host = patterned_f16_data(k * n);

    let a = device.alloc_from(&a_host).expect("alloc A");
    let b = device.alloc_from(&b_host).expect("alloc B");
    let mut c = device.alloc_zeros::<f32>(m * n).expect("alloc C");

    matmul_tc(&device, &a, &b, &mut c, m as u32, n as u32, k as u32)
        .unwrap_or_else(|e| panic!("{label}: matmul_tc failed: {e}"));

    let got = c.to_host(&device).expect("C to host");
    let expected = cpu_matmul_f16xf16_f32(&a_host, &b_host, m, n, k);
    assert_close_with_k_scaled_tol(&got, &expected, &a_host, &b_host, m, n, k, label);
}

#[test]
#[ignore] // requires NVIDIA GPU (SM 8.0+)
fn tc_matmul_tiny_16_8_16() {
    run_matmul_tc_test(16, 8, 16, "tiny");
}

#[test]
#[ignore]
fn tc_matmul_small_32_16_32() {
    run_matmul_tc_test(32, 16, 32, "small");
}

/// Non-square test (Opus review catch): many M blocks, one N block,
/// one K tile. Catches `block_idx_x` ↔ `block_idx_y` swaps that square
/// dimensions would mask.
#[test]
#[ignore]
fn tc_matmul_rect_128_8_16() {
    run_matmul_tc_test(128, 8, 16, "rect");
}

#[test]
#[ignore]
fn tc_matmul_medium_64_64_64() {
    run_matmul_tc_test(64, 64, 64, "medium");
}

/// Sprint 6.7 Gate A canary — per-warp distinguishable input pattern at
/// 64×64×64 (one block, all 4 warps in-bounds). A is row-index-folded
/// (`A[i,j] = (i+1) * 0.01`), B is col-index-folded (`B[i,j] = (j+1) *
/// 0.01`); each output cell `(i, j) = (i+1) * 0.01 * sum_k (j+1) * 0.01
/// = (i+1) * (j+1) * K * 0.0001`. Per-cell value depends uniquely on
/// `(i, j)`, so any per-warp routing bug (e.g. warp 1 writing into
/// warp 0's quadrant) produces a wrong, detectable value at that cell.
/// Catches multi-warp restructure misalignment that uniform-input tests
/// can mask.
#[test]
#[ignore]
fn tc_matmul_multi_warp_quadrant_canary_64_64_64() {
    use half::f16;
    let device = KaioDevice::new(0).expect("GPU required");
    let info = device.info().expect("device info");
    assert!(
        info.compute_capability.0 >= 8,
        "quadrant canary: matmul_tc requires SM 8.0+"
    );

    let (m, n, k) = (64usize, 64, 64);
    let a_host: Vec<f16> = (0..m * k)
        .map(|idx| {
            let i = idx / k;
            f16::from_f32((i as f32 + 1.0) * 0.01)
        })
        .collect();
    let b_host: Vec<f16> = (0..k * n)
        .map(|idx| {
            let j = idx % n;
            f16::from_f32((j as f32 + 1.0) * 0.01)
        })
        .collect();

    let a = device.alloc_from(&a_host).expect("alloc A");
    let b = device.alloc_from(&b_host).expect("alloc B");
    let mut c = device.alloc_zeros::<f32>(m * n).expect("alloc C");

    matmul_tc(&device, &a, &b, &mut c, m as u32, n as u32, k as u32).expect("matmul_tc launch");
    let got = c.to_host(&device).expect("C to host");

    let expected = cpu_matmul_f16xf16_f32(&a_host, &b_host, m, n, k);
    assert_close_with_k_scaled_tol(
        &got,
        &expected,
        &a_host,
        &b_host,
        m,
        n,
        k,
        "quadrant_canary",
    );

    // Spot-check a cell in each of the 4 warp quadrants.
    // Expected: (i+1) * (j+1) * K * 0.01 * 0.01 = (i+1) * (j+1) * 0.0064
    let probes = [
        (8usize, 8), // warp 0 (rows 0-31, cols 0-31)
        (8, 40),     // warp 1 (rows 0-31, cols 32-63)
        (40, 8),     // warp 2 (rows 32-63, cols 0-31)
        (40, 40),    // warp 3 (rows 32-63, cols 32-63)
    ];
    for (i, j) in probes {
        let analytic = (i as f32 + 1.0) * (j as f32 + 1.0) * (k as f32) * 0.0001;
        let observed = got[i * n + j];
        let rel_err = (observed - analytic).abs() / analytic.abs().max(1e-6);
        assert!(
            rel_err < 0.01,
            "warp quadrant canary at ({i},{j}): expected ~{analytic:.4}, got {observed:.4} (rel_err {rel_err:.4})"
        );
    }
}
