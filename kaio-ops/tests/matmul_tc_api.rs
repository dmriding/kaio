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
