//! GPU correctness tests for `matmul_tc` (Sprint 6.3 gate).
//!
//! Four size configurations, each compared element-wise against a fp32
//! CPU reference. Tolerance scales with K using `K * 2^-10 *
//! max_abs_input_product` — tighter than a fixed constant, catches
//! regressions a loose tolerance would hide.
//!
//! Inputs are patterned (not random) and scaled to `|x| ≤ 1.0` to keep
//! the tolerance bound interpretable.

use half::f16;
use kaio::prelude::*;
use kaio_ops::matmul_tc;

/// CPU reference: promote f16 to f32, multiply, accumulate in f32.
fn cpu_matmul_f16xf16_f32(a: &[f16], b: &[f16], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                let av = a[i * k + p].to_f32();
                let bv = b[p * n + j].to_f32();
                sum += av * bv;
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// Generate patterned f16 data scaled to |x| ≤ 1.0.
/// Avoids randomness for reproducibility; avoids all-zeros / all-ones
/// to keep products non-trivial.
fn patterned_f16_data(len: usize) -> Vec<f16> {
    (0..len)
        .map(|i| {
            // (i % 17) / 17 - 0.5 → uniform-ish in [-0.5, 0.5].
            let v = ((i % 17) as f32) / 17.0 - 0.5;
            f16::from_f32(v)
        })
        .collect()
}

/// Run one matmul_tc correctness test with full diagnostic on failure.
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

    // K-scaled tolerance: K * 2^-10 * max_abs_input_product
    let max_abs_a = a_host
        .iter()
        .map(|x| x.to_f32().abs())
        .fold(0.0f32, f32::max);
    let max_abs_b = b_host
        .iter()
        .map(|x| x.to_f32().abs())
        .fold(0.0f32, f32::max);
    let max_abs_input_product = max_abs_a * max_abs_b;
    let abs_tol = (k as f32) * 2f32.powi(-10) * max_abs_input_product;

    // Walk outputs; track worst case; on failure, emit full diagnostic.
    let mut worst_idx: Option<(usize, usize)> = None;
    let mut worst_abs_err = 0.0f32;
    let mut worst_got = 0.0f32;
    let mut worst_expected = 0.0f32;
    for i in 0..m {
        for j in 0..n {
            let idx = i * n + j;
            let g = got[idx];
            let e = expected[idx];
            let abs_err = (g - e).abs();
            if abs_err > worst_abs_err {
                worst_abs_err = abs_err;
                worst_idx = Some((i, j));
                worst_got = g;
                worst_expected = e;
            }
        }
    }

    if worst_abs_err >= abs_tol {
        let (wi, wj) = worst_idx.unwrap_or((0, 0));
        let rel = if worst_expected.abs() > 1e-6 {
            (worst_got - worst_expected) / worst_expected
        } else {
            worst_got - worst_expected
        };
        panic!(
            "{label} ({m}×{k} × {k}×{n}) FAILED bit-close tolerance:\n\
             \n\
             K                    = {k}\n\
             max_abs_input_product = {max_abs_input_product:e}\n\
             abs_tol (= K * 2^-10 * max_abs_input_product) = {abs_tol:e}\n\
             worst_abs_err        = {worst_abs_err:e}\n\
             worst_abs_err / tol  = {:.3}\n\
             worst index (i, j)   = ({wi}, {wj})\n\
             CPU reference value  = {worst_expected}\n\
             GPU value            = {worst_got}\n\
             (got - expected)     = {}\n\
             relative error       = {rel:e}\n",
            worst_abs_err / abs_tol,
            worst_got - worst_expected,
        );
    }

    // Success diagnostic — prints the usage ratio so regressions near
    // the bound become visible before they start failing.
    eprintln!(
        "{label} ({m}×{k} × {k}×{n}): max_abs_err = {worst_abs_err:e}, \
         abs_tol = {abs_tol:e}, usage = {:.1}% of tolerance",
        100.0 * worst_abs_err / abs_tol
    );
}

// --- GPU correctness tests ---

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
