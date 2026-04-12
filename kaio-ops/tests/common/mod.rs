//! Shared test helpers for kaio-ops integration tests.
//!
//! Extracted in Sprint 6.5 when the third matmul-TC test file
//! (`tuner_tc_test.rs`) would have required a fourth copy of the
//! patterned f16 generator + CPU reference + K-scaled tolerance
//! machinery. Three copies is tolerable; four is the breaking
//! point.
//!
//! Each integration-test file declares `mod common;` — Rust
//! compiles one copy of this module per test binary (that's how
//! the integration-test harness works; not ideal, but cheaper than
//! every file re-inlining the helpers).

use half::f16;

/// Generate patterned f16 data scaled to |x| ≤ 1.0.
///
/// Patterned (not random) for reproducibility; avoids all-zero /
/// all-one inputs so products are non-trivial and sign changes
/// don't accidentally cancel. Used by every kaio-ops TC test.
pub fn patterned_f16_data(len: usize) -> Vec<f16> {
    (0..len)
        .map(|i| {
            // (i % 17) / 17 - 0.5 → uniform-ish in [-0.5, 0.5].
            let v = ((i % 17) as f32) / 17.0 - 0.5;
            f16::from_f32(v)
        })
        .collect()
}

/// fp32 CPU reference matmul: promote f16 to f32, multiply,
/// accumulate in f32. Matches the GPU kernels' mixed-precision
/// contract (f16 inputs, f32 accumulate).
pub fn cpu_matmul_f16xf16_f32(a: &[f16], b: &[f16], m: usize, n: usize, k: usize) -> Vec<f32> {
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

/// Compare a GPU result against a CPU reference using the K-scaled
/// tolerance formula `K * 2^-10 * max_abs_input_product`. Panics
/// with a full diagnostic block (label, dims, tolerance, worst
/// element, CPU and GPU values at that element, relative error)
/// if any output exceeds the bound. On success, prints the
/// tolerance-usage percentage to stderr so regressions approaching
/// the bound are visible before they fail.
///
/// Callers pass `a_host` / `b_host` so `max_abs_input_product` can
/// be computed against the actual inputs rather than assumed to be
/// at the |x| ≤ 1.0 ceiling.
#[allow(clippy::too_many_arguments)]
pub fn assert_close_with_k_scaled_tol(
    got: &[f32],
    expected: &[f32],
    a_host: &[f16],
    b_host: &[f16],
    m: usize,
    n: usize,
    k: usize,
    label: &str,
) {
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

    eprintln!(
        "{label} ({m}×{k} × {k}×{n}): max_abs_err = {worst_abs_err:e}, \
         abs_tol = {abs_tol:e}, usage = {:.1}% of tolerance",
        100.0 * worst_abs_err / abs_tol
    );
}
