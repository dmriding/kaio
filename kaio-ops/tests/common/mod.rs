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
//! every file re-inlining the helpers). Per-binary the unused
//! helpers look dead; `#[allow(dead_code)]` silences the false
//! positives without hiding genuinely-unused code in this module
//! (verified by `cargo test --workspace` — any pruned helper
//! would break the binary that uses it, not this warning path).

#![allow(dead_code)]

use half::{bf16, f16};

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

// ============================================================================
// Sprint 9.1 — bf16 TC matmul helpers
// ============================================================================
//
// Sibling of the f16 trio above. Per the Sprint 9.1 D5 reference-strategy
// table, small/medium bf16 correctness tests use a dense f64 CPU reference
// (cheaper than f32-accumulated reference would be misleading: bf16's weak
// 7-bit mantissa makes the f64 reference materially tighter than f32 for
// dot-product cancellation patterns). Large shapes use sampled-cell f64 —
// that helper lands at C5 when the full correctness suite arrives.

/// Generate patterned bf16 data scaled to |x| ≤ 1.0.
///
/// Same `(i % 17) / 17 - 0.5` deterministic pattern as
/// [`patterned_f16_data`] — keeps the comparison apples-to-apples
/// across the f16 and bf16 paths. bf16's wider exponent doesn't
/// matter at this magnitude; mantissa precision is what differs.
pub fn patterned_bf16_data(len: usize) -> Vec<bf16> {
    (0..len)
        .map(|i| {
            let v = ((i % 17) as f32) / 17.0 - 0.5;
            bf16::from_f32(v)
        })
        .collect()
}

/// f64 CPU reference matmul for bf16 inputs.
///
/// Promotes each bf16 input to f64 and accumulates in f64. Per D5,
/// f64 is the reference for the bf16 correctness suite because bf16's
/// 7-bit mantissa can lose precision in cancellation-prone dot products
/// at sizes where f32 accumulation would also be lossy enough to mask
/// real kernel bugs. Used at small/medium shapes; sampled-cell f64
/// for large shapes lands at C5.
pub fn cpu_matmul_bf16xbf16_f64(a: &[bf16], b: &[bf16], m: usize, n: usize, k: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f64;
            for p in 0..k {
                let av = a[i * k + p].to_f32() as f64;
                let bv = b[p * n + j].to_f32() as f64;
                sum += av * bv;
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// Generate patterned bf16 data with a target magnitude.
///
/// Same base pattern as [`patterned_bf16_data`] (`(i % 17) / 17 - 0.5`),
/// scaled by `magnitude_scale` so peak |x| ≈ `0.5 * magnitude_scale`.
/// Used for the medium (`scale = 200`, |x| ≤ 100) and large
/// (`scale = 2e14`, |x| ≤ 1e14) magnitude classes in the bf16
/// correctness suite. The large-magnitude cap of 1e14 keeps
/// `|a| × |b| × K` comfortably under f32 max in the accumulator
/// while still exercising bf16's full f32-style exponent range —
/// well past f16's 65504 overflow point.
pub fn patterned_bf16_magnitude(len: usize, magnitude_scale: f32) -> Vec<bf16> {
    (0..len)
        .map(|i| {
            let v = (((i % 17) as f32) / 17.0 - 0.5) * magnitude_scale;
            bf16::from_f32(v)
        })
        .collect()
}

/// Generate positive-only near-denorm bf16 data in `[1e-18, 2e-18]`.
///
/// The bf16 near-denorm magnitude class for the correctness suite.
/// Positive-only and non-cancelling — a symmetric `[-1e-18, 1e-18]`
/// generator would let an all-zero kernel pass the absolute-tolerance
/// check vacuously at expected outputs `~K × 1e-36`. The positive
/// bias keeps products from cancelling toward zero so the
/// nonzero-output assertion in [`assert_bf16_close_d5_near_denorm`]
/// has signal.
pub fn patterned_bf16_near_denorm(len: usize) -> Vec<bf16> {
    (0..len)
        .map(|i| {
            // (i % 17) / 17 ∈ [0, ~0.94], shifted to [1.0, ~1.94] then × 1e-18
            // → [1e-18, ~1.94e-18]. Strictly positive, deterministic.
            let v = (1.0 + ((i % 17) as f32) / 17.0) * 1e-18;
            bf16::from_f32(v)
        })
        .collect()
}

/// Deterministic-seed sampling of `n_samples` cell coordinates from an
/// `m × n` output grid. Uses an inline Numerical-Recipes LCG so we
/// don't pull in `rand` as a dev-dep just for this — the requirement
/// is reproducibility and decorrelation between cells, not crypto
/// quality.
///
/// Returns each cell as `(i, j)` with `0 ≤ i < m`, `0 ≤ j < n`. May
/// return duplicates at small grid sizes; that's fine — the assertion
/// just compares each sampled cell independently.
pub fn sample_cells(m: usize, n: usize, n_samples: usize, seed: u64) -> Vec<(usize, usize)> {
    let mut state: u32 = (seed ^ (seed >> 32)) as u32;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        state
    };
    (0..n_samples)
        .map(|_| {
            let i = (next() as usize) % m;
            let j = (next() as usize) % n;
            (i, j)
        })
        .collect()
}

/// Sampled-cell f64 reference: compute the f64 dot product for each
/// cell in `cells`. Used for the D5 large-shape (2048³, 4096³) tests
/// where a dense f64 matmul would be prohibitive (~68B FMAs at 4096³).
/// Returns one f64 per cell, indexed the same as `cells`.
pub fn sampled_cell_f64_reference(
    a: &[bf16],
    b: &[bf16],
    _m: usize,
    n: usize,
    k: usize,
    cells: &[(usize, usize)],
) -> Vec<f64> {
    cells
        .iter()
        .map(|&(i, j)| {
            let mut sum = 0.0f64;
            for p in 0..k {
                let av = a[i * k + p].to_f32() as f64;
                let bv = b[p * n + j].to_f32() as f64;
                sum += av * bv;
            }
            sum
        })
        .collect()
}

/// Sprint 9.1 D5 standard-tolerance assertion at sampled cells:
/// `rel_err < 1e-2 || abs_err < 1e-3` against an f64 reference computed
/// only at the sampled cells. Used at large shapes (2048³, 4096³) where
/// a dense reference is prohibitive.
pub fn assert_bf16_close_d5_sampled(
    got: &[f32],
    cells: &[(usize, usize)],
    expected: &[f64],
    _m: usize,
    n: usize,
    label: &str,
) {
    const REL_BOUND: f64 = 1e-2;
    const ABS_BOUND: f64 = 1e-3;

    let mut worst: Option<(usize, usize, f64, f64, f64, f64)> = None;
    for (idx, &(i, j)) in cells.iter().enumerate() {
        let g = got[i * n + j] as f64;
        let e = expected[idx];
        let abs_err = (g - e).abs();
        let rel_err = if e.abs() > 0.0 {
            abs_err / e.abs()
        } else {
            abs_err
        };
        if abs_err < ABS_BOUND || rel_err < REL_BOUND {
            continue;
        }
        if worst
            .map(|(_, _, _, _, _, wr)| rel_err > wr)
            .unwrap_or(true)
        {
            worst = Some((i, j, e, g, abs_err, rel_err));
        }
    }

    if let Some((wi, wj, e, g, abs_err, rel_err)) = worst {
        panic!(
            "{label} (sampled, {} cells) FAILED Sprint 9.1 D5 tolerance:\n\
             \n\
             bounds:                rel_err < {REL_BOUND:e} || abs_err < {ABS_BOUND:e}\n\
             worst index (i, j)     = ({wi}, {wj})\n\
             f64 reference value    = {e}\n\
             GPU value (f32 → f64)  = {g}\n\
             abs_err                = {abs_err:e}\n\
             rel_err                = {rel_err:e}\n",
            cells.len(),
        );
    }
}

/// Sprint 9.1 D5 standard-tolerance assertion: `rel_err < 1e-2 ||
/// abs_err < 1e-3` against an f64 reference, computed in f64 and only
/// downcast when reporting the failure. Used for small/medium/large
/// magnitude classes. The near-denorm class uses a stricter rel-only
/// bound + a nonzero-output assertion in
/// [`assert_bf16_close_d5_near_denorm`].
///
/// `got` is the GPU output (f32 from the bf16 mma's f32 accumulator);
/// `expected` is the f64 reference. The element-wise check is f64
/// throughout, with `worst_*` reported in f64 for diagnostic fidelity.
pub fn assert_bf16_close_d5(got: &[f32], expected: &[f64], m: usize, n: usize, label: &str) {
    const REL_BOUND: f64 = 1e-2;
    const ABS_BOUND: f64 = 1e-3;

    let mut worst_idx: Option<(usize, usize)> = None;
    let mut worst_rel = 0.0f64;
    let mut worst_abs = 0.0f64;
    let mut worst_got = 0.0f64;
    let mut worst_expected = 0.0f64;

    for i in 0..m {
        for j in 0..n {
            let idx = i * n + j;
            let g = got[idx] as f64;
            let e = expected[idx];
            let abs_err = (g - e).abs();
            let rel_err = if e.abs() > 0.0 {
                abs_err / e.abs()
            } else {
                abs_err
            };
            if abs_err < ABS_BOUND || rel_err < REL_BOUND {
                continue;
            }
            // Track the worst failing element.
            if rel_err > worst_rel {
                worst_idx = Some((i, j));
                worst_rel = rel_err;
                worst_abs = abs_err;
                worst_got = g;
                worst_expected = e;
            }
        }
    }

    if let Some((wi, wj)) = worst_idx {
        panic!(
            "{label} ({m}×_ × _×{n}) FAILED Sprint 9.1 D5 tolerance:\n\
             \n\
             bounds:                rel_err < {REL_BOUND:e} || abs_err < {ABS_BOUND:e}\n\
             worst index (i, j)     = ({wi}, {wj})\n\
             f64 reference value    = {worst_expected}\n\
             GPU value (f32 → f64)  = {worst_got}\n\
             abs_err                = {worst_abs:e}\n\
             rel_err                = {worst_rel:e}\n",
        );
    }
}

/// Near-denorm tolerance assertion for the bf16 correctness suite:
/// `rel_err < 1e-1` **only** (no absolute-error fallback), plus a
/// "non-zero output" assertion that catches the "kernel returns zero
/// on small inputs" bug class. The looser relative bound reflects
/// bf16's genuinely worse mantissa precision at near-denorm
/// magnitudes; the nonzero gate is the bug-class catch that a
/// permissive `abs_err < 1e-3` would silently let through (since
/// expected outputs are `~K × 1e-36`, far below any reasonable
/// absolute tolerance).
pub fn assert_bf16_close_d5_near_denorm(
    got: &[f32],
    expected: &[f64],
    m: usize,
    n: usize,
    label: &str,
) {
    const REL_BOUND: f64 = 1e-1;

    // (1) Nonzero-output assertion: at least one output cell must be
    //     non-trivially non-zero. An all-zeros (or near-zeros) result
    //     is the "kernel returns zero on small inputs" bug class the
    //     near-denorm test is specifically targeting.
    let max_abs_got = got.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
    let max_abs_expected = expected.iter().fold(0.0f64, |acc, &v| acc.max(v.abs()));
    let nonzero_floor = (max_abs_expected * 0.01).max(f64::MIN_POSITIVE);
    assert!(
        (max_abs_got as f64) > nonzero_floor,
        "{label}: near-denorm output is all-zero or far-below-expected — \
         max_abs(got)={max_abs_got:e}, max_abs(expected)={max_abs_expected:e}, \
         floor={nonzero_floor:e}. Likely the kernel underflowed at small inputs \
         (the bug class this assertion is here to catch)."
    );

    // (2) Strict relative-only tolerance on cells where the f64 reference
    //     itself is meaningful (above the nonzero floor). Cells where the
    //     reference is at-or-near zero are skipped — relative error is
    //     undefined and any abs check would be vacuous at these magnitudes.
    let mut worst: Option<(usize, usize, f64, f64, f64)> = None;
    for i in 0..m {
        for j in 0..n {
            let idx = i * n + j;
            let g = got[idx] as f64;
            let e = expected[idx];
            if e.abs() < nonzero_floor {
                continue;
            }
            let rel_err = (g - e).abs() / e.abs();
            if rel_err < REL_BOUND {
                continue;
            }
            if worst.map(|(_, _, _, _, wr)| rel_err > wr).unwrap_or(true) {
                worst = Some((i, j, e, g, rel_err));
            }
        }
    }

    if let Some((wi, wj, e, g, rel_err)) = worst {
        panic!(
            "{label} ({m}×_ × _×{n}) FAILED Sprint 9.1 D5 near-denorm tolerance:\n\
             \n\
             bound:                 rel_err < {REL_BOUND:e} (no abs fallback at this magnitude)\n\
             worst index (i, j)     = ({wi}, {wj})\n\
             f64 reference value    = {e:e}\n\
             GPU value (f32 → f64)  = {g:e}\n\
             rel_err                = {rel_err:e}\n",
        );
    }
}

// ============================================================================
// Sprint 6.6 — TC attention helpers
// ============================================================================
//
// Extracted alongside the matmul helpers above. The CPU reference uses
// the same mixed-precision promotion contract (f16 inputs → f32 accumulate)
// plus a row-wise softmax in f32 to match the TC kernel's intermediate
// shape, then a second f32-accumulate matmul into f32 output. Reference
// does NOT do a mid-pipeline f32→f16 cvt on probs — that's a precision
// concession the GPU kernel makes because mma.sync requires f16 inputs.
// The assertion tolerance below is widened to account for the resulting
// drift.

/// Scaled dot-product attention reference: `out = softmax(Q·Kᵀ / √d_k) · V`.
///
/// Optionally applies a causal mask (`scores[i,j] = -inf for j > i`) before
/// softmax, matching the scalar `attention_causal` semantics (mask cells
/// contribute 0 to softmax, so row sums stay normalized over the allowed
/// columns).
///
/// Inputs are `half::f16`, promoted to f32 for all arithmetic. Output is
/// f32 (matches the TC kernel's fp32 accumulator contract).
#[allow(clippy::too_many_arguments)]
pub fn cpu_attention_f16xf16_f32(
    q: &[f16],
    k: &[f16],
    v: &[f16],
    seq_q: usize,
    seq_k: usize,
    d_k: usize,
    d_v: usize,
    causal: bool,
) -> Vec<f32> {
    let inv_sqrt_dk = 1.0f32 / (d_k as f32).sqrt();

    // scores[i, j] = Σ_d q[i,d] * k[j,d] * inv_sqrt_dk
    let mut scores = vec![0.0f32; seq_q * seq_k];
    for i in 0..seq_q {
        for j in 0..seq_k {
            let mut dot = 0.0f32;
            for d in 0..d_k {
                dot += q[i * d_k + d].to_f32() * k[j * d_k + d].to_f32();
            }
            scores[i * seq_k + j] = dot * inv_sqrt_dk;
        }
    }

    if causal {
        for i in 0..seq_q {
            for j in 0..seq_k {
                if j > i {
                    scores[i * seq_k + j] = f32::NEG_INFINITY;
                }
            }
        }
    }

    // Row-wise softmax (max-subtract for numerical stability).
    let mut probs = vec![0.0f32; seq_q * seq_k];
    for i in 0..seq_q {
        let row = &scores[i * seq_k..(i + 1) * seq_k];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for j in 0..seq_k {
            let e = if row[j] == f32::NEG_INFINITY {
                0.0
            } else {
                (row[j] - max).exp()
            };
            probs[i * seq_k + j] = e;
            sum += e;
        }
        // Sum is >0 unless the entire row is masked (all -inf); that's
        // a valid degenerate case (causal row before position 0) but
        // doesn't occur for the causal patterns we test (row 0 always
        // has at least the diagonal entry alive).
        if sum > 0.0 {
            for j in 0..seq_k {
                probs[i * seq_k + j] /= sum;
            }
        }
    }

    // out[i, d] = Σ_j probs[i, j] * v[j, d]
    let mut out = vec![0.0f32; seq_q * d_v];
    for i in 0..seq_q {
        for d in 0..d_v {
            let mut sum = 0.0f32;
            for j in 0..seq_k {
                sum += probs[i * seq_k + j] * v[j * d_v + d].to_f32();
            }
            out[i * d_v + d] = sum;
        }
    }
    out
}

/// Attention-tolerance comparator. Absolute OR relative — the `OR` is
/// **load-bearing**, not defensive: near-zero outputs (masked-out
/// positions, attention weights far from the dominant key) have
/// exploding `rel_err` on any noise, and failing on that is a false
/// positive. If tests flake on specific shapes, tighten `abs_err`
/// first; only adjust `rel_err` if the offending shape has large
/// outputs. Do not remove the OR.
///
/// Tolerance is slightly looser than scalar attention's `1e-3 / 1e-2`
/// to account for the f32→f16 cvt on probs (correct per D8 but adds
/// up to ~1 ULP of f16 per prob element) plus compound error through
/// two matmuls + one nonlinear softmax.
pub fn assert_close_attention(
    got: &[f32],
    expected: &[f32],
    seq_q: usize,
    d_v: usize,
    label: &str,
) {
    let abs_tol = 5e-3f32;
    let rel_tol = 2e-2f32;
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for idx in 0..seq_q * d_v {
        let g = got[idx];
        let e = expected[idx];
        let abs_err = (g - e).abs();
        let rel_err = if e.abs() > 1e-6 {
            abs_err / e.abs()
        } else {
            abs_err
        };
        if abs_err > max_abs {
            max_abs = abs_err;
        }
        if rel_err > max_rel {
            max_rel = rel_err;
        }
        assert!(
            abs_err < abs_tol || rel_err < rel_tol,
            "{label}: error at idx {idx}: got {g}, expected {e}, abs={abs_err:.2e}, rel={rel_err:.2e}"
        );
    }
    eprintln!(
        "{label} ({seq_q}×{d_v}): max_abs={max_abs:.2e}, max_rel={max_rel:.2e}, \
         abs_tol={abs_tol:.0e}, rel_tol={rel_tol:.0e}"
    );
}
