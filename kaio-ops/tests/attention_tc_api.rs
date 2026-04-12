//! Sprint 6.6 integration tests for `attention_tc` (fused TC attention).
//!
//! Gate A tests exercise `attention_tc_gate_a` — the matmul1-only dev
//! kernel that computes `scores = Q · Kᵀ · inv_sqrt_dk`. This isolates
//! the first mma.sync contract in the fused kernel before Gate B adds
//! softmax + cvt and Gate C adds the second matmul.

mod common;

use common::{
    assert_close_attention, cpu_attention_f16xf16_f32, cpu_attention_probs_f32,
    cpu_attention_scores_f16xf16_f32, patterned_f16_data,
};
use half::f16;
use kaio::prelude::*;
use kaio_ops::{attention_tc, attention_tc_causal, attention_tc_gate_a, attention_tc_gate_b};

// ---------------------------------------------------------------------
// Gate A correctness — matmul1 only
// ---------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn check_gate_a(seq_q: u32, seq_k: u32, d_k: u32, label: &str) {
    let device = KaioDevice::new(0).expect("GPU required");
    let q_host = patterned_f16_data((seq_q * d_k) as usize);
    let k_host = patterned_f16_data((seq_k * d_k) as usize);

    let q = device.alloc_from(&q_host).unwrap();
    let k = device.alloc_from(&k_host).unwrap();
    let mut scores = device
        .alloc_zeros::<f32>((seq_q * seq_k) as usize)
        .unwrap();

    attention_tc_gate_a(&device, &q, &k, &mut scores, seq_q, seq_k, d_k)
        .expect("attention_tc_gate_a launch failed");

    let got = scores.to_host(&device).unwrap();
    let expected = cpu_attention_scores_f16xf16_f32(
        &q_host,
        &k_host,
        seq_q as usize,
        seq_k as usize,
        d_k as usize,
        false, // Gate A: no causal mask
    );

    // The scores matrix is the output of a f16×f16→f32 matmul shape
    // M=seq_q, N=seq_k, K=d_k, scaled by inv_sqrt_dk. Use K-scaled
    // tolerance against the K dimension (d_k), inflated by the scale
    // factor's 1.0 multiplier (inv_sqrt_dk ≤ 1/√16 = 0.25).
    //
    // For simplicity, hand-roll a per-element check here with a
    // slightly relaxed tolerance vs matmul_tc (inv_sqrt_dk introduces
    // a small amount of extra f32 noise).
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for idx in 0..(seq_q * seq_k) as usize {
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
        let abs_tol = (d_k as f32) * 2f32.powi(-10) * 0.5; // K-scaled, inputs ≤ 0.5
        assert!(
            abs_err < abs_tol || rel_err < 5e-3,
            "{label}: scores mismatch at idx {idx}: got {g}, expected {e}, \
             abs={abs_err:.2e}, rel={rel_err:.2e}, tol={abs_tol:.2e}"
        );
    }
    eprintln!(
        "{label} (seq_q={seq_q}, seq_k={seq_k}, d_k={d_k}): \
         max_abs={max_abs:.2e}, max_rel={max_rel:.2e}"
    );
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn gate_a_scores_16x16x16() {
    check_gate_a(16, 16, 16, "gate_a_scores_16x16x16");
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn gate_a_scores_32x32x32() {
    check_gate_a(32, 32, 32, "gate_a_scores_32x32x32");
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn gate_a_scores_64x128x64() {
    check_gate_a(64, 128, 64, "gate_a_scores_64x128x64");
}

// ---------------------------------------------------------------------
// Gate B correctness — matmul1 + softmax + cvt bridge → f16 probs
// ---------------------------------------------------------------------

fn check_gate_b(seq_q: u32, seq_k: u32, d_k: u32, label: &str) {
    let device = KaioDevice::new(0).expect("GPU required");
    let q_host = patterned_f16_data((seq_q * d_k) as usize);
    let k_host = patterned_f16_data((seq_k * d_k) as usize);

    let q = device.alloc_from(&q_host).unwrap();
    let k = device.alloc_from(&k_host).unwrap();
    let mut probs = device
        .alloc_zeros::<f16>((seq_q * seq_k) as usize)
        .unwrap();

    attention_tc_gate_b(&device, &q, &k, &mut probs, seq_q, seq_k, d_k)
        .expect("attention_tc_gate_b launch failed");

    // GPU output is f16. Promote to f32 for comparison.
    let got_f16 = probs.to_host(&device).unwrap();
    let got: Vec<f32> = got_f16.iter().map(|h| h.to_f32()).collect();

    // CPU reference: scaled scores → row-wise softmax (f32 probs).
    let cpu_scores = cpu_attention_scores_f16xf16_f32(
        &q_host,
        &k_host,
        seq_q as usize,
        seq_k as usize,
        d_k as usize,
        false,
    );
    let expected = cpu_attention_probs_f32(&cpu_scores, seq_q as usize, seq_k as usize);

    // Probs ∈ [0, 1]. f16 representable resolution near 0 is ~6e-5
    // (smallest positive subnormal). Near 1 the resolution is ~1/2048
    // ≈ 5e-4. Use abs_err < 2e-3 OR rel_err < 1e-2 — slightly looser
    // than the full attention tolerance because we're sensitive to the
    // raw cvt rounding in isolation (Gate C's full-path tolerance
    // absorbs it more naturally).
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for idx in 0..(seq_q * seq_k) as usize {
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
            abs_err < 2e-3 || rel_err < 1e-2,
            "{label}: probs mismatch at idx {idx}: got {g}, expected {e}, \
             abs={abs_err:.2e}, rel={rel_err:.2e}"
        );
    }
    eprintln!(
        "{label} (seq_q={seq_q}, seq_k={seq_k}, d_k={d_k}): \
         max_abs={max_abs:.2e}, max_rel={max_rel:.2e}"
    );
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn gate_b_probs_16x16x16() {
    check_gate_b(16, 16, 16, "gate_b_probs_16x16x16");
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn gate_b_probs_32x32x32() {
    check_gate_b(32, 32, 32, "gate_b_probs_32x32x32");
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn gate_b_probs_64x128x64() {
    check_gate_b(64, 128, 64, "gate_b_probs_64x128x64");
}

// ---------------------------------------------------------------------
// Gate C correctness — full fused non-causal attention_tc
// ---------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn check_attention_tc(seq_q: u32, seq_k: u32, d_k: u32, d_v: u32, label: &str) {
    let device = KaioDevice::new(0).expect("GPU required");
    let q_host = patterned_f16_data((seq_q * d_k) as usize);
    let k_host = patterned_f16_data((seq_k * d_k) as usize);
    let v_host = patterned_f16_data((seq_k * d_v) as usize);

    let q = device.alloc_from(&q_host).unwrap();
    let k = device.alloc_from(&k_host).unwrap();
    let v = device.alloc_from(&v_host).unwrap();
    let mut out = device
        .alloc_zeros::<f32>((seq_q * d_v) as usize)
        .unwrap();

    attention_tc(&device, &q, &k, &v, &mut out, seq_q, seq_k, d_k, d_v)
        .expect("attention_tc launch failed");

    let got = out.to_host(&device).unwrap();
    let expected = cpu_attention_f16xf16_f32(
        &q_host,
        &k_host,
        &v_host,
        seq_q as usize,
        seq_k as usize,
        d_k as usize,
        d_v as usize,
        false,
    );

    assert_close_attention(&got, &expected, seq_q as usize, d_v as usize, label);
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn attention_tc_smallest_16x16x16x8() {
    check_attention_tc(16, 16, 16, 8, "attention_tc_smallest_16x16x16x8");
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn attention_tc_32x32x32x32() {
    check_attention_tc(32, 32, 32, 32, "attention_tc_32x32x32x32");
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn attention_tc_64x128x64x64() {
    check_attention_tc(64, 128, 64, 64, "attention_tc_64x128x64x64");
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn attention_tc_hard_cap_64x384x64x64() {
    check_attention_tc(64, 384, 64, 64, "attention_tc_hard_cap_64x384x64x64");
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn attention_tc_small_shape_16x32x16x8() {
    check_attention_tc(16, 32, 16, 8, "attention_tc_small_shape_16x32x16x8");
}

// ---------------------------------------------------------------------
// 6.6b causal correctness — full fused causal attention_tc_causal
// ---------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn check_attention_tc_causal(seq_q: u32, seq_k: u32, d_k: u32, d_v: u32, label: &str) {
    let device = KaioDevice::new(0).expect("GPU required");
    let q_host = patterned_f16_data((seq_q * d_k) as usize);
    let k_host = patterned_f16_data((seq_k * d_k) as usize);
    let v_host = patterned_f16_data((seq_k * d_v) as usize);

    let q = device.alloc_from(&q_host).unwrap();
    let k = device.alloc_from(&k_host).unwrap();
    let v = device.alloc_from(&v_host).unwrap();
    let mut out = device
        .alloc_zeros::<f32>((seq_q * d_v) as usize)
        .unwrap();

    attention_tc_causal(&device, &q, &k, &v, &mut out, seq_q, seq_k, d_k, d_v)
        .expect("attention_tc_causal launch failed");

    let got = out.to_host(&device).unwrap();
    let expected = cpu_attention_f16xf16_f32(
        &q_host,
        &k_host,
        &v_host,
        seq_q as usize,
        seq_k as usize,
        d_k as usize,
        d_v as usize,
        true, // causal
    );

    assert_close_attention(&got, &expected, seq_q as usize, d_v as usize, label);
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn attention_tc_causal_smallest_16x16x16x8() {
    check_attention_tc_causal(16, 16, 16, 8, "attention_tc_causal_smallest_16x16x16x8");
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn attention_tc_causal_32x32x32x32() {
    check_attention_tc_causal(32, 32, 32, 32, "attention_tc_causal_32x32x32x32");
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn attention_tc_causal_64x128x64x64() {
    check_attention_tc_causal(64, 128, 64, 64, "attention_tc_causal_64x128x64x64");
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn attention_tc_causal_hard_cap_64x384x64x64() {
    check_attention_tc_causal(
        64,
        384,
        64,
        64,
        "attention_tc_causal_hard_cap_64x384x64x64",
    );
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn attention_tc_causal_small_shape_16x32x16x8() {
    check_attention_tc_causal(
        16,
        32,
        16,
        8,
        "attention_tc_causal_small_shape_16x32x16x8",
    );
}

/// Canary test for the causal-mask global-coordinate math. Uses
/// V = I (identity-like: each V row has a distinct pattern that
/// makes the output equal to the attended-to V rows). Verifies
/// row 0 of the output equals row 0 of V (row 0 can only attend
/// to position 0 under causal masking). Reproduces the same trap
/// the scalar `attention_causal_row0_self_only` test gates against —
/// an off-by-one in the mask predicate (`c > r` vs `c >= r`) would
/// either let row 0 attend to col 1 or block attention to col 0.
#[test]
#[ignore = "requires NVIDIA GPU"]
fn attention_tc_causal_row0_self_only() {
    let seq_q: u32 = 16;
    let seq_k: u32 = 16;
    let d_k: u32 = 16;
    let d_v: u32 = 8;

    let device = KaioDevice::new(0).expect("GPU required");

    // Q with large row-0 query that should dominate — but causal
    // forces row 0 to attend only to row 0 of K anyway. Pattern
    // doesn't matter for this test's invariant; use patterned_f16_data.
    let q_host = patterned_f16_data((seq_q * d_k) as usize);
    let k_host = patterned_f16_data((seq_k * d_k) as usize);

    // V: each row has a distinct magnitude: row i has all cols = i+1
    // (scaled to fit f16 comfortably). Output row 0 should equal V row
    // 0 exactly (within softmax tolerance for a 1-element-alive row
    // which has probs = [1.0, 0, 0, ...]).
    let mut v_vals = vec![0.0f32; (seq_k * d_v) as usize];
    for i in 0..seq_k as usize {
        for j in 0..d_v as usize {
            v_vals[i * d_v as usize + j] = (i as f32 + 1.0) * 0.1;
        }
    }
    let v_host: Vec<f16> = v_vals.iter().map(|x| f16::from_f32(*x)).collect();

    let q = device.alloc_from(&q_host).unwrap();
    let k = device.alloc_from(&k_host).unwrap();
    let v = device.alloc_from(&v_host).unwrap();
    let mut out = device
        .alloc_zeros::<f32>((seq_q * d_v) as usize)
        .unwrap();

    attention_tc_causal(&device, &q, &k, &v, &mut out, seq_q, seq_k, d_k, d_v)
        .expect("attention_tc_causal launch failed");

    let got = out.to_host(&device).unwrap();

    // Row 0 of output should equal V row 0 (0.1 in every col).
    for j in 0..d_v as usize {
        let g = got[j];
        let e = 0.1f32;
        let err = (g - e).abs();
        assert!(
            err < 5e-3,
            "row0_self_only: col {j} got {g}, expected {e} (diff {err:.2e}); \
             this typically indicates an off-by-one in the causal mask \
             global-coord predicate (`c > r` vs `c >= r`)"
        );
    }
    eprintln!("attention_tc_causal_row0_self_only: row 0 matches V row 0 (diagonal-attended).");
}

// Keep full attention CPU reference and tolerance helper in-scope so
// they are tested for compilation during Gate A. Gates B/C will use
// them directly. This silences the `unused` warnings on the helpers
// until they are reached.
#[test]
fn common_helpers_compile() {
    let q: Vec<f16> = patterned_f16_data(16 * 16);
    let k: Vec<f16> = patterned_f16_data(16 * 16);
    let v: Vec<f16> = patterned_f16_data(16 * 16);
    let out = cpu_attention_f16xf16_f32(&q, &k, &v, 16, 16, 16, 16, false);
    assert_eq!(out.len(), 16 * 16);
    // Re-run same inputs, compare to itself for non-trivial equality.
    let out2 = cpu_attention_f16xf16_f32(&q, &k, &v, 16, 16, 16, 16, false);
    assert_close_attention(&out, &out2, 16, 16, "helpers_self_consistency");
}
