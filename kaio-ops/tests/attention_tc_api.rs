//! Sprint 6.6 integration tests for `attention_tc` (fused TC attention).
//!
//! Gate A tests exercise `attention_tc_gate_a` — the matmul1-only dev
//! kernel that computes `scores = Q · Kᵀ · inv_sqrt_dk`. This isolates
//! the first mma.sync contract in the fused kernel before Gate B adds
//! softmax + cvt and Gate C adds the second matmul.

mod common;

use common::{
    assert_close_attention, cpu_attention_f16xf16_f32, cpu_attention_scores_f16xf16_f32,
    patterned_f16_data,
};
use half::f16;
use kaio::prelude::*;
use kaio_ops::attention_tc_gate_a;

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
