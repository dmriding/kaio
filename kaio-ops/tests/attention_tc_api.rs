//! Sprint 6.6 integration tests for `attention_tc` (fused TC attention).
//!
//! Covers correctness of the full fused kernel + its causal variant
//! against the CPU f16 attention reference. Five shapes × 2 causal
//! variants = 10 correctness tests plus the `row0_self_only` canary
//! guarding the causal mask's global-coordinate math.

mod common;

use common::{assert_close_attention, cpu_attention_f16xf16_f32, patterned_f16_data};
use half::f16;
use kaio::prelude::*;
use kaio_ops::{attention_tc, attention_tc_causal};

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
    let mut out = device.alloc_zeros::<f32>((seq_q * d_v) as usize).unwrap();

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
    let mut out = device.alloc_zeros::<f32>((seq_q * d_v) as usize).unwrap();

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
    check_attention_tc_causal(64, 384, 64, 64, "attention_tc_causal_hard_cap_64x384x64x64");
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn attention_tc_causal_small_shape_16x32x16x8() {
    check_attention_tc_causal(16, 32, 16, 8, "attention_tc_causal_small_shape_16x32x16x8");
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
    let mut out = device.alloc_zeros::<f32>((seq_q * d_v) as usize).unwrap();

    attention_tc_causal(&device, &q, &k, &v, &mut out, seq_q, seq_k, d_k, d_v)
        .expect("attention_tc_causal launch failed");

    let got = out.to_host(&device).unwrap();

    // Row 0 of output should equal V row 0 (0.1 in every col).
    for (j, &g) in got.iter().enumerate().take(d_v as usize) {
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

/// Host-only self-consistency check on the CPU reference. Ensures
/// `cpu_attention_f16xf16_f32` is deterministic (same inputs → same
/// outputs). A trivial property test but catches the case where a
/// future refactor accidentally introduces non-determinism (e.g.,
/// parallelism reordering the softmax reduction).
#[test]
fn cpu_reference_is_deterministic() {
    let q: Vec<f16> = patterned_f16_data(16 * 16);
    let k: Vec<f16> = patterned_f16_data(16 * 16);
    let v: Vec<f16> = patterned_f16_data(16 * 16);
    let out_a = cpu_attention_f16xf16_f32(&q, &k, &v, 16, 16, 16, 16, false);
    let out_b = cpu_attention_f16xf16_f32(&q, &k, &v, 16, 16, 16, 16, false);
    assert_eq!(out_a.len(), 16 * 16);
    assert_close_attention(&out_a, &out_b, 16, 16, "cpu_reference_is_deterministic");
}
