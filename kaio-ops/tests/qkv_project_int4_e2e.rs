//! GPU round-trip correctness tests for `qkv_project_int4` (W4A16).
//!
//! Tests exercise the full INT4 dequant chain: packed signed-INT4 weights
//! × f16 activations × f16 group scales → f16 outputs. CPU reference
//! reproduces the exact `shr.s32 + cvt.rn.f32.s32 + cvt.rn.f16.f32 +
//! mul.f16 + mma.sync.f16.f16.f32 + cvt.rn.f16.f32` arithmetic chain the
//! kernel uses.
//!
//! All tests are `#[ignore]` — run via
//! `cargo test -p kaio-ops --test qkv_project_int4_e2e -- --ignored`
//! on a machine with an Ampere+ GPU.
//!
//! # Group-boundary coverage
//!
//! [`qkv_project_int4_one_group_64_16_128`],
//! [`qkv_project_int4_two_groups_64_16_256`], and
//! [`qkv_project_int4_eight_groups_64_16_1024`] exercise the
//! group-scale reload cadence (`K_TILE_GROUP_RATIO = 8` K-tiles per
//! group). The 8-group case forces seven group transitions inside the
//! K-loop — catches any off-by-one in `group_idx = k_tile /
//! K_TILE_GROUP_RATIO`.
//!
//! # Sign-extend canary
//!
//! [`qkv_project_int4_sign_extend_canary_negative_eights`] — all weights
//! packed as `0x8` (INT4 = -8). Output should be `K × (-8) × scale` per
//! cell with X = 1. If `shr.s32` collapsed to `shr.u32`, the weights
//! dequant as +8 and output sign flips — loud failure.
//!
//! # Q/K/V differentiation canary
//!
//! [`qkv_project_int4_qkv_differentiation_canary`] verifies the three
//! outputs differ by expected ratios — the tri-output sibling of the
//! sign-extend canary, catches fragment-C register aliasing across
//! projections.

use half::f16;
use kaio::prelude::*;
use kaio_ops::qkv_project_int4;

const GROUP_SIZE: usize = 128;

// ============================================================================
// CPU packer + reference
// ============================================================================

fn pack_s4_weights(w: &[i8], k: usize, n: usize) -> Vec<u32> {
    assert!(k.is_multiple_of(8), "K must be a multiple of 8");
    let k_words = k / 8;
    let mut out = vec![0u32; k_words * n];
    for col in 0..n {
        for word_idx in 0..k_words {
            let mut word = 0u32;
            for nibble_idx in 0..8 {
                let k_pos = word_idx * 8 + nibble_idx;
                let val = w[k_pos * n + col];
                let nibble = (val as u32) & 0xF;
                word |= nibble << (4 * nibble_idx);
            }
            out[word_idx + col * k_words] = word;
        }
    }
    out
}

/// CPU reference for one INT4 W4A16 projection. Reproduces the exact GPU
/// arithmetic chain: unpack → sign-extend via `(x << 28) >> 28` arith
/// shift → s32 → f32 → f16 → mul by f16 group scale → f16 × f16 mma
/// accumulating to f32 → cvt.rn.f16.f32 output narrow.
fn cpu_reference_qkv_int4(
    x: &[f16],
    w_packed: &[u32],
    scales: &[f16],
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
) -> Vec<f16> {
    let k_words = k / 8;
    let mut out = vec![f16::from_f32(0.0); m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                let word_idx = (kk / 8) + j * k_words;
                let nibble_idx = kk % 8;
                let nibble_bits = (w_packed[word_idx] >> (4 * nibble_idx)) & 0xF;
                let signed_i32 = ((nibble_bits << 28) as i32) >> 28;
                let f32_val = signed_i32 as f32;
                let f16_raw = f16::from_f32(f32_val);
                let scale = scales[(kk / group_size) * n + j];
                let f16_scaled = f16::from_f32(f16_raw.to_f32() * scale.to_f32());
                let x_f16 = x[i * k + kk];
                acc += x_f16.to_f32() * f16_scaled.to_f32();
            }
            out[i * n + j] = f16::from_f32(acc);
        }
    }
    out
}

// ============================================================================
// Round-trip helpers
// ============================================================================

type QkvOutTuple = (Vec<f16>, Vec<f16>, Vec<f16>, Vec<f16>, Vec<f16>, Vec<f16>);

#[allow(clippy::too_many_arguments)]
fn round_trip(
    m: u32,
    n: u32,
    k: u32,
    x: &[f16],
    w_q: &[i8],
    w_k: &[i8],
    w_v: &[i8],
    scales_q: &[f16],
    scales_k: &[f16],
    scales_v: &[f16],
) -> std::result::Result<QkvOutTuple, Box<dyn std::error::Error>> {
    let device = KaioDevice::new(0)?;
    let info = device.info()?;
    let (major, minor) = info.compute_capability;
    assert!(
        major >= 8,
        "qkv_project_int4 requires SM 8.0+; got sm_{major}{minor}"
    );

    let packed_q = pack_s4_weights(w_q, k as usize, n as usize);
    let packed_k = pack_s4_weights(w_k, k as usize, n as usize);
    let packed_v = pack_s4_weights(w_v, k as usize, n as usize);

    let x_gpu = device.alloc_from(x)?;
    let w_q_gpu = device.alloc_from(&packed_q)?;
    let w_k_gpu = device.alloc_from(&packed_k)?;
    let w_v_gpu = device.alloc_from(&packed_v)?;
    let s_q_gpu = device.alloc_from(scales_q)?;
    let s_k_gpu = device.alloc_from(scales_k)?;
    let s_v_gpu = device.alloc_from(scales_v)?;
    let mut q_out = device.alloc_zeros::<f16>((m * n) as usize)?;
    let mut k_out = device.alloc_zeros::<f16>((m * n) as usize)?;
    let mut v_out = device.alloc_zeros::<f16>((m * n) as usize)?;

    qkv_project_int4(
        &device,
        &x_gpu,
        &w_q_gpu,
        &w_k_gpu,
        &w_v_gpu,
        &s_q_gpu,
        &s_k_gpu,
        &s_v_gpu,
        &mut q_out,
        &mut k_out,
        &mut v_out,
        m,
        n,
        k,
        GROUP_SIZE as u32,
    )?;
    device.stream().synchronize()?;

    let q_got = q_out.to_host(&device)?;
    let k_got = k_out.to_host(&device)?;
    let v_got = v_out.to_host(&device)?;
    let q_want = cpu_reference_qkv_int4(
        x, &packed_q, scales_q, m as usize, n as usize, k as usize, GROUP_SIZE,
    );
    let k_want = cpu_reference_qkv_int4(
        x, &packed_k, scales_k, m as usize, n as usize, k as usize, GROUP_SIZE,
    );
    let v_want = cpu_reference_qkv_int4(
        x, &packed_v, scales_v, m as usize, n as usize, k as usize, GROUP_SIZE,
    );
    Ok((q_got, k_got, v_got, q_want, k_want, v_want))
}

fn assert_close(got: &[f16], want: &[f16], tol: f32, label: &str) {
    assert_eq!(got.len(), want.len(), "length mismatch for {label}");
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let g32 = g.to_f32();
        let w32 = w.to_f32();
        let abs = (g32 - w32).abs();
        let rel = abs / w32.abs().max(1.0);
        if abs > max_abs {
            max_abs = abs;
        }
        if rel > max_rel {
            max_rel = rel;
        }
        assert!(
            rel < tol,
            "{label}: at [{i}] rel={rel:.2e} > {tol:.2e} (got={g32}, want={w32})"
        );
    }
    eprintln!("{label}: max_abs={max_abs:.3e}, max_rel={max_rel:.3e}");
}

fn deterministic_f16(n_elem: usize, seed: u64) -> Vec<f16> {
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits_top = (state >> 32) as u32;
        let f = (bits_top as f32 / u32::MAX as f32) - 0.5;
        f16::from_f32(f)
    };
    (0..n_elem).map(|_| next()).collect()
}

fn deterministic_s4(n_elem: usize, seed: u64) -> Vec<i8> {
    let mut state = seed;
    let mut next = || -> i8 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 60) as i8).wrapping_sub(8).clamp(-8, 7)
    };
    (0..n_elem).map(|_| next()).collect()
}

fn small_scales(num_groups: usize, n: usize, seed: u64) -> Vec<f16> {
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits_top = (state >> 32) as u32;
        // Scales in [0.0625, 0.125) — small enough to keep accumulated f32
        // well inside f16's representable range even at large K.
        let f = (bits_top as f32 / u32::MAX as f32) * 0.0625 + 0.0625;
        f16::from_f32(f)
    };
    (0..num_groups * n).map(|_| next()).collect()
}

fn run_correctness_case(m: u32, n: u32, k: u32, seed: u64, tol: f32, label: &str) {
    let x = deterministic_f16((m * k) as usize, seed);
    let w_q = deterministic_s4((k * n) as usize, seed.wrapping_add(1));
    let w_k = deterministic_s4((k * n) as usize, seed.wrapping_add(2));
    let w_v = deterministic_s4((k * n) as usize, seed.wrapping_add(3));
    let num_groups = (k as usize) / GROUP_SIZE;
    let s_q = small_scales(num_groups, n as usize, seed.wrapping_add(4));
    let s_k = small_scales(num_groups, n as usize, seed.wrapping_add(5));
    let s_v = small_scales(num_groups, n as usize, seed.wrapping_add(6));
    let (q_got, k_got, v_got, q_want, k_want, v_want) =
        round_trip(m, n, k, &x, &w_q, &w_k, &w_v, &s_q, &s_k, &s_v)
            .unwrap_or_else(|e| panic!("round_trip {label}: {e}"));
    assert_close(&q_got, &q_want, tol, &format!("{label} Q"));
    assert_close(&k_got, &k_want, tol, &format!("{label} K"));
    assert_close(&v_got, &v_want, tol, &format!("{label} V"));
}

// ============================================================================
// Correctness tests — group-boundary coverage
// ============================================================================

/// K = 128 = exactly one group. Smallest shape that still exercises the
/// full dequant + mma + store chain.
#[test]
#[ignore]
fn qkv_project_int4_one_group_64_16_128() {
    run_correctness_case(64, 16, 128, 0xDEAD_BEEF, 1e-3, "one_group_64_16_128");
}

/// K = 256 = 2 groups. Exercises exactly one group transition inside the
/// K-loop.
#[test]
#[ignore]
fn qkv_project_int4_two_groups_64_16_256() {
    run_correctness_case(64, 16, 256, 0xCAFE_BABE, 1e-3, "two_groups_64_16_256");
}

/// K = 1024 = 8 groups. Seven group transitions — catches off-by-one in
/// `group_idx = k_tile / K_TILE_GROUP_RATIO`.
#[test]
#[ignore]
fn qkv_project_int4_eight_groups_64_16_1024() {
    run_correctness_case(64, 16, 1024, 0xFEED_FACE, 1e-3, "eight_groups_64_16_1024");
}

// ============================================================================
// Correctness tests — multi-block shape coverage
// ============================================================================

#[test]
#[ignore]
fn qkv_project_int4_multiblock_n_64_64_128() {
    // BN_BLOCK = 16 → N=64 spans 4 N-blocks.
    run_correctness_case(64, 64, 128, 0xBEEF_F00D, 1e-3, "multi_n_64_64_128");
}

#[test]
#[ignore]
fn qkv_project_int4_multiblock_m_128_32_128() {
    // BM_BLOCK = 64 → M=128 spans 2 M-blocks.
    run_correctness_case(128, 32, 128, 0xDEAD_F00D, 1e-3, "multi_m_128_32_128");
}

#[test]
#[ignore]
fn qkv_project_int4_larger_128_128_256() {
    run_correctness_case(128, 128, 256, 0xABCD_EF01, 1e-3, "larger_128_128_256");
}

// ============================================================================
// Sign-extend canary (GPU layer)
// ============================================================================

/// All weights = -8 (nibble 0x8). Scales = 1.0. X = 1.0.
/// Output per cell should be `K × (-8) × 1 × 1 = -K × 8`.
/// If sign-extend collapsed to zero-extend, -8 → +8 and output sign flips.
#[test]
#[ignore]
fn qkv_project_int4_sign_extend_canary_negative_eights() {
    let m = 64u32;
    let n = 16u32;
    let k = 128u32;
    let x = vec![f16::from_f32(1.0); (m * k) as usize];
    let w_q = vec![-8i8; (k * n) as usize];
    let w_k = vec![-8i8; (k * n) as usize];
    let w_v = vec![-8i8; (k * n) as usize];
    let num_groups = (k as usize) / GROUP_SIZE;
    let s_q = vec![f16::from_f32(1.0); num_groups * n as usize];
    let s_k = vec![f16::from_f32(1.0); num_groups * n as usize];
    let s_v = vec![f16::from_f32(1.0); num_groups * n as usize];
    let (q_got, k_got, v_got, _, _, _) =
        round_trip(m, n, k, &x, &w_q, &w_k, &w_v, &s_q, &s_k, &s_v).expect("round_trip");
    let expected = -(k as f32) * 8.0; // -1024
    for (label, got) in [("Q", &q_got), ("K", &k_got), ("V", &v_got)] {
        for (i, h) in got.iter().enumerate() {
            let v = h.to_f32();
            let err = (v - expected).abs();
            assert!(
                err < 4.0,
                "{label}: at [{i}] expected ≈{expected} (sign-extend), got {v} (err={err})"
            );
        }
        eprintln!("{label} sign-extend canary: all cells ≈{expected}");
    }
}

// ============================================================================
// Q/K/V differentiation canary (Codex round 2)
// ============================================================================

#[test]
#[ignore]
fn qkv_project_int4_qkv_differentiation_canary() {
    let m = 64u32;
    let n = 32u32;
    let k = 128u32;
    let x = vec![f16::from_f32(1.0); (m * k) as usize];
    // INT4 nibble range is [-8, +7] — use 1 / 2 / 3 for clean ratios.
    let w_q = vec![1i8; (k * n) as usize];
    let w_k = vec![2i8; (k * n) as usize];
    let w_v = vec![3i8; (k * n) as usize];
    let num_groups = (k as usize) / GROUP_SIZE;
    // Scales = 1/K so expected outputs = 1 / 2 / 3.
    let inv_k = f16::from_f32(1.0 / k as f32);
    let s_q = vec![inv_k; num_groups * n as usize];
    let s_k = vec![inv_k; num_groups * n as usize];
    let s_v = vec![inv_k; num_groups * n as usize];
    let (q_got, k_got, v_got, _, _, _) =
        round_trip(m, n, k, &x, &w_q, &w_k, &w_v, &s_q, &s_k, &s_v).expect("round_trip");

    for (label, got, expected) in [
        ("Q", &q_got, 1.0f32),
        ("K", &k_got, 2.0),
        ("V", &v_got, 3.0),
    ] {
        for (i, h) in got.iter().enumerate() {
            let v = h.to_f32();
            let err = (v - expected).abs();
            assert!(
                err < 1e-2,
                "{label}: at [{i}] expected ≈{expected}, got {v} (err={err})"
            );
        }
        eprintln!("{label} canary: expected ≈{expected}, all cells within 1e-2");
    }
    assert!(
        (q_got[0].to_f32() - k_got[0].to_f32()).abs() > 0.5,
        "Q and K outputs alias (canary tripped)"
    );
    assert!(
        (k_got[0].to_f32() - v_got[0].to_f32()).abs() > 0.5,
        "K and V outputs alias (canary tripped)"
    );
}

// ============================================================================
// S+½P slot-mapping canary (Sprint 7.3.5)
// ============================================================================
//
// Distinct from the `qkv_project_int4_qkv_differentiation_canary` above:
// scale=1.0 (identity) → expected per-element outputs are exact integers
// `{K, 2K, 3K}` for Q/K/V, crisper failure signal if the ping-pong slot
// wiring mis-reads a projection's slot.
//
// INT4 note: K must be a multiple of `group_size = 128`, so we cannot
// directly mirror the INT8 K=32/K=48 shapes. K=128 (8 K-tiles) and K=256
// (16 K-tiles = 2 groups) are the minimal equivalents: both exercise
// steady → steady back-edge transitions, and K=256 adds coverage of the
// K_TILE_GROUP_RATIO=8 group boundary transition where scale hoist re-
// reads from a new row of the scales tensor.

fn run_slot_mapping_canary_int4(m: u32, n: u32, k: u32, label: &str) {
    let x = vec![f16::from_f32(1.0); (m * k) as usize];
    let w_q = vec![1i8; (k * n) as usize];
    let w_k = vec![2i8; (k * n) as usize];
    let w_v = vec![3i8; (k * n) as usize];
    let num_groups = (k as usize) / GROUP_SIZE;
    // scale = 1.0 so expected outputs are exact integers K/2K/3K.
    let one = f16::from_f32(1.0);
    let s_q = vec![one; num_groups * n as usize];
    let s_k = vec![one; num_groups * n as usize];
    let s_v = vec![one; num_groups * n as usize];
    let (q_got, k_got, v_got, _, _, _) =
        round_trip(m, n, k, &x, &w_q, &w_k, &w_v, &s_q, &s_k, &s_v).expect("round_trip");
    let expected_q = k as f32;
    let expected_k = 2.0 * k as f32;
    let expected_v = 3.0 * k as f32;
    for (proj_label, got, expected) in [
        ("Q", &q_got, expected_q),
        ("K", &k_got, expected_k),
        ("V", &v_got, expected_v),
    ] {
        for (i, h) in got.iter().enumerate() {
            let v = h.to_f32();
            let err = (v - expected).abs();
            // f16 can represent all integers ≤ 2048 exactly; at K=256
            // the max expected (V=768) is well within that range. A
            // slot swap would shift the magnitude by at least 1×K, far
            // exceeding the 0.5 tolerance.
            assert!(
                err < 0.5,
                "{label} {proj_label}: at [{i}] expected {expected}, got {v} (err={err})"
            );
        }
    }
    let q0 = q_got[0].to_f32();
    let k0 = k_got[0].to_f32();
    let v0 = v_got[0].to_f32();
    assert!(
        (q0 - k0).abs() > 0.5 * k as f32,
        "{label}: Q≈{q0} aliases K≈{k0} (slot-mapping tripped)"
    );
    assert!(
        (k0 - v0).abs() > 0.5 * k as f32,
        "{label}: K≈{k0} aliases V≈{v0} (slot-mapping tripped)"
    );
    eprintln!("{label}: Q≈{q0} K≈{k0} V≈{v0} (expected {expected_q} {expected_k} {expected_v})");
}

/// K=128 = 1 group = 8 K-tiles. Smallest valid INT4 shape; exercises the
/// initial → 7× steady → final ping-pong cadence without crossing a group
/// boundary.
#[test]
#[ignore]
fn qkv_project_int4_slot_mapping_canary_1_group() {
    run_slot_mapping_canary_int4(64, 16, 128, "slot_mapping_1g_64_16_128");
}

/// K=256 = 2 groups = 16 K-tiles. Exercises the group transition at
/// k_tile=8 where `g = k_tile / K_TILE_GROUP_RATIO` flips 0→1 and the
/// hoisted scales re-read from scales[1, :]. A bug specifically at the
/// group transition (e.g. wrong `current_group` division, wrong scales
/// row-stride math) is caught here that the 1-group test passes.
#[test]
#[ignore]
fn qkv_project_int4_slot_mapping_canary_2_groups() {
    run_slot_mapping_canary_int4(64, 16, 256, "slot_mapping_2g_64_16_256");
}

// ============================================================================
// Determinism stress — cross-launch bit-exactness under the 2-W-slot layout
// ============================================================================
//
// Sprint 7.3.5 S+½P sibling of the INT8 determinism stress. Same rationale:
// the overlap window between cooperative loads and mma reads on the other
// slot is a barrier-misplacement race surface. Warp scheduling varies
// across launches, so such races manifest as cross-launch nondeterminism
// — same input, same kernel, bit-different output.
//
// INT4 adds a second race surface: the scales register-hoist path. If the
// per-lane `ld.global.f16` is reordered against B1 or a downstream dequant,
// the dequanted fragment B depends on whether the load reached cache
// before the mma. Both shapes assert bit-exactness from run 1 onward (no
// warmup discards).
//
// Two shapes cover different regimes:
//   - Short / group-boundary stress: K=256 = 2 groups × 8 K-tiles each.
//     16 K-tiles per launch; one group transition per launch. Scheduler
//     gets little room to drift but the group-boundary scale re-hoist is
//     exercised once per launch × 100 launches = 100 group transitions.
//   - Prefill regime / long: K=4096 = 32 groups × 8 K-tiles each.
//     256 K-tiles per launch, the shape that motivated the sprint's
//     S+½P rework. Deep K + many overlapping load/mma epochs = maximum
//     scheduler drift room.

const DETERMINISM_REPS_DEFAULT: usize = 100;

fn run_determinism_stress_int4(m: u32, n: u32, k: u32, reps: usize, label: &str) {
    let x = deterministic_f16((m * k) as usize, 0xDEAD_BEEF);
    let w_q = deterministic_s4((k * n) as usize, 0x1111_1111);
    let w_k = deterministic_s4((k * n) as usize, 0x2222_2222);
    let w_v = deterministic_s4((k * n) as usize, 0x3333_3333);
    let num_groups = (k as usize) / GROUP_SIZE;
    let s_q = small_scales(num_groups, n as usize, 0x4444_4444);
    let s_k = small_scales(num_groups, n as usize, 0x5555_5555);
    let s_v = small_scales(num_groups, n as usize, 0x6666_6666);

    let device = KaioDevice::new(0).expect("KaioDevice::new");
    let (major, _) = device.info().unwrap().compute_capability;
    assert!(major >= 8, "qkv_project_int4 requires SM 8.0+");

    let packed_q = pack_s4_weights(&w_q, k as usize, n as usize);
    let packed_k = pack_s4_weights(&w_k, k as usize, n as usize);
    let packed_v = pack_s4_weights(&w_v, k as usize, n as usize);
    let x_gpu = device.alloc_from(&x).unwrap();
    let w_q_gpu = device.alloc_from(&packed_q).unwrap();
    let w_k_gpu = device.alloc_from(&packed_k).unwrap();
    let w_v_gpu = device.alloc_from(&packed_v).unwrap();
    let s_q_gpu = device.alloc_from(&s_q).unwrap();
    let s_k_gpu = device.alloc_from(&s_k).unwrap();
    let s_v_gpu = device.alloc_from(&s_v).unwrap();

    // Run 1 establishes the reference. Bit-exactness is asserted from
    // run 1 onward — no warmup runs are discarded. Warmup-driven
    // variance (cold-cache first-run differences from steady-state) is
    // treated as a real failure: barrier races we're hunting don't have
    // warmup semantics, and if run 1 differs from runs 2..N, that's
    // either a race OR a cache-subsystem determinism issue — both worth
    // surfacing rather than papering over.
    let mut reference: Option<(Vec<f16>, Vec<f16>, Vec<f16>)> = None;
    for rep in 0..reps {
        let mut q_out = device.alloc_zeros::<f16>((m * n) as usize).unwrap();
        let mut k_out = device.alloc_zeros::<f16>((m * n) as usize).unwrap();
        let mut v_out = device.alloc_zeros::<f16>((m * n) as usize).unwrap();
        qkv_project_int4(
            &device,
            &x_gpu,
            &w_q_gpu,
            &w_k_gpu,
            &w_v_gpu,
            &s_q_gpu,
            &s_k_gpu,
            &s_v_gpu,
            &mut q_out,
            &mut k_out,
            &mut v_out,
            m,
            n,
            k,
            GROUP_SIZE as u32,
        )
        .expect("qkv_project_int4");
        device.stream().synchronize().unwrap();
        let q = q_out.to_host(&device).unwrap();
        let k_h = k_out.to_host(&device).unwrap();
        let v = v_out.to_host(&device).unwrap();
        match &reference {
            None => reference = Some((q, k_h, v)),
            Some((qr, kr, vr)) => {
                for (proj_label, got, want) in [("Q", &q, qr), ("K", &k_h, kr), ("V", &v, vr)] {
                    let mismatch = got
                        .iter()
                        .zip(want.iter())
                        .enumerate()
                        .find(|(_, (g, w))| g.to_bits() != w.to_bits());
                    if let Some((i, (g, w))) = mismatch {
                        panic!(
                            "{label} {proj_label} rep={rep} i={i}: bits differ \
                             (got 0x{:04x} want 0x{:04x}, f32 got={} want={})",
                            g.to_bits(),
                            w.to_bits(),
                            g.to_f32(),
                            w.to_f32()
                        );
                    }
                }
            }
        }
    }
    eprintln!("{label}: {reps} runs, bit-exact across all runs");
}

/// Short shape with group-boundary stress. K=256 = 2 groups × 8 K-tiles,
/// crossing one group boundary per launch. Bumping reps to ~1000 is the
/// sprint-close / commit-boundary gate; 100 is the default for reasonable
/// iteration time.
#[test]
#[ignore]
fn qkv_project_int4_determinism_stress_short() {
    run_determinism_stress_int4(
        256,
        512,
        256,
        DETERMINISM_REPS_DEFAULT,
        "determinism_short_256_512_256",
    );
}

/// Prefill-regime shape. 32 groups × 8 K-tiles = 256 K-tiles per launch
/// — the shape that motivated the Sprint 7.3.5 S+½P rework. Schedulers
/// have the most room to drift between long mma epochs and overlapped
/// cooperative loads at this depth. 100 reps is ~2-5 minutes of GPU
/// time on sm_89 — acceptable for an `#[ignore]`d sprint-gate.
#[test]
#[ignore]
fn qkv_project_int4_determinism_stress_prefill() {
    run_determinism_stress_int4(
        2048,
        512,
        4096,
        DETERMINISM_REPS_DEFAULT,
        "determinism_prefill_2048_512_4096",
    );
}
