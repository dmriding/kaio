//! GPU round-trip correctness tests for `qkv_project_int8` (W8A16).
//!
//! Tests exercise the full INT8 dequant chain: f16 activations × i8
//! weights × f32 scalar per-projection scale → f16 outputs. CPU reference
//! reproduces the exact `cvt.rn.f16.s8 + mma.f32 + mul.f32 + cvt.rn.f16.f32`
//! arithmetic chain the kernel uses.
//!
//! All tests are `#[ignore]` — run via
//! `cargo test -p kaio-ops --test qkv_project_int8_e2e -- --ignored`
//! on a machine with an Ampere+ GPU.
//!
//! # Tolerance posture (round 2)
//!
//! The fused kernel uses the same arithmetic primitives as a
//! "3 × INT8-W8A16 dequant + cast f32→f16" reference in the same
//! precision envelope. Outputs are **expected to be bit-close and often
//! bit-exact**. The hard pass gate is `max_rel_err < 1e-3` (loose
//! enough to accept f16-rounding-step difference between
//! `f32 acc → mul → cvt.rn.f16.f32` and the reference's identical
//! chain). Observed bit-exactness is a bonus, not the only acceptable
//! outcome.
//!
//! # Q/K/V differentiation canary
//!
//! [`qkv_project_int8_qkv_differentiation_canary`] sets
//! `W_Q = +1`, `W_K = +2`, `W_V = +3`, `X = ones`, all scales = 1.0, and
//! verifies the three outputs differ by exactly the expected ratios.
//! Catches the class of tri-output bugs where a buggy alloc_grid pattern
//! reuses fragment-C registers across projections, silently merging
//! accumulators (the tri-output sibling of 7.2's sign-extend canary).

use half::f16;
use kaio::prelude::*;
use kaio_ops::qkv_project_int8;

// ============================================================================
// CPU reference
// ============================================================================

/// CPU reference for one INT8 W8A16 projection.
///
/// Reproduces the exact PTX arithmetic chain:
///   - per-weight `cvt.rn.f16.s8` (the s8 → f16 conversion happens inside
///     the dequant chain; the mma sees f16 × f16),
///   - `mma.sync.m16n8k16.f16.f16.f32` accumulator (f32 sum of f16 muls),
///   - per-projection scalar `mul.f32` scale fold,
///   - `cvt.rn.f16.f32` store-out narrow.
fn cpu_reference_qkv_int8(
    x: &[f16],
    w_i8: &[i8],
    scale: f32,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f16> {
    let mut out = vec![f16::from_f32(0.0); m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                // s8 → f16 (cvt.rn.f16.s8 is exact for s8 magnitudes).
                let w_f16 = f16::from_f32(w_i8[kk * n + j] as f32);
                let x_f16 = x[i * k + kk];
                // f16 mul, f32 accumulate (mma.sync semantics).
                acc += x_f16.to_f32() * w_f16.to_f32();
            }
            // Post-accumulation scalar scale + narrow.
            out[i * n + j] = f16::from_f32(acc * scale);
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
    scale_q: f32,
    scale_k: f32,
    scale_v: f32,
) -> std::result::Result<QkvOutTuple, Box<dyn std::error::Error>> {
    let device = KaioDevice::new(0)?;
    let info = device.info()?;
    let (major, minor) = info.compute_capability;
    assert!(
        major >= 8,
        "qkv_project_int8 requires SM 8.0+; got sm_{major}{minor}"
    );

    let x_gpu = device.alloc_from(x)?;
    let w_q_gpu = device.alloc_from(w_q)?;
    let w_k_gpu = device.alloc_from(w_k)?;
    let w_v_gpu = device.alloc_from(w_v)?;
    let mut q_out = device.alloc_zeros::<f16>((m * n) as usize)?;
    let mut k_out = device.alloc_zeros::<f16>((m * n) as usize)?;
    let mut v_out = device.alloc_zeros::<f16>((m * n) as usize)?;

    qkv_project_int8(
        &device, &x_gpu, &w_q_gpu, &w_k_gpu, &w_v_gpu, scale_q, scale_k, scale_v, &mut q_out,
        &mut k_out, &mut v_out, m, n, k,
    )?;
    device.stream().synchronize()?;

    let q_got = q_out.to_host(&device)?;
    let k_got = k_out.to_host(&device)?;
    let v_got = v_out.to_host(&device)?;
    let q_want = cpu_reference_qkv_int8(x, w_q, scale_q, m as usize, n as usize, k as usize);
    let k_want = cpu_reference_qkv_int8(x, w_k, scale_k, m as usize, n as usize, k as usize);
    let v_want = cpu_reference_qkv_int8(x, w_v, scale_v, m as usize, n as usize, k as usize);
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

fn deterministic_f16_activations(n_elem: usize, seed: u64) -> Vec<f16> {
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits_top = (state >> 32) as u32;
        // Map to [-0.5, 0.5) f32 — keep magnitudes small to avoid f32→f16
        // overflow at large K accumulations.
        let f = (bits_top as f32 / u32::MAX as f32) - 0.5;
        f16::from_f32(f)
    };
    (0..n_elem).map(|_| next()).collect()
}

fn deterministic_i8_weights(n_elem: usize, seed: u64) -> Vec<i8> {
    let mut state = seed;
    let mut next = || -> i8 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Map to [-32, 31] — small magnitude reduces accumulator overflow risk.
        ((state >> 56) as i8) / 4
    };
    (0..n_elem).map(|_| next()).collect()
}

fn run_correctness_case(m: u32, n: u32, k: u32, seed: u64, tol: f32, label: &str) {
    let x = deterministic_f16_activations((m * k) as usize, seed);
    let w_q = deterministic_i8_weights((k * n) as usize, seed.wrapping_add(1));
    let w_k = deterministic_i8_weights((k * n) as usize, seed.wrapping_add(2));
    let w_v = deterministic_i8_weights((k * n) as usize, seed.wrapping_add(3));
    let scale_q = 1.0 / (k as f32);
    let scale_k = 1.0 / (k as f32);
    let scale_v = 1.0 / (k as f32);
    let (q_got, k_got, v_got, q_want, k_want, v_want) =
        round_trip(m, n, k, &x, &w_q, &w_k, &w_v, scale_q, scale_k, scale_v)
            .unwrap_or_else(|e| panic!("round_trip {label}: {e}"));
    assert_close(&q_got, &q_want, tol, &format!("{label} Q"));
    assert_close(&k_got, &k_want, tol, &format!("{label} K"));
    assert_close(&v_got, &v_want, tol, &format!("{label} V"));
}

// ============================================================================
// Correctness tests across canonical LLM-ish shapes
// ============================================================================

#[test]
#[ignore]
fn qkv_project_int8_smallest_64_16_16() {
    run_correctness_case(64, 16, 16, 0xDEAD_BEEF, 1e-3, "smallest_64_16_16");
}

#[test]
#[ignore]
fn qkv_project_int8_one_block_tile_64_16_128() {
    run_correctness_case(64, 16, 128, 0xCAFE_BABE, 1e-3, "one_block_64_16_128");
}

#[test]
#[ignore]
fn qkv_project_int8_multiblock_n_64_64_64() {
    // BN_BLOCK = 16 (Rollback #1) → N=64 spans 4 N-blocks.
    run_correctness_case(64, 64, 64, 0xFEED_FACE, 1e-3, "multi_n_64_64_64");
}

#[test]
#[ignore]
fn qkv_project_int8_multiblock_m_128_64_64() {
    // BM_BLOCK = 64 → M=128 spans 2 M-blocks; combined with N=64 = 8 blocks.
    run_correctness_case(128, 64, 64, 0xBEEF_F00D, 1e-3, "multi_m_128_64_64");
}

#[test]
#[ignore]
fn qkv_project_int8_larger_k_64_32_512() {
    // Stress the K-loop iteration count (32 K-tiles).
    run_correctness_case(64, 32, 512, 0xDEAD_F00D, 1e-3, "larger_k_64_32_512");
}

#[test]
#[ignore]
fn qkv_project_int8_medium_256_128_256() {
    // Realistic per-block decomposition stress.
    run_correctness_case(256, 128, 256, 0xABCD_EF01, 1e-3, "medium_256_128_256");
}

// ============================================================================
// Q/K/V differentiation canary (round 2)
// ============================================================================

/// W_Q = +1, W_K = +2, W_V = +3, X = ones, scales = 1.0/K (so output ≈ 1, 2, 3
/// per cell). Catches the class of tri-output bugs where fragment-C grids
/// alias across projections — a buggy alloc would silently merge the three
/// accumulators and outputs would all match instead of differing by ratio.
#[test]
#[ignore]
fn qkv_project_int8_qkv_differentiation_canary() {
    let m = 64u32;
    let n = 32u32;
    let k = 128u32;
    let x = vec![f16::from_f32(1.0); (m * k) as usize];
    let w_q = vec![1i8; (k * n) as usize];
    let w_k = vec![2i8; (k * n) as usize];
    let w_v = vec![3i8; (k * n) as usize];
    let scale = 1.0 / (k as f32);
    let (q_got, k_got, v_got, _q_want, _k_want, _v_want) =
        round_trip(m, n, k, &x, &w_q, &w_k, &w_v, scale, scale, scale).expect("round_trip");

    // Expected: Q ≈ 1.0, K ≈ 2.0, V ≈ 3.0 per cell.
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
    // And the outputs must NOT be equal across projections.
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
// Slot-mapping canary — ping-pong correctness under the 2-W-slot layout
// ============================================================================
//
// The 2-W-slot ping-pong design selects one of two shared-memory slots
// per projection per K-tile via runtime `k_tile & 1` arithmetic. A
// silent mis-wiring — e.g. `cur_w_base` and `next_w_base` accidentally
// swapped, or the mma reads the wrong slot for a given projection —
// would produce wrong outputs deterministically. Random-data tests
// (the correctness cases above) can mask this: if the wrong weights
// happen to produce plausible accumulator values, the reference
// comparison still falls within tolerance.
//
// The slot-mapping canary makes the wiring observable by seeding W
// with per-projection-distinguishable constants (W_Q=1, W_K=2, W_V=3,
// X=1, scale=1.0) so the expected output per element is exactly
// `{K, 2*K, 3*K}` for `{Q, K, V}`. A wrong slot read flips projections
// and the output mismatches by an integer ratio — impossible to
// explain by rounding.
//
// The 2-K-tile variant exercises the `steady → final` transition; the
// 3-K-tile variant exercises `steady → steady → final`, catching
// wiring bugs that live specifically at the back-edge between two
// steady-state iterations.

fn run_slot_mapping_canary(m: u32, n: u32, k: u32, label: &str) {
    let x = vec![f16::from_f32(1.0); (m * k) as usize];
    let w_q = vec![1i8; (k * n) as usize];
    let w_k = vec![2i8; (k * n) as usize];
    let w_v = vec![3i8; (k * n) as usize];
    // scale = 1.0 (identity) — expected output = exactly {K, 2K, 3K}
    // per element. Unscaled integer products in f16 representable range:
    // K=48 → V=144 which is well under f16's max normal (65504).
    let (q_got, k_got, v_got, _qw, _kw, _vw) =
        round_trip(m, n, k, &x, &w_q, &w_k, &w_v, 1.0, 1.0, 1.0).expect("round_trip");
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
            // f16 has ~3 decimal digits of precision. At K=48, V=144
            // rounds to exactly 144 (representable). Tolerance 0.5 is
            // more than loose enough and tight enough to catch a slot
            // swap (which would flip magnitudes by at least 1×K).
            assert!(
                err < 0.5,
                "{label} {proj_label}: at [{i}] expected {expected}, got {v} (err={err})"
            );
        }
    }
    // Sanity check: the three projections must still be distinguishable.
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

/// 2 K-tiles at K_TILE_SHARED=16 → covers steady-state iteration
/// followed by final iteration. N=16 spans one BN_BLOCK = single
/// N-block; M=64 spans one BM_BLOCK.
#[test]
#[ignore]
fn qkv_project_int8_slot_mapping_canary_2_k_tiles() {
    run_slot_mapping_canary(64, 16, 32, "slot_mapping_2ktile_64_16_32");
}

/// 3 K-tiles at K_TILE_SHARED=16 → steady → steady → final, exercises
/// the back-edge transition between two steady-state iterations that
/// the 2-K-tile variant doesn't hit. If the ping-pong slot indexing
/// breaks specifically when going from k_tile=0 → k_tile=1 (both
/// steady), the 2-K-tile test passes but this one fails.
#[test]
#[ignore]
fn qkv_project_int8_slot_mapping_canary_3_k_tiles() {
    run_slot_mapping_canary(64, 16, 48, "slot_mapping_3ktile_64_16_48");
}

// ============================================================================
// Determinism stress — cross-launch bit-exactness under the 2-W-slot layout
// ============================================================================
//
// The 2-W-slot design introduces an overlap window where cooperative
// `ld.global → st.shared` stores to one slot run concurrently with
// mma reads from the other slot. If a barrier is misplaced or missing,
// the overlap creates a race window whose resolution depends on warp
// scheduling. Warp scheduling can vary across launches, so such races
// manifest as **cross-launch nondeterminism** — same input, same
// kernel, different bit-exact output between runs.
//
// The slot-mapping canary above catches *deterministic* mis-wiring.
// These determinism tests catch *nondeterministic* barrier races. Both
// classes exist; neither subsumes the other.
//
// Two shapes cover different regimes on the race-window manifold:
// short/many-K-tiles exercises a lot of steady-state transitions per
// launch; prefill-regime (M=2048, K=4096) is the shape that motivated
// the S+½P rework and is where schedulers have the most room to drift.

const DETERMINISM_REPS_DEFAULT: usize = 100;

fn run_determinism_stress(m: u32, n: u32, k: u32, reps: usize, label: &str) {
    let x = deterministic_f16_activations((m * k) as usize, 0xDEAD_BEEF);
    let w_q = deterministic_i8_weights((k * n) as usize, 0x1111_1111);
    let w_k = deterministic_i8_weights((k * n) as usize, 0x2222_2222);
    let w_v = deterministic_i8_weights((k * n) as usize, 0x3333_3333);
    let scale = 1.0 / (k as f32);

    let device = KaioDevice::new(0).expect("KaioDevice::new");
    let (major, _) = device.info().unwrap().compute_capability;
    assert!(major >= 8, "qkv_project_int8 requires SM 8.0+");

    let x_gpu = device.alloc_from(&x).unwrap();
    let w_q_gpu = device.alloc_from(&w_q).unwrap();
    let w_k_gpu = device.alloc_from(&w_k).unwrap();
    let w_v_gpu = device.alloc_from(&w_v).unwrap();

    // Run 1 establishes the reference. Bit-exactness is asserted from
    // run 1 onward — no warmup runs are discarded. Warmup-driven
    // variance (cold-cache first-run differences from steady state) is
    // treated as a real failure, because barrier races we're hunting
    // don't have warmup semantics: if run 1 differs from runs 2..N,
    // that's either a race OR a cache-subsystem determinism issue,
    // both worth surfacing rather than papering over.
    let mut reference: Option<(Vec<f16>, Vec<f16>, Vec<f16>)> = None;
    for rep in 0..reps {
        let mut q_out = device.alloc_zeros::<f16>((m * n) as usize).unwrap();
        let mut k_out = device.alloc_zeros::<f16>((m * n) as usize).unwrap();
        let mut v_out = device.alloc_zeros::<f16>((m * n) as usize).unwrap();
        qkv_project_int8(
            &device, &x_gpu, &w_q_gpu, &w_k_gpu, &w_v_gpu, scale, scale, scale, &mut q_out,
            &mut k_out, &mut v_out, m, n, k,
        )
        .expect("qkv_project_int8");
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

/// Short shape, many K-tiles per launch. Stresses the steady-state
/// ping-pong transitions at moderate K and gives the scheduler many
/// micro-opportunities to drift across the 100 launches. Bumping the
/// rep count to ~1000 is the sprint-close / commit-boundary gate;
/// 100 is the default for reasonable iteration time.
#[test]
#[ignore]
fn qkv_project_int8_determinism_stress_short() {
    run_determinism_stress(
        256,
        512,
        1024,
        DETERMINISM_REPS_DEFAULT,
        "determinism_short_256_512_1024",
    );
}

/// Prefill-regime shape. The S+½P rework was motivated by recovering
/// `prefill_m2048` from the 0.85× Design-S ratio; this is the regime
/// where schedulers have the most room to drift between long mma
/// epochs and overlapped cooperative loads. 100 reps is ~30-60s of
/// GPU time on sm_89 — acceptable for an `#[ignore]`d sprint-gate.
#[test]
#[ignore]
fn qkv_project_int8_determinism_stress_prefill() {
    run_determinism_stress(
        2048,
        512,
        4096,
        DETERMINISM_REPS_DEFAULT,
        "determinism_prefill_2048_512_4096",
    );
}
