//! GPU round-trip correctness tests for `matmul_int4`.
//!
//! Tests exercise the full DEQUANT-F16 chain: signed-INT4 packed
//! weights × f16 activations × f16 group scales → f32 accumulator.
//! CPU reference reproduces the exact `s32 → f32 → f16` cvt chain
//! the kernel uses for bit-comparable correctness.
//!
//! All tests are `#[ignore]` — run via
//! `cargo test -p kaio-ops --test matmul_int4_e2e -- --ignored`
//! on a machine with an Ampere+ GPU.
//!
//! # Sign-extend canary tests (R1 mitigation — GPU layer)
//!
//! [`matmul_int4_sign_extend_canary_negative_eights`] and
//! [`matmul_int4_sign_extend_canary_mixed_positions`] are the
//! GPU-level leg of the triple-layer sign-extend canary. They
//! assert that all-negative and position-sensitive nibble patterns
//! round-trip correctly on hardware. If `shr.s32` collapsed to
//! `shr.u32` anywhere along the pipeline, these tests produce
//! wildly wrong outputs.

use half::f16;
use kaio::prelude::*;
use kaio_ops::matmul_int4;

const GROUP_SIZE: usize = 128;

// ============================================================================
// CPU packer + reference
// ============================================================================

/// Pack a logical `[K, N]` signed-INT4 weight matrix into the KAIO
/// `[K/8, N]` col-major `u32` layout (8 nibbles per u32, K-contiguous).
///
/// Each weight must be in `[-8, +7]`. Values outside this range are
/// masked to 4 bits — caller's responsibility to clamp upstream.
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

/// CPU reference for `matmul_int4`: unpack nibble → sign-extend → mul
/// by f16 group scale → accumulate in f32. Reproduces the exact GPU
/// arithmetic chain for bit-comparable correctness. Matches PTX
/// semantics: `cvt.rn.f32.s32 + cvt.rn.f16.f32 + mul.f16 + mma.sync`.
fn cpu_reference_matmul_int4(
    a: &[f16],
    w_packed: &[u32],
    scales: &[f16],
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
) -> Vec<f32> {
    let k_words = k / 8;
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                // Unpack + sign-extend INT4.
                let word_idx = (kk / 8) + j * k_words;
                let nibble_idx = kk % 8;
                let nibble_bits = (w_packed[word_idx] >> (4 * nibble_idx)) & 0xF;
                // Sign-extend from 4 bits via (x << 28) >> 28 arith shift.
                let signed_i32 = ((nibble_bits << 28) as i32) >> 28;
                // Reproduce PTX cvt chain: s32 → f32 → f16 → mul by f16 scale.
                let f32_val = signed_i32 as f32;
                let f16_raw = f16::from_f32(f32_val);
                let scale = scales[(kk / group_size) * n + j];
                let f16_scaled = f16::from_f32(f16_raw.to_f32() * scale.to_f32());
                // mma.sync f16 * f16 → f32: each multiply is already f16,
                // accumulation is f32 (wide).
                let a_val = a[i * k + kk].to_f32();
                acc += a_val * f16_scaled.to_f32();
            }
            out[i * n + j] = acc;
        }
    }
    out
}

// ============================================================================
// Round-trip helpers
// ============================================================================

fn round_trip(
    m: u32,
    n: u32,
    k: u32,
    w: &[i8],
    a: &[f16],
    scales: &[f16],
) -> std::result::Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
    assert_eq!(w.len(), (k as usize) * (n as usize), "W must be [K, N]");
    assert_eq!(a.len(), (m as usize) * (k as usize), "A must be [M, K]");
    assert_eq!(
        scales.len(),
        (k as usize / GROUP_SIZE) * (n as usize),
        "scales must be [K/128, N]"
    );

    let device = KaioDevice::new(0)?;
    let info = device.info()?;
    let (major, minor) = info.compute_capability;
    assert!(
        major >= 8,
        "matmul_int4 requires SM 8.0+; got sm_{major}{minor}"
    );

    let packed = pack_s4_weights(w, k as usize, n as usize);

    let a_gpu = device.alloc_from(a)?;
    let b_gpu = device.alloc_from(&packed)?;
    let s_gpu = device.alloc_from(scales)?;
    let mut c_gpu = device.alloc_zeros::<f32>((m * n) as usize)?;

    matmul_int4(&device, &a_gpu, &b_gpu, &s_gpu, &mut c_gpu, m, n, k, 128)?;
    let got: Vec<f32> = c_gpu.to_host(&device)?;

    let want = cpu_reference_matmul_int4(
        a, &packed, scales, m as usize, n as usize, k as usize, GROUP_SIZE,
    );
    Ok((got, want))
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, label: &str) {
    assert_eq!(got.len(), want.len(), "length mismatch for {label}");
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let abs = (g - w).abs();
        let rel = abs / w.abs().max(1.0);
        if abs > max_abs {
            max_abs = abs;
        }
        if rel > max_rel {
            max_rel = rel;
        }
        assert!(
            rel < tol,
            "{label}: at [{i}] rel={rel:.2e} > {tol:.2e} (got={g}, want={w})"
        );
    }
    eprintln!("{label}: max_abs={max_abs:.3e}, max_rel={max_rel:.3e}");
}

fn ones_activations(m: u32, k: u32) -> Vec<f16> {
    vec![f16::from_f32(1.0); (m * k) as usize]
}

fn deterministic_f16_activations(m: u32, k: u32, seed: u64) -> Vec<f16> {
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Map to [-1, 1) f32.
        let bits_top = (state >> 32) as u32;
        let f = (bits_top as f32 / u32::MAX as f32) * 2.0 - 1.0;
        f16::from_f32(f)
    };
    (0..(m * k) as usize).map(|_| next()).collect()
}

fn deterministic_s4_weights(k: u32, n: u32, seed: u64) -> Vec<i8> {
    let mut state = seed;
    let mut next = || -> i8 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Map to [-8, 7]
        ((state >> 60) as i8).wrapping_sub(8).clamp(-8, 7)
    };
    (0..(k * n) as usize).map(|_| next()).collect()
}

fn constant_scales(num_groups: usize, n: usize, value: f32) -> Vec<f16> {
    vec![f16::from_f32(value); num_groups * n]
}

// ============================================================================
// Smallest-shape sanity
// ============================================================================

/// Smallest viable shape: one block tile × one K-tile × one group.
/// Weights all +1, scale 1.0, activations = deterministic random.
/// Output should be `sum(a[i, :])` per row.
#[test]
#[ignore]
fn matmul_int4_smallest_16_8_128_all_ones_weights() {
    let m = 16u32;
    let n = 8u32;
    let k = 128u32;
    let w = vec![1i8; (k * n) as usize];
    let a = ones_activations(m, k);
    let scales = constant_scales((k as usize) / GROUP_SIZE, n as usize, 1.0);
    let (got, want) = round_trip(m, n, k, &w, &a, &scales).expect("round_trip");
    // Each output cell should be approximately k (128) × 1 × 1 = 128.
    for v in &got {
        assert!((v - 128.0).abs() < 1.0, "expected ≈128, got {v}");
    }
    assert_close(&got, &want, 1e-3, "smallest_16_8_128");
}

// ============================================================================
// Sign-extend GPU canaries (R1 mitigation — hardware layer)
// ============================================================================

/// All weights are -8 (packed `0x88888888`). Scale = 1.0. Activations = 1.0.
/// Output should be `K × (-8) × 1 × 1 = -K * 8` per output cell.
/// If sign-extend is collapsed to zero-extend, the weights dequant as +8
/// instead of -8 and output is +K*8 — loud failure.
#[test]
#[ignore]
fn matmul_int4_sign_extend_canary_negative_eights() {
    let m = 16u32;
    let n = 8u32;
    let k = 128u32;
    let w = vec![-8i8; (k * n) as usize];
    let a = ones_activations(m, k);
    let scales = constant_scales((k as usize) / GROUP_SIZE, n as usize, 1.0);
    let (got, want) = round_trip(m, n, k, &w, &a, &scales).expect("round_trip");
    let expected = -(k as f32) * 8.0; // -1024
    for v in &got {
        assert!(
            (v - expected).abs() < 2.0,
            "sign-extend canary: expected {expected}, got {v} (if close to +1024, sign bit was zero-extended)"
        );
    }
    assert_close(&got, &want, 1e-3, "sign_extend_negative_eights");
}

/// Mixed-position nibble canary (round 3 fold). One column has
/// weights {+7, -8, +7, -8, ...} alternating along K; others are 0.
/// Catches position-mapping bugs the homogeneous canary misses.
#[test]
#[ignore]
fn matmul_int4_sign_extend_canary_mixed_positions() {
    let m = 16u32;
    let n = 8u32;
    let k = 128u32;
    // Weights [K, N]: col 0 alternates +7/-8, other cols zero.
    let mut w = vec![0i8; (k * n) as usize];
    for kk in 0..k as usize {
        w[kk * n as usize] = if kk % 2 == 0 { 7 } else { -8 };
    }
    let a = ones_activations(m, k);
    let scales = constant_scales((k as usize) / GROUP_SIZE, n as usize, 1.0);
    let (got, want) = round_trip(m, n, k, &w, &a, &scales).expect("round_trip");
    // Col 0 per-row: K/2 × 7 + K/2 × (-8) = K/2 × (7 - 8) = -K/2 = -64.
    // Other cols: 0.
    for row in 0..m as usize {
        for col in 0..n as usize {
            let v = got[row * n as usize + col];
            let expected = if col == 0 { -64.0 } else { 0.0 };
            assert!(
                (v - expected).abs() < 2.0,
                "mixed-position canary: at ({row},{col}) expected {expected}, got {v}"
            );
        }
    }
    assert_close(&got, &want, 1e-3, "sign_extend_mixed_positions");
}

// ============================================================================
// Group-boundary correctness (R3 mitigation)
// ============================================================================

/// K=256 covers 2 groups. Per-group scales differ (group 0 = 1.0,
/// group 1 = 2.0), per-column same. Weights all +1, activations all 1.
/// Each row sum = 128 × 1 × 1 + 128 × 1 × 2 = 128 + 256 = 384.
#[test]
#[ignore]
fn matmul_int4_multi_group_k256() {
    let m = 32u32;
    let n = 32u32;
    let k = 256u32;
    let w = vec![1i8; (k * n) as usize];
    let a = ones_activations(m, k);
    // scales: [2, 32], group 0 = 1.0, group 1 = 2.0
    let mut scales = Vec::with_capacity(2 * n as usize);
    for _ in 0..n as usize {
        scales.push(f16::from_f32(1.0));
    }
    for _ in 0..n as usize {
        scales.push(f16::from_f32(2.0));
    }
    let (got, want) = round_trip(m, n, k, &w, &a, &scales).expect("round_trip");
    for v in &got {
        assert!((v - 384.0).abs() < 2.0, "expected 384, got {v}");
    }
    assert_close(&got, &want, 1e-3, "multi_group_k256");
}

// ============================================================================
// Larger realistic shapes
// ============================================================================

#[test]
#[ignore]
fn matmul_int4_64_64_128() {
    let (got, want) = round_trip(
        64,
        64,
        128,
        &deterministic_s4_weights(128, 64, 0xCAFE_BABE_DEAD_BEEF),
        &deterministic_f16_activations(64, 128, 0x1234_5678_9ABC_DEF0),
        &constant_scales(1, 64, 0.1),
    )
    .expect("round_trip");
    assert_close(&got, &want, 1e-2, "64_64_128");
}

#[test]
#[ignore]
fn matmul_int4_128_128_256() {
    let (got, want) = round_trip(
        128,
        128,
        256,
        &deterministic_s4_weights(256, 128, 0xF00D_FACE_D00D_1337),
        &deterministic_f16_activations(128, 256, 0xA5A5_5A5A_D00D_FEED),
        &constant_scales(2, 128, 0.05),
    )
    .expect("round_trip");
    assert_close(&got, &want, 1e-2, "128_128_256");
}

#[test]
#[ignore]
fn matmul_int4_256_256_512() {
    let (got, want) = round_trip(
        256,
        256,
        512,
        &deterministic_s4_weights(512, 256, 0x0123_4567_89AB_CDEF),
        &deterministic_f16_activations(256, 512, 0xFEDC_BA98_7654_3210),
        &constant_scales(4, 256, 0.02),
    )
    .expect("round_trip");
    assert_close(&got, &want, 1e-2, "256_256_512");
}

// ============================================================================
// Edge-tile M / N
// ============================================================================

#[test]
#[ignore]
fn matmul_int4_edge_m_17() {
    let m = 17u32;
    let n = 8u32;
    let k = 128u32;
    let (got, want) = round_trip(
        m,
        n,
        k,
        &deterministic_s4_weights(k, n, 0xDEAD_0001_BEEF_0002),
        &deterministic_f16_activations(m, k, 0xFACE_0003_BEEF_0004),
        &constant_scales(1, n as usize, 0.1),
    )
    .expect("round_trip");
    assert_close(&got, &want, 1e-2, "edge_m_17");
}

#[test]
#[ignore]
fn matmul_int4_edge_n_13() {
    let m = 16u32;
    let n = 13u32;
    let k = 128u32;
    let (got, want) = round_trip(
        m,
        n,
        k,
        &deterministic_s4_weights(k, n, 0x5555_5555_AAAA_AAAA),
        &deterministic_f16_activations(m, k, 0x1111_2222_3333_4444),
        &constant_scales(1, n as usize, 0.1),
    )
    .expect("round_trip");
    assert_close(&got, &want, 1e-2, "edge_n_13");
}

// ============================================================================
// Validation-error tests
// ============================================================================

#[test]
#[ignore]
fn matmul_int4_rejects_k_not_multiple_of_128() {
    let device = KaioDevice::new(0).expect("no GPU");
    let m = 16u32;
    let n = 8u32;
    let k = 127u32; // NOT a multiple of 128
    let a = device.alloc_zeros::<f16>((m * k) as usize).unwrap();
    let b = device.alloc_zeros::<u32>(((k / 8) * n) as usize).unwrap();
    let s = device.alloc_zeros::<f16>(n as usize).unwrap();
    let mut c = device.alloc_zeros::<f32>((m * n) as usize).unwrap();
    let err = matmul_int4(&device, &a, &b, &s, &mut c, m, n, k, 128).unwrap_err();
    match err {
        KaioError::InvalidConfig(msg) => assert!(
            msg.contains("K must be a multiple of group_size"),
            "got: {msg}"
        ),
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}

#[test]
#[ignore]
fn matmul_int4_rejects_non_128_group_size() {
    let device = KaioDevice::new(0).expect("no GPU");
    let m = 16u32;
    let n = 8u32;
    let k = 128u32;
    let a = device.alloc_zeros::<f16>((m * k) as usize).unwrap();
    let b = device.alloc_zeros::<u32>(((k / 8) * n) as usize).unwrap();
    let s = device.alloc_zeros::<f16>(n as usize).unwrap();
    let mut c = device.alloc_zeros::<f32>((m * n) as usize).unwrap();
    let err = matmul_int4(&device, &a, &b, &s, &mut c, m, n, k, 64).unwrap_err();
    match err {
        KaioError::InvalidConfig(msg) => {
            assert!(msg.contains("group_size must be 128"), "got: {msg}")
        }
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}

#[test]
#[ignore]
fn matmul_int4_rejects_zero_dim() {
    let device = KaioDevice::new(0).expect("no GPU");
    let a = device.alloc_zeros::<f16>(1).unwrap();
    let b = device.alloc_zeros::<u32>(1).unwrap();
    let s = device.alloc_zeros::<f16>(1).unwrap();
    let mut c = device.alloc_zeros::<f32>(1).unwrap();
    let err = matmul_int4(&device, &a, &b, &s, &mut c, 0, 8, 128, 128).unwrap_err();
    assert!(matches!(err, KaioError::InvalidConfig(_)));
}

// ============================================================================
// Host-only CPU reference sanity (runs without --ignored)
// ============================================================================

/// Verify the CPU packer + reference produce sensible values for a
/// trivial case. Runs in every test run (no `#[ignore]`) as a
/// regression guard on the packer / reference pair.
#[test]
fn cpu_packer_plus_reference_smoke() {
    let m = 2usize;
    let n = 2usize;
    let k = 8usize;
    // W = [[1, -1], [1, -1], ..., [1, -1]] — 8 rows, 2 cols.
    let w: Vec<i8> = (0..k * n)
        .map(|i| if i % 2 == 0 { 1 } else { -1 })
        .collect();
    let packed = pack_s4_weights(&w, k, n);
    // Reconstruct: col 0 should be 8 nibbles of +1 packed → 0x11111111.
    assert_eq!(packed[0], 0x11111111, "col-0 pack mismatch");
    // Col 1 should be 8 nibbles of -1 (= 0xF) → 0xFFFFFFFF.
    assert_eq!(packed[1], 0xFFFFFFFF, "col-1 pack mismatch");

    // Reference: A = identity-like [2, 8] with row 0 = [1, 1, ..., 1],
    // row 1 = [0, 0, ..., 0]. One group covers all 8 K elements.
    // Need group_size = 8 for this toy test.
    let a: Vec<f16> = (0..m * k)
        .map(|i| f16::from_f32(if i < k { 1.0 } else { 0.0 }))
        .collect();
    let scales = vec![f16::from_f32(1.0); n]; // 1 group × 2 cols
    let out = cpu_reference_matmul_int4(&a, &packed, &scales, m, n, k, k);
    // Row 0 col 0: sum(1 * 1 * 1) × 8 = 8
    // Row 0 col 1: sum(1 * 1 * -1) × 8 = -8
    // Row 1: all zeros
    assert_eq!(out, vec![8.0, -8.0, 0.0, 0.0]);
}
