//! GPU round-trip correctness tests for `matmul_int8`.
//!
//! Small-to-medium sizes are tested bit-exact against a CPU
//! `i8 × i8 → i32` reference scaled to `f32`. Larger sizes use a
//! relative-error tolerance because fp32 accumulation noise grows with
//! K; `matmul_int8` runs accumulation in `s32` (exact) but the
//! post-accumulation scale cast introduces fp32 rounding per output
//! element.
//!
//! All tests are `#[ignore]` — run via
//! `cargo test -p kaio-ops --test matmul_int8_e2e -- --ignored`
//! on a machine with an Ampere+ GPU.

use kaio::prelude::*;
use kaio_ops::matmul_int8;

/// CPU reference: accumulate `i8 × i8 → i32` exactly, then scale to
/// `f32` with a cast-then-multiply identical to the kernel's
/// `cvt.rn.f32.s32 + mul.f32`.
fn cpu_reference(a: &[i8], b: &[i8], scale: f32, m: u32, n: u32, k: u32) -> Vec<f32> {
    let mut out = vec![0.0f32; (m * n) as usize];
    for i in 0..m {
        for j in 0..n {
            let mut acc: i32 = 0;
            for kk in 0..k {
                let av = a[(i * k + kk) as usize] as i32;
                let bv = b[(kk * n + j) as usize] as i32;
                acc += av * bv;
            }
            out[(i * n + j) as usize] = (acc as f32) * scale;
        }
    }
    out
}

fn round_trip(m: u32, n: u32, k: u32, scale: f32, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let device = KaioDevice::new(0).expect("no GPU");
    let info = device.info().expect("device info");
    let (major, minor) = info.compute_capability;
    assert!(
        major >= 8,
        "matmul_int8 requires SM 8.0+; got sm_{major}{minor}"
    );

    let mk = (m * k) as usize;
    let kn = (k * n) as usize;
    let mn = (m * n) as usize;

    // Deterministic pseudo-random i8 fill.
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 56) as i8
    };
    let a: Vec<i8> = (0..mk).map(|_| next()).collect();
    let b: Vec<i8> = (0..kn).map(|_| next()).collect();

    let a_gpu = device.alloc_from(&a).expect("alloc a");
    let b_gpu = device.alloc_from(&b).expect("alloc b");
    let mut c_gpu = device.alloc_zeros::<f32>(mn).expect("alloc c");
    matmul_int8(&device, &a_gpu, &b_gpu, &mut c_gpu, scale, m, n, k).expect("matmul_int8");
    let c_gpu_host = c_gpu.to_host(&device).expect("dtoh");

    let c_ref = cpu_reference(&a, &b, scale, m, n, k);
    (c_gpu_host, c_ref)
}

#[test]
#[ignore]
fn matmul_int8_16_8_32_scale_one() {
    // Smallest shape that fits exactly one mma.sync.m16n8k32 output.
    let (got, want) = round_trip(16, 8, 32, 1.0, 0xA5A5_5A5A_D00D_FEED);
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!((g - w).abs() < 1e-4, "mismatch at {i}: got {g}, want {w}");
    }
}

#[test]
#[ignore]
fn matmul_int8_64_64_32() {
    // One full block tile on the output, one K-tile.
    let (got, want) = round_trip(64, 64, 32, 0.5, 0x1234_5678_9ABC_DEF0);
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!((g - w).abs() < 1e-4, "mismatch at {i}: got {g}, want {w}");
    }
}

#[test]
#[ignore]
fn matmul_int8_64_64_128() {
    // 4 K-tiles — exercises the K-loop bar.sync pattern.
    let (got, want) = round_trip(64, 64, 128, 0.125, 0xCAFE_BABE_DEAD_BEEF);
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!((g - w).abs() < 1e-4, "mismatch at {i}: got {g}, want {w}");
    }
}

#[test]
#[ignore]
fn matmul_int8_128_128_128() {
    // 4 block tiles on output × 4 K-tiles — exercises multi-block +
    // multi-K parallelism.
    let (got, want) = round_trip(128, 128, 128, 0.01, 0x0123_4567_89AB_CDEF);
    let mut max_abs = 0.0f32;
    for (g, w) in got.iter().zip(want.iter()) {
        max_abs = max_abs.max((g - w).abs());
    }
    assert!(max_abs < 1e-3, "max abs err {max_abs} exceeds 1e-3");
}

#[test]
#[ignore]
fn matmul_int8_256_256_256() {
    let (got, want) = round_trip(256, 256, 256, 0.001, 0xF00D_FACE_D00D_1337);
    let mut max_rel = 0.0f32;
    for (g, w) in got.iter().zip(want.iter()) {
        let denom = w.abs().max(1.0);
        max_rel = max_rel.max((g - w).abs() / denom);
    }
    assert!(max_rel < 1e-4, "max rel err {max_rel} exceeds 1e-4");
}

/// Edge-tile M path: M=17 is not a multiple of 64. The kernel should
/// handle ragged M via row-bounds predication.
#[test]
#[ignore]
fn matmul_int8_17_8_32_edge_m() {
    let (got, want) = round_trip(17, 8, 32, 1.0, 0xDEAD_0001_BEEF_0002);
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!((g - w).abs() < 1e-4, "mismatch at {i}: got {g}, want {w}");
    }
}

/// Edge-tile N path: N=13 is not a multiple of 64. The kernel should
/// handle ragged N via col-bounds predication.
#[test]
#[ignore]
fn matmul_int8_16_13_32_edge_n() {
    let (got, want) = round_trip(16, 13, 32, 1.0, 0xFACE_0003_BEEF_0004);
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!((g - w).abs() < 1e-4, "mismatch at {i}: got {g}, want {w}");
    }
}

/// K=31 must be rejected cleanly with `KaioError::InvalidConfig`.
#[test]
#[ignore]
fn matmul_int8_rejects_k_not_multiple_of_32() {
    let device = KaioDevice::new(0).expect("no GPU");
    let a = device.alloc_zeros::<i8>(16 * 31).unwrap();
    let b = device.alloc_zeros::<i8>(31 * 8).unwrap();
    let mut c = device.alloc_zeros::<f32>(16 * 8).unwrap();
    let err = matmul_int8(&device, &a, &b, &mut c, 1.0, 16, 8, 31).unwrap_err();
    match err {
        KaioError::InvalidConfig(msg) => {
            assert!(msg.contains("K must be a multiple of 32"), "got: {msg}");
        }
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}

/// Boundary i8 values: ensure the signed-i8 multiply path handles
/// `i8::MIN`, `i8::MAX`, `-1`, `0`, `1` at known positions.
#[test]
#[ignore]
fn matmul_int8_boundary_values() {
    let device = KaioDevice::new(0).expect("no GPU");
    let m = 16u32;
    let n = 8u32;
    let k = 32u32;
    // Construct A: first row = [i8::MIN, i8::MAX, -1, 0, 1, 2, 3, ..., 27]
    let mut a = vec![0i8; (m * k) as usize];
    a[0] = i8::MIN;
    a[1] = i8::MAX;
    a[2] = -1;
    a[3] = 0;
    a[4] = 1;
    for (offset, slot) in a.iter_mut().enumerate().take(k as usize).skip(5) {
        *slot = (offset - 5) as i8; // 0..=26
    }
    // B: identity-like in col 0 (first K entries = 1, rest = 0); varied in col 1.
    let mut b = vec![0i8; (k * n) as usize];
    for kk in 0..k as usize {
        b[kk * n as usize] = 1; // col 0
        b[kk * n as usize + 1] = if kk % 2 == 0 { 1 } else { -1 }; // col 1
    }
    let scale = 1.0f32;

    let a_gpu = device.alloc_from(&a).unwrap();
    let b_gpu = device.alloc_from(&b).unwrap();
    let mut c_gpu = device.alloc_zeros::<f32>((m * n) as usize).unwrap();
    matmul_int8(&device, &a_gpu, &b_gpu, &mut c_gpu, scale, m, n, k).unwrap();
    let got: Vec<f32> = c_gpu.to_host(&device).unwrap();

    let want = cpu_reference(&a, &b, scale, m, n, k);
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!((g - w).abs() < 1.0, "mismatch at {i}: got {g}, want {w}");
    }
}
