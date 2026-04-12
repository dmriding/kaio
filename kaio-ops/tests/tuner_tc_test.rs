//! Sprint 6.5 — integration tests for the tensor-core auto-tuner
//! (`tune_matmul_tc` / `matmul_auto_tc`).
//!
//! Host-only tests for pre-dispatch rejection live alongside these
//! (Rust treats any `#[test]` in an integration-test file as a test,
//! whether or not it actually needs a GPU). The GPU-requiring tests
//! are `#[ignore]`'d.
//!
//! Shared helpers come from `tests/common/mod.rs`.

use kaio::prelude::*;
use kaio_ops::{matmul_auto_tc, tune_matmul_tc};

mod common;
use common::{assert_close_with_k_scaled_tol, cpu_matmul_f16xf16_f32, patterned_f16_data};

/// Redirect the tuner cache to an isolated temp file for the duration
/// of a test. Caller must keep the guard alive across the test body;
/// the guard restores the previous env var (or removes it) on drop so
/// tests don't leak state into each other.
struct CacheEnvGuard {
    previous: Option<String>,
}

impl CacheEnvGuard {
    fn set(path: &str) -> Self {
        let previous = std::env::var("KAIO_TUNE_CACHE").ok();
        // SAFETY: test binaries run single-threaded by default
        // (--test-threads=1 effectively, or at least these env-mutation
        // tests serialize via the harness). See the parallel note in
        // kaio-core's ptxas_verify tech-debt entry.
        unsafe { std::env::set_var("KAIO_TUNE_CACHE", path) };
        Self { previous }
    }
}

impl Drop for CacheEnvGuard {
    fn drop(&mut self) {
        // SAFETY: see set().
        unsafe {
            match &self.previous {
                Some(v) => std::env::set_var("KAIO_TUNE_CACHE", v),
                None => std::env::remove_var("KAIO_TUNE_CACHE"),
            }
        }
    }
}

fn temp_cache_path(tag: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "kaio_tuner_tc_test_{}_{}.json",
        tag,
        std::process::id()
    ));
    // Clear any leftover file from a previous run.
    let _ = std::fs::remove_file(&p);
    p
}

// ---------------------------------------------------------------------------
// Host-only pre-dispatch rejection tests (no GPU required beyond device
// probe; these touch `device.info()` so we still mark `#[ignore]` —
// spawning a GPU device is what they cost).
// ---------------------------------------------------------------------------

/// Sprint 6.7 Gate C replaces the prior "rejects non-divisible M"
/// behavior — M and N may now be any positive value (edge-tile
/// predication in the kernel handles the OOB rows/cols). This test
/// is the positive counterpart: a non-divisible M, N pair is accepted
/// by `matmul_auto_tc` and produces correct output via the dispatched
/// kernel.
#[test]
#[ignore]
fn matmul_auto_tc_handles_non_divisible_dims() {
    let device = KaioDevice::new(0).expect("GPU required");
    let info = device.info().expect("device info");
    if info.compute_capability.0 < 8 {
        return;
    }

    // M=17, N=9, K=16 — both M and N ragged within the 64×64 block tile.
    let m: usize = 17;
    let n: usize = 9;
    let k: usize = 16;
    let a_host: Vec<half::f16> = (0..m * k)
        .map(|idx| half::f16::from_f32(((idx % 7) as f32 - 3.0) * 0.1))
        .collect();
    let b_host: Vec<half::f16> = (0..k * n)
        .map(|idx| half::f16::from_f32(((idx % 5) as f32 - 2.0) * 0.1))
        .collect();

    let a = device.alloc_from(&a_host).unwrap();
    let b = device.alloc_from(&b_host).unwrap();
    let mut c = device.alloc_zeros::<f32>(m * n).unwrap();

    matmul_auto_tc(&device, &a, &b, &mut c, m as u32, n as u32, k as u32)
        .expect("non-divisible dims should now be accepted (Gate C)");

    let got = c.to_host(&device).unwrap();

    // CPU reference (mirrors common::cpu_matmul_f16xf16_f32 inline).
    let mut expected = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a_host[i * k + p].to_f32() * b_host[p * n + j].to_f32();
            }
            expected[i * n + j] = sum;
        }
    }
    let mut max_abs = 0.0f32;
    for idx in 0..(m * n) {
        let e = (got[idx] - expected[idx]).abs();
        if e > max_abs {
            max_abs = e;
        }
    }
    assert!(
        max_abs < 1e-2,
        "non-divisible 17×9×16 max_abs_err = {max_abs:.2e} (expected < 1e-2)"
    );
}

/// K still must be a multiple of 16 (mma K-tile is structural).
#[test]
#[ignore]
fn matmul_auto_tc_rejects_k_not_multiple_of_16() {
    let device = KaioDevice::new(0).expect("GPU required");
    let info = device.info().expect("device info");
    if info.compute_capability.0 < 8 {
        return;
    }
    let a = device.alloc_zeros::<half::f16>(16 * 24).unwrap();
    let b = device.alloc_zeros::<half::f16>(24 * 8).unwrap();
    let mut c = device.alloc_zeros::<f32>(16 * 8).unwrap();
    // K=24 violates K%16=0.
    let err = matmul_auto_tc(&device, &a, &b, &mut c, 16, 8, 24)
        .expect_err("K not multiple of 16 must still be rejected");
    match err {
        KaioError::InvalidConfig(msg) => {
            assert!(
                msg.contains("K%") || msg.contains("K must be a multiple"),
                "error should name K constraint; got: {msg}"
            );
        }
        other => panic!("expected InvalidConfig, got: {other:?}"),
    }
}

#[test]
#[ignore]
fn matmul_auto_tc_rejects_zero_dim() {
    let device = KaioDevice::new(0).expect("GPU required");
    let a = device.alloc_zeros::<half::f16>(16).unwrap();
    let b = device.alloc_zeros::<half::f16>(8).unwrap();
    let mut c = device.alloc_zeros::<f32>(8).unwrap();
    let err =
        matmul_auto_tc(&device, &a, &b, &mut c, 0, 8, 16).expect_err("zero-dim must be rejected");
    assert!(matches!(err, KaioError::InvalidConfig(_)));
}

// ---------------------------------------------------------------------------
// GPU correctness / dispatch / fallback tests
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn tune_matmul_tc_returns_valid_variant() {
    let device = KaioDevice::new(0).expect("GPU required");
    let info = device.info().expect("device info");
    if info.compute_capability.0 < 8 {
        eprintln!(
            "skipping: SM 8.0+ required (have sm_{}{})",
            info.compute_capability.0, info.compute_capability.1
        );
        return;
    }

    let _guard = CacheEnvGuard::set(temp_cache_path("tune_variant").to_str().unwrap());

    let variant = tune_matmul_tc(&device, 64, 64, 64)
        .expect("tune_matmul_tc should succeed on SM 8.0+ with divisible dims");
    assert!(
        variant == "tensor_core" || variant == "tensor_core_async",
        "unexpected variant string: {variant}"
    );
}

#[test]
#[ignore]
fn matmul_auto_tc_produces_correct_output() {
    let device = KaioDevice::new(0).expect("GPU required");
    let info = device.info().expect("device info");
    if info.compute_capability.0 < 8 {
        return;
    }

    let _guard = CacheEnvGuard::set(temp_cache_path("correctness").to_str().unwrap());

    // Prime the cache by tuning first, then dispatch through matmul_auto_tc.
    tune_matmul_tc(&device, 32, 16, 32).expect("tune");

    let m = 32usize;
    let n = 16usize;
    let k = 32usize;
    let a_host = patterned_f16_data(m * k);
    let b_host = patterned_f16_data(k * n);
    let a = device.alloc_from(&a_host).expect("alloc A");
    let b = device.alloc_from(&b_host).expect("alloc B");
    let mut c = device.alloc_zeros::<f32>(m * n).expect("alloc C");

    matmul_auto_tc(&device, &a, &b, &mut c, m as u32, n as u32, k as u32)
        .expect("matmul_auto_tc failed");

    let got = c.to_host(&device).expect("C to host");
    let expected = cpu_matmul_f16xf16_f32(&a_host, &b_host, m, n, k);
    assert_close_with_k_scaled_tol(&got, &expected, &a_host, &b_host, m, n, k, "auto_tc");
}

#[test]
#[ignore]
fn matmul_auto_tc_falls_back_without_cache() {
    // No tune_matmul_tc call first — cache is empty for these dims.
    // Expected: dispatch falls back to the conservative default
    // (MatmulTcVariant::TensorCore per Sprint 6.5 D4), which matches
    // 6.4's timing observation (async is slower at 1 warp/block).
    let device = KaioDevice::new(0).expect("GPU required");
    let info = device.info().expect("device info");
    if info.compute_capability.0 < 8 {
        return;
    }

    let _guard = CacheEnvGuard::set(temp_cache_path("fallback").to_str().unwrap());

    let m = 16usize;
    let n = 8usize;
    let k = 16usize;
    let a_host = patterned_f16_data(m * k);
    let b_host = patterned_f16_data(k * n);
    let a = device.alloc_from(&a_host).unwrap();
    let b = device.alloc_from(&b_host).unwrap();
    let mut c = device.alloc_zeros::<f32>(m * n).unwrap();

    matmul_auto_tc(&device, &a, &b, &mut c, m as u32, n as u32, k as u32)
        .expect("fallback dispatch should succeed");

    let got = c.to_host(&device).unwrap();
    let expected = cpu_matmul_f16xf16_f32(&a_host, &b_host, m, n, k);
    assert_close_with_k_scaled_tol(
        &got,
        &expected,
        &a_host,
        &b_host,
        m,
        n,
        k,
        "auto_tc_fallback",
    );
}
