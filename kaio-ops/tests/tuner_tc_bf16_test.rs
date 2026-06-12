//! Sprint 9.1.2 — integration tests for the bf16 tensor-core auto-tuner
//! (`tune_matmul_tc_bf16` / `matmul_auto_tc_bf16`).
//!
//! Sibling to `tuner_tc_test.rs` (the f16 tuner integration suite).
//! Host-only pre-dispatch rejection tests live alongside the GPU
//! correctness/dispatch/fallback tests; the GPU-requiring tests are
//! `#[ignore]`'d.
//!
//! The cache coexistence invariant (f16-TC vs bf16-TC cache entries
//! sharing the same JSON file without collision) is locked by a
//! separate in-module unit test inside `tuner.rs`'s `tc_tuner_tests`,
//! because `TuneCache` / `TuneResult` are private to that module and
//! not reachable from an integration test.
//!
//! Shared helpers come from `tests/common/mod.rs`.

use half::bf16;
use kaio::prelude::*;
use kaio_ops::{matmul_auto_tc_bf16, tune_matmul_tc_bf16};

mod common;
use common::{assert_bf16_close_d5, cpu_matmul_bf16xbf16_f64, patterned_bf16_data};

/// Redirect the tuner cache to an isolated temp file for the duration
/// of a test. Caller must keep the guard alive across the test body;
/// the guard restores the previous env var (or removes it) on drop so
/// tests don't leak state into each other.
///
/// Cargo's test harness runs tests within a test binary in parallel by
/// default. Without serialisation, two concurrent `CacheEnvGuard::set`
/// calls would race on the global `KAIO_TUNE_CACHE` env var and one
/// test's "previous" snapshot could capture another test's redirect
/// path, leaking state across tests on guard drop. The static
/// `ENV_MUTEX` below serialises env mutations within this binary for
/// the lifetime of any `CacheEnvGuard` instance.
///
/// Each integration test file is its own process, so this mutex only
/// needs to cover within-binary parallelism. The sibling
/// `tuner_tc_test.rs` uses the same pattern with its own local
/// mutex.
static ENV_MUTEX: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();

struct CacheEnvGuard {
    previous: Option<String>,
    _lock: std::sync::MutexGuard<'static, ()>,
}

impl CacheEnvGuard {
    fn set(path: &str) -> Self {
        let mutex = ENV_MUTEX.get_or_init(|| std::sync::Mutex::new(()));
        // Poison-tolerant: if a prior test panicked while holding the
        // lock, we still want subsequent tests to acquire it and run.
        let lock = mutex
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let previous = std::env::var("KAIO_TUNE_CACHE").ok();
        // SAFETY: env mutation is serialised by the static `ENV_MUTEX`
        // held in `_lock`; no other thread inside this test binary can
        // observe a partial update or race with our `previous` snapshot.
        unsafe { std::env::set_var("KAIO_TUNE_CACHE", path) };
        Self {
            previous,
            _lock: lock,
        }
    }
}

impl Drop for CacheEnvGuard {
    fn drop(&mut self) {
        // SAFETY: `_lock` is still held — see set().
        unsafe {
            match &self.previous {
                Some(v) => std::env::set_var("KAIO_TUNE_CACHE", v),
                None => std::env::remove_var("KAIO_TUNE_CACHE"),
            }
        }
        // `_lock` drops here, releasing serialisation for the next guard.
    }
}

fn temp_cache_path(tag: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "kaio_tuner_tc_bf16_test_{}_{}.json",
        tag,
        std::process::id()
    ));
    // Clear any leftover file from a previous run.
    let _ = std::fs::remove_file(&p);
    p
}

// ---------------------------------------------------------------------------
// Host-only pre-dispatch rejection tests (still `#[ignore]` because
// they spawn a GPU device for the `device.info()` probe).
// ---------------------------------------------------------------------------

/// Sprint 9.1 / 9.1.1 edge-tile predication: M and N may be any
/// positive value. This test is the positive counterpart — a
/// non-divisible (M, N) pair is accepted by `matmul_auto_tc_bf16` and
/// produces correct output via the dispatched kernel.
#[test]
#[ignore]
fn matmul_auto_tc_bf16_handles_non_divisible_dims() {
    let device = KaioDevice::new(0).expect("GPU required");
    let info = device.info().expect("device info");
    if info.compute_capability.0 < 8 {
        return;
    }

    // M=17, N=9, K=16 — both M and N ragged within the 64×64 block tile.
    let m: usize = 17;
    let n: usize = 9;
    let k: usize = 16;
    let a_host: Vec<bf16> = (0..m * k)
        .map(|idx| bf16::from_f32(((idx % 7) as f32 - 3.0) * 0.1))
        .collect();
    let b_host: Vec<bf16> = (0..k * n)
        .map(|idx| bf16::from_f32(((idx % 5) as f32 - 2.0) * 0.1))
        .collect();

    let a = device.alloc_from(&a_host).unwrap();
    let b = device.alloc_from(&b_host).unwrap();
    let mut c = device.alloc_zeros::<f32>(m * n).unwrap();

    matmul_auto_tc_bf16(&device, &a, &b, &mut c, m as u32, n as u32, k as u32)
        .expect("non-divisible dims should be accepted (edge-tile predication)");

    let got = c.to_host(&device).unwrap();
    let expected = cpu_matmul_bf16xbf16_f64(&a_host, &b_host, m, n, k);
    assert_bf16_close_d5(&got, &expected, m, n, "auto_tc_bf16_non_divisible");
}

/// K still must be a multiple of 16 (mma K-tile is structural).
#[test]
#[ignore]
fn matmul_auto_tc_bf16_rejects_k_not_multiple_of_16() {
    let device = KaioDevice::new(0).expect("GPU required");
    let info = device.info().expect("device info");
    if info.compute_capability.0 < 8 {
        return;
    }
    let a = device.alloc_zeros::<bf16>(16 * 24).unwrap();
    let b = device.alloc_zeros::<bf16>(24 * 8).unwrap();
    let mut c = device.alloc_zeros::<f32>(16 * 8).unwrap();
    // K=24 violates K%16=0.
    let err = matmul_auto_tc_bf16(&device, &a, &b, &mut c, 16, 8, 24)
        .expect_err("K not multiple of 16 must still be rejected");
    match err {
        KaioError::InvalidConfig(msg) => {
            assert!(
                msg.contains("K%") || msg.contains("K must be a multiple"),
                "error should name K constraint; got: {msg}"
            );
            assert!(
                msg.contains("matmul_auto_tc_bf16"),
                "bf16 error message should name the bf16 entry point; got: {msg}"
            );
        }
        other => panic!("expected InvalidConfig, got: {other:?}"),
    }
}

#[test]
#[ignore]
fn matmul_auto_tc_bf16_rejects_zero_dim() {
    let device = KaioDevice::new(0).expect("GPU required");
    let a = device.alloc_zeros::<bf16>(16).unwrap();
    let b = device.alloc_zeros::<bf16>(8).unwrap();
    let mut c = device.alloc_zeros::<f32>(8).unwrap();
    let err = matmul_auto_tc_bf16(&device, &a, &b, &mut c, 0, 8, 16)
        .expect_err("zero-dim must be rejected");
    assert!(matches!(err, KaioError::InvalidConfig(_)));
}

// ---------------------------------------------------------------------------
// GPU correctness / dispatch / fallback tests
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn tune_matmul_tc_bf16_returns_valid_variant() {
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

    let variant = tune_matmul_tc_bf16(&device, 64, 64, 64)
        .expect("tune_matmul_tc_bf16 should succeed on SM 8.0+ with divisible dims");
    assert!(
        variant == "tensor_core_bf16" || variant == "tensor_core_bf16_async",
        "unexpected variant string: {variant}"
    );
}

#[test]
#[ignore]
fn matmul_auto_tc_bf16_produces_correct_output() {
    let device = KaioDevice::new(0).expect("GPU required");
    let info = device.info().expect("device info");
    if info.compute_capability.0 < 8 {
        return;
    }

    let _guard = CacheEnvGuard::set(temp_cache_path("correctness").to_str().unwrap());

    // Prime the cache by tuning first, then dispatch through matmul_auto_tc_bf16.
    tune_matmul_tc_bf16(&device, 32, 16, 32).expect("tune");

    let m = 32usize;
    let n = 16usize;
    let k = 32usize;
    let a_host = patterned_bf16_data(m * k);
    let b_host = patterned_bf16_data(k * n);
    let a = device.alloc_from(&a_host).expect("alloc A");
    let b = device.alloc_from(&b_host).expect("alloc B");
    let mut c = device.alloc_zeros::<f32>(m * n).expect("alloc C");

    matmul_auto_tc_bf16(&device, &a, &b, &mut c, m as u32, n as u32, k as u32)
        .expect("matmul_auto_tc_bf16 failed");

    let got = c.to_host(&device).expect("C to host");
    let expected = cpu_matmul_bf16xbf16_f64(&a_host, &b_host, m, n, k);
    assert_bf16_close_d5(&got, &expected, m, n, "auto_tc_bf16");
}

#[test]
#[ignore]
fn matmul_auto_tc_bf16_falls_back_without_cache() {
    // No tune_matmul_tc_bf16 call first — cache is empty for these dims.
    // Expected: dispatch falls back to the size-heuristic default
    // (`cache_miss_default_bf16`, inheriting the f16 3072 threshold).
    // 16×8×16 is well below 3072, so the sync variant
    // (MatmulTcBf16Variant::TensorCore → matmul_tc_bf16) is selected.
    let device = KaioDevice::new(0).expect("GPU required");
    let info = device.info().expect("device info");
    if info.compute_capability.0 < 8 {
        return;
    }

    let _guard = CacheEnvGuard::set(temp_cache_path("fallback").to_str().unwrap());

    let m = 16usize;
    let n = 8usize;
    let k = 16usize;
    let a_host = patterned_bf16_data(m * k);
    let b_host = patterned_bf16_data(k * n);
    let a = device.alloc_from(&a_host).unwrap();
    let b = device.alloc_from(&b_host).unwrap();
    let mut c = device.alloc_zeros::<f32>(m * n).unwrap();

    matmul_auto_tc_bf16(&device, &a, &b, &mut c, m as u32, n as u32, k as u32)
        .expect("fallback dispatch should succeed");

    let got = c.to_host(&device).unwrap();
    let expected = cpu_matmul_bf16xbf16_f64(&a_host, &b_host, m, n, k);
    assert_bf16_close_d5(&got, &expected, m, n, "auto_tc_bf16_fallback");
}
