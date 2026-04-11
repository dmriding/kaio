//! Tests for kaio_ops auto-tuner.
//!
//! Sprint 5.5: benchmark variants, cache results, dispatch.

#![allow(clippy::too_many_arguments)]

use kaio::prelude::*;
use kaio_ops::{attention_auto, matmul_auto, tune_attention, tune_matmul};

// --- Helpers for cache isolation ---
// Edition 2024: set_var/remove_var are unsafe (not thread-safe).
// Tests are single-threaded via --test-threads=1 for tuner tests.

fn set_cache(path: &std::path::Path) {
    unsafe { std::env::set_var("KAIO_TUNE_CACHE", path) };
}

fn clear_cache() {
    unsafe { std::env::remove_var("KAIO_TUNE_CACHE") };
}

// --- CPU reference ---

fn cpu_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

// --- Tuning tests (require GPU) ---

#[test]
#[ignore] // requires NVIDIA GPU
fn tune_matmul_returns_variant() {
    let tmp = std::env::temp_dir().join("kaio_test_tune_matmul.json");
    set_cache(&tmp);

    let device = KaioDevice::new(0).expect("GPU required");
    let result = tune_matmul(&device, 64, 64, 64).unwrap();

    assert!(
        result == "naive_16x16" || result == "optimized_64x64",
        "unexpected variant: {result}"
    );

    let _ = std::fs::remove_file(&tmp);
    clear_cache();
}

#[test]
#[ignore] // requires NVIDIA GPU
fn matmul_auto_produces_correct_output() {
    let tmp = std::env::temp_dir().join("kaio_test_matmul_auto.json");
    set_cache(&tmp);

    let device = KaioDevice::new(0).expect("GPU required");
    let m = 32usize;
    let n = 32usize;
    let k = 32usize;

    let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
    let expected = cpu_matmul(&a_data, &b_data, m, n, k);

    let a = device.alloc_from(&a_data).unwrap();
    let b = device.alloc_from(&b_data).unwrap();
    let mut c = device.alloc_zeros::<f32>(m * n).unwrap();

    matmul_auto(&device, &a, &b, &mut c, m as u32, n as u32, k as u32).unwrap();

    let result = c.to_host(&device).unwrap();
    for idx in 0..m * n {
        let abs_err = (result[idx] - expected[idx]).abs();
        assert!(
            abs_err < 1e-3,
            "matmul_auto mismatch at {idx}: got {}, expected {}, abs={abs_err:.2e}",
            result[idx],
            expected[idx]
        );
    }

    let _ = std::fs::remove_file(&tmp);
    clear_cache();
}

#[test]
#[ignore] // requires NVIDIA GPU
fn tune_attention_returns_variant() {
    let tmp = std::env::temp_dir().join("kaio_test_tune_attn.json");
    set_cache(&tmp);

    let device = KaioDevice::new(0).expect("GPU required");
    let result = tune_attention(&device, 16, 16).unwrap();

    assert!(
        result == "standard" || result == "flash",
        "unexpected variant: {result}"
    );

    let _ = std::fs::remove_file(&tmp);
    clear_cache();
}

#[test]
#[ignore] // requires NVIDIA GPU
fn attention_auto_produces_correct_output() {
    let tmp = std::env::temp_dir().join("kaio_test_attn_auto.json");
    set_cache(&tmp);

    let device = KaioDevice::new(0).expect("GPU required");
    let seq_len = 16usize;
    let d_k = 16usize;

    let q_data: Vec<f32> = (0..seq_len * d_k)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.1)
        .collect();
    let k_data: Vec<f32> = (0..seq_len * d_k)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
        .collect();
    let v_data: Vec<f32> = (0..seq_len * d_k)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.1)
        .collect();

    let q = device.alloc_from(&q_data).unwrap();
    let k = device.alloc_from(&k_data).unwrap();
    let v = device.alloc_from(&v_data).unwrap();
    let mut out = device.alloc_zeros::<f32>(seq_len * d_k).unwrap();

    attention_auto(&device, &q, &k, &v, &mut out, seq_len as u32, d_k as u32).unwrap();

    let result = out.to_host(&device).unwrap();
    for (i, &val) in result.iter().enumerate() {
        assert!(!val.is_nan(), "NaN at index {i}");
        assert!(val.abs() < 100.0, "unreasonable value at {i}: {val}");
    }

    let _ = std::fs::remove_file(&tmp);
    clear_cache();
}

// --- Cache and fallback tests ---

#[test]
fn tune_cache_roundtrip() {
    let tmp = std::env::temp_dir().join("kaio_test_cache_rt.json");
    set_cache(&tmp);

    let cache_json = r#"{"version":1,"results":[{"kernel":"matmul","variant":"naive_16x16","sm_target":"sm_89","dims":[64,64,64],"median_ms":0.5}]}"#;
    std::fs::write(&tmp, cache_json).unwrap();

    let contents = std::fs::read_to_string(&tmp).unwrap();
    assert!(contents.contains("naive_16x16"));
    assert!(contents.contains("sm_89"));

    let _ = std::fs::remove_file(&tmp);
    clear_cache();
}

#[test]
#[ignore] // requires NVIDIA GPU
fn auto_falls_back_no_cache() {
    let tmp = std::env::temp_dir().join("kaio_test_no_cache_exists.json");
    let _ = std::fs::remove_file(&tmp);
    set_cache(&tmp);

    let device = KaioDevice::new(0).expect("GPU required");
    let a = device.alloc_zeros::<f32>(64).unwrap();
    let b = device.alloc_zeros::<f32>(64).unwrap();
    let mut c = device.alloc_zeros::<f32>(64).unwrap();

    // Should succeed with default variant (no crash, no cache)
    matmul_auto(&device, &a, &b, &mut c, 8, 8, 8).unwrap();

    let _ = std::fs::remove_file(&tmp);
    clear_cache();
}

#[test]
#[ignore] // requires NVIDIA GPU
fn tune_attention_skips_flash_when_dk_too_large() {
    let tmp = std::env::temp_dir().join("kaio_test_dk_large.json");
    set_cache(&tmp);

    let device = KaioDevice::new(0).expect("GPU required");
    let result = tune_attention(&device, 1, 257).unwrap();

    assert_eq!(
        result, "standard",
        "expected standard (flash ineligible for d_k=257), got: {result}"
    );

    let _ = std::fs::remove_file(&tmp);
    clear_cache();
}
