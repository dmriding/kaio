//! Tests for kaio_ops::attention() — standard single-head attention.
//!
//! Sprint 5.2: correctness baseline for FlashAttention.
//! All tests compare GPU output against CPU reference.

#![allow(clippy::too_many_arguments)]

use kaio::prelude::*;
use kaio_ops::attention;

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

fn cpu_softmax_rows(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row = &data[r * cols..(r + 1) * cols];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for c in 0..cols {
            out[r * cols + c] = exps[c] / sum;
        }
    }
    out
}

/// CPU reference: out = softmax(Q * K^T / sqrt(d_k)) * V
fn cpu_attention(q: &[f32], k: &[f32], v: &[f32], seq_len: usize, d_k: usize) -> Vec<f32> {
    // Step 1: S = Q * K^T / sqrt(d_k)
    // Q is (seq_len, d_k), K^T is (d_k, seq_len) -> S is (seq_len, seq_len)
    let inv_sqrt_dk = 1.0f32 / (d_k as f32).sqrt();
    let mut scores = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let mut dot = 0.0f32;
            for d in 0..d_k {
                dot += q[i * d_k + d] * k[j * d_k + d];
            }
            scores[i * seq_len + j] = dot * inv_sqrt_dk;
        }
    }

    // Step 2: P = softmax(S) row-wise
    let probs = cpu_softmax_rows(&scores, seq_len, seq_len);

    // Step 3: out = P * V
    cpu_matmul(&probs, v, seq_len, d_k, seq_len)
}

fn check_attention(
    seq_len: usize,
    d_k: usize,
    q_data: &[f32],
    k_data: &[f32],
    v_data: &[f32],
    label: &str,
) {
    let device = KaioDevice::new(0).expect("GPU required");

    let q = device.alloc_from(q_data).unwrap();
    let k = device.alloc_from(k_data).unwrap();
    let v = device.alloc_from(v_data).unwrap();
    let mut out = device.alloc_zeros::<f32>(seq_len * d_k).unwrap();

    attention(&device, &q, &k, &v, &mut out, seq_len as u32, d_k as u32).unwrap();

    let result = out.to_host(&device).unwrap();
    let expected = cpu_attention(q_data, k_data, v_data, seq_len, d_k);

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;

    for idx in 0..seq_len * d_k {
        let got = result[idx];
        let exp = expected[idx];
        let abs_err = (got - exp).abs();
        let rel_err = if exp.abs() > 1e-6 {
            abs_err / exp.abs()
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
            abs_err < 1e-3 || rel_err < 1e-2,
            "{label}: error at index {idx}: got {got}, expected {exp}, abs={abs_err:.2e}, rel={rel_err:.2e}"
        );
    }

    eprintln!("{label} ({seq_len}×{d_k}): max_abs={max_abs:.2e}, max_rel={max_rel:.2e}");
}

// --- Correctness tests ---

#[test]
#[ignore] // requires NVIDIA GPU
fn attention_tiny() {
    let seq_len = 4;
    let d_k = 4;
    let q: Vec<f32> = (0..seq_len * d_k).map(|i| (i % 7) as f32 * 0.1).collect();
    let k: Vec<f32> = (0..seq_len * d_k)
        .map(|i| ((i + 3) % 5) as f32 * 0.1)
        .collect();
    let v: Vec<f32> = (0..seq_len * d_k)
        .map(|i| ((i + 1) % 11) as f32 * 0.1)
        .collect();
    check_attention(seq_len, d_k, &q, &k, &v, "tiny");
}

#[test]
#[ignore] // requires NVIDIA GPU
fn attention_16x16() {
    let seq_len = 16;
    let d_k = 16;
    let q: Vec<f32> = (0..seq_len * d_k)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.1)
        .collect();
    let k: Vec<f32> = (0..seq_len * d_k)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
        .collect();
    let v: Vec<f32> = (0..seq_len * d_k)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.1)
        .collect();
    check_attention(seq_len, d_k, &q, &k, &v, "16x16");
}

#[test]
#[ignore] // requires NVIDIA GPU
fn attention_non_aligned() {
    let seq_len = 17;
    let d_k = 19;
    let q: Vec<f32> = (0..seq_len * d_k)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
        .collect();
    let k: Vec<f32> = (0..seq_len * d_k)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.05)
        .collect();
    let v: Vec<f32> = (0..seq_len * d_k)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.05)
        .collect();
    check_attention(seq_len, d_k, &q, &k, &v, "non_aligned");
}

#[test]
#[ignore] // requires NVIDIA GPU
fn attention_medium() {
    let seq_len = 64;
    let d_k = 64;
    let q: Vec<f32> = (0..seq_len * d_k)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.1)
        .collect();
    let k: Vec<f32> = (0..seq_len * d_k)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
        .collect();
    let v: Vec<f32> = (0..seq_len * d_k)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.1)
        .collect();
    check_attention(seq_len, d_k, &q, &k, &v, "medium");
}

#[test]
#[ignore] // requires NVIDIA GPU
fn attention_identity() {
    // Q == K == V — structured input that catches transposition bugs.
    // If K^T indexing is wrong, the Q*K^T result will differ from Q*Q^T.
    let seq_len = 8;
    let d_k = 8;
    let data: Vec<f32> = (0..seq_len * d_k).map(|i| (i as f32) * 0.1).collect();
    check_attention(seq_len, d_k, &data, &data, &data, "identity");
}

// --- Validation tests ---

#[test]
#[ignore] // requires NVIDIA GPU
fn attention_rejects_zero_seq_len() {
    let device = KaioDevice::new(0).expect("GPU required");
    let buf = device.alloc_zeros::<f32>(1).unwrap();
    let mut out = device.alloc_zeros::<f32>(1).unwrap();
    let err = attention(&device, &buf, &buf, &buf, &mut out, 0, 1).unwrap_err();
    assert!(
        err.to_string().contains("non-zero"),
        "expected zero-dim error, got: {err}"
    );
}

#[test]
#[ignore] // requires NVIDIA GPU
fn attention_rejects_zero_dk() {
    let device = KaioDevice::new(0).expect("GPU required");
    let buf = device.alloc_zeros::<f32>(1).unwrap();
    let mut out = device.alloc_zeros::<f32>(1).unwrap();
    let err = attention(&device, &buf, &buf, &buf, &mut out, 1, 0).unwrap_err();
    assert!(
        err.to_string().contains("non-zero"),
        "expected zero-dim error, got: {err}"
    );
}
