//! Tests for kaio_ops attention — standard and FlashAttention.
//!
//! Sprint 5.2: standard attention correctness baseline.
//! Sprint 5.3: causal masking.
//! Sprint 5.4: FlashAttention (online softmax + tiled attention).

#![allow(clippy::too_many_arguments)]

use kaio::prelude::*;
use kaio_ops::{attention, attention_causal, attention_flash, attention_flash_causal};

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

// --- Causal masking tests (Sprint 5.3) ---

/// CPU reference for causal attention: apply mask before softmax.
fn cpu_attention_causal(q: &[f32], k: &[f32], v: &[f32], seq_len: usize, d_k: usize) -> Vec<f32> {
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
    // Apply causal mask: S[i,j] = -FLT_MAX where j > i
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            scores[i * seq_len + j] = f32::MIN;
        }
    }
    let probs = cpu_softmax_rows(&scores, seq_len, seq_len);
    cpu_matmul(&probs, v, seq_len, d_k, seq_len)
}

fn check_attention_causal(
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

    attention_causal(&device, &q, &k, &v, &mut out, seq_len as u32, d_k as u32).unwrap();

    let result = out.to_host(&device).unwrap();
    let expected = cpu_attention_causal(q_data, k_data, v_data, seq_len, d_k);

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
            "{label}: error at index {idx}: got {got}, expected {exp}, \
             abs={abs_err:.2e}, rel={rel_err:.2e}"
        );
    }

    eprintln!("{label} ({seq_len}×{d_k}): max_abs={max_abs:.2e}, max_rel={max_rel:.2e}");
}

#[test]
#[ignore] // requires NVIDIA GPU
fn attention_causal_tiny() {
    let seq_len = 4;
    let d_k = 4;
    let q: Vec<f32> = (0..seq_len * d_k).map(|i| (i % 7) as f32 * 0.1).collect();
    let k: Vec<f32> = (0..seq_len * d_k)
        .map(|i| ((i + 3) % 5) as f32 * 0.1)
        .collect();
    let v: Vec<f32> = (0..seq_len * d_k)
        .map(|i| ((i + 1) % 11) as f32 * 0.1)
        .collect();
    check_attention_causal(seq_len, d_k, &q, &k, &v, "causal_tiny");
}

#[test]
#[ignore] // requires NVIDIA GPU
fn attention_causal_16x16() {
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
    check_attention_causal(seq_len, d_k, &q, &k, &v, "causal_16x16");
}

#[test]
#[ignore] // requires NVIDIA GPU
fn attention_causal_non_aligned() {
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
    check_attention_causal(seq_len, d_k, &q, &k, &v, "causal_non_aligned");
}

#[test]
#[ignore] // requires NVIDIA GPU
fn attention_causal_medium() {
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
    check_attention_causal(seq_len, d_k, &q, &k, &v, "causal_medium");
}

#[test]
#[ignore] // requires NVIDIA GPU
fn attention_causal_row0_self_only() {
    // Row 0 should only attend to position 0 (all j > 0 masked).
    // If mask uses col >= row instead of col > row, row 0 gets fully
    // masked and softmax produces NaN.
    let seq_len = 8;
    let d_k = 4;
    let q: Vec<f32> = (0..seq_len * d_k).map(|i| (i as f32) * 0.1).collect();
    let k: Vec<f32> = (0..seq_len * d_k).map(|i| (i as f32) * 0.1).collect();
    let v: Vec<f32> = (0..seq_len * d_k).map(|i| (i as f32) * 0.1).collect();

    let device = KaioDevice::new(0).expect("GPU required");
    let q_buf = device.alloc_from(&q).unwrap();
    let k_buf = device.alloc_from(&k).unwrap();
    let v_buf = device.alloc_from(&v).unwrap();
    let mut out = device.alloc_zeros::<f32>(seq_len * d_k).unwrap();

    attention_causal(
        &device,
        &q_buf,
        &k_buf,
        &v_buf,
        &mut out,
        seq_len as u32,
        d_k as u32,
    )
    .unwrap();

    let result = out.to_host(&device).unwrap();

    // Row 0 output should equal V[0,:] (softmax is [1, 0, 0, ...])
    for d in 0..d_k {
        let got = result[d];
        let exp = v[d]; // V[0, d]
        let abs_err = (got - exp).abs();
        assert!(
            abs_err < 1e-4,
            "row0: col {d}: got {got}, expected {exp} (V[0,{d}]), abs={abs_err:.2e}"
        );
    }

    // No NaN anywhere
    for (i, &val) in result.iter().enumerate() {
        assert!(!val.is_nan(), "NaN at index {i} — mask off-by-one?");
    }
}

#[test]
#[ignore] // requires NVIDIA GPU
fn causal_mask_direct() {
    // Verify causal mask produces lower-triangular attention weights.
    // Set V = identity matrix. Then out = P * I = P, giving us the
    // attention weights directly. Verify upper triangle is ~0 and
    // each row sums to ~1.
    let device = KaioDevice::new(0).expect("GPU required");
    let seq_len = 8usize;
    let d_k = seq_len; // d_k == seq_len so V can be identity

    // Q, K: small values so softmax doesn't saturate
    let q: Vec<f32> = (0..seq_len * d_k).map(|i| (i % 7) as f32 * 0.01).collect();
    let k: Vec<f32> = (0..seq_len * d_k)
        .map(|i| ((i + 3) % 5) as f32 * 0.01)
        .collect();
    // V = identity matrix
    let mut v = vec![0.0f32; seq_len * d_k];
    for i in 0..seq_len {
        v[i * d_k + i] = 1.0;
    }

    let q_buf = device.alloc_from(&q).unwrap();
    let k_buf = device.alloc_from(&k).unwrap();
    let v_buf = device.alloc_from(&v).unwrap();
    let mut out = device.alloc_zeros::<f32>(seq_len * d_k).unwrap();

    attention_causal(
        &device,
        &q_buf,
        &k_buf,
        &v_buf,
        &mut out,
        seq_len as u32,
        d_k as u32,
    )
    .unwrap();

    let weights = out.to_host(&device).unwrap();

    // out = P * I = P, so weights[i,j] = attention probability from i to j.
    // Causal: weights[i,j] should be ~0 for j > i, sum to ~1 per row.
    for i in 0..seq_len {
        let mut row_sum = 0.0f32;
        for j in 0..seq_len {
            let w = weights[i * seq_len + j];
            if j > i {
                assert!(
                    w.abs() < 1e-4,
                    "mask leak: weights[{i},{j}] = {w} (should be ~0)"
                );
            } else {
                assert!(w >= 0.0, "negative weight at [{i},{j}]: {w}");
                row_sum += w;
            }
        }
        assert!(
            (row_sum - 1.0).abs() < 1e-3,
            "row {i} weights sum to {row_sum}, expected 1.0"
        );
    }
}

// --- FlashAttention tests (Sprint 5.4) ---

/// Compare flash attention output against standard attention output.
fn check_flash_vs_standard(seq_len: usize, d_k: usize, label: &str) {
    let device = KaioDevice::new(0).expect("GPU required");

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

    // Standard attention (reference)
    let mut out_std = device.alloc_zeros::<f32>(seq_len * d_k).unwrap();
    attention(
        &device,
        &q,
        &k,
        &v,
        &mut out_std,
        seq_len as u32,
        d_k as u32,
    )
    .unwrap();
    let result_std = out_std.to_host(&device).unwrap();

    // FlashAttention
    let mut out_flash = device.alloc_zeros::<f32>(seq_len * d_k).unwrap();
    attention_flash(
        &device,
        &q,
        &k,
        &v,
        &mut out_flash,
        seq_len as u32,
        d_k as u32,
    )
    .unwrap();
    let result_flash = out_flash.to_host(&device).unwrap();

    let mut max_abs = 0.0f32;
    for idx in 0..seq_len * d_k {
        let abs_err = (result_flash[idx] - result_std[idx]).abs();
        let rel_err = if result_std[idx].abs() > 1e-6 {
            abs_err / result_std[idx].abs()
        } else {
            abs_err
        };
        if abs_err > max_abs {
            max_abs = abs_err;
        }
        assert!(
            abs_err < 1e-2 || rel_err < 1e-1,
            "{label}: flash vs standard mismatch at {idx}: \
             flash={}, std={}, abs={abs_err:.2e}, rel={rel_err:.2e}",
            result_flash[idx],
            result_std[idx]
        );
    }
    eprintln!("{label} ({seq_len}×{d_k}): flash_vs_std max_abs={max_abs:.2e}");
}

#[test]
#[ignore]
fn flash_attention_tiny() {
    let seq_len = 4;
    let d_k = 4;
    let q: Vec<f32> = (0..seq_len * d_k).map(|i| (i % 7) as f32 * 0.1).collect();
    let k: Vec<f32> = (0..seq_len * d_k)
        .map(|i| ((i + 3) % 5) as f32 * 0.1)
        .collect();
    let v: Vec<f32> = (0..seq_len * d_k)
        .map(|i| ((i + 1) % 11) as f32 * 0.1)
        .collect();
    check_attention(seq_len, d_k, &q, &k, &v, "flash_tiny");
}

#[test]
#[ignore]
fn flash_attention_16x16() {
    check_flash_vs_standard(16, 16, "flash_16x16");
}

#[test]
#[ignore]
fn flash_attention_non_aligned() {
    check_flash_vs_standard(17, 19, "flash_non_aligned");
}

#[test]
#[ignore]
fn flash_attention_medium() {
    check_flash_vs_standard(64, 64, "flash_medium");
}

#[test]
#[ignore]
fn flash_matches_standard() {
    // The most important test: validates flash output against the
    // known-correct standard attention from Sprint 5.2.
    check_flash_vs_standard(32, 32, "flash_matches_standard");
}

#[test]
#[ignore]
fn flash_attention_causal_medium() {
    let device = KaioDevice::new(0).expect("GPU required");
    let seq_len = 64usize;
    let d_k = 64usize;

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

    // Standard causal (reference)
    let mut out_std = device.alloc_zeros::<f32>(seq_len * d_k).unwrap();
    attention_causal(
        &device,
        &q,
        &k,
        &v,
        &mut out_std,
        seq_len as u32,
        d_k as u32,
    )
    .unwrap();
    let result_std = out_std.to_host(&device).unwrap();

    // Flash causal
    let mut out_flash = device.alloc_zeros::<f32>(seq_len * d_k).unwrap();
    attention_flash_causal(
        &device,
        &q,
        &k,
        &v,
        &mut out_flash,
        seq_len as u32,
        d_k as u32,
    )
    .unwrap();
    let result_flash = out_flash.to_host(&device).unwrap();

    let mut max_abs = 0.0f32;
    for idx in 0..seq_len * d_k {
        let abs_err = (result_flash[idx] - result_std[idx]).abs();
        if abs_err > max_abs {
            max_abs = abs_err;
        }
        assert!(
            abs_err < 1e-2,
            "flash_causal vs standard at {idx}: flash={}, std={}, abs={abs_err:.2e}",
            result_flash[idx],
            result_std[idx]
        );
    }
    eprintln!("flash_causal_medium (64×64): max_abs={max_abs:.2e}");
}

#[test]
#[ignore]
fn flash_causal_first_rows() {
    // Verify rows 0, 1, 2 specifically in causal mode.
    // Row 0 attends only to position 0 (output = V[0]).
    // Catches masked-tile / tiny-valid-set bugs.
    let device = KaioDevice::new(0).expect("GPU required");
    let seq_len = 8usize;
    let d_k = 4usize;
    let q: Vec<f32> = (0..seq_len * d_k).map(|i| (i as f32) * 0.1).collect();
    let k: Vec<f32> = (0..seq_len * d_k).map(|i| (i as f32) * 0.1).collect();
    let v: Vec<f32> = (0..seq_len * d_k).map(|i| (i as f32) * 0.1).collect();

    let q_buf = device.alloc_from(&q).unwrap();
    let k_buf = device.alloc_from(&k).unwrap();
    let v_buf = device.alloc_from(&v).unwrap();
    let mut out = device.alloc_zeros::<f32>(seq_len * d_k).unwrap();

    attention_flash_causal(
        &device,
        &q_buf,
        &k_buf,
        &v_buf,
        &mut out,
        seq_len as u32,
        d_k as u32,
    )
    .unwrap();

    let result = out.to_host(&device).unwrap();

    // Row 0: only attends to pos 0 → output = V[0,:]
    for d in 0..d_k {
        let got = result[d];
        let exp = v[d];
        assert!(
            (got - exp).abs() < 1e-3,
            "row 0 col {d}: got {got}, expected {exp}"
        );
    }

    // No NaN in first 3 rows
    for (i, &val) in result.iter().enumerate().take(3 * d_k) {
        assert!(!val.is_nan(), "NaN at index {i}");
    }
}

#[test]
#[ignore]
fn flash_all_in_one_tile() {
    // seq_len < 256: single-tile degenerate case.
    check_flash_vs_standard(8, 8, "flash_one_tile");
}

#[test]
#[ignore]
fn flash_last_tile_partial() {
    // seq_len = 257 = 256 + 1: tests final partial tile handling.
    check_flash_vs_standard(257, 32, "flash_partial_tile");
}
