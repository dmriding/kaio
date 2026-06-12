//! FlashAttention backward — correctness suite (Sprint 9.2).
//!
//! Hosts the CPU f64 analytical attention-backward reference — the
//! primary correctness oracle for the `attention_flash_bwd` GPU kernels
//! — plus the GPU correctness matrix added as the sprint lands.
//!
//! The oracle is self-checked against f64 central finite differences
//! before it judges any GPU output: an unverified reference is just a
//! second untested implementation. The self-check tests are host-only
//! and run in the default suite; GPU tests are `#[ignore]`-gated per
//! kaio-ops convention.
//!
//! Backward identities (single-head self-attention, `d_v == d_k`):
//!
//! ```text
//! S  = scale · Q Kᵀ          scale = 1/√d_k
//! P  = softmax(S) row-wise   (causal: S_ij = −∞ for j > i ⇒ P_ij = 0)
//! O  = P V
//!
//! dV_j  = Σ_i P_ij · dO_i
//! dP_ij = dO_i · V_j
//! D_i   = Σ_d dO[i,d] · O[i,d]
//! dS_ij = P_ij · (dP_ij − D_i)
//! dQ_i  = scale · Σ_j dS_ij · K_j
//! dK_j  = scale · Σ_i dS_ij · Q_i
//! ```
//!
//! Closed form at `seq_len = 1`: softmax of one element is exactly 1, so
//! `O = V`, `dP_11 = D_1`, `dS = 0` ⇒ `dV = dO`, `dQ = 0`, `dK = 0`.

#![allow(clippy::too_many_arguments)]

use kaio::prelude::*;
use kaio_ops::{
    attention_flash, attention_flash_bwd, attention_flash_bwd_causal, attention_flash_bwd_dkdv,
    attention_flash_bwd_dq, attention_flash_bwd_preprocess, attention_flash_causal,
    attention_flash_causal_with_stats, attention_flash_with_stats,
};

// --- CPU f64 analytical reference ---

/// f64 attention forward. Returns `(O, P)` — backward needs both.
/// `causal = true` masks scores at `j > i` to −∞ (probability 0).
fn cpu_attention_fwd_f64(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    seq_len: usize,
    d_k: usize,
    causal: bool,
) -> (Vec<f64>, Vec<f64>) {
    let scale = 1.0f64 / (d_k as f64).sqrt();

    // S = scale * Q K^T, with causal mask applied pre-softmax.
    let mut s = vec![0.0f64; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if causal && j > i {
                s[i * seq_len + j] = f64::NEG_INFINITY;
            } else {
                let mut dot = 0.0f64;
                for d in 0..d_k {
                    dot += q[i * d_k + d] * k[j * d_k + d];
                }
                s[i * seq_len + j] = dot * scale;
            }
        }
    }

    // P = softmax(S) row-wise (max-subtracted; exp(−∞ − max) = 0).
    let mut p = vec![0.0f64; seq_len * seq_len];
    for i in 0..seq_len {
        let row = &s[i * seq_len..(i + 1) * seq_len];
        let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = row.iter().map(|&x| (x - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        for j in 0..seq_len {
            p[i * seq_len + j] = exps[j] / sum;
        }
    }

    // O = P V.
    let mut o = vec![0.0f64; seq_len * d_k];
    for i in 0..seq_len {
        for d in 0..d_k {
            let mut acc = 0.0f64;
            for j in 0..seq_len {
                acc += p[i * seq_len + j] * v[j * d_k + d];
            }
            o[i * d_k + d] = acc;
        }
    }

    (o, p)
}

/// f64 analytical attention backward. Takes the forward's `O` and `P`;
/// the causal mask is already encoded in `P` (`P_ij = 0` for masked
/// positions), so no separate causal flag is needed here.
fn cpu_attention_bwd_f64(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    o: &[f64],
    p: &[f64],
    d_out: &[f64],
    seq_len: usize,
    d_k: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let scale = 1.0f64 / (d_k as f64).sqrt();

    // dV_j = Σ_i P_ij · dO_i
    let mut dv = vec![0.0f64; seq_len * d_k];
    for j in 0..seq_len {
        for d in 0..d_k {
            let mut acc = 0.0f64;
            for i in 0..seq_len {
                acc += p[i * seq_len + j] * d_out[i * d_k + d];
            }
            dv[j * d_k + d] = acc;
        }
    }

    // dP_ij = dO_i · V_j
    let mut dp = vec![0.0f64; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let mut acc = 0.0f64;
            for d in 0..d_k {
                acc += d_out[i * d_k + d] * v[j * d_k + d];
            }
            dp[i * seq_len + j] = acc;
        }
    }

    // D_i = Σ_d dO[i,d] · O[i,d]
    let mut dd = vec![0.0f64; seq_len];
    for i in 0..seq_len {
        let mut acc = 0.0f64;
        for d in 0..d_k {
            acc += d_out[i * d_k + d] * o[i * d_k + d];
        }
        dd[i] = acc;
    }

    // dS_ij = P_ij · (dP_ij − D_i)
    let mut ds = vec![0.0f64; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            ds[i * seq_len + j] = p[i * seq_len + j] * (dp[i * seq_len + j] - dd[i]);
        }
    }

    // dQ_i = scale · Σ_j dS_ij · K_j
    let mut dq = vec![0.0f64; seq_len * d_k];
    for i in 0..seq_len {
        for d in 0..d_k {
            let mut acc = 0.0f64;
            for j in 0..seq_len {
                acc += ds[i * seq_len + j] * k[j * d_k + d];
            }
            dq[i * d_k + d] = acc * scale;
        }
    }

    // dK_j = scale · Σ_i dS_ij · Q_i
    let mut dk = vec![0.0f64; seq_len * d_k];
    for j in 0..seq_len {
        for d in 0..d_k {
            let mut acc = 0.0f64;
            for i in 0..seq_len {
                acc += ds[i * seq_len + j] * q[i * d_k + d];
            }
            dk[j * d_k + d] = acc * scale;
        }
    }

    (dq, dk, dv)
}

// --- Oracle self-check: f64 finite differences ---

/// Scalar loss for the FD probe: `loss = Σ_{i,d} w[i,d] · O[i,d]`,
/// which makes the upstream gradient exactly `dO = w`.
fn loss_f64(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    w: &[f64],
    seq_len: usize,
    d_k: usize,
    causal: bool,
) -> f64 {
    let (o, _) = cpu_attention_fwd_f64(q, k, v, seq_len, d_k, causal);
    o.iter().zip(w.iter()).map(|(a, b)| a * b).sum()
}

/// Central finite difference of `loss` w.r.t. one input buffer.
/// `which`: 0 = Q, 1 = K, 2 = V.
fn fd_grad_f64(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    w: &[f64],
    seq_len: usize,
    d_k: usize,
    causal: bool,
    which: usize,
    h: f64,
) -> Vec<f64> {
    let n = seq_len * d_k;
    let mut grad = vec![0.0f64; n];
    for e in 0..n {
        let mut bufs = [q.to_vec(), k.to_vec(), v.to_vec()];
        bufs[which][e] += h;
        let plus = loss_f64(&bufs[0], &bufs[1], &bufs[2], w, seq_len, d_k, causal);
        bufs[which][e] -= 2.0 * h;
        let minus = loss_f64(&bufs[0], &bufs[1], &bufs[2], w, seq_len, d_k, causal);
        grad[e] = (plus - minus) / (2.0 * h);
    }
    grad
}

fn assert_close_f64(got: &[f64], expected: &[f64], abs_tol: f64, rel_tol: f64, label: &str) {
    assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    for (idx, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let abs_err = (g - e).abs();
        let rel_err = if e.abs() > 1e-12 {
            abs_err / e.abs()
        } else {
            abs_err
        };
        max_abs = max_abs.max(abs_err);
        max_rel = max_rel.max(rel_err);
        assert!(
            abs_err < abs_tol || rel_err < rel_tol,
            "{label}: mismatch at index {idx}: got {g}, expected {e}, abs={abs_err:.2e}, rel={rel_err:.2e}"
        );
    }
    eprintln!("{label}: max_abs={max_abs:.2e}, max_rel={max_rel:.2e}");
}

/// Deterministic patterned inputs (house style: modular arithmetic,
/// zero-centered, no RNG dependency).
fn test_inputs_f64(seq_len: usize, d_k: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = seq_len * d_k;
    let q: Vec<f64> = (0..n).map(|i| ((i % 17) as f64 - 8.0) * 0.1).collect();
    let k: Vec<f64> = (0..n).map(|i| ((i % 13) as f64 - 6.0) * 0.1).collect();
    let v: Vec<f64> = (0..n).map(|i| ((i % 19) as f64 - 9.0) * 0.1).collect();
    let w: Vec<f64> = (0..n).map(|i| ((i % 11) as f64 - 5.0) * 0.2).collect();
    (q, k, v, w)
}

fn oracle_self_check(seq_len: usize, d_k: usize, causal: bool, w: &[f64], label: &str) {
    let (q, k, v, _) = test_inputs_f64(seq_len, d_k);

    let (o, p) = cpu_attention_fwd_f64(&q, &k, &v, seq_len, d_k, causal);
    let (dq_a, dk_a, dv_a) = cpu_attention_bwd_f64(&q, &k, &v, &o, &p, w, seq_len, d_k);

    // Central differences: truncation error O(h²), round-off O(ε/h).
    // h = 1e-5 puts both far below the tolerances asserted here.
    let h = 1e-5f64;
    let dq_fd = fd_grad_f64(&q, &k, &v, w, seq_len, d_k, causal, 0, h);
    let dk_fd = fd_grad_f64(&q, &k, &v, w, seq_len, d_k, causal, 1, h);
    let dv_fd = fd_grad_f64(&q, &k, &v, w, seq_len, d_k, causal, 2, h);

    assert_close_f64(&dq_a, &dq_fd, 1e-7, 1e-6, &format!("{label}/dQ"));
    assert_close_f64(&dk_a, &dk_fd, 1e-7, 1e-6, &format!("{label}/dK"));
    assert_close_f64(&dv_a, &dv_fd, 1e-7, 1e-6, &format!("{label}/dV"));
}

// --- Oracle self-check tests (host-only) ---

#[test]
fn oracle_self_check_fd_plain() {
    let (_, _, _, w) = test_inputs_f64(6, 5);
    oracle_self_check(6, 5, false, &w, "fd_plain_6x5");
}

#[test]
fn oracle_self_check_fd_causal() {
    let (_, _, _, w) = test_inputs_f64(6, 5);
    oracle_self_check(6, 5, true, &w, "fd_causal_6x5");
}

#[test]
fn oracle_self_check_fd_wide_range_dout() {
    // Upstream-gradient rows spanning orders of magnitude, so the
    // per-row D_i term varies widely. Exercises the cancellation path
    // in dS = P·(dP − D) that uniform-magnitude data cannot reach.
    let seq_len = 6;
    let d_k = 5;
    let row_scale = [1e-3, 1.0, 5e1, 1e-2, 1e2, 0.5];
    let w: Vec<f64> = (0..seq_len * d_k)
        .map(|i| {
            let base = ((i % 11) as f64 - 5.0) * 0.2;
            base * row_scale[i / d_k]
        })
        .collect();
    oracle_self_check(seq_len, d_k, false, &w, "fd_wide_range_6x5");
    oracle_self_check(seq_len, d_k, true, &w, "fd_wide_range_causal_6x5");
}

#[test]
fn oracle_seq1_closed_form() {
    // seq_len = 1: softmax of one element is exactly 1.0, so O = V
    // bit-for-bit, dP_11 and D_1 are the same sum in the same order,
    // dS = 0, and the closed form dV = dO, dQ = 0, dK = 0 holds to
    // f64 exactness in the reference. This validates the oracle against
    // the closed form before the GPU gate relies on either.
    let d_k = 8;
    let (q, k, v, w) = test_inputs_f64(1, d_k);

    for causal in [false, true] {
        let (o, p) = cpu_attention_fwd_f64(&q, &k, &v, 1, d_k, causal);
        assert_eq!(p, vec![1.0f64], "seq=1 softmax must be exactly 1.0");
        assert_eq!(o, v, "seq=1 output must equal V exactly");

        let (dq, dk, dv) = cpu_attention_bwd_f64(&q, &k, &v, &o, &p, &w, 1, d_k);
        for (idx, &g) in dq.iter().enumerate() {
            assert!(g.abs() <= 1e-12, "seq=1 dQ[{idx}] must be 0, got {g}");
        }
        for (idx, &g) in dk.iter().enumerate() {
            assert!(g.abs() <= 1e-12, "seq=1 dK[{idx}] must be 0, got {g}");
        }
        assert_close_f64(&dv, &w, 1e-15, 1e-15, "seq1/dV");
    }
}

// ---------------------------------------------------------------------------
// GPU: `_with_stats` forward variants — L-stats gate (Sprint 9.2)
// ---------------------------------------------------------------------------

/// Deterministic f32 inputs for the GPU tests (house pattern).
fn gpu_inputs_f32(seq_len: usize, d_k: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = seq_len * d_k;
    let q: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
    let k: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
    let v: Vec<f32> = (0..n).map(|i| ((i % 19) as f32 - 9.0) * 0.1).collect();
    (q, k, v)
}

/// f64 row-wise logsumexp of the scaled score matrix, computed on host
/// from the same f32 inputs the GPU sees.
fn cpu_logsumexp_rows_f64(
    q: &[f32],
    k: &[f32],
    seq_len: usize,
    d_k: usize,
    causal: bool,
) -> Vec<f64> {
    let scale = 1.0f64 / (d_k as f64).sqrt();
    let mut l_ref = vec![0.0f64; seq_len];
    for i in 0..seq_len {
        let lim = if causal { i + 1 } else { seq_len };
        let mut scores = Vec::with_capacity(lim);
        for j in 0..lim {
            let mut dot = 0.0f64;
            for d in 0..d_k {
                dot += q[i * d_k + d] as f64 * k[j * d_k + d] as f64;
            }
            scores.push(dot * scale);
        }
        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = scores.iter().map(|&s| (s - max).exp()).sum();
        l_ref[i] = max + sum.ln();
    }
    l_ref
}

fn check_with_stats(seq_len: usize, d_k: usize, causal: bool) {
    let device = KaioDevice::new(0).expect("GPU required");
    let (q_h, k_h, v_h) = gpu_inputs_f32(seq_len, d_k);

    let q = device.alloc_from(&q_h).unwrap();
    let k = device.alloc_from(&k_h).unwrap();
    let v = device.alloc_from(&v_h).unwrap();
    let mut out_plain = device.alloc_zeros::<f32>(seq_len * d_k).unwrap();
    let mut out_stats = device.alloc_zeros::<f32>(seq_len * d_k).unwrap();
    let mut stats = device.alloc_zeros::<f32>(seq_len).unwrap();

    let sl = seq_len as u32;
    let dk_u = d_k as u32;
    if causal {
        attention_flash_causal(&device, &q, &k, &v, &mut out_plain, sl, dk_u).unwrap();
        attention_flash_causal_with_stats(
            &device,
            &q,
            &k,
            &v,
            &mut out_stats,
            &mut stats,
            sl,
            dk_u,
        )
        .unwrap();
    } else {
        attention_flash(&device, &q, &k, &v, &mut out_plain, sl, dk_u).unwrap();
        attention_flash_with_stats(&device, &q, &k, &v, &mut out_stats, &mut stats, sl, dk_u)
            .unwrap();
    }

    let label = if causal { "causal" } else { "plain" };

    // Contract: the stats variant must not change the numerical output.
    // The zero-diff (bit-level) assertion holds today because both
    // kernels are built by the same toolchain in the same run; if a
    // toolchain or lowering change ever fails it, re-evaluate numerical
    // equivalence rather than blindly loosening.
    let plain_h = out_plain.to_host(&device).unwrap();
    let stats_out_h = out_stats.to_host(&device).unwrap();
    for idx in 0..seq_len * d_k {
        assert_eq!(
            plain_h[idx].to_bits(),
            stats_out_h[idx].to_bits(),
            "{label} {seq_len}x{d_k}: with_stats out differs from plain forward at {idx}"
        );
    }

    // L vs f64 logsumexp, dual tolerance — L can sit near zero (e.g.
    // seq_len = 1 with q·k ≈ 0) where pure relative error is undefined.
    let l_gpu = stats.to_host(&device).unwrap();
    let l_ref = cpu_logsumexp_rows_f64(&q_h, &k_h, seq_len, d_k, causal);
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    for i in 0..seq_len {
        let got = l_gpu[i] as f64;
        let expected = l_ref[i];
        let abs_err = (got - expected).abs();
        let rel_err = if expected.abs() > 1e-12 {
            abs_err / expected.abs()
        } else {
            abs_err
        };
        max_abs = max_abs.max(abs_err);
        max_rel = max_rel.max(rel_err);
        assert!(
            abs_err < 1e-5 || rel_err < 1e-5,
            "{label} {seq_len}x{d_k}: L[{i}] got {got}, expected {expected}, abs={abs_err:.2e}, rel={rel_err:.2e}"
        );
    }
    eprintln!("{label} {seq_len}x{d_k}: L max_abs={max_abs:.2e}, max_rel={max_rel:.2e}");

    // Property: Σ_j exp(S_ij − L_i) = 1 per row — the definition of the
    // normalizer, with S recomputed on host in f64.
    let scale = 1.0f64 / (d_k as f64).sqrt();
    for i in 0..seq_len {
        let lim = if causal { i + 1 } else { seq_len };
        let mut sum = 0.0f64;
        for j in 0..lim {
            let mut dot = 0.0f64;
            for d in 0..d_k {
                dot += q_h[i * d_k + d] as f64 * k_h[j * d_k + d] as f64;
            }
            sum += (dot * scale - l_gpu[i] as f64).exp();
        }
        assert!(
            (sum - 1.0).abs() < 1e-3,
            "{label} {seq_len}x{d_k}: rowsum property failed at row {i}: {sum}"
        );
    }

    // Determinism canary: identical inputs must produce bit-identical
    // stats (no atomics, fixed reduction order). The candle backward
    // recovers L by re-running this kernel, so its correctness rests on
    // exactly this property.
    let mut out2 = device.alloc_zeros::<f32>(seq_len * d_k).unwrap();
    let mut stats2 = device.alloc_zeros::<f32>(seq_len).unwrap();
    if causal {
        attention_flash_causal_with_stats(&device, &q, &k, &v, &mut out2, &mut stats2, sl, dk_u)
            .unwrap();
    } else {
        attention_flash_with_stats(&device, &q, &k, &v, &mut out2, &mut stats2, sl, dk_u).unwrap();
    }
    let stats2_h = stats2.to_host(&device).unwrap();
    for i in 0..seq_len {
        assert_eq!(
            l_gpu[i].to_bits(),
            stats2_h[i].to_bits(),
            "{label} {seq_len}x{d_k}: stats not deterministic at row {i}"
        );
    }
}

#[test]
#[ignore] // GPU required
fn with_stats_1x8() {
    check_with_stats(1, 8, false);
    check_with_stats(1, 8, true);
}

#[test]
#[ignore] // GPU required
fn with_stats_32x32() {
    check_with_stats(32, 32, false);
    check_with_stats(32, 32, true);
}

#[test]
#[ignore] // GPU required
fn with_stats_64x64() {
    check_with_stats(64, 64, false);
    check_with_stats(64, 64, true);
}

#[test]
#[ignore] // GPU required
fn with_stats_128x128() {
    check_with_stats(128, 128, false);
    check_with_stats(128, 128, true);
}

#[test]
#[ignore] // GPU required
fn with_stats_non_aligned_17x19() {
    check_with_stats(17, 19, false);
    check_with_stats(17, 19, true);
}

#[test]
#[ignore] // GPU required
fn with_stats_tile_boundary_257x32() {
    check_with_stats(257, 32, false);
    check_with_stats(257, 32, true);
}

// ---------------------------------------------------------------------------
// GPU: backward dK/dV kernel vs the f64 oracle (Sprint 9.2)
// ---------------------------------------------------------------------------

/// Upstream-gradient pattern for the GPU backward tests.
fn gpu_grad_f32(seq_len: usize, d_k: usize) -> Vec<f32> {
    (0..seq_len * d_k)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.2)
        .collect()
}

/// Tolerance gate for GPU f32 gradients vs the f64 oracle.
fn assert_grads_close(got: &[f32], expected: &[f64], label: &str) {
    let got_f64: Vec<f64> = got.iter().map(|&x| x as f64).collect();
    assert_close_f64(&got_f64, expected, 1e-3, 1e-2, label);
}

/// Runs the real GPU pipeline (forward `_with_stats` → preprocess →
/// dkdv) and compares dK/dV against the all-f64 analytical oracle.
fn check_dkdv(seq_len: usize, d_k: usize, causal: bool) {
    let device = KaioDevice::new(0).expect("GPU required");
    let (q_h, k_h, v_h) = gpu_inputs_f32(seq_len, d_k);
    let g_h = gpu_grad_f32(seq_len, d_k);

    let sl = seq_len as u32;
    let dk_u = d_k as u32;
    let n = seq_len * d_k;

    let q = device.alloc_from(&q_h).unwrap();
    let k = device.alloc_from(&k_h).unwrap();
    let v = device.alloc_from(&v_h).unwrap();
    let g = device.alloc_from(&g_h).unwrap();
    let mut out = device.alloc_zeros::<f32>(n).unwrap();
    let mut stats = device.alloc_zeros::<f32>(seq_len).unwrap();
    let mut d_buf = device.alloc_zeros::<f32>(seq_len).unwrap();
    let mut dk_gpu = device.alloc_zeros::<f32>(n).unwrap();
    let mut dv_gpu = device.alloc_zeros::<f32>(n).unwrap();

    if causal {
        attention_flash_causal_with_stats(&device, &q, &k, &v, &mut out, &mut stats, sl, dk_u)
            .unwrap();
    } else {
        attention_flash_with_stats(&device, &q, &k, &v, &mut out, &mut stats, sl, dk_u).unwrap();
    }
    attention_flash_bwd_preprocess(&device, &g, &out, &mut d_buf, sl, dk_u).unwrap();
    attention_flash_bwd_dkdv(
        &device,
        &g,
        &q,
        &k,
        &v,
        &stats,
        &d_buf,
        &mut dk_gpu,
        &mut dv_gpu,
        sl,
        dk_u,
        causal,
    )
    .unwrap();

    let label = if causal { "causal" } else { "plain" };

    // Intermediate diagnostic: the preprocess D buffer vs an f64 host
    // recompute from the GPU's own out — isolates preprocess bugs from
    // dkdv bugs when bisecting a failure.
    let out_h = out.to_host(&device).unwrap();
    let d_gpu_h = d_buf.to_host(&device).unwrap();
    let d_ref: Vec<f64> = (0..seq_len)
        .map(|i| {
            (0..d_k)
                .map(|d| g_h[i * d_k + d] as f64 * out_h[i * d_k + d] as f64)
                .sum()
        })
        .collect();
    let d_gpu_f64: Vec<f64> = d_gpu_h.iter().map(|&x| x as f64).collect();
    assert_close_f64(
        &d_gpu_f64,
        &d_ref,
        1e-4,
        1e-4,
        &format!("dkdv_{label}_{seq_len}x{d_k}/D"),
    );

    // Primary gate: dK/dV vs the all-f64 analytical oracle.
    let q64: Vec<f64> = q_h.iter().map(|&x| x as f64).collect();
    let k64: Vec<f64> = k_h.iter().map(|&x| x as f64).collect();
    let v64: Vec<f64> = v_h.iter().map(|&x| x as f64).collect();
    let g64: Vec<f64> = g_h.iter().map(|&x| x as f64).collect();
    let (o_ref, p_ref) = cpu_attention_fwd_f64(&q64, &k64, &v64, seq_len, d_k, causal);
    let (_, dk_ref, dv_ref) =
        cpu_attention_bwd_f64(&q64, &k64, &v64, &o_ref, &p_ref, &g64, seq_len, d_k);

    let dk_h = dk_gpu.to_host(&device).unwrap();
    let dv_h = dv_gpu.to_host(&device).unwrap();
    assert_grads_close(&dk_h, &dk_ref, &format!("dkdv_{label}_{seq_len}x{d_k}/dK"));
    assert_grads_close(&dv_h, &dv_ref, &format!("dkdv_{label}_{seq_len}x{d_k}/dV"));
}

#[test]
#[ignore] // GPU required
fn bwd_dkdv_32x32() {
    check_dkdv(32, 32, false);
    check_dkdv(32, 32, true);
}

#[test]
#[ignore] // GPU required
fn bwd_dkdv_64x64() {
    check_dkdv(64, 64, false);
    check_dkdv(64, 64, true);
}

#[test]
#[ignore] // GPU required
fn bwd_dkdv_128x128() {
    check_dkdv(128, 128, false);
    check_dkdv(128, 128, true);
}

#[test]
#[ignore] // GPU required
fn bwd_dkdv_non_aligned_17x19() {
    check_dkdv(17, 19, false);
    check_dkdv(17, 19, true);
}

// ---------------------------------------------------------------------------
// GPU: backward dQ kernel vs the f64 oracle (Sprint 9.2)
// ---------------------------------------------------------------------------

/// Runs the real GPU pipeline (forward `_with_stats` → preprocess →
/// dq) and compares dQ against the all-f64 analytical oracle.
fn check_dq(seq_len: usize, d_k: usize, causal: bool) {
    let device = KaioDevice::new(0).expect("GPU required");
    let (q_h, k_h, v_h) = gpu_inputs_f32(seq_len, d_k);
    let g_h = gpu_grad_f32(seq_len, d_k);

    let sl = seq_len as u32;
    let dk_u = d_k as u32;
    let n = seq_len * d_k;

    let q = device.alloc_from(&q_h).unwrap();
    let k = device.alloc_from(&k_h).unwrap();
    let v = device.alloc_from(&v_h).unwrap();
    let g = device.alloc_from(&g_h).unwrap();
    let mut out = device.alloc_zeros::<f32>(n).unwrap();
    let mut stats = device.alloc_zeros::<f32>(seq_len).unwrap();
    let mut d_buf = device.alloc_zeros::<f32>(seq_len).unwrap();
    let mut dq_gpu = device.alloc_zeros::<f32>(n).unwrap();

    if causal {
        attention_flash_causal_with_stats(&device, &q, &k, &v, &mut out, &mut stats, sl, dk_u)
            .unwrap();
    } else {
        attention_flash_with_stats(&device, &q, &k, &v, &mut out, &mut stats, sl, dk_u).unwrap();
    }
    attention_flash_bwd_preprocess(&device, &g, &out, &mut d_buf, sl, dk_u).unwrap();
    attention_flash_bwd_dq(
        &device,
        &g,
        &q,
        &k,
        &v,
        &stats,
        &d_buf,
        &mut dq_gpu,
        sl,
        dk_u,
        causal,
    )
    .unwrap();

    let label = if causal { "causal" } else { "plain" };

    let q64: Vec<f64> = q_h.iter().map(|&x| x as f64).collect();
    let k64: Vec<f64> = k_h.iter().map(|&x| x as f64).collect();
    let v64: Vec<f64> = v_h.iter().map(|&x| x as f64).collect();
    let g64: Vec<f64> = g_h.iter().map(|&x| x as f64).collect();
    let (o_ref, p_ref) = cpu_attention_fwd_f64(&q64, &k64, &v64, seq_len, d_k, causal);
    let (dq_ref, _, _) =
        cpu_attention_bwd_f64(&q64, &k64, &v64, &o_ref, &p_ref, &g64, seq_len, d_k);

    let dq_h = dq_gpu.to_host(&device).unwrap();
    assert_grads_close(&dq_h, &dq_ref, &format!("dq_{label}_{seq_len}x{d_k}/dQ"));
}

#[test]
#[ignore] // GPU required
fn bwd_dq_32x32() {
    check_dq(32, 32, false);
    check_dq(32, 32, true);
}

#[test]
#[ignore] // GPU required
fn bwd_dq_64x64() {
    check_dq(64, 64, false);
    check_dq(64, 64, true);
}

#[test]
#[ignore] // GPU required
fn bwd_dq_128x128() {
    check_dq(128, 128, false);
    check_dq(128, 128, true);
}

#[test]
#[ignore] // GPU required
fn bwd_dq_non_aligned_17x19() {
    check_dq(17, 19, false);
    check_dq(17, 19, true);
}

// ---------------------------------------------------------------------------
// GPU: full backward via the public API vs the f64 oracle (Sprint 9.2)
// ---------------------------------------------------------------------------

/// Full pipeline through the public API: `_with_stats` forward →
/// `attention_flash_bwd[_causal]` → all three gradients vs the oracle.
/// `g_h` is the upstream gradient (caller controls its distribution).
fn run_bwd_full(seq_len: usize, d_k: usize, causal: bool, g_h: &[f32], label: &str) {
    let device = KaioDevice::new(0).expect("GPU required");
    let (q_h, k_h, v_h) = gpu_inputs_f32(seq_len, d_k);

    let sl = seq_len as u32;
    let dk_u = d_k as u32;
    let n = seq_len * d_k;

    let q = device.alloc_from(&q_h).unwrap();
    let k = device.alloc_from(&k_h).unwrap();
    let v = device.alloc_from(&v_h).unwrap();
    let g = device.alloc_from(g_h).unwrap();
    let mut out = device.alloc_zeros::<f32>(n).unwrap();
    let mut stats = device.alloc_zeros::<f32>(seq_len).unwrap();
    let mut dq_gpu = device.alloc_zeros::<f32>(n).unwrap();
    let mut dk_gpu = device.alloc_zeros::<f32>(n).unwrap();
    let mut dv_gpu = device.alloc_zeros::<f32>(n).unwrap();

    if causal {
        attention_flash_causal_with_stats(&device, &q, &k, &v, &mut out, &mut stats, sl, dk_u)
            .unwrap();
        attention_flash_bwd_causal(
            &device,
            &g,
            &q,
            &k,
            &v,
            &out,
            &stats,
            &mut dq_gpu,
            &mut dk_gpu,
            &mut dv_gpu,
            sl,
            dk_u,
        )
        .unwrap();
    } else {
        attention_flash_with_stats(&device, &q, &k, &v, &mut out, &mut stats, sl, dk_u).unwrap();
        attention_flash_bwd(
            &device,
            &g,
            &q,
            &k,
            &v,
            &out,
            &stats,
            &mut dq_gpu,
            &mut dk_gpu,
            &mut dv_gpu,
            sl,
            dk_u,
        )
        .unwrap();
    }

    let q64: Vec<f64> = q_h.iter().map(|&x| x as f64).collect();
    let k64: Vec<f64> = k_h.iter().map(|&x| x as f64).collect();
    let v64: Vec<f64> = v_h.iter().map(|&x| x as f64).collect();
    let g64: Vec<f64> = g_h.iter().map(|&x| x as f64).collect();
    let (o_ref, p_ref) = cpu_attention_fwd_f64(&q64, &k64, &v64, seq_len, d_k, causal);
    let (dq_ref, dk_ref, dv_ref) =
        cpu_attention_bwd_f64(&q64, &k64, &v64, &o_ref, &p_ref, &g64, seq_len, d_k);

    let dq_h = dq_gpu.to_host(&device).unwrap();
    let dk_h = dk_gpu.to_host(&device).unwrap();
    let dv_h = dv_gpu.to_host(&device).unwrap();
    assert_grads_close(&dq_h, &dq_ref, &format!("{label}/dQ"));
    assert_grads_close(&dk_h, &dk_ref, &format!("{label}/dK"));
    assert_grads_close(&dv_h, &dv_ref, &format!("{label}/dV"));
}

fn check_bwd_full(seq_len: usize, d_k: usize, causal: bool) {
    let g_h = gpu_grad_f32(seq_len, d_k);
    let label = format!(
        "bwd_full_{}_{seq_len}x{d_k}",
        if causal { "causal" } else { "plain" }
    );
    run_bwd_full(seq_len, d_k, causal, &g_h, &label);
}

/// Closed-form gate at `seq_len = 1`: `dV = grad_out`, `dQ = 0`,
/// `dK = 0`. Wiring errors (wrong sign, swapped buffers, missing
/// scale) produce O(0.1+) violations here; the small tolerances below
/// only absorb FP noise — `dP_11` (serial per-thread sum) and `D_1`
/// (block tree reduction) sum in different orders, so `dS` is ~1e-7
/// rather than exactly zero.
#[test]
#[ignore] // GPU required
fn bwd_seq1_closed_form_gate() {
    let device = KaioDevice::new(0).expect("GPU required");
    let d_k = 16usize;
    let (q_h, k_h, v_h) = gpu_inputs_f32(1, d_k);
    let g_h = gpu_grad_f32(1, d_k);

    for causal in [false, true] {
        let q = device.alloc_from(&q_h).unwrap();
        let k = device.alloc_from(&k_h).unwrap();
        let v = device.alloc_from(&v_h).unwrap();
        let g = device.alloc_from(&g_h).unwrap();
        let mut out = device.alloc_zeros::<f32>(d_k).unwrap();
        let mut stats = device.alloc_zeros::<f32>(1).unwrap();
        let mut dq = device.alloc_zeros::<f32>(d_k).unwrap();
        let mut dk = device.alloc_zeros::<f32>(d_k).unwrap();
        let mut dv = device.alloc_zeros::<f32>(d_k).unwrap();

        if causal {
            attention_flash_causal_with_stats(&device, &q, &k, &v, &mut out, &mut stats, 1, 16)
                .unwrap();
            attention_flash_bwd_causal(
                &device, &g, &q, &k, &v, &out, &stats, &mut dq, &mut dk, &mut dv, 1, 16,
            )
            .unwrap();
        } else {
            attention_flash_with_stats(&device, &q, &k, &v, &mut out, &mut stats, 1, 16).unwrap();
            attention_flash_bwd(
                &device, &g, &q, &k, &v, &out, &stats, &mut dq, &mut dk, &mut dv, 1, 16,
            )
            .unwrap();
        }

        let label = if causal { "causal" } else { "plain" };
        let dq_h = dq.to_host(&device).unwrap();
        let dk_h = dk.to_host(&device).unwrap();
        let dv_h = dv.to_host(&device).unwrap();
        for d in 0..d_k {
            assert!(
                (dv_h[d] - g_h[d]).abs() < 1e-6,
                "seq1 {label}: dV[{d}] = {} != grad {}",
                dv_h[d],
                g_h[d]
            );
            assert!(
                dq_h[d].abs() < 1e-5,
                "seq1 {label}: dQ[{d}] = {} != 0",
                dq_h[d]
            );
            assert!(
                dk_h[d].abs() < 1e-5,
                "seq1 {label}: dK[{d}] = {} != 0",
                dk_h[d]
            );
        }
    }
}

#[test]
#[ignore] // GPU required
fn bwd_full_seq32() {
    for d_k in [32, 64, 128] {
        check_bwd_full(32, d_k, false);
        check_bwd_full(32, d_k, true);
    }
}

#[test]
#[ignore] // GPU required
fn bwd_full_seq64() {
    for d_k in [32, 64, 128] {
        check_bwd_full(64, d_k, false);
        check_bwd_full(64, d_k, true);
    }
}

#[test]
#[ignore] // GPU required
fn bwd_full_seq128() {
    for d_k in [32, 64, 128] {
        check_bwd_full(128, d_k, false);
        check_bwd_full(128, d_k, true);
    }
}

#[test]
#[ignore] // GPU required
fn bwd_full_non_aligned_17x19() {
    check_bwd_full(17, 19, false);
    check_bwd_full(17, 19, true);
}

#[test]
#[ignore] // GPU required
fn bwd_full_tile_boundary_257x32() {
    check_bwd_full(257, 32, false);
    check_bwd_full(257, 32, true);
}

/// Upstream-gradient rows spanning orders of magnitude so `D_i` varies
/// widely across rows — exercises the cancellation path in
/// `dS = P·(dP − D)` that uniform-magnitude data cannot reach.
#[test]
#[ignore] // GPU required
fn bwd_full_wide_range_dout() {
    let seq_len = 64usize;
    let d_k = 64usize;
    let row_scale: Vec<f32> = (0..seq_len)
        .map(|i| match i % 6 {
            0 => 1e-3,
            1 => 1.0,
            2 => 5e1,
            3 => 1e-2,
            4 => 1e2,
            _ => 0.5,
        })
        .collect();
    let g_h: Vec<f32> = (0..seq_len * d_k)
        .map(|i| {
            let base = ((i % 11) as f32 - 5.0) * 0.2;
            base * row_scale[i / d_k]
        })
        .collect();
    run_bwd_full(seq_len, d_k, false, &g_h, "bwd_wide_range_plain_64x64");
    run_bwd_full(seq_len, d_k, true, &g_h, "bwd_wide_range_causal_64x64");
}

/// Finite-difference smoke on the GPU at a tiny shape — a sanity layer
/// independent of the analytical oracle, not the primary gate.
#[test]
#[ignore] // GPU required
fn bwd_fd_smoke_8x16() {
    let seq_len = 8usize;
    let d_k = 16usize;
    let (q_h, k_h, v_h) = gpu_inputs_f32(seq_len, d_k);
    let g_h = gpu_grad_f32(seq_len, d_k);

    let device = KaioDevice::new(0).expect("GPU required");
    let sl = seq_len as u32;
    let n = seq_len * d_k;

    for causal in [false, true] {
        let q = device.alloc_from(&q_h).unwrap();
        let k = device.alloc_from(&k_h).unwrap();
        let v = device.alloc_from(&v_h).unwrap();
        let g = device.alloc_from(&g_h).unwrap();
        let mut out = device.alloc_zeros::<f32>(n).unwrap();
        let mut stats = device.alloc_zeros::<f32>(seq_len).unwrap();
        let mut dq = device.alloc_zeros::<f32>(n).unwrap();
        let mut dk = device.alloc_zeros::<f32>(n).unwrap();
        let mut dv = device.alloc_zeros::<f32>(n).unwrap();

        if causal {
            attention_flash_causal_with_stats(&device, &q, &k, &v, &mut out, &mut stats, sl, 16)
                .unwrap();
            attention_flash_bwd_causal(
                &device, &g, &q, &k, &v, &out, &stats, &mut dq, &mut dk, &mut dv, sl, 16,
            )
            .unwrap();
        } else {
            attention_flash_with_stats(&device, &q, &k, &v, &mut out, &mut stats, sl, 16).unwrap();
            attention_flash_bwd(
                &device, &g, &q, &k, &v, &out, &stats, &mut dq, &mut dk, &mut dv, sl, 16,
            )
            .unwrap();
        }

        let q64: Vec<f64> = q_h.iter().map(|&x| x as f64).collect();
        let k64: Vec<f64> = k_h.iter().map(|&x| x as f64).collect();
        let v64: Vec<f64> = v_h.iter().map(|&x| x as f64).collect();
        let g64: Vec<f64> = g_h.iter().map(|&x| x as f64).collect();
        let h = 1e-5f64;
        let dq_fd = fd_grad_f64(&q64, &k64, &v64, &g64, seq_len, d_k, causal, 0, h);
        let dk_fd = fd_grad_f64(&q64, &k64, &v64, &g64, seq_len, d_k, causal, 1, h);
        let dv_fd = fd_grad_f64(&q64, &k64, &v64, &g64, seq_len, d_k, causal, 2, h);

        let label = if causal { "fd_causal" } else { "fd_plain" };
        assert_grads_close(
            &dq.to_host(&device).unwrap(),
            &dq_fd,
            &format!("{label}/dQ"),
        );
        assert_grads_close(
            &dk.to_host(&device).unwrap(),
            &dk_fd,
            &format!("{label}/dK"),
        );
        assert_grads_close(
            &dv.to_host(&device).unwrap(),
            &dv_fd,
            &format!("{label}/dV"),
        );
    }
}

// ============================================================================
// Validation-error tests — the per-kernel building blocks must reject
// bad dims/buffers before launch, like every other launch wrapper in
// kaio-ops. A short buffer here is otherwise a GPU out-of-bounds
// write; an oversized d_k is a silently truncated reduction.
// ============================================================================

/// `d_k > 256` must be rejected by the preprocess helper specifically:
/// its kernel covers dims with one 256-thread block per row, so without
/// the guard a 512-dim call computes `D_i` over the first 256 dims only
/// — silently wrong, no fault. All buffers here are correctly sized for
/// `d_k = 512`, so the d_k bound is the only check that can fire.
#[test]
#[ignore] // GPU required
fn bwd_preprocess_rejects_dk_over_256() {
    let device = KaioDevice::new(0).expect("GPU required");
    let seq_len = 2u32;
    let d_k = 512u32;
    let n = (seq_len * d_k) as usize;
    let g = device.alloc_zeros::<f32>(n).unwrap();
    let o = device.alloc_zeros::<f32>(n).unwrap();
    let mut d_buf = device.alloc_zeros::<f32>(seq_len as usize).unwrap();
    let err =
        attention_flash_bwd_preprocess(&device, &g, &o, &mut d_buf, seq_len, d_k).unwrap_err();
    match err {
        KaioError::InvalidConfig(msg) => {
            assert!(msg.contains("d_k <= 256"), "got: {msg}")
        }
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}

/// `d_buf` is the one buffer the orchestrators never validate (they
/// allocate it exact-size internally) — the helper must.
#[test]
#[ignore] // GPU required
fn bwd_preprocess_rejects_short_d_buf() {
    let device = KaioDevice::new(0).expect("GPU required");
    let seq_len = 8u32;
    let d_k = 16u32;
    let n = (seq_len * d_k) as usize;
    let g = device.alloc_zeros::<f32>(n).unwrap();
    let o = device.alloc_zeros::<f32>(n).unwrap();
    let mut d_buf = device.alloc_zeros::<f32>(seq_len as usize - 1).unwrap();
    let err =
        attention_flash_bwd_preprocess(&device, &g, &o, &mut d_buf, seq_len, d_k).unwrap_err();
    match err {
        KaioError::InvalidConfig(msg) => {
            assert!(msg.contains("d_buf buffer too small"), "got: {msg}")
        }
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}

#[test]
#[ignore] // GPU required
fn bwd_dkdv_rejects_short_dk_buffer() {
    let device = KaioDevice::new(0).expect("GPU required");
    let seq_len = 8u32;
    let d_k = 16u32;
    let n = (seq_len * d_k) as usize;
    let g = device.alloc_zeros::<f32>(n).unwrap();
    let q = device.alloc_zeros::<f32>(n).unwrap();
    let k = device.alloc_zeros::<f32>(n).unwrap();
    let v = device.alloc_zeros::<f32>(n).unwrap();
    let stats = device.alloc_zeros::<f32>(seq_len as usize).unwrap();
    let d_buf = device.alloc_zeros::<f32>(seq_len as usize).unwrap();
    let mut dk = device.alloc_zeros::<f32>(n - 1).unwrap();
    let mut dv = device.alloc_zeros::<f32>(n).unwrap();
    let err = attention_flash_bwd_dkdv(
        &device, &g, &q, &k, &v, &stats, &d_buf, &mut dk, &mut dv, seq_len, d_k, false,
    )
    .unwrap_err();
    match err {
        KaioError::InvalidConfig(msg) => {
            assert!(msg.contains("dK buffer too small"), "got: {msg}")
        }
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}

#[test]
#[ignore] // GPU required
fn bwd_dq_rejects_short_dq_buffer() {
    let device = KaioDevice::new(0).expect("GPU required");
    let seq_len = 8u32;
    let d_k = 16u32;
    let n = (seq_len * d_k) as usize;
    let g = device.alloc_zeros::<f32>(n).unwrap();
    let q = device.alloc_zeros::<f32>(n).unwrap();
    let k = device.alloc_zeros::<f32>(n).unwrap();
    let v = device.alloc_zeros::<f32>(n).unwrap();
    let stats = device.alloc_zeros::<f32>(seq_len as usize).unwrap();
    let d_buf = device.alloc_zeros::<f32>(seq_len as usize).unwrap();
    let mut dq = device.alloc_zeros::<f32>(n - 1).unwrap();
    let err = attention_flash_bwd_dq(
        &device, &g, &q, &k, &v, &stats, &d_buf, &mut dq, seq_len, d_k, true,
    )
    .unwrap_err();
    match err {
        KaioError::InvalidConfig(msg) => {
            assert!(msg.contains("dQ buffer too small"), "got: {msg}")
        }
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}

#[test]
#[ignore] // GPU required
fn bwd_dq_rejects_zero_seq_len() {
    let device = KaioDevice::new(0).expect("GPU required");
    let g = device.alloc_zeros::<f32>(1).unwrap();
    let q = device.alloc_zeros::<f32>(1).unwrap();
    let k = device.alloc_zeros::<f32>(1).unwrap();
    let v = device.alloc_zeros::<f32>(1).unwrap();
    let stats = device.alloc_zeros::<f32>(1).unwrap();
    let d_buf = device.alloc_zeros::<f32>(1).unwrap();
    let mut dq = device.alloc_zeros::<f32>(1).unwrap();
    let err = attention_flash_bwd_dq(
        &device, &g, &q, &k, &v, &stats, &d_buf, &mut dq, 0, 16, false,
    )
    .unwrap_err();
    match err {
        KaioError::InvalidConfig(msg) => {
            assert!(msg.contains("non-zero"), "got: {msg}")
        }
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}
