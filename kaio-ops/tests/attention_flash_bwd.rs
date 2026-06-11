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
