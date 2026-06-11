//! GPU tests for the FlashAttention candle bindings (Sprint 9.2).
//!
//! Separate from `candle_gpu_roundtrip.rs`: the attention family brings
//! its own f64 oracle apparatus (gradient checks land alongside the
//! backward), and keeping it here keeps both files bisectable.
//!
//! All tests require an NVIDIA GPU and are `#[ignore]`-gated; run with
//! `cargo test --features cuda -- --ignored`.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use candle_core::{Device, Tensor};
use kaio::prelude::KaioDevice;

// --- deterministic inputs (house pattern) ---

fn inputs_f32(seq_len: usize, d_k: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = seq_len * d_k;
    let q: Vec<f32> = (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
    let k: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
    let v: Vec<f32> = (0..n).map(|i| ((i % 19) as f32 - 9.0) * 0.1).collect();
    (q, k, v)
}

// --- bit-exact forward: candle path vs direct kaio-ops call ---

/// The candle binding must produce bit-identical output to a direct
/// `kaio_ops::attention_flash[_causal]` call on the same input bits —
/// the bridge adds no arithmetic.
fn bit_exact_attention_flash(seq_len: usize, d_k: usize, causal: bool) -> anyhow::Result<()> {
    let candle_dev = Device::new_cuda(0)?;
    let kaio_dev = Arc::new(KaioDevice::new(0)?);
    let (q_h, k_h, v_h) = inputs_f32(seq_len, d_k);

    // Candle path.
    let q = Tensor::from_vec(q_h.clone(), (seq_len, d_k), &candle_dev)?;
    let k = Tensor::from_vec(k_h.clone(), (seq_len, d_k), &candle_dev)?;
    let v = Tensor::from_vec(v_h.clone(), (seq_len, d_k), &candle_dev)?;
    let out_candle = if causal {
        kaio_candle::attention_flash_causal(&kaio_dev, &q, &k, &v)?
    } else {
        kaio_candle::attention_flash(&kaio_dev, &q, &k, &v)?
    };
    let out_candle_h: Vec<f32> = out_candle.flatten_all()?.to_vec1::<f32>()?;

    // Direct kaio-ops path.
    let q_buf = kaio_dev.alloc_from(&q_h)?;
    let k_buf = kaio_dev.alloc_from(&k_h)?;
    let v_buf = kaio_dev.alloc_from(&v_h)?;
    let mut out_buf = kaio_dev.alloc_zeros::<f32>(seq_len * d_k)?;
    if causal {
        kaio_ops::attention_flash_causal(
            &kaio_dev,
            &q_buf,
            &k_buf,
            &v_buf,
            &mut out_buf,
            seq_len as u32,
            d_k as u32,
        )?;
    } else {
        kaio_ops::attention_flash(
            &kaio_dev,
            &q_buf,
            &k_buf,
            &v_buf,
            &mut out_buf,
            seq_len as u32,
            d_k as u32,
        )?;
    }
    let out_kaio_h: Vec<f32> = out_buf.to_host(&kaio_dev)?;

    for (idx, (c, d)) in out_candle_h.iter().zip(out_kaio_h.iter()).enumerate() {
        assert_eq!(
            c.to_bits(),
            d.to_bits(),
            "bit mismatch at {idx}: candle {c} vs direct {d}"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn attention_flash_bit_exact_32x32() -> anyhow::Result<()> {
    bit_exact_attention_flash(32, 32, false)
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn attention_flash_bit_exact_64x64() -> anyhow::Result<()> {
    bit_exact_attention_flash(64, 64, false)
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn attention_flash_bit_exact_non_aligned_17x19() -> anyhow::Result<()> {
    bit_exact_attention_flash(17, 19, false)
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn attention_flash_causal_bit_exact_32x32() -> anyhow::Result<()> {
    bit_exact_attention_flash(32, 32, true)
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn attention_flash_causal_bit_exact_64x64() -> anyhow::Result<()> {
    bit_exact_attention_flash(64, 64, true)
}

// --- rejection tests ---

#[test]
#[ignore = "requires NVIDIA GPU"]
fn attention_flash_rejects_f16_dtype() -> anyhow::Result<()> {
    let candle_dev = Device::new_cuda(0)?;
    let kaio_dev = Arc::new(KaioDevice::new(0)?);

    let q = Tensor::ones((32, 32), candle_core::DType::F16, &candle_dev)?;
    let k = Tensor::ones((32, 32), candle_core::DType::F16, &candle_dev)?;
    let v = Tensor::ones((32, 32), candle_core::DType::F16, &candle_dev)?;

    let err = kaio_candle::attention_flash(&kaio_dev, &q, &k, &v).expect_err("must reject f16");
    let msg = format!("{err}");
    assert!(
        msg.to_lowercase().contains("f32") || msg.to_lowercase().contains("dtype"),
        "expected f32 dtype rejection message, got: {msg}"
    );
    Ok(())
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn attention_flash_rejects_noncontiguous() -> anyhow::Result<()> {
    let candle_dev = Device::new_cuda(0)?;
    let kaio_dev = Arc::new(KaioDevice::new(0)?);

    let q = Tensor::ones((32, 32), candle_core::DType::F32, &candle_dev)?;
    let k = Tensor::ones((32, 32), candle_core::DType::F32, &candle_dev)?;
    let v = Tensor::ones((32, 32), candle_core::DType::F32, &candle_dev)?;

    let q_nc = q.t()?;
    assert!(
        !q_nc.is_contiguous(),
        "setup sanity: .t() should be non-contiguous"
    );

    let err = kaio_candle::attention_flash(&kaio_dev, &q_nc, &k, &v).expect_err("must reject");
    let msg = format!("{err}");
    assert!(
        msg.contains("contiguous"),
        "expected contiguity rejection message, got: {msg}"
    );
    Ok(())
}

/// This binding is self-attention only: K with a different seq_len than
/// Q (a shape `attention_tc` would accept) must be loudly rejected.
#[test]
#[ignore = "requires NVIDIA GPU"]
fn attention_flash_rejects_seq_mismatch() -> anyhow::Result<()> {
    let candle_dev = Device::new_cuda(0)?;
    let kaio_dev = Arc::new(KaioDevice::new(0)?);

    let q = Tensor::ones((32, 32), candle_core::DType::F32, &candle_dev)?;
    let k = Tensor::ones((16, 32), candle_core::DType::F32, &candle_dev)?;
    let v = Tensor::ones((16, 32), candle_core::DType::F32, &candle_dev)?;

    let err = kaio_candle::attention_flash(&kaio_dev, &q, &k, &v).expect_err("must reject");
    let msg = format!("{err}");
    assert!(
        msg.contains("self-attention") && msg.contains("attention_tc"),
        "expected self-attention shape rejection with attention_tc hint, got: {msg}"
    );
    Ok(())
}

/// V with a different head dim than Q/K (`d_v != d_k`) must be rejected
/// — the flash kernels require d_v == d_k.
#[test]
#[ignore = "requires NVIDIA GPU"]
fn attention_flash_causal_rejects_dv_mismatch() -> anyhow::Result<()> {
    let candle_dev = Device::new_cuda(0)?;
    let kaio_dev = Arc::new(KaioDevice::new(0)?);

    let q = Tensor::ones((32, 32), candle_core::DType::F32, &candle_dev)?;
    let k = Tensor::ones((32, 32), candle_core::DType::F32, &candle_dev)?;
    let v = Tensor::ones((32, 16), candle_core::DType::F32, &candle_dev)?;

    let err = kaio_candle::attention_flash_causal(&kaio_dev, &q, &k, &v).expect_err("must reject");
    let msg = format!("{err}");
    assert!(
        msg.contains("self-attention"),
        "expected self-attention shape rejection, got: {msg}"
    );
    Ok(())
}

/// Rank-3 inputs must be rejected with the rank-2 hint (multi-head
/// callers flatten first).
#[test]
#[ignore = "requires NVIDIA GPU"]
fn attention_flash_rejects_rank3() -> anyhow::Result<()> {
    let candle_dev = Device::new_cuda(0)?;
    let kaio_dev = Arc::new(KaioDevice::new(0)?);

    let q = Tensor::ones((2, 32, 32), candle_core::DType::F32, &candle_dev)?;
    let k = Tensor::ones((2, 32, 32), candle_core::DType::F32, &candle_dev)?;
    let v = Tensor::ones((2, 32, 32), candle_core::DType::F32, &candle_dev)?;

    let err = kaio_candle::attention_flash(&kaio_dev, &q, &k, &v).expect_err("must reject");
    let msg = format!("{err}");
    assert!(
        msg.contains("rank-2") || msg.contains("rank 2"),
        "expected rank-2 rejection message, got: {msg}"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Gradient correctness (backward via candle autograd)
// ---------------------------------------------------------------------------
// CPU f64 analytical reference — host-side copy of the oracle that
// gates the kaio-ops kernels in kaio-ops/tests/attention_flash_bwd.rs
// (that file also self-checks the identities against f64 finite
// differences before anything trusts them).

fn cpu_attention_fwd_f64(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    seq_len: usize,
    d_k: usize,
    causal: bool,
) -> (Vec<f64>, Vec<f64>) {
    let scale = 1.0f64 / (d_k as f64).sqrt();
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

#[allow(clippy::too_many_arguments)]
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
    let mut dd = vec![0.0f64; seq_len];
    for i in 0..seq_len {
        let mut acc = 0.0f64;
        for d in 0..d_k {
            acc += d_out[i * d_k + d] * o[i * d_k + d];
        }
        dd[i] = acc;
    }
    let mut ds = vec![0.0f64; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            ds[i * seq_len + j] = p[i * seq_len + j] * (dp[i * seq_len + j] - dd[i]);
        }
    }
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

fn assert_grad_close(got: &[f32], expected: &[f64], label: &str) {
    assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
    for (idx, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let g = g as f64;
        let abs_err = (g - e).abs();
        let rel_err = if e.abs() > 1e-12 {
            abs_err / e.abs()
        } else {
            abs_err
        };
        assert!(
            rel_err < 1e-2 || abs_err < 1e-3,
            "{label}: grad mismatch at [{idx}]: got={g}, expected={e}, \
             rel_err={rel_err:.4e}, abs_err={abs_err:.4e}"
        );
    }
}

/// Builds a candle graph `loss = sum(W ∘ attention_flash(Q, K, V))`
/// (`W = ones` for the unweighted variant, so `dO = W` either way),
/// runs `.backward()`, and checks all three input gradients against
/// the CPU f64 analytical oracle.
fn gradient_check_attention_flash(
    seq_len: usize,
    d_k: usize,
    causal: bool,
    weighted: bool,
) -> anyhow::Result<()> {
    let candle_dev = Device::new_cuda(0)?;
    let kaio_dev = Arc::new(KaioDevice::new(0)?);
    let (q_h, k_h, v_h) = inputs_f32(seq_len, d_k);
    let w_h: Vec<f32> = if weighted {
        (0..seq_len * d_k)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
            .collect()
    } else {
        vec![1.0f32; seq_len * d_k]
    };

    let q = candle_core::Var::from_vec(q_h.clone(), (seq_len, d_k), &candle_dev)?;
    let k = candle_core::Var::from_vec(k_h.clone(), (seq_len, d_k), &candle_dev)?;
    let v = candle_core::Var::from_vec(v_h.clone(), (seq_len, d_k), &candle_dev)?;

    let out = if causal {
        kaio_candle::attention_flash_causal(&kaio_dev, q.as_tensor(), k.as_tensor(), v.as_tensor())?
    } else {
        kaio_candle::attention_flash(&kaio_dev, q.as_tensor(), k.as_tensor(), v.as_tensor())?
    };
    let w = Tensor::from_vec(w_h.clone(), (seq_len, d_k), &candle_dev)?;
    let loss = (out * w)?.sum_all()?;

    let grads = loss.backward()?;
    let grad_q = grads.get(q.as_tensor()).expect("Q should have gradient");
    let grad_k = grads.get(k.as_tensor()).expect("K should have gradient");
    let grad_v = grads.get(v.as_tensor()).expect("V should have gradient");

    let gq: Vec<f32> = grad_q.flatten_all()?.to_vec1::<f32>()?;
    let gk: Vec<f32> = grad_k.flatten_all()?.to_vec1::<f32>()?;
    let gv: Vec<f32> = grad_v.flatten_all()?.to_vec1::<f32>()?;

    let q64: Vec<f64> = q_h.iter().map(|&x| x as f64).collect();
    let k64: Vec<f64> = k_h.iter().map(|&x| x as f64).collect();
    let v64: Vec<f64> = v_h.iter().map(|&x| x as f64).collect();
    let w64: Vec<f64> = w_h.iter().map(|&x| x as f64).collect();
    let (o_ref, p_ref) = cpu_attention_fwd_f64(&q64, &k64, &v64, seq_len, d_k, causal);
    let (dq_ref, dk_ref, dv_ref) =
        cpu_attention_bwd_f64(&q64, &k64, &v64, &o_ref, &p_ref, &w64, seq_len, d_k);

    let label = format!(
        "attention_flash{}{}_{seq_len}x{d_k}",
        if causal { "_causal" } else { "" },
        if weighted { "_weighted" } else { "" }
    );
    assert_grad_close(&gq, &dq_ref, &format!("{label}/dQ"));
    assert_grad_close(&gk, &dk_ref, &format!("{label}/dK"));
    assert_grad_close(&gv, &dv_ref, &format!("{label}/dV"));
    Ok(())
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn attention_flash_backward_32x32() -> anyhow::Result<()> {
    gradient_check_attention_flash(32, 32, false, false)
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn attention_flash_backward_64x64() -> anyhow::Result<()> {
    gradient_check_attention_flash(64, 64, false, false)
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn attention_flash_backward_128x128() -> anyhow::Result<()> {
    gradient_check_attention_flash(128, 128, false, false)
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn attention_flash_backward_weighted_64x64() -> anyhow::Result<()> {
    gradient_check_attention_flash(64, 64, false, true)
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn attention_flash_causal_backward_32x32() -> anyhow::Result<()> {
    gradient_check_attention_flash(32, 32, true, false)
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn attention_flash_causal_backward_64x64() -> anyhow::Result<()> {
    gradient_check_attention_flash(64, 64, true, false)
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn attention_flash_causal_backward_128x128() -> anyhow::Result<()> {
    gradient_check_attention_flash(128, 128, true, false)
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn attention_flash_causal_backward_weighted_64x64() -> anyhow::Result<()> {
    gradient_check_attention_flash(64, 64, true, true)
}

/// Closed-form sanity through the full candle graph at `seq_len = 1`:
/// `dV = W` (the loss weight), `dQ = dK = 0` up to FP noise.
#[test]
#[ignore = "requires NVIDIA GPU"]
fn attention_flash_backward_seq1_closed_form() -> anyhow::Result<()> {
    let candle_dev = Device::new_cuda(0)?;
    let kaio_dev = Arc::new(KaioDevice::new(0)?);
    let d_k = 16usize;
    let (q_h, k_h, v_h) = inputs_f32(1, d_k);
    let w_h: Vec<f32> = (0..d_k).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();

    for causal in [false, true] {
        let q = candle_core::Var::from_vec(q_h.clone(), (1, d_k), &candle_dev)?;
        let k = candle_core::Var::from_vec(k_h.clone(), (1, d_k), &candle_dev)?;
        let v = candle_core::Var::from_vec(v_h.clone(), (1, d_k), &candle_dev)?;

        let out = if causal {
            kaio_candle::attention_flash_causal(
                &kaio_dev,
                q.as_tensor(),
                k.as_tensor(),
                v.as_tensor(),
            )?
        } else {
            kaio_candle::attention_flash(&kaio_dev, q.as_tensor(), k.as_tensor(), v.as_tensor())?
        };
        let w = Tensor::from_vec(w_h.clone(), (1, d_k), &candle_dev)?;
        let loss = (out * w)?.sum_all()?;
        let grads = loss.backward()?;

        let gq: Vec<f32> = grads
            .get(q.as_tensor())
            .expect("Q grad")
            .flatten_all()?
            .to_vec1::<f32>()?;
        let gk: Vec<f32> = grads
            .get(k.as_tensor())
            .expect("K grad")
            .flatten_all()?
            .to_vec1::<f32>()?;
        let gv: Vec<f32> = grads
            .get(v.as_tensor())
            .expect("V grad")
            .flatten_all()?
            .to_vec1::<f32>()?;

        let label = if causal { "causal" } else { "plain" };
        for d in 0..d_k {
            assert!(
                (gv[d] - w_h[d]).abs() < 1e-6,
                "seq1 {label}: dV[{d}] = {} != {}",
                gv[d],
                w_h[d]
            );
            assert!(gq[d].abs() < 1e-5, "seq1 {label}: dQ[{d}] = {} != 0", gq[d]);
            assert!(gk[d].abs() < 1e-5, "seq1 {label}: dK[{d}] = {} != 0", gk[d]);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Path equivalence: candle autograd vs direct kaio-ops backward
// ---------------------------------------------------------------------------

/// The gradient tests above check the candle path against the f64
/// oracle, and the kaio-ops suite checks the direct path against the
/// same oracle — so the two paths only agree transitively, within the
/// oracle tolerance. This pins them to each other bit-exactly, so a
/// binding orchestration bug (e.g. the wrong buffer wired as `out` or
/// `stats` in the backward call) cannot hide inside that tolerance.
/// Bit-exactness is the right bar: the binding adds no arithmetic of
/// its own, the `_with_stats` recompute is bit-deterministic (locked
/// by the determinism canary in the kaio-ops suite), and the upstream
/// gradient of `sum(W ∘ out)` is `W` exactly (`1.0 * x == x`).
fn bwd_path_equivalence(seq_len: usize, d_k: usize, causal: bool) -> anyhow::Result<()> {
    let candle_dev = Device::new_cuda(0)?;
    let kaio_dev = Arc::new(KaioDevice::new(0)?);
    let (q_h, k_h, v_h) = inputs_f32(seq_len, d_k);
    // Non-trivial upstream gradient: loss = sum(W ∘ out) makes dO = W.
    let w_h: Vec<f32> = (0..seq_len * d_k)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
        .collect();

    // Candle autograd path.
    let q = candle_core::Var::from_vec(q_h.clone(), (seq_len, d_k), &candle_dev)?;
    let k = candle_core::Var::from_vec(k_h.clone(), (seq_len, d_k), &candle_dev)?;
    let v = candle_core::Var::from_vec(v_h.clone(), (seq_len, d_k), &candle_dev)?;
    let out = if causal {
        kaio_candle::attention_flash_causal(&kaio_dev, q.as_tensor(), k.as_tensor(), v.as_tensor())?
    } else {
        kaio_candle::attention_flash(&kaio_dev, q.as_tensor(), k.as_tensor(), v.as_tensor())?
    };
    let w = Tensor::from_vec(w_h.clone(), (seq_len, d_k), &candle_dev)?;
    let loss = (out * w)?.sum_all()?;
    let grads = loss.backward()?;
    let gq: Vec<f32> = grads
        .get(q.as_tensor())
        .expect("Q grad")
        .flatten_all()?
        .to_vec1::<f32>()?;
    let gk: Vec<f32> = grads
        .get(k.as_tensor())
        .expect("K grad")
        .flatten_all()?
        .to_vec1::<f32>()?;
    let gv: Vec<f32> = grads
        .get(v.as_tensor())
        .expect("V grad")
        .flatten_all()?
        .to_vec1::<f32>()?;

    // Direct kaio-ops path on the same input bits: `_with_stats`
    // forward for out + L, then the three-kernel backward with dO = W.
    let q_buf = kaio_dev.alloc_from(&q_h)?;
    let k_buf = kaio_dev.alloc_from(&k_h)?;
    let v_buf = kaio_dev.alloc_from(&v_h)?;
    let w_buf = kaio_dev.alloc_from(&w_h)?;
    let mut out_buf = kaio_dev.alloc_zeros::<f32>(seq_len * d_k)?;
    let mut stats_buf = kaio_dev.alloc_zeros::<f32>(seq_len)?;
    let mut dq_buf = kaio_dev.alloc_zeros::<f32>(seq_len * d_k)?;
    let mut dk_buf = kaio_dev.alloc_zeros::<f32>(seq_len * d_k)?;
    let mut dv_buf = kaio_dev.alloc_zeros::<f32>(seq_len * d_k)?;
    if causal {
        kaio_ops::attention_flash_causal_with_stats(
            &kaio_dev,
            &q_buf,
            &k_buf,
            &v_buf,
            &mut out_buf,
            &mut stats_buf,
            seq_len as u32,
            d_k as u32,
        )?;
        kaio_ops::attention_flash_bwd_causal(
            &kaio_dev,
            &w_buf,
            &q_buf,
            &k_buf,
            &v_buf,
            &out_buf,
            &stats_buf,
            &mut dq_buf,
            &mut dk_buf,
            &mut dv_buf,
            seq_len as u32,
            d_k as u32,
        )?;
    } else {
        kaio_ops::attention_flash_with_stats(
            &kaio_dev,
            &q_buf,
            &k_buf,
            &v_buf,
            &mut out_buf,
            &mut stats_buf,
            seq_len as u32,
            d_k as u32,
        )?;
        kaio_ops::attention_flash_bwd(
            &kaio_dev,
            &w_buf,
            &q_buf,
            &k_buf,
            &v_buf,
            &out_buf,
            &stats_buf,
            &mut dq_buf,
            &mut dk_buf,
            &mut dv_buf,
            seq_len as u32,
            d_k as u32,
        )?;
    }
    let dq_h: Vec<f32> = dq_buf.to_host(&kaio_dev)?;
    let dk_h: Vec<f32> = dk_buf.to_host(&kaio_dev)?;
    let dv_h: Vec<f32> = dv_buf.to_host(&kaio_dev)?;

    let label = if causal { "causal" } else { "plain" };
    for (name, candle_g, direct_g) in [("dQ", &gq, &dq_h), ("dK", &gk, &dk_h), ("dV", &gv, &dv_h)] {
        assert_eq!(candle_g.len(), direct_g.len(), "{label} {name}: length");
        for (idx, (c, d)) in candle_g.iter().zip(direct_g.iter()).enumerate() {
            assert_eq!(
                c.to_bits(),
                d.to_bits(),
                "{label} {name}: bit mismatch at {idx}: candle {c} vs direct {d}"
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn attention_flash_bwd_paths_bit_exact_64x64() -> anyhow::Result<()> {
    bwd_path_equivalence(64, 64, false)
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn attention_flash_causal_bwd_paths_bit_exact_64x64() -> anyhow::Result<()> {
    bwd_path_equivalence(64, 64, true)
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn attention_flash_bwd_paths_bit_exact_non_aligned_17x19() -> anyhow::Result<()> {
    bwd_path_equivalence(17, 19, false)
}
