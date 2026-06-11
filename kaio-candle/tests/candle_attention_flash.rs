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
