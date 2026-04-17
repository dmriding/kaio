//! GPU integration tests — bit-exact cross-check between candle-routed
//! bridge calls and direct `kaio_ops::*` calls on the same bits.
//!
//! **Marked `#[ignore]`.** Run on a GPU host via `cargo test --features cuda
//! -- --ignored`. Compiles as empty when the `cuda` feature is off.
//!
//! The point of bit-exactness here is not tolerance-within-fp16 — it is
//! that the bridge introduces NO reinterpretation. Same kernel + same bits
//! → same output bits. Any divergence is a bridge bug (wrong slice
//! offset, dtype coercion, missed contiguity check). No tolerance
//! relaxation — Rollback #4 in the plan explicitly rejects that.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use candle_core::{Device, Tensor};
use half::f16;
use kaio::prelude::KaioDevice;

// ---------------------------------------------------------------------------
// matmul_tc bit-exact cross-check
// ---------------------------------------------------------------------------

fn bit_exact_matmul_tc(m: usize, n: usize, k: usize) -> anyhow::Result<()> {
    // Deterministic host-side inputs so candle and kaio-ops see identical
    // bits going in. Using patterned data (not randn) keeps the test
    // reproducible across runs without seeding work.
    let a_host: Vec<f16> = (0..m * k)
        .map(|i| f16::from_f32(((i % 31) as f32) * 0.1 - 1.5))
        .collect();
    let b_host: Vec<f16> = (0..k * n)
        .map(|i| f16::from_f32(((i % 17) as f32) * 0.05 - 0.4))
        .collect();

    let candle_dev = Device::new_cuda(0)?;
    let kaio_dev = Arc::new(KaioDevice::new(0)?);

    // Candle path
    let a_candle = Tensor::from_vec(a_host.clone(), (m, k), &candle_dev)?;
    let b_candle = Tensor::from_vec(b_host.clone(), (k, n), &candle_dev)?;
    let c_candle = kaio_candle::matmul_tc(&kaio_dev, &a_candle, &b_candle)?;
    let c_candle_host: Vec<f32> = c_candle.flatten_all()?.to_vec1::<f32>()?;

    // Direct kaio-ops path
    let a_buf = kaio_dev.alloc_from(&a_host)?;
    let b_buf = kaio_dev.alloc_from(&b_host)?;
    let mut c_buf = kaio_dev.alloc_zeros::<f32>(m * n)?;
    kaio_ops::matmul_tc(
        &kaio_dev, &a_buf, &b_buf, &mut c_buf, m as u32, n as u32, k as u32,
    )?;
    let c_kaio_host: Vec<f32> = c_buf.to_host(&kaio_dev)?;

    assert_eq!(
        c_candle_host.len(),
        c_kaio_host.len(),
        "output length mismatch"
    );
    for (i, (cc, ck)) in c_candle_host.iter().zip(c_kaio_host.iter()).enumerate() {
        assert_eq!(
            cc.to_bits(),
            ck.to_bits(),
            "bit mismatch at element {i}: candle={cc} kaio={ck} (shapes {m}x{n}x{k})"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn matmul_tc_bit_exact_64() -> anyhow::Result<()> {
    bit_exact_matmul_tc(64, 64, 64)
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn matmul_tc_bit_exact_256() -> anyhow::Result<()> {
    bit_exact_matmul_tc(256, 256, 256)
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn matmul_tc_bit_exact_1024() -> anyhow::Result<()> {
    bit_exact_matmul_tc(1024, 1024, 1024)
}

// ---------------------------------------------------------------------------
// matmul_tc_async bit-exact cross-check — identical pattern, cp.async kernel
// ---------------------------------------------------------------------------

fn bit_exact_matmul_tc_async(m: usize, n: usize, k: usize) -> anyhow::Result<()> {
    let a_host: Vec<f16> = (0..m * k)
        .map(|i| f16::from_f32(((i % 31) as f32) * 0.1 - 1.5))
        .collect();
    let b_host: Vec<f16> = (0..k * n)
        .map(|i| f16::from_f32(((i % 17) as f32) * 0.05 - 0.4))
        .collect();

    let candle_dev = Device::new_cuda(0)?;
    let kaio_dev = Arc::new(KaioDevice::new(0)?);

    let a_candle = Tensor::from_vec(a_host.clone(), (m, k), &candle_dev)?;
    let b_candle = Tensor::from_vec(b_host.clone(), (k, n), &candle_dev)?;
    let c_candle = kaio_candle::matmul_tc_async(&kaio_dev, &a_candle, &b_candle)?;
    let c_candle_host: Vec<f32> = c_candle.flatten_all()?.to_vec1::<f32>()?;

    let a_buf = kaio_dev.alloc_from(&a_host)?;
    let b_buf = kaio_dev.alloc_from(&b_host)?;
    let mut c_buf = kaio_dev.alloc_zeros::<f32>(m * n)?;
    kaio_ops::matmul_tc_async(
        &kaio_dev, &a_buf, &b_buf, &mut c_buf, m as u32, n as u32, k as u32,
    )?;
    let c_kaio_host: Vec<f32> = c_buf.to_host(&kaio_dev)?;

    assert_eq!(c_candle_host.len(), c_kaio_host.len());
    for (i, (cc, ck)) in c_candle_host.iter().zip(c_kaio_host.iter()).enumerate() {
        assert_eq!(
            cc.to_bits(),
            ck.to_bits(),
            "bit mismatch at element {i}: candle={cc} kaio={ck} (shapes {m}x{n}x{k})"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn matmul_tc_async_bit_exact_64() -> anyhow::Result<()> {
    bit_exact_matmul_tc_async(64, 64, 64)
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn matmul_tc_async_bit_exact_256() -> anyhow::Result<()> {
    bit_exact_matmul_tc_async(256, 256, 256)
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn matmul_tc_async_bit_exact_1024() -> anyhow::Result<()> {
    bit_exact_matmul_tc_async(1024, 1024, 1024)
}

// ---------------------------------------------------------------------------
// Rejection-path tests — non-contiguous and non-zero-offset inputs
// ---------------------------------------------------------------------------

/// Transposed tensor → non-contiguous → bridge must reject loudly, not
/// silently produce wrong bits (Codex R3, Opus #7).
#[test]
#[ignore = "requires NVIDIA GPU"]
fn matmul_tc_rejects_noncontiguous() -> anyhow::Result<()> {
    let candle_dev = Device::new_cuda(0)?;
    let kaio_dev = Arc::new(KaioDevice::new(0)?);

    let a = Tensor::ones((64, 64), candle_core::DType::F16, &candle_dev)?;
    let b = Tensor::ones((64, 64), candle_core::DType::F16, &candle_dev)?;

    // `.t()` (transpose last two dims) produces a non-contiguous view.
    let a_nc = a.t()?;
    assert!(
        !a_nc.is_contiguous(),
        "setup sanity: .t() should be non-contiguous"
    );

    let err = kaio_candle::matmul_tc(&kaio_dev, &a_nc, &b).expect_err("must reject");
    let msg = format!("{err}");
    assert!(
        msg.contains("contiguous"),
        "expected contiguity rejection message, got: {msg}"
    );
    Ok(())
}

/// `.narrow(...)` on a base tensor produces a contiguous view with non-zero
/// offset → bridge must reject with the offset hint (Codex R3).
#[test]
#[ignore = "requires NVIDIA GPU"]
fn matmul_tc_rejects_nonzero_offset() -> anyhow::Result<()> {
    let candle_dev = Device::new_cuda(0)?;
    let kaio_dev = Arc::new(KaioDevice::new(0)?);

    // Build a [128, 64] base; narrow dim-0 from row 64 onward to get a
    // contiguous [64, 64] view starting mid-storage.
    let base = Tensor::ones((128, 64), candle_core::DType::F16, &candle_dev)?;
    let a_off = base.narrow(0, 64, 64)?;
    let b = Tensor::ones((64, 64), candle_core::DType::F16, &candle_dev)?;

    // `.narrow(...)` in candle produces a non-contiguous result in some
    // cases and a non-zero-offset contiguous view in others, depending on
    // which axis is narrowed. The bridge rejects both via separate message
    // paths (contiguity AND offset checks). Assert that SOME rejection
    // fires — whichever path catches it first is acceptable for the test.
    let err = kaio_candle::matmul_tc(&kaio_dev, &a_off, &b).expect_err("must reject");
    let msg = format!("{err}");
    assert!(
        msg.contains("contiguous") || msg.contains("offset"),
        "expected contiguity or offset rejection, got: {msg}"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// matmul_int4 bit-exact cross-check (INT4 GPTQ-style, group_size=128)
// ---------------------------------------------------------------------------

fn bit_exact_matmul_int4(m: usize, n: usize, k: usize) -> anyhow::Result<()> {
    assert!(
        k.is_multiple_of(128),
        "test setup: K must be multiple of 128"
    );
    let packed_rows = k / 8; // 8 INT4 per u32
    let scale_rows = k / 128;

    // Deterministic patterned data — same bits on both paths.
    let a_host: Vec<f16> = (0..m * k)
        .map(|i| f16::from_f32(((i % 23) as f32) * 0.08 - 0.9))
        .collect();
    let b_packed_host: Vec<u32> = (0..packed_rows * n)
        .map(|i| ((i as u32).wrapping_mul(0x9E37_79B1)) ^ 0x1234_5678)
        .collect();
    let scales_host: Vec<f16> = (0..scale_rows * n)
        .map(|i| f16::from_f32(0.01 + ((i % 7) as f32) * 0.005))
        .collect();

    let candle_dev = Device::new_cuda(0)?;
    let kaio_dev = Arc::new(KaioDevice::new(0)?);

    // Candle path — rank-2 tensors matching the kaio-ops layout interpretation.
    let a_candle = Tensor::from_vec(a_host.clone(), (m, k), &candle_dev)?;
    let b_candle = Tensor::from_vec(b_packed_host.clone(), (packed_rows, n), &candle_dev)?;
    let s_candle = Tensor::from_vec(scales_host.clone(), (scale_rows, n), &candle_dev)?;
    let c_candle = kaio_candle::matmul_int4(&kaio_dev, &a_candle, &b_candle, &s_candle)?;
    let c_candle_host: Vec<f32> = c_candle.flatten_all()?.to_vec1::<f32>()?;

    // Direct kaio-ops path — flat GpuBuffers.
    let a_buf = kaio_dev.alloc_from(&a_host)?;
    let b_buf = kaio_dev.alloc_from(&b_packed_host)?;
    let s_buf = kaio_dev.alloc_from(&scales_host)?;
    let mut c_buf = kaio_dev.alloc_zeros::<f32>(m * n)?;
    kaio_ops::matmul_int4(
        &kaio_dev, &a_buf, &b_buf, &s_buf, &mut c_buf, m as u32, n as u32, k as u32, 128,
    )?;
    let c_kaio_host: Vec<f32> = c_buf.to_host(&kaio_dev)?;

    assert_eq!(c_candle_host.len(), c_kaio_host.len());
    for (i, (cc, ck)) in c_candle_host.iter().zip(c_kaio_host.iter()).enumerate() {
        assert_eq!(
            cc.to_bits(),
            ck.to_bits(),
            "bit mismatch at {i}: candle={cc} kaio={ck} (shapes {m}x{n}x{k}, group_size=128)"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn matmul_int4_bit_exact_256x256x128() -> anyhow::Result<()> {
    bit_exact_matmul_int4(256, 256, 128)
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn matmul_int4_bit_exact_1024x1024x512() -> anyhow::Result<()> {
    bit_exact_matmul_int4(1024, 1024, 512)
}

/// INT4 bridge must reject K that isn't a multiple of group_size (128).
/// Pure rejection test — host-reachable error, no kernel launch.
#[test]
#[ignore = "requires NVIDIA GPU"]
fn matmul_int4_rejects_k_not_multiple_of_group_size() -> anyhow::Result<()> {
    let candle_dev = Device::new_cuda(0)?;
    let kaio_dev = Arc::new(KaioDevice::new(0)?);

    // K=100 → not a multiple of 128. Use tiny shapes to minimize allocation.
    let m = 16usize;
    let k = 100usize;
    let n = 16usize;
    let a = Tensor::zeros((m, k), candle_core::DType::F16, &candle_dev)?;
    let b_packed = Tensor::zeros((k.div_ceil(8), n), candle_core::DType::U32, &candle_dev)?;
    let scales = Tensor::zeros((1, n), candle_core::DType::F16, &candle_dev)?;

    let err = kaio_candle::matmul_int4(&kaio_dev, &a, &b_packed, &scales)
        .expect_err("K=100 must be rejected");
    let msg = format!("{err}");
    assert!(
        msg.contains("group_size") || msg.contains("128"),
        "expected group-size rejection message, got: {msg}"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// attention_tc + attention_tc_causal bit-exact cross-check
// ---------------------------------------------------------------------------

fn bit_exact_attention_tc(
    seq_q: usize,
    seq_k: usize,
    d_k: usize,
    d_v: usize,
    causal: bool,
) -> anyhow::Result<()> {
    let q_host: Vec<f16> = (0..seq_q * d_k)
        .map(|i| f16::from_f32(((i % 19) as f32) * 0.03 - 0.3))
        .collect();
    let k_host: Vec<f16> = (0..seq_k * d_k)
        .map(|i| f16::from_f32(((i % 23) as f32) * 0.025 - 0.29))
        .collect();
    let v_host: Vec<f16> = (0..seq_k * d_v)
        .map(|i| f16::from_f32(((i % 29) as f32) * 0.02 - 0.28))
        .collect();

    let candle_dev = Device::new_cuda(0)?;
    let kaio_dev = Arc::new(KaioDevice::new(0)?);

    // Candle path
    let q_candle = Tensor::from_vec(q_host.clone(), (seq_q, d_k), &candle_dev)?;
    let k_candle = Tensor::from_vec(k_host.clone(), (seq_k, d_k), &candle_dev)?;
    let v_candle = Tensor::from_vec(v_host.clone(), (seq_k, d_v), &candle_dev)?;
    let out_candle = if causal {
        kaio_candle::attention_tc_causal(&kaio_dev, &q_candle, &k_candle, &v_candle)?
    } else {
        kaio_candle::attention_tc(&kaio_dev, &q_candle, &k_candle, &v_candle)?
    };
    let out_candle_host: Vec<f32> = out_candle.flatten_all()?.to_vec1::<f32>()?;

    // Direct kaio-ops path — use the #[doc(hidden)] pub re-exports from kaio-ops.
    let q_buf = kaio_dev.alloc_from(&q_host)?;
    let k_buf = kaio_dev.alloc_from(&k_host)?;
    let v_buf = kaio_dev.alloc_from(&v_host)?;
    let mut out_buf = kaio_dev.alloc_zeros::<f32>(seq_q * d_v)?;
    if causal {
        kaio_ops::attention_tc_causal(
            &kaio_dev,
            &q_buf,
            &k_buf,
            &v_buf,
            &mut out_buf,
            seq_q as u32,
            seq_k as u32,
            d_k as u32,
            d_v as u32,
        )?;
    } else {
        kaio_ops::attention_tc(
            &kaio_dev,
            &q_buf,
            &k_buf,
            &v_buf,
            &mut out_buf,
            seq_q as u32,
            seq_k as u32,
            d_k as u32,
            d_v as u32,
        )?;
    }
    let out_kaio_host: Vec<f32> = out_buf.to_host(&kaio_dev)?;

    assert_eq!(out_candle_host.len(), out_kaio_host.len());
    for (i, (c, k)) in out_candle_host.iter().zip(out_kaio_host.iter()).enumerate() {
        assert_eq!(
            c.to_bits(),
            k.to_bits(),
            "bit mismatch at {i}: candle={c} kaio={k} \
             (seq_q={seq_q} seq_k={seq_k} d_k={d_k} d_v={d_v} causal={causal})"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn attention_tc_bit_exact_64x64() -> anyhow::Result<()> {
    bit_exact_attention_tc(64, 64, 64, 64, false)
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn attention_tc_bit_exact_256x128() -> anyhow::Result<()> {
    // seq_k ≤ 384 per kaio-ops attention_tc shared-memory score-buffer cap;
    // FlashAttention-TC will lift this in a later sprint. 256 fits cleanly.
    bit_exact_attention_tc(256, 256, 128, 128, false)
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn attention_tc_causal_bit_exact_64x64() -> anyhow::Result<()> {
    bit_exact_attention_tc(64, 64, 64, 64, true)
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn attention_tc_causal_bit_exact_256x128() -> anyhow::Result<()> {
    bit_exact_attention_tc(256, 256, 128, 128, true)
}

// ---------------------------------------------------------------------------
// matmul_int8 bit-exact cross-check (W8A8 symmetric, scalar f32 scale)
// ---------------------------------------------------------------------------

fn bit_exact_matmul_int8(m: usize, n: usize, k: usize, scale: f32) -> anyhow::Result<()> {
    // Deterministic signed-INT8 patterned data. `as u8` is a bit-preserving
    // cast for i8 → u8 (-1_i8 is 255_u8), so candle's DType::U8 storage
    // and kaio-ops' GpuBuffer<i8> see the exact same bytes on the device.
    let a_host_i8: Vec<i8> = (0..m * k)
        .map(|i| (((i % 127) as i32) - 63) as i8)
        .collect();
    let b_host_i8: Vec<i8> = (0..k * n).map(|i| (((i % 97) as i32) - 48) as i8).collect();
    let a_host_u8: Vec<u8> = a_host_i8.iter().map(|&x| x as u8).collect();
    let b_host_u8: Vec<u8> = b_host_i8.iter().map(|&x| x as u8).collect();

    let candle_dev = Device::new_cuda(0)?;
    let kaio_dev = Arc::new(KaioDevice::new(0)?);

    // Candle path — DType::U8 tensors, bridge reinterprets as i8 internally.
    let a_candle = Tensor::from_vec(a_host_u8, (m, k), &candle_dev)?;
    let b_candle = Tensor::from_vec(b_host_u8, (k, n), &candle_dev)?;
    let c_candle = kaio_candle::matmul_int8(&kaio_dev, &a_candle, &b_candle, scale)?;
    let c_candle_host: Vec<f32> = c_candle.flatten_all()?.to_vec1::<f32>()?;

    // Direct kaio-ops path — native i8 GpuBuffers.
    let a_buf = kaio_dev.alloc_from(&a_host_i8)?;
    let b_buf = kaio_dev.alloc_from(&b_host_i8)?;
    let mut c_buf = kaio_dev.alloc_zeros::<f32>(m * n)?;
    kaio_ops::matmul_int8(
        &kaio_dev, &a_buf, &b_buf, &mut c_buf, scale, m as u32, n as u32, k as u32,
    )?;
    let c_kaio_host: Vec<f32> = c_buf.to_host(&kaio_dev)?;

    assert_eq!(
        c_candle_host.len(),
        c_kaio_host.len(),
        "output length mismatch"
    );
    for (i, (cc, ck)) in c_candle_host.iter().zip(c_kaio_host.iter()).enumerate() {
        assert_eq!(
            cc.to_bits(),
            ck.to_bits(),
            "bit mismatch at element {i}: candle={cc} kaio={ck} (shapes {m}x{n}x{k}, scale={scale})"
        );
    }
    Ok(())
}

// Scale values span three regimes:
//   - 0.00125 — realistic INT8 quant (≈ max_abs / 127)
//   - 1.0     — identity
//   - 47.3    — large
// A dropped-scale bug produces obviously-wrong output in at least two of
// these, and the failing test's scale localises which regime broke.

#[test]
#[ignore = "requires NVIDIA GPU"]
fn matmul_int8_bit_exact_256_small_scale() -> anyhow::Result<()> {
    bit_exact_matmul_int8(256, 256, 256, 0.00125)
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn matmul_int8_bit_exact_1024_identity_scale() -> anyhow::Result<()> {
    bit_exact_matmul_int8(1024, 1024, 1024, 1.0)
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn matmul_int8_bit_exact_4096_large_scale() -> anyhow::Result<()> {
    bit_exact_matmul_int8(4096, 4096, 4096, 47.3)
}

/// Transposed tensor → non-contiguous → bridge must reject loudly.
#[test]
#[ignore = "requires NVIDIA GPU"]
fn matmul_int8_rejects_noncontiguous() -> anyhow::Result<()> {
    let candle_dev = Device::new_cuda(0)?;
    let kaio_dev = Arc::new(KaioDevice::new(0)?);

    let a = Tensor::zeros((64, 64), candle_core::DType::U8, &candle_dev)?;
    let b = Tensor::zeros((64, 64), candle_core::DType::U8, &candle_dev)?;

    let a_nc = a.t()?;
    assert!(
        !a_nc.is_contiguous(),
        "setup sanity: .t() should be non-contiguous"
    );

    let err = kaio_candle::matmul_int8(&kaio_dev, &a_nc, &b, 1.0).expect_err("must reject");
    let msg = format!("{err}");
    assert!(
        msg.contains("contiguous"),
        "expected contiguity rejection message, got: {msg}"
    );
    Ok(())
}

/// `.narrow(...)` on a base tensor produces a contiguous view with non-zero
/// offset (or non-contiguous, depending on axis) → bridge must reject.
#[test]
#[ignore = "requires NVIDIA GPU"]
fn matmul_int8_rejects_nonzero_offset() -> anyhow::Result<()> {
    let candle_dev = Device::new_cuda(0)?;
    let kaio_dev = Arc::new(KaioDevice::new(0)?);

    let base = Tensor::zeros((128, 64), candle_core::DType::U8, &candle_dev)?;
    let a_off = base.narrow(0, 64, 64)?;
    let b = Tensor::zeros((64, 64), candle_core::DType::U8, &candle_dev)?;

    let err = kaio_candle::matmul_int8(&kaio_dev, &a_off, &b, 1.0).expect_err("must reject");
    let msg = format!("{err}");
    assert!(
        msg.contains("contiguous") || msg.contains("offset"),
        "expected contiguity or offset rejection, got: {msg}"
    );
    Ok(())
}
