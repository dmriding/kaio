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

// ---------------------------------------------------------------------------
// qkv_project_int8 bit-exact cross-check (W8A16 fused tri-output)
// ---------------------------------------------------------------------------

fn bit_exact_qkv_project_int8(
    m: usize,
    n: usize,
    k: usize,
    scale_q: f32,
    scale_k: f32,
    scale_v: f32,
) -> anyhow::Result<()> {
    let x_host: Vec<f16> = (0..m * k)
        .map(|i| f16::from_f32(((i % 31) as f32) * 0.02 - 0.3))
        .collect();
    // i8 weights — candle side is u8
    let wq_host_i8: Vec<i8> = (0..k * n)
        .map(|i| (((i % 127) as i32) - 63) as i8)
        .collect();
    let wk_host_i8: Vec<i8> = (0..k * n).map(|i| (((i % 97) as i32) - 48) as i8).collect();
    let wv_host_i8: Vec<i8> = (0..k * n)
        .map(|i| (((i % 113) as i32) - 56) as i8)
        .collect();
    let wq_host_u8: Vec<u8> = wq_host_i8.iter().map(|&x| x as u8).collect();
    let wk_host_u8: Vec<u8> = wk_host_i8.iter().map(|&x| x as u8).collect();
    let wv_host_u8: Vec<u8> = wv_host_i8.iter().map(|&x| x as u8).collect();

    let candle_dev = Device::new_cuda(0)?;
    let kaio_dev = Arc::new(KaioDevice::new(0)?);

    // Candle path
    let x_c = Tensor::from_vec(x_host.clone(), (m, k), &candle_dev)?;
    let wq_c = Tensor::from_vec(wq_host_u8, (k, n), &candle_dev)?;
    let wk_c = Tensor::from_vec(wk_host_u8, (k, n), &candle_dev)?;
    let wv_c = Tensor::from_vec(wv_host_u8, (k, n), &candle_dev)?;
    let (q_c, k_c, v_c) = kaio_candle::qkv_project_int8(
        &kaio_dev, &x_c, &wq_c, &wk_c, &wv_c, scale_q, scale_k, scale_v,
    )?;
    let q_c_host: Vec<f16> = q_c.flatten_all()?.to_vec1::<f16>()?;
    let k_c_host: Vec<f16> = k_c.flatten_all()?.to_vec1::<f16>()?;
    let v_c_host: Vec<f16> = v_c.flatten_all()?.to_vec1::<f16>()?;

    // Direct kaio-ops path
    let x_buf = kaio_dev.alloc_from(&x_host)?;
    let wq_buf = kaio_dev.alloc_from(&wq_host_i8)?;
    let wk_buf = kaio_dev.alloc_from(&wk_host_i8)?;
    let wv_buf = kaio_dev.alloc_from(&wv_host_i8)?;
    let mut q_buf = kaio_dev.alloc_zeros::<f16>(m * n)?;
    let mut k_buf = kaio_dev.alloc_zeros::<f16>(m * n)?;
    let mut v_buf = kaio_dev.alloc_zeros::<f16>(m * n)?;
    kaio_ops::qkv_project_int8(
        &kaio_dev, &x_buf, &wq_buf, &wk_buf, &wv_buf, scale_q, scale_k, scale_v, &mut q_buf,
        &mut k_buf, &mut v_buf, m as u32, n as u32, k as u32,
    )?;
    let q_k_host: Vec<f16> = q_buf.to_host(&kaio_dev)?;
    let k_k_host: Vec<f16> = k_buf.to_host(&kaio_dev)?;
    let v_k_host: Vec<f16> = v_buf.to_host(&kaio_dev)?;

    // Independent per-projection bit-exact assertions.
    for (label, candle_out, kaio_out) in [
        ("Q", &q_c_host, &q_k_host),
        ("K", &k_c_host, &k_k_host),
        ("V", &v_c_host, &v_k_host),
    ] {
        assert_eq!(candle_out.len(), kaio_out.len(), "{label} length mismatch");
        for (i, (cv, kv)) in candle_out.iter().zip(kaio_out.iter()).enumerate() {
            assert_eq!(
                cv.to_bits(),
                kv.to_bits(),
                "{label} projection mismatch at element {i}: candle={cv} kaio={kv} \
                 (M={m} N={n} K={k})"
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn qkv_project_int8_bit_exact_64x64x64() -> anyhow::Result<()> {
    bit_exact_qkv_project_int8(64, 64, 64, 0.005, 0.01, 0.02)
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn qkv_project_int8_bit_exact_256x128x256() -> anyhow::Result<()> {
    bit_exact_qkv_project_int8(256, 128, 256, 1.0, 0.5, 2.0)
}

/// Q/K/V differentiation canary: W_Q=1, W_K=2, W_V=3, X=ones,
/// scale=1/K. Expected outputs ≈ 1, 2, 3 per cell. Catches projection-
/// routing bugs where fragment-C grids alias across projections.
#[test]
#[ignore = "requires NVIDIA GPU"]
fn qkv_project_int8_differentiation_canary() -> anyhow::Result<()> {
    let m = 64usize;
    let n = 32usize;
    let k = 128usize;
    let scale = 1.0 / (k as f32);

    let x_host: Vec<f16> = vec![f16::from_f32(1.0); m * k];
    let wq_u8: Vec<u8> = vec![1u8; k * n]; // i8 = 1
    let wk_u8: Vec<u8> = vec![2u8; k * n]; // i8 = 2
    let wv_u8: Vec<u8> = vec![3u8; k * n]; // i8 = 3

    let candle_dev = Device::new_cuda(0)?;
    let kaio_dev = Arc::new(KaioDevice::new(0)?);

    let x_c = Tensor::from_vec(x_host, (m, k), &candle_dev)?;
    let wq_c = Tensor::from_vec(wq_u8, (k, n), &candle_dev)?;
    let wk_c = Tensor::from_vec(wk_u8, (k, n), &candle_dev)?;
    let wv_c = Tensor::from_vec(wv_u8, (k, n), &candle_dev)?;

    let (q, kk, v) =
        kaio_candle::qkv_project_int8(&kaio_dev, &x_c, &wq_c, &wk_c, &wv_c, scale, scale, scale)?;
    let q_host: Vec<f16> = q.flatten_all()?.to_vec1::<f16>()?;
    let k_host: Vec<f16> = kk.flatten_all()?.to_vec1::<f16>()?;
    let v_host: Vec<f16> = v.flatten_all()?.to_vec1::<f16>()?;

    // Q ≈ 1.0, K ≈ 2.0, V ≈ 3.0
    for (label, out, expected) in [
        ("Q", &q_host, 1.0f32),
        ("K", &k_host, 2.0),
        ("V", &v_host, 3.0),
    ] {
        for (i, h) in out.iter().enumerate() {
            let val = h.to_f32();
            let err = (val - expected).abs();
            assert!(
                err < 0.1,
                "{label}: at [{i}] expected ≈{expected}, got {val} (err={err})"
            );
        }
    }
    // Outputs must differ across projections.
    assert!(
        (q_host[0].to_f32() - k_host[0].to_f32()).abs() > 0.5,
        "Q and K outputs alias (canary tripped)"
    );
    Ok(())
}

/// Non-contiguous weight tensor → must reject.
#[test]
#[ignore = "requires NVIDIA GPU"]
fn qkv_project_int8_rejects_noncontiguous() -> anyhow::Result<()> {
    let candle_dev = Device::new_cuda(0)?;
    let kaio_dev = Arc::new(KaioDevice::new(0)?);

    let x = Tensor::zeros((64, 64), candle_core::DType::F16, &candle_dev)?;
    let w = Tensor::zeros((64, 64), candle_core::DType::U8, &candle_dev)?;
    let w_nc = w.t()?;
    let w_ok = Tensor::zeros((64, 64), candle_core::DType::U8, &candle_dev)?;

    let err = kaio_candle::qkv_project_int8(&kaio_dev, &x, &w_nc, &w_ok, &w_ok, 1.0, 1.0, 1.0)
        .expect_err("must reject");
    let msg = format!("{err}");
    assert!(
        msg.contains("contiguous"),
        "expected contiguity rejection, got: {msg}"
    );
    Ok(())
}

/// Non-zero offset weight tensor → must reject.
#[test]
#[ignore = "requires NVIDIA GPU"]
fn qkv_project_int8_rejects_nonzero_offset() -> anyhow::Result<()> {
    let candle_dev = Device::new_cuda(0)?;
    let kaio_dev = Arc::new(KaioDevice::new(0)?);

    let x = Tensor::zeros((64, 64), candle_core::DType::F16, &candle_dev)?;
    let base = Tensor::zeros((128, 64), candle_core::DType::U8, &candle_dev)?;
    let w_off = base.narrow(0, 64, 64)?;
    let w_ok = Tensor::zeros((64, 64), candle_core::DType::U8, &candle_dev)?;

    let err = kaio_candle::qkv_project_int8(&kaio_dev, &x, &w_off, &w_ok, &w_ok, 1.0, 1.0, 1.0)
        .expect_err("must reject");
    let msg = format!("{err}");
    assert!(
        msg.contains("contiguous") || msg.contains("offset"),
        "expected rejection, got: {msg}"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// qkv_project_int4 bit-exact cross-check (W4A16 fused tri-output)
// ---------------------------------------------------------------------------

fn bit_exact_qkv_project_int4(m: usize, n: usize, k: usize) -> anyhow::Result<()> {
    assert!(k.is_multiple_of(128), "K must be a multiple of 128");
    let packed_rows = k / 8;
    let scale_rows = k / 128;

    let x_host: Vec<f16> = (0..m * k)
        .map(|i| f16::from_f32(((i % 23) as f32) * 0.02 - 0.22))
        .collect();
    // Packed u32 weights — deterministic patterned data.
    let wq_host: Vec<u32> = (0..packed_rows * n)
        .map(|i| ((i as u32).wrapping_mul(0x9E37_79B1)) ^ 0x1111_1111)
        .collect();
    let wk_host: Vec<u32> = (0..packed_rows * n)
        .map(|i| ((i as u32).wrapping_mul(0x9E37_79B1)) ^ 0x2222_2222)
        .collect();
    let wv_host: Vec<u32> = (0..packed_rows * n)
        .map(|i| ((i as u32).wrapping_mul(0x9E37_79B1)) ^ 0x3333_3333)
        .collect();
    let sq_host: Vec<f16> = (0..scale_rows * n)
        .map(|i| f16::from_f32(0.01 + ((i % 7) as f32) * 0.005))
        .collect();
    let sk_host: Vec<f16> = (0..scale_rows * n)
        .map(|i| f16::from_f32(0.02 + ((i % 11) as f32) * 0.003))
        .collect();
    let sv_host: Vec<f16> = (0..scale_rows * n)
        .map(|i| f16::from_f32(0.015 + ((i % 13) as f32) * 0.004))
        .collect();

    let candle_dev = Device::new_cuda(0)?;
    let kaio_dev = Arc::new(KaioDevice::new(0)?);

    // Candle path
    let x_c = Tensor::from_vec(x_host.clone(), (m, k), &candle_dev)?;
    let wq_c = Tensor::from_vec(wq_host.clone(), (packed_rows, n), &candle_dev)?;
    let wk_c = Tensor::from_vec(wk_host.clone(), (packed_rows, n), &candle_dev)?;
    let wv_c = Tensor::from_vec(wv_host.clone(), (packed_rows, n), &candle_dev)?;
    let sq_c = Tensor::from_vec(sq_host.clone(), (scale_rows, n), &candle_dev)?;
    let sk_c = Tensor::from_vec(sk_host.clone(), (scale_rows, n), &candle_dev)?;
    let sv_c = Tensor::from_vec(sv_host.clone(), (scale_rows, n), &candle_dev)?;
    let (q_c, k_c, v_c) =
        kaio_candle::qkv_project_int4(&kaio_dev, &x_c, &wq_c, &wk_c, &wv_c, &sq_c, &sk_c, &sv_c)?;
    let q_c_host: Vec<f16> = q_c.flatten_all()?.to_vec1::<f16>()?;
    let k_c_host: Vec<f16> = k_c.flatten_all()?.to_vec1::<f16>()?;
    let v_c_host: Vec<f16> = v_c.flatten_all()?.to_vec1::<f16>()?;

    // Direct kaio-ops path
    let x_buf = kaio_dev.alloc_from(&x_host)?;
    let wq_buf = kaio_dev.alloc_from(&wq_host)?;
    let wk_buf = kaio_dev.alloc_from(&wk_host)?;
    let wv_buf = kaio_dev.alloc_from(&wv_host)?;
    let sq_buf = kaio_dev.alloc_from(&sq_host)?;
    let sk_buf = kaio_dev.alloc_from(&sk_host)?;
    let sv_buf = kaio_dev.alloc_from(&sv_host)?;
    let mut q_buf = kaio_dev.alloc_zeros::<f16>(m * n)?;
    let mut k_buf = kaio_dev.alloc_zeros::<f16>(m * n)?;
    let mut v_buf = kaio_dev.alloc_zeros::<f16>(m * n)?;
    kaio_ops::qkv_project_int4(
        &kaio_dev, &x_buf, &wq_buf, &wk_buf, &wv_buf, &sq_buf, &sk_buf, &sv_buf, &mut q_buf,
        &mut k_buf, &mut v_buf, m as u32, n as u32, k as u32, 128,
    )?;
    let q_k_host: Vec<f16> = q_buf.to_host(&kaio_dev)?;
    let k_k_host: Vec<f16> = k_buf.to_host(&kaio_dev)?;
    let v_k_host: Vec<f16> = v_buf.to_host(&kaio_dev)?;

    for (label, candle_out, kaio_out) in [
        ("Q", &q_c_host, &q_k_host),
        ("K", &k_c_host, &k_k_host),
        ("V", &v_c_host, &v_k_host),
    ] {
        assert_eq!(candle_out.len(), kaio_out.len(), "{label} length mismatch");
        for (i, (c, ko)) in candle_out.iter().zip(kaio_out.iter()).enumerate() {
            assert_eq!(
                c.to_bits(),
                ko.to_bits(),
                "{label} projection mismatch at element {i}: candle={c} kaio={ko} \
                 (M={m} N={n} K={k})"
            );
        }
    }
    Ok(())
}

// 3 shapes: 1 group, 2 groups, 4 groups.
#[test]
#[ignore = "requires NVIDIA GPU"]
fn qkv_project_int4_bit_exact_64x64x128() -> anyhow::Result<()> {
    bit_exact_qkv_project_int4(64, 64, 128)
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn qkv_project_int4_bit_exact_256x128x256() -> anyhow::Result<()> {
    bit_exact_qkv_project_int4(256, 128, 256)
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn qkv_project_int4_bit_exact_128x64x512() -> anyhow::Result<()> {
    bit_exact_qkv_project_int4(128, 64, 512)
}

/// Q/K/V differentiation canary for INT4: packed nibble values 1/2/3
/// per projection with scale=1.0. Catches deterministic slot-mapping
/// bugs that random data can pass by coincidence.
#[test]
#[ignore = "requires NVIDIA GPU"]
fn qkv_project_int4_differentiation_canary() -> anyhow::Result<()> {
    let m = 64usize;
    let n = 32usize;
    let k = 128usize;
    let packed_rows = k / 8;
    let scale_rows = k / 128;

    let x_host: Vec<f16> = vec![f16::from_f32(1.0); m * k];
    // Pack 8 identical nibble values per u32: nibble=1 → 0x11111111, etc.
    let wq_host: Vec<u32> = vec![0x1111_1111u32; packed_rows * n];
    let wk_host: Vec<u32> = vec![0x2222_2222u32; packed_rows * n];
    let wv_host: Vec<u32> = vec![0x3333_3333u32; packed_rows * n];
    let sq_host: Vec<f16> = vec![f16::from_f32(1.0 / (k as f32)); scale_rows * n];
    let sk_host: Vec<f16> = vec![f16::from_f32(1.0 / (k as f32)); scale_rows * n];
    let sv_host: Vec<f16> = vec![f16::from_f32(1.0 / (k as f32)); scale_rows * n];

    let candle_dev = Device::new_cuda(0)?;
    let kaio_dev = Arc::new(KaioDevice::new(0)?);

    let x_c = Tensor::from_vec(x_host, (m, k), &candle_dev)?;
    let wq_c = Tensor::from_vec(wq_host, (packed_rows, n), &candle_dev)?;
    let wk_c = Tensor::from_vec(wk_host, (packed_rows, n), &candle_dev)?;
    let wv_c = Tensor::from_vec(wv_host, (packed_rows, n), &candle_dev)?;
    let sq_c = Tensor::from_vec(sq_host, (scale_rows, n), &candle_dev)?;
    let sk_c = Tensor::from_vec(sk_host, (scale_rows, n), &candle_dev)?;
    let sv_c = Tensor::from_vec(sv_host, (scale_rows, n), &candle_dev)?;

    let (q, kk, v) =
        kaio_candle::qkv_project_int4(&kaio_dev, &x_c, &wq_c, &wk_c, &wv_c, &sq_c, &sk_c, &sv_c)?;
    let q_host: Vec<f16> = q.flatten_all()?.to_vec1::<f16>()?;
    let k_host: Vec<f16> = kk.flatten_all()?.to_vec1::<f16>()?;
    let v_host: Vec<f16> = v.flatten_all()?.to_vec1::<f16>()?;

    // Outputs must differ across projections.
    let q0 = q_host[0].to_f32();
    let k0 = k_host[0].to_f32();
    let v0 = v_host[0].to_f32();
    assert!(
        (q0 - k0).abs() > 0.01,
        "Q and K outputs alias: q0={q0}, k0={k0}"
    );
    assert!(
        (k0 - v0).abs() > 0.01,
        "K and V outputs alias: k0={k0}, v0={v0}"
    );
    Ok(())
}

/// K not multiple of 128 → must reject.
#[test]
#[ignore = "requires NVIDIA GPU"]
fn qkv_project_int4_rejects_k_not_multiple_of_128() -> anyhow::Result<()> {
    let candle_dev = Device::new_cuda(0)?;
    let kaio_dev = Arc::new(KaioDevice::new(0)?);

    // K=64, not a multiple of 128. Use shapes that are otherwise valid for
    // the packed/scale row counts at K=64 to isolate the group_size check.
    let m = 16usize;
    let k = 64usize;
    let n = 16usize;
    let packed_rows = k / 8;
    let scale_rows = 1; // would be K/128 = 0, but we need at least 1 row

    let x = Tensor::zeros((m, k), candle_core::DType::F16, &candle_dev)?;
    let w = Tensor::zeros((packed_rows, n), candle_core::DType::U32, &candle_dev)?;
    let s = Tensor::zeros((scale_rows, n), candle_core::DType::F16, &candle_dev)?;

    let err = kaio_candle::qkv_project_int4(&kaio_dev, &x, &w, &w, &w, &s, &s, &s)
        .expect_err("must reject");
    let msg = format!("{err}");
    assert!(
        msg.contains("multiple of group_size") || msg.contains("multiple of 128"),
        "expected group_size rejection, got: {msg}"
    );
    Ok(())
}

/// Non-contiguous packed weight → must reject.
#[test]
#[ignore = "requires NVIDIA GPU"]
fn qkv_project_int4_rejects_noncontiguous() -> anyhow::Result<()> {
    let candle_dev = Device::new_cuda(0)?;
    let kaio_dev = Arc::new(KaioDevice::new(0)?);

    let m = 16usize;
    let k = 128usize;
    let n = 16usize;
    let packed_rows = k / 8;
    let scale_rows = k / 128;

    let x = Tensor::zeros((m, k), candle_core::DType::F16, &candle_dev)?;
    let w_ok = Tensor::zeros((packed_rows, n), candle_core::DType::U32, &candle_dev)?;
    let w_nc = w_ok.t()?;
    let s = Tensor::zeros((scale_rows, n), candle_core::DType::F16, &candle_dev)?;

    let err = kaio_candle::qkv_project_int4(&kaio_dev, &x, &w_nc, &w_ok, &w_ok, &s, &s, &s)
        .expect_err("must reject");
    let msg = format!("{err}");
    assert!(
        msg.contains("contiguous"),
        "expected contiguity rejection, got: {msg}"
    );
    Ok(())
}

/// Non-zero offset weight → must reject.
#[test]
#[ignore = "requires NVIDIA GPU"]
fn qkv_project_int4_rejects_nonzero_offset() -> anyhow::Result<()> {
    let candle_dev = Device::new_cuda(0)?;
    let kaio_dev = Arc::new(KaioDevice::new(0)?);

    let m = 16usize;
    let k = 128usize;
    let n = 16usize;
    let packed_rows = k / 8;
    let scale_rows = k / 128;

    let x = Tensor::zeros((m, k), candle_core::DType::F16, &candle_dev)?;
    let base = Tensor::zeros((packed_rows * 2, n), candle_core::DType::U32, &candle_dev)?;
    let w_off = base.narrow(0, packed_rows, packed_rows)?;
    let w_ok = Tensor::zeros((packed_rows, n), candle_core::DType::U32, &candle_dev)?;
    let s = Tensor::zeros((scale_rows, n), candle_core::DType::F16, &candle_dev)?;

    let err = kaio_candle::qkv_project_int4(&kaio_dev, &x, &w_off, &w_ok, &w_ok, &s, &s, &s)
        .expect_err("must reject");
    let msg = format!("{err}");
    assert!(
        msg.contains("contiguous") || msg.contains("offset"),
        "expected rejection, got: {msg}"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Event-based stream sync API-path smoke test (Sprint 7.4c)
// ---------------------------------------------------------------------------

/// Validates that the event-based sync API path (join / record_event /
/// wait) executes without error on real hardware. Does NOT prove cross-
/// stream ordering semantics — both sides use the default stream, so
/// ordering is already guaranteed by FIFO. Catches Rollback #1's failure
/// mode: driver rejection of the event API calls.
#[test]
#[ignore = "requires NVIDIA GPU"]
fn event_based_sync_smoke_test() -> anyhow::Result<()> {
    let candle_dev = Device::new_cuda(0)?;
    let kaio_dev = Arc::new(KaioDevice::new(0)?);

    // Calling a bridge op exercises both sync_before_launch and
    // sync_after_launch with the new event-based path internally.
    let a = Tensor::ones((16, 16), candle_core::DType::F16, &candle_dev)?;
    let b = Tensor::ones((16, 16), candle_core::DType::F16, &candle_dev)?;
    let _c = kaio_candle::matmul_tc(&kaio_dev, &a, &b)?;

    // If we get here without error, the event/join API path works.
    Ok(())
}

// ---------------------------------------------------------------------------
// matmul_tc / matmul_tc_async backward (gradient correctness, Sprint 7.4d)
// ---------------------------------------------------------------------------

/// Numerical gradient check via finite differences. For C = A @ B with
/// loss = C.sum(), the analytical gradients are dA = ones @ B^T and
/// dB = A^T @ ones (since dL/dC = ones for sum()). We compare against
/// the autograd-computed gradients from bwd().
///
/// Inputs initialized near zero with small variance (Gemini G3-3) so
/// f16 mantissa can represent the values accurately.
///
/// Note: this backward is numerically approximate (f32→f16 downcast on
/// grad_res + output grad cast back to f16). Dual tolerance accounts
/// for f16 precision.
fn gradient_check_matmul(m: usize, k: usize, n: usize, use_async: bool) -> anyhow::Result<()> {
    let candle_dev = Device::new_cuda(0)?;
    let kaio_dev = Arc::new(KaioDevice::new(0)?);

    // Small-magnitude patterned data in [-0.1, 0.1] for f16 stability.
    let a_data: Vec<f16> = (0..m * k)
        .map(|i| f16::from_f32(((i % 19) as f32 - 9.0) * 0.01))
        .collect();
    let b_data: Vec<f16> = (0..k * n)
        .map(|i| f16::from_f32(((i % 23) as f32 - 11.0) * 0.01))
        .collect();

    // Create tensors with gradient tracking via Var.
    let a = candle_core::Var::from_vec(a_data.clone(), (m, k), &candle_dev)?;
    let b = candle_core::Var::from_vec(b_data.clone(), (k, n), &candle_dev)?;

    // Forward: C = A @ B via the bridge op
    let c = if use_async {
        kaio_candle::matmul_tc_async(&kaio_dev, a.as_tensor(), b.as_tensor())?
    } else {
        kaio_candle::matmul_tc(&kaio_dev, a.as_tensor(), b.as_tensor())?
    };

    // Scalar loss: sum of all elements
    let loss = c.sum_all()?;

    // Backward — returns GradStore, extract per-tensor gradients.
    let grads = loss.backward()?;

    let grad_a = grads.get(a.as_tensor()).expect("a should have gradient");
    let grad_b = grads.get(b.as_tensor()).expect("b should have gradient");

    let grad_a_host: Vec<f16> = grad_a.flatten_all()?.to_vec1::<f16>()?;
    let grad_b_host: Vec<f16> = grad_b.flatten_all()?.to_vec1::<f16>()?;

    // For loss = sum(A @ B), the analytical gradient is:
    //   dA = ones[M,N] @ B^T = sum over N of each row of B^T
    //   dB = A^T @ ones[M,N] = sum over M of each col of A^T
    // We compute expected gradients on the host for comparison.

    // Expected dA[i,j] = sum over l of B[j,l] (since dC is all-ones,
    // dA = ones @ B^T, so dA[i,j] = sum_l B[j,l])
    let mut expected_grad_a = vec![0.0f32; m * k];
    for i in 0..m {
        for j in 0..k {
            let mut sum = 0.0f32;
            for l in 0..n {
                sum += b_data[j * n + l].to_f32();
            }
            expected_grad_a[i * k + j] = sum;
        }
    }

    // Expected dB[i,j] = sum over l of A[l,i] (dB = A^T @ ones,
    // so dB[i,j] = sum_l A[l,i])
    let mut expected_grad_b = vec![0.0f32; k * n];
    for i in 0..k {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..m {
                sum += a_data[l * k + i].to_f32();
            }
            expected_grad_b[i * n + j] = sum;
        }
    }

    // Compare with dual tolerance (Opus R1): rel < 1e-2 OR abs < 1e-3
    let variant = if use_async {
        "matmul_tc_async"
    } else {
        "matmul_tc"
    };
    for (idx, (got, expected)) in grad_a_host.iter().zip(expected_grad_a.iter()).enumerate() {
        let got_f32 = got.to_f32();
        let exp = *expected;
        let abs_err = (got_f32 - exp).abs();
        let rel_err = if exp.abs() > 1e-6 {
            abs_err / exp.abs()
        } else {
            abs_err
        };
        assert!(
            rel_err < 1e-2 || abs_err < 1e-3,
            "{variant} grad_a mismatch at [{idx}]: got={got_f32}, expected={expected}, \
             rel_err={rel_err:.4e}, abs_err={abs_err:.4e} (M={m} K={k} N={n})"
        );
    }
    for (idx, (got, expected)) in grad_b_host.iter().zip(expected_grad_b.iter()).enumerate() {
        let got_f32 = got.to_f32();
        let exp = *expected;
        let abs_err = (got_f32 - exp).abs();
        let rel_err = if exp.abs() > 1e-6 {
            abs_err / exp.abs()
        } else {
            abs_err
        };
        assert!(
            rel_err < 1e-2 || abs_err < 1e-3,
            "{variant} grad_b mismatch at [{idx}]: got={got_f32}, expected={expected}, \
             rel_err={rel_err:.4e}, abs_err={abs_err:.4e} (M={m} K={k} N={n})"
        );
    }

    Ok(())
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn matmul_tc_backward_32x32x32() -> anyhow::Result<()> {
    gradient_check_matmul(32, 32, 32, false)
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn matmul_tc_backward_128x128x128() -> anyhow::Result<()> {
    gradient_check_matmul(128, 128, 128, false)
}

/// Non-square shape catches transposition bugs where M=K=N would mask a swap.
#[test]
#[ignore = "requires NVIDIA GPU"]
fn matmul_tc_backward_64x32x128() -> anyhow::Result<()> {
    gradient_check_matmul(64, 32, 128, false)
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn matmul_tc_async_backward_32x32x32() -> anyhow::Result<()> {
    gradient_check_matmul(32, 32, 32, true)
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn matmul_tc_async_backward_128x128x128() -> anyhow::Result<()> {
    gradient_check_matmul(128, 128, 128, true)
}

#[test]
#[ignore = "requires NVIDIA GPU"]
fn matmul_tc_async_backward_64x32x128() -> anyhow::Result<()> {
    gradient_check_matmul(64, 32, 128, true)
}
