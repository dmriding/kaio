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
