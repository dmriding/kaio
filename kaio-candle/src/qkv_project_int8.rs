//! `qkv_project_int8` — direct-call bridge for the W8A16 fused
//! tri-output QKV projection kernel.
//!
//! This module does NOT use candle's `CustomOpN` traits (max 3 inputs,
//! single output). Instead it provides a free function that takes
//! `&Tensor` inputs, extracts `CudaStorage` itself, validates, calls the
//! fused kernel, and returns `(Tensor, Tensor, Tensor)`.
//!
//! ## Output dtype
//!
//! Returns `DType::F16` — the fused kernel performs `f32→f16` conversion
//! internally as part of the projection fusion. This differs from the
//! `CustomOp`-based ops (`matmul_tc`, `matmul_int4`, `attention_tc`)
//! which return `DType::F32` matching the kaio-ops accumulator.
//!
//! ## Autograd
//!
//! Forward-only. Gradient-tracked inputs are rejected with a loud error
//! requiring `.detach()` — see Gemini G3-1 in the sprint plan.

use std::sync::Arc;

use candle_core::op::BackpropOp;
use candle_core::{Error, Result, Storage, Tensor};
use half::f16;
use kaio::prelude::{GpuBuffer, KaioDevice};
use kaio_ops::qkv_project_int8 as kaio_qkv_project_int8;

use crate::bridge;

const OP_NAME: &str = "qkv_project_int8";

/// Reject gradient-tracked tensors (G3-1: silent autograd disconnection
/// guard). Direct-call ops bypass candle's BackpropOp graph tracking, so
/// passing a tracked tensor would silently sever the computation graph.
fn reject_if_variable(tensor: &Tensor, param_name: &str) -> Result<()> {
    if tensor.is_variable() {
        return Err(Error::Msg(format!(
            "kaio-candle::{OP_NAME}: {param_name} has gradient tracking enabled. \
             Direct-call ops are forward-only and do not support autograd. \
             Call `.detach()` on the tensor before passing it to {OP_NAME}."
        )));
    }
    Ok(())
}

/// Fused W8A16 tri-output QKV projection via KAIO's tensor-core kernel.
///
/// Single kernel launch produces three `f16[M, N]` output tensors (Q, K, V)
/// from shared f16 activations and per-projection i8 weights with scalar
/// f32 scales.
///
/// - `x`: `f16[M, K]` — activations. Contiguous, zero-offset.
/// - `w_q`, `w_k`, `w_v`: `u8[K, N]` — weights (candle `DType::U8`,
///   interpreted as signed INT8 by the kernel). Contiguous, zero-offset.
/// - `scale_q`, `scale_k`, `scale_v`: scalar `f32` per-projection scales.
/// - Returns: `(f16[M, N], f16[M, N], f16[M, N])` — (Q, K, V).
///
/// All inputs must be on the same CUDA device, matching the supplied
/// `Arc<KaioDevice>`. Gradient-tracked inputs are rejected (forward-only).
///
/// Requires SM 8.0+ (Ampere or newer).
#[allow(clippy::too_many_arguments)]
pub fn qkv_project_int8(
    device: &Arc<KaioDevice>,
    x: &Tensor,
    w_q: &Tensor,
    w_k: &Tensor,
    w_v: &Tensor,
    scale_q: f32,
    scale_k: f32,
    scale_v: f32,
) -> Result<(Tensor, Tensor, Tensor)> {
    // G3-1: reject gradient-tracked inputs.
    reject_if_variable(x, "x")?;
    reject_if_variable(w_q, "w_q")?;
    reject_if_variable(w_k, "w_k")?;
    reject_if_variable(w_v, "w_v")?;

    // Extract storage + layout from each input (G3-3: guards live in
    // this scope, Storage::Cuda match is inline).
    let (guard_x, layout_x) = x.storage_and_layout();
    let (guard_wq, layout_wq) = w_q.storage_and_layout();
    let (guard_wk, layout_wk) = w_k.storage_and_layout();
    let (guard_wv, layout_wv) = w_v.storage_and_layout();

    let s_x = match &*guard_x {
        Storage::Cuda(s) => s,
        _ => {
            return Err(Error::Msg(format!(
                "kaio-candle::{OP_NAME}: x must be on a CUDA device"
            )));
        }
    };
    let s_wq = match &*guard_wq {
        Storage::Cuda(s) => s,
        _ => {
            return Err(Error::Msg(format!(
                "kaio-candle::{OP_NAME}: w_q must be on a CUDA device"
            )));
        }
    };
    let s_wk = match &*guard_wk {
        Storage::Cuda(s) => s,
        _ => {
            return Err(Error::Msg(format!(
                "kaio-candle::{OP_NAME}: w_k must be on a CUDA device"
            )));
        }
    };
    let s_wv = match &*guard_wv {
        Storage::Cuda(s) => s,
        _ => {
            return Err(Error::Msg(format!(
                "kaio-candle::{OP_NAME}: w_v must be on a CUDA device"
            )));
        }
    };

    // Device-consistency: all inputs same CUDA ordinal, matching KaioDevice.
    let candle_dev = s_x.device.clone();
    bridge::ensure_ordinal_match(&candle_dev, device)?;
    bridge::ensure_ordinal_match(&s_wq.device, device)?;
    bridge::ensure_ordinal_match(&s_wk.device, device)?;
    bridge::ensure_ordinal_match(&s_wv.device, device)?;

    // Shape validation with named parameters.
    let (m, k_x) = bridge::ensure_rank2_contiguous_zero_offset_named(OP_NAME, "x", layout_x)?;
    let (k_wq, n_wq) =
        bridge::ensure_rank2_contiguous_zero_offset_named(OP_NAME, "w_q", layout_wq)?;
    let (k_wk, n_wk) =
        bridge::ensure_rank2_contiguous_zero_offset_named(OP_NAME, "w_k", layout_wk)?;
    let (k_wv, n_wv) =
        bridge::ensure_rank2_contiguous_zero_offset_named(OP_NAME, "w_v", layout_wv)?;

    // K dimension must match across x and all weights.
    if k_x != k_wq || k_x != k_wk || k_x != k_wv {
        return Err(Error::Msg(format!(
            "kaio-candle::{OP_NAME}: K mismatch — x has K={k_x}, \
             w_q has K={k_wq}, w_k has K={k_wk}, w_v has K={k_wv}. \
             All must share the same K dimension."
        )));
    }
    // N dimension must match across all weights.
    if n_wq != n_wk || n_wq != n_wv {
        return Err(Error::Msg(format!(
            "kaio-candle::{OP_NAME}: N mismatch — w_q has N={n_wq}, \
             w_k has N={n_wk}, w_v has N={n_wv}. \
             All weight tensors must share the same N dimension."
        )));
    }
    let n = n_wq;
    let k = k_x;

    let m_u32 =
        u32::try_from(m).map_err(|_| Error::Msg(format!("{OP_NAME}: M ({m}) exceeds u32")))?;
    let n_u32 =
        u32::try_from(n).map_err(|_| Error::Msg(format!("{OP_NAME}: N ({n}) exceeds u32")))?;
    let k_u32 =
        u32::try_from(k).map_err(|_| Error::Msg(format!("{OP_NAME}: K ({k}) exceeds u32")))?;

    // Dtype extraction: x is f16, weights are u8-as-i8.
    let x_slice = bridge::slice_ref_from_storage::<f16>(s_x)?;
    let wq_slice_u8 = bridge::slice_ref_from_storage::<u8>(s_wq)?;
    let wk_slice_u8 = bridge::slice_ref_from_storage::<u8>(s_wk)?;
    let wv_slice_u8 = bridge::slice_ref_from_storage::<u8>(s_wv)?;

    let wq_slice = bridge::reinterpret_u8_slice_as_i8(wq_slice_u8);
    let wk_slice = bridge::reinterpret_u8_slice_as_i8(wk_slice_u8);
    let wv_slice = bridge::reinterpret_u8_slice_as_i8(wv_slice_u8);

    let x_buf: &GpuBuffer<f16> = bridge::buffer_ref_from_slice_readonly(x_slice);
    let wq_buf: &GpuBuffer<i8> = bridge::buffer_ref_from_slice_readonly(wq_slice);
    let wk_buf: &GpuBuffer<i8> = bridge::buffer_ref_from_slice_readonly(wk_slice);
    let wv_buf: &GpuBuffer<i8> = bridge::buffer_ref_from_slice_readonly(wv_slice);

    // Allocate 3 output buffers (f16).
    let mut q_buf: GpuBuffer<f16> = device.alloc_zeros::<f16>(m * n).map_err(bridge::kaio_err)?;
    let mut k_buf: GpuBuffer<f16> = device.alloc_zeros::<f16>(m * n).map_err(bridge::kaio_err)?;
    let mut v_buf: GpuBuffer<f16> = device.alloc_zeros::<f16>(m * n).map_err(bridge::kaio_err)?;

    bridge::sync_before_launch(&candle_dev, device)?;

    kaio_qkv_project_int8(
        device, x_buf, wq_buf, wk_buf, wv_buf, scale_q, scale_k, scale_v, &mut q_buf, &mut k_buf,
        &mut v_buf, m_u32, n_u32, k_u32,
    )
    .map_err(bridge::kaio_err)?;

    bridge::sync_after_launch(&candle_dev, device)?;

    // Drop the read guards before wrapping outputs — we're done reading
    // input storage. (Not strictly required since they'd drop at fn end,
    // but makes the lifetime boundary explicit.)
    drop(guard_x);
    drop(guard_wq);
    drop(guard_wk);
    drop(guard_wv);

    // Wrap outputs into Tensors. from_storage internally uses
    // Layout::contiguous(shape) for the strides (G3-2).
    let shape = (m, n);
    let q_storage = Storage::Cuda(bridge::storage_from_slice::<f16>(
        q_buf.into_cuda_slice(),
        candle_dev.clone(),
    ));
    let k_storage = Storage::Cuda(bridge::storage_from_slice::<f16>(
        k_buf.into_cuda_slice(),
        candle_dev.clone(),
    ));
    let v_storage = Storage::Cuda(bridge::storage_from_slice::<f16>(
        v_buf.into_cuda_slice(),
        candle_dev,
    ));

    let q_tensor = Tensor::from_storage(q_storage, shape, BackpropOp::none(), false);
    let k_tensor = Tensor::from_storage(k_storage, shape, BackpropOp::none(), false);
    let v_tensor = Tensor::from_storage(v_storage, shape, BackpropOp::none(), false);

    Ok((q_tensor, k_tensor, v_tensor))
}
