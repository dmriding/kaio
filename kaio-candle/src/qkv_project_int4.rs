//! `qkv_project_int4` — direct-call bridge for the W4A16 fused
//! tri-output QKV projection kernel.
//!
//! Same direct-call pattern as [`qkv_project_int8`](crate::qkv_project_int8)
//! but with 7 input tensors (activations + 3 packed-INT4 weight tensors +
//! 3 f16 group-scale tensors). `group_size=128` is locked per the
//! kaio-ops kernel contract; `K` must be a multiple of 128.
//!
//! ## Output dtype
//!
//! Returns `DType::F16` — the fused kernel performs `f32→f16` conversion
//! internally. See module docs on `qkv_project_int8` for the rationale.
//!
//! ## Autograd
//!
//! Forward-only. Gradient-tracked inputs are rejected (Gemini G3-1).

use std::sync::Arc;

use candle_core::op::BackpropOp;
use candle_core::{Error, Result, Storage, Tensor};
use half::f16;
use kaio::prelude::{GpuBuffer, KaioDevice};
use kaio_ops::qkv_project_int4 as kaio_qkv_project_int4;

use crate::bridge;

const OP_NAME: &str = "qkv_project_int4";
const GROUP_SIZE: u32 = 128;

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

/// Fused W4A16 tri-output QKV projection via KAIO's tensor-core kernel.
///
/// Single kernel launch produces three `f16[M, N]` output tensors (Q, K, V)
/// from shared f16 activations, per-projection packed-INT4 weights, and
/// per-projection f16 group scales.
///
/// - `x`: `f16[M, K]` — activations. Contiguous, zero-offset.
/// - `w_q_packed`, `w_k_packed`, `w_v_packed`: `u32[K/8, N]` — 8 INT4
///   values packed per `u32`. Contiguous, zero-offset.
/// - `scales_q`, `scales_k`, `scales_v`: `f16[K/128, N]` — one f16 scale
///   per group of 128 elements. Contiguous, zero-offset.
/// - Returns: `(f16[M, N], f16[M, N], f16[M, N])` — (Q, K, V).
///
/// `K` must be a multiple of 128 (`group_size` locked). All inputs must be
/// on the same CUDA device, matching the supplied `Arc<KaioDevice>`.
/// Gradient-tracked inputs are rejected (forward-only).
///
/// Requires SM 8.0+ (Ampere or newer).
#[allow(clippy::too_many_arguments)]
pub fn qkv_project_int4(
    device: &Arc<KaioDevice>,
    x: &Tensor,
    w_q_packed: &Tensor,
    w_k_packed: &Tensor,
    w_v_packed: &Tensor,
    scales_q: &Tensor,
    scales_k: &Tensor,
    scales_v: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    // G3-1: reject gradient-tracked inputs.
    reject_if_variable(x, "x")?;
    reject_if_variable(w_q_packed, "w_q_packed")?;
    reject_if_variable(w_k_packed, "w_k_packed")?;
    reject_if_variable(w_v_packed, "w_v_packed")?;
    reject_if_variable(scales_q, "scales_q")?;
    reject_if_variable(scales_k, "scales_k")?;
    reject_if_variable(scales_v, "scales_v")?;

    // Extract storage + layout from all 7 inputs.
    let (guard_x, layout_x) = x.storage_and_layout();
    let (guard_wq, layout_wq) = w_q_packed.storage_and_layout();
    let (guard_wk, layout_wk) = w_k_packed.storage_and_layout();
    let (guard_wv, layout_wv) = w_v_packed.storage_and_layout();
    let (guard_sq, layout_sq) = scales_q.storage_and_layout();
    let (guard_sk, layout_sk) = scales_k.storage_and_layout();
    let (guard_sv, layout_sv) = scales_v.storage_and_layout();

    macro_rules! cuda_storage {
        ($guard:expr, $name:literal) => {
            match &*$guard {
                Storage::Cuda(s) => s,
                _ => {
                    return Err(Error::Msg(format!(
                        "kaio-candle::{OP_NAME}: {} must be on a CUDA device",
                        $name
                    )))
                }
            }
        };
    }

    let s_x = cuda_storage!(guard_x, "x");
    let s_wq = cuda_storage!(guard_wq, "w_q_packed");
    let s_wk = cuda_storage!(guard_wk, "w_k_packed");
    let s_wv = cuda_storage!(guard_wv, "w_v_packed");
    let s_sq = cuda_storage!(guard_sq, "scales_q");
    let s_sk = cuda_storage!(guard_sk, "scales_k");
    let s_sv = cuda_storage!(guard_sv, "scales_v");

    // Device consistency: all 7 inputs on same ordinal, matching KaioDevice.
    let candle_dev = s_x.device.clone();
    bridge::ensure_ordinal_match(&candle_dev, device)?;
    bridge::ensure_ordinal_match(&s_wq.device, device)?;
    bridge::ensure_ordinal_match(&s_wk.device, device)?;
    bridge::ensure_ordinal_match(&s_wv.device, device)?;
    bridge::ensure_ordinal_match(&s_sq.device, device)?;
    bridge::ensure_ordinal_match(&s_sk.device, device)?;
    bridge::ensure_ordinal_match(&s_sv.device, device)?;

    // Shape validation.
    let (m, k_x) = bridge::ensure_rank2_contiguous_zero_offset_named(OP_NAME, "x", layout_x)?;
    let (packed_rows_q, n_wq) =
        bridge::ensure_rank2_contiguous_zero_offset_named(OP_NAME, "w_q_packed", layout_wq)?;
    let (packed_rows_k, n_wk) =
        bridge::ensure_rank2_contiguous_zero_offset_named(OP_NAME, "w_k_packed", layout_wk)?;
    let (packed_rows_v, n_wv) =
        bridge::ensure_rank2_contiguous_zero_offset_named(OP_NAME, "w_v_packed", layout_wv)?;
    let (scale_rows_q, n_sq) =
        bridge::ensure_rank2_contiguous_zero_offset_named(OP_NAME, "scales_q", layout_sq)?;
    let (scale_rows_k, n_sk) =
        bridge::ensure_rank2_contiguous_zero_offset_named(OP_NAME, "scales_k", layout_sk)?;
    let (scale_rows_v, n_sv) =
        bridge::ensure_rank2_contiguous_zero_offset_named(OP_NAME, "scales_v", layout_sv)?;

    // K must be a multiple of GROUP_SIZE.
    if !k_x.is_multiple_of(GROUP_SIZE as usize) {
        return Err(Error::Msg(format!(
            "kaio-candle::{OP_NAME}: K ({k_x}) must be a multiple of group_size ({GROUP_SIZE})."
        )));
    }

    let expected_packed_rows = k_x / 8;
    let expected_scale_rows = k_x / GROUP_SIZE as usize;

    // Cross-check packed-weight row counts.
    for (name, actual) in [
        ("w_q_packed", packed_rows_q),
        ("w_k_packed", packed_rows_k),
        ("w_v_packed", packed_rows_v),
    ] {
        if actual != expected_packed_rows {
            return Err(Error::Msg(format!(
                "kaio-candle::{OP_NAME}: {name} has {actual} rows, expected K/8 = {expected_packed_rows} \
                 (K={k_x})."
            )));
        }
    }

    // Cross-check scale row counts.
    for (name, actual) in [
        ("scales_q", scale_rows_q),
        ("scales_k", scale_rows_k),
        ("scales_v", scale_rows_v),
    ] {
        if actual != expected_scale_rows {
            return Err(Error::Msg(format!(
                "kaio-candle::{OP_NAME}: {name} has {actual} rows, expected K/{GROUP_SIZE} = {expected_scale_rows} \
                 (K={k_x})."
            )));
        }
    }

    // N dimension must be consistent across all weight + scale tensors.
    let all_n = [n_wq, n_wk, n_wv, n_sq, n_sk, n_sv];
    let n = n_wq;
    for &ni in &all_n[1..] {
        if ni != n {
            return Err(Error::Msg(format!(
                "kaio-candle::{OP_NAME}: N mismatch across weight/scale tensors — \
                 w_q_packed N={n_wq}, w_k_packed N={n_wk}, w_v_packed N={n_wv}, \
                 scales_q N={n_sq}, scales_k N={n_sk}, scales_v N={n_sv}. All must match."
            )));
        }
    }
    let k = k_x;

    let m_u32 =
        u32::try_from(m).map_err(|_| Error::Msg(format!("{OP_NAME}: M ({m}) exceeds u32")))?;
    let n_u32 =
        u32::try_from(n).map_err(|_| Error::Msg(format!("{OP_NAME}: N ({n}) exceeds u32")))?;
    let k_u32 =
        u32::try_from(k).map_err(|_| Error::Msg(format!("{OP_NAME}: K ({k}) exceeds u32")))?;

    // Dtype extraction.
    let x_slice = bridge::slice_ref_from_storage::<f16>(s_x)?;
    let wq_slice = bridge::slice_ref_from_storage::<u32>(s_wq)?;
    let wk_slice = bridge::slice_ref_from_storage::<u32>(s_wk)?;
    let wv_slice = bridge::slice_ref_from_storage::<u32>(s_wv)?;
    let sq_slice = bridge::slice_ref_from_storage::<f16>(s_sq)?;
    let sk_slice = bridge::slice_ref_from_storage::<f16>(s_sk)?;
    let sv_slice = bridge::slice_ref_from_storage::<f16>(s_sv)?;

    let x_buf: &GpuBuffer<f16> = bridge::buffer_ref_from_slice_readonly(x_slice);
    let wq_buf: &GpuBuffer<u32> = bridge::buffer_ref_from_slice_readonly(wq_slice);
    let wk_buf: &GpuBuffer<u32> = bridge::buffer_ref_from_slice_readonly(wk_slice);
    let wv_buf: &GpuBuffer<u32> = bridge::buffer_ref_from_slice_readonly(wv_slice);
    let sq_buf: &GpuBuffer<f16> = bridge::buffer_ref_from_slice_readonly(sq_slice);
    let sk_buf: &GpuBuffer<f16> = bridge::buffer_ref_from_slice_readonly(sk_slice);
    let sv_buf: &GpuBuffer<f16> = bridge::buffer_ref_from_slice_readonly(sv_slice);

    // Allocate 3 output buffers (f16).
    let mut q_buf: GpuBuffer<f16> = device.alloc_zeros::<f16>(m * n).map_err(bridge::kaio_err)?;
    let mut k_buf: GpuBuffer<f16> = device.alloc_zeros::<f16>(m * n).map_err(bridge::kaio_err)?;
    let mut v_buf: GpuBuffer<f16> = device.alloc_zeros::<f16>(m * n).map_err(bridge::kaio_err)?;

    bridge::sync_before_launch(&candle_dev, device)?;

    kaio_qkv_project_int4(
        device, x_buf, wq_buf, wk_buf, wv_buf, sq_buf, sk_buf, sv_buf, &mut q_buf, &mut k_buf,
        &mut v_buf, m_u32, n_u32, k_u32, GROUP_SIZE,
    )
    .map_err(bridge::kaio_err)?;

    bridge::sync_after_launch(&candle_dev, device)?;

    // Drop read guards before wrapping outputs.
    drop(guard_x);
    drop(guard_wq);
    drop(guard_wk);
    drop(guard_wv);
    drop(guard_sq);
    drop(guard_sk);
    drop(guard_sv);

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
