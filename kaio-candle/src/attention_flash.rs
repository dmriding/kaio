//! `AttentionFlashOp` (CustomOp3, `causal: bool` on the struct) + the
//! two user-facing wrappers [`attention_flash`] and
//! [`attention_flash_causal`].
//!
//! FlashAttention: single-head scaled-dot-product attention without
//! materializing the O(seq²) score matrix. f32 end-to-end, rank-2
//! self-attention only — Q, K, V all `[seq_len, d_k]` (no cross-
//! attention shapes; `attention_tc` accepts `seq_q ≠ seq_k`, this op
//! deliberately does not). Multi-head callers flatten `[heads, seq, d]`
//! to `[heads * seq, d]` or call per-head.

use std::sync::Arc;

use candle_core::{CpuStorage, CudaStorage, CustomOp3, Error, Layout, Result, Shape, Tensor};
use kaio::prelude::{GpuBuffer, KaioDevice};
use kaio_ops::{
    attention_flash as kaio_attention_flash, attention_flash_causal as kaio_attention_flash_causal,
};

use crate::bridge;

/// Candle [`CustomOp3`] wrapper around `kaio_ops::attention_flash` /
/// `attention_flash_causal`.
///
/// Users call [`attention_flash`] or [`attention_flash_causal`] rather
/// than constructing this directly. The `causal` field selects the
/// kernel variant (masked scores for `j > i` when true).
pub struct AttentionFlashOp {
    /// The KAIO device this op launches on. Must have the same CUDA
    /// ordinal as the input tensors' candle device.
    pub device: Arc<KaioDevice>,
    /// `true` → causal (decoder) mask, `false` → full attention.
    pub causal: bool,
}

impl AttentionFlashOp {
    fn op_name(&self) -> &'static str {
        if self.causal {
            "attention_flash_causal"
        } else {
            "attention_flash"
        }
    }

    /// Validates the rank-2 self-attention shape contract shared by
    /// fwd and bwd: Q, K, V all `[seq_len, d_k]`, contiguous,
    /// zero-offset. Returns `(seq_len, d_k)`.
    fn ensure_square_qkv(
        &self,
        l_q: &Layout,
        l_k: &Layout,
        l_v: &Layout,
    ) -> Result<(usize, usize)> {
        let op_name = self.op_name();
        let (seq_q, d_q) = bridge::ensure_rank2_contiguous_zero_offset(op_name, 0, l_q)?;
        let (seq_k, d_kk) = bridge::ensure_rank2_contiguous_zero_offset(op_name, 1, l_k)?;
        let (seq_v, d_v) = bridge::ensure_rank2_contiguous_zero_offset(op_name, 2, l_v)?;

        // Self-attention contract: one seq_len, d_v == d_k. Reject the
        // cross-attention-ish shapes attention_tc would accept.
        if seq_q != seq_k || seq_q != seq_v || d_q != d_kk || d_q != d_v {
            return Err(Error::Msg(format!(
                "kaio-candle::{op_name}: Q, K, V must all be [seq_len, d_k] \
                 (single-head self-attention; d_v == d_k) — got Q [{seq_q}, {d_q}], \
                 K [{seq_k}, {d_kk}], V [{seq_v}, {d_v}]. For cross-attention \
                 shapes use attention_tc, which accepts seq_q != seq_k."
            )));
        }
        Ok((seq_q, d_q))
    }
}

impl CustomOp3 for AttentionFlashOp {
    fn name(&self) -> &'static str {
        if self.causal {
            "kaio::attention_flash_causal"
        } else {
            "kaio::attention_flash"
        }
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
        _s3: &CpuStorage,
        _l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        Err(Error::Msg(format!(
            "kaio-candle::{}: CPU fallback not supported. \
             This op requires a CUDA device. \
             Call `.to_device(&Device::new_cuda(0)?)` on your tensors first.",
            self.name()
        )))
    }

    fn cuda_fwd(
        &self,
        s_q: &CudaStorage,
        l_q: &Layout,
        s_k: &CudaStorage,
        l_k: &Layout,
        s_v: &CudaStorage,
        l_v: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let op_name = self.op_name();
        let (seq_len, d_k) = self.ensure_square_qkv(l_q, l_k, l_v)?;

        let seq_u32 = u32::try_from(seq_len)
            .map_err(|_| Error::Msg(format!("{op_name}: seq_len exceeds u32")))?;
        let d_k_u32 =
            u32::try_from(d_k).map_err(|_| Error::Msg(format!("{op_name}: d_k exceeds u32")))?;

        let candle_dev = s_q.device.clone();
        bridge::ensure_ordinal_match(&candle_dev, &self.device)?;

        // Dtype gate — Q/K/V all f32 (the flash kernels are f32-only).
        let q_slice = bridge::slice_ref_from_storage::<f32>(s_q)?;
        let k_slice = bridge::slice_ref_from_storage::<f32>(s_k)?;
        let v_slice = bridge::slice_ref_from_storage::<f32>(s_v)?;

        // The flash kernels only read Q/K/V; safe under the readonly
        // contract.
        let q_buf: &GpuBuffer<f32> = bridge::buffer_ref_from_slice_readonly(q_slice);
        let k_buf: &GpuBuffer<f32> = bridge::buffer_ref_from_slice_readonly(k_slice);
        let v_buf: &GpuBuffer<f32> = bridge::buffer_ref_from_slice_readonly(v_slice);

        let mut out_buf: GpuBuffer<f32> = self
            .device
            .alloc_zeros::<f32>(seq_len * d_k)
            .map_err(bridge::kaio_err)?;

        bridge::sync_before_launch(&candle_dev, &self.device)?;

        // Plain forward — no stats. The backward recovers the logsumexp
        // stats by re-running the `_with_stats` variant at bwd time
        // (candle's CustomOp3 has no saved-intermediate channel), so a
        // stats buffer written here would have nowhere to go.
        let kernel_result = if self.causal {
            kaio_attention_flash_causal(
                &self.device,
                q_buf,
                k_buf,
                v_buf,
                &mut out_buf,
                seq_u32,
                d_k_u32,
            )
        } else {
            kaio_attention_flash(
                &self.device,
                q_buf,
                k_buf,
                v_buf,
                &mut out_buf,
                seq_u32,
                d_k_u32,
            )
        };
        kernel_result.map_err(bridge::kaio_err)?;

        bridge::sync_after_launch(&candle_dev, &self.device)?;

        let out_slice = out_buf.into_cuda_slice();
        let out_storage = bridge::storage_from_slice::<f32>(out_slice, candle_dev);
        Ok((out_storage, Shape::from_dims(&[seq_len, d_k])))
    }
}

/// FlashAttention (non-causal) on candle tensors.
///
/// - `q`, `k`, `v`: `f32[seq_len, d_k]` — single-head self-attention,
///   all three the same shape (`d_v == d_k`)
/// - Returns: `f32[seq_len, d_k]`
///
/// All inputs must be contiguous, zero-offset, rank-2, on the same
/// CUDA device as `device`. `d_k ≤ 256` (kernel contract). No O(seq²)
/// memory — no `seq_k` cap, unlike `attention_tc`.
pub fn attention_flash(
    device: &Arc<KaioDevice>,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
) -> Result<Tensor> {
    q.apply_op3(
        k,
        v,
        AttentionFlashOp {
            device: device.clone(),
            causal: false,
        },
    )
}

/// FlashAttention with decoder causal mask.
///
/// Same shape contract as [`attention_flash`]; scores at positions
/// `j > i` (query index `i`, key index `j`) are masked so they
/// contribute zero probability, matching `attention_flash_causal`
/// semantics in kaio-ops.
pub fn attention_flash_causal(
    device: &Arc<KaioDevice>,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
) -> Result<Tensor> {
    q.apply_op3(
        k,
        v,
        AttentionFlashOp {
            device: device.clone(),
            causal: true,
        },
    )
}
