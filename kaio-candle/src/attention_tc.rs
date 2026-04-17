//! `AttentionTcOp` (CustomOp3, `causal: bool` on the struct) + the two
//! user-facing wrappers [`attention_tc`] and [`attention_tc_causal`].
//!
//! Fused tensor-core scaled-dot-product attention. Rank-2 Q/K/V inputs;
//! multi-head attention callers must flatten `[heads, seq, d]` to
//! `[heads * seq, d]` or reshape to `[seq, d]` per-head before calling.

use std::sync::Arc;

use candle_core::{CpuStorage, CudaStorage, CustomOp3, Error, Layout, Result, Shape, Tensor};
use half::f16;
use kaio::prelude::{GpuBuffer, KaioDevice};
use kaio_ops::{
    attention_tc as kaio_attention_tc, attention_tc_causal as kaio_attention_tc_causal,
};

use crate::bridge;

/// Candle [`CustomOp3`] wrapper around `kaio_ops::attention_tc` /
/// `attention_tc_causal`.
///
/// Users call [`attention_tc`] or [`attention_tc_causal`] rather than
/// constructing this directly. The `causal` field selects the kernel
/// variant (masked scores for j > i when true).
pub struct AttentionTcOp {
    /// The KAIO device this op launches on. Must have the same CUDA
    /// ordinal as the input tensors' candle device.
    pub device: Arc<KaioDevice>,
    /// `true` → causal (decoder) mask, `false` → full attention.
    pub causal: bool,
}

impl CustomOp3 for AttentionTcOp {
    fn name(&self) -> &'static str {
        if self.causal {
            "kaio::attention_tc_causal"
        } else {
            "kaio::attention_tc"
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
             This op requires a CUDA device (SM 8.0+). \
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
        let op_name: &'static str = if self.causal {
            "attention_tc_causal"
        } else {
            "attention_tc"
        };

        let (seq_q, d_k_q) = bridge::ensure_rank2_contiguous_zero_offset(op_name, 0, l_q)?;
        let (seq_k_k, d_k_k) = bridge::ensure_rank2_contiguous_zero_offset(op_name, 1, l_k)?;
        let (seq_k_v, d_v) = bridge::ensure_rank2_contiguous_zero_offset(op_name, 2, l_v)?;

        // Shape contract (P5): Q [seq_q, d_k], K [seq_k, d_k], V [seq_k, d_v].
        if d_k_q != d_k_k {
            return Err(Error::Msg(format!(
                "kaio-candle::{op_name}: d_k mismatch between Q and K — \
                 Q has shape [{seq_q}, {d_k_q}], K has shape [{seq_k_k}, {d_k_k}]. \
                 Q and K must share the last dimension (head_dim)."
            )));
        }
        if seq_k_k != seq_k_v {
            return Err(Error::Msg(format!(
                "kaio-candle::{op_name}: seq_k mismatch between K and V — \
                 K has shape [{seq_k_k}, {d_k_k}], V has shape [{seq_k_v}, {d_v}]. \
                 K and V must share the first dimension (key sequence length)."
            )));
        }
        let seq_k = seq_k_k; // canonicalise now that the mismatch check passed
        let d_k = d_k_q;

        let seq_q_u32 = u32::try_from(seq_q)
            .map_err(|_| Error::Msg(format!("{op_name}: seq_q exceeds u32")))?;
        let seq_k_u32 = u32::try_from(seq_k)
            .map_err(|_| Error::Msg(format!("{op_name}: seq_k exceeds u32")))?;
        let d_k_u32 =
            u32::try_from(d_k).map_err(|_| Error::Msg(format!("{op_name}: d_k exceeds u32")))?;
        let d_v_u32 =
            u32::try_from(d_v).map_err(|_| Error::Msg(format!("{op_name}: d_v exceeds u32")))?;

        let candle_dev = s_q.device.clone();
        bridge::ensure_ordinal_match(&candle_dev, &self.device)?;

        // Dtype gate — Q/K/V all f16.
        let q_slice = bridge::slice_ref_from_storage::<f16>(s_q)?;
        let k_slice = bridge::slice_ref_from_storage::<f16>(s_k)?;
        let v_slice = bridge::slice_ref_from_storage::<f16>(s_v)?;

        // kaio_ops::attention_tc(_causal) reads Q/K/V into shared-mem
        // stagings and fragments; inputs never mutated. Safe under the
        // readonly contract.
        let q_buf: &GpuBuffer<f16> = bridge::buffer_ref_from_slice_readonly(q_slice);
        let k_buf: &GpuBuffer<f16> = bridge::buffer_ref_from_slice_readonly(k_slice);
        let v_buf: &GpuBuffer<f16> = bridge::buffer_ref_from_slice_readonly(v_slice);

        let mut out_buf: GpuBuffer<f32> = self
            .device
            .alloc_zeros::<f32>(seq_q * d_v)
            .map_err(bridge::kaio_err)?;

        bridge::sync_before_launch(&candle_dev, &self.device)?;

        let kernel_result = if self.causal {
            kaio_attention_tc_causal(
                &self.device,
                q_buf,
                k_buf,
                v_buf,
                &mut out_buf,
                seq_q_u32,
                seq_k_u32,
                d_k_u32,
                d_v_u32,
            )
        } else {
            kaio_attention_tc(
                &self.device,
                q_buf,
                k_buf,
                v_buf,
                &mut out_buf,
                seq_q_u32,
                seq_k_u32,
                d_k_u32,
                d_v_u32,
            )
        };
        kernel_result.map_err(bridge::kaio_err)?;

        bridge::sync_after_launch(&candle_dev, &self.device)?;

        let out_slice = out_buf.into_cuda_slice();
        let out_storage = bridge::storage_from_slice::<f32>(out_slice, candle_dev);
        Ok((out_storage, Shape::from_dims(&[seq_q, d_v])))
    }
}

/// Fused TC scaled-dot-product attention (non-causal).
///
/// - `q`: `f16[seq_q, d_k]`
/// - `k`: `f16[seq_k, d_k]`
/// - `v`: `f16[seq_k, d_v]`
/// - Returns: `f32[seq_q, d_v]`
///
/// All inputs must be contiguous, zero-offset, rank-2. Multi-head
/// attention callers should flatten `[heads, seq, d]` to `[heads * seq, d]`
/// or call per-head with rank-2 slices. Requires SM 8.0+.
pub fn attention_tc(
    device: &Arc<KaioDevice>,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
) -> Result<Tensor> {
    q.apply_op3(
        k,
        v,
        AttentionTcOp {
            device: device.clone(),
            causal: false,
        },
    )
}

/// Fused TC scaled-dot-product attention with decoder causal mask.
///
/// Same shape contract as [`attention_tc`]; scores at positions
/// `j > i` (query index `i`, key index `j`) are masked to `-3.4e38` so
/// `exp(score - row_max)` underflows to 0, matching the scalar
/// `attention_causal` semantics.
pub fn attention_tc_causal(
    device: &Arc<KaioDevice>,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
) -> Result<Tensor> {
    q.apply_op3(
        k,
        v,
        AttentionTcOp {
            device: device.clone(),
            causal: true,
        },
    )
}
