//! `MatmulTcBf16AsyncOp` (CustomOp2) + [`matmul_tc_bf16_async`] wrapper.
//!
//! bf16 × bf16 → f32 tensor-core matmul, `cp.async` variant. The async
//! sibling to [`matmul_tc_bf16`](super::matmul_tc_bf16::matmul_tc_bf16).
//! See [crate-level docs](crate) for limitations.
//!
//! Backward via forward-reuse pattern (mirror of `MatmulTcAsyncOp::bwd`):
//! `dA = grad @ B^T`, `dB = A^T @ grad`, both via this kernel — no new
//! PTX. See `MatmulTcBf16AsyncOp::bwd` for the precision note.
//!
//! For the bf16 precision contract and the `bf16 mma requires sm_80+` gate
//! rationale, see [`MatmulTcBf16Op`](super::matmul_tc_bf16::MatmulTcBf16Op)
//! — the contracts are identical; only the underlying kernel differs
//! (sync vs cp.async-pipelined).

use std::sync::Arc;

use candle_core::{
    CpuStorage, CudaStorage, CustomOp2, DType, Error, Layout, Result, Shape, Tensor,
};
use half::bf16;
use kaio::prelude::{GpuBuffer, KaioDevice};
use kaio_ops::matmul_tc_bf16_async as kaio_matmul_tc_bf16_async;

use crate::bridge;

/// Candle [`CustomOp2`] wrapper around [`kaio_ops::matmul_tc_bf16_async`].
///
/// Users call the free function [`matmul_tc_bf16_async`] rather than
/// constructing this directly.
///
/// Backward via forward-reuse — see [`MatmulTcBf16AsyncOp::bwd`].
pub struct MatmulTcBf16AsyncOp {
    /// The KAIO device this op launches on. Must have the same CUDA
    /// ordinal as the input tensors' candle device.
    pub device: Arc<KaioDevice>,
}

impl CustomOp2 for MatmulTcBf16AsyncOp {
    fn name(&self) -> &'static str {
        "kaio::matmul_tc_bf16_async"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        Err(Error::Msg(
            "kaio-candle::matmul_tc_bf16_async: CPU fallback not supported. \
             This op requires a CUDA device (bf16 variant requires SM 8.0+ \
             for bf16 mma; cp.async is sm_80+). Call \
             `.to_device(&Device::new_cuda(0)?)` on your tensors first."
                .to_string(),
        ))
    }

    fn cuda_fwd(
        &self,
        s1: &CudaStorage,
        l1: &Layout,
        s2: &CudaStorage,
        l2: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (m_a, k_a) =
            bridge::ensure_rank2_contiguous_zero_offset("matmul_tc_bf16_async", 0, l1)?;
        let (k_b, n_b) =
            bridge::ensure_rank2_contiguous_zero_offset("matmul_tc_bf16_async", 1, l2)?;
        if k_a != k_b {
            return Err(Error::Msg(format!(
                "kaio-candle::matmul_tc_bf16_async: K mismatch between inputs — \
                 input #0 has shape [{m_a}, {k_a}] (K = {k_a}), \
                 input #1 has shape [{k_b}, {n_b}] (K = {k_b}). \
                 Inner dimensions must match."
            )));
        }
        let m = u32::try_from(m_a)
            .map_err(|_| Error::Msg(format!("matmul_tc_bf16_async: M ({m_a}) exceeds u32")))?;
        let n = u32::try_from(n_b)
            .map_err(|_| Error::Msg(format!("matmul_tc_bf16_async: N ({n_b}) exceeds u32")))?;
        let k = u32::try_from(k_a)
            .map_err(|_| Error::Msg(format!("matmul_tc_bf16_async: K ({k_a}) exceeds u32")))?;

        // CudaStorage.device is a public field.
        let candle_dev = s1.device.clone();
        bridge::ensure_ordinal_match(&candle_dev, &self.device)?;

        // Dtype gate — bf16 × bf16.
        let a_slice = bridge::slice_ref_from_storage::<bf16>(s1)?;
        let b_slice = bridge::slice_ref_from_storage::<bf16>(s2)?;

        // kaio_ops::matmul_tc_bf16_async shares the same kernel structure
        // as matmul_tc_bf16 — validate_dims_tc (read-only), then cp.async
        // staged loads that READ from global into shared. Inputs never
        // mutated.
        let a_buf: &GpuBuffer<bf16> = bridge::buffer_ref_from_slice_readonly(a_slice);
        let b_buf: &GpuBuffer<bf16> = bridge::buffer_ref_from_slice_readonly(b_slice);

        let mut out_buf: GpuBuffer<f32> = self
            .device
            .alloc_zeros::<f32>(m_a * n_b)
            .map_err(bridge::kaio_err)?;

        bridge::sync_before_launch(&candle_dev, &self.device)?;

        kaio_matmul_tc_bf16_async(&self.device, a_buf, b_buf, &mut out_buf, m, n, k)
            .map_err(bridge::kaio_err)?;

        bridge::sync_after_launch(&candle_dev, &self.device)?;

        let out_slice = out_buf.into_cuda_slice();
        let out_storage = bridge::storage_from_slice::<f32>(out_slice, candle_dev);
        Ok((out_storage, Shape::from_dims(&[m_a, n_b])))
    }

    /// Backward pass: dA = grad @ B^T, dB = A^T @ grad.
    ///
    /// Reuses the forward `matmul_tc_bf16_async` kernel for both
    /// directions for consistent perf — no new PTX. See
    /// [`MatmulTcBf16Op::bwd`](super::matmul_tc_bf16::MatmulTcBf16Op)
    /// for the full precision + memory documentation; the bf16 precision
    /// note (double bf16 cast, ~7-bit mantissa vs f16's 10-bit, 8-bit
    /// exponent range advantage) applies identically.
    fn bwd(
        &self,
        a: &Tensor,
        b: &Tensor,
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        // grad_res is f32 [M, N]; matmul_tc_bf16_async needs bf16 inputs.
        let grad_bf16 = grad_res.to_dtype(DType::BF16)?;

        // dA = grad @ B^T → f32 [M, K]
        let b_t = b.t()?.contiguous()?;
        let grad_a = matmul_tc_bf16_async(&self.device, &grad_bf16, &b_t)?;

        // dB = A^T @ grad → f32 [K, N]
        let a_t = a.t()?.contiguous()?;
        let grad_b = matmul_tc_bf16_async(&self.device, &a_t, &grad_bf16)?;

        // Cast output gradients to bf16 to match input dtypes.
        Ok((
            Some(grad_a.to_dtype(DType::BF16)?),
            Some(grad_b.to_dtype(DType::BF16)?),
        ))
    }
}

/// Matrix multiply two `bf16` tensors via KAIO's `cp.async` tensor-core
/// kernel (sibling to [`matmul_tc_bf16`](super::matmul_tc_bf16::matmul_tc_bf16),
/// async-pipelined K-loop variant from Sprint 9.1.1).
///
/// Same input contract as
/// [`matmul_tc_bf16`](super::matmul_tc_bf16::matmul_tc_bf16): rank-2,
/// contiguous, zero-offset, `K % 16 == 0`, SM 8.0+.
///
/// **Backward supported** via the forward-reuse pattern in
/// [`MatmulTcBf16AsyncOp::bwd`] (no new PTX; mirrors the f16 async
/// sibling from Sprint 7.4d).
///
/// See [crate-level docs](crate) for the full list of limitations.
pub fn matmul_tc_bf16_async(device: &Arc<KaioDevice>, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    a.apply_op2(
        b,
        MatmulTcBf16AsyncOp {
            device: device.clone(),
        },
    )
}
