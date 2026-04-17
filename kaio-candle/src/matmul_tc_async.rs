//! `MatmulTcAsyncOp` (CustomOp2) + [`matmul_tc_async`] wrapper.
//!
//! f16 × f16 → f32 tensor-core matmul, `cp.async` variant (Sprint 6.7b —
//! 92.5% cuBLAS sgemm at 4096² on RTX 4090 sm_89 when called directly;
//! bridge calls include AD9 sync-fence overhead). See
//! [crate-level docs](crate) for limitations.

use std::sync::Arc;

use candle_core::{
    CpuStorage, CudaStorage, CustomOp2, DType, Error, Layout, Result, Shape, Tensor,
};
use half::f16;
use kaio::prelude::{GpuBuffer, KaioDevice};
use kaio_ops::matmul_tc_async as kaio_matmul_tc_async;

use crate::bridge;

/// Candle [`CustomOp2`] wrapper around [`kaio_ops::matmul_tc_async`].
///
/// Users call the free function [`matmul_tc_async`] rather than
/// constructing this directly.
pub struct MatmulTcAsyncOp {
    /// The KAIO device this op launches on. Must have the same CUDA
    /// ordinal as the input tensors' candle device.
    pub device: Arc<KaioDevice>,
}

impl CustomOp2 for MatmulTcAsyncOp {
    fn name(&self) -> &'static str {
        "kaio::matmul_tc_async"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        Err(Error::Msg(
            "kaio-candle::matmul_tc_async: CPU fallback not supported. \
             This op requires a CUDA device (SM 8.0+, cp.async is sm_80+). \
             Call `.to_device(&Device::new_cuda(0)?)` on your tensors first."
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
        // AD4: rank + contiguity + offset gate on each input.
        let (m_a, k_a) = bridge::ensure_rank2_contiguous_zero_offset("matmul_tc_async", 0, l1)?;
        let (k_b, n_b) = bridge::ensure_rank2_contiguous_zero_offset("matmul_tc_async", 1, l2)?;
        if k_a != k_b {
            return Err(Error::Msg(format!(
                "kaio-candle::matmul_tc_async: K mismatch between inputs — \
                 input #0 has shape [{m_a}, {k_a}] (K = {k_a}), \
                 input #1 has shape [{k_b}, {n_b}] (K = {k_b}). \
                 Inner dimensions must match."
            )));
        }
        let m = u32::try_from(m_a)
            .map_err(|_| Error::Msg(format!("matmul_tc_async: M ({m_a}) exceeds u32")))?;
        let n = u32::try_from(n_b)
            .map_err(|_| Error::Msg(format!("matmul_tc_async: N ({n_b}) exceeds u32")))?;
        let k = u32::try_from(k_a)
            .map_err(|_| Error::Msg(format!("matmul_tc_async: K ({k_a}) exceeds u32")))?;

        // CudaStorage.device is a public field.
        let candle_dev = s1.device.clone();
        bridge::ensure_ordinal_match(&candle_dev, &self.device)?;

        // AD4: dtype gate — f16 × f16.
        let a_slice = bridge::slice_ref_from_storage::<f16>(s1)?;
        let b_slice = bridge::slice_ref_from_storage::<f16>(s2)?;

        // AD2-Audit D3: kaio_ops::matmul_tc_async shares the same kernel
        // structure as matmul_tc — validate_dims_tc (read-only), then
        // cp.async staged loads that READ from global into shared. Inputs
        // never mutated.
        let a_buf: &GpuBuffer<f16> = bridge::buffer_ref_from_slice_readonly(a_slice);
        let b_buf: &GpuBuffer<f16> = bridge::buffer_ref_from_slice_readonly(b_slice);

        let mut out_buf: GpuBuffer<f32> = self
            .device
            .alloc_zeros::<f32>(m_a * n_b)
            .map_err(bridge::kaio_err)?;

        // AD9: stream-safety fences.
        bridge::sync_before_launch(&candle_dev, &self.device)?;

        kaio_matmul_tc_async(&self.device, a_buf, b_buf, &mut out_buf, m, n, k)
            .map_err(bridge::kaio_err)?;

        bridge::sync_after_launch(&candle_dev, &self.device)?;

        let out_slice = out_buf.into_cuda_slice();
        let out_storage = bridge::storage_from_slice::<f32>(out_slice, candle_dev);
        Ok((out_storage, Shape::from_dims(&[m_a, n_b])))
    }

    /// Backward pass: dA = grad @ B^T, dB = A^T @ grad.
    ///
    /// Uses the `matmul_tc_async` forward kernel for backward — same
    /// cp.async variant in both directions for consistent perf.
    /// See [`MatmulTcOp::bwd`](super::matmul_tc::MatmulTcOp) for the
    /// full precision + memory documentation; identical constraints apply.
    fn bwd(
        &self,
        a: &Tensor,
        b: &Tensor,
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let grad_f16 = grad_res.to_dtype(DType::F16)?;

        let b_t = b.t()?.contiguous()?;
        let grad_a = matmul_tc_async(&self.device, &grad_f16, &b_t)?;

        let a_t = a.t()?.contiguous()?;
        let grad_b = matmul_tc_async(&self.device, &a_t, &grad_f16)?;

        Ok((
            Some(grad_a.to_dtype(DType::F16)?),
            Some(grad_b.to_dtype(DType::F16)?),
        ))
    }
}

/// Matrix multiply two `f16` tensors via KAIO's `cp.async` tensor-core
/// kernel (Sprint 6.7b — 92.5% cuBLAS sgemm at 4096² direct-call).
///
/// Same input contract as [`matmul_tc`](super::matmul_tc::matmul_tc).
///
/// See [crate-level docs](crate) for limitations.
pub fn matmul_tc_async(device: &Arc<KaioDevice>, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    a.apply_op2(
        b,
        MatmulTcAsyncOp {
            device: device.clone(),
        },
    )
}
