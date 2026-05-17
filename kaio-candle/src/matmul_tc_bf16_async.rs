//! `MatmulTcBf16AsyncOp` (CustomOp2) + [`matmul_tc_bf16_async`] wrapper.
//!
//! bf16 × bf16 → f32 tensor-core matmul, `cp.async` variant. The async
//! sibling to [`matmul_tc_bf16`](super::matmul_tc_bf16::matmul_tc_bf16).
//! See [crate-level docs](crate) for limitations.
//!
//! **Forward-only in Sprint 9.1.3.** Backward arrives in Sprint 9.1.4 via
//! forward-reuse (mirror of `MatmulTcAsyncOp::bwd`).
//!
//! For the bf16 precision contract and the `bf16 mma requires sm_80+` gate
//! rationale, see [`MatmulTcBf16Op`](super::matmul_tc_bf16::MatmulTcBf16Op)
//! — the contracts are identical; only the underlying kernel differs
//! (sync vs cp.async-pipelined).

use std::sync::Arc;

use candle_core::{CpuStorage, CudaStorage, CustomOp2, Error, Layout, Result, Shape, Tensor};
use half::bf16;
use kaio::prelude::{GpuBuffer, KaioDevice};
use kaio_ops::matmul_tc_bf16_async as kaio_matmul_tc_bf16_async;

use crate::bridge;

/// Candle [`CustomOp2`] wrapper around [`kaio_ops::matmul_tc_bf16_async`].
///
/// Users call the free function [`matmul_tc_bf16_async`] rather than
/// constructing this directly.
///
/// **Forward-only in Sprint 9.1.3.** The `bwd()` override returns an
/// explicit `Err` naming Sprint 9.1.4.
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

    /// Backward pass: explicit `Err` naming Sprint 9.1.4.
    ///
    /// Sprint 9.1.3 ships forward-only. 9.1.4 adds backward via the same
    /// forward-reuse pattern used by `MatmulTcAsyncOp::bwd` (the async
    /// kernel for both directions for consistent perf).
    ///
    /// See [`MatmulTcBf16Op::bwd`](super::matmul_tc_bf16::MatmulTcBf16Op)
    /// for the rationale on explicit-`Err` vs default `BackwardNotSupported`.
    fn bwd(
        &self,
        _a: &Tensor,
        _b: &Tensor,
        _res: &Tensor,
        _grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        Err(Error::Msg(
            "matmul_tc_bf16_async backward is sprint 9.1.4; not yet implemented — \
             use kaio_ops::matmul_tc_bf16_async directly or downcast to f16"
                .to_string(),
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
/// **Forward-only in Sprint 9.1.3.** Calling `.backward()` on a graph
/// containing this op returns an explicit error pointing at Sprint 9.1.4.
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
