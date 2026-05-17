//! `MatmulTcBf16Op` (CustomOp2) + [`matmul_tc_bf16`] wrapper.
//!
//! bf16 × bf16 → f32 tensor-core matmul, bridging candle's Tensor API onto
//! `kaio_ops::matmul_tc_bf16`. See [crate-level docs](crate) for limitations
//! (contiguity, offset, rank-2, CUDA Graphs).
//!
//! **Forward-only in Sprint 9.1.3.** Backward arrives in Sprint 9.1.4 via
//! forward-reuse (mirror of `MatmulTcOp::bwd`).

use std::sync::Arc;

use candle_core::{CpuStorage, CudaStorage, CustomOp2, Error, Layout, Result, Shape, Tensor};
use half::bf16;
use kaio::prelude::{GpuBuffer, KaioDevice};
use kaio_ops::matmul_tc_bf16 as kaio_matmul_tc_bf16;

use crate::bridge;

/// Candle [`CustomOp2`] wrapper around [`kaio_ops::matmul_tc_bf16`].
///
/// Users call the free function [`matmul_tc_bf16`] rather than constructing
/// this directly. Carries the `Arc<KaioDevice>` into `cuda_fwd`.
///
/// **Forward-only in Sprint 9.1.3.** The `bwd()` override returns an
/// explicit `Err` naming Sprint 9.1.4 (which adds backward via the
/// forward-reuse pattern used by `MatmulTcOp`).
pub struct MatmulTcBf16Op {
    /// The KAIO device this op launches on. Must have the same CUDA
    /// ordinal as the input tensors' candle device (checked per-call via
    /// the private `bridge::ensure_ordinal_match`).
    pub device: Arc<KaioDevice>,
}

impl CustomOp2 for MatmulTcBf16Op {
    fn name(&self) -> &'static str {
        "kaio::matmul_tc_bf16"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        Err(Error::Msg(
            "kaio-candle::matmul_tc_bf16: CPU fallback not supported. \
             This op requires a CUDA device (bf16 variant requires SM 8.0+ \
             for bf16 mma). KAIO's value prop is GPU-specific PTX — falling \
             back to CPU would silently route around every perf claim. Call \
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
        let (m_a, k_a) = bridge::ensure_rank2_contiguous_zero_offset("matmul_tc_bf16", 0, l1)?;
        let (k_b, n_b) = bridge::ensure_rank2_contiguous_zero_offset("matmul_tc_bf16", 1, l2)?;
        if k_a != k_b {
            return Err(Error::Msg(format!(
                "kaio-candle::matmul_tc_bf16: K mismatch between inputs — \
                 input #0 has shape [{m_a}, {k_a}] (K = {k_a}), \
                 input #1 has shape [{k_b}, {n_b}] (K = {k_b}). \
                 Inner dimensions must match."
            )));
        }
        let m = u32::try_from(m_a)
            .map_err(|_| Error::Msg(format!("matmul_tc_bf16: M ({m_a}) exceeds u32")))?;
        let n = u32::try_from(n_b)
            .map_err(|_| Error::Msg(format!("matmul_tc_bf16: N ({n_b}) exceeds u32")))?;
        let k = u32::try_from(k_a)
            .map_err(|_| Error::Msg(format!("matmul_tc_bf16: K ({k_a}) exceeds u32")))?;

        // CudaStorage.device is a public field.
        let candle_dev = s1.device.clone();
        bridge::ensure_ordinal_match(&candle_dev, &self.device)?;

        // Dtype gate — kaio-ops matmul_tc_bf16 is bf16 × bf16 only.
        // as_cuda_slice::<bf16>() errors with candle's own dtype-mismatch
        // message if the storage isn't bf16.
        let a_slice = bridge::slice_ref_from_storage::<bf16>(s1)?;
        let b_slice = bridge::slice_ref_from_storage::<bf16>(s2)?;

        // Shared-borrow view into candle-owned buffers. Same lifetime and
        // aliasing contracts as `MatmulTcOp` — see
        // `bridge::buffer_ref_from_slice_readonly`. kaio_ops::matmul_tc_bf16
        // does not mutate its input GpuBuffers; inputs are read-only into
        // MMA fragments via shared memory.
        let a_buf: &GpuBuffer<bf16> = bridge::buffer_ref_from_slice_readonly(a_slice);
        let b_buf: &GpuBuffer<bf16> = bridge::buffer_ref_from_slice_readonly(b_slice);

        // Allocate the f32 output. Bridge owns this allocation end-to-end
        // until it's re-wrapped into CudaStorage below.
        let mut out_buf: GpuBuffer<f32> = self
            .device
            .alloc_zeros::<f32>(m_a * n_b)
            .map_err(bridge::kaio_err)?;

        bridge::sync_before_launch(&candle_dev, &self.device)?;

        kaio_matmul_tc_bf16(&self.device, a_buf, b_buf, &mut out_buf, m, n, k)
            .map_err(bridge::kaio_err)?;

        bridge::sync_after_launch(&candle_dev, &self.device)?;

        let out_slice = out_buf.into_cuda_slice();
        let out_storage = bridge::storage_from_slice::<f32>(out_slice, candle_dev);
        Ok((out_storage, Shape::from_dims(&[m_a, n_b])))
    }

    /// Backward pass: explicit `Err` naming Sprint 9.1.4.
    ///
    /// Sprint 9.1.3 ships forward-only. 9.1.4 adds backward via the same
    /// forward-reuse pattern used by `MatmulTcOp::bwd` (no new PTX needed —
    /// `dA = grad @ B^T`, `dB = A^T @ grad`, both via this kernel).
    ///
    /// Returning an explicit `Err` instead of falling through to the
    /// default `BackwardNotSupported` makes the diagnostic actionable:
    /// the user sees the sprint that fills the gap AND concrete workarounds.
    fn bwd(
        &self,
        _a: &Tensor,
        _b: &Tensor,
        _res: &Tensor,
        _grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        Err(Error::Msg(
            "matmul_tc_bf16 backward is sprint 9.1.4; not yet implemented — \
             use kaio_ops::matmul_tc_bf16 directly or downcast to f16"
                .to_string(),
        ))
    }
}

/// Matrix multiply two `bf16` tensors via KAIO's tensor-core kernel.
///
/// - `a`: `bf16[M, K]`, contiguous, zero-offset.
/// - `b`: `bf16[K, N]`, contiguous, zero-offset.
/// - Returns: `f32[M, N]`. Cast with `.to_dtype(DType::BF16)?` if you need
///   `bf16` for downstream graph continuation.
///
/// Requires SM 8.0+ (Ampere or newer; bf16 mma is sm_80+) and
/// `K % 16 == 0`.
///
/// **Forward-only in Sprint 9.1.3.** Calling `.backward()` on a graph
/// containing this op returns an explicit error pointing at Sprint 9.1.4
/// (which adds backward via forward-reuse).
///
/// See [crate-level docs](crate) for the full list of limitations
/// (contiguity/offset rejection, rank-2 only, CUDA Graph incompatibility,
/// bench-methodology caveat).
pub fn matmul_tc_bf16(device: &Arc<KaioDevice>, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    a.apply_op2(
        b,
        MatmulTcBf16Op {
            device: device.clone(),
        },
    )
}
