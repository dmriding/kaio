//! `MatmulTcBf16Op` (CustomOp2) + [`matmul_tc_bf16`] wrapper.
//!
//! bf16 × bf16 → f32 tensor-core matmul, bridging candle's Tensor API onto
//! `kaio_ops::matmul_tc_bf16`. See [crate-level docs](crate) for limitations
//! (contiguity, offset, rank-2, CUDA Graphs).
//!
//! Backward via forward-reuse pattern (mirror of `MatmulTcOp::bwd`):
//! `dA = grad @ B^T`, `dB = A^T @ grad`, both via this kernel — no new
//! PTX. See `MatmulTcBf16Op::bwd` for the precision note.

use std::sync::Arc;

use candle_core::{
    CpuStorage, CudaStorage, CustomOp2, DType, Error, Layout, Result, Shape, Tensor,
};
use half::bf16;
use kaio::prelude::{GpuBuffer, KaioDevice};
use kaio_ops::matmul_tc_bf16 as kaio_matmul_tc_bf16;

use crate::bridge;

/// Candle [`CustomOp2`] wrapper around [`kaio_ops::matmul_tc_bf16`].
///
/// Users call the free function [`matmul_tc_bf16`] rather than constructing
/// this directly. Carries the `Arc<KaioDevice>` into `cuda_fwd`.
///
/// Backward via forward-reuse — see [`MatmulTcBf16Op::bwd`].
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

    /// Backward pass: dA = grad @ B^T, dB = A^T @ grad.
    ///
    /// Reuses the forward `matmul_tc_bf16` kernel — no new PTX. The f32
    /// `grad_res` is downcast to bf16 before each matmul call, and the
    /// f32 output gradients are cast back to bf16 to match the input
    /// dtypes (candle's gradient accumulator requires matching dtypes —
    /// verified in `backprop.rs:672`).
    ///
    /// **Precision note:** the double bf16 cast (input grad + output
    /// grad) is a known approximation. bf16's 7-bit mantissa is lower
    /// precision than f16's 10-bit, so per-element quantization noise
    /// from this round-trip is higher in absolute terms; bf16's 8-bit
    /// exponent gives values representable at scales where f16 would
    /// overflow or underflow. The dual-tolerance gradient check
    /// (`rel < 1e-2 || abs < 1e-3`, identical to f16) covers the
    /// shapes tested in `candle_gpu_roundtrip.rs`; larger shapes or
    /// different magnitude regimes may require recalibration.
    ///
    /// **Memory:** allocates two materialized transposes (`.t()?.contiguous()?`)
    /// plus the casted `grad_res`. Peak backward memory ≈ 2-3× forward
    /// input size.
    fn bwd(
        &self,
        a: &Tensor,
        b: &Tensor,
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        // grad_res is f32 [M, N]; matmul_tc_bf16 needs bf16 inputs.
        let grad_bf16 = grad_res.to_dtype(DType::BF16)?;

        // dA = grad @ B^T → f32 [M, K]
        let b_t = b.t()?.contiguous()?;
        let grad_a = matmul_tc_bf16(&self.device, &grad_bf16, &b_t)?;

        // dB = A^T @ grad → f32 [K, N]
        let a_t = a.t()?.contiguous()?;
        let grad_b = matmul_tc_bf16(&self.device, &a_t, &grad_bf16)?;

        // Cast output gradients to bf16 to match input dtypes.
        // Candle's gradient accumulation (backprop.rs:672) uses
        // sum_grad.add(&arg_grad) without auto-casting — dtype
        // mismatch would error.
        Ok((
            Some(grad_a.to_dtype(DType::BF16)?),
            Some(grad_b.to_dtype(DType::BF16)?),
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
/// **Backward supported** via the forward-reuse pattern in
/// [`MatmulTcBf16Op::bwd`] (no new PTX; mirrors the f16 sibling from
/// Sprint 7.4d).
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
