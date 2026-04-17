//! `MatmulInt8Op` (CustomOp2, `scale: f32` on the struct) + [`matmul_int8`]
//! wrapper.
//!
//! W8A8 symmetric-quant matmul: `i8 × i8 → f32` with a single scalar
//! `f32` scale factor applied in the accumulator. See
//! [crate-level docs](crate) for the shared limitations (contiguity,
//! offset, rank-2, CUDA Graphs, f32 output).
//!
//! ## Dtype convention: `DType::U8` on the candle side
//!
//! `kaio_ops::matmul_int8` takes `GpuBuffer<i8>`, but candle's `DType`
//! enum has no `I8` variant (see `candle_core::DType` — only `U8` at
//! byte-width). Users quantize activations/weights to the signed INT8
//! range (`-128..=127`) and store the bytes in `DType::U8` tensors —
//! this is the established candle convention for INT8 quant models.
//! The bridge reinterprets the `u8` storage as `i8` via a
//! `#[repr(transparent)]` same-layout transmute (sound: identical size
//! and alignment, phantom-T type parameter in `cudarc::CudaSlice`).

use std::sync::Arc;

use candle_core::{CpuStorage, CudaStorage, CustomOp2, Error, Layout, Result, Shape, Tensor};
use kaio::prelude::{GpuBuffer, KaioDevice};
use kaio_ops::matmul_int8 as kaio_matmul_int8;

use crate::bridge;

/// Candle [`CustomOp2`] wrapper around [`kaio_ops::matmul_int8`].
///
/// Users call [`matmul_int8`] rather than constructing this directly.
/// The `scale` field is the scalar quant scale applied by the kernel
/// during accumulation; it is threaded through the op struct so it
/// reaches `cuda_fwd` (candle's `CustomOp2` only hands through the two
/// tensor inputs — scalars ride on the op itself).
pub struct MatmulInt8Op {
    /// The KAIO device this op launches on. Must have the same CUDA
    /// ordinal as the input tensors' candle device.
    pub device: Arc<KaioDevice>,
    /// Scalar quant scale applied in the kernel accumulator.
    /// Typical realistic value: `max_abs / 127` for symmetric INT8.
    pub scale: f32,
}

impl CustomOp2 for MatmulInt8Op {
    fn name(&self) -> &'static str {
        "kaio::matmul_int8"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        Err(Error::Msg(
            "kaio-candle::matmul_int8: CPU fallback not supported. \
             This op requires a CUDA device (SM 8.0+). \
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
        let (m_a, k_a) = bridge::ensure_rank2_contiguous_zero_offset("matmul_int8", 0, l1)?;
        let (k_b, n_b) = bridge::ensure_rank2_contiguous_zero_offset("matmul_int8", 1, l2)?;
        if k_a != k_b {
            return Err(Error::Msg(format!(
                "kaio-candle::matmul_int8: K mismatch between inputs — \
                 input #0 has shape [{m_a}, {k_a}] (K = {k_a}), \
                 input #1 has shape [{k_b}, {n_b}] (K = {k_b}). \
                 Inner dimensions must match."
            )));
        }
        let m = u32::try_from(m_a)
            .map_err(|_| Error::Msg(format!("matmul_int8: M ({m_a}) exceeds u32")))?;
        let n = u32::try_from(n_b)
            .map_err(|_| Error::Msg(format!("matmul_int8: N ({n_b}) exceeds u32")))?;
        let k = u32::try_from(k_a)
            .map_err(|_| Error::Msg(format!("matmul_int8: K ({k_a}) exceeds u32")))?;

        let candle_dev = s1.device.clone();
        bridge::ensure_ordinal_match(&candle_dev, &self.device)?;

        // Dtype gate — candle has no DType::I8, so the convention is
        // DType::U8 with the bytes interpreted as signed INT8.
        let a_slice_u8 = bridge::slice_ref_from_storage::<u8>(s1)?;
        let b_slice_u8 = bridge::slice_ref_from_storage::<u8>(s2)?;
        let a_slice = bridge::reinterpret_u8_slice_as_i8(a_slice_u8);
        let b_slice = bridge::reinterpret_u8_slice_as_i8(b_slice_u8);

        // kaio_ops::matmul_int8 reads inputs through LDMATRIX into
        // fragment A/B; never mutated. Safe under the readonly contract.
        let a_buf: &GpuBuffer<i8> = bridge::buffer_ref_from_slice_readonly(a_slice);
        let b_buf: &GpuBuffer<i8> = bridge::buffer_ref_from_slice_readonly(b_slice);

        let mut out_buf: GpuBuffer<f32> = self
            .device
            .alloc_zeros::<f32>(m_a * n_b)
            .map_err(bridge::kaio_err)?;

        bridge::sync_before_launch(&candle_dev, &self.device)?;

        kaio_matmul_int8(
            &self.device,
            a_buf,
            b_buf,
            &mut out_buf,
            self.scale,
            m,
            n,
            k,
        )
        .map_err(bridge::kaio_err)?;

        bridge::sync_after_launch(&candle_dev, &self.device)?;

        let out_slice = out_buf.into_cuda_slice();
        let out_storage = bridge::storage_from_slice::<f32>(out_slice, candle_dev);
        Ok((out_storage, Shape::from_dims(&[m_a, n_b])))
    }
}

/// W8A8 symmetric-quant matmul via KAIO's `mma.s8.s8.s32` kernel.
///
/// - `a`: `u8[M, K]` where the bytes are interpreted as signed INT8
///   (`-128..=127`). Contiguous, zero-offset.
/// - `b`: `u8[K, N]`, same convention. Contiguous, zero-offset.
/// - `scale`: scalar `f32` quant scale applied in the kernel accumulator
///   (typical value: `max_abs / 127` for symmetric INT8).
/// - Returns: `f32[M, N]`. Cast via `.to_dtype(DType::F16)?` if you need
///   f16 for downstream graph continuation.
///
/// See the module-level docs for the `DType::U8`-as-INT8 dtype
/// convention (candle has no `DType::I8`).
///
/// Requires SM 8.0+ (Ampere or newer).
///
/// See [crate-level docs](crate) for the full list of limitations
/// (contiguity/offset rejection, rank-2 only, CUDA Graph incompatibility,
/// bench-methodology caveat).
pub fn matmul_int8(device: &Arc<KaioDevice>, a: &Tensor, b: &Tensor, scale: f32) -> Result<Tensor> {
    a.apply_op2(
        b,
        MatmulInt8Op {
            device: device.clone(),
            scale,
        },
    )
}
