//! `MatmulInt4Op` (CustomOp3) + [`matmul_int4`] wrapper.
//!
//! GPTQ-style INT4 dequantize-matmul (W4A16): f16 activation × packed-INT4
//! weights × f16 group scales → f32 accumulator. `group_size` is locked to
//! 128 (kaio-ops contract); `K` must be a multiple of 128. See
//! [crate-level docs](crate) for the broader limitations (contiguity,
//! offset, rank-2, CUDA Graphs).

use std::sync::Arc;

use candle_core::{CpuStorage, CudaStorage, CustomOp3, Error, Layout, Result, Shape, Tensor};
use half::f16;
use kaio::prelude::{GpuBuffer, KaioDevice};
use kaio_ops::matmul_int4 as kaio_matmul_int4;

use crate::bridge;

/// Fixed group size expected by `kaio_ops::matmul_int4`. K must be a
/// multiple of this. Reference: `matmul_int4_kernel::GROUP_SIZE`.
const GROUP_SIZE: u32 = 128;

/// Candle [`CustomOp3`] wrapper around [`kaio_ops::matmul_int4`].
///
/// Users call the free function [`matmul_int4`] rather than constructing
/// this directly.
pub struct MatmulInt4Op {
    /// The KAIO device this op launches on. Must have the same CUDA
    /// ordinal as the input tensors' candle device.
    pub device: Arc<KaioDevice>,
}

impl CustomOp3 for MatmulInt4Op {
    fn name(&self) -> &'static str {
        "kaio::matmul_int4"
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
        Err(Error::Msg(
            "kaio-candle::matmul_int4: CPU fallback not supported. \
             This op requires a CUDA device (SM 8.0+). \
             Call `.to_device(&Device::new_cuda(0)?)` on your tensors first."
                .to_string(),
        ))
    }

    fn cuda_fwd(
        &self,
        s_a: &CudaStorage,
        l_a: &Layout,
        s_b: &CudaStorage,
        l_b: &Layout,
        s_s: &CudaStorage,
        l_s: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        // AD4: rank + contiguity + offset gate on every input.
        let (m_a, k_a) = bridge::ensure_rank2_contiguous_zero_offset("matmul_int4", 0, l_a)?;
        let (packed_rows, n_b) =
            bridge::ensure_rank2_contiguous_zero_offset("matmul_int4", 1, l_b)?;
        let (scale_rows, n_s) = bridge::ensure_rank2_contiguous_zero_offset("matmul_int4", 2, l_s)?;

        // kaio_ops::matmul_int4 contract:
        //   a: f16 [M, K]
        //   b_packed: u32 [K/8, N]   (8 INT4 values packed per u32)
        //   scales:   f16 [K/128, N] (one f16 scale per group of 128)
        //   out:      f32 [M, N]
        // K must be a multiple of GROUP_SIZE (= 128).
        if !k_a.is_multiple_of(GROUP_SIZE as usize) {
            return Err(Error::Msg(format!(
                "kaio-candle::matmul_int4: K ({k_a}) must be a multiple of group_size ({GROUP_SIZE}). \
                 This op is locked to GPTQ-style group_size=128 per the kaio-ops kernel contract."
            )));
        }
        let expected_packed_rows = k_a / 8; // 8 INT4 / u32
        let expected_scale_rows = k_a / GROUP_SIZE as usize;
        if packed_rows != expected_packed_rows {
            return Err(Error::Msg(format!(
                "kaio-candle::matmul_int4: packed-weight rows mismatch — \
                 input #0 shape [{m_a}, {k_a}] implies K/8 = {expected_packed_rows} \
                 packed rows, input #1 has shape [{packed_rows}, {n_b}]."
            )));
        }
        if scale_rows != expected_scale_rows {
            return Err(Error::Msg(format!(
                "kaio-candle::matmul_int4: scales rows mismatch — \
                 K/{GROUP_SIZE} = {expected_scale_rows} groups expected, \
                 input #2 has shape [{scale_rows}, {n_s}]."
            )));
        }
        if n_b != n_s {
            return Err(Error::Msg(format!(
                "kaio-candle::matmul_int4: N mismatch between packed weights ({n_b}) \
                 and scales ({n_s})."
            )));
        }

        let m = u32::try_from(m_a)
            .map_err(|_| Error::Msg(format!("matmul_int4: M ({m_a}) exceeds u32")))?;
        let n = u32::try_from(n_b)
            .map_err(|_| Error::Msg(format!("matmul_int4: N ({n_b}) exceeds u32")))?;
        let k = u32::try_from(k_a)
            .map_err(|_| Error::Msg(format!("matmul_int4: K ({k_a}) exceeds u32")))?;

        let candle_dev = s_a.device.clone();
        bridge::ensure_ordinal_match(&candle_dev, &self.device)?;

        // AD4: dtype gates — a = f16, b_packed = u32, scales = f16.
        let a_slice = bridge::slice_ref_from_storage::<f16>(s_a)?;
        let b_slice = bridge::slice_ref_from_storage::<u32>(s_b)?;
        let s_slice = bridge::slice_ref_from_storage::<f16>(s_s)?;

        // AD2-Audit D4: kaio_ops::matmul_int4 runs validate_dims_int4
        // (read-only) then launches the dequant-fused kernel. Inner loop
        // reads A / B_packed / scales into shared memory stagings and
        // fragments; inputs never mutated. Safe under the readonly
        // transmute contract.
        let a_buf: &GpuBuffer<f16> = bridge::buffer_ref_from_slice_readonly(a_slice);
        let b_buf: &GpuBuffer<u32> = bridge::buffer_ref_from_slice_readonly(b_slice);
        let s_buf: &GpuBuffer<f16> = bridge::buffer_ref_from_slice_readonly(s_slice);

        let mut out_buf: GpuBuffer<f32> = self
            .device
            .alloc_zeros::<f32>(m_a * n_b)
            .map_err(bridge::kaio_err)?;

        bridge::sync_before_launch(&candle_dev, &self.device)?;

        kaio_matmul_int4(
            &self.device,
            a_buf,
            b_buf,
            s_buf,
            &mut out_buf,
            m,
            n,
            k,
            GROUP_SIZE,
        )
        .map_err(bridge::kaio_err)?;

        bridge::sync_after_launch(&candle_dev, &self.device)?;

        let out_slice = out_buf.into_cuda_slice();
        let out_storage = bridge::storage_from_slice::<f32>(out_slice, candle_dev);
        Ok((out_storage, Shape::from_dims(&[m_a, n_b])))
    }
}

/// INT4 dequantize-matmul through candle's Tensor API.
///
/// - `a`: `f16[M, K]`, contiguous, zero-offset.
/// - `b_packed`: `u32[K/8, N]` — 8 INT4 values packed per `u32`.
/// - `scales`: `f16[K/128, N]` — one f16 scale per group of 128 elements.
///
/// Returns: `f32[M, N]`. `K` must be a multiple of 128 (GPTQ-style
/// `group_size=128` is locked in). Requires SM 8.0+.
pub fn matmul_int4(
    device: &Arc<KaioDevice>,
    a: &Tensor,
    b_packed: &Tensor,
    scales: &Tensor,
) -> Result<Tensor> {
    a.apply_op3(
        b_packed,
        scales,
        MatmulInt4Op {
            device: device.clone(),
        },
    )
}
