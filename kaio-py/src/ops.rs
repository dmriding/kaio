//! GPU ops exposed from Python.
//!
//! Sprint 8.1 ships a single op — `matmul_tc` — as the end-to-end
//! smoke demo for the Phase 8 Python bindings. The call path stays
//! on device from input Tensor to output Tensor; no CPU round-trip
//! between ops.
//!
//! Every op wrapper follows the same pattern established here:
//!
//! 1. Dtype-validate the inputs (match on `TensorStorage`) with
//!    clear error messages.
//! 2. Cross-Device-identity check via `Arc::ptr_eq` — Tensors must
//!    come from the **same `kaio.Device` object**, not merely the
//!    same GPU ordinal.
//! 3. Shape validation, including `usize → u32` range checks for the
//!    kernel API's `u32` dim parameters. No silent truncation.
//! 4. Allocate output buffer on the shared `Arc<KaioDevice>`.
//! 5. Release the GIL around the kernel launch + synchronize:
//!    `py.detach(|| kaio_ops::matmul_tc(...))`. (PyO3 0.28 renamed
//!    the former `allow_threads` to `detach`.)
//! 6. Wrap the output buffer as a new `Tensor` and return it.

use std::sync::Arc;

use pyo3::prelude::*;

use crate::errors::{kaio_err, map_kaio_err};
use crate::tensor::{Tensor, TensorStorage};

/// Tensor-core matmul: `C = A @ B` with f16 inputs → f32 output.
///
/// Shapes: `A = [M, K]`, `B = [K, N]`, `C = [M, N]`. Both inputs
/// must be dtype `float16`; the output Tensor is `float32`.
///
/// Kernel launch happens under `py.allow_threads`, so concurrent
/// Python threads make progress while the GPU runs the matmul.
#[pyfunction]
pub fn matmul_tc(py: Python<'_>, a: &Tensor, b: &Tensor) -> PyResult<Tensor> {
    let a_buf = match &a.storage {
        TensorStorage::F16(buf) => buf,
        other => {
            return Err(kaio_err(format!(
                "matmul_tc requires float16 inputs, got {} for argument 0",
                other.dtype_name()
            )));
        }
    };
    let b_buf = match &b.storage {
        TensorStorage::F16(buf) => buf,
        other => {
            return Err(kaio_err(format!(
                "matmul_tc requires float16 inputs, got {} for argument 1",
                other.dtype_name()
            )));
        }
    };

    if !Arc::ptr_eq(&a.device, &b.device) {
        return Err(kaio_err(
            "matmul_tc: tensors must be on the same Device object (two kaio.Device(0) calls create distinct devices)",
        ));
    }

    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(kaio_err(format!(
            "matmul_tc requires 2-D inputs, got A.shape={:?}, B.shape={:?}",
            a.shape, b.shape
        )));
    }
    let (m, k_a) = (a.shape[0], a.shape[1]);
    let (k_b, n) = (b.shape[0], b.shape[1]);
    if k_a != k_b {
        return Err(kaio_err(format!(
            "matmul_tc shape mismatch: A=[M={m}, K={k_a}], B=[K={k_b}, N={n}]"
        )));
    }
    let k = k_a;

    let (m_u32, n_u32, k_u32) = to_u32_dims(m, n, k)?;

    let device = Arc::clone(&a.device);
    let mut c_buf = device.alloc_zeros::<f32>(m * n).map_err(map_kaio_err)?;

    py.detach(|| kaio_ops::matmul_tc(&device, a_buf, b_buf, &mut c_buf, m_u32, n_u32, k_u32))
        .map_err(map_kaio_err)?;

    Ok(Tensor {
        device,
        storage: TensorStorage::F32(c_buf),
        shape: vec![m, n],
    })
}

fn to_u32_dims(m: usize, n: usize, k: usize) -> PyResult<(u32, u32, u32)> {
    let to_u32 = |dim: usize, label: &str| -> PyResult<u32> {
        u32::try_from(dim)
            .map_err(|_| kaio_err(format!("matmul_tc: {label} dim exceeds u32 range: {dim}")))
    };
    Ok((to_u32(m, "M")?, to_u32(n, "N")?, to_u32(k, "K")?))
}
