//! Python wrapper around `kaio_rs::KaioDevice`.
//!
//! `kaio.Device` holds an `Arc<KaioDevice>`. Every `kaio.Tensor` created
//! through the device stores a clone of this Arc, which enforces the
//! master-plan invariant that a Tensor cannot outlive its Device's CUDA
//! context. Rust's reference counting handles teardown when the last
//! Python reference drops.
//!
//! `KaioDevice` is `Send + Sync` (wraps `Arc<CudaContext>` + `Arc<CudaStream>`,
//! both thread-safe in cudarc 0.19), so no `#[pyclass(unsendable)]` is
//! needed. If a future refactor makes `KaioDevice` thread-local, this
//! attribute is the correct safety fallback.

use std::sync::Arc;

use kaio_rs::prelude::KaioDevice;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Python handle to a CUDA device.
///
/// Construction opens a CUDA context on the given device ordinal (0 for
/// the first GPU) and fails with an exception if the driver is missing,
/// no GPU is present, or the ordinal is out of range.
///
/// Stub exception mapping uses `PyRuntimeError` until `kaio.KaioError`
/// lands in C4; swap-over is a mechanical `.map_err(...)` rewrite.
#[pyclass(module = "kaio", name = "Device")]
pub struct Device {
    pub(crate) inner: Arc<KaioDevice>,
}

#[pymethods]
impl Device {
    /// Create a handle to the GPU at `index`. Defaults to 0 (first GPU).
    #[new]
    #[pyo3(signature = (index = 0))]
    fn new(index: usize) -> PyResult<Self> {
        let device = KaioDevice::new(index)
            .map_err(|e| PyRuntimeError::new_err(format!("failed to open device {index}: {e}")))?;
        Ok(Self {
            inner: Arc::new(device),
        })
    }

    /// Human-readable GPU name (e.g. "NVIDIA GeForce RTX 4090").
    #[getter]
    fn name(&self) -> PyResult<String> {
        let info = self
            .inner
            .info()
            .map_err(|e| PyRuntimeError::new_err(format!("device info failed: {e}")))?;
        Ok(info.name)
    }

    /// Compute capability as a `(major, minor)` tuple, e.g. `(8, 9)`
    /// for SM 8.9.
    #[getter]
    fn compute_capability(&self) -> PyResult<(u32, u32)> {
        let info = self
            .inner
            .info()
            .map_err(|e| PyRuntimeError::new_err(format!("device info failed: {e}")))?;
        Ok(info.compute_capability)
    }

    fn __repr__(&self) -> PyResult<String> {
        let info = self
            .inner
            .info()
            .map_err(|e| PyRuntimeError::new_err(format!("device info failed: {e}")))?;
        Ok(format!(
            "Device(name={:?}, sm=sm_{}{})",
            info.name, info.compute_capability.0, info.compute_capability.1
        ))
    }
}
