//! PTX module loading and kernel function handles.

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule};

use crate::error::Result;

/// A loaded PTX module on the GPU device.
///
/// Created via [`PyrosDevice::load_ptx`](crate::device::PyrosDevice::load_ptx).
/// Use [`function`](Self::function) to get a handle to a specific kernel
/// entry point, then launch it via cudarc's `launch_builder`.
pub struct PyrosModule {
    inner: Arc<CudaModule>,
}

impl PyrosModule {
    /// Wrap a raw cudarc module.
    pub(crate) fn from_raw(inner: Arc<CudaModule>) -> Self {
        Self { inner }
    }

    /// Get a kernel function handle by name.
    ///
    /// The name must match the `.entry` name in the PTX source
    /// (e.g. `"vector_add"`).
    pub fn function(&self, name: &str) -> Result<PyrosFunction> {
        let func = self.inner.load_function(name)?;
        Ok(PyrosFunction { inner: func })
    }
}

/// A handle to a kernel function within a loaded PTX module.
///
/// Use [`inner`](Self::inner) to access the underlying `CudaFunction`
/// for passing to cudarc's `launch_builder`. In Phase 1, kernel launch
/// goes through cudarc directly — Phase 2's macro will generate typed
/// safe wrappers.
pub struct PyrosFunction {
    inner: CudaFunction,
}

impl PyrosFunction {
    /// Access the underlying [`CudaFunction`] for cudarc's launch builder.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let cfg = LaunchConfig::for_num_elems(n);
    /// unsafe {
    ///     device.stream()
    ///         .launch_builder(func.inner())
    ///         .arg(buf_a.inner())
    ///         .arg(&n)
    ///         .launch(cfg)?;
    /// }
    /// ```
    pub fn inner(&self) -> &CudaFunction {
        &self.inner
    }
}
