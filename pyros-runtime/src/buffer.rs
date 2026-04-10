//! Typed device memory buffers.

use cudarc::driver::CudaSlice;

use crate::device::PyrosDevice;
use crate::error::Result;

/// A typed buffer in GPU device memory, wrapping cudarc's [`CudaSlice<T>`].
///
/// Created via [`PyrosDevice::alloc_from`] or [`PyrosDevice::alloc_zeros`].
///
/// # Memory management
///
/// `GpuBuffer` does **not** implement [`Drop`] manually — cudarc's
/// [`CudaSlice`] handles device memory deallocation automatically when
/// the buffer is dropped. The `CudaSlice` holds an `Arc<CudaContext>`
/// internally, ensuring the CUDA context outlives the allocation.
pub struct GpuBuffer<T> {
    inner: CudaSlice<T>,
}

impl<T> GpuBuffer<T> {
    /// Create a `GpuBuffer` from an existing `CudaSlice`.
    pub(crate) fn from_raw(inner: CudaSlice<T>) -> Self {
        Self { inner }
    }

    /// Number of elements in the buffer.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the buffer contains no elements.
    pub fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }

    /// Access the underlying [`CudaSlice`] for passing to cudarc launch
    /// operations.
    ///
    /// This is the escape hatch for Sprint 1.7's launch builder — the
    /// caller pushes `&buf.inner()` as a kernel argument.
    pub fn inner(&self) -> &CudaSlice<T> {
        &self.inner
    }

    /// Mutable access to the underlying [`CudaSlice`].
    pub fn inner_mut(&mut self) -> &mut CudaSlice<T> {
        &mut self.inner
    }
}

impl<T: cudarc::driver::DeviceRepr + Default + Clone + Unpin> GpuBuffer<T> {
    /// Transfer buffer contents from device to host.
    ///
    /// Requires a reference to the [`PyrosDevice`] that created this buffer
    /// (for stream access). The device is borrowed, not consumed.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let device = PyrosDevice::new(0)?;
    /// let buf = device.alloc_from(&[1.0f32, 2.0, 3.0])?;
    /// let host_data = buf.to_host(&device)?;
    /// assert_eq!(host_data, vec![1.0, 2.0, 3.0]);
    /// ```
    pub fn to_host(&self, device: &PyrosDevice) -> Result<Vec<T>> {
        Ok(device.stream().clone_dtoh(&self.inner)?)
    }
}
