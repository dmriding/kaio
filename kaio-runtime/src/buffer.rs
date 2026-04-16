//! Typed device memory buffers.

use cudarc::driver::CudaSlice;

use crate::device::KaioDevice;
use crate::error::Result;

/// A typed buffer in GPU device memory, wrapping cudarc's [`CudaSlice<T>`].
///
/// Created via [`KaioDevice::alloc_from`] or [`KaioDevice::alloc_zeros`].
///
/// # Memory management
///
/// `GpuBuffer` does **not** implement [`Drop`] manually — cudarc's
/// [`CudaSlice`] handles device memory deallocation automatically when
/// the buffer is dropped. The `CudaSlice` holds an `Arc<CudaContext>`
/// internally, ensuring the CUDA context outlives the allocation.
///
/// # Representation — load-bearing
///
/// `#[repr(transparent)]` guarantees this newtype has identical memory
/// layout, size, and alignment to its sole field [`CudaSlice<T>`]. The
/// `kaio-candle` bridge crate relies on this to cast `&CudaSlice<T>`
/// (borrowed from candle's `CudaStorage`) to `&GpuBuffer<T>` for passing
/// into `kaio-ops` kernel entry points without round-tripping through an
/// owned clone.
///
/// **Do not remove `#[repr(transparent)]` or add a second field without
/// coordinating with `kaio-candle`.** The soundness-assertion tests at the
/// bottom of this module will fail at compile time if the layout diverges.
#[repr(transparent)]
pub struct GpuBuffer<T> {
    inner: CudaSlice<T>,
}

impl<T> GpuBuffer<T> {
    /// Wrap an existing cudarc [`CudaSlice`] as a [`GpuBuffer`].
    ///
    /// Takes ownership of the slice. The returned `GpuBuffer` drops the
    /// underlying device allocation via cudarc's normal `Drop` on its own
    /// drop.
    ///
    /// Used by bridge crates (e.g. `kaio-candle`) to consume a
    /// fresh-allocated slice back into the KAIO buffer type after a kernel
    /// produces its output.
    pub fn from_cuda_slice(inner: CudaSlice<T>) -> Self {
        Self { inner }
    }

    /// Consume the buffer and return the underlying cudarc [`CudaSlice`].
    ///
    /// Used by bridge crates to hand the owned output slice back to the
    /// host framework (e.g. wrapping into `candle_core::CudaStorage`) after
    /// a KAIO kernel has written into the buffer.
    pub fn into_cuda_slice(self) -> CudaSlice<T> {
        self.inner
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
    /// Requires a reference to the [`KaioDevice`] that created this buffer
    /// (for stream access). The device is borrowed, not consumed.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let device = KaioDevice::new(0)?;
    /// let buf = device.alloc_from(&[1.0f32, 2.0, 3.0])?;
    /// let host_data = buf.to_host(&device)?;
    /// assert_eq!(host_data, vec![1.0, 2.0, 3.0]);
    /// ```
    pub fn to_host(&self, device: &KaioDevice) -> Result<Vec<T>> {
        Ok(device.stream().clone_dtoh(&self.inner)?)
    }
}

// Soundness assertions for the `#[repr(transparent)]` contract above.
// Compile-time: any future change to `GpuBuffer`'s layout (adding a field,
// removing `#[repr(transparent)]`, changing the inner type) fails the build
// here instead of producing UB at the `kaio-candle` transmute site.
// Placed at end-of-file to satisfy the `clippy::items_after_test_module`
// lint.
#[cfg(test)]
mod repr_soundness {
    use super::GpuBuffer;
    use cudarc::driver::CudaSlice;
    use half::f16;
    use static_assertions::{assert_eq_align, assert_eq_size};

    assert_eq_size!(GpuBuffer<f32>, CudaSlice<f32>);
    assert_eq_align!(GpuBuffer<f32>, CudaSlice<f32>);
    assert_eq_size!(GpuBuffer<f16>, CudaSlice<f16>);
    assert_eq_align!(GpuBuffer<f16>, CudaSlice<f16>);
    assert_eq_size!(GpuBuffer<i8>, CudaSlice<i8>);
    assert_eq_align!(GpuBuffer<i8>, CudaSlice<i8>);
    assert_eq_size!(GpuBuffer<u32>, CudaSlice<u32>);
    assert_eq_align!(GpuBuffer<u32>, CudaSlice<u32>);
}
