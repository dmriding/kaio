//! CUDA device management.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream, DeviceRepr, ValidAsZeroBits};

use crate::buffer::GpuBuffer;
use crate::error::Result;

/// A PYROS GPU device — wraps a CUDA context and its default stream.
///
/// Created via [`PyrosDevice::new`] with a device ordinal (0 for the first GPU).
/// All allocation and transfer operations go through the default stream.
///
/// # Example
///
/// ```ignore
/// let device = PyrosDevice::new(0)?;
/// let buf = device.alloc_from(&[1.0f32, 2.0, 3.0])?;
/// let host = buf.to_host(&device)?;
/// ```
pub struct PyrosDevice {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
}

impl std::fmt::Debug for PyrosDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyrosDevice")
            .field("ordinal", &self.ctx.ordinal())
            .finish()
    }
}

impl PyrosDevice {
    /// Create a new device targeting the GPU at the given ordinal.
    ///
    /// Ordinal 0 is the first GPU. Returns an error if no GPU exists at
    /// that ordinal or if the CUDA driver fails to initialize.
    pub fn new(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal)?;
        let stream = ctx.default_stream();
        Ok(Self { ctx, stream })
    }

    /// Query basic information about this device.
    pub fn info(&self) -> Result<DeviceInfo> {
        DeviceInfo::from_context(&self.ctx)
    }

    /// Allocate device memory and copy data from a host slice.
    pub fn alloc_from<T: DeviceRepr>(&self, data: &[T]) -> Result<GpuBuffer<T>> {
        let slice = self.stream.clone_htod(data)?;
        Ok(GpuBuffer::from_raw(slice))
    }

    /// Allocate zero-initialized device memory.
    pub fn alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(&self, len: usize) -> Result<GpuBuffer<T>> {
        let slice = self.stream.alloc_zeros::<T>(len)?;
        Ok(GpuBuffer::from_raw(slice))
    }

    /// Access the underlying CUDA stream for kernel launch operations.
    ///
    /// Used with cudarc's `launch_builder` to launch kernels. In Phase 2,
    /// the proc macro will generate typed wrappers that hide this.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Load a PTX module from source text and return a [`crate::module::PyrosModule`].
    ///
    /// The PTX text is passed to the CUDA driver's `cuModuleLoadData` —
    /// no NVRTC compilation occurs. The driver JIT-compiles the PTX for
    /// the current GPU.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let module = device.load_ptx(&ptx_text)?;
    /// let func = module.function("vector_add")?;
    /// ```
    pub fn load_ptx(&self, ptx_text: &str) -> Result<crate::module::PyrosModule> {
        let ptx = cudarc::nvrtc::Ptx::from_src(ptx_text);
        let module = self.ctx.load_module(ptx)?;
        Ok(crate::module::PyrosModule::from_raw(module))
    }
}

/// Basic information about a CUDA device.
///
/// Phase 1 includes name, compute capability, and total memory.
/// Additional fields (SM count, max threads per block, max shared memory,
/// warp size) are planned for Phase 3/4 when shared memory and occupancy
/// calculations matter.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// GPU device name (e.g. "NVIDIA GeForce RTX 4090").
    pub name: String,
    /// Compute capability as (major, minor) — e.g. (8, 9) for SM 8.9.
    pub compute_capability: (u32, u32),
    /// Total device memory in bytes.
    pub total_memory: usize,
}

impl DeviceInfo {
    /// Query device info from a CUDA context.
    fn from_context(ctx: &Arc<CudaContext>) -> Result<Self> {
        use cudarc::driver::result::device;

        let ordinal = ctx.ordinal();
        let dev = device::get(ordinal as i32)?;
        let name = device::get_name(dev)?;
        let total_memory = unsafe { device::total_mem(dev)? };

        // SAFETY: dev is a valid device handle obtained from device::get().
        // get_attribute reads a device property — no mutation, no aliasing.
        let major = unsafe {
            device::get_attribute(
                dev,
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            )?
        };
        let minor = unsafe {
            device::get_attribute(
                dev,
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            )?
        };

        Ok(Self {
            name,
            compute_capability: (major as u32, minor as u32),
            total_memory,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    static DEVICE: OnceLock<PyrosDevice> = OnceLock::new();
    fn device() -> &'static PyrosDevice {
        DEVICE.get_or_init(|| PyrosDevice::new(0).expect("GPU required for tests"))
    }

    #[test]
    #[ignore] // requires NVIDIA GPU
    fn device_creation() {
        let dev = PyrosDevice::new(0);
        assert!(dev.is_ok(), "PyrosDevice::new(0) failed: {dev:?}");
    }

    #[test]
    #[ignore]
    fn device_info_name() {
        let info = device().info().expect("info() failed");
        assert!(!info.name.is_empty(), "device name should not be empty");
        // RTX 4090 should contain "4090" somewhere in the name
        eprintln!("GPU name: {}", info.name);
    }

    #[test]
    #[ignore]
    fn device_info_compute_capability() {
        let info = device().info().expect("info() failed");
        // RTX 4090 = SM 8.9
        assert_eq!(
            info.compute_capability,
            (8, 9),
            "expected SM 8.9 for RTX 4090, got {:?}",
            info.compute_capability
        );
    }

    #[test]
    #[ignore]
    fn buffer_roundtrip_f32() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let buf = device().alloc_from(&data).expect("alloc_from failed");
        let result = buf.to_host(device()).expect("to_host failed");
        assert_eq!(result, data, "roundtrip data mismatch");
    }

    #[test]
    #[ignore]
    fn buffer_alloc_zeros() {
        let buf = device()
            .alloc_zeros::<f32>(100)
            .expect("alloc_zeros failed");
        let result = buf.to_host(device()).expect("to_host failed");
        assert_eq!(result, vec![0.0f32; 100]);
    }

    #[test]
    #[ignore]
    fn buffer_len() {
        let buf = device()
            .alloc_from(&[1.0f32, 2.0, 3.0])
            .expect("alloc_from failed");
        assert_eq!(buf.len(), 3);
        assert!(!buf.is_empty());
    }

    #[test]
    #[ignore]
    fn invalid_device_ordinal() {
        let result = PyrosDevice::new(999);
        assert!(result.is_err(), "expected error for ordinal 999");
    }
}
