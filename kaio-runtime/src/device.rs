//! CUDA device management.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaStream, DeviceRepr, ValidAsZeroBits};

use crate::buffer::GpuBuffer;
use crate::error::Result;

/// Process-wide latch for the debug-build performance note.
///
/// Sprint 7.0.5 A2: emit a one-time stderr note on first `KaioDevice::new`
/// when the binary is built in debug mode. Prevents the common "benchmarked
/// in debug, bounced" adoption failure where new users run a showcase example
/// with `cargo run` (defaulting to debug) and conclude KAIO is slow. The note
/// is performance-framed only — debug-mode does not affect correctness, and a
/// `cargo test`-in-debug user checking kernel output should not see their
/// correctness results cast into doubt.
static DEBUG_WARNED: OnceLock<()> = OnceLock::new();

/// Performance-framed debug-mode note body. `const` so tests can assert on
/// its content without re-typing the message.
const DEBUG_WARNING_MESSAGE: &str = "[kaio] Note: debug build — GPU kernel performance is ~10-20x slower than --release. Use `cargo run --release` / `cargo test --release` for representative performance numbers. Correctness is unaffected. Set KAIO_SUPPRESS_DEBUG_WARNING=1 to silence.";

/// Pure decision function: should the debug-mode note fire on this
/// process? Split out from [`maybe_warn_debug_build`] so the env-var
/// logic is testable without the static `OnceLock` interfering.
fn should_emit_debug_warning() -> bool {
    cfg!(debug_assertions) && std::env::var("KAIO_SUPPRESS_DEBUG_WARNING").is_err()
}

/// Emit the debug-mode performance note to stderr once per process, if
/// [`should_emit_debug_warning`] returns true.
///
/// Called from [`KaioDevice::new`] — every KAIO program hits this path on
/// first launch, so the note surfaces exactly when a user would first
/// benefit from knowing. In release builds, `cfg!(debug_assertions)` folds
/// to `false` and the whole body compiles out.
fn maybe_warn_debug_build() {
    if should_emit_debug_warning() {
        DEBUG_WARNED.get_or_init(|| {
            eprintln!("{DEBUG_WARNING_MESSAGE}");
        });
    }
}

/// A KAIO GPU device — wraps a CUDA context and its default stream.
///
/// Created via [`KaioDevice::new`] with a device ordinal (0 for the first GPU).
/// All allocation and transfer operations go through the default stream.
///
/// # Example
///
/// ```ignore
/// let device = KaioDevice::new(0)?;
/// let buf = device.alloc_from(&[1.0f32, 2.0, 3.0])?;
/// let host = buf.to_host(&device)?;
/// ```
pub struct KaioDevice {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
}

impl std::fmt::Debug for KaioDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KaioDevice")
            .field("ordinal", &self.ctx.ordinal())
            .finish()
    }
}

impl KaioDevice {
    /// Create a new device targeting the GPU at the given ordinal.
    ///
    /// Ordinal 0 is the first GPU. Returns an error if no GPU exists at
    /// that ordinal or if the CUDA driver fails to initialize.
    pub fn new(ordinal: usize) -> Result<Self> {
        maybe_warn_debug_build();
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

    /// Load a PTX module from source text and return a [`crate::module::KaioModule`].
    ///
    /// The PTX text is passed to the CUDA driver's `cuModuleLoadData` —
    /// no NVRTC compilation occurs. The driver JIT-compiles the PTX for
    /// the current GPU.
    ///
    /// # Deprecated — prefer [`load_module`](Self::load_module)
    ///
    /// The module path runs
    /// [`PtxModule::validate`](kaio_core::ir::PtxModule::validate)
    /// before the driver sees the PTX, catching SM mismatches (e.g.
    /// `mma.sync` on sub-Ampere targets) with readable
    /// [`KaioError::Validation`](crate::error::KaioError::Validation)
    /// errors instead of cryptic `ptxas` failures deep in the driver.
    ///
    /// This function remains public for raw-PTX use cases (external PTX
    /// files, hand-written PTX for research, bypassing validation
    /// intentionally). It is not scheduled for removal in the 0.2.x line.
    ///
    /// # Migration
    ///
    /// Before:
    /// ```ignore
    /// let ptx_text: String = build_my_ptx();
    /// let module = device.load_ptx(&ptx_text)?;
    /// ```
    ///
    /// After:
    /// ```ignore
    /// use kaio_core::ir::PtxModule;
    /// let ptx_module: PtxModule = build_my_module("sm_80");
    /// let module = device.load_module(&ptx_module)?;
    /// ```
    #[deprecated(
        since = "0.2.1",
        note = "use load_module(&PtxModule) — runs PtxModule::validate() for readable SM-mismatch errors"
    )]
    pub fn load_ptx(&self, ptx_text: &str) -> Result<crate::module::KaioModule> {
        let ptx = cudarc::nvrtc::Ptx::from_src(ptx_text);
        let module = self.ctx.load_module(ptx)?;
        Ok(crate::module::KaioModule::from_raw(module))
    }

    /// Validate, emit, and load a [`kaio_core::ir::PtxModule`] on the device.
    ///
    /// This is the preferred entrypoint when the caller has an in-memory
    /// `PtxModule` (as opposed to raw PTX text). Before the PTX text is
    /// handed to the driver, [`kaio_core::ir::PtxModule::validate`]
    /// checks that the module's target SM supports every feature used by
    /// its kernels — raising
    /// [`KaioError::Validation`](crate::error::KaioError::Validation) if
    /// e.g. a `mma.sync` op is present but the target is `sm_70`.
    ///
    /// Surfacing the error at this layer gives the user a readable
    /// message ("`mma.sync.m16n8k16 requires sm_80+, target is sm_70`")
    /// instead of a cryptic `ptxas` error from deep in the driver.
    pub fn load_module(
        &self,
        module: &kaio_core::ir::PtxModule,
    ) -> Result<crate::module::KaioModule> {
        use kaio_core::emit::{Emit, PtxWriter};

        module.validate()?;

        let mut w = PtxWriter::new();
        module
            .emit(&mut w)
            .map_err(|e| crate::error::KaioError::PtxLoad(format!("emit failed: {e}")))?;
        let ptx_text = w.finish();

        // `load_ptx` is #[deprecated] as a public API to steer users to the
        // validated module path, but it's still the correct internal
        // implementation detail after we've emitted the PTX text here.
        #[allow(deprecated)]
        self.load_ptx(&ptx_text)
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

    static DEVICE: OnceLock<KaioDevice> = OnceLock::new();
    fn device() -> &'static KaioDevice {
        DEVICE.get_or_init(|| KaioDevice::new(0).expect("GPU required for tests"))
    }

    // Sprint 7.0.5 A2: debug-mode performance note tests.
    //
    // These verify the pure-function half of the warning logic. The
    // once-per-process behavior mediated by the static `DEBUG_WARNED`
    // OnceLock is not testable in-process without restructuring (the
    // latch is set for the lifetime of the test binary); manual/subprocess
    // verification is in sprint_7_0_5.md.

    #[test]
    fn debug_warning_message_is_performance_framed_not_correctness_framed() {
        // Regression canary (Sprint 7.0.5 A2 message framing): if the
        // wording ever drifts to imply correctness is affected ("results
        // are not meaningful," "output is invalid," etc.) this test
        // fails. The whole point of the message is to prevent perf
        // misunderstandings WITHOUT scaring off correctness testing.
        let msg = DEBUG_WARNING_MESSAGE;
        assert!(
            msg.contains("performance"),
            "debug warning must mention performance: {msg}"
        );
        assert!(
            msg.contains("Correctness is unaffected") || msg.contains("correctness is unaffected"),
            "debug warning must explicitly state correctness is unaffected: {msg}"
        );
        assert!(
            !msg.to_lowercase().contains("not meaningful")
                && !msg.to_lowercase().contains("invalid"),
            "debug warning must NOT imply results are invalid/not meaningful: {msg}"
        );
        assert!(
            msg.contains("KAIO_SUPPRESS_DEBUG_WARNING"),
            "debug warning must document the opt-out env var: {msg}"
        );
    }

    #[test]
    fn debug_warning_opt_out_env_var_suppresses() {
        // SAFETY: single-threaded env-var manipulation inside a test.
        // Restore the prior value (if any) before returning so other
        // tests in the same binary don't observe stale state.
        let prev = std::env::var("KAIO_SUPPRESS_DEBUG_WARNING").ok();
        unsafe {
            std::env::set_var("KAIO_SUPPRESS_DEBUG_WARNING", "1");
        }
        assert!(
            !should_emit_debug_warning(),
            "KAIO_SUPPRESS_DEBUG_WARNING=1 must suppress the warning"
        );
        unsafe {
            std::env::remove_var("KAIO_SUPPRESS_DEBUG_WARNING");
        }
        // In debug builds the warning should now be allowed; in release
        // builds cfg!(debug_assertions) is false so it's suppressed either
        // way. Assert the cfg-consistent expectation.
        assert_eq!(should_emit_debug_warning(), cfg!(debug_assertions));
        // Restore
        if let Some(v) = prev {
            unsafe {
                std::env::set_var("KAIO_SUPPRESS_DEBUG_WARNING", v);
            }
        }
    }

    #[test]
    #[ignore] // requires NVIDIA GPU
    fn device_creation() {
        let dev = KaioDevice::new(0);
        assert!(dev.is_ok(), "KaioDevice::new(0) failed: {dev:?}");
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
        // Any SM 7.0+ GPU should work (Volta and newer)
        let (major, _minor) = info.compute_capability;
        assert!(
            major >= 7,
            "expected SM 7.0+ GPU, got SM {}.{}",
            info.compute_capability.0,
            info.compute_capability.1,
        );
        eprintln!(
            "GPU compute capability: SM {}.{}",
            info.compute_capability.0, info.compute_capability.1
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
        let result = KaioDevice::new(999);
        assert!(result.is_err(), "expected error for ordinal 999");
    }
}
