#![allow(dead_code)]

//! Bridge primitives between candle's `CudaStorage` / `CudaDevice` and
//! KAIO's `GpuBuffer<T>` / `KaioDevice`.
//!
//! All items here are crate-private (`pub(crate)`) — consumers use the
//! per-op wrapper functions in [`matmul_tc`](crate::matmul_tc),
//! [`matmul_tc_async`](crate::matmul_tc_async),
//! [`matmul_int4`](crate::matmul_int4), and
//! [`attention_tc`](crate::attention_tc) modules, not these primitives
//! directly.
//!
//! # Contract map
//!
//! - [`slice_ref_from_storage`] / [`storage_from_slice`]: cudarc-handle
//!   plumbing across the candle boundary.
//! - [`buffer_ref_from_slice_readonly`]: `#[repr(transparent)]` cast into
//!   KAIO's buffer type. Semantic + lifetime contracts documented on the
//!   function.
//! - [`ensure_ordinal_match`]: the user-supplied `Arc<KaioDevice>` must
//!   reference the same CUDA ordinal as the tensor's candle device.
//!   Both sides share the same primary context via
//!   `cuDevicePrimaryCtxRetain`.
//! - [`sync_before_launch`] / [`sync_after_launch`]: stream-safety
//!   fences. See `lib.rs` crate docs for the CUDA-Graph limitation this
//!   implies.
//! - [`kaio_err`]: `KaioError` → `candle_core::Error` conversion (orphan
//!   rule prevents `impl From`; call sites use `.map_err(bridge::kaio_err)`).

use candle_core::backend::BackendDevice;
use candle_core::cuda_backend::CudaDType;
use candle_core::{CudaDevice, CudaStorage, DeviceLocation, Error, Layout, Result};
use cudarc::driver::CudaSlice;
use kaio::prelude::{GpuBuffer, KaioDevice, KaioError};

/// Extract the cudarc [`CudaSlice<T>`] handle from a candle [`CudaStorage`].
///
/// Errors if the storage's dtype doesn't match `T`.
pub(crate) fn slice_ref_from_storage<T: CudaDType>(storage: &CudaStorage) -> Result<&CudaSlice<T>> {
    storage.as_cuda_slice::<T>()
}

/// Wrap a cudarc [`CudaSlice<T>`] back into a candle [`CudaStorage`],
/// consuming the slice. Used on the output path after a KAIO kernel has
/// written into the buffer we allocated.
pub(crate) fn storage_from_slice<T: CudaDType>(
    slice: CudaSlice<T>,
    device: CudaDevice,
) -> CudaStorage {
    T::wrap_cuda_slice(slice, device)
}

/// Cast a cudarc `&CudaSlice<T>` to a `&GpuBuffer<T>` via the
/// `#[repr(transparent)]` guarantee on `GpuBuffer`.
///
/// # Invariants
///
/// - **Memory layout** (enforced by
///   `kaio_runtime::buffer::repr_soundness` compile-time asserts):
///   `GpuBuffer<T>` and `CudaSlice<T>` have identical size + alignment.
/// - **Aliasing semantics:** the returned `&GpuBuffer<T>` is
///   shared-immutable. Caller MUST NOT route it to any kaio-ops path that
///   mutates the input buffer. Each op module audits its kernel to confirm
///   inputs are read-only (loaded into MMA fragments via shared memory,
///   never written).
/// - **Lifetime:** the returned reference MUST NOT escape the current
///   synchronous function scope (i.e. `cuda_fwd`'s stack frame). cudarc's
///   `CudaSlice` is `Arc`-managed by candle; this transmute does NOT
///   increment the refcount. If the reference leaks into a detached
///   thread, async task, or static storage, candle could drop the slice
///   while a kernel is still reading device memory → UB. The post-launch
///   sync fence makes this safe-by-convention: the kernel completes before
///   `cuda_fwd` returns, so the borrow cannot outlive candle's ownership.
pub(crate) fn buffer_ref_from_slice_readonly<T>(slice: &CudaSlice<T>) -> &GpuBuffer<T> {
    // SAFETY:
    // - GpuBuffer<T> is #[repr(transparent)] over CudaSlice<T> (verified
    //   at compile time by kaio_runtime::buffer::repr_soundness).
    // - Returned reference inherits the borrow lifetime from `slice`.
    // - Caller contract (aliasing + lifetime) documented above.
    unsafe { &*(slice as *const CudaSlice<T> as *const GpuBuffer<T>) }
}

/// Reinterpret `&CudaSlice<u8>` as `&CudaSlice<i8>` without touching
/// device memory. Used by `matmul_int8` and `qkv_project_int8` to
/// bridge candle's `DType::U8` convention for INT8 tensors to the
/// `GpuBuffer<i8>` that kaio-ops expects.
///
/// # Soundness
///
/// cudarc's `CudaSlice<T>` carries `T` only as a `PhantomData` marker;
/// the underlying storage is a raw `CUdeviceptr` plus a length. Since
/// `u8` and `i8` have identical size (1 byte) and alignment (1), a
/// `&CudaSlice<u8>` and a `&CudaSlice<i8>` describe the same bits in
/// the same addresses. The transmute is metadata-only: no device
/// memory is read or written.
///
/// The input is read-only inside the kaio-ops kernels (matmul / QKV
/// projection inputs are loaded into MMA fragments via LDMATRIX, never
/// written), so aliasing this reinterpretation against any other `u8`
/// view of the same storage is safe.
pub(crate) fn reinterpret_u8_slice_as_i8(slice: &CudaSlice<u8>) -> &CudaSlice<i8> {
    // SAFETY: see function-level docs. Same-layout T-swap inside a ref
    // on a phantom-T newtype. No device I/O.
    unsafe { &*(slice as *const CudaSlice<u8> as *const CudaSlice<i8>) }
}

/// Extract the CUDA ordinal from a candle [`CudaDevice`].
///
/// candle expresses device location via the [`DeviceLocation`] enum; for
/// CUDA devices the ordinal lives in `DeviceLocation::Cuda { gpu_id }`.
/// Returns an error if somehow the device reports a non-CUDA location
/// (shouldn't happen inside a `cuda_fwd` call but the compiler needs the
/// exhaustive-match safety net).
fn candle_ordinal(dev: &CudaDevice) -> Result<usize> {
    match dev.location() {
        DeviceLocation::Cuda { gpu_id } => Ok(gpu_id),
        other => Err(Error::Msg(format!(
            "kaio-candle: expected CUDA device, candle reports location {other:?}"
        ))),
    }
}

/// The user-supplied [`Arc<KaioDevice>`](std::sync::Arc) must reference
/// the same CUDA ordinal as the tensor's candle device.
///
/// `cudarc::CudaContext::new(ord)` wraps `cuDevicePrimaryCtxRetain`, so
/// two same-ordinal context constructions share the same underlying
/// primary context at the driver level. Ordinal equality is the right
/// check; Arc-identity would be spurious.
pub(crate) fn ensure_ordinal_match(candle_dev: &CudaDevice, kaio_dev: &KaioDevice) -> Result<()> {
    let candle_ord = candle_ordinal(candle_dev)?;
    let kaio_ord = kaio_dev.ordinal();
    if candle_ord != kaio_ord {
        return Err(Error::Msg(format!(
            "kaio-candle: input tensor is on CUDA ordinal {candle_ord}, \
             but the Arc<KaioDevice> passed is ordinal {kaio_ord}. \
             Construct a KaioDevice on the same ordinal as the candle Device."
        )));
    }
    Ok(())
}

/// Pre-launch sync: ensure candle's prior work on the input tensors
/// completes before the KAIO kernel reads from them.
///
/// Uses event-based cross-stream synchronization (Sprint 7.4c):
/// `kaio_stream.join(&candle_stream)` records candle's current work into
/// a CUDA event and makes KAIO's stream wait on it. GPU-side only — no
/// CPU blocking. The underlying `cuEventRecord` + `cuStreamWaitEvent`
/// calls are CUDA Graph capture-compatible (unlike the prior
/// `cuCtxSynchronize` which was banned during capture).
///
/// Both streams must belong to the same CUDA device/context — enforced
/// by `ensure_ordinal_match` at every call site.
pub(crate) fn sync_before_launch(candle_dev: &CudaDevice, kaio_dev: &KaioDevice) -> Result<()> {
    let candle_stream = candle_dev.cuda_stream();
    let kaio_stream = kaio_dev.stream();
    kaio_stream.join(&candle_stream).map_err(driver_err)
}

/// Post-launch sync: ensure the KAIO kernel completes before candle
/// uses the output storage (which may be scheduled on a different stream).
///
/// Same event-based mechanism as [`sync_before_launch`], reversed:
/// `candle_stream.join(&kaio_stream)` makes candle wait for KAIO's
/// kernel completion.
pub(crate) fn sync_after_launch(candle_dev: &CudaDevice, kaio_dev: &KaioDevice) -> Result<()> {
    let kaio_stream = kaio_dev.stream();
    let candle_stream = candle_dev.cuda_stream();
    candle_stream.join(kaio_stream).map_err(driver_err)
}

/// Convert a [`KaioError`] into a [`candle_core::Error`]. Use at each
/// `.map_err(bridge::kaio_err)?` boundary. Orphan rule prevents the
/// natural `impl From<KaioError> for candle_core::Error`.
pub(crate) fn kaio_err(e: KaioError) -> Error {
    Error::Msg(format!("kaio: {e}"))
}

/// Convert a cudarc [`DriverError`] into a [`candle_core::Error`].
/// Used by the event-based sync functions (Sprint 7.4c) where cudarc's
/// stream/event API returns `DriverError` directly.
pub(crate) fn driver_err(e: cudarc::driver::DriverError) -> Error {
    Error::Msg(format!("kaio-candle stream sync: {e}"))
}

/// Shared inner gate logic — `input_label` is either `"input #0"` or a
/// named parameter like `"w_k"`.
fn ensure_rank2_contiguous_zero_offset_inner(
    op_name: &str,
    input_label: &str,
    layout: &Layout,
) -> Result<(usize, usize)> {
    let shape = layout.shape();
    let dims = shape.dims();
    if dims.len() != 2 {
        return Err(Error::Msg(format!(
            "kaio-candle::{op_name}: {input_label} must be rank-2; \
             got rank-{rank} input of shape {shape:?}. \
             For multi-head attention, reshape to rank-2 via \
             `.reshape((seq, d))?` after flattening batch+heads.",
            rank = dims.len()
        )));
    }
    if !layout.is_contiguous() {
        return Err(Error::Msg(format!(
            "kaio-candle::{op_name}: {input_label} must be contiguous; \
             got shape {shape:?} with strides {strides:?}. \
             Call `.contiguous()?` first.",
            strides = layout.stride()
        )));
    }
    if layout.start_offset() != 0 {
        return Err(Error::Msg(format!(
            "kaio-candle::{op_name}: {input_label} must start at storage offset 0; \
             got offset {off} (likely from a `.slice(..)` / `.narrow(..)` call). \
             Call `.contiguous()?` to compact.",
            off = layout.start_offset()
        )));
    }
    Ok((dims[0], dims[1]))
}

/// Shape + rank + contiguity + offset gate. Called by CustomOp impls on
/// every input layout. Input is identified by numeric index (`input #0`).
pub(crate) fn ensure_rank2_contiguous_zero_offset(
    op_name: &'static str,
    input_index: usize,
    layout: &Layout,
) -> Result<(usize, usize)> {
    ensure_rank2_contiguous_zero_offset_inner(op_name, &format!("input #{input_index}"), layout)
}

/// Named-parameter variant of [`ensure_rank2_contiguous_zero_offset`].
/// Used by direct-call bridge functions (qkv_project_*) where inputs
/// are identified by parameter name (`w_k`, `scales_q`) rather than
/// numeric index.
pub(crate) fn ensure_rank2_contiguous_zero_offset_named(
    op_name: &str,
    param_name: &str,
    layout: &Layout,
) -> Result<(usize, usize)> {
    ensure_rank2_contiguous_zero_offset_inner(op_name, param_name, layout)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Layout, Shape};

    /// Happy path: rank-2 contiguous zero-offset layout returns (rows, cols).
    #[test]
    fn rank2_contiguous_zero_offset_returns_dims() {
        let layout = Layout::contiguous(Shape::from_dims(&[64, 128]));
        let (rows, cols) =
            ensure_rank2_contiguous_zero_offset("test", 0, &layout).expect("happy path");
        assert_eq!(rows, 64);
        assert_eq!(cols, 128);
    }

    /// Rank-1 input rejected with a concrete reshape hint.
    #[test]
    fn rank1_rejected_with_reshape_hint() {
        let layout = Layout::contiguous(Shape::from_dims(&[128]));
        let err = ensure_rank2_contiguous_zero_offset("test_op", 0, &layout)
            .expect_err("rank-1 must fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("must be rank-2"),
            "expected 'must be rank-2' in {msg}"
        );
        assert!(msg.contains("reshape"), "expected reshape hint in {msg}");
        assert!(msg.contains("test_op"), "expected op name in {msg}");
        assert!(msg.contains("input #0"), "expected input index in {msg}");
    }

    /// Rank-3 (common for multi-head attention before flatten) rejected
    /// with the same message shape.
    #[test]
    fn rank3_rejected_with_reshape_hint() {
        let layout = Layout::contiguous(Shape::from_dims(&[8, 64, 64]));
        let err = ensure_rank2_contiguous_zero_offset("test_op", 1, &layout)
            .expect_err("rank-3 must fail");
        let msg = format!("{err}");
        assert!(msg.contains("must be rank-2"));
        assert!(msg.contains("input #1"));
    }

    /// Rank-4 (batch + heads + seq + d) — same rejection.
    #[test]
    fn rank4_rejected() {
        let layout = Layout::contiguous(Shape::from_dims(&[2, 8, 64, 64]));
        assert!(ensure_rank2_contiguous_zero_offset("test", 0, &layout).is_err());
    }
}
