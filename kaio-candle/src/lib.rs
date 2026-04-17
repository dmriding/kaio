//! # kaio-candle — Candle bridge for KAIO
//!
//! `CustomOp2` / `CustomOp3` bindings between
//! [candle](https://github.com/huggingface/candle) and the
//! [KAIO](https://github.com/dmriding/kaio) GPU kernel library.
//!
//! ## Status — v0.1.0 (Sprint 7.4a–7.4d)
//!
//! Bridges 8 ops across two patterns:
//!
//! **CustomOp-based** (single-output, return `f32`):
//! - `matmul_tc` — f16 × f16 → f32 matmul via KAIO tensor-core kernel. **Backward supported.**
//! - `matmul_tc_async` — same, `cp.async` variant (92.5% cuBLAS sgemm at 4096² on sm_89). **Backward supported.**
//! - `matmul_int4` — GPTQ-style INT4 dequantize-matmul with f16 group scales. Forward-only.
//! - `matmul_int8` — W8A8 symmetric-quant matmul with scalar f32 scale (80–94 TOPS at 4096³ on sm_89). Forward-only.
//! - `attention_tc` — fused tensor-core scaled-dot-product attention. Forward-only.
//! - `attention_tc_causal` — same, with decoder causal mask. Forward-only.
//!
//! **Direct-call** (multi-output, return `f16`, forward-only):
//! - `qkv_project_int8` — W8A16 fused tri-output QKV projection (4 inputs → 3 f16 outputs).
//! - `qkv_project_int4` — W4A16 fused tri-output QKV projection (7 inputs → 3 f16 outputs).
//!
//! CustomOp-based ops return `f32` matching the kaio-ops accumulator.
//! Direct-call ops return `f16` because the fused kernel performs the
//! `f32→f16` conversion internally as part of the projection fusion.
//!
//! ## Backward support
//!
//! `matmul_tc` and `matmul_tc_async` implement `CustomOp2::bwd()` for
//! candle autograd integration. The backward pass computes
//! `dA = grad @ B^T` and `dB = A^T @ grad` by reusing the same
//! forward kernel — no new PTX.
//!
//! **Numerically approximate:** the f32 upstream gradient is downcast to
//! f16 before the tensor-core matmul, and output gradients are cast back
//! to f16 to satisfy candle's dtype-matching constraint. This is an
//! initial autograd integration, not a final mixed-precision training
//! stack.
//!
//! Remaining ops are forward-only: attention backward requires new PTX
//! kernels (Phase 8); quantized ops are inference-only by design.
//!
//! ## Build requirements
//!
//! This crate MUST be built with the `cuda` feature enabled:
//!
//! ```toml
//! [dependencies]
//! kaio-candle = { version = "0.1", features = ["cuda"] }
//! ```
//!
//! The default (feature-less) build produces an empty shell — attempting
//! to call a bridge function like `kaio_candle::matmul_tc(...)` surfaces
//! a "function not found" compile error pointing at the missing feature.
//! This matches candle-core's own opt-in model and keeps `cargo doc` /
//! no-CUDA CI legs working without the toolkit.
//!
//! The `cuda` feature requires the CUDA toolkit at build time (candle-core's
//! cudarc feature uses `dynamic-linking`). Downstream consumers who already
//! build candle with the `cuda` feature see no new system requirement.
//!
//! ## Standalone-crate rationale
//!
//! `kaio-candle` is NOT a member of the main KAIO workspace. cudarc rejects
//! `dynamic-loading` + `dynamic-linking` as simultaneously active; the main
//! KAIO workspace defaults to `dynamic-loading` (no CUDA toolkit required to
//! build), candle requires `dynamic-linking`. Cargo unions features across a
//! workspace build, so including `kaio-candle` in the main workspace would
//! break every no-CUDA CI runner. Standalone keeps the two worlds apart.
//!
//! See `kaio-candle/README.md` for the full rationale.
//!
//! ## Device lifetime
//!
//! The [`std::sync::Arc<kaio::prelude::KaioDevice>`] you construct and pass
//! to `kaio-candle` wrapper functions is independent of the
//! `candle_core::Device` you use for your tensors. Both retain the same CUDA
//! primary context via `cuDevicePrimaryCtxRetain` (verified via scratch
//! probe in `candle-probe/src/bin/probe_ctx.rs`); neither owns the other.
//! Drop order between them is unconstrained.
//!
//! ## Known limitations (v0.1)
//!
//! - **Non-contiguous tensors rejected.** Call `.contiguous()?` upstream.
//! - **Non-zero storage offset rejected** (e.g. from `.narrow(...)` / `.slice(...)`).
//!   Call `.contiguous()?` to compact.
//! - **Rank-2 only.** Multi-head attention reshape to rank-2 before calling;
//!   wrappers error with a concrete reshape hint for higher-rank inputs.
//! - **CUDA Graph capture partially unblocked.** Event-based sync (Sprint
//!   7.4c) removes the prior `cuCtxSynchronize` blocker. However, full CUDA
//!   Graph capture requires non-default streams on both the candle and KAIO
//!   sides, which is not yet verified. Default-stream users should not
//!   attempt graph capture.
//! - **f32 output contract (CustomOp ops).** `matmul_tc`, `matmul_int4`,
//!   `matmul_int8`, and `attention_tc` return `DType::F32` matching the
//!   kaio-ops accumulator. Direct-call ops (`qkv_project_int{4,8}`) return
//!   `DType::F16` because the fused kernel converts internally.
//! - **Bench numbers vs direct-call gap.** Each bridge call issues event-
//!   based stream sync (two `join()` calls — `cuEventRecord` +
//!   `cuStreamWaitEvent` per sync point). This replaced the heavier
//!   `cuCtxSynchronize` from v0.1 but still allocates a transient
//!   `CudaEvent` per call. KAIO's published %-of-cuBLAS numbers are
//!   measured via direct kaio-ops calls, not through the bridge.

#![warn(missing_docs)]

// Without the `cuda` feature, kaio-candle is an empty shell. This matches
// candle-core's own opt-in `cuda` model: consumers who forget the feature
// get a clear "function not found" when they try to call into the bridge
// (e.g. `kaio_candle::matmul_tc(...)`), rather than a lib-level
// `compile_error!` that breaks `cargo check` / `cargo doc` on no-CUDA CI
// legs. The no-default build is exercised in CI — it must succeed on a
// no-CUDA-toolkit host.
#[cfg(feature = "cuda")]
mod bridge;

#[cfg(feature = "cuda")]
mod matmul_tc;
#[cfg(feature = "cuda")]
pub use matmul_tc::{MatmulTcOp, matmul_tc};

#[cfg(feature = "cuda")]
mod matmul_tc_async;
#[cfg(feature = "cuda")]
pub use matmul_tc_async::{MatmulTcAsyncOp, matmul_tc_async};

#[cfg(feature = "cuda")]
mod matmul_int4;
#[cfg(feature = "cuda")]
pub use matmul_int4::{MatmulInt4Op, matmul_int4};

#[cfg(feature = "cuda")]
mod attention_tc;
#[cfg(feature = "cuda")]
pub use attention_tc::{AttentionTcOp, attention_tc, attention_tc_causal};

#[cfg(feature = "cuda")]
mod matmul_int8;
#[cfg(feature = "cuda")]
pub use matmul_int8::{MatmulInt8Op, matmul_int8};

#[cfg(feature = "cuda")]
mod qkv_project_int8;
#[cfg(feature = "cuda")]
pub use qkv_project_int8::qkv_project_int8;

#[cfg(feature = "cuda")]
mod qkv_project_int4;
#[cfg(feature = "cuda")]
pub use qkv_project_int4::qkv_project_int4;
