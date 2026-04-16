//! # kaio-candle — Candle bridge for KAIO
//!
//! Forward-only `CustomOp2` / `CustomOp3` bindings between
//! [candle](https://github.com/huggingface/candle) and the
//! [KAIO](https://github.com/dmriding/kaio) GPU kernel library.
//!
//! ## Status — v0.1.0 (Sprint 7.4a)
//!
//! Currently bridges the LLM-inference-forward trinity plus two
//! zero-incremental-cost neighbours:
//!
//! - `matmul_tc` — f16 × f16 → f32 matmul via KAIO tensor-core kernel.
//! - `matmul_tc_async` — same, `cp.async` variant (92.5% cuBLAS sgemm at 4096² on sm_89).
//! - `matmul_int4` — GPTQ-style INT4 dequantize-matmul with f16 group scales.
//! - `attention_tc` — fused tensor-core scaled-dot-product attention.
//! - `attention_tc_causal` — same, with decoder causal mask.
//!
//! Backward kernels + remaining quant ops (`matmul_int8`, `qkv_project_int{4,8}`)
//! land in 7.4b / 7.4c.
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
//! See `docs/development/sprints/phase7/reviews/internal_7.4a_plan.md`
//! Pre-flight P2 and Round 3 G3-1 for the full history.
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
//! - **CUDA Graph capture incompatible** — wrappers call `cuCtxSynchronize`
//!   which is banned inside a stream-capture region. Returns
//!   `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`. Stream-plumbing in 7.4c
//!   (event-based sync) unblocks Graph usage.
//! - **f32 output contract.** Wrappers return `DType::F32` tensors matching
//!   the kaio-ops accumulator. Cast via `.to_dtype(DType::F16)?` if you need
//!   f16 for downstream graph continuation.
//! - **Bench numbers vs direct-call gap.** Each bridge call issues two
//!   `cuCtxSynchronize`s (one before, one after kaio-ops launch) for stream
//!   safety. Tiny-shape decode is dominated by this; prefill shapes less so.
//!   KAIO's published %-of-cuBLAS numbers are measured via direct kaio-ops
//!   calls, not through the bridge.

#![warn(missing_docs)]

// Without the `cuda` feature, kaio-candle is an empty shell. This matches
// candle-core's own opt-in `cuda` model: consumers who forget the feature
// get a clear "function not found" when they try to call into the bridge
// (e.g. `kaio_candle::matmul_tc(...)`), rather than a lib-level
// `compile_error!` that breaks `cargo check` / `cargo doc` on no-CUDA CI
// legs. Ship gate #10 (AD8) specifically exercises the no-default build —
// it must succeed on a no-CUDA-toolkit host.
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
