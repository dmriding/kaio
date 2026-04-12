// Generated launch functions for matmul have many parameters (device + kernel args + grid).
#![allow(clippy::too_many_arguments)]
#![warn(missing_docs)]

//! Pre-built GPU operations for KAIO.
//!
//! Provides high-level functions for common GPU compute operations.
//! Users don't need to write kernels — just call the operation.
//!
//! # Operations
//!
//! - [`matmul`] / [`matmul_auto`] — scalar f32 × f32 → f32 matmul,
//!   works on any NVIDIA GPU (SM 7.0+).
//! - [`matmul_auto_tc`] — tensor-core f16 × f16 → f32 matmul with
//!   auto-tuner dispatch between the synchronous and `cp.async`
//!   double-buffered variants. **Sprint 6.5 preview surface**:
//!   requires SM 8.0+ (Ampere) and `M%16 = N%8 = K%16 = 0`.
//!   Production performance lands in Sprint 6.7's multi-warp
//!   restructure; see the `matmul_auto_tc` rustdoc for the full
//!   contract.
//! - [`attention`] / [`attention_auto`] and causal variants —
//!   fused attention for f32.
//!
//! # Example
//!
//! ```ignore
//! use kaio::prelude::*;
//! use kaio_ops::matmul;
//!
//! let device = KaioDevice::new(0)?;
//! let a = device.alloc_from(&a_data)?;
//! let b = device.alloc_from(&b_data)?;
//! let mut c = device.alloc_zeros::<f32>(m * n)?;
//! matmul(&device, &a, &b, &mut c, m, n, k)?;
//! ```

mod attention_kernel;
mod attention_tc_kernel;
mod matmul_kernel;
mod matmul_tc_async_kernel;
mod matmul_tc_kernel;
mod tuner;

pub use attention_kernel::{attention, attention_causal, attention_flash, attention_flash_causal};
pub use matmul_kernel::matmul;
pub use tuner::{
    attention_auto, attention_auto_causal, matmul_auto, matmul_auto_tc, tune_attention,
    tune_attention_causal, tune_matmul, tune_matmul_tc,
};

// Expose naive kernel for benchmarking (not public API)
#[doc(hidden)]
pub use matmul_kernel::matmul_naive;

// TEMP: hidden export for integration tests. Promote to stable `pub`
// (and announce in README + CHANGELOG) when Sprint 6.7 lifts the
// divisibility constraint and adds edge-tile handling. Until then
// this is crate-internal API that happens to be test-reachable.
#[doc(hidden)]
pub use matmul_tc_kernel::matmul_tc;

// TEMP: hidden export for integration tests — Sprint 6.4 cp.async
// double-buffered variant of matmul_tc. Same dimension constraints
// and promotion trigger as matmul_tc above (Sprint 6.7 lifts
// divisibility + promotes to stable `pub` with README/CHANGELOG
// announcement). Sprint 6.5's `matmul_auto_tc` tuner dispatches
// between this and `matmul_tc` based on benchmarked latency.
#[doc(hidden)]
pub use matmul_tc_async_kernel::matmul_tc_async;

// TEMP: Sprint 6.6 final `attention_tc` + `attention_tc_causal` —
// fused TC scaled dot-product attention. #[doc(hidden)] pub use
// until Phase 7 lifts the divisibility + seq_k constraints and adds
// `attention_flash_tc` (at which point `attention_auto_tc` becomes
// the real user-facing dispatcher, matching the `matmul_auto_tc`
// pattern from Sprint 6.5).
#[doc(hidden)]
pub use attention_tc_kernel::{attention_tc, attention_tc_causal};
