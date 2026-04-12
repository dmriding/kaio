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
//! - [`matmul_auto_tc`] / [`matmul_tc`] / [`matmul_tc_async`] —
//!   tensor-core f16 × f16 → f32 matmul. Multi-warp 64×64 block tile,
//!   4 warps × 32×32 sub-quadrant via 8 mma.sync.m16n8k16 per K-tile.
//!   Edge-tile predication on M and N — only `K % 16 == 0` is
//!   required (mma K-tile is structural). Requires SM 8.0+ (Ampere).
//!   Sprint 6.7 measured performance: **79.9%** of cuBLAS sgemm at
//!   4096² for the sync variant, **85.1%** for the cp.async double-
//!   buffered variant. See `docs/performance.md` for the full table
//!   and the project-local-baseline disclaimer.
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

// Sprint 6.7 D7 promotion (multi-warp restructure + edge tiles +
// 79.9% cuBLAS sgemm at 4096² + benchmark): stable public API.
// Both kernels accept any positive M, N and require K%16=0
// (mma.sync.m16n8k16 K-tile is structural). f16 inputs, f32
// accumulation, SM 8.0+ (Ampere).
pub use matmul_tc_async_kernel::matmul_tc_async;
pub use matmul_tc_kernel::matmul_tc;

// TEMP: Sprint 6.6 final `attention_tc` + `attention_tc_causal` —
// fused TC scaled dot-product attention. #[doc(hidden)] pub use
// until Phase 7 lifts the divisibility + seq_k constraints and adds
// `attention_flash_tc` (at which point `attention_auto_tc` becomes
// the real user-facing dispatcher, matching the `matmul_auto_tc`
// pattern from Sprint 6.5).
#[doc(hidden)]
pub use attention_tc_kernel::{attention_tc, attention_tc_causal};
