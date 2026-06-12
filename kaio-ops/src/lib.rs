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
//!   Bank-conflict-padded shared Tile B (Sprint 6.7b col-stride 36 B)
//!   plus fragment-loader `(group_id, tig)` hoist for minimum
//!   shared-memory serialisation on the fragment-B read hot path.
//!   Edge-tile predication on M and N — only `K % 16 == 0` is
//!   required (mma K-tile is structural). Requires SM 8.0+ (Ampere).
//!   **Measured: 107% (sync) / 115% (async) of cuBLAS sgemm at 4096³
//!   on RTX 4090 sm_89, worst-of-10 consecutive runs.** See
//!   `docs/performance.md` for the full distribution tables
//!   (256–4096), the apples-to-apples disclaimer (KAIO is
//!   fp16 × fp16 → fp32 accumulation vs cuBLAS sgemm f32 × f32 → f32),
//!   and the rationale for why async benefits more than sync from
//!   the shared-memory layout improvements.
//! - [`matmul_auto_tc_bf16`] / [`matmul_tc_bf16`] /
//!   [`matmul_tc_bf16_async`] — tensor-core bf16 × bf16 → f32 matmul.
//!   Sibling family to the f16 TC matmul above; same tile shape, warp
//!   layout, mma count per K-iter, and cp.async pipeline. Uses the
//!   dedicated `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
//!   instance — bf16 is a distinct IR boundary, not a runtime cvt
//!   from f16. Bf16 sync ≈ f16 sync and bf16 async ≈ f16 async on
//!   RTX 4090 sm_89 within measurement noise (Sprint 9.1 SC-2 and
//!   Sprint 9.1.1 SC-2 split-bound gates green).
//! - [`matmul_int8`] / [`matmul_int4`] — quantized dequantize-matmul
//!   (W8A8 symmetric and W4A16 GPTQ-style with f16 group scales).
//! - [`qkv_project_int8`] / [`qkv_project_int4`] — fused tri-output
//!   QKV projections (one launch → Q, K, V).
//! - [`attention`] / [`attention_auto`] and causal variants —
//!   fused attention for f32.
//! - [`attention_flash`] / [`attention_flash_causal`] — FlashAttention
//!   forward (online softmax, no O(seq²) memory), plus the
//!   [`attention_flash_with_stats`] / [`attention_flash_causal_with_stats`]
//!   variants that additionally export the per-row logsumexp.
//! - [`attention_flash_bwd`] / [`attention_flash_bwd_causal`] —
//!   FlashAttention backward (three dedicated kernels: D-term
//!   preprocess, dK/dV, dQ; no atomics), consuming the saved stats.
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
mod matmul_int4_kernel;
mod matmul_int8_kernel;
mod matmul_kernel;
mod matmul_tc_async_kernel;
// Sprint 9.1 — bf16 tensor-core matmul. Sync-only sibling of
// `matmul_tc_kernel`; reuses its `pub(crate)` shared-tile loaders,
// store helper, and tile constants (the bf16 byte layout in shared is
// bit-identical). Ships with a cvt-free hot-path gate (host-only test
// asserting zero `cvt.*` between fragment load and mma in the K-loop).
mod matmul_tc_bf16_kernel;
// Sprint 9.1.1 — bf16 async tensor-core matmul. cp.async-pipelined
// sibling of `matmul_tc_bf16_kernel`; cross-product of (f16 async ×
// bf16 sync). Reuses the precision-agnostic cp.async A-tile loader
// from `matmul_tc_async_kernel` and the dedicated bf16 mma helper
// from `matmul_tc_bf16_kernel` (both `pub(crate)`), with its own
// cvt-free hot-path gate.
mod matmul_tc_bf16_async_kernel;
mod matmul_tc_kernel;
// Sprint 7.3 — fused tri-output QKV projection: INT8 (W8A16) and
// INT4 (W4A16) variants.
mod qkv_project_int4_kernel;
mod qkv_project_int8_kernel;
// Shared emit helpers that outlive any single kernel module.
// Currently hosts the fragment-C → packed-f16 global store path used by
// both qkv_project variants (D2).
mod store_out;
// D2.5 register-pressure skeleton for the tri-output QKV projection.
// Test-only module; excluded from release builds via its inner `#![cfg(test)]`.
#[cfg(test)]
mod qkv_skeleton;
mod tuner;

pub use attention_kernel::{
    attention, attention_causal, attention_flash, attention_flash_bwd, attention_flash_bwd_causal,
    attention_flash_causal, attention_flash_causal_with_stats, attention_flash_with_stats,
};
pub use matmul_kernel::matmul;
pub use tuner::{
    attention_auto, attention_auto_causal, matmul_auto, matmul_auto_tc, matmul_auto_tc_bf16,
    tune_attention, tune_attention_causal, tune_matmul, tune_matmul_tc, tune_matmul_tc_bf16,
};

// Expose naive kernel for benchmarking (not public API)
#[doc(hidden)]
pub use matmul_kernel::matmul_naive;

// Built-and-parked ldmatrix fragment-A variant of matmul_tc (Sprint
// 9.3 D6: measured uplift at the bench noise floor, default stays
// ld.shared). Exists so the ldmatrix A/B regression bench can
// interleave both load paths in one process, and as the entry point
// for the XOR-swizzle follow-up's default revisit. Not public API.
#[doc(hidden)]
pub use matmul_tc_kernel::matmul_tc_ldmatrix;

// FlashAttention backward building blocks — exposed for the per-kernel
// correctness tests; the public API is the orchestrating
// attention_flash_bwd / attention_flash_bwd_causal functions.
#[doc(hidden)]
pub use attention_kernel::{
    attention_flash_bwd_dkdv, attention_flash_bwd_dq, attention_flash_bwd_preprocess,
};

// Sprint 6.7 D7 promotion (multi-warp restructure + edge tiles +
// benchmark) + Sprint 6.7b (bank-conflict padding + D10 hoist,
// 82.3% sync / 92.5% async cuBLAS sgemm at 4096²). Stable public
// API as of v0.2.0. Both kernels accept any positive M, N and
// require K%16=0 (mma.sync.m16n8k16 K-tile is structural). f16
// inputs, f32 accumulation, SM 8.0+ (Ampere).
pub use matmul_tc_async_kernel::matmul_tc_async;
pub use matmul_tc_kernel::matmul_tc;

// Sprint 9.1 — bf16 × bf16 → f32 sync tensor-core matmul. Sibling of
// `matmul_tc` (f16); same 64×64 block tile / 4-warp 32×32 quadrant /
// edge-tile predication / Sprint 6.7b D10 fragment hoist. Uses the
// dedicated `TensorCoreOp::MmaSyncBf16` IR variant. Requires
// SM 8.0+ and K%16==0. The async / auto-tuner / candle bf16 variants
// shipped as sub-sprints 9.1.1–9.1.4.
pub use matmul_tc_bf16_kernel::matmul_tc_bf16;

// Sprint 9.1.1 — bf16 × bf16 → f32 cp.async-pipelined tensor-core
// matmul. Async sibling of `matmul_tc_bf16`; cross-product of (f16
// async × bf16 sync). Same 64×64 block tile / 4-warp 32×32 quadrant
// / edge-tile predication / Sprint 6.7b D10 fragment hoist as the
// sync sibling, with double-buffered `cp.async.ca` A staging
// overlapping the K-loop's memory fetch with the previous iteration's
// mma compute. Requires SM 8.0+ and K%16==0.
pub use matmul_tc_bf16_async_kernel::matmul_tc_bf16_async;

// Sprint 7.1 — INT8 symmetric dequantize-matmul (W8A8, i8 × i8 → f32).
// Path FAST: direct mma.sync.m16n8k32.s8.s8.s32 with s32 accumulator,
// single global scalar scale applied post-accumulation. Reference quant
// op for v0.3.0 — GPTQ/AWQ/per-channel/asymmetric/INT4 land as future
// additive refinements. Sync-only; async INT8 deferred to 7.1.5+.
// Requires SM 8.0+ (Ampere) and K%32==0.
pub use matmul_int8_kernel::matmul_int8;

// Sprint 7.2 — INT4 symmetric dequantize-matmul, W4A16 GPTQ-style
// packed 4-bit weights with f16 group scales (group_size=128).
// DEQUANT-F16 path: per-lane unpack + sign-extend + cvt + scale-fold →
// mma.sync.m16n8k16.f16.f16.f32. Reference quant op — not a drop-in
// for external GPTQ/GGUF model formats. Requires SM 8.0+ and K%128==0.
pub use matmul_int4_kernel::matmul_int4;

// Sprint 7.3 MVS — fused tri-output INT8 QKV projection (W8A16: f16
// activations × i8 weights, scalar per-projection scales). Single launch
// produces three GpuBuffer<f16> outputs ready for attention_tc. Saves
// 2× global activation reads vs three matmul_int8 calls. Per-block tile:
// 64×16 (Rollback #1 from D3.4 register-pressure trigger). 7-barrier
// Design-S K-tile cadence. Requires SM 8.0+ and K%16==0, N%2==0.
pub use qkv_project_int8_kernel::qkv_project_int8;

// Sprint 7.3 contingent — fused tri-output INT4 QKV projection (W4A16:
// f16 activations × packed-INT4 weights × f16 group scales, group_size=128).
// Same Design-S serial fusion as the INT8 path; group-scale reload folds
// into the per-projection W load epoch on group-boundary K-tiles. Per-block
// tile 64×16 (Rollback #1 mirror). Requires SM 8.0+ and K%128==0, N%2==0.
pub use qkv_project_int4_kernel::qkv_project_int4;

// `attention_tc` + `attention_tc_causal` — fused TC scaled
// dot-product attention, #[doc(hidden)]: short-sequence (seq_k ≤ 384)
// inference ops with divisibility constraints, consumed by the
// kaio-candle bridge and the QKV showcase. Training and long-sequence
// users route to `attention_flash`, which shipped with backward in
// Sprint 9.2 and has no seq cap.
#[doc(hidden)]
pub use attention_tc_kernel::{attention_tc, attention_tc_causal};
