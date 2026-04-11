//! # KAIO
//!
//! **Rust-native GPU kernel authoring framework.**
//!
//! KAIO (καίω — to kindle, to ignite) lets developers write GPU compute
//! kernels in Rust and lower them to PTX for execution on NVIDIA GPUs.
//! A Rust alternative to OpenAI's Triton, targeting Windows and Linux
//! from day one, with automatic PTX generation and Rust's type-safety
//! guarantees.
//!
//! ## Crates
//!
//! - [`kaio_core`] — PTX IR types, instruction emitters, PtxWriter
//! - [`kaio_runtime`] — CUDA device management, buffers, PTX loading, kernel launch
//! - `kaio_macros` — `#[gpu_kernel]` proc macro (re-exported here)
//! - `kaio_ops` — pre-built GPU operations (matmul, more planned).
//!   Separate crate, not re-exported from `kaio`. Add with `cargo add kaio-ops`.
//!
//! ## Status
//!
//! **Phase 4 complete.** Tiled matmul (31% of cuBLAS sgemm), `kaio-ops`
//! crate, 2D thread blocks, FMA, PTX inspection tools. See the
//! repository README for the full feature table and roadmap.

#![warn(missing_docs)]

pub mod gpu_builtins;
pub mod prelude;

/// PTX code generation — IR types, instruction emitters, and PTX text emission.
pub use kaio_core as core;

/// CUDA runtime — device management, buffers, PTX loading, kernel launch.
pub use kaio_runtime as runtime;

/// Re-export the `#[gpu_kernel]` attribute macro.
pub use kaio_macros::gpu_kernel;
