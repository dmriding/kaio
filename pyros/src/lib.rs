//! # PYROS
//!
//! **Rust-native GPU kernel authoring framework.**
//!
//! PYROS (πῦρ — fire) lets developers write GPU compute kernels in Rust and
//! compile them to PTX for execution on NVIDIA GPUs. It is a Rust alternative
//! to OpenAI's Triton, targeting Windows and Linux from day one, with
//! compile-time PTX emission and Rust's type-safety guarantees.
//!
//! ## Crates
//!
//! - [`pyros_core`] — PTX IR types, instruction emitters, PtxWriter
//! - [`pyros_runtime`] — CUDA device management, buffers, PTX loading, kernel launch
//!
//! ## Status
//!
//! **Phase 2 in progress.** `#[gpu_kernel]` proc macro transforms Rust
//! function syntax into PTX codegen + typed launch wrappers.

#![warn(missing_docs)]

pub mod gpu_builtins;
pub mod prelude;

/// PTX code generation — IR types, instruction emitters, and PTX text emission.
pub use pyros_core as core;

/// CUDA runtime — device management, buffers, PTX loading, kernel launch.
pub use pyros_runtime as runtime;

/// Re-export the `#[gpu_kernel]` attribute macro.
pub use pyros_macros::gpu_kernel;
