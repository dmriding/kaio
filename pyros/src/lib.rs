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
//! **Phase 1 complete.** The IR and runtime layers can construct, emit, load,
//! and execute a `vector_add` kernel on a real GPU. The user-facing proc macro
//! API (`#[gpu_kernel]`) is Phase 2.

#![warn(missing_docs)]

/// PTX code generation — IR types, instruction emitters, and PTX text emission.
pub use pyros_core as core;

/// CUDA runtime — device management, buffers, PTX loading, kernel launch.
pub use pyros_runtime as runtime;

/// Re-export the `#[gpu_kernel]` attribute macro.
pub use pyros_macros::gpu_kernel;
