//! KAIO Python bindings — PyO3 extension module.
//!
//! Exposes the KAIO GPU-kernel framework to Python code through a
//! thin wrapper around the `kaio` and `kaio-ops` Rust crates. The
//! Rust crate `kaio` is aliased to `kaio-rs` in `Cargo.toml` (the
//! `package = "kaio"` key) so its identifier does not collide with
//! this crate's `[lib] name = "kaio"`. All internal imports use
//! `kaio_rs::...`.
//!
//! # Design principles (Phase 8 master plan)
//!
//! - **GIL released during kernel execution.** Every op wrapper
//!   calls `Python::allow_threads(|py| ...)` around the
//!   `kaio-ops` invocation so concurrent Python threads make
//!   progress while the kernel runs.
//! - **Reference-counted Device + Tensor.** `kaio.Device` wraps
//!   `Arc<KaioDevice>`; every `kaio.Tensor` holds a clone of that
//!   `Arc`. The Device cannot be dropped before the Tensors that
//!   reference its CUDA context — Rust's type system enforces
//!   this invariant without lifetime annotations.
//! - **No raw pointers in Python code.** Python users see
//!   `kaio.Device` / `kaio.Tensor` / exception types. Never a raw
//!   CUDA handle.
//! - **Thin wrapper.** 1:1 mapping to the `kaio-ops` host API;
//!   this module does not re-design the surface. Ergonomic
//!   expansion is a later sprint's job, driven by user feedback.

use pyo3::prelude::*;

/// Python module entry point.
///
/// Sprint 8.1 exposes a minimal surface — the scaffold + one smoke
/// kernel. Device, Tensor, KaioError, and matmul_tc land in C2–C5
/// respectively. This module registration is the foundation every
/// later commit attaches to.
#[pymodule]
fn kaio(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
