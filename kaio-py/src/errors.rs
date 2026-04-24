//! Python exception class + Rust error lift helper.
//!
//! Sprint 8.1 exposes a single `kaio.KaioError` Python exception class
//! under which every `kaio_rs::KaioError` variant surfaces. Subclassing
//! (`KaioValidationError`, `KaioDeviceError`, `KaioPtxError`) is
//! deferred to 8.2 when broader op coverage produces enough error-path
//! diversity for subclasses to earn their weight.
//!
//! # Why a function, not `From<_> for PyErr`
//!
//! `impl From<kaio_rs::KaioError> for PyErr` would be an **orphan
//! impl** (both trait and type are foreign to this crate) and Rust
//! rejects it. A local `map_kaio_err` function is the correct
//! workaround — every call site uses `.map_err(map_kaio_err)?` instead
//! of an implicit `?`-through-`From`, which is slightly more verbose
//! but Rust-legal and zero-footgun.

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

create_exception!(kaio, KaioError, PyException);

/// Lift a `kaio_rs::KaioError` to a Python `KaioError` exception.
///
/// Every Rust-to-Python boundary in this crate routes its error path
/// through this helper, keeping the exception class consistent across
/// the module.
pub(crate) fn map_kaio_err(e: kaio_rs::prelude::KaioError) -> PyErr {
    PyErr::new::<KaioError, _>(format!("{e}"))
}

/// Raise a `KaioError` from a Python-layer validation message (not a
/// lifted `kaio_rs::KaioError`). Keeps input-validation errors on the
/// same exception class as kernel errors.
pub(crate) fn kaio_err(msg: impl Into<String>) -> PyErr {
    PyErr::new::<KaioError, _>(msg.into())
}
