//! # pyros-runtime
//!
//! CUDA runtime layer for the PYROS GPU kernel authoring framework. This
//! crate wraps [`cudarc`] to provide device management, typed device
//! buffers, PTX module loading, and a builder-style kernel launch API. It
//! is Layer 2 of PYROS, sitting on top of [`pyros-core`] (PTX emission).
//!
//! **Status:** Phase 1 scaffolding. No public API yet.
//!
//! [`cudarc`]: https://crates.io/crates/cudarc
//! [`pyros-core`]: https://crates.io/crates/pyros-core

#![warn(missing_docs)]
