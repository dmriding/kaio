//! Kernel-level intermediate representation.
//!
//! Types that bridge `syn`'s AST to the generated `pyros-core` API calls.
//! Internal to the macro crate.

pub mod types;

pub use types::{KernelConfig, KernelParam, KernelSignature, KernelType};
