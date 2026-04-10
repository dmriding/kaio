//! Parsing layer for `#[gpu_kernel]`.
//!
//! Converts `syn` AST into kernel IR types.

pub mod attrs;
pub mod signature;
