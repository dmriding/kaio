//! Kernel-level intermediate representation.
//!
//! Types that bridge `syn`'s AST to the generated `pyros-core` API calls.
//! Internal to the macro crate.

pub mod expr;
pub mod stmt;
pub mod types;

// Re-exports for convenience within the crate.
#[allow(unused_imports)]
pub use expr::{BinOpKind, KernelExpr, UnaryOpKind};
#[allow(unused_imports)]
pub use stmt::KernelStmt;
pub use types::{KernelConfig, KernelParam, KernelSignature, KernelType};
