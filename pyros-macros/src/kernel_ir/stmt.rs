//! Kernel statement types for the `#[gpu_kernel]` macro.

use proc_macro2::Span;

use super::KernelType;
use super::expr::KernelExpr;

/// A kernel statement node.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Variants used progressively across Sprints 2.2-2.8
pub enum KernelStmt {
    /// `let x = expr;` or `let x: Type = expr;`
    Let {
        /// Variable name.
        name: String,
        /// Optional explicit type annotation.
        ty: Option<KernelType>,
        /// Initializer expression.
        value: KernelExpr,
        /// Source span.
        span: Span,
    },
    /// `x = expr;` (reassignment of an existing let binding).
    Assign {
        /// Variable name.
        name: String,
        /// New value.
        value: KernelExpr,
        /// Source span.
        span: Span,
    },
    /// `arr[idx] = expr;` (write to a mutable slice).
    IndexAssign {
        /// Slice parameter name.
        array: String,
        /// Index expression.
        index: KernelExpr,
        /// Value to store.
        value: KernelExpr,
        /// Source span.
        span: Span,
    },
    /// `if cond { ... }` or `if cond { ... } else { ... }`.
    If {
        /// Condition expression (must evaluate to bool).
        condition: KernelExpr,
        /// Then-branch statements.
        then_body: Vec<KernelStmt>,
        /// Optional else-branch statements.
        else_body: Option<Vec<KernelStmt>>,
        /// Source span.
        span: Span,
    },
    /// Bare expression statement (e.g., a function call).
    Expr(KernelExpr, Span),
}
