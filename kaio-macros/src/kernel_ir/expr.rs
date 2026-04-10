//! Kernel expression types for the `#[gpu_kernel]` macro.
//!
//! These types represent parsed expressions from the kernel body,
//! bridging `syn`'s AST to the lowering pass that generates
//! `kaio-core` IR construction code.

use proc_macro2::Span;

use super::KernelType;

/// A kernel expression node.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Variants used progressively across Sprints 2.2-2.8
pub enum KernelExpr {
    /// Integer literal: `42`, `42u32`, `100_i64`.
    LitInt(i64, KernelType, Span),
    /// Float literal: `1.0`, `3.14f64`.
    LitFloat(f64, KernelType, Span),
    /// Boolean literal: `true`, `false`.
    LitBool(bool, Span),
    /// Variable reference: `x`, `idx`, `n`.
    Var(String, Span),
    /// Binary operation: `a + b`, `x * y`, `idx < n`.
    BinOp {
        /// The binary operator.
        op: BinOpKind,
        /// Left-hand operand.
        lhs: Box<KernelExpr>,
        /// Right-hand operand.
        rhs: Box<KernelExpr>,
        /// Source span for error reporting.
        span: Span,
    },
    /// Unary operation: `-x`, `!flag`.
    UnaryOp {
        /// The unary operator.
        op: UnaryOpKind,
        /// Operand expression.
        expr: Box<KernelExpr>,
        /// Source span.
        span: Span,
    },
    /// Array index read: `a[idx]`.
    Index {
        /// Array parameter name (must be a slice parameter).
        array: String,
        /// Index expression.
        index: Box<KernelExpr>,
        /// Source span.
        span: Span,
    },
    /// Built-in function call: `thread_idx_x()`, `sqrt(x)`.
    BuiltinCall {
        /// Function name.
        name: String,
        /// Arguments.
        args: Vec<KernelExpr>,
        /// Source span.
        span: Span,
    },
    /// Type cast: `x as u32`.
    Cast {
        /// Expression to cast.
        expr: Box<KernelExpr>,
        /// Target type.
        target_ty: KernelType,
        /// Source span.
        span: Span,
    },
    /// Parenthesized expression: `(a + b)`.
    Paren(Box<KernelExpr>, Span),
}

/// Binary operator kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // Variants used across multiple sprints
pub enum BinOpKind {
    // Arithmetic (Sprint 2.2)
    /// `+`
    Add,
    /// `-`
    Sub,
    /// `*`
    Mul,
    /// `/`
    Div,
    /// `%`
    Rem,

    // Comparison (Sprint 2.3)
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`
    Ge,
    /// `==`
    Eq,
    /// `!=`
    Ne,

    // Bitwise
    /// `&`
    BitAnd,
    /// `|`
    BitOr,
    /// `^`
    BitXor,
    /// `<<`
    Shl,
    /// `>>`
    Shr,

    // Logical
    /// `&&`
    And,
    /// `||`
    Or,
}

#[allow(dead_code)] // Used in lower/mod.rs, clippy doesn't trace cross-module test usage
impl BinOpKind {
    /// Returns `true` if this is an arithmetic operator.
    pub fn is_arithmetic(&self) -> bool {
        matches!(
            self,
            BinOpKind::Add | BinOpKind::Sub | BinOpKind::Mul | BinOpKind::Div | BinOpKind::Rem
        )
    }

    /// Returns `true` if this is a comparison operator.
    pub fn is_comparison(&self) -> bool {
        matches!(
            self,
            BinOpKind::Lt
                | BinOpKind::Le
                | BinOpKind::Gt
                | BinOpKind::Ge
                | BinOpKind::Eq
                | BinOpKind::Ne
        )
    }
}

/// Unary operator kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // Variants used in parse/body.rs + lower/mod.rs
pub enum UnaryOpKind {
    /// `-x` (arithmetic negation)
    Neg,
    /// `!x` (bitwise not for ints, logical not for bool)
    Not,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binop_is_arithmetic() {
        assert!(BinOpKind::Add.is_arithmetic());
        assert!(BinOpKind::Mul.is_arithmetic());
        assert!(!BinOpKind::Lt.is_arithmetic());
        assert!(!BinOpKind::And.is_arithmetic());
    }

    #[test]
    fn binop_is_comparison() {
        assert!(BinOpKind::Lt.is_comparison());
        assert!(BinOpKind::Ge.is_comparison());
        assert!(!BinOpKind::Add.is_comparison());
        assert!(!BinOpKind::Or.is_comparison());
    }
}
