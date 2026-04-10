//! Parse a kernel function body into `KernelStmt` / `KernelExpr`.
//!
//! Converts `syn`'s rich AST into the limited subset supported by
//! GPU kernels. Unsupported constructs produce `syn::Error` with
//! spans pointing at the offending code.

#![allow(dead_code)] // All functions used in Sprint 2.6 codegen; tested in this module

use proc_macro2::Span;
use syn::spanned::Spanned;
use syn::{BinOp, Block, Expr, ExprLit, Lit, Local, RangeLimits, Stmt, UnOp};

use crate::kernel_ir::KernelType;
use crate::kernel_ir::expr::{BinOpKind, KernelExpr, UnaryOpKind};
use crate::kernel_ir::stmt::KernelStmt;

/// Parse a function body block into a sequence of kernel statements.
pub fn parse_body(block: &Block) -> syn::Result<Vec<KernelStmt>> {
    let mut stmts = Vec::new();
    for stmt in &block.stmts {
        stmts.push(parse_stmt(stmt)?);
    }
    Ok(stmts)
}

fn parse_stmt(stmt: &Stmt) -> syn::Result<KernelStmt> {
    match stmt {
        Stmt::Local(local) => parse_let(local),
        Stmt::Expr(expr, _semi) => parse_expr_stmt(expr),
        _ => Err(syn::Error::new_spanned(
            stmt,
            "unsupported statement in GPU kernel",
        )),
    }
}

fn parse_let(local: &Local) -> syn::Result<KernelStmt> {
    let span = local.span();

    // Extract variable name
    let name = match &local.pat {
        syn::Pat::Ident(pat_ident) => pat_ident.ident.to_string(),
        syn::Pat::Type(pat_type) => {
            // `let x: Type = expr;` — name is inside the pattern
            match pat_type.pat.as_ref() {
                syn::Pat::Ident(pat_ident) => pat_ident.ident.to_string(),
                _ => {
                    return Err(syn::Error::new_spanned(
                        &pat_type.pat,
                        "only simple identifier patterns are supported in GPU kernel let bindings",
                    ));
                }
            }
        }
        _ => {
            return Err(syn::Error::new_spanned(
                &local.pat,
                "only simple identifier patterns are supported in GPU kernel let bindings",
            ));
        }
    };

    // Extract optional type annotation
    let ty = match &local.pat {
        syn::Pat::Type(pat_type) => Some(parse_type_from_syn(&pat_type.ty)?),
        _ => None,
    };

    // Extract initializer
    let init = local.init.as_ref().ok_or_else(|| {
        syn::Error::new(span, "let bindings in GPU kernels must have an initializer")
    })?;

    // Check for shared_mem! macro: `let buf = shared_mem![f32; 256];`
    if let Expr::Macro(expr_macro) = init.expr.as_ref()
        && expr_macro.mac.path.is_ident("shared_mem")
    {
        let (elem_ty, count) = parse_shared_mem_tokens(&expr_macro.mac, span)?;
        return Ok(KernelStmt::SharedMemDecl {
            name,
            elem_ty,
            count,
            span,
        });
    }

    let value = parse_expr(&init.expr)?;

    Ok(KernelStmt::Let {
        name,
        ty,
        value,
        span,
    })
}

/// Parse the inner tokens of `shared_mem![T; N]`.
fn parse_shared_mem_tokens(mac: &syn::Macro, span: Span) -> syn::Result<(KernelType, usize)> {
    use syn::parse::Parser;

    let parser = |input: syn::parse::ParseStream| {
        // Parse type: f32, u32, etc.
        let ty: syn::Type = input.parse()?;
        // Parse semicolon separator
        input.parse::<syn::Token![;]>()?;
        // Parse count: integer literal
        let count: syn::LitInt = input.parse()?;
        Ok((ty, count))
    };

    let (ty, count_lit) = parser.parse2(mac.tokens.clone()).map_err(|_| {
        syn::Error::new(
            span,
            "invalid shared_mem! syntax — expected `shared_mem![Type; count]` \
             (e.g., `shared_mem![f32; 256]`)",
        )
    })?;

    let elem_ty = parse_type_from_syn(&ty)?;
    if elem_ty.is_slice() {
        return Err(syn::Error::new(
            span,
            "shared_mem! element type must be a scalar (f32, u32, etc.), not a slice",
        ));
    }

    let count: usize = count_lit
        .base10_parse()
        .map_err(|_| syn::Error::new(span, "shared_mem! count must be a positive integer"))?;
    if count == 0 {
        return Err(syn::Error::new(
            span,
            "shared_mem! count must be greater than 0",
        ));
    }

    Ok((elem_ty, count))
}

/// Parse an expression as a statement (handles assignment, if, and bare expressions).
fn parse_expr_stmt(expr: &Expr) -> syn::Result<KernelStmt> {
    let span = expr.span();
    match expr {
        // if/else
        Expr::If(expr_if) => {
            let condition = parse_expr(&expr_if.cond)?;
            let then_body = parse_body(&expr_if.then_branch)?;
            let else_body = match &expr_if.else_branch {
                Some((_else_token, else_expr)) => match else_expr.as_ref() {
                    Expr::Block(expr_block) => Some(parse_body(&expr_block.block)?),
                    Expr::If(_) => {
                        // else if — wrap in a single-element vec
                        Some(vec![parse_expr_stmt(else_expr)?])
                    }
                    _ => {
                        return Err(syn::Error::new_spanned(
                            else_expr,
                            "unsupported else branch in GPU kernel",
                        ));
                    }
                },
                None => None,
            };
            Ok(KernelStmt::If {
                condition,
                then_body,
                else_body,
                span,
            })
        }
        // for i in start..end { body }
        Expr::ForLoop(expr_for) => {
            // Extract loop variable name (must be a simple ident)
            let var = match &*expr_for.pat {
                syn::Pat::Ident(pat_ident) => pat_ident.ident.to_string(),
                _ => {
                    return Err(syn::Error::new_spanned(
                        &expr_for.pat,
                        "only simple identifiers are supported as `for` loop variables \
                         in GPU kernels (e.g., `for i in 0..n`)",
                    ));
                }
            };
            // Extract range bounds — must be `start..end` (half-open)
            let (start, end) = match &*expr_for.expr {
                Expr::Range(expr_range) => {
                    match &expr_range.limits {
                        RangeLimits::HalfOpen(_) => {}
                        RangeLimits::Closed(_) => {
                            return Err(syn::Error::new_spanned(
                                expr_range.limits,
                                "inclusive ranges (`..=`) are not supported in GPU kernel \
                                 `for` loops — use `start..end` instead",
                            ));
                        }
                    }
                    let start = expr_range.start.as_ref().ok_or_else(|| {
                        syn::Error::new_spanned(
                            &expr_for.expr,
                            "`for` loop range must have a start bound (e.g., `0..n`)",
                        )
                    })?;
                    let end = expr_range.end.as_ref().ok_or_else(|| {
                        syn::Error::new_spanned(
                            &expr_for.expr,
                            "`for` loop range must have an end bound (e.g., `0..n`)",
                        )
                    })?;
                    (parse_expr(start)?, parse_expr(end)?)
                }
                _ => {
                    return Err(syn::Error::new_spanned(
                        &expr_for.expr,
                        "only `start..end` ranges are supported in GPU kernel `for` loops \
                         (iterators like `.iter()` or `.step_by()` are not supported)",
                    ));
                }
            };
            let body = parse_body(&expr_for.body)?;
            Ok(KernelStmt::For {
                var,
                start,
                end,
                body,
                span,
            })
        }
        // while condition { body }
        Expr::While(expr_while) => {
            let condition = parse_expr(&expr_while.cond)?;
            let body = parse_body(&expr_while.body)?;
            Ok(KernelStmt::While {
                condition,
                body,
                span,
            })
        }
        // Assignment: x = expr or arr[idx] = expr
        Expr::Assign(expr_assign) => {
            match expr_assign.left.as_ref() {
                // arr[idx] = expr
                Expr::Index(expr_index) => {
                    let array = extract_var_name(&expr_index.expr)?;
                    let index = parse_expr(&expr_index.index)?;
                    let value = parse_expr(&expr_assign.right)?;
                    Ok(KernelStmt::IndexAssign {
                        array,
                        index,
                        value,
                        span,
                    })
                }
                // x = expr
                Expr::Path(expr_path) => {
                    let name = extract_path_name(expr_path)?;
                    let value = parse_expr(&expr_assign.right)?;
                    Ok(KernelStmt::Assign { name, value, span })
                }
                _ => Err(syn::Error::new_spanned(
                    &expr_assign.left,
                    "unsupported assignment target in GPU kernel",
                )),
            }
        }
        // Compound assignment: x += expr, arr[i] += expr
        // In syn 2, compound assignment is Expr::Binary with BinOp::AddAssign etc.
        Expr::Binary(expr_bin) if desugar_compound_op(&expr_bin.op).is_some() => {
            let base_op = desugar_compound_op(&expr_bin.op).unwrap();
            let rhs = parse_expr(&expr_bin.right)?;
            match expr_bin.left.as_ref() {
                // x += expr → Assign { name: "x", value: BinOp(Var("x"), Add, expr) }
                Expr::Path(expr_path) => {
                    let name = extract_path_name(expr_path)?;
                    let lhs_expr = KernelExpr::Var(name.clone(), span);
                    let value = KernelExpr::BinOp {
                        op: base_op,
                        lhs: Box::new(lhs_expr),
                        rhs: Box::new(rhs),
                        span,
                    };
                    Ok(KernelStmt::Assign { name, value, span })
                }
                // arr[i] += expr → IndexAssign { ..., value: BinOp(Index(...), Add, expr) }
                Expr::Index(expr_index) => {
                    let array = extract_var_name(&expr_index.expr)?;
                    let index = parse_expr(&expr_index.index)?;
                    let lhs_expr = KernelExpr::Index {
                        array: array.clone(),
                        index: Box::new(index.clone()),
                        span,
                    };
                    let value = KernelExpr::BinOp {
                        op: base_op,
                        lhs: Box::new(lhs_expr),
                        rhs: Box::new(rhs),
                        span,
                    };
                    Ok(KernelStmt::IndexAssign {
                        array,
                        index,
                        value,
                        span,
                    })
                }
                _ => Err(syn::Error::new_spanned(
                    &expr_bin.left,
                    "unsupported compound assignment target in GPU kernel",
                )),
            }
        }
        // Bare expression
        _ => {
            let kernel_expr = parse_expr(expr)?;
            Ok(KernelStmt::Expr(kernel_expr, span))
        }
    }
}

/// Map a compound assignment operator to its base arithmetic operator.
/// Returns `None` for non-compound operators.
fn desugar_compound_op(op: &BinOp) -> Option<BinOpKind> {
    match op {
        BinOp::AddAssign(_) => Some(BinOpKind::Add),
        BinOp::SubAssign(_) => Some(BinOpKind::Sub),
        BinOp::MulAssign(_) => Some(BinOpKind::Mul),
        BinOp::DivAssign(_) => Some(BinOpKind::Div),
        BinOp::RemAssign(_) => Some(BinOpKind::Rem),
        _ => None,
    }
}

/// Parse a `syn::Expr` into a `KernelExpr`.
pub fn parse_expr(expr: &Expr) -> syn::Result<KernelExpr> {
    let span = expr.span();
    match expr {
        // Binary: a + b, x < n, etc.
        Expr::Binary(expr_bin) => {
            let op = convert_binop(&expr_bin.op)?;
            let lhs = parse_expr(&expr_bin.left)?;
            let rhs = parse_expr(&expr_bin.right)?;
            Ok(KernelExpr::BinOp {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
                span,
            })
        }
        // Unary: -x, !flag
        Expr::Unary(expr_unary) => {
            let op = match expr_unary.op {
                UnOp::Neg(_) => UnaryOpKind::Neg,
                UnOp::Not(_) => UnaryOpKind::Not,
                _ => {
                    return Err(syn::Error::new_spanned(
                        expr,
                        "unsupported unary operator in GPU kernel",
                    ));
                }
            };
            let inner = parse_expr(&expr_unary.expr)?;
            Ok(KernelExpr::UnaryOp {
                op,
                expr: Box::new(inner),
                span,
            })
        }
        // Literal
        Expr::Lit(ExprLit { lit, .. }) => parse_literal(lit),
        // Variable reference: x, idx
        Expr::Path(expr_path) => {
            let name = extract_path_name(expr_path)?;
            Ok(KernelExpr::Var(name, span))
        }
        // Array index: a[idx]
        Expr::Index(expr_index) => {
            let array = extract_var_name(&expr_index.expr)?;
            let index = parse_expr(&expr_index.index)?;
            Ok(KernelExpr::Index {
                array,
                index: Box::new(index),
                span,
            })
        }
        // Function call: thread_idx_x(), sqrt(x)
        Expr::Call(expr_call) => {
            let name = extract_var_name(&expr_call.func)?;
            let mut args = Vec::new();
            for arg in &expr_call.args {
                args.push(parse_expr(arg)?);
            }
            Ok(KernelExpr::BuiltinCall { name, args, span })
        }
        // Type cast: x as u32
        Expr::Cast(expr_cast) => {
            let inner = parse_expr(&expr_cast.expr)?;
            let target_ty = parse_type_from_syn(&expr_cast.ty)?;
            Ok(KernelExpr::Cast {
                expr: Box::new(inner),
                target_ty,
                span,
            })
        }
        // Parenthesized: (a + b)
        Expr::Paren(expr_paren) => {
            let inner = parse_expr(&expr_paren.expr)?;
            Ok(KernelExpr::Paren(Box::new(inner), span))
        }
        // Block expression: { ... } (used in if/else)
        Expr::Block(expr_block) => {
            // A block with a single trailing expression — parse it
            if expr_block.block.stmts.len() == 1
                && let Stmt::Expr(inner_expr, None) = &expr_block.block.stmts[0]
            {
                return parse_expr(inner_expr);
            }
            Err(syn::Error::new_spanned(
                expr,
                "block expressions are not supported in GPU kernels",
            ))
        }
        // --- Unsupported constructs with specific error messages ---
        Expr::Loop(_) => Err(syn::Error::new_spanned(
            expr,
            "`loop` is not supported in GPU kernels — use `for` or `while` instead",
        )),
        Expr::Match(_) => Err(syn::Error::new_spanned(
            expr,
            "`match` is not supported in GPU kernels",
        )),
        Expr::Closure(_) => Err(syn::Error::new_spanned(
            expr,
            "closures are not supported in GPU kernels",
        )),
        Expr::Return(_) => Err(syn::Error::new_spanned(
            expr,
            "`return` is not supported in GPU kernels",
        )),
        Expr::Unsafe(_) => Err(syn::Error::new_spanned(
            expr,
            "`unsafe` blocks are not supported in GPU kernels",
        )),
        Expr::Macro(expr_macro) => {
            if expr_macro.mac.path.is_ident("shared_mem") {
                Err(syn::Error::new_spanned(
                    expr,
                    "shared_mem![] must be assigned to a variable: \
                     `let buf = shared_mem![f32; 256];`",
                ))
            } else {
                Err(syn::Error::new_spanned(
                    expr,
                    "macro invocations are not supported in GPU kernels",
                ))
            }
        }
        Expr::MethodCall(_) => Err(syn::Error::new_spanned(
            expr,
            "method calls are not supported in GPU kernels",
        )),
        Expr::Struct(_) => Err(syn::Error::new_spanned(
            expr,
            "struct construction is not supported in GPU kernels",
        )),
        Expr::Tuple(_) => Err(syn::Error::new_spanned(
            expr,
            "tuples are not supported in GPU kernels",
        )),
        Expr::Range(_) => Err(syn::Error::new_spanned(
            expr,
            "range expressions are not supported in GPU kernels",
        )),
        Expr::Try(_) => Err(syn::Error::new_spanned(
            expr,
            "the `?` operator is not supported in GPU kernels",
        )),
        _ => Err(syn::Error::new_spanned(
            expr,
            "unsupported expression in GPU kernel",
        )),
    }
}

fn parse_literal(lit: &Lit) -> syn::Result<KernelExpr> {
    let span = lit.span();
    match lit {
        Lit::Int(lit_int) => {
            let value: i64 = lit_int.base10_parse().map_err(|_| {
                syn::Error::new(span, "integer literal out of range for GPU kernel")
            })?;
            let ty = match lit_int.suffix() {
                "" => KernelType::I32, // default
                "i32" => KernelType::I32,
                "u32" => KernelType::U32,
                "i64" => KernelType::I64,
                "u64" => KernelType::U64,
                other => {
                    return Err(syn::Error::new(
                        span,
                        format!("unsupported integer suffix `{other}` in GPU kernel"),
                    ));
                }
            };
            Ok(KernelExpr::LitInt(value, ty, span))
        }
        Lit::Float(lit_float) => {
            let value: f64 = lit_float
                .base10_parse()
                .map_err(|_| syn::Error::new(span, "float literal out of range for GPU kernel"))?;
            let ty = match lit_float.suffix() {
                "" => KernelType::F32, // default
                "f32" => KernelType::F32,
                "f64" => KernelType::F64,
                other => {
                    return Err(syn::Error::new(
                        span,
                        format!("unsupported float suffix `{other}` in GPU kernel"),
                    ));
                }
            };
            Ok(KernelExpr::LitFloat(value, ty, span))
        }
        Lit::Bool(lit_bool) => Ok(KernelExpr::LitBool(lit_bool.value, span)),
        _ => Err(syn::Error::new(
            span,
            "unsupported literal type in GPU kernel (only int, float, and bool are supported)",
        )),
    }
}

/// Extract a simple variable name from a single-segment path expression.
fn extract_path_name(expr_path: &syn::ExprPath) -> syn::Result<String> {
    if expr_path.qself.is_some() || expr_path.path.segments.len() != 1 {
        return Err(syn::Error::new_spanned(
            expr_path,
            "only simple variable names are supported in GPU kernels (no paths like a::b)",
        ));
    }
    Ok(expr_path.path.segments[0].ident.to_string())
}

/// Extract a variable name from an expression (must be a simple path).
fn extract_var_name(expr: &Expr) -> syn::Result<String> {
    match expr {
        Expr::Path(expr_path) => extract_path_name(expr_path),
        _ => Err(syn::Error::new_spanned(
            expr,
            "expected a variable name in GPU kernel",
        )),
    }
}

/// Convert a `syn::BinOp` to our `BinOpKind`.
fn convert_binop(op: &BinOp) -> syn::Result<BinOpKind> {
    match op {
        BinOp::Add(_) => Ok(BinOpKind::Add),
        BinOp::Sub(_) => Ok(BinOpKind::Sub),
        BinOp::Mul(_) => Ok(BinOpKind::Mul),
        BinOp::Div(_) => Ok(BinOpKind::Div),
        BinOp::Rem(_) => Ok(BinOpKind::Rem),
        BinOp::Lt(_) => Ok(BinOpKind::Lt),
        BinOp::Le(_) => Ok(BinOpKind::Le),
        BinOp::Gt(_) => Ok(BinOpKind::Gt),
        BinOp::Ge(_) => Ok(BinOpKind::Ge),
        BinOp::Eq(_) => Ok(BinOpKind::Eq),
        BinOp::Ne(_) => Ok(BinOpKind::Ne),
        BinOp::BitAnd(_) => Ok(BinOpKind::BitAnd),
        BinOp::BitOr(_) => Ok(BinOpKind::BitOr),
        BinOp::BitXor(_) => Ok(BinOpKind::BitXor),
        BinOp::Shl(_) => Ok(BinOpKind::Shl),
        BinOp::Shr(_) => Ok(BinOpKind::Shr),
        BinOp::And(_) => Ok(BinOpKind::And),
        BinOp::Or(_) => Ok(BinOpKind::Or),
        _ => Err(syn::Error::new_spanned(
            op,
            "unsupported binary operator in GPU kernel",
        )),
    }
}

/// Parse a `syn::Type` into a `KernelType` (for cast targets and let annotations).
fn parse_type_from_syn(ty: &syn::Type) -> syn::Result<KernelType> {
    match ty {
        syn::Type::Path(type_path) => {
            if type_path.qself.is_some() || type_path.path.segments.len() != 1 {
                return Err(syn::Error::new_spanned(
                    ty,
                    "unsupported type in GPU kernel",
                ));
            }
            match type_path.path.segments[0].ident.to_string().as_str() {
                "f32" => Ok(KernelType::F32),
                "f64" => Ok(KernelType::F64),
                "i32" => Ok(KernelType::I32),
                "u32" => Ok(KernelType::U32),
                "i64" => Ok(KernelType::I64),
                "u64" => Ok(KernelType::U64),
                "bool" => Ok(KernelType::Bool),
                other => Err(syn::Error::new_spanned(
                    ty,
                    format!("unsupported type `{other}` in GPU kernel"),
                )),
            }
        }
        _ => Err(syn::Error::new_spanned(
            ty,
            "unsupported type in GPU kernel",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;

    /// Helper: parse a block of statements from tokens.
    fn parse_block(tokens: proc_macro2::TokenStream) -> Block {
        let func: syn::ItemFn = syn::parse2(quote! { fn test() #tokens }).unwrap();
        *func.block
    }

    /// Helper: parse a single expression from tokens.
    fn parse_single_expr(tokens: proc_macro2::TokenStream) -> KernelExpr {
        let expr: syn::Expr = syn::parse2(tokens).unwrap();
        parse_expr(&expr).unwrap()
    }

    #[test]
    fn parse_let_simple() {
        let block = parse_block(quote! { { let x = a + b; } });
        let stmts = parse_body(&block).unwrap();
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            KernelStmt::Let {
                name, ty, value, ..
            } => {
                assert_eq!(name, "x");
                assert!(ty.is_none());
                match value {
                    KernelExpr::BinOp { op, lhs, rhs, .. } => {
                        assert_eq!(*op, BinOpKind::Add);
                        assert!(matches!(lhs.as_ref(), KernelExpr::Var(n, _) if n == "a"));
                        assert!(matches!(rhs.as_ref(), KernelExpr::Var(n, _) if n == "b"));
                    }
                    _ => panic!("expected BinOp"),
                }
            }
            _ => panic!("expected Let"),
        }
    }

    #[test]
    fn parse_let_with_type() {
        let block = parse_block(quote! { { let x: u32 = n; } });
        let stmts = parse_body(&block).unwrap();
        match &stmts[0] {
            KernelStmt::Let { name, ty, .. } => {
                assert_eq!(name, "x");
                assert_eq!(ty.as_ref(), Some(&KernelType::U32));
            }
            _ => panic!("expected Let"),
        }
    }

    #[test]
    fn parse_precedence() {
        // a + b * c should parse as Add(a, Mul(b, c)) due to syn's precedence
        let expr = parse_single_expr(quote! { a + b * c });
        match &expr {
            KernelExpr::BinOp { op, lhs, rhs, .. } => {
                assert_eq!(*op, BinOpKind::Add);
                assert!(matches!(lhs.as_ref(), KernelExpr::Var(n, _) if n == "a"));
                match rhs.as_ref() {
                    KernelExpr::BinOp { op: inner_op, .. } => {
                        assert_eq!(*inner_op, BinOpKind::Mul);
                    }
                    _ => panic!("expected inner BinOp(Mul)"),
                }
            }
            _ => panic!("expected BinOp"),
        }
    }

    #[test]
    fn parse_int_literals() {
        // Unsuffixed -> I32
        match parse_single_expr(quote! { 42 }) {
            KernelExpr::LitInt(val, ty, _) => {
                assert_eq!(val, 42);
                assert_eq!(ty, KernelType::I32);
            }
            _ => panic!("expected LitInt"),
        }

        // Suffixed u32
        match parse_single_expr(quote! { 42u32 }) {
            KernelExpr::LitInt(val, ty, _) => {
                assert_eq!(val, 42);
                assert_eq!(ty, KernelType::U32);
            }
            _ => panic!("expected LitInt"),
        }

        // Suffixed i64
        match parse_single_expr(quote! { 100i64 }) {
            KernelExpr::LitInt(val, ty, _) => {
                assert_eq!(val, 100);
                assert_eq!(ty, KernelType::I64);
            }
            _ => panic!("expected LitInt"),
        }
    }

    #[test]
    fn parse_float_literals() {
        // Unsuffixed -> F32
        match parse_single_expr(quote! { 1.0 }) {
            KernelExpr::LitFloat(val, ty, _) => {
                assert!((val - 1.0).abs() < f64::EPSILON);
                assert_eq!(ty, KernelType::F32);
            }
            _ => panic!("expected LitFloat"),
        }

        // Suffixed f64
        match parse_single_expr(quote! { 3.14f64 }) {
            KernelExpr::LitFloat(val, ty, _) => {
                assert!((val - 3.14).abs() < 0.001);
                assert_eq!(ty, KernelType::F64);
            }
            _ => panic!("expected LitFloat"),
        }
    }

    #[test]
    fn parse_bool_literal() {
        match parse_single_expr(quote! { true }) {
            KernelExpr::LitBool(val, _) => assert!(val),
            _ => panic!("expected LitBool"),
        }
    }

    #[test]
    fn parse_unary_neg() {
        match parse_single_expr(quote! { -x }) {
            KernelExpr::UnaryOp { op, expr, .. } => {
                assert_eq!(op, UnaryOpKind::Neg);
                assert!(matches!(expr.as_ref(), KernelExpr::Var(n, _) if n == "x"));
            }
            _ => panic!("expected UnaryOp"),
        }
    }

    #[test]
    fn parse_function_call() {
        match parse_single_expr(quote! { thread_idx_x() }) {
            KernelExpr::BuiltinCall { name, args, .. } => {
                assert_eq!(name, "thread_idx_x");
                assert!(args.is_empty());
            }
            _ => panic!("expected BuiltinCall"),
        }
    }

    #[test]
    fn parse_function_call_with_args() {
        match parse_single_expr(quote! { sqrt(x) }) {
            KernelExpr::BuiltinCall { name, args, .. } => {
                assert_eq!(name, "sqrt");
                assert_eq!(args.len(), 1);
            }
            _ => panic!("expected BuiltinCall"),
        }
    }

    #[test]
    fn parse_index() {
        match parse_single_expr(quote! { a[idx] }) {
            KernelExpr::Index { array, index, .. } => {
                assert_eq!(array, "a");
                assert!(matches!(index.as_ref(), KernelExpr::Var(n, _) if n == "idx"));
            }
            _ => panic!("expected Index"),
        }
    }

    #[test]
    fn parse_cast() {
        match parse_single_expr(quote! { x as f32 }) {
            KernelExpr::Cast {
                expr, target_ty, ..
            } => {
                assert!(matches!(expr.as_ref(), KernelExpr::Var(n, _) if n == "x"));
                assert_eq!(target_ty, KernelType::F32);
            }
            _ => panic!("expected Cast"),
        }
    }

    #[test]
    fn parse_if_else() {
        let block = parse_block(quote! { {
            if idx < n {
                out[idx] = a[idx] + b[idx];
            }
        } });
        let stmts = parse_body(&block).unwrap();
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            KernelStmt::If {
                condition,
                then_body,
                else_body,
                ..
            } => {
                assert!(matches!(
                    condition,
                    KernelExpr::BinOp {
                        op: BinOpKind::Lt,
                        ..
                    }
                ));
                assert_eq!(then_body.len(), 1);
                assert!(else_body.is_none());
            }
            _ => panic!("expected If"),
        }
    }

    #[test]
    fn parse_index_assign() {
        let block = parse_block(quote! { { out[idx] = x + y; } });
        let stmts = parse_body(&block).unwrap();
        match &stmts[0] {
            KernelStmt::IndexAssign {
                array,
                index,
                value,
                ..
            } => {
                assert_eq!(array, "out");
                assert!(matches!(index, KernelExpr::Var(n, _) if n == "idx"));
                assert!(matches!(
                    value,
                    KernelExpr::BinOp {
                        op: BinOpKind::Add,
                        ..
                    }
                ));
            }
            _ => panic!("expected IndexAssign"),
        }
    }

    // --- Loop parsing tests ---

    #[test]
    fn parse_for_loop() {
        let block = parse_block(quote! { {
            for i in 0..n {
                out[i] = 0.0;
            }
        } });
        let stmts = parse_body(&block).unwrap();
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            KernelStmt::For {
                var,
                start,
                end,
                body,
                ..
            } => {
                assert_eq!(var, "i");
                assert!(matches!(start, KernelExpr::LitInt(0, KernelType::I32, _)));
                assert!(matches!(end, KernelExpr::Var(n, _) if n == "n"));
                assert_eq!(body.len(), 1);
            }
            _ => panic!("expected For"),
        }
    }

    #[test]
    fn parse_while_loop() {
        let block = parse_block(quote! { {
            while x > 0 {
                x = x - 1;
            }
        } });
        let stmts = parse_body(&block).unwrap();
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            KernelStmt::While {
                condition, body, ..
            } => {
                assert!(matches!(
                    condition,
                    KernelExpr::BinOp {
                        op: BinOpKind::Gt,
                        ..
                    }
                ));
                assert_eq!(body.len(), 1);
            }
            _ => panic!("expected While"),
        }
    }

    #[test]
    fn parse_compound_assign_variable() {
        let block = parse_block(quote! { { x += 1; } });
        let stmts = parse_body(&block).unwrap();
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            KernelStmt::Assign { name, value, .. } => {
                assert_eq!(name, "x");
                // value should be BinOp(Var("x"), Add, LitInt(1))
                match value {
                    KernelExpr::BinOp { op, lhs, rhs, .. } => {
                        assert_eq!(*op, BinOpKind::Add);
                        assert!(matches!(lhs.as_ref(), KernelExpr::Var(n, _) if n == "x"));
                        assert!(matches!(rhs.as_ref(), KernelExpr::LitInt(1, _, _)));
                    }
                    _ => panic!("expected BinOp"),
                }
            }
            _ => panic!("expected Assign"),
        }
    }

    #[test]
    fn parse_compound_assign_index() {
        let block = parse_block(quote! { { arr[i] += val; } });
        let stmts = parse_body(&block).unwrap();
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            KernelStmt::IndexAssign {
                array,
                index,
                value,
                ..
            } => {
                assert_eq!(array, "arr");
                assert!(matches!(index, KernelExpr::Var(n, _) if n == "i"));
                // value should be BinOp(Index("arr", Var("i")), Add, Var("val"))
                match value {
                    KernelExpr::BinOp { op, lhs, rhs, .. } => {
                        assert_eq!(*op, BinOpKind::Add);
                        assert!(matches!(
                            lhs.as_ref(),
                            KernelExpr::Index { array: a, .. } if a == "arr"
                        ));
                        assert!(matches!(rhs.as_ref(), KernelExpr::Var(n, _) if n == "val"));
                    }
                    _ => panic!("expected BinOp"),
                }
            }
            _ => panic!("expected IndexAssign"),
        }
    }

    #[test]
    fn parse_compound_sub_assign() {
        let block = parse_block(quote! { { x -= 2; } });
        let stmts = parse_body(&block).unwrap();
        match &stmts[0] {
            KernelStmt::Assign { name, value, .. } => {
                assert_eq!(name, "x");
                assert!(matches!(
                    value,
                    KernelExpr::BinOp {
                        op: BinOpKind::Sub,
                        ..
                    }
                ));
            }
            _ => panic!("expected Assign"),
        }
    }

    #[test]
    fn reject_for_inclusive_range() {
        let block = parse_block(quote! { { for i in 0..=n {} } });
        let err = parse_body(&block).unwrap_err();
        assert!(err.to_string().contains("inclusive range"));
    }

    #[test]
    fn reject_for_iterator() {
        let block = parse_block(quote! { { for i in data {} } });
        let err = parse_body(&block).unwrap_err();
        assert!(err.to_string().contains("start..end"));
    }

    // --- Rejection tests ---

    #[test]
    fn reject_loop() {
        let expr: syn::Expr = syn::parse2(quote! { loop {} }).unwrap();
        let err = parse_expr(&expr).unwrap_err();
        assert!(err.to_string().contains("loop"));
        assert!(err.to_string().contains("use `for` or `while`"));
    }

    #[test]
    fn reject_match() {
        let expr: syn::Expr = syn::parse2(quote! { match x { _ => {} } }).unwrap();
        let err = parse_expr(&expr).unwrap_err();
        assert!(err.to_string().contains("match"));
    }

    #[test]
    fn reject_closure() {
        let expr: syn::Expr = syn::parse2(quote! { || {} }).unwrap();
        let err = parse_expr(&expr).unwrap_err();
        assert!(err.to_string().contains("closure"));
    }

    #[test]
    fn reject_method_call() {
        let expr: syn::Expr = syn::parse2(quote! { x.foo() }).unwrap();
        let err = parse_expr(&expr).unwrap_err();
        assert!(err.to_string().contains("method call"));
    }

    // --- shared_mem! parsing tests ---

    #[test]
    fn parse_shared_mem_f32() {
        let block = parse_block(quote! { { let sdata = shared_mem![f32; 256]; } });
        let stmts = parse_body(&block).unwrap();
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            KernelStmt::SharedMemDecl {
                name,
                elem_ty,
                count,
                ..
            } => {
                assert_eq!(name, "sdata");
                assert_eq!(*elem_ty, KernelType::F32);
                assert_eq!(*count, 256);
            }
            _ => panic!("expected SharedMemDecl"),
        }
    }

    #[test]
    fn parse_shared_mem_u32() {
        let block = parse_block(quote! { { let buf = shared_mem![u32; 128]; } });
        let stmts = parse_body(&block).unwrap();
        match &stmts[0] {
            KernelStmt::SharedMemDecl {
                name,
                elem_ty,
                count,
                ..
            } => {
                assert_eq!(name, "buf");
                assert_eq!(*elem_ty, KernelType::U32);
                assert_eq!(*count, 128);
            }
            _ => panic!("expected SharedMemDecl"),
        }
    }

    #[test]
    fn reject_shared_mem_bare() {
        // shared_mem! without let binding
        let expr: syn::Expr = syn::parse2(quote! { shared_mem![f32; 256] }).unwrap();
        let err = parse_expr(&expr).unwrap_err();
        assert!(err.to_string().contains("must be assigned to a variable"));
    }
}
