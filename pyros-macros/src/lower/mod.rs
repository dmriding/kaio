//! Lowering pass: transform kernel IR into `TokenStream` fragments
//! that construct `pyros-core` IR at runtime.

pub mod arith;

use std::collections::HashMap;

use proc_macro2::{Ident, Span, TokenStream};
use quote::{format_ident, quote};

use crate::kernel_ir::KernelType;
use crate::kernel_ir::expr::{KernelExpr, UnaryOpKind};

/// Context threaded through all lowering functions.
#[allow(dead_code)] // Used in Sprint 2.6 codegen; tested via lower/arith.rs and lower/mod.rs tests
pub struct LoweringContext {
    /// Monotonic counter for generating unique register variable names
    /// (`_pyros_r0`, `_pyros_r1`, ...) in the generated `build_ptx()` code.
    reg_counter: u32,
    /// Counter for generating unique label names (Sprint 2.3+).
    #[allow(dead_code)] // Used in Sprint 2.3 for if/else labels
    label_counter: u32,
    /// Variable-to-register mapping.
    /// Key: variable name, Value: (register Ident in generated code, type).
    /// Populated by parameter loading (Sprint 2.6) and let-binding lowering.
    pub locals: HashMap<String, (Ident, KernelType)>,
}

#[allow(dead_code)] // Methods used in lower/arith.rs + Sprint 2.6 codegen
impl LoweringContext {
    /// Create a new lowering context.
    pub fn new() -> Self {
        Self {
            reg_counter: 0,
            label_counter: 0,
            locals: HashMap::new(),
        }
    }

    /// Allocate a fresh register variable name for the generated code.
    pub fn fresh_reg(&mut self) -> Ident {
        let id = self.reg_counter;
        self.reg_counter += 1;
        format_ident!("_pyros_r{}", id)
    }

    /// Convert a `KernelType` to the `PtxType` variant name as an `Ident`
    /// for use in generated code (e.g., `F32`, `S32`, `U64`).
    pub fn ptx_type_tokens(&self, ty: &KernelType) -> Ident {
        Ident::new(ty.ptx_type_token(), Span::call_site())
    }
}

/// Recursively lower a `KernelExpr` to a `TokenStream` that builds IR.
///
/// Returns `(register_ident, result_type, token_stream)`:
/// - `register_ident`: the Ident of the register in generated code holding the result
/// - `result_type`: the `KernelType` of the expression
/// - `token_stream`: the generated Rust code that constructs the IR
///
/// For `Var` lookups, the token stream is empty (the register already exists).
/// For everything else, the token stream contains `alloc.alloc()` + `kernel.push()` calls.
#[allow(dead_code)] // Used in Sprint 2.6 codegen; tested in this module's tests
pub fn lower_expr(
    ctx: &mut LoweringContext,
    expr: &KernelExpr,
) -> syn::Result<(Ident, KernelType, TokenStream)> {
    match expr {
        // Variable reference: look up in locals, no codegen needed
        KernelExpr::Var(name, span) => {
            let (reg, ty) = ctx.locals.get(name).cloned().ok_or_else(|| {
                syn::Error::new(*span, format!("undefined variable `{name}` in GPU kernel"))
            })?;
            Ok((reg, ty, TokenStream::new()))
        }

        // Integer literal: allocate register, mov immediate
        KernelExpr::LitInt(value, ty, _span) => {
            let dst = ctx.fresh_reg();
            let ptx_ty = ctx.ptx_type_tokens(ty);

            // Choose the right Operand constructor for the type
            let operand = match ty {
                KernelType::I32 => {
                    let v = *value as i32;
                    quote! { Operand::ImmI32(#v) }
                }
                KernelType::U32 => {
                    let v = *value as u32;
                    quote! { Operand::ImmU32(#v) }
                }
                KernelType::I64 => {
                    let v = *value;
                    quote! { Operand::ImmI64(#v) }
                }
                KernelType::U64 => {
                    let v = *value as u64;
                    quote! { Operand::ImmU64(#v) }
                }
                _ => {
                    return Err(syn::Error::new(
                        Span::call_site(),
                        format!("integer literal cannot have type {}", ty.display_name()),
                    ));
                }
            };

            let tokens = quote! {
                let #dst = alloc.alloc(PtxType::#ptx_ty);
                kernel.push(PtxInstruction::Mov {
                    dst: #dst,
                    src: #operand,
                    ty: PtxType::#ptx_ty,
                });
            };
            Ok((dst, ty.clone(), tokens))
        }

        // Float literal: allocate register, mov immediate
        KernelExpr::LitFloat(value, ty, _span) => {
            let dst = ctx.fresh_reg();
            let ptx_ty = ctx.ptx_type_tokens(ty);

            let operand = match ty {
                KernelType::F32 => {
                    let v = *value as f32;
                    quote! { Operand::ImmF32(#v) }
                }
                KernelType::F64 => {
                    let v = *value;
                    quote! { Operand::ImmF64(#v) }
                }
                _ => {
                    return Err(syn::Error::new(
                        Span::call_site(),
                        format!("float literal cannot have type {}", ty.display_name()),
                    ));
                }
            };

            let tokens = quote! {
                let #dst = alloc.alloc(PtxType::#ptx_ty);
                kernel.push(PtxInstruction::Mov {
                    dst: #dst,
                    src: #operand,
                    ty: PtxType::#ptx_ty,
                });
            };
            Ok((dst, ty.clone(), tokens))
        }

        // Bool literal
        KernelExpr::LitBool(_value, span) => Err(syn::Error::new(
            *span,
            "boolean literals in expressions are not yet supported (use comparisons)",
        )),

        // Binary operation: lower both sides recursively, then lower the op
        KernelExpr::BinOp {
            op, lhs, rhs, span, ..
        } => {
            if op.is_arithmetic() {
                let (lhs_reg, lhs_ty, lhs_tokens) = lower_expr(ctx, lhs)?;
                let (rhs_reg, _rhs_ty, rhs_tokens) = lower_expr(ctx, rhs)?;
                // TODO (Sprint 2.7): type-check that lhs_ty == rhs_ty
                let (dst, op_tokens) = arith::lower_binop(ctx, op, &lhs_reg, &rhs_reg, &lhs_ty);
                let combined = quote! { #lhs_tokens #rhs_tokens #op_tokens };
                Ok((dst, lhs_ty, combined))
            } else if op.is_comparison() {
                Err(syn::Error::new(
                    *span,
                    "comparison lowering not yet implemented (Sprint 2.3)",
                ))
            } else {
                Err(syn::Error::new(
                    *span,
                    format!("operator {op:?} lowering not yet implemented"),
                ))
            }
        }

        // Unary negation
        KernelExpr::UnaryOp { op, expr, span } => match op {
            UnaryOpKind::Neg => {
                let (src_reg, src_ty, src_tokens) = lower_expr(ctx, expr)?;
                let (dst, neg_tokens) = arith::lower_neg(ctx, &src_reg, &src_ty);
                let combined = quote! { #src_tokens #neg_tokens };
                Ok((dst, src_ty, combined))
            }
            UnaryOpKind::Not => Err(syn::Error::new(
                *span,
                "logical not (`!`) lowering not yet implemented",
            )),
        },

        // Parenthesized: just recurse
        KernelExpr::Paren(inner, _span) => lower_expr(ctx, inner),

        // --- Not yet implemented (Sprint 2.3-2.5) ---
        KernelExpr::Index { span, .. } => Err(syn::Error::new(
            *span,
            "array indexing lowering not yet implemented (Sprint 2.4)",
        )),
        KernelExpr::BuiltinCall { span, .. } => Err(syn::Error::new(
            *span,
            "built-in function lowering not yet implemented (Sprint 2.5)",
        )),
        KernelExpr::Cast { span, .. } => Err(syn::Error::new(
            *span,
            "type cast lowering not yet implemented (Sprint 2.6)",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_ir::expr::BinOpKind;

    #[test]
    fn lower_var_lookup() {
        let mut ctx = LoweringContext::new();
        let reg = Ident::new("_pyros_r5", Span::call_site());
        ctx.locals
            .insert("x".to_string(), (reg.clone(), KernelType::F32));

        let expr = KernelExpr::Var("x".to_string(), Span::call_site());
        let (result_reg, result_ty, tokens) = lower_expr(&mut ctx, &expr).unwrap();

        assert_eq!(result_reg.to_string(), "_pyros_r5");
        assert_eq!(result_ty, KernelType::F32);
        assert!(tokens.is_empty()); // No codegen for var lookup
    }

    #[test]
    fn lower_var_undefined() {
        let mut ctx = LoweringContext::new();
        let expr = KernelExpr::Var("nonexistent".to_string(), Span::call_site());
        let err = lower_expr(&mut ctx, &expr).unwrap_err();
        assert!(err.to_string().contains("undefined variable"));
    }

    #[test]
    fn lower_int_literal() {
        let mut ctx = LoweringContext::new();
        let expr = KernelExpr::LitInt(42, KernelType::I32, Span::call_site());
        let (dst, ty, tokens) = lower_expr(&mut ctx, &expr).unwrap();

        assert_eq!(ty, KernelType::I32);
        assert!(dst.to_string().starts_with("_pyros_r"));
        let code = tokens.to_string();
        assert!(code.contains("alloc . alloc"));
        assert!(code.contains("Mov"));
        assert!(code.contains("ImmI32"));
    }

    #[test]
    fn lower_float_literal() {
        let mut ctx = LoweringContext::new();
        let expr = KernelExpr::LitFloat(1.0, KernelType::F32, Span::call_site());
        let (_dst, ty, tokens) = lower_expr(&mut ctx, &expr).unwrap();

        assert_eq!(ty, KernelType::F32);
        let code = tokens.to_string();
        assert!(code.contains("ImmF32"));
    }

    #[test]
    fn lower_binop_add() {
        let mut ctx = LoweringContext::new();
        ctx.locals.insert(
            "a".to_string(),
            (Ident::new("_pyros_r0", Span::call_site()), KernelType::F32),
        );
        ctx.locals.insert(
            "b".to_string(),
            (Ident::new("_pyros_r1", Span::call_site()), KernelType::F32),
        );

        let expr = KernelExpr::BinOp {
            op: BinOpKind::Add,
            lhs: Box::new(KernelExpr::Var("a".to_string(), Span::call_site())),
            rhs: Box::new(KernelExpr::Var("b".to_string(), Span::call_site())),
            span: Span::call_site(),
        };
        let (dst, ty, tokens) = lower_expr(&mut ctx, &expr).unwrap();

        assert_eq!(ty, KernelType::F32);
        assert!(dst.to_string().starts_with("_pyros_r"));
        let code = tokens.to_string();
        assert!(code.contains("ArithOp :: Add"));
    }

    #[test]
    fn lower_nested_a_plus_b_times_c() {
        let mut ctx = LoweringContext::new();
        ctx.locals.insert(
            "a".to_string(),
            (Ident::new("_pyros_r0", Span::call_site()), KernelType::F32),
        );
        ctx.locals.insert(
            "b".to_string(),
            (Ident::new("_pyros_r1", Span::call_site()), KernelType::F32),
        );
        ctx.locals.insert(
            "c".to_string(),
            (Ident::new("_pyros_r2", Span::call_site()), KernelType::F32),
        );

        // a + b * c -> Add(a, Mul(b, c))
        let expr = KernelExpr::BinOp {
            op: BinOpKind::Add,
            lhs: Box::new(KernelExpr::Var("a".to_string(), Span::call_site())),
            rhs: Box::new(KernelExpr::BinOp {
                op: BinOpKind::Mul,
                lhs: Box::new(KernelExpr::Var("b".to_string(), Span::call_site())),
                rhs: Box::new(KernelExpr::Var("c".to_string(), Span::call_site())),
                span: Span::call_site(),
            }),
            span: Span::call_site(),
        };
        let (_dst, ty, tokens) = lower_expr(&mut ctx, &expr).unwrap();

        assert_eq!(ty, KernelType::F32);
        let code = tokens.to_string();
        // Mul should appear before Add (evaluation order: inner first)
        let mul_pos = code.find("ArithOp :: Mul").expect("should contain Mul");
        let add_pos = code.find("ArithOp :: Add").expect("should contain Add");
        assert!(
            mul_pos < add_pos,
            "Mul should be emitted before Add in evaluation order"
        );
    }

    #[test]
    fn lower_unary_neg() {
        let mut ctx = LoweringContext::new();
        ctx.locals.insert(
            "x".to_string(),
            (Ident::new("_pyros_r0", Span::call_site()), KernelType::F32),
        );

        let expr = KernelExpr::UnaryOp {
            op: UnaryOpKind::Neg,
            expr: Box::new(KernelExpr::Var("x".to_string(), Span::call_site())),
            span: Span::call_site(),
        };
        let (_dst, ty, tokens) = lower_expr(&mut ctx, &expr).unwrap();

        assert_eq!(ty, KernelType::F32);
        let code = tokens.to_string();
        assert!(code.contains("ArithOp :: Neg"));
    }

    #[test]
    fn lower_paren_recurses() {
        let mut ctx = LoweringContext::new();
        ctx.locals.insert(
            "x".to_string(),
            (Ident::new("_pyros_r0", Span::call_site()), KernelType::F32),
        );

        let expr = KernelExpr::Paren(
            Box::new(KernelExpr::Var("x".to_string(), Span::call_site())),
            Span::call_site(),
        );
        let (reg, ty, tokens) = lower_expr(&mut ctx, &expr).unwrap();

        assert_eq!(reg.to_string(), "_pyros_r0");
        assert_eq!(ty, KernelType::F32);
        assert!(tokens.is_empty());
    }
}
