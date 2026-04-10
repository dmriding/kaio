//! Lowering pass: transform kernel IR into `TokenStream` fragments
//! that construct `pyros-core` IR at runtime.

pub mod arith;
pub mod compare;
pub mod memory;

use std::collections::HashMap;

use proc_macro2::{Ident, Span, TokenStream};
use quote::{format_ident, quote};

use crate::kernel_ir::KernelType;
use crate::kernel_ir::expr::{KernelExpr, UnaryOpKind};
use crate::kernel_ir::stmt::KernelStmt;

/// Context threaded through all lowering functions.
#[allow(dead_code)] // Used in Sprint 2.6 codegen; tested via lower/arith.rs and lower/mod.rs tests
pub struct LoweringContext {
    /// Monotonic counter for generating unique register variable names
    /// (`_pyros_r0`, `_pyros_r1`, ...) in the generated `build_ptx()` code.
    reg_counter: u32,
    /// Counter for generating unique label names (`IF_END_0`, `IF_ELSE_1`, ...).
    label_counter: u32,
    /// Variable-to-register mapping.
    /// Key: variable name, Value: (register Ident in generated code, type).
    /// Populated by parameter loading (Sprint 2.6) and let-binding lowering.
    pub locals: HashMap<String, (Ident, KernelType)>,
    /// Cached `cvta.to.global` results per pointer parameter.
    /// Key: param name, Value: register Ident holding the global address.
    /// One CvtaToGlobal per pointer, reused across multiple index accesses.
    pub global_addrs: HashMap<String, Ident>,
}

#[allow(dead_code)] // Methods used in lower/arith.rs + Sprint 2.6 codegen
impl LoweringContext {
    /// Create a new lowering context.
    pub fn new() -> Self {
        Self {
            reg_counter: 0,
            label_counter: 0,
            locals: HashMap::new(),
            global_addrs: HashMap::new(),
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

    /// Generate a unique label name (e.g., `"IF_END_0"`, `"IF_ELSE_3"`).
    pub fn fresh_label(&mut self, prefix: &str) -> String {
        let id = self.label_counter;
        self.label_counter += 1;
        format!("{prefix}_{id}")
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
                let (lhs_reg, lhs_ty, lhs_tokens) = lower_expr(ctx, lhs)?;
                let (rhs_reg, _rhs_ty, rhs_tokens) = lower_expr(ctx, rhs)?;
                let (pred, cmp_tokens) =
                    compare::lower_comparison(ctx, op, &lhs_reg, &rhs_reg, &lhs_ty);
                let combined = quote! { #lhs_tokens #rhs_tokens #cmp_tokens };
                Ok((pred, KernelType::Bool, combined))
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

        // Array index read: a[idx]
        KernelExpr::Index { array, index, span } => {
            let (array_reg, array_ty) = ctx.locals.get(array).cloned().ok_or_else(|| {
                syn::Error::new(*span, format!("undefined array `{array}` in GPU kernel"))
            })?;
            let elem_ty = array_ty.elem_type().cloned().ok_or_else(|| {
                syn::Error::new(
                    *span,
                    format!(
                        "cannot index into `{array}`: type `{}` is not a slice",
                        array_ty.display_name()
                    ),
                )
            })?;
            let (idx_reg, _idx_ty, idx_tokens) = lower_expr(ctx, index)?;
            let (result, mem_tokens) =
                memory::lower_index_read(ctx, array, &array_reg, &idx_reg, &elem_ty);
            Ok((result, elem_ty, quote! { #idx_tokens #mem_tokens }))
        }
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

/// Lower a sequence of kernel statements to a combined `TokenStream`.
#[allow(dead_code)] // Used in Sprint 2.6 codegen
pub fn lower_stmts(ctx: &mut LoweringContext, stmts: &[KernelStmt]) -> syn::Result<TokenStream> {
    let mut combined = TokenStream::new();
    for stmt in stmts {
        let tokens = lower_stmt(ctx, stmt)?;
        combined.extend(tokens);
    }
    Ok(combined)
}

/// Lower a single kernel statement to a `TokenStream`.
#[allow(dead_code)] // Used in Sprint 2.6 codegen; tested here
pub fn lower_stmt(ctx: &mut LoweringContext, stmt: &KernelStmt) -> syn::Result<TokenStream> {
    match stmt {
        // let x = expr; — lower value, register in locals
        KernelStmt::Let {
            name, value, span, ..
        } => {
            let (reg, ty, expr_tokens) = lower_expr(ctx, value)?;
            if ctx.locals.contains_key(name) {
                return Err(syn::Error::new(
                    *span,
                    format!("variable `{name}` already defined in this kernel"),
                ));
            }
            ctx.locals.insert(name.clone(), (reg, ty));
            Ok(expr_tokens)
        }

        // if cond { then } [else { otherwise }]
        KernelStmt::If {
            condition,
            then_body,
            else_body,
            ..
        } => {
            // 1. Lower condition to predicate register
            let (pred_reg, _pred_ty, cond_tokens) = lower_expr(ctx, condition)?;

            // 2. Generate labels
            let has_else = else_body.is_some();
            let end_label = ctx.fresh_label("IF_END");
            let else_label = if has_else {
                Some(ctx.fresh_label("IF_ELSE"))
            } else {
                None
            };

            // 3. Branch: @!pred bra target (skip then-block when condition is false)
            let skip_target = else_label.as_deref().unwrap_or(&end_label);
            let skip_target_str = skip_target.to_string();
            let branch_tokens = quote! {
                kernel.push(PtxInstruction::Control(ControlOp::BraPred {
                    pred: #pred_reg,
                    target: #skip_target_str.to_string(),
                    negate: true,
                }));
            };

            // 4. Lower then-body
            let then_tokens = lower_stmts(ctx, then_body)?;

            // 5. If else: unconditional branch past else, else label, else body
            let else_tokens = if let Some(else_stmts) = else_body {
                let else_lbl = else_label.as_ref().unwrap();
                let end_lbl_str = end_label.clone();
                let else_body_tokens = lower_stmts(ctx, else_stmts)?;
                quote! {
                    kernel.push(PtxInstruction::Control(ControlOp::Bra {
                        target: #end_lbl_str.to_string(),
                    }));
                    kernel.push(PtxInstruction::Label(#else_lbl.to_string()));
                    #else_body_tokens
                }
            } else {
                TokenStream::new()
            };

            // 6. End label
            let end_label_tokens = quote! {
                kernel.push(PtxInstruction::Label(#end_label.to_string()));
            };

            Ok(quote! {
                #cond_tokens
                #branch_tokens
                #then_tokens
                #else_tokens
                #end_label_tokens
            })
        }

        // Bare expression statement
        KernelStmt::Expr(expr, _span) => {
            let (_reg, _ty, tokens) = lower_expr(ctx, expr)?;
            Ok(tokens)
        }

        // --- Not yet implemented (Sprint 2.4+) ---
        KernelStmt::Assign { span, .. } => Err(syn::Error::new(
            *span,
            "variable reassignment lowering not yet implemented",
        )),
        KernelStmt::IndexAssign {
            array,
            index,
            value,
            span,
        } => {
            let (array_reg, array_ty) = ctx.locals.get(array).cloned().ok_or_else(|| {
                syn::Error::new(*span, format!("undefined array `{array}` in GPU kernel"))
            })?;
            // Must be &mut [T] for writes
            if !array_ty.is_mut_slice() {
                return Err(syn::Error::new(
                    *span,
                    format!(
                        "cannot write to immutable slice parameter `{array}`: \
                         declare as `&mut [T]`"
                    ),
                ));
            }
            let elem_ty = array_ty.elem_type().cloned().ok_or_else(|| {
                syn::Error::new(*span, "internal error: mut slice has no element type")
            })?;
            let (idx_reg, _idx_ty, idx_tokens) = lower_expr(ctx, index)?;
            let (val_reg, _val_ty, val_tokens) = lower_expr(ctx, value)?;
            let store_tokens =
                memory::lower_index_write(ctx, array, &array_reg, &idx_reg, &val_reg, &elem_ty);
            Ok(quote! { #idx_tokens #val_tokens #store_tokens })
        }
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

    // --- Sprint 2.3: Comparisons + If/Else ---

    #[test]
    fn lower_comparison_in_expr() {
        let mut ctx = LoweringContext::new();
        ctx.locals.insert(
            "x".to_string(),
            (Ident::new("_pyros_r0", Span::call_site()), KernelType::U32),
        );
        ctx.locals.insert(
            "n".to_string(),
            (Ident::new("_pyros_r1", Span::call_site()), KernelType::U32),
        );

        let expr = KernelExpr::BinOp {
            op: BinOpKind::Lt,
            lhs: Box::new(KernelExpr::Var("x".to_string(), Span::call_site())),
            rhs: Box::new(KernelExpr::Var("n".to_string(), Span::call_site())),
            span: Span::call_site(),
        };
        let (pred, ty, tokens) = lower_expr(&mut ctx, &expr).unwrap();

        assert_eq!(ty, KernelType::Bool);
        assert!(pred.to_string().starts_with("_pyros_r"));
        let code = tokens.to_string();
        assert!(code.contains("ControlOp :: SetP"));
        assert!(code.contains("CmpOp :: Lt"));
    }

    #[test]
    fn lower_let_registers_local() {
        let mut ctx = LoweringContext::new();
        ctx.locals.insert(
            "a".to_string(),
            (Ident::new("_pyros_r0", Span::call_site()), KernelType::F32),
        );
        ctx.locals.insert(
            "b".to_string(),
            (Ident::new("_pyros_r1", Span::call_site()), KernelType::F32),
        );

        let stmt = KernelStmt::Let {
            name: "x".to_string(),
            ty: None,
            value: KernelExpr::BinOp {
                op: BinOpKind::Add,
                lhs: Box::new(KernelExpr::Var("a".to_string(), Span::call_site())),
                rhs: Box::new(KernelExpr::Var("b".to_string(), Span::call_site())),
                span: Span::call_site(),
            },
            span: Span::call_site(),
        };
        let tokens = lower_stmt(&mut ctx, &stmt).unwrap();

        // "x" should now be in locals
        assert!(ctx.locals.contains_key("x"));
        let (reg, ty) = &ctx.locals["x"];
        assert_eq!(ty, &KernelType::F32);
        assert!(reg.to_string().starts_with("_pyros_r"));

        // Should have generated ArithOp::Add
        let code = tokens.to_string();
        assert!(code.contains("ArithOp :: Add"));
    }

    #[test]
    fn lower_if_simple() {
        let mut ctx = LoweringContext::new();
        ctx.locals.insert(
            "idx".to_string(),
            (Ident::new("_pyros_r0", Span::call_site()), KernelType::U32),
        );
        ctx.locals.insert(
            "n".to_string(),
            (Ident::new("_pyros_r1", Span::call_site()), KernelType::U32),
        );

        // if idx < n { (bare expression for now) }
        let stmt = KernelStmt::If {
            condition: KernelExpr::BinOp {
                op: BinOpKind::Lt,
                lhs: Box::new(KernelExpr::Var("idx".to_string(), Span::call_site())),
                rhs: Box::new(KernelExpr::Var("n".to_string(), Span::call_site())),
                span: Span::call_site(),
            },
            then_body: vec![],
            else_body: None,
            span: Span::call_site(),
        };
        let tokens = lower_stmt(&mut ctx, &stmt).unwrap();
        let code = tokens.to_string();

        // Should contain: SetP, BraPred with negate: true, Label
        assert!(code.contains("SetP"));
        assert!(code.contains("negate : true"));
        assert!(code.contains("IF_END_0"));
        assert!(code.contains("Label"));
    }

    #[test]
    fn lower_if_else() {
        let mut ctx = LoweringContext::new();
        ctx.locals.insert(
            "x".to_string(),
            (Ident::new("_pyros_r0", Span::call_site()), KernelType::U32),
        );
        ctx.locals.insert(
            "n".to_string(),
            (Ident::new("_pyros_r1", Span::call_site()), KernelType::U32),
        );

        let stmt = KernelStmt::If {
            condition: KernelExpr::BinOp {
                op: BinOpKind::Lt,
                lhs: Box::new(KernelExpr::Var("x".to_string(), Span::call_site())),
                rhs: Box::new(KernelExpr::Var("n".to_string(), Span::call_site())),
                span: Span::call_site(),
            },
            then_body: vec![],
            else_body: Some(vec![]),
            span: Span::call_site(),
        };
        let tokens = lower_stmt(&mut ctx, &stmt).unwrap();
        let code = tokens.to_string();

        // If/else should have: BraPred -> IF_ELSE, Bra -> IF_END, IF_ELSE label, IF_END label
        // Label allocation order: IF_END first (0), IF_ELSE second (1)
        assert!(code.contains("negate : true"));
        assert!(code.contains("IF_ELSE_1"));
        assert!(code.contains("IF_END_0"));
        // Unconditional branch to skip else
        assert!(code.contains("ControlOp :: Bra"));
    }

    #[test]
    fn fresh_labels_are_unique() {
        let mut ctx = LoweringContext::new();
        let l1 = ctx.fresh_label("IF_END");
        let l2 = ctx.fresh_label("IF_ELSE");
        let l3 = ctx.fresh_label("IF_END");
        assert_eq!(l1, "IF_END_0");
        assert_eq!(l2, "IF_ELSE_1");
        assert_eq!(l3, "IF_END_2");
    }

    // --- Sprint 2.4: Array Indexing ---

    #[test]
    fn lower_expr_index_read() {
        let mut ctx = LoweringContext::new();
        // Simulate a pointer param "a" loaded as SliceRef(F32)
        ctx.locals.insert(
            "a".to_string(),
            (
                Ident::new("_pyros_r0", Span::call_site()),
                KernelType::SliceRef(Box::new(KernelType::F32)),
            ),
        );
        ctx.locals.insert(
            "idx".to_string(),
            (Ident::new("_pyros_r1", Span::call_site()), KernelType::U32),
        );

        let expr = KernelExpr::Index {
            array: "a".to_string(),
            index: Box::new(KernelExpr::Var("idx".to_string(), Span::call_site())),
            span: Span::call_site(),
        };
        let (result, ty, tokens) = lower_expr(&mut ctx, &expr).unwrap();

        assert_eq!(ty, KernelType::F32); // result is the element type
        assert!(result.to_string().starts_with("_pyros_r"));
        let code = tokens.to_string();
        assert!(code.contains("CvtaToGlobal"));
        assert!(code.contains("MulWide"));
        assert!(code.contains("LdGlobal"));
    }

    #[test]
    fn lower_stmt_index_assign() {
        let mut ctx = LoweringContext::new();
        ctx.locals.insert(
            "out".to_string(),
            (
                Ident::new("_pyros_r0", Span::call_site()),
                KernelType::SliceMutRef(Box::new(KernelType::F32)),
            ),
        );
        ctx.locals.insert(
            "idx".to_string(),
            (Ident::new("_pyros_r1", Span::call_site()), KernelType::U32),
        );
        ctx.locals.insert(
            "val".to_string(),
            (Ident::new("_pyros_r2", Span::call_site()), KernelType::F32),
        );

        let stmt = KernelStmt::IndexAssign {
            array: "out".to_string(),
            index: KernelExpr::Var("idx".to_string(), Span::call_site()),
            value: KernelExpr::Var("val".to_string(), Span::call_site()),
            span: Span::call_site(),
        };
        let tokens = lower_stmt(&mut ctx, &stmt).unwrap();
        let code = tokens.to_string();

        assert!(code.contains("CvtaToGlobal"));
        assert!(code.contains("StGlobal"));
        assert!(code.contains("PtxType :: F32"));
    }

    #[test]
    fn reject_write_to_immutable_slice() {
        let mut ctx = LoweringContext::new();
        ctx.locals.insert(
            "a".to_string(),
            (
                Ident::new("_pyros_r0", Span::call_site()),
                KernelType::SliceRef(Box::new(KernelType::F32)), // &[f32], NOT &mut
            ),
        );
        ctx.locals.insert(
            "idx".to_string(),
            (Ident::new("_pyros_r1", Span::call_site()), KernelType::U32),
        );
        ctx.locals.insert(
            "val".to_string(),
            (Ident::new("_pyros_r2", Span::call_site()), KernelType::F32),
        );

        let stmt = KernelStmt::IndexAssign {
            array: "a".to_string(),
            index: KernelExpr::Var("idx".to_string(), Span::call_site()),
            value: KernelExpr::Var("val".to_string(), Span::call_site()),
            span: Span::call_site(),
        };
        let err = lower_stmt(&mut ctx, &stmt).unwrap_err();
        assert!(err.to_string().contains("immutable slice"));
    }

    #[test]
    fn reject_index_into_scalar() {
        let mut ctx = LoweringContext::new();
        ctx.locals.insert(
            "n".to_string(),
            (Ident::new("_pyros_r0", Span::call_site()), KernelType::U32),
        );
        ctx.locals.insert(
            "idx".to_string(),
            (Ident::new("_pyros_r1", Span::call_site()), KernelType::U32),
        );

        let expr = KernelExpr::Index {
            array: "n".to_string(),
            index: Box::new(KernelExpr::Var("idx".to_string(), Span::call_site())),
            span: Span::call_site(),
        };
        let err = lower_expr(&mut ctx, &expr).unwrap_err();
        assert!(err.to_string().contains("not a slice"));
    }
}
