//! Sprint 7.0 D4: lowering for short-circuit logical operators `&&` and `||`.
//!
//! Two codepaths share one builder:
//!
//! - **Expression position** (`let mask = a && b;`) — [`lower_logical_expr`]
//!   materializes the short-circuit result into a fresh `.pred` register so
//!   the predicate can flow to downstream consumers (stored, compared,
//!   passed as an `if` condition after being bound to a name, etc.).
//!
//! - **If-condition position** (`if a && b { body }`) — [`lower_logical_if`]
//!   emits a branch-direct form: no intermediate `p_out` register; the
//!   short-circuit target jumps straight to the caller-supplied `skip_label`.
//!   This is an optimization over the expression-position path; correctness
//!   is identical.
//!
//! Both paths preserve Rust's left-to-right, short-circuit semantics: if
//! `a` determines the result (`a == false` for `&&`, `a == true` for `||`),
//! `b` is never evaluated. This is why `if i < n && arr[i] > 0` is safe as
//! a bounds-guarded access — the `arr[i]` read only fires when `i < n`.
//!
//! See [`crate::kernel_ir::expr::BinOpKind::is_logical`] and the D4 GPU
//! tests in `kaio/tests/logical_short_circuit_macro.rs`.

use proc_macro2::{Ident, TokenStream};
use quote::quote;

use crate::kernel_ir::KernelType;
use crate::kernel_ir::expr::{BinOpKind, KernelExpr};

use super::{LoweringContext, lower_expr};

/// Lower a logical `&&` / `||` used in expression position.
///
/// Returns `(p_out_ident, tokens)` where `p_out_ident` is a fresh `.pred`
/// register holding the short-circuit result. The tokens allocate `p_out`,
/// evaluate `lhs`, conditionally short-circuit, evaluate `rhs`, and leave
/// the final predicate in `p_out`.
///
/// The shape of the emitted sequence for `a && b`:
/// ```text
/// <evaluate lhs — sets p_lhs>
/// mov.pred p_out, p_lhs
/// @!p_lhs bra DONE_k   // short-circuit: lhs false → result is p_lhs (false)
/// <evaluate rhs — sets p_rhs>
/// mov.pred p_out, p_rhs
/// DONE_k:
/// ```
/// For `a || b`, swap the `@!p_lhs` branch for `@p_lhs` (short-circuit when
/// lhs is true).
pub fn lower_logical_expr(
    ctx: &mut LoweringContext,
    op: &BinOpKind,
    lhs: &KernelExpr,
    rhs: &KernelExpr,
) -> syn::Result<(Ident, TokenStream)> {
    debug_assert!(op.is_logical(), "lower_logical_expr called with non-logical op: {op:?}");

    let (lhs_reg, lhs_ty, lhs_tokens) = lower_expr(ctx, lhs)?;
    ensure_bool(&lhs_ty, lhs, op)?;

    let p_out = ctx.fresh_reg();
    let done_label = ctx.fresh_label("LOGICAL_DONE");
    let done_label_str = done_label.clone();

    // `&&` short-circuits on LHS false (negate bra); `||` short-circuits on LHS true.
    let negate_bra = matches!(op, BinOpKind::And);

    // Eagerly lower RHS tokens so we can decide whether it was pure from inspection.
    let (rhs_reg, rhs_ty, rhs_tokens) = lower_expr(ctx, rhs)?;
    ensure_bool(&rhs_ty, rhs, op)?;

    let tokens = quote! {
        #lhs_tokens
        let #p_out = alloc.alloc(PtxType::Pred);
        kernel.push(PtxInstruction::Mov {
            dst: #p_out,
            src: Operand::Reg(#lhs_reg),
            ty: PtxType::Pred,
        });
        kernel.push(PtxInstruction::Control(ControlOp::BraPred {
            pred: #lhs_reg,
            target: #done_label_str.to_string(),
            negate: #negate_bra,
        }));
        #rhs_tokens
        kernel.push(PtxInstruction::Mov {
            dst: #p_out,
            src: Operand::Reg(#rhs_reg),
            ty: PtxType::Pred,
        });
        kernel.push(PtxInstruction::Label(#done_label_str.to_string()));
    };

    Ok((p_out, tokens))
}

/// Lower a logical `&&` / `||` that is directly the condition of an `if`.
///
/// Branch-direct form: on short-circuit skip, branch directly to
/// `skip_label` (the else/end label of the surrounding `if`). Returns
/// tokens that, after execution, leave control flow in one of two states:
/// - falls through → if-body should execute (condition was true)
/// - jumps to `skip_label` → if-body should be skipped (condition false)
///
/// For `if a && b`:
/// ```text
/// <evaluate lhs — p_lhs>
/// @!p_lhs bra skip_label
/// <evaluate rhs — p_rhs>
/// @!p_rhs bra skip_label
/// // fall through to body
/// ```
/// For `if a || b`:
/// ```text
/// <evaluate lhs — p_lhs>
/// @p_lhs bra TAKE_k
/// <evaluate rhs — p_rhs>
/// @!p_rhs bra skip_label
/// TAKE_k:
/// // fall through to body
/// ```
pub fn lower_logical_if(
    ctx: &mut LoweringContext,
    op: &BinOpKind,
    lhs: &KernelExpr,
    rhs: &KernelExpr,
    skip_label: &str,
) -> syn::Result<TokenStream> {
    debug_assert!(op.is_logical(), "lower_logical_if called with non-logical op: {op:?}");

    let (lhs_reg, lhs_ty, lhs_tokens) = lower_expr(ctx, lhs)?;
    ensure_bool(&lhs_ty, lhs, op)?;
    let (rhs_reg, rhs_ty, rhs_tokens) = lower_expr(ctx, rhs)?;
    ensure_bool(&rhs_ty, rhs, op)?;

    let skip = skip_label.to_string();

    let tokens = match op {
        BinOpKind::And => {
            // @!p_lhs bra skip; rhs; @!p_rhs bra skip;
            quote! {
                #lhs_tokens
                kernel.push(PtxInstruction::Control(ControlOp::BraPred {
                    pred: #lhs_reg,
                    target: #skip.to_string(),
                    negate: true,
                }));
                #rhs_tokens
                kernel.push(PtxInstruction::Control(ControlOp::BraPred {
                    pred: #rhs_reg,
                    target: #skip.to_string(),
                    negate: true,
                }));
            }
        }
        BinOpKind::Or => {
            // @p_lhs bra TAKE; rhs; @!p_rhs bra skip; TAKE:
            let take_label = ctx.fresh_label("LOGICAL_OR_TAKE");
            quote! {
                #lhs_tokens
                kernel.push(PtxInstruction::Control(ControlOp::BraPred {
                    pred: #lhs_reg,
                    target: #take_label.to_string(),
                    negate: false,
                }));
                #rhs_tokens
                kernel.push(PtxInstruction::Control(ControlOp::BraPred {
                    pred: #rhs_reg,
                    target: #skip.to_string(),
                    negate: true,
                }));
                kernel.push(PtxInstruction::Label(#take_label.to_string()));
            }
        }
        _ => unreachable!("lower_logical_if guarded by is_logical()"),
    };

    Ok(tokens)
}

fn ensure_bool(ty: &KernelType, expr: &KernelExpr, op: &BinOpKind) -> syn::Result<()> {
    if *ty != KernelType::Bool {
        return Err(syn::Error::new(
            expr_span(expr),
            format!(
                "logical operator {} requires bool operands, got {}",
                op_display(op),
                ty.display_name()
            ),
        ));
    }
    Ok(())
}

fn op_display(op: &BinOpKind) -> &'static str {
    match op {
        BinOpKind::And => "&&",
        BinOpKind::Or => "||",
        _ => "<?>",
    }
}

fn expr_span(expr: &KernelExpr) -> proc_macro2::Span {
    match expr {
        KernelExpr::BinOp { span, .. }
        | KernelExpr::UnaryOp { span, .. }
        | KernelExpr::Index { span, .. }
        | KernelExpr::BuiltinCall { span, .. }
        | KernelExpr::Cast { span, .. }
        | KernelExpr::LitInt(_, _, span)
        | KernelExpr::LitFloat(_, _, span)
        | KernelExpr::LitBool(_, span)
        | KernelExpr::Var(_, span) => *span,
        KernelExpr::Paren(_, span) => *span,
    }
}
