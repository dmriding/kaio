//! Lower arithmetic binary operations to `kaio-core` IR construction code.

use proc_macro2::{Ident, TokenStream};
use quote::quote;

use crate::kernel_ir::KernelType;
use crate::kernel_ir::expr::BinOpKind;

use super::LoweringContext;

/// Lower a single arithmetic binary operation.
///
/// `BinOpKind::Mul` always lowers to `ArithOp::Mul` (same-width), never
/// `MulWide`. `MulWide` is only used for address offset calculation in
/// Sprint 2.4's memory lowering (32-bit index → 64-bit byte offset).
///
/// Returns `(result_register_ident, token_stream_fragment)`.
#[allow(dead_code)] // Called from lower/mod.rs::lower_expr; tested here
pub fn lower_binop(
    ctx: &mut LoweringContext,
    op: &BinOpKind,
    lhs_reg: &Ident,
    rhs_reg: &Ident,
    ty: &KernelType,
) -> (Ident, TokenStream) {
    let dst = ctx.fresh_reg();
    let ptx_ty = ctx.ptx_type_tokens(ty);

    let arith_variant = match op {
        BinOpKind::Add => quote! {
            ArithOp::Add {
                dst: #dst,
                lhs: Operand::Reg(#lhs_reg),
                rhs: Operand::Reg(#rhs_reg),
                ty: PtxType::#ptx_ty,
            }
        },
        BinOpKind::Sub => quote! {
            ArithOp::Sub {
                dst: #dst,
                lhs: Operand::Reg(#lhs_reg),
                rhs: Operand::Reg(#rhs_reg),
                ty: PtxType::#ptx_ty,
            }
        },
        BinOpKind::Mul => quote! {
            ArithOp::Mul {
                dst: #dst,
                lhs: Operand::Reg(#lhs_reg),
                rhs: Operand::Reg(#rhs_reg),
                ty: PtxType::#ptx_ty,
            }
        },
        BinOpKind::Div => quote! {
            ArithOp::Div {
                dst: #dst,
                lhs: Operand::Reg(#lhs_reg),
                rhs: Operand::Reg(#rhs_reg),
                ty: PtxType::#ptx_ty,
            }
        },
        BinOpKind::Rem => quote! {
            ArithOp::Rem {
                dst: #dst,
                lhs: Operand::Reg(#lhs_reg),
                rhs: Operand::Reg(#rhs_reg),
                ty: PtxType::#ptx_ty,
            }
        },
        _ => panic!("lower_binop called with non-arithmetic op: {op:?}"),
    };

    let tokens = quote! {
        let #dst = alloc.alloc(PtxType::#ptx_ty);
        kernel.push(PtxInstruction::Arith(#arith_variant));
    };

    (dst, tokens)
}

/// Lower a unary negation to `ArithOp::Neg`.
#[allow(dead_code)] // Called from lower/mod.rs::lower_expr; tested here
pub fn lower_neg(
    ctx: &mut LoweringContext,
    src_reg: &Ident,
    ty: &KernelType,
) -> (Ident, TokenStream) {
    let dst = ctx.fresh_reg();
    let ptx_ty = ctx.ptx_type_tokens(ty);

    let tokens = quote! {
        let #dst = alloc.alloc(PtxType::#ptx_ty);
        kernel.push(PtxInstruction::Arith(ArithOp::Neg {
            dst: #dst,
            src: Operand::Reg(#src_reg),
            ty: PtxType::#ptx_ty,
        }));
    };

    (dst, tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lower_add_produces_arith_op() {
        let mut ctx = LoweringContext::new();
        // Simulate pre-existing variables
        let lhs = Ident::new("_kaio_r0", proc_macro2::Span::call_site());
        let rhs = Ident::new("_kaio_r1", proc_macro2::Span::call_site());

        let (dst, tokens) = lower_binop(&mut ctx, &BinOpKind::Add, &lhs, &rhs, &KernelType::F32);
        let code = tokens.to_string();

        assert!(code.contains("ArithOp :: Add"));
        assert!(code.contains("Operand :: Reg"));
        assert!(code.contains("PtxType :: F32"));
        assert!(dst.to_string().starts_with("_kaio_r"));
    }

    #[test]
    fn lower_mul_produces_mul_not_mulwide() {
        let mut ctx = LoweringContext::new();
        let lhs = Ident::new("_kaio_r0", proc_macro2::Span::call_site());
        let rhs = Ident::new("_kaio_r1", proc_macro2::Span::call_site());

        let (_dst, tokens) = lower_binop(&mut ctx, &BinOpKind::Mul, &lhs, &rhs, &KernelType::U32);
        let code = tokens.to_string();

        // Must produce ArithOp::Mul, NOT MulWide
        assert!(code.contains("ArithOp :: Mul"));
        assert!(!code.contains("MulWide"));
    }

    #[test]
    fn lower_neg_produces_neg_op() {
        let mut ctx = LoweringContext::new();
        let src = Ident::new("_kaio_r0", proc_macro2::Span::call_site());

        let (_dst, tokens) = lower_neg(&mut ctx, &src, &KernelType::F32);
        let code = tokens.to_string();

        assert!(code.contains("ArithOp :: Neg"));
        assert!(code.contains("PtxType :: F32"));
    }

    #[test]
    fn fresh_regs_are_unique() {
        let mut ctx = LoweringContext::new();
        let lhs = Ident::new("_kaio_r0", proc_macro2::Span::call_site());
        let rhs = Ident::new("_kaio_r1", proc_macro2::Span::call_site());

        let (r1, _) = lower_binop(&mut ctx, &BinOpKind::Add, &lhs, &rhs, &KernelType::F32);
        let (r2, _) = lower_binop(&mut ctx, &BinOpKind::Sub, &lhs, &rhs, &KernelType::F32);

        assert_ne!(r1.to_string(), r2.to_string());
    }
}
