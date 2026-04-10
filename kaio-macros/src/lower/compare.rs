//! Lower comparison operations to `kaio-core` IR construction code.

use proc_macro2::{Ident, TokenStream};
use quote::quote;

use crate::kernel_ir::KernelType;
use crate::kernel_ir::expr::BinOpKind;

use super::LoweringContext;

/// Lower a comparison binary operation to `ControlOp::SetP`.
///
/// Returns `(predicate_register_ident, token_stream_fragment)`.
/// The result is always a predicate register (`PtxType::Pred`).
#[allow(dead_code)] // Called from lower/mod.rs::lower_expr
pub fn lower_comparison(
    ctx: &mut LoweringContext,
    op: &BinOpKind,
    lhs_reg: &Ident,
    rhs_reg: &Ident,
    operand_ty: &KernelType,
) -> (Ident, TokenStream) {
    let dst = ctx.fresh_reg();
    let ptx_ty = ctx.ptx_type_tokens(operand_ty);

    let cmp_op = match op {
        BinOpKind::Lt => quote! { CmpOp::Lt },
        BinOpKind::Le => quote! { CmpOp::Le },
        BinOpKind::Gt => quote! { CmpOp::Gt },
        BinOpKind::Ge => quote! { CmpOp::Ge },
        BinOpKind::Eq => quote! { CmpOp::Eq },
        BinOpKind::Ne => quote! { CmpOp::Ne },
        _ => panic!("lower_comparison called with non-comparison op: {op:?}"),
    };

    let tokens = quote! {
        let #dst = alloc.alloc(PtxType::Pred);
        kernel.push(PtxInstruction::Control(ControlOp::SetP {
            dst: #dst,
            cmp_op: #cmp_op,
            lhs: Operand::Reg(#lhs_reg),
            rhs: Operand::Reg(#rhs_reg),
            ty: PtxType::#ptx_ty,
        }));
    };

    (dst, tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proc_macro2::Span;

    #[test]
    fn lower_comparison_lt() {
        let mut ctx = LoweringContext::new();
        let lhs = Ident::new("_kaio_r0", Span::call_site());
        let rhs = Ident::new("_kaio_r1", Span::call_site());

        let (pred, tokens) =
            lower_comparison(&mut ctx, &BinOpKind::Lt, &lhs, &rhs, &KernelType::U32);
        let code = tokens.to_string();

        assert!(pred.to_string().starts_with("_kaio_r"));
        assert!(code.contains("ControlOp :: SetP"));
        assert!(code.contains("CmpOp :: Lt"));
        assert!(code.contains("PtxType :: Pred"));
        assert!(code.contains("PtxType :: U32"));
    }

    #[test]
    fn lower_comparison_ge() {
        let mut ctx = LoweringContext::new();
        let lhs = Ident::new("_kaio_r0", Span::call_site());
        let rhs = Ident::new("_kaio_r1", Span::call_site());

        let (_pred, tokens) =
            lower_comparison(&mut ctx, &BinOpKind::Ge, &lhs, &rhs, &KernelType::F32);
        let code = tokens.to_string();

        assert!(code.contains("CmpOp :: Ge"));
        assert!(code.contains("PtxType :: F32"));
    }

    #[test]
    fn lower_all_comparison_ops() {
        let ops = [
            (BinOpKind::Lt, "Lt"),
            (BinOpKind::Le, "Le"),
            (BinOpKind::Gt, "Gt"),
            (BinOpKind::Ge, "Ge"),
            (BinOpKind::Eq, "Eq"),
            (BinOpKind::Ne, "Ne"),
        ];
        for (op, expected_str) in ops {
            let mut ctx = LoweringContext::new();
            let lhs = Ident::new("_kaio_r0", Span::call_site());
            let rhs = Ident::new("_kaio_r1", Span::call_site());
            let (_pred, tokens) = lower_comparison(&mut ctx, &op, &lhs, &rhs, &KernelType::U32);
            let code = tokens.to_string();
            assert!(
                code.contains(&format!("CmpOp :: {expected_str}")),
                "expected CmpOp::{expected_str} for {op:?}"
            );
        }
    }
}
