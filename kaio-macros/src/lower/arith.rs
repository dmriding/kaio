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

/// Lower a bitwise binary op (`&`, `|`, `^`, `<<`, `>>`) to the matching
/// `ArithOp` variant.
///
/// Signed vs unsigned `Shr` is driven by `ty` — `I32`/`I64` → `shr.s{size}`,
/// `U32`/`U64` → `shr.u{size}`. This is the macro-level half of the AD2
/// canary; paired with the IR-level `emit_shr_s32_arithmetic` /
/// `emit_shr_u32_logical` unit tests and the GPU round-trip in D5.
///
/// Returns `(result_register_ident, token_stream_fragment)`.
#[allow(dead_code)] // Called from lower/mod.rs::lower_expr; tested here
pub fn lower_bitop(
    ctx: &mut LoweringContext,
    op: &BinOpKind,
    lhs_reg: &Ident,
    rhs_reg: &Ident,
    ty: &KernelType,
) -> (Ident, TokenStream) {
    let dst = ctx.fresh_reg();
    let ptx_ty = ctx.ptx_type_tokens(ty);

    let arith_variant = match op {
        BinOpKind::BitAnd => quote! {
            ArithOp::And {
                dst: #dst,
                lhs: Operand::Reg(#lhs_reg),
                rhs: Operand::Reg(#rhs_reg),
                ty: PtxType::#ptx_ty,
            }
        },
        BinOpKind::BitOr => quote! {
            ArithOp::Or {
                dst: #dst,
                lhs: Operand::Reg(#lhs_reg),
                rhs: Operand::Reg(#rhs_reg),
                ty: PtxType::#ptx_ty,
            }
        },
        BinOpKind::BitXor => quote! {
            ArithOp::Xor {
                dst: #dst,
                lhs: Operand::Reg(#lhs_reg),
                rhs: Operand::Reg(#rhs_reg),
                ty: PtxType::#ptx_ty,
            }
        },
        BinOpKind::Shl => quote! {
            ArithOp::Shl {
                dst: #dst,
                lhs: Operand::Reg(#lhs_reg),
                rhs: Operand::Reg(#rhs_reg),
                ty: PtxType::#ptx_ty,
            }
        },
        BinOpKind::Shr => quote! {
            ArithOp::Shr {
                dst: #dst,
                lhs: Operand::Reg(#lhs_reg),
                rhs: Operand::Reg(#rhs_reg),
                ty: PtxType::#ptx_ty,
            }
        },
        _ => panic!("lower_bitop called with non-bitwise op: {op:?}"),
    };

    let tokens = quote! {
        let #dst = alloc.alloc(PtxType::#ptx_ty);
        kernel.push(PtxInstruction::Arith(#arith_variant));
    };

    (dst, tokens)
}

/// Lower a unary NOT to either bitwise `not.b{size}` (integer) or logical
/// `not.pred` (bool). Context dispatch — the macro caller decides based on
/// `src_ty`.
#[allow(dead_code)] // Called from lower/mod.rs::lower_expr; tested here
pub fn lower_not(
    ctx: &mut LoweringContext,
    src_reg: &Ident,
    ty: &KernelType,
) -> (Ident, TokenStream) {
    let dst = ctx.fresh_reg();
    let ptx_ty = ctx.ptx_type_tokens(ty);

    let tokens = quote! {
        let #dst = alloc.alloc(PtxType::#ptx_ty);
        kernel.push(PtxInstruction::Arith(ArithOp::Not {
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
    fn lower_bitop_and_produces_and_variant() {
        let mut ctx = LoweringContext::new();
        let lhs = Ident::new("_kaio_r0", proc_macro2::Span::call_site());
        let rhs = Ident::new("_kaio_r1", proc_macro2::Span::call_site());

        let (_dst, tokens) =
            lower_bitop(&mut ctx, &BinOpKind::BitAnd, &lhs, &rhs, &KernelType::U32);
        let code = tokens.to_string();
        assert!(code.contains("ArithOp :: And"));
        assert!(code.contains("PtxType :: U32"));
    }

    #[test]
    fn lower_bitop_shr_preserves_signedness_u32() {
        // AD2 canary (macro-level): u32 >> n must emit ArithOp::Shr { ty: U32 }
        // so PTX emits `shr.u32` (logical shift).
        let mut ctx = LoweringContext::new();
        let lhs = Ident::new("_kaio_r0", proc_macro2::Span::call_site());
        let rhs = Ident::new("_kaio_r1", proc_macro2::Span::call_site());

        let (_dst, tokens) = lower_bitop(&mut ctx, &BinOpKind::Shr, &lhs, &rhs, &KernelType::U32);
        let code = tokens.to_string();
        assert!(code.contains("ArithOp :: Shr"));
        assert!(
            code.contains("PtxType :: U32"),
            "u32 >> n must carry U32 through to ArithOp, got: {code}"
        );
    }

    #[test]
    fn lower_bitop_shr_preserves_signedness_i32() {
        // AD2 canary (macro-level): i32 >> n must emit ArithOp::Shr { ty: S32 }
        // so PTX emits `shr.s32` (arithmetic shift, sign-extends). If this test
        // ever emits PtxType::U32, quant INT8 dequant on negative packed values
        // will silently zero-extend and produce wrong weights.
        let mut ctx = LoweringContext::new();
        let lhs = Ident::new("_kaio_r0", proc_macro2::Span::call_site());
        let rhs = Ident::new("_kaio_r1", proc_macro2::Span::call_site());

        let (_dst, tokens) = lower_bitop(&mut ctx, &BinOpKind::Shr, &lhs, &rhs, &KernelType::I32);
        let code = tokens.to_string();
        assert!(code.contains("ArithOp :: Shr"));
        assert!(
            code.contains("PtxType :: S32"),
            "i32 >> n must carry S32 through to ArithOp (arithmetic shift), got: {code}"
        );
    }

    #[test]
    fn lower_bitop_shl_typeless_on_signedness() {
        // AD2: shl is typeless — both I32 and U32 must produce valid PTX.
        // The IR carries whichever type was passed; the emit layer collapses
        // to `.b32` via reg_decl_type. Test the IR-level round-trip.
        let mut ctx = LoweringContext::new();
        let lhs = Ident::new("_kaio_r0", proc_macro2::Span::call_site());
        let rhs = Ident::new("_kaio_r1", proc_macro2::Span::call_site());

        let (_dst, tokens) = lower_bitop(&mut ctx, &BinOpKind::Shl, &lhs, &rhs, &KernelType::I32);
        let code = tokens.to_string();
        assert!(code.contains("ArithOp :: Shl"));
    }

    #[test]
    fn lower_not_emits_not_variant() {
        let mut ctx = LoweringContext::new();
        let src = Ident::new("_kaio_r0", proc_macro2::Span::call_site());

        let (_dst, tokens) = lower_not(&mut ctx, &src, &KernelType::U32);
        let code = tokens.to_string();
        assert!(code.contains("ArithOp :: Not"));
        assert!(code.contains("PtxType :: U32"));
    }

    #[test]
    fn lower_not_pred_for_bool() {
        // AD3: unary `!` on Bool must dispatch to ArithOp::Not { ty: Pred }
        // so PTX emits `not.pred`. If this ever emits `.b32` / `.b8`, ptxas
        // will reject it outright (loud failure) or silently misbehave.
        let mut ctx = LoweringContext::new();
        let src = Ident::new("_kaio_p0", proc_macro2::Span::call_site());

        let (_dst, tokens) = lower_not(&mut ctx, &src, &KernelType::Bool);
        let code = tokens.to_string();
        assert!(code.contains("ArithOp :: Not"));
        assert!(
            code.contains("PtxType :: Pred"),
            "!bool must dispatch to Pred type, got: {code}"
        );
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
