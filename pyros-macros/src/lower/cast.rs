//! Lower type casts (`expr as Type`) to PTX `cvt` instructions.

#![allow(dead_code)]

use proc_macro2::{Ident, TokenStream};
use quote::quote;

use crate::kernel_ir::KernelType;

use super::LoweringContext;

/// Lower a type cast: `expr as target_ty` → `PtxInstruction::Cvt`.
///
/// Returns `(result_register, token_stream)`.
pub fn lower_cast(
    ctx: &mut LoweringContext,
    src_reg: &Ident,
    src_ty: &KernelType,
    target_ty: &KernelType,
) -> (Ident, TokenStream) {
    let dst = ctx.fresh_reg();
    let dst_ptx = ctx.ptx_type_tokens(target_ty);
    let src_ptx = ctx.ptx_type_tokens(src_ty);

    let tokens = quote! {
        let #dst = alloc.alloc(PtxType::#dst_ptx);
        kernel.push(PtxInstruction::Cvt {
            dst: #dst,
            src: #src_reg,
            dst_ty: PtxType::#dst_ptx,
            src_ty: PtxType::#src_ptx,
        });
    };

    (dst, tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proc_macro2::Span;

    #[test]
    fn lower_cast_f32_to_i32() {
        let mut ctx = LoweringContext::new();
        let src = Ident::new("_pyros_r0", Span::call_site());

        let (dst, tokens) = lower_cast(&mut ctx, &src, &KernelType::F32, &KernelType::I32);
        let code = tokens.to_string();

        assert!(dst.to_string().starts_with("_pyros_r"));
        assert!(code.contains("Cvt"));
        assert!(code.contains("PtxType :: S32")); // dst_ty
        assert!(code.contains("PtxType :: F32")); // src_ty
    }

    #[test]
    fn lower_cast_u32_to_f32() {
        let mut ctx = LoweringContext::new();
        let src = Ident::new("_pyros_r0", Span::call_site());

        let (_dst, tokens) = lower_cast(&mut ctx, &src, &KernelType::U32, &KernelType::F32);
        let code = tokens.to_string();

        assert!(code.contains("PtxType :: F32")); // dst_ty
        assert!(code.contains("PtxType :: U32")); // src_ty
    }
}
