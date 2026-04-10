//! Lower kernel parameters to PTX param declarations + ld.param instructions.

#![allow(dead_code)]

use proc_macro2::TokenStream;
use quote::quote;

use crate::kernel_ir::{KernelParam, KernelType};

use super::LoweringContext;

/// Lower all kernel parameters: generate `add_param` + `ld.param` + register in locals.
///
/// Returns a `TokenStream` that declares and loads all parameters.
pub fn lower_params(ctx: &mut LoweringContext, params: &[KernelParam]) -> syn::Result<TokenStream> {
    let mut tokens = TokenStream::new();
    for param in params {
        tokens.extend(lower_one_param(ctx, param)?);
    }
    Ok(tokens)
}

fn lower_one_param(ctx: &mut LoweringContext, param: &KernelParam) -> syn::Result<TokenStream> {
    match &param.ty {
        KernelType::SliceRef(elem_ty) | KernelType::SliceMutRef(elem_ty) => {
            lower_pointer_param(ctx, &param.name, elem_ty, &param.ty)
        }
        scalar_ty => lower_scalar_param(ctx, &param.name, scalar_ty),
    }
}

/// Pointer param: `.param .u64 name_ptr` + `ld.param.u64` + `cvta.to.global` + register.
///
/// The `cvta.to.global` is emitted eagerly here (not lazily on first access)
/// because lazy emission breaks when the first access is inside a conditional
/// branch — the register would be uninitialized on the other path.
/// Eager emission matches nvcc behavior: all pointer conversions happen at
/// kernel entry, before any control flow.
fn lower_pointer_param(
    ctx: &mut LoweringContext,
    name: &str,
    elem_ty: &KernelType,
    full_ty: &KernelType,
) -> syn::Result<TokenStream> {
    let ptx_elem_ty = ctx.ptx_type_tokens(elem_ty);
    let param_name = format!("{name}_ptr");
    let param_reg = ctx.fresh_reg();
    let global_reg = ctx.fresh_reg();

    // Register in locals with full slice type (SliceRef/SliceMutRef) so
    // lower_expr Index can extract elem_type and check mutability.
    // The locals entry points to the PARAM register (pre-cvta), but
    // global_addrs caches the cvta'd register for memory lowering.
    ctx.locals
        .insert(name.to_string(), (param_reg.clone(), full_ty.clone()));
    ctx.global_addrs
        .insert(name.to_string(), global_reg.clone());

    Ok(quote! {
        kernel.add_param(PtxParam::pointer(#param_name, PtxType::#ptx_elem_ty));
        let #param_reg = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
            dst: #param_reg,
            param_name: #param_name.to_string(),
            ty: PtxType::U64,
        }));
        let #global_reg = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Memory(MemoryOp::CvtaToGlobal {
            dst: #global_reg,
            src: #param_reg,
        }));
    })
}

/// Scalar param: `.param .{ty} name` + `ld.param.{ty}` + register in locals.
fn lower_scalar_param(
    ctx: &mut LoweringContext,
    name: &str,
    ty: &KernelType,
) -> syn::Result<TokenStream> {
    let ptx_ty = ctx.ptx_type_tokens(ty);
    let reg = ctx.fresh_reg();

    ctx.locals
        .insert(name.to_string(), (reg.clone(), ty.clone()));

    Ok(quote! {
        kernel.add_param(PtxParam::scalar(#name, PtxType::#ptx_ty));
        let #reg = alloc.alloc(PtxType::#ptx_ty);
        kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
            dst: #reg,
            param_name: #name.to_string(),
            ty: PtxType::#ptx_ty,
        }));
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use proc_macro2::Span;

    #[test]
    fn lower_scalar_u32() {
        let mut ctx = LoweringContext::new();
        let param = KernelParam {
            name: "n".to_string(),
            ty: KernelType::U32,
            span: Span::call_site(),
        };
        let tokens = lower_params(&mut ctx, &[param]).unwrap();
        let code = tokens.to_string();

        assert!(code.contains("PtxParam :: scalar"));
        assert!(code.contains("\"n\""));
        assert!(code.contains("PtxType :: U32"));
        assert!(code.contains("MemoryOp :: LdParam"));

        // Should register in locals
        assert!(ctx.locals.contains_key("n"));
        assert_eq!(ctx.locals["n"].1, KernelType::U32);
    }

    #[test]
    fn lower_pointer_f32() {
        let mut ctx = LoweringContext::new();
        let param = KernelParam {
            name: "a".to_string(),
            ty: KernelType::SliceRef(Box::new(KernelType::F32)),
            span: Span::call_site(),
        };
        let tokens = lower_params(&mut ctx, &[param]).unwrap();
        let code = tokens.to_string();

        assert!(code.contains("PtxParam :: pointer"));
        assert!(code.contains("\"a_ptr\""));
        assert!(code.contains("PtxType :: F32")); // elem type
        assert!(code.contains("PtxType :: U64")); // ld.param type for pointers
        assert!(code.contains("MemoryOp :: LdParam"));

        // Should register as SliceRef in locals
        assert!(ctx.locals.contains_key("a"));
        assert!(ctx.locals["a"].1.is_slice());
    }

    #[test]
    fn lower_vector_add_params() {
        let mut ctx = LoweringContext::new();
        let params = vec![
            KernelParam {
                name: "a".to_string(),
                ty: KernelType::SliceRef(Box::new(KernelType::F32)),
                span: Span::call_site(),
            },
            KernelParam {
                name: "b".to_string(),
                ty: KernelType::SliceRef(Box::new(KernelType::F32)),
                span: Span::call_site(),
            },
            KernelParam {
                name: "out".to_string(),
                ty: KernelType::SliceMutRef(Box::new(KernelType::F32)),
                span: Span::call_site(),
            },
            KernelParam {
                name: "n".to_string(),
                ty: KernelType::U32,
                span: Span::call_site(),
            },
        ];
        let _tokens = lower_params(&mut ctx, &params).unwrap();

        // All four should be in locals
        assert_eq!(ctx.locals.len(), 4);
        assert!(ctx.locals["a"].1.is_slice());
        assert!(ctx.locals["out"].1.is_mut_slice());
        assert_eq!(ctx.locals["n"].1, KernelType::U32);
    }
}
