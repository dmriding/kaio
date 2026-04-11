//! Parse a kernel function signature into a `KernelSignature`.

use syn::spanned::Spanned;
use syn::{FnArg, ItemFn, Pat, ReturnType, Type};

use crate::kernel_ir::{KernelConfig, KernelParam, KernelSignature, KernelType};

/// Parse a Rust type into a `KernelType`.
///
/// Supported types:
/// - Scalars: `f32`, `f64`, `i32`, `u32`, `i64`, `u64`, `bool`
/// - Slices: `&[T]`, `&mut [T]` where T is a supported scalar
fn parse_type(ty: &Type) -> syn::Result<KernelType> {
    match ty {
        // Scalar: path type like `f32`, `u32`, etc.
        Type::Path(type_path) => {
            if type_path.qself.is_some() || type_path.path.segments.len() != 1 {
                return Err(syn::Error::new_spanned(
                    ty,
                    "unsupported type in GPU kernel parameter. \
                     Supported: f32, f64, i32, u32, i64, u64, bool, &[T], &mut [T]",
                ));
            }
            let ident = &type_path.path.segments[0].ident;
            match ident.to_string().as_str() {
                "f32" => Ok(KernelType::F32),
                "f64" => Ok(KernelType::F64),
                "i32" => Ok(KernelType::I32),
                "u32" => Ok(KernelType::U32),
                "i64" => Ok(KernelType::I64),
                "u64" => Ok(KernelType::U64),
                "bool" => Ok(KernelType::Bool),
                other => Err(syn::Error::new_spanned(
                    ty,
                    format!(
                        "unsupported type `{other}` in GPU kernel parameter. \
                         Supported: f32, f64, i32, u32, i64, u64, bool, &[T], &mut [T]"
                    ),
                )),
            }
        }

        // Reference: &[T] or &mut [T]
        Type::Reference(type_ref) => {
            if type_ref.lifetime.is_some() {
                return Err(syn::Error::new_spanned(
                    ty,
                    "lifetime parameters are not supported in GPU kernels",
                ));
            }

            match type_ref.elem.as_ref() {
                Type::Slice(type_slice) => {
                    let elem_ty = parse_type(&type_slice.elem)?;
                    if !elem_ty.is_scalar() {
                        return Err(syn::Error::new_spanned(
                            &type_slice.elem,
                            "nested slices are not supported in GPU kernels",
                        ));
                    }
                    if type_ref.mutability.is_some() {
                        Ok(KernelType::SliceMutRef(Box::new(elem_ty)))
                    } else {
                        Ok(KernelType::SliceRef(Box::new(elem_ty)))
                    }
                }
                _ => Err(syn::Error::new_spanned(
                    ty,
                    "only slice references (&[T] / &mut [T]) are supported in GPU kernels",
                )),
            }
        }

        _ => Err(syn::Error::new_spanned(
            ty,
            "unsupported type in GPU kernel parameter. \
             Supported: f32, f64, i32, u32, i64, u64, bool, &[T], &mut [T]",
        )),
    }
}

/// Parse a `syn::ItemFn` into a `KernelSignature` with the given config.
pub fn parse_kernel_signature(func: &ItemFn, config: KernelConfig) -> syn::Result<KernelSignature> {
    let name = func.sig.ident.to_string();
    let name_span = func.sig.ident.span();

    // Reject return types other than ()
    if let ReturnType::Type(_, ref ty) = func.sig.output {
        return Err(syn::Error::new_spanned(
            ty,
            "GPU kernels must return `()`. Found a return type.",
        ));
    }

    // Reject generics
    if !func.sig.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &func.sig.generics,
            "generic type parameters are not supported in GPU kernels",
        ));
    }

    // Reject async
    if func.sig.asyncness.is_some() {
        return Err(syn::Error::new_spanned(
            func.sig.asyncness,
            "`async` is not supported in GPU kernels",
        ));
    }

    // Reject unsafe
    if func.sig.unsafety.is_some() {
        return Err(syn::Error::new_spanned(
            func.sig.unsafety,
            "`unsafe` is not supported in GPU kernels",
        ));
    }

    // Parse parameters
    let mut params = Vec::new();
    for arg in &func.sig.inputs {
        match arg {
            FnArg::Receiver(_) => {
                return Err(syn::Error::new_spanned(
                    arg,
                    "`self` parameters are not supported in GPU kernels",
                ));
            }
            FnArg::Typed(pat_type) => {
                let param_name = match pat_type.pat.as_ref() {
                    Pat::Ident(pat_ident) => pat_ident.ident.to_string(),
                    _ => {
                        return Err(syn::Error::new_spanned(
                            &pat_type.pat,
                            "only simple identifier patterns are supported in GPU kernel parameters",
                        ));
                    }
                };
                let param_ty = parse_type(&pat_type.ty)?;
                let span = pat_type.pat.span();
                params.push(KernelParam {
                    name: param_name,
                    ty: param_ty,
                    span,
                });
            }
        }
    }

    Ok(KernelSignature {
        name,
        params,
        config,
        name_span,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;

    fn parse_fn(tokens: proc_macro2::TokenStream) -> ItemFn {
        syn::parse2(tokens).expect("failed to parse function")
    }

    fn dummy_config() -> KernelConfig {
        KernelConfig {
            block_size: 256,
            block_size_y: None,
            block_size_span: proc_macro2::Span::call_site(),
        }
    }

    #[test]
    fn parse_vector_add_signature() {
        let func = parse_fn(quote! {
            fn vector_add(a: &[f32], b: &[f32], out: &mut [f32], n: u32) {}
        });
        let sig = parse_kernel_signature(&func, dummy_config()).unwrap();
        assert_eq!(sig.name, "vector_add");
        assert_eq!(sig.params.len(), 4);

        assert_eq!(sig.params[0].name, "a");
        assert_eq!(
            sig.params[0].ty,
            KernelType::SliceRef(Box::new(KernelType::F32))
        );

        assert_eq!(sig.params[1].name, "b");
        assert_eq!(
            sig.params[1].ty,
            KernelType::SliceRef(Box::new(KernelType::F32))
        );

        assert_eq!(sig.params[2].name, "out");
        assert_eq!(
            sig.params[2].ty,
            KernelType::SliceMutRef(Box::new(KernelType::F32))
        );

        assert_eq!(sig.params[3].name, "n");
        assert_eq!(sig.params[3].ty, KernelType::U32);

        assert_eq!(sig.config.block_size, 256);
    }

    #[test]
    fn parse_all_scalar_types() {
        let func = parse_fn(quote! {
            fn kernel(a: f32, b: f64, c: i32, d: u32, e: i64, f: u64, g: bool) {}
        });
        let sig = parse_kernel_signature(&func, dummy_config()).unwrap();
        assert_eq!(sig.params[0].ty, KernelType::F32);
        assert_eq!(sig.params[1].ty, KernelType::F64);
        assert_eq!(sig.params[2].ty, KernelType::I32);
        assert_eq!(sig.params[3].ty, KernelType::U32);
        assert_eq!(sig.params[4].ty, KernelType::I64);
        assert_eq!(sig.params[5].ty, KernelType::U64);
        assert_eq!(sig.params[6].ty, KernelType::Bool);
    }

    #[test]
    fn parse_f64_slices() {
        let func = parse_fn(quote! {
            fn kernel(data: &[f64], out: &mut [f64]) {}
        });
        let sig = parse_kernel_signature(&func, dummy_config()).unwrap();
        assert_eq!(
            sig.params[0].ty,
            KernelType::SliceRef(Box::new(KernelType::F64))
        );
        assert_eq!(
            sig.params[1].ty,
            KernelType::SliceMutRef(Box::new(KernelType::F64))
        );
    }

    #[test]
    fn reject_return_type() {
        let func = parse_fn(quote! {
            fn kernel(n: u32) -> u32 { n }
        });
        let err = parse_kernel_signature(&func, dummy_config()).unwrap_err();
        assert!(err.to_string().contains("return"));
    }

    #[test]
    fn reject_generics() {
        let func = parse_fn(quote! {
            fn kernel<T>(data: &[f32]) {}
        });
        let err = parse_kernel_signature(&func, dummy_config()).unwrap_err();
        assert!(err.to_string().contains("generic"));
    }

    #[test]
    fn reject_unsupported_type() {
        let func = parse_fn(quote! {
            fn kernel(name: String) {}
        });
        let err = parse_kernel_signature(&func, dummy_config()).unwrap_err();
        assert!(err.to_string().contains("unsupported type"));
    }

    #[test]
    fn reject_lifetime() {
        let func = parse_fn(quote! {
            fn kernel(data: &'a [f32]) {}
        });
        let err = parse_kernel_signature(&func, dummy_config()).unwrap_err();
        assert!(err.to_string().contains("lifetime"));
    }

    #[test]
    fn reject_self_param() {
        let func: ItemFn = syn::parse2(quote! {
            fn kernel(self, n: u32) {}
        })
        .unwrap();
        let err = parse_kernel_signature(&func, dummy_config()).unwrap_err();
        assert!(err.to_string().contains("self"));
    }

    #[test]
    fn reject_async() {
        let func = parse_fn(quote! {
            async fn kernel(n: u32) {}
        });
        let err = parse_kernel_signature(&func, dummy_config()).unwrap_err();
        assert!(err.to_string().contains("async"));
    }

    #[test]
    fn reject_unsafe() {
        let func = parse_fn(quote! {
            unsafe fn kernel(n: u32) {}
        });
        let err = parse_kernel_signature(&func, dummy_config()).unwrap_err();
        assert!(err.to_string().contains("unsafe"));
    }

    #[test]
    fn reject_non_slice_reference() {
        let func = parse_fn(quote! {
            fn kernel(data: &f32) {}
        });
        let err = parse_kernel_signature(&func, dummy_config()).unwrap_err();
        assert!(err.to_string().contains("slice references"));
    }
}
