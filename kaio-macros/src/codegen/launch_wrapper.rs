//! Generate the `launch()` function.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::Ident;

use crate::kernel_ir::{KernelSignature, KernelType};

/// Generate the `pub fn launch(...)` function.
///
/// Maps kernel params to launch function params:
/// - `&[T]` → `&GpuBuffer<T>`
/// - `&mut [T]` → `&mut GpuBuffer<T>`
/// - scalar → same type
///
/// Uses the last `u32` scalar param for `LaunchConfig::for_num_elems()`.
///
/// **Known limitation (Phase 2):** The last u32 parameter is assumed to be
/// the element count. Custom grid config support is deferred to Phase 3.
pub fn generate_launch_fn(sig: &KernelSignature) -> syn::Result<TokenStream> {
    let kernel_name = &sig.name;

    // Build launch function params and .arg() calls
    let mut launch_params: Vec<TokenStream> = Vec::new();
    let mut arg_calls: Vec<TokenStream> = Vec::new();
    let mut elem_count_ident: Option<Ident> = None;

    // First param is always device
    launch_params.push(quote! { device: &kaio::runtime::KaioDevice });

    for param in &sig.params {
        let name = format_ident!("{}", param.name);
        match &param.ty {
            KernelType::SliceRef(elem) => {
                let rust_ty = rust_type_tokens(elem);
                launch_params.push(quote! { #name: &kaio::runtime::GpuBuffer<#rust_ty> });
                arg_calls.push(quote! { .arg(#name.inner()) });
            }
            KernelType::SliceMutRef(elem) => {
                let rust_ty = rust_type_tokens(elem);
                launch_params.push(quote! { #name: &mut kaio::runtime::GpuBuffer<#rust_ty> });
                arg_calls.push(quote! { .arg(#name.inner_mut()) });
            }
            scalar_ty => {
                let rust_ty = rust_type_tokens(scalar_ty);
                launch_params.push(quote! { #name: #rust_ty });
                arg_calls.push(quote! { .arg(&#name) });
                // Track last u32 for grid config
                if *scalar_ty == KernelType::U32 {
                    elem_count_ident = Some(name.clone());
                }
            }
        }
    }

    let n_ident = elem_count_ident.ok_or_else(|| {
        syn::Error::new(
            sig.name_span,
            "GPU kernel must have at least one `u32` parameter for element count \
             (used by LaunchConfig::for_num_elems)",
        )
    })?;

    Ok(quote! {
        /// Launch this GPU kernel on the given device.
        pub fn launch(#(#launch_params),*) -> Result<(), kaio::runtime::KaioError> {
            use kaio::runtime::PushKernelArg;

            let ptx = PTX_CACHE.get_or_init(build_ptx);
            let module = device.load_ptx(ptx)?;
            let func = module.function(#kernel_name)?;
            let cfg = kaio::runtime::LaunchConfig::for_num_elems(#n_ident);

            // SAFETY: kernel signature matches params (enforced by macro codegen),
            // buffers are valid device pointers (enforced by GpuBuffer construction),
            // launch config within device limits (enforced by LaunchConfig).
            unsafe {
                device
                    .stream()
                    .launch_builder(func.inner())
                    #(#arg_calls)*
                    .launch(cfg)?;
            }
            Ok(())
        }
    })
}

/// Convert a `KernelType` to the Rust type tokens for the launch function signature.
fn rust_type_tokens(ty: &KernelType) -> TokenStream {
    match ty {
        KernelType::F32 => quote! { f32 },
        KernelType::F64 => quote! { f64 },
        KernelType::I32 => quote! { i32 },
        KernelType::U32 => quote! { u32 },
        KernelType::I64 => quote! { i64 },
        KernelType::U64 => quote! { u64 },
        KernelType::Bool => quote! { bool },
        _ => panic!("rust_type_tokens called on slice type"),
    }
}
