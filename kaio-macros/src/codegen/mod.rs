//! Code generation — assemble the final `TokenStream` from lowered fragments.

mod launch_wrapper;
mod ptx_builder;

use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

use crate::kernel_ir::KernelSignature;
use crate::kernel_ir::stmt::KernelStmt;

/// Generate the complete kernel module from a parsed signature and body.
///
/// Produces:
/// ```ignore
/// mod kernel_name {
///     static PTX_CACHE: OnceLock<String> = OnceLock::new();
///     fn build_ptx() -> String { ... }
///     pub fn launch(device, params...) -> Result<(), KaioError> { ... }
/// }
/// ```
pub fn generate_kernel_module(
    sig: &KernelSignature,
    body: &[KernelStmt],
) -> syn::Result<TokenStream> {
    let mod_name = Ident::new(&sig.name, sig.name_span);

    let build_ptx_fn = ptx_builder::generate_build_ptx(sig, body)?;
    let launch_fn = launch_wrapper::generate_launch_fn(sig)?;

    Ok(quote! {
        #[allow(non_snake_case, unused_imports, dead_code)]
        mod #mod_name {
            use std::sync::OnceLock;
            static PTX_CACHE: OnceLock<String> = OnceLock::new();

            #build_ptx_fn
            #launch_fn
        }
    })
}
