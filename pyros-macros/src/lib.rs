//! # pyros-macros
//!
//! Proc macro crate for PYROS. Provides the `#[gpu_kernel]` attribute
//! macro that transforms Rust function syntax into PTX codegen +
//! typed launch wrappers.
//!
//! This crate is not intended to be used directly — use `pyros` and
//! import via `pyros::prelude::*`.

#![warn(missing_docs)]

mod kernel_ir;
mod lower;
mod parse;

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::ItemFn;

use parse::attrs::parse_kernel_config;
use parse::signature::parse_kernel_signature;

/// Marks a function as a GPU kernel compiled to PTX.
///
/// # Attributes
///
/// - `block_size = N` (required): Number of threads per block. Must be
///   a power of 2 in the range `[1, 1024]`.
///
/// # Example
///
/// ```ignore
/// use pyros::prelude::*;
///
/// #[gpu_kernel(block_size = 256)]
/// fn vector_add(a: &[f32], b: &[f32], out: &mut [f32], n: u32) {
///     let idx = thread_idx_x() + block_idx_x() * block_dim_x();
///     if idx < n {
///         out[idx] = a[idx] + b[idx];
///     }
/// }
/// ```
#[proc_macro_attribute]
pub fn gpu_kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    match gpu_kernel_impl(attr.into(), item.into()) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn gpu_kernel_impl(attr: TokenStream2, item: TokenStream2) -> syn::Result<TokenStream2> {
    // Parse the function
    let func: ItemFn = syn::parse2(item)?;

    // Parse attribute config
    let config = parse_kernel_config(attr)?;

    // Parse and validate signature
    let sig = parse_kernel_signature(&func, config)?;

    // Phase 2 Sprint 2.1: return a placeholder module.
    // Sprints 2.2-2.6 will replace this with real codegen.
    let mod_name = syn::Ident::new(&sig.name, sig.name_span);
    let _block_size = sig.config.block_size;

    Ok(quote! {
        /// Generated GPU kernel module. Codegen not yet implemented —
        /// will produce `build_ptx()` + `launch()` after Sprint 2.6.
        mod #mod_name {
            // Placeholder: signature parsed successfully.
            // Parameters: #(#param_names: #param_types),*
        }
    })
}
