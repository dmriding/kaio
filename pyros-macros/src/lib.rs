//! # pyros-macros
//!
//! Proc macro crate for PYROS. Provides the `#[gpu_kernel]` attribute
//! macro that transforms Rust function syntax into PTX codegen +
//! typed launch wrappers.
//!
//! This crate is not intended to be used directly — use `pyros` and
//! import via `pyros::prelude::*`.

#![warn(missing_docs)]

mod codegen;
mod kernel_ir;
mod lower;
mod parse;

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use syn::ItemFn;

use parse::attrs::parse_kernel_config;
use parse::body::parse_body;
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

    // Parse body into kernel IR
    let body = parse_body(&func.block)?;

    // Generate the kernel module (build_ptx + launch)
    codegen::generate_kernel_module(&sig, &body)
}
