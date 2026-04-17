//! # kaio-macros
//!
//! Proc macro crate for KAIO. Provides the `#[gpu_kernel]` attribute
//! macro that transforms Rust function syntax into PTX codegen +
//! typed launch wrappers.
//!
//! This crate is not intended to be used directly — use `kaio` and
//! import via `kaio::prelude::*`.

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
/// # DSL, not compiled Rust
///
/// The function body uses Rust syntax but is **not compiled by rustc**.
/// The proc macro parses it into KAIO's own IR (`KernelStmt`) and emits
/// PTX text directly. No LLVM, no MIR, no borrow checker runs on the
/// kernel body.
///
/// This has an important consequence for `&mut [T]` parameters: in
/// standard Rust, `&mut T` carries a `noalias` guarantee — the compiler
/// assumes exclusive access. In a GPU kernel, thousands of threads
/// execute the same function body concurrently, all accessing the same
/// buffer. Because the body never reaches rustc's backend, no `noalias`
/// attribute is emitted — ptxas sees a plain `.u64` param. There is no
/// UB from the aliasing mismatch, but the `&mut` syntax is misleading:
/// correctness depends on the kernel author writing disjoint access
/// patterns (e.g. `if idx < n` bounds guards), not on compiler-enforced
/// uniqueness.
///
/// A future release will accept `*mut [T]` / `*const [T]` as the
/// primary kernel parameter syntax to better communicate this. See
/// RFC-0001 in the repository for the design direction.
///
/// You cannot call Rust functions declared outside the kernel inside the
/// kernel body. The supported syntax subset includes: arithmetic,
/// comparisons, bitwise ops, short-circuit `&&`/`||`, compound
/// assignment, `if`/`else`, `for`/`while` loops, `let` bindings, and
/// KAIO GPU builtins (`thread_idx_x()`, `shared_mem!`, etc.).
///
/// # Attributes
///
/// - `block_size = N` (required): Number of threads per block. Must be
///   a power of 2 in the range `[1, 1024]`.
///
/// # Example
///
/// ```ignore
/// use kaio::prelude::*;
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
