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
/// # Parameter syntax
///
/// Kernel parameters are written as `*const [T]` (primary) or `&[T]`
/// (sugar) for read-only slices, and `*mut [T]` (primary) or `&mut [T]`
/// (sugar) for read-write slices. Both forms lower to identical PTX.
/// The pointer form is recommended because it accurately signals
/// "device pointer, no aliasing contract" — see RFC-0001. The
/// reference form is accepted as permanent ergonomic sugar; it will
/// not be deprecated.
///
/// Scalar types (`f32`, `f64`, `i32`, `u32`, `i64`, `u64`, `bool`) are
/// passed by value.
///
/// # DSL, not compiled Rust
///
/// The function body uses Rust syntax but is **not compiled by rustc**.
/// The proc macro parses it into KAIO's own IR (`KernelStmt`) and emits
/// PTX text directly. No LLVM, no MIR, no borrow checker runs on the
/// kernel body. ptxas sees a plain `.u64` param for every slice
/// parameter regardless of which surface syntax you wrote.
///
/// Thousands of threads execute the kernel body concurrently, all
/// accessing the same device buffers. Correctness depends on writing
/// disjoint access patterns (e.g. `if idx < n` bounds guards), not on
/// compiler-enforced uniqueness.
///
/// You cannot call Rust functions declared outside the kernel inside
/// the kernel body. The supported syntax subset includes: arithmetic,
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
/// fn vector_add(a: *const [f32], b: *const [f32], out: *mut [f32], n: u32) {
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
