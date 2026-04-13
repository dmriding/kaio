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
///     fn build_module(sm: &str) -> PtxModule { ... }
///     pub fn launch(device, params...) -> Result<(), KaioError> { ... }
/// }
/// ```
///
/// Sprint 6.10 D1a: the `PTX_CACHE: OnceLock<String>` cache is removed.
/// Each `launch()` call rebuilds the `PtxModule` fresh, using the device's
/// own compute capability as the SM target. Per-call rebuild cost is
/// microseconds (IR construction, not compilation); if it later becomes
/// hot, a cache-design sprint can re-introduce caching deliberately.
pub fn generate_kernel_module(
    sig: &KernelSignature,
    body: &[KernelStmt],
) -> syn::Result<TokenStream> {
    let mod_name = Ident::new(&sig.name, sig.name_span);

    let build_module_fn = ptx_builder::generate_build_module(sig, body)?;
    let launch_fn = launch_wrapper::generate_launch_fn(sig)?;

    Ok(quote! {
        #[allow(non_snake_case, unused_imports, dead_code)]
        mod #mod_name {
            #build_module_fn
            #launch_fn
        }
    })
}

#[cfg(test)]
mod tests {
    //! Host-level codegen regression tests for the lowering pipeline.
    //!
    //! These tests drive `parse_body` + `generate_kernel_module` end to end
    //! and inspect the emitted `TokenStream` for specific semantic patterns
    //! the lowering must produce. No GPU required.
    //!
    //! Added Sprint 6.10 (D2). Each test has a regression canary comment
    //! documenting the mutation it guards against.

    use super::*;
    use crate::kernel_ir::KernelConfig;
    use crate::parse::body::parse_body;
    use crate::parse::signature::parse_kernel_signature;
    use proc_macro2::Span;
    use quote::quote;
    use syn::ItemFn;

    fn dummy_config(block_size: u32) -> KernelConfig {
        KernelConfig {
            block_size,
            block_size_y: None,
            block_size_span: Span::call_site(),
        }
    }

    fn parse_kernel(tokens: proc_macro2::TokenStream) -> ItemFn {
        syn::parse2(tokens).expect("failed to parse test kernel")
    }

    #[test]
    fn shared_memory_lowering_emits_shared_addr_pattern() {
        // Regression canary: if shared memory lowering ever stops emitting
        //   Operand::SharedAddr("<name>".to_string())
        // as the base-address source for shared_mem![] access, or if it
        // switches to a raw pointer-arithmetic path that bypasses the
        // named-symbol scheme, this test fails.
        //
        // The SharedAddr pattern is load-bearing because shared memory
        // allocations are named .shared symbols in PTX, not anonymous
        // offsets. Losing the named-symbol path breaks cross-instruction
        // referencing and any future debugger/profiler integration.
        let func = parse_kernel(quote! {
            fn shared_kernel(n: u32) {
                let sdata = shared_mem![f32; 256];
                sdata[0] = 0.0f32;
            }
        });
        let sig = parse_kernel_signature(&func, dummy_config(256))
            .expect("signature should parse");
        let body = parse_body(&func.block).expect("body should parse");

        let module = generate_kernel_module(&sig, &body)
            .expect("codegen should succeed for valid shared_mem kernel");
        let output = module.to_string();

        assert!(
            output.contains("SharedAddr"),
            "expected Operand::SharedAddr(...) in shared-memory lowering output, \
             but did not find it. First 800 chars:\n{}",
            &output[..output.len().min(800)]
        );
    }

    #[test]
    fn reduction_lowering_uses_named_symbol() {
        // Regression canary: if block_reduce_sum / block_reduce_max lowering
        // ever stops using the literal string "_kaio_reduce_smem" as the
        // shared-memory allocation name (e.g., switches to an anonymous
        // allocation, renames the symbol, or inlines the shared region
        // into a different layout), this test fails.
        //
        // The named symbol is required because the reduction lowering
        // performs multiple load/store operations against the same shared
        // region across warp rounds and cross-warp broadcast. All of those
        // touch the same SharedAddr by name. Losing the stable name breaks
        // the multi-phase reduction in a silent correctness-killing way.
        let func = parse_kernel(quote! {
            fn reduce_kernel(out: &mut [f32], n: u32) {
                let x = 1.0f32;
                let s = block_reduce_sum(x);
                out[0] = s;
            }
        });
        let sig = parse_kernel_signature(&func, dummy_config(256))
            .expect("signature should parse");
        let body = parse_body(&func.block).expect("body should parse");

        let module = generate_kernel_module(&sig, &body)
            .expect("codegen should succeed for valid reduction kernel");
        let output = module.to_string();

        assert!(
            output.contains("\"_kaio_reduce_smem\""),
            "expected \"_kaio_reduce_smem\" string literal in reduction lowering output, \
             but did not find it. First 800 chars:\n{}",
            &output[..output.len().min(800)]
        );
    }
}
