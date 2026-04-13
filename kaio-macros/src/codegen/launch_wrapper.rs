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
/// Launch configuration depends on block dimensionality:
/// - **1D** (`block_size = N`): uses the last `u32` scalar param to
///   compute grid size. Block dim is set to the declared block_size.
/// - **2D** (`block_size = (X, Y)`): the generated `launch()` takes a
///   `grid: (u32, u32, u32)` parameter. Block dims are hardcoded from
///   the attribute to prevent mismatches.
pub fn generate_launch_fn(sig: &KernelSignature) -> syn::Result<TokenStream> {
    let kernel_name = &sig.name;
    let is_2d = sig.config.block_size_y.is_some();

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
                // Track last u32 for 1D grid config
                if !is_2d && *scalar_ty == KernelType::U32 {
                    elem_count_ident = Some(name.clone());
                }
            }
        }
    }

    let launch_config_expr = if is_2d {
        // 2D: accept grid dims, hardcode block dims from attribute.
        // This prevents mismatches between declared block_size and runtime config.
        let bx = sig.config.block_size;
        let by = sig.config.block_size_y.unwrap(); // safe: is_2d checked
        launch_params.push(quote! { grid: (u32, u32, u32) });
        quote! {
            kaio::runtime::LaunchConfig {
                grid_dim: grid,
                block_dim: (#bx, #by, 1),
                shared_mem_bytes: 0,
            }
        }
    } else {
        // 1D: infer grid from last u32 (existing behavior)
        let n_ident = elem_count_ident.ok_or_else(|| {
            syn::Error::new(
                sig.name_span,
                "GPU kernel must have at least one `u32` parameter for element count \
                 (used to compute grid size). For 2D kernels, use \
                 `block_size = (X, Y)` which accepts a `grid: (u32, u32, u32)` parameter.",
            )
        })?;
        {
            let bs = sig.config.block_size;
            quote! {
                kaio::runtime::LaunchConfig {
                    grid_dim: (#n_ident.div_ceil(#bs), 1, 1),
                    block_dim: (#bs, 1, 1),
                    shared_mem_bytes: 0,
                }
            }
        }
    };

    Ok(quote! {
        /// Launch this GPU kernel on the given device.
        pub fn launch(#(#launch_params),*) -> Result<(), kaio::runtime::KaioError> {
            use kaio::runtime::PushKernelArg;

            // Derive the SM target from the device's compute capability
            // so PtxModule::validate() can gate SM-specific features
            // (mma.sync, cp.async, etc.) before ptxas sees the PTX.
            // Sprint 6.10 D1a: closes the trust-boundary gap where
            // load_ptx(&str) bypassed PtxModule::validate().
            let info = device.info()?;
            let (major, minor) = info.compute_capability;
            let sm = format!("sm_{major}{minor}");
            let ptx_module = build_module(&sm);
            let module = device.load_module(&ptx_module)?;
            let func = module.function(#kernel_name)?;
            let cfg = #launch_config_expr;

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

#[cfg(test)]
mod tests {
    //! Host-level codegen regression tests for the launch wrapper.
    //!
    //! These tests protect critical structural invariants of the emitted
    //! `launch()` function without requiring a GPU. They run in standard
    //! `cargo test` and in CI on non-GPU runners.
    //!
    //! Added Sprint 6.10 (D2). Each test has a regression canary comment
    //! documenting the specific mutation it guards against.

    use super::*;
    use crate::kernel_ir::{KernelConfig, KernelParam, KernelSignature, KernelType};
    use proc_macro2::Span;

    fn mock_param_slice_mut_f32(name: &str) -> KernelParam {
        KernelParam {
            name: name.to_string(),
            ty: KernelType::SliceMutRef(Box::new(KernelType::F32)),
            span: Span::call_site(),
        }
    }

    fn mock_param_u32(name: &str) -> KernelParam {
        KernelParam {
            name: name.to_string(),
            ty: KernelType::U32,
            span: Span::call_site(),
        }
    }

    fn mock_signature_1d(name: &str, block_size: u32) -> KernelSignature {
        KernelSignature {
            name: name.to_string(),
            // Minimal 1D kernel: `fn k(out: &mut [f32], n: u32)`.
            // The `n: u32` is required because 1D codegen uses the last
            // u32 scalar to compute grid_dim.
            params: vec![
                mock_param_slice_mut_f32("out"),
                mock_param_u32("n"),
            ],
            config: KernelConfig {
                block_size,
                block_size_y: None,
                block_size_span: Span::call_site(),
            },
            name_span: Span::call_site(),
        }
    }

    fn mock_signature_2d(name: &str, block_size_x: u32, block_size_y: u32) -> KernelSignature {
        KernelSignature {
            name: name.to_string(),
            // Minimal 2D kernel: `fn k(out: &mut [f32])`.
            // 2D codegen takes an explicit `grid` parameter, no u32 required.
            params: vec![mock_param_slice_mut_f32("out")],
            config: KernelConfig {
                block_size: block_size_x,
                block_size_y: Some(block_size_y),
                block_size_span: Span::call_site(),
            },
            name_span: Span::call_site(),
        }
    }

    #[test]
    fn launch_wrapper_emits_correct_block_dim_1d() {
        // Regression canary: if launch_wrapper.rs ever emits
        //   block_dim: (1, 1, 1)
        // or mismatches the declared #[gpu_kernel(block_size = N)] value,
        // or omits block_dim from the LaunchConfig entirely,
        // this test fails.
        //
        // This is the critical block_dim fix from Sprint 4.2 that was
        // previously only caught by --ignored GPU tests. Now protected
        // host-side.
        let sig = mock_signature_1d("test_kernel_1d", 256);
        let output = generate_launch_fn(&sig)
            .expect("codegen should succeed for mock signature")
            .to_string();

        // TokenStream.to_string() renders with spaces between tokens, so
        // "block_dim: (256u32, 1, 1)" serializes as "block_dim : (256u32 , 1 , 1)".
        // We assert on semantic structure, not exact formatting.
        assert!(
            output.contains("block_dim : (256u32 , 1 , 1)"),
            "expected block_dim (256u32, 1, 1) for block_size=256, got:\n{output}"
        );
    }

    #[test]
    fn launch_wrapper_emits_correct_block_dim_2d() {
        // Regression canary: if 2D launch wrapper omits block_size_y,
        // swaps x/y, or mismatches the declared
        //   #[gpu_kernel(block_size = (X, Y))]
        // values, this test fails.
        //
        // 2D block_dim must be literally (X, Y, 1) — hardcoded from the
        // attribute, not inferred from runtime args. Prevents user from
        // accidentally launching with the wrong shape.
        let sig = mock_signature_2d("test_kernel_2d", 16, 8);
        let output = generate_launch_fn(&sig)
            .expect("codegen should succeed for mock signature")
            .to_string();

        assert!(
            output.contains("block_dim : (16u32 , 8u32 , 1)"),
            "expected block_dim (16u32, 8u32, 1) for block_size=(16, 8), got:\n{output}"
        );
    }

    #[test]
    fn launch_wrapper_threads_compute_capability_into_module_build() {
        // Regression canary: the launch wrapper must structurally:
        //   1. Obtain device.info().compute_capability
        //   2. Format it as sm_XX (e.g. sm_80)
        //   3. Pass that SM target into the module build function
        //   4. Call device.load_module(&module) (not device.load_ptx(&str))
        //
        // If any of those steps is missing or hardcoded to a fixed SM, the
        // trust-boundary fix from Sprint 6.10 D1a is broken and user-authored
        // kernels with Ampere-gated features fall back to driver-level
        // ptxas errors instead of structured PtxModule::validate() failures.
        //
        // Activated Sprint 6.10 D1a. Assertions verify:
        //   - output contains "compute_capability" (source of SM)
        //   - output contains "load_module" (correct loader)
        //   - output does NOT contain "load_ptx" (old loader gone from macro)

        let sig = mock_signature_1d("test_kernel_sm", 256);
        let output = generate_launch_fn(&sig)
            .expect("codegen should succeed for mock signature")
            .to_string();

        assert!(
            output.contains("compute_capability"),
            "expected launch wrapper to read device.info().compute_capability, got:\n{output}"
        );
        assert!(
            output.contains("load_module"),
            "expected launch wrapper to call device.load_module, got:\n{output}"
        );
        assert!(
            !output.contains("load_ptx"),
            "launch wrapper should not call device.load_ptx after D1a migration, got:\n{output}"
        );
    }
}
