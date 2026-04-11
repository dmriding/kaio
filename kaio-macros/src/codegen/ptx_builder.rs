//! Generate the `build_ptx()` function body.

use proc_macro2::TokenStream;
use quote::quote;

use crate::kernel_ir::KernelSignature;
use crate::kernel_ir::stmt::KernelStmt;
use crate::lower;
use crate::lower::LoweringContext;

/// Generate the `build_ptx() -> String` function.
///
/// This produces a function that, at runtime:
/// 1. Creates a `RegisterAllocator` and `PtxKernel`
/// 2. Declares and loads all parameters
/// 3. Executes the lowered kernel body
/// 4. Adds `Ret`, finalizes registers
/// 5. Emits PTX via `PtxWriter`
/// 6. Optionally dumps PTX if `KAIO_DUMP_PTX` is set
pub fn generate_build_ptx(sig: &KernelSignature, body: &[KernelStmt]) -> syn::Result<TokenStream> {
    let kernel_name = &sig.name;
    let mut ctx = LoweringContext::new();
    ctx.block_size = Some(sig.config.block_size);

    // Lower parameters (populates ctx.locals)
    let param_tokens = lower::params::lower_params(&mut ctx, &sig.params)?;

    // Lower body statements
    let body_tokens = lower::lower_stmts(&mut ctx, body)?;

    // Compute total shared memory for compile-time reporting
    let total_shared_bytes: u32 = ctx
        .shared_arrays
        .values()
        .map(|(ty, count)| (ty.size_bytes() * count) as u32)
        .sum::<u32>()
        + if ctx.reduce_smem_allocated {
            (ctx.block_size.unwrap_or(256) / 32) * 4
        } else {
            0
        };

    // Emit shared memory diagnostic in generated code
    let shared_mem_diagnostic = if total_shared_bytes > 0 {
        let kb = total_shared_bytes as f64 / 1024.0;
        if total_shared_bytes > 49152 {
            // 48KB limit warning
            quote! {
                eprintln!("KAIO warning: kernel '{}' uses {} bytes ({:.1} KB) of shared memory — exceeds 48 KB default limit",
                    #kernel_name, #total_shared_bytes, #kb);
            }
        } else {
            quote! {}
        }
    } else {
        quote! {}
    };

    Ok(quote! {
        fn build_ptx() -> String {
            use kaio::core::emit::{Emit, PtxWriter};
            use kaio::core::instr::ArithOp;
            use kaio::core::instr::control::{CmpOp, ControlOp};
            use kaio::core::instr::memory::MemoryOp;
            use kaio::core::instr::special;
            use kaio::core::ir::{
                Operand, PtxInstruction, PtxKernel, PtxModule, PtxParam, RegisterAllocator,
                SharedDecl,
            };
            use kaio::core::types::PtxType;

            let mut alloc = RegisterAllocator::new();
            let mut kernel = PtxKernel::new(#kernel_name);

            #param_tokens
            #body_tokens

            kernel.push(PtxInstruction::Control(ControlOp::Ret));
            kernel.set_registers(alloc.into_allocated());

            let sm_target = std::env::var("KAIO_SM_TARGET")
                .unwrap_or_else(|_| "sm_70".to_string());
            let mut module = PtxModule::new(&sm_target);
            module.add_kernel(kernel);

            let mut w = PtxWriter::new();
            module.emit(&mut w).unwrap();
            let ptx = w.finish();

            #shared_mem_diagnostic

            if std::env::var("KAIO_DUMP_PTX").is_ok() {
                let dump_dir = std::env::var("OUT_DIR")
                    .unwrap_or_else(|_| ".".to_string());
                let dump_path = format!("{}/{}.ptx", dump_dir, #kernel_name);
                match std::fs::write(&dump_path, &ptx) {
                    Ok(()) => eprintln!("KAIO: wrote {}", dump_path),
                    Err(e) => eprintln!("KAIO: failed to write {}: {}", dump_path, e),
                }
            }

            ptx
        }
    })
}
