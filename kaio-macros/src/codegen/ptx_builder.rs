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

            let mut module = PtxModule::new("sm_89");
            module.add_kernel(kernel);

            let mut w = PtxWriter::new();
            module.emit(&mut w).unwrap();
            let ptx = w.finish();

            if std::env::var("KAIO_DUMP_PTX").is_ok() {
                eprintln!("=== KAIO PTX: {} ===\n{}", #kernel_name, ptx);
            }

            ptx
        }
    })
}
