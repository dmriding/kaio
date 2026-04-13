//! Generate the `build_module()` function body.

use proc_macro2::TokenStream;
use quote::quote;

use crate::kernel_ir::KernelSignature;
use crate::kernel_ir::stmt::KernelStmt;
use crate::lower;
use crate::lower::LoweringContext;

/// Generate the `build_module(sm: &str) -> PtxModule` function.
///
/// Produces a function that, at runtime:
/// 1. Creates a `RegisterAllocator` and `PtxKernel`
/// 2. Declares and loads all parameters
/// 3. Executes the lowered kernel body
/// 4. Adds `Ret`, finalizes registers
/// 5. Wraps the `PtxKernel` in a `PtxModule` targeting the caller's SM
///    (or `KAIO_SM_TARGET` env var if set, for debugging/cross-compile)
/// 6. Optionally dumps PTX if `KAIO_DUMP_PTX` is set
///
/// Sprint 6.10 D1a migration: returns `PtxModule` instead of a PTX text
/// string. The launch wrapper passes `sm_XX` derived from
/// `device.info().compute_capability`, so the driver call path hits
/// `PtxModule::validate()` before ptxas — closing the trust-boundary gap
/// flagged in tech_debt.md:129.
pub fn generate_build_module(sig: &KernelSignature, body: &[KernelStmt]) -> syn::Result<TokenStream> {
    let kernel_name = &sig.name;
    let mut ctx = LoweringContext::new();
    let total_threads = sig.config.block_size * sig.config.block_size_y.unwrap_or(1);
    ctx.block_size = Some(total_threads);
    if sig.config.block_size_y.is_some() {
        ctx.block_size_x = Some(sig.config.block_size);
        ctx.block_size_y = sig.config.block_size_y;
    }

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
        fn build_module(sm: &str) -> kaio::core::ir::PtxModule {
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

            let _kaio_annotate = std::env::var("KAIO_PTX_ANNOTATE").is_ok();

            let mut alloc = RegisterAllocator::new();
            let mut kernel = PtxKernel::new(#kernel_name);

            #param_tokens
            #body_tokens

            kernel.push(PtxInstruction::Control(ControlOp::Ret));
            kernel.set_registers(alloc.into_allocated());

            if std::env::var("KAIO_PTX_STATS").is_ok() {
                let _s = kernel.stats();
                eprintln!("KAIO stats: kernel '{}' (PTX structure, not runtime profile)", #kernel_name);
                eprintln!("  Instructions: {} total", _s.total_instructions);
                eprintln!("  Arithmetic:   {} fma, {} other", _s.fma, _s.arith_other);
                eprintln!("  Memory:       {} ld.global, {} st.global, {} ld.shared, {} st.shared",
                    _s.ld_global, _s.st_global, _s.ld_shared, _s.st_shared);
                eprintln!("  Control:      {} bar.sync, {} branches, {} setp, {} mov, {} cvt",
                    _s.bar_sync, _s.branches, _s.setp, _s.mov, _s.cvt);
                eprintln!("  Registers:    {} r32, {} r64, {} f32, {} f64, {} pred, {} f16, {} bf16  (PTX-level, not final HW allocation)",
                    _s.registers_r, _s.registers_rd, _s.registers_f, _s.registers_fd, _s.registers_p, _s.registers_h, _s.registers_hb);
                eprintln!("  Shared mem:   {} bytes", _s.shared_bytes);
            }

            // Caller passes `sm` from device.info().compute_capability.
            // KAIO_SM_TARGET env var still overrides (debugging / cross-compile).
            let sm_target = std::env::var("KAIO_SM_TARGET")
                .unwrap_or_else(|_| sm.to_string());
            let mut module = PtxModule::new(&sm_target);
            module.add_kernel(kernel);

            #shared_mem_diagnostic

            if std::env::var("KAIO_DUMP_PTX").is_ok() {
                // Emit PTX text for dump purposes; the runtime load_module path
                // will re-emit on its own, which is cheap.
                let mut w = PtxWriter::new();
                module.emit(&mut w).unwrap();
                let ptx = w.finish();
                let dump_dir = std::env::var("OUT_DIR")
                    .unwrap_or_else(|_| ".".to_string());
                let dump_path = format!("{}/{}.ptx", dump_dir, #kernel_name);
                match std::fs::write(&dump_path, &ptx) {
                    Ok(()) => eprintln!("KAIO: wrote {}", dump_path),
                    Err(e) => eprintln!("KAIO: failed to write {}: {}", dump_path, e),
                }
            }

            module
        }
    })
}
