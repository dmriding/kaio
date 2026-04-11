//! Lower array indexing to address calculation + memory load/store.
//!
//! **Global memory** (pointer parameters):
//! `cvta.to.global → mul.wide (byte offset, 64-bit) → add.s64 → ld/st.global`
//!
//! **Shared memory** (`shared_mem!` buffers):
//! `mov base (shared name) → mul.lo (byte offset, 32-bit) → add.u32 → ld/st.shared`

use proc_macro2::{Ident, TokenStream};
use quote::quote;

use crate::kernel_ir::KernelType;

use super::LoweringContext;

/// Compute the global memory address for `array[index]`.
///
/// Pattern:
/// 1. `cvta.to.global.u64` on the pointer register (cached per param in `ctx.global_addrs`)
/// 2. `mul.wide.u32` to compute byte offset: `index * sizeof(T)`
/// 3. `add.s64` to compute final address: `global_base + byte_offset`
///
/// Returns `(address_register_ident, token_stream)`.
#[allow(dead_code)] // Called from lower_index_read/write
fn compute_address(
    ctx: &mut LoweringContext,
    array_name: &str,
    array_reg: &Ident,
    index_reg: &Ident,
    elem_ty: &KernelType,
) -> (Ident, TokenStream) {
    let mut tokens = TokenStream::new();

    // 1. CvtaToGlobal — once per pointer, cached in ctx.global_addrs
    let global_reg = if let Some(cached) = ctx.global_addrs.get(array_name) {
        cached.clone()
    } else {
        let reg = ctx.fresh_reg();
        let cvta_tokens = quote! {
            let #reg = alloc.alloc(PtxType::U64);
            kernel.push(PtxInstruction::Memory(MemoryOp::CvtaToGlobal {
                dst: #reg,
                src: #array_reg,
            }));
        };
        tokens.extend(cvta_tokens);
        ctx.global_addrs.insert(array_name.to_string(), reg.clone());
        reg
    };

    // 2. MulWide — byte offset = index * sizeof(T)
    let size = elem_ty.size_bytes() as u32;
    let offset_reg = ctx.fresh_reg();
    let offset_tokens = quote! {
        let #offset_reg = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
            dst: #offset_reg,
            lhs: Operand::Reg(#index_reg),
            rhs: Operand::ImmU32(#size),
            src_ty: PtxType::U32,
        }));
    };
    tokens.extend(offset_tokens);

    // 3. Add — final address = global_base + byte_offset
    let addr_reg = ctx.fresh_reg();
    let addr_tokens = quote! {
        let #addr_reg = alloc.alloc(PtxType::S64);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: #addr_reg,
            lhs: Operand::Reg(#global_reg),
            rhs: Operand::Reg(#offset_reg),
            ty: PtxType::S64,
        }));
    };
    tokens.extend(addr_tokens);

    (addr_reg, tokens)
}

/// Lower an array index read: `a[idx]` → address calc + `ld.global`.
///
/// Returns `(result_register_ident, token_stream)`.
#[allow(dead_code)] // Called from lower/mod.rs::lower_expr
pub fn lower_index_read(
    ctx: &mut LoweringContext,
    array_name: &str,
    array_reg: &Ident,
    index_reg: &Ident,
    elem_ty: &KernelType,
) -> (Ident, TokenStream) {
    let (addr_reg, addr_tokens) = compute_address(ctx, array_name, array_reg, index_reg, elem_ty);

    let ptx_ty = ctx.ptx_type_tokens(elem_ty);
    let result_reg = ctx.fresh_reg();
    let load_tokens = quote! {
        let #result_reg = alloc.alloc(PtxType::#ptx_ty);
        kernel.push(PtxInstruction::Memory(MemoryOp::LdGlobal {
            dst: #result_reg,
            addr: #addr_reg,
            ty: PtxType::#ptx_ty,
        }));
    };

    (result_reg, quote! { #addr_tokens #load_tokens })
}

/// Lower an array index write: `out[idx] = val` → address calc + `st.global`.
///
/// Returns the token stream (no result register — stores don't produce values).
#[allow(dead_code)] // Called from lower/mod.rs::lower_stmt
pub fn lower_index_write(
    ctx: &mut LoweringContext,
    array_name: &str,
    array_reg: &Ident,
    index_reg: &Ident,
    value_reg: &Ident,
    elem_ty: &KernelType,
) -> TokenStream {
    let (addr_reg, addr_tokens) = compute_address(ctx, array_name, array_reg, index_reg, elem_ty);

    let ptx_ty = ctx.ptx_type_tokens(elem_ty);
    let store_tokens = quote! {
        kernel.push(PtxInstruction::Memory(MemoryOp::StGlobal {
            addr: #addr_reg,
            src: #value_reg,
            ty: PtxType::#ptx_ty,
        }));
    };

    quote! { #addr_tokens #store_tokens }
}

/// Compute a shared memory address for `array_name[index]`.
///
/// **Shared memory uses 32-bit addressing (U32)** — this is a deliberate
/// PTX/shared-space choice. `.shared` is per-SM SRAM with a 32-bit address
/// space, distinct from 64-bit global memory.
///
/// Uses named-symbol base addressing:
/// 1. `mov.u32 base, <symbol>` — PTX resolves to the allocation's base
/// 2. `mul.u32 byte_off, index, sizeof(T)` — byte offset within allocation
/// 3. `add.u32 addr, base, byte_off` — final address
///
/// This pattern works correctly for any number of shared allocations.
/// The base load + add are emitted unconditionally (even when there's only
/// one allocation where base = 0). This is an intentional simplicity
/// tradeoff — may be optimized via base hoisting in Phase 4.6+.
fn compute_shared_address(
    ctx: &mut LoweringContext,
    array_name: &str,
    index_reg: &Ident,
    elem_ty: &KernelType,
) -> (Ident, TokenStream) {
    let size = elem_ty.size_bytes() as u32;
    let ptx_name = array_name.to_string();
    let base_reg = ctx.fresh_reg();
    let byte_off_reg = ctx.fresh_reg();
    let addr_reg = ctx.fresh_reg();

    let tokens = quote! {
        // Load base address of shared allocation (32-bit shared space)
        let #base_reg = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Mov {
            dst: #base_reg,
            src: Operand::SharedAddr(#ptx_name.to_string()),
            ty: PtxType::U32,
        });
        // Byte offset within allocation
        let #byte_off_reg = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Mul {
            dst: #byte_off_reg,
            lhs: Operand::Reg(#index_reg),
            rhs: Operand::ImmU32(#size),
            ty: PtxType::U32,
        }));
        // Final address = base + byte_offset
        let #addr_reg = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: #addr_reg,
            lhs: Operand::Reg(#base_reg),
            rhs: Operand::Reg(#byte_off_reg),
            ty: PtxType::U32,
        }));
    };

    (addr_reg, tokens)
}

/// Lower a shared memory index read: `sdata[idx]` → address calc + `ld.shared`.
pub fn lower_shared_index_read(
    ctx: &mut LoweringContext,
    array_name: &str,
    index_reg: &Ident,
    elem_ty: &KernelType,
) -> (Ident, TokenStream) {
    let (addr_reg, addr_tokens) = compute_shared_address(ctx, array_name, index_reg, elem_ty);

    let ptx_ty = ctx.ptx_type_tokens(elem_ty);
    let result_reg = ctx.fresh_reg();
    let load_tokens = quote! {
        let #result_reg = alloc.alloc(PtxType::#ptx_ty);
        kernel.push(PtxInstruction::Memory(MemoryOp::LdShared {
            dst: #result_reg,
            addr: #addr_reg,
            ty: PtxType::#ptx_ty,
        }));
    };

    (result_reg, quote! { #addr_tokens #load_tokens })
}

/// Lower a shared memory index write: `sdata[idx] = val` → address calc + `st.shared`.
pub fn lower_shared_index_write(
    ctx: &mut LoweringContext,
    array_name: &str,
    index_reg: &Ident,
    value_reg: &Ident,
    elem_ty: &KernelType,
) -> TokenStream {
    let (addr_reg, addr_tokens) = compute_shared_address(ctx, array_name, index_reg, elem_ty);

    let ptx_ty = ctx.ptx_type_tokens(elem_ty);
    let store_tokens = quote! {
        kernel.push(PtxInstruction::Memory(MemoryOp::StShared {
            addr: #addr_reg,
            src: #value_reg,
            ty: PtxType::#ptx_ty,
        }));
    };

    quote! { #addr_tokens #store_tokens }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proc_macro2::Span;

    // --- Standalone address calculation tests (risk mitigation) ---

    #[test]
    fn address_calc_f32_sizeof_4() {
        let mut ctx = LoweringContext::new();
        let array_reg = Ident::new("_kaio_r0", Span::call_site());
        let index_reg = Ident::new("_kaio_r1", Span::call_site());

        let (_addr, tokens) =
            compute_address(&mut ctx, "a", &array_reg, &index_reg, &KernelType::F32);
        let code = tokens.to_string();

        // Must use ImmU32(4u32) for f32 sizeof
        assert!(code.contains("ImmU32 (4u32)"), "f32 sizeof must be 4");
        assert!(code.contains("MulWide"));
        assert!(code.contains("CvtaToGlobal"));
        assert!(code.contains("PtxType :: S64")); // address is s64
    }

    #[test]
    fn address_calc_f64_sizeof_8() {
        let mut ctx = LoweringContext::new();
        let array_reg = Ident::new("_kaio_r0", Span::call_site());
        let index_reg = Ident::new("_kaio_r1", Span::call_site());

        let (_addr, tokens) =
            compute_address(&mut ctx, "a", &array_reg, &index_reg, &KernelType::F64);
        let code = tokens.to_string();

        assert!(code.contains("ImmU32 (8u32)"), "f64 sizeof must be 8");
    }

    #[test]
    fn address_calc_u32_sizeof_4() {
        let mut ctx = LoweringContext::new();
        let array_reg = Ident::new("_kaio_r0", Span::call_site());
        let index_reg = Ident::new("_kaio_r1", Span::call_site());

        let (_addr, tokens) =
            compute_address(&mut ctx, "a", &array_reg, &index_reg, &KernelType::U32);
        let code = tokens.to_string();

        assert!(code.contains("ImmU32 (4u32)"), "u32 sizeof must be 4");
    }

    // --- Index read/write tests ---

    #[test]
    fn lower_index_read_f32() {
        let mut ctx = LoweringContext::new();
        let array_reg = Ident::new("_kaio_r0", Span::call_site());
        let index_reg = Ident::new("_kaio_r1", Span::call_site());

        let (result, tokens) =
            lower_index_read(&mut ctx, "a", &array_reg, &index_reg, &KernelType::F32);
        let code = tokens.to_string();

        assert!(result.to_string().starts_with("_kaio_r"));
        assert!(code.contains("CvtaToGlobal"));
        assert!(code.contains("MulWide"));
        assert!(code.contains("ArithOp :: Add"));
        assert!(code.contains("MemoryOp :: LdGlobal"));
        assert!(code.contains("PtxType :: F32"));
    }

    #[test]
    fn lower_index_write_f32() {
        let mut ctx = LoweringContext::new();
        let array_reg = Ident::new("_kaio_r0", Span::call_site());
        let index_reg = Ident::new("_kaio_r1", Span::call_site());
        let value_reg = Ident::new("_kaio_r2", Span::call_site());

        let tokens = lower_index_write(
            &mut ctx,
            "out",
            &array_reg,
            &index_reg,
            &value_reg,
            &KernelType::F32,
        );
        let code = tokens.to_string();

        assert!(code.contains("CvtaToGlobal"));
        assert!(code.contains("MulWide"));
        assert!(code.contains("MemoryOp :: StGlobal"));
        assert!(code.contains("PtxType :: F32"));
    }

    #[test]
    fn cvta_cached_across_accesses() {
        let mut ctx = LoweringContext::new();
        let array_reg = Ident::new("_kaio_r0", Span::call_site());
        let idx1 = Ident::new("_kaio_r1", Span::call_site());
        let idx2 = Ident::new("_kaio_r2", Span::call_site());

        // First access: should emit CvtaToGlobal
        let (_r1, tokens1) = lower_index_read(&mut ctx, "a", &array_reg, &idx1, &KernelType::F32);
        let code1 = tokens1.to_string();
        assert!(code1.contains("CvtaToGlobal"), "first access should cvta");

        // Second access to same array: should NOT emit another CvtaToGlobal
        let (_r2, tokens2) = lower_index_read(&mut ctx, "a", &array_reg, &idx2, &KernelType::F32);
        let code2 = tokens2.to_string();
        assert!(
            !code2.contains("CvtaToGlobal"),
            "second access should reuse cached cvta"
        );
        // But should still have MulWide + Add + LdGlobal
        assert!(code2.contains("MulWide"));
        assert!(code2.contains("LdGlobal"));
    }

    // --- Shared memory addressing tests ---

    #[test]
    fn shared_address_calc_f32() {
        let mut ctx = LoweringContext::new();
        let index_reg = Ident::new("_kaio_r0", Span::call_site());

        let (_addr, tokens) =
            compute_shared_address(&mut ctx, "sdata", &index_reg, &KernelType::F32);
        let code = tokens.to_string();

        // Should use 32-bit addressing (U32), NOT 64-bit
        assert!(
            code.contains("ArithOp :: Mul"),
            "should compute byte offset"
        );
        assert!(code.contains("ImmU32 (4u32)"), "f32 sizeof must be 4");
        assert!(code.contains("PtxType :: U32"), "should use 32-bit regs");
        // Should NOT contain 64-bit operations
        assert!(
            !code.contains("MulWide"),
            "shared memory uses 32-bit mul, not wide"
        );
        assert!(!code.contains("CvtaToGlobal"), "shared memory has no cvta");
    }

    #[test]
    fn lower_shared_index_read_f32() {
        let mut ctx = LoweringContext::new();
        let index_reg = Ident::new("_kaio_r0", Span::call_site());

        let (result, tokens) =
            lower_shared_index_read(&mut ctx, "sdata", &index_reg, &KernelType::F32);
        let code = tokens.to_string();

        assert!(result.to_string().starts_with("_kaio_r"));
        assert!(
            code.contains("LdShared"),
            "should use ld.shared, not ld.global"
        );
        assert!(code.contains("PtxType :: F32"));
        assert!(!code.contains("LdGlobal"), "must not use global load");
    }

    #[test]
    fn lower_shared_index_write_f32() {
        let mut ctx = LoweringContext::new();
        let index_reg = Ident::new("_kaio_r0", Span::call_site());
        let value_reg = Ident::new("_kaio_r1", Span::call_site());

        let tokens =
            lower_shared_index_write(&mut ctx, "sdata", &index_reg, &value_reg, &KernelType::F32);
        let code = tokens.to_string();

        assert!(
            code.contains("StShared"),
            "should use st.shared, not st.global"
        );
        assert!(code.contains("PtxType :: F32"));
        assert!(!code.contains("StGlobal"), "must not use global store");
    }
}
