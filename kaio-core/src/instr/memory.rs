//! Memory PTX operations.
//!
//! Contains load/store instructions for global and shared memory:
//! [`LdParam`](MemoryOp::LdParam), [`LdGlobal`](MemoryOp::LdGlobal),
//! [`StGlobal`](MemoryOp::StGlobal), [`LdShared`](MemoryOp::LdShared),
//! [`StShared`](MemoryOp::StShared), and
//! [`CvtaToGlobal`](MemoryOp::CvtaToGlobal).

use std::fmt;

use crate::emit::{Emit, PtxWriter};
use crate::ir::Register;
use crate::types::PtxType;

/// Memory PTX instruction variants.
///
/// Operand conventions:
/// - All addresses and values are [`Register`]s (not [`Operand`](crate::ir::Operand)).
///   You can't `ld.global` from an immediate address or `st.global` an immediate
///   value in PTX — those go through `mov` first.
/// - [`LdParam`](Self::LdParam) is the exception: it references a kernel parameter
///   by name (a `String`), not by register.
#[derive(Debug, Clone)]
pub enum MemoryOp {
    /// Load kernel parameter: `ld.param{ty} dst, [param_name];`
    ///
    /// References the parameter by name from the kernel signature.
    /// Example: `ld.param.u64 %rd1, [vector_add_param_0];`
    LdParam {
        /// Destination register.
        dst: Register,
        /// Parameter name from the kernel signature.
        param_name: String,
        /// PTX type of the parameter value.
        ty: PtxType,
    },
    /// Load from global memory: `ld.global{ty} dst, [addr];`
    ///
    /// The `addr` register holds the computed memory address.
    /// Example: `ld.global.f32 %f1, [%rd8];`
    LdGlobal {
        /// Destination register.
        dst: Register,
        /// Register holding the memory address.
        addr: Register,
        /// PTX type of the loaded value.
        ty: PtxType,
    },
    /// Store to global memory: `st.global{ty} [addr], src;`
    ///
    /// **Operand order is reversed in PTX** — address comes first,
    /// value second. This matches PTX convention but is opposite to
    /// loads and arithmetic where `dst` is first.
    ///
    /// Example: `st.global.f32 [%rd10], %f3;`
    StGlobal {
        /// Register holding the memory address.
        addr: Register,
        /// Source register (value to store).
        src: Register,
        /// PTX type of the stored value.
        ty: PtxType,
    },
    /// Load from shared memory: `ld.shared{ty} dst, [addr];`
    ///
    /// Shared memory is block-scoped SRAM. The `addr` register holds the
    /// offset into the declared shared allocation.
    /// Example: `ld.shared.f32 %f0, [%r0];`
    LdShared {
        /// Destination register.
        dst: Register,
        /// Register holding the shared memory offset.
        addr: Register,
        /// PTX type of the loaded value.
        ty: PtxType,
    },
    /// Store to shared memory: `st.shared{ty} [addr], src;`
    ///
    /// **Operand order is reversed in PTX** — address first, value second
    /// (same convention as [`StGlobal`](Self::StGlobal)).
    /// Example: `st.shared.f32 [%r0], %f1;`
    StShared {
        /// Register holding the shared memory offset.
        addr: Register,
        /// Source register (value to store).
        src: Register,
        /// PTX type of the stored value.
        ty: PtxType,
    },
    /// Convert generic address to global: `cvta.to.global.u64 dst, src;`
    ///
    /// Always `.u64` (64-bit address space, matching `.address_size 64`).
    /// Required because `ld.param` returns generic-space pointers —
    /// `ld.global` needs global-space addresses.
    CvtaToGlobal {
        /// Destination register (global-space address).
        dst: Register,
        /// Source register (generic-space address from `ld.param`).
        src: Register,
    },
}

impl Emit for MemoryOp {
    fn emit(&self, w: &mut PtxWriter) -> fmt::Result {
        match self {
            MemoryOp::LdParam {
                dst,
                param_name,
                ty,
            } => {
                let mnemonic = format!("ld.param{}", ty.ptx_suffix());
                let addr = format!("[{param_name}]");
                w.instruction(&mnemonic, &[dst as &dyn fmt::Display, &addr])
            }
            MemoryOp::LdGlobal { dst, addr, ty } => {
                let mnemonic = format!("ld.global{}", ty.ptx_suffix());
                let addr_str = format!("[{addr}]");
                w.instruction(&mnemonic, &[dst as &dyn fmt::Display, &addr_str])
            }
            MemoryOp::StGlobal { addr, src, ty } => {
                let mnemonic = format!("st.global{}", ty.ptx_suffix());
                let addr_str = format!("[{addr}]");
                // PTX store order: [address], source (reversed from load)
                w.instruction(&mnemonic, &[&addr_str as &dyn fmt::Display, src])
            }
            MemoryOp::LdShared { dst, addr, ty } => {
                let mnemonic = format!("ld.shared{}", ty.ptx_suffix());
                let addr_str = format!("[{addr}]");
                w.instruction(&mnemonic, &[dst as &dyn fmt::Display, &addr_str])
            }
            MemoryOp::StShared { addr, src, ty } => {
                let mnemonic = format!("st.shared{}", ty.ptx_suffix());
                let addr_str = format!("[{addr}]");
                w.instruction(&mnemonic, &[&addr_str as &dyn fmt::Display, src])
            }
            MemoryOp::CvtaToGlobal { dst, src } => {
                w.instruction("cvta.to.global.u64", &[dst as &dyn fmt::Display, src])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::RegKind;

    /// Helper to make a register without going through the allocator.
    fn reg(kind: RegKind, index: u32, ptx_type: PtxType) -> Register {
        Register {
            kind,
            index,
            ptx_type,
        }
    }

    // --- nvcc golden comparisons (byte-for-byte match against nvcc --ptx -arch=sm_89) ---

    #[test]
    fn emit_ld_param_u64() {
        // nvcc line 28: ld.param.u64 %rd1, [vector_add_param_0]
        let mut w = PtxWriter::new();
        w.indent();
        let op = MemoryOp::LdParam {
            dst: reg(RegKind::Rd, 1, PtxType::U64),
            param_name: "vector_add_param_0".to_string(),
            ty: PtxType::U64,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    ld.param.u64 %rd1, [vector_add_param_0];\n");
    }

    #[test]
    fn emit_ld_param_u32() {
        // nvcc line 31: ld.param.u32 %r2, [vector_add_param_3]
        let mut w = PtxWriter::new();
        w.indent();
        let op = MemoryOp::LdParam {
            dst: reg(RegKind::R, 2, PtxType::U32),
            param_name: "vector_add_param_3".to_string(),
            ty: PtxType::U32,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    ld.param.u32 %r2, [vector_add_param_3];\n");
    }

    #[test]
    fn emit_cvta_to_global() {
        // nvcc line 39: cvta.to.global.u64 %rd4, %rd1
        let mut w = PtxWriter::new();
        w.indent();
        let op = MemoryOp::CvtaToGlobal {
            dst: reg(RegKind::Rd, 4, PtxType::U64),
            src: reg(RegKind::Rd, 1, PtxType::U64),
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    cvta.to.global.u64 %rd4, %rd1;\n");
    }

    #[test]
    fn emit_ld_global_f32() {
        // nvcc line 44: ld.global.f32 %f1, [%rd8]
        let mut w = PtxWriter::new();
        w.indent();
        let op = MemoryOp::LdGlobal {
            dst: reg(RegKind::F, 1, PtxType::F32),
            addr: reg(RegKind::Rd, 8, PtxType::U64),
            ty: PtxType::F32,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    ld.global.f32 %f1, [%rd8];\n");
    }

    #[test]
    fn emit_st_global_f32() {
        // nvcc line 49: st.global.f32 [%rd10], %f3
        let mut w = PtxWriter::new();
        w.indent();
        let op = MemoryOp::StGlobal {
            addr: reg(RegKind::Rd, 10, PtxType::U64),
            src: reg(RegKind::F, 3, PtxType::F32),
            ty: PtxType::F32,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    st.global.f32 [%rd10], %f3;\n");
    }

    // --- Dispatch and ordering validation ---

    #[test]
    fn memory_via_ptx_instruction() {
        use crate::ir::PtxInstruction;

        let mut w = PtxWriter::new();
        w.indent();
        let instr = PtxInstruction::Memory(MemoryOp::LdGlobal {
            dst: reg(RegKind::F, 0, PtxType::F32),
            addr: reg(RegKind::Rd, 0, PtxType::U64),
            ty: PtxType::F32,
        });
        instr.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    ld.global.f32 %f0, [%rd0];\n");
    }

    // --- Shared memory ops ---

    #[test]
    fn emit_ld_shared_f32() {
        let mut w = PtxWriter::new();
        w.indent();
        let op = MemoryOp::LdShared {
            dst: reg(RegKind::F, 0, PtxType::F32),
            addr: reg(RegKind::R, 0, PtxType::U32),
            ty: PtxType::F32,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    ld.shared.f32 %f0, [%r0];\n");
    }

    #[test]
    fn emit_st_shared_f32() {
        let mut w = PtxWriter::new();
        w.indent();
        let op = MemoryOp::StShared {
            addr: reg(RegKind::R, 0, PtxType::U32),
            src: reg(RegKind::F, 1, PtxType::F32),
            ty: PtxType::F32,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    st.shared.f32 [%r0], %f1;\n");
    }

    // --- Half-precision load/store (Sprint 6.1) ---

    #[test]
    fn emit_ld_global_f16() {
        let mut w = PtxWriter::new();
        w.indent();
        let op = MemoryOp::LdGlobal {
            dst: reg(RegKind::H, 0, PtxType::F16),
            addr: reg(RegKind::Rd, 0, PtxType::U64),
            ty: PtxType::F16,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    ld.global.f16 %h0, [%rd0];\n");
    }

    #[test]
    fn emit_st_global_f16() {
        let mut w = PtxWriter::new();
        w.indent();
        let op = MemoryOp::StGlobal {
            addr: reg(RegKind::Rd, 0, PtxType::U64),
            src: reg(RegKind::H, 0, PtxType::F16),
            ty: PtxType::F16,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    st.global.f16 [%rd0], %h0;\n");
    }

    #[test]
    fn emit_ld_shared_bf16() {
        let mut w = PtxWriter::new();
        w.indent();
        let op = MemoryOp::LdShared {
            dst: reg(RegKind::Hb, 0, PtxType::BF16),
            addr: reg(RegKind::R, 0, PtxType::U32),
            ty: PtxType::BF16,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    ld.shared.bf16 %hb0, [%r0];\n");
    }

    #[test]
    fn st_global_operand_order() {
        // Verify store has [addr], src order — NOT src, [addr]
        let mut w = PtxWriter::new();
        w.indent();
        let op = MemoryOp::StGlobal {
            addr: reg(RegKind::Rd, 0, PtxType::U64),
            src: reg(RegKind::F, 0, PtxType::F32),
            ty: PtxType::F32,
        };
        op.emit(&mut w).unwrap();
        let output = w.finish();
        // [%rd0] must appear BEFORE %f0
        let addr_pos = output.find("[%rd0]").expect("address not found");
        let src_pos = output.find("%f0").expect("source not found");
        assert!(
            addr_pos < src_pos,
            "store operand order wrong: address must come before source in PTX"
        );
    }
}
