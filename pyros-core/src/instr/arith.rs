//! Arithmetic PTX operations.
//!
//! Contains the minimum instruction set needed for `vector_add`:
//! [`Add`](ArithOp::Add), [`Mad`](ArithOp::Mad), and
//! [`MulWide`](ArithOp::MulWide). Additional arithmetic operations
//! (Sub, Mul, Div, Fma, etc.) will be added when kernels need them.

use std::fmt;

use crate::emit::{Emit, PtxWriter};
use crate::ir::{Operand, Register};
use crate::types::PtxType;

/// Mode for the `mad` (multiply-add) instruction.
///
/// Determines which bits of the intermediate product are used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MadMode {
    /// Low bits of the product (`mad.lo`).
    ///
    /// `mad.lo.s32 %r1, %r3, %r4, %r5` computes `%r1 = (%r3 * %r4)[31:0] + %r5`.
    Lo,
    // Hi — high bits of the product. Deferred until a kernel needs it.
    // Wide — full 64-bit product from 32-bit inputs, then add. Deferred.
}

impl MadMode {
    /// PTX modifier string (e.g. `"lo"`).
    pub fn ptx_str(&self) -> &'static str {
        match self {
            Self::Lo => "lo",
        }
    }
}

/// Arithmetic PTX instruction variants.
///
/// Operand conventions:
/// - `dst` is always a [`Register`] (destination).
/// - Source operands are [`Operand`] (register or immediate).
/// - `ty` / `src_ty` is the PTX type suffix in the mnemonic.
#[derive(Debug, Clone)]
pub enum ArithOp {
    /// Typed addition: `add{ty} dst, lhs, rhs;`
    ///
    /// Examples from nvcc output:
    /// - `add.f32 %f3, %f2, %f1;` — float add
    /// - `add.s64 %rd6, %rd4, %rd5;` — address arithmetic
    Add {
        /// Destination register.
        dst: Register,
        /// Left-hand source operand.
        lhs: Operand,
        /// Right-hand source operand.
        rhs: Operand,
        /// PTX type suffix (`.f32`, `.s64`, etc.).
        ty: PtxType,
    },
    /// Multiply-add: `mad.{mode}{ty} dst, a, b, c;`
    ///
    /// Computes `dst = (a * b) + c` with the mode selecting which
    /// bits of the intermediate product to use.
    ///
    /// Example: `mad.lo.s32 %r1, %r3, %r4, %r5;`
    Mad {
        /// Destination register.
        dst: Register,
        /// First multiplicand.
        a: Operand,
        /// Second multiplicand.
        b: Operand,
        /// Addend.
        c: Operand,
        /// PTX type suffix.
        ty: PtxType,
        /// Product bit selection mode.
        mode: MadMode,
    },
    /// Widening multiply: `mul.wide{src_ty} dst, lhs, rhs;`
    ///
    /// Multiplies two N-bit values and produces a 2N-bit result.
    /// `src_ty` is the input type; the destination register is
    /// twice the width.
    ///
    /// Example: `mul.wide.u32 %rd5, %r1, 4;` — 32-bit → 64-bit
    MulWide {
        /// Destination register (2× width of inputs).
        dst: Register,
        /// Left-hand source operand (N-bit).
        lhs: Operand,
        /// Right-hand source operand (N-bit, can be immediate).
        rhs: Operand,
        /// Input type (determines the `.wide` suffix).
        src_ty: PtxType,
    },
}

impl Emit for ArithOp {
    fn emit(&self, w: &mut PtxWriter) -> fmt::Result {
        match self {
            ArithOp::Add { dst, lhs, rhs, ty } => {
                let mnemonic = format!("add{}", ty.ptx_suffix());
                w.instruction(&mnemonic, &[dst as &dyn fmt::Display, lhs, rhs])
            }
            ArithOp::Mad {
                dst,
                a,
                b,
                c,
                ty,
                mode,
            } => {
                let mnemonic = format!("mad.{}{}", mode.ptx_str(), ty.ptx_suffix());
                w.instruction(&mnemonic, &[dst as &dyn fmt::Display, a, b, c])
            }
            ArithOp::MulWide {
                dst,
                lhs,
                rhs,
                src_ty,
            } => {
                let mnemonic = format!("mul.wide{}", src_ty.ptx_suffix());
                w.instruction(&mnemonic, &[dst as &dyn fmt::Display, lhs, rhs])
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
    fn emit_add_f32() {
        // nvcc line 46: add.f32 %f3, %f2, %f1
        let mut w = PtxWriter::new();
        w.indent();
        let op = ArithOp::Add {
            dst: reg(RegKind::F, 3, PtxType::F32),
            lhs: Operand::Reg(reg(RegKind::F, 2, PtxType::F32)),
            rhs: Operand::Reg(reg(RegKind::F, 1, PtxType::F32)),
            ty: PtxType::F32,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    add.f32 %f3, %f2, %f1;\n");
    }

    #[test]
    fn emit_add_s64() {
        // nvcc line 41: add.s64 %rd6, %rd4, %rd5
        let mut w = PtxWriter::new();
        w.indent();
        let op = ArithOp::Add {
            dst: reg(RegKind::Rd, 6, PtxType::S64),
            lhs: Operand::Reg(reg(RegKind::Rd, 4, PtxType::S64)),
            rhs: Operand::Reg(reg(RegKind::Rd, 5, PtxType::U64)),
            ty: PtxType::S64,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    add.s64 %rd6, %rd4, %rd5;\n");
    }

    #[test]
    fn emit_mad_lo_s32() {
        // nvcc line 35: mad.lo.s32 %r1, %r3, %r4, %r5
        let mut w = PtxWriter::new();
        w.indent();
        let op = ArithOp::Mad {
            dst: reg(RegKind::R, 1, PtxType::S32),
            a: Operand::Reg(reg(RegKind::R, 3, PtxType::S32)),
            b: Operand::Reg(reg(RegKind::R, 4, PtxType::S32)),
            c: Operand::Reg(reg(RegKind::R, 5, PtxType::S32)),
            ty: PtxType::S32,
            mode: MadMode::Lo,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    mad.lo.s32 %r1, %r3, %r4, %r5;\n");
    }

    #[test]
    fn emit_mul_wide_u32() {
        // nvcc line 40: mul.wide.u32 %rd5, %r1, 4
        let mut w = PtxWriter::new();
        w.indent();
        let op = ArithOp::MulWide {
            dst: reg(RegKind::Rd, 5, PtxType::U64),
            lhs: Operand::Reg(reg(RegKind::R, 1, PtxType::U32)),
            rhs: Operand::ImmU32(4),
            src_ty: PtxType::U32,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    mul.wide.u32 %rd5, %r1, 4;\n");
    }

    // --- Additional validation tests ---

    #[test]
    fn emit_add_with_immediate() {
        let mut w = PtxWriter::new();
        w.indent();
        let op = ArithOp::Add {
            dst: reg(RegKind::R, 0, PtxType::S32),
            lhs: Operand::Reg(reg(RegKind::R, 1, PtxType::S32)),
            rhs: Operand::ImmI32(1),
            ty: PtxType::S32,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    add.s32 %r0, %r1, 1;\n");
    }

    #[test]
    fn arith_via_ptx_instruction() {
        use crate::ir::PtxInstruction;

        let mut w = PtxWriter::new();
        w.indent();
        let instr = PtxInstruction::Arith(ArithOp::Add {
            dst: reg(RegKind::F, 0, PtxType::F32),
            lhs: Operand::Reg(reg(RegKind::F, 1, PtxType::F32)),
            rhs: Operand::Reg(reg(RegKind::F, 2, PtxType::F32)),
            ty: PtxType::F32,
        });
        instr.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    add.f32 %f0, %f1, %f2;\n");
    }

    #[test]
    fn mad_mode_ptx_str() {
        assert_eq!(MadMode::Lo.ptx_str(), "lo");
    }
}
