//! Arithmetic PTX operations.
//!
//! Phase 1 instructions: [`Add`](ArithOp::Add), [`Mad`](ArithOp::Mad),
//! [`MulWide`](ArithOp::MulWide).
//!
//! Phase 2 (Sprint 2.2): [`Sub`](ArithOp::Sub), [`Mul`](ArithOp::Mul),
//! [`Div`](ArithOp::Div), [`Rem`](ArithOp::Rem), [`Neg`](ArithOp::Neg).

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
    /// Fused multiply-add: `fma.rn{ty} dst, a, b, c;`
    ///
    /// Computes `dst = (a * b) + c` with a single rounding step (IEEE
    /// round-to-nearest). Float-only (F32, F64). More precise and faster
    /// than separate mul + add — essential for matmul inner loops.
    ///
    /// Example: `fma.rn.f32 %f0, %f1, %f2, %f3;`
    Fma {
        /// Destination register.
        dst: Register,
        /// First multiplicand.
        a: Operand,
        /// Second multiplicand.
        b: Operand,
        /// Addend.
        c: Operand,
        /// PTX type suffix (F32 or F64).
        ty: PtxType,
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
    /// Typed subtraction: `sub{ty} dst, lhs, rhs;`
    ///
    /// Example: `sub.s32 %r0, %r1, %r2;`
    Sub {
        /// Destination register.
        dst: Register,
        /// Left-hand source operand.
        lhs: Operand,
        /// Right-hand source operand.
        rhs: Operand,
        /// PTX type suffix.
        ty: PtxType,
    },
    /// Same-width multiply: `mul.lo{ty}` (integer) or `mul{ty}` (float).
    ///
    /// For integers, PTX requires `.lo` to select the low N bits of the
    /// 2N-bit product. For floats, no mode is needed.
    ///
    /// Distinct from [`MulWide`](ArithOp::MulWide) which produces a
    /// 2N-bit result from N-bit inputs (used for address calculation).
    ///
    /// Examples:
    /// - `mul.lo.s32 %r0, %r1, %r2;` — integer multiply
    /// - `mul.f32 %f0, %f1, %f2;` — float multiply
    Mul {
        /// Destination register (same width as inputs).
        dst: Register,
        /// Left-hand source operand.
        lhs: Operand,
        /// Right-hand source operand.
        rhs: Operand,
        /// PTX type suffix.
        ty: PtxType,
    },
    /// Typed division: `div{ty} dst, lhs, rhs;`
    ///
    /// Example: `div.f32 %f0, %f1, %f2;`
    Div {
        /// Destination register.
        dst: Register,
        /// Left-hand source operand (dividend).
        lhs: Operand,
        /// Right-hand source operand (divisor).
        rhs: Operand,
        /// PTX type suffix.
        ty: PtxType,
    },
    /// Integer remainder: `rem{ty} dst, lhs, rhs;`
    ///
    /// Only valid for integer types in PTX ISA (`.s32`, `.u32`, `.s64`, `.u64`).
    ///
    /// Example: `rem.u32 %r0, %r1, %r2;`
    Rem {
        /// Destination register.
        dst: Register,
        /// Left-hand source operand (dividend).
        lhs: Operand,
        /// Right-hand source operand (divisor).
        rhs: Operand,
        /// PTX type suffix (must be integer).
        ty: PtxType,
    },
    /// Unary negation: `neg{ty} dst, src;`
    ///
    /// Example: `neg.f32 %f0, %f1;`
    Neg {
        /// Destination register.
        dst: Register,
        /// Source operand.
        src: Operand,
        /// PTX type suffix.
        ty: PtxType,
    },
    /// Absolute value: `abs{ty} dst, src;`
    ///
    /// Example: `abs.f32 %f0, %f1;`
    Abs {
        /// Destination register.
        dst: Register,
        /// Source operand.
        src: Operand,
        /// PTX type suffix.
        ty: PtxType,
    },
    /// Minimum: `min{ty} dst, lhs, rhs;`
    ///
    /// Example: `min.f32 %f0, %f1, %f2;`
    Min {
        /// Destination register.
        dst: Register,
        /// Left-hand source operand.
        lhs: Operand,
        /// Right-hand source operand.
        rhs: Operand,
        /// PTX type suffix.
        ty: PtxType,
    },
    /// Maximum: `max{ty} dst, lhs, rhs;`
    ///
    /// Example: `max.f32 %f0, %f1, %f2;`
    Max {
        /// Destination register.
        dst: Register,
        /// Left-hand source operand.
        lhs: Operand,
        /// Right-hand source operand.
        rhs: Operand,
        /// PTX type suffix.
        ty: PtxType,
    },
    /// Approximate square root: `sqrt.approx.f32 dst, src;`
    ///
    /// f32 only in Phase 2. Uses `.approx` for fast-math (matches GPU behavior).
    Sqrt {
        /// Destination register.
        dst: Register,
        /// Source operand.
        src: Operand,
    },
    /// Approximate base-2 exponential: `ex2.approx.f32 dst, src;`
    ///
    /// Used to synthesize `exp(x) = 2^(x * log2(e))`.
    Ex2 {
        /// Destination register.
        dst: Register,
        /// Source operand.
        src: Operand,
    },
    /// Approximate base-2 logarithm: `lg2.approx.f32 dst, src;`
    ///
    /// Used to synthesize `ln(x) = log2(x) * ln(2)`.
    Lg2 {
        /// Destination register.
        dst: Register,
        /// Source operand.
        src: Operand,
    },
    /// Approximate reciprocal: `rcp.approx.f32 dst, src;`
    Rcp {
        /// Destination register.
        dst: Register,
        /// Source operand.
        src: Operand,
    },
    /// Select-if-predicate: `selp{ty} dst, a, b, p;` → `dst = p ? a : b`.
    ///
    /// Branchless conditional assignment — no warp divergence. Useful
    /// for applying per-lane predicated writes (e.g. Sprint 6.6b's
    /// causal mask emits `setp.gt.u32 %p, %col, %row; selp.f32 %score,
    /// -3.4e38, %score, %p`). PTX ISA §9.7.8.1.
    ///
    /// Example: `selp.f32 %f5, %f3, %f4, %p1;`
    Selp {
        /// Destination register.
        dst: Register,
        /// Operand selected when predicate is true.
        a: Operand,
        /// Operand selected when predicate is false.
        b: Operand,
        /// Predicate register.
        pred: Register,
        /// PTX type suffix.
        ty: PtxType,
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
            ArithOp::Fma { dst, a, b, c, ty } => {
                let mnemonic = format!("fma.rn{}", ty.ptx_suffix());
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
            ArithOp::Sub { dst, lhs, rhs, ty } => {
                let mnemonic = format!("sub{}", ty.ptx_suffix());
                w.instruction(&mnemonic, &[dst as &dyn fmt::Display, lhs, rhs])
            }
            ArithOp::Mul { dst, lhs, rhs, ty } => {
                // Integer multiply needs .lo (low N bits of 2N product).
                // Float multiply has no mode suffix.
                let mnemonic = match ty {
                    PtxType::F32 | PtxType::F64 => format!("mul{}", ty.ptx_suffix()),
                    _ => format!("mul.lo{}", ty.ptx_suffix()),
                };
                w.instruction(&mnemonic, &[dst as &dyn fmt::Display, lhs, rhs])
            }
            ArithOp::Div { dst, lhs, rhs, ty } => {
                // Float division requires .approx (fast-math) or .rn (IEEE).
                // Integer division has no modifier.
                let mnemonic = match ty {
                    PtxType::F32 => "div.approx.f32".to_string(),
                    PtxType::F64 => "div.rn.f64".to_string(),
                    _ => format!("div{}", ty.ptx_suffix()),
                };
                w.instruction(&mnemonic, &[dst as &dyn fmt::Display, lhs, rhs])
            }
            ArithOp::Rem { dst, lhs, rhs, ty } => {
                let mnemonic = format!("rem{}", ty.ptx_suffix());
                w.instruction(&mnemonic, &[dst as &dyn fmt::Display, lhs, rhs])
            }
            ArithOp::Neg { dst, src, ty } => {
                let mnemonic = format!("neg{}", ty.ptx_suffix());
                w.instruction(&mnemonic, &[dst as &dyn fmt::Display, src])
            }
            ArithOp::Abs { dst, src, ty } => {
                let mnemonic = format!("abs{}", ty.ptx_suffix());
                w.instruction(&mnemonic, &[dst as &dyn fmt::Display, src])
            }
            ArithOp::Min { dst, lhs, rhs, ty } => {
                let mnemonic = format!("min{}", ty.ptx_suffix());
                w.instruction(&mnemonic, &[dst as &dyn fmt::Display, lhs, rhs])
            }
            ArithOp::Max { dst, lhs, rhs, ty } => {
                let mnemonic = format!("max{}", ty.ptx_suffix());
                w.instruction(&mnemonic, &[dst as &dyn fmt::Display, lhs, rhs])
            }
            ArithOp::Sqrt { dst, src } => {
                w.instruction("sqrt.approx.f32", &[dst as &dyn fmt::Display, src])
            }
            ArithOp::Ex2 { dst, src } => {
                w.instruction("ex2.approx.f32", &[dst as &dyn fmt::Display, src])
            }
            ArithOp::Lg2 { dst, src } => {
                w.instruction("lg2.approx.f32", &[dst as &dyn fmt::Display, src])
            }
            ArithOp::Rcp { dst, src } => {
                w.instruction("rcp.approx.f32", &[dst as &dyn fmt::Display, src])
            }
            ArithOp::Selp {
                dst,
                a,
                b,
                pred,
                ty,
            } => {
                let mnemonic = format!("selp{}", ty.ptx_suffix());
                w.instruction(&mnemonic, &[dst as &dyn fmt::Display, a, b, pred])
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
    fn emit_fma_f32() {
        let mut w = PtxWriter::new();
        w.indent();
        let op = ArithOp::Fma {
            dst: reg(RegKind::F, 0, PtxType::F32),
            a: Operand::Reg(reg(RegKind::F, 1, PtxType::F32)),
            b: Operand::Reg(reg(RegKind::F, 2, PtxType::F32)),
            c: Operand::Reg(reg(RegKind::F, 3, PtxType::F32)),
            ty: PtxType::F32,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    fma.rn.f32 %f0, %f1, %f2, %f3;\n");
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

    // --- Sprint 2.2: Sub, Mul, Div, Rem, Neg ---

    #[test]
    fn emit_sub_s32() {
        let mut w = PtxWriter::new();
        w.indent();
        let op = ArithOp::Sub {
            dst: reg(RegKind::R, 0, PtxType::S32),
            lhs: Operand::Reg(reg(RegKind::R, 1, PtxType::S32)),
            rhs: Operand::Reg(reg(RegKind::R, 2, PtxType::S32)),
            ty: PtxType::S32,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    sub.s32 %r0, %r1, %r2;\n");
    }

    #[test]
    fn emit_sub_f32() {
        let mut w = PtxWriter::new();
        w.indent();
        let op = ArithOp::Sub {
            dst: reg(RegKind::F, 0, PtxType::F32),
            lhs: Operand::Reg(reg(RegKind::F, 1, PtxType::F32)),
            rhs: Operand::Reg(reg(RegKind::F, 2, PtxType::F32)),
            ty: PtxType::F32,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    sub.f32 %f0, %f1, %f2;\n");
    }

    #[test]
    fn emit_mul_lo_s32() {
        // Integer multiply uses mul.lo (low 32 bits of 64-bit product)
        let mut w = PtxWriter::new();
        w.indent();
        let op = ArithOp::Mul {
            dst: reg(RegKind::R, 0, PtxType::S32),
            lhs: Operand::Reg(reg(RegKind::R, 1, PtxType::S32)),
            rhs: Operand::Reg(reg(RegKind::R, 2, PtxType::S32)),
            ty: PtxType::S32,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    mul.lo.s32 %r0, %r1, %r2;\n");
    }

    #[test]
    fn emit_mul_f32() {
        // Float multiply has no .lo mode
        let mut w = PtxWriter::new();
        w.indent();
        let op = ArithOp::Mul {
            dst: reg(RegKind::F, 0, PtxType::F32),
            lhs: Operand::Reg(reg(RegKind::F, 1, PtxType::F32)),
            rhs: Operand::Reg(reg(RegKind::F, 2, PtxType::F32)),
            ty: PtxType::F32,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    mul.f32 %f0, %f1, %f2;\n");
    }

    #[test]
    fn emit_mul_lo_u32_with_immediate() {
        let mut w = PtxWriter::new();
        w.indent();
        let op = ArithOp::Mul {
            dst: reg(RegKind::R, 0, PtxType::U32),
            lhs: Operand::Reg(reg(RegKind::R, 1, PtxType::U32)),
            rhs: Operand::ImmU32(3),
            ty: PtxType::U32,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    mul.lo.u32 %r0, %r1, 3;\n");
    }

    #[test]
    fn emit_div_f32() {
        let mut w = PtxWriter::new();
        w.indent();
        let op = ArithOp::Div {
            dst: reg(RegKind::F, 0, PtxType::F32),
            lhs: Operand::Reg(reg(RegKind::F, 1, PtxType::F32)),
            rhs: Operand::Reg(reg(RegKind::F, 2, PtxType::F32)),
            ty: PtxType::F32,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    div.approx.f32 %f0, %f1, %f2;\n");
    }

    #[test]
    fn emit_div_s32() {
        let mut w = PtxWriter::new();
        w.indent();
        let op = ArithOp::Div {
            dst: reg(RegKind::R, 0, PtxType::S32),
            lhs: Operand::Reg(reg(RegKind::R, 1, PtxType::S32)),
            rhs: Operand::Reg(reg(RegKind::R, 2, PtxType::S32)),
            ty: PtxType::S32,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    div.s32 %r0, %r1, %r2;\n");
    }

    #[test]
    fn emit_rem_u32() {
        let mut w = PtxWriter::new();
        w.indent();
        let op = ArithOp::Rem {
            dst: reg(RegKind::R, 0, PtxType::U32),
            lhs: Operand::Reg(reg(RegKind::R, 1, PtxType::U32)),
            rhs: Operand::Reg(reg(RegKind::R, 2, PtxType::U32)),
            ty: PtxType::U32,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    rem.u32 %r0, %r1, %r2;\n");
    }

    #[test]
    fn emit_neg_f32() {
        let mut w = PtxWriter::new();
        w.indent();
        let op = ArithOp::Neg {
            dst: reg(RegKind::F, 0, PtxType::F32),
            src: Operand::Reg(reg(RegKind::F, 1, PtxType::F32)),
            ty: PtxType::F32,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    neg.f32 %f0, %f1;\n");
    }

    #[test]
    fn emit_neg_s32() {
        let mut w = PtxWriter::new();
        w.indent();
        let op = ArithOp::Neg {
            dst: reg(RegKind::R, 0, PtxType::S32),
            src: Operand::Reg(reg(RegKind::R, 1, PtxType::S32)),
            ty: PtxType::S32,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    neg.s32 %r0, %r1;\n");
    }

    #[test]
    fn sub_via_ptx_instruction() {
        use crate::ir::PtxInstruction;
        let mut w = PtxWriter::new();
        w.indent();
        let instr = PtxInstruction::Arith(ArithOp::Sub {
            dst: reg(RegKind::R, 0, PtxType::S32),
            lhs: Operand::Reg(reg(RegKind::R, 1, PtxType::S32)),
            rhs: Operand::ImmI32(1),
            ty: PtxType::S32,
        });
        instr.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    sub.s32 %r0, %r1, 1;\n");
    }

    // --- Sprint 2.5: Math operations ---

    #[test]
    fn emit_abs_f32() {
        let mut w = PtxWriter::new();
        w.indent();
        ArithOp::Abs {
            dst: reg(RegKind::F, 0, PtxType::F32),
            src: Operand::Reg(reg(RegKind::F, 1, PtxType::F32)),
            ty: PtxType::F32,
        }
        .emit(&mut w)
        .unwrap();
        assert_eq!(w.finish(), "    abs.f32 %f0, %f1;\n");
    }

    #[test]
    fn emit_min_f32() {
        let mut w = PtxWriter::new();
        w.indent();
        ArithOp::Min {
            dst: reg(RegKind::F, 0, PtxType::F32),
            lhs: Operand::Reg(reg(RegKind::F, 1, PtxType::F32)),
            rhs: Operand::Reg(reg(RegKind::F, 2, PtxType::F32)),
            ty: PtxType::F32,
        }
        .emit(&mut w)
        .unwrap();
        assert_eq!(w.finish(), "    min.f32 %f0, %f1, %f2;\n");
    }

    #[test]
    fn emit_max_f32() {
        let mut w = PtxWriter::new();
        w.indent();
        ArithOp::Max {
            dst: reg(RegKind::F, 0, PtxType::F32),
            lhs: Operand::Reg(reg(RegKind::F, 1, PtxType::F32)),
            rhs: Operand::Reg(reg(RegKind::F, 2, PtxType::F32)),
            ty: PtxType::F32,
        }
        .emit(&mut w)
        .unwrap();
        assert_eq!(w.finish(), "    max.f32 %f0, %f1, %f2;\n");
    }

    #[test]
    fn emit_sqrt_approx_f32() {
        let mut w = PtxWriter::new();
        w.indent();
        ArithOp::Sqrt {
            dst: reg(RegKind::F, 0, PtxType::F32),
            src: Operand::Reg(reg(RegKind::F, 1, PtxType::F32)),
        }
        .emit(&mut w)
        .unwrap();
        assert_eq!(w.finish(), "    sqrt.approx.f32 %f0, %f1;\n");
    }

    #[test]
    fn emit_ex2_approx_f32() {
        let mut w = PtxWriter::new();
        w.indent();
        ArithOp::Ex2 {
            dst: reg(RegKind::F, 0, PtxType::F32),
            src: Operand::Reg(reg(RegKind::F, 1, PtxType::F32)),
        }
        .emit(&mut w)
        .unwrap();
        assert_eq!(w.finish(), "    ex2.approx.f32 %f0, %f1;\n");
    }

    #[test]
    fn emit_lg2_approx_f32() {
        let mut w = PtxWriter::new();
        w.indent();
        ArithOp::Lg2 {
            dst: reg(RegKind::F, 0, PtxType::F32),
            src: Operand::Reg(reg(RegKind::F, 1, PtxType::F32)),
        }
        .emit(&mut w)
        .unwrap();
        assert_eq!(w.finish(), "    lg2.approx.f32 %f0, %f1;\n");
    }

    #[test]
    fn emit_rcp_approx_f32() {
        let mut w = PtxWriter::new();
        w.indent();
        ArithOp::Rcp {
            dst: reg(RegKind::F, 0, PtxType::F32),
            src: Operand::Reg(reg(RegKind::F, 1, PtxType::F32)),
        }
        .emit(&mut w)
        .unwrap();
        assert_eq!(w.finish(), "    rcp.approx.f32 %f0, %f1;\n");
    }

    #[test]
    fn emit_selp_f32() {
        // Branchless select: `selp.f32 dst, a, b, p` → if p then a else b.
        // Used by Sprint 6.6b causal attention mask:
        //   setp.gt.u32 %p, %col, %row;
        //   selp.f32 %score, -3.4e38, %score, %p;
        // Operand::ImmF32 emits decimal form (same convention as
        // matmul_tc's FragmentC zero-init); ptxas accepts it.
        let mut w = PtxWriter::new();
        w.indent();
        ArithOp::Selp {
            dst: reg(RegKind::F, 5, PtxType::F32),
            a: Operand::ImmF32(1.0),
            b: Operand::Reg(reg(RegKind::F, 4, PtxType::F32)),
            pred: reg(RegKind::P, 1, PtxType::Pred),
            ty: PtxType::F32,
        }
        .emit(&mut w)
        .unwrap();
        assert_eq!(w.finish(), "    selp.f32 %f5, 1.0, %f4, %p1;\n");
    }
}
