//! Arithmetic PTX operations.
//!
//! Phase 1 instructions: [`Add`](ArithOp::Add), [`Mad`](ArithOp::Mad),
//! [`MulWide`](ArithOp::MulWide).
//!
//! Phase 2 (Sprint 2.2): [`Sub`](ArithOp::Sub), [`Mul`](ArithOp::Mul),
//! [`Div`](ArithOp::Div), [`Rem`](ArithOp::Rem), [`Neg`](ArithOp::Neg).
//!
//! Sprint 7.0 bitwise: [`And`](ArithOp::And), [`Or`](ArithOp::Or),
//! [`Xor`](ArithOp::Xor), [`Shl`](ArithOp::Shl), [`Shr`](ArithOp::Shr),
//! [`Not`](ArithOp::Not). Shr preserves signed/unsigned distinction
//! (arithmetic vs logical right shift); all others are typeless (`.b{size}`
//! or `.pred`).

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
    /// Bitwise AND: `and{suffix} dst, lhs, rhs;` — suffix is `.b{size}` for
    /// integers or `.pred` for predicates (typeless on signedness).
    ///
    /// Examples:
    /// - `and.b32 %r0, %r1, %r2;` — bitwise AND on integer registers
    /// - `and.pred %p0, %p1, %p2;` — predicate AND (emitted by `!` context
    ///   dispatch and any future branch-free logical combinator)
    And {
        /// Destination register.
        dst: Register,
        /// Left-hand source operand.
        lhs: Operand,
        /// Right-hand source operand.
        rhs: Operand,
        /// Operand type — determines suffix via [`PtxType::reg_decl_type`].
        ty: PtxType,
    },
    /// Bitwise OR: `or{suffix} dst, lhs, rhs;` — suffix mirrors `And`.
    Or {
        /// Destination register.
        dst: Register,
        /// Left-hand source operand.
        lhs: Operand,
        /// Right-hand source operand.
        rhs: Operand,
        /// Operand type — determines suffix via [`PtxType::reg_decl_type`].
        ty: PtxType,
    },
    /// Bitwise XOR: `xor{suffix} dst, lhs, rhs;` — suffix mirrors `And`.
    Xor {
        /// Destination register.
        dst: Register,
        /// Left-hand source operand.
        lhs: Operand,
        /// Right-hand source operand.
        rhs: Operand,
        /// Operand type — determines suffix via [`PtxType::reg_decl_type`].
        ty: PtxType,
    },
    /// Left shift: `shl{suffix} dst, lhs, rhs;` — suffix is `.b{size}`.
    ///
    /// Left shift is bit-width-only; PTX does not distinguish signed vs
    /// unsigned (the bits are the same). The `rhs` shift count operand is
    /// always treated as `.u32` per PTX ISA §9.7.3 (caller must ensure the
    /// shift count register is 32-bit).
    Shl {
        /// Destination register.
        dst: Register,
        /// Value being shifted.
        lhs: Operand,
        /// Shift count (PTX requires this operand to be 32-bit).
        rhs: Operand,
        /// Operand type — determines suffix via [`PtxType::reg_decl_type`].
        ty: PtxType,
    },
    /// Right shift: `shr{suffix} dst, lhs, rhs;` — suffix is `.s{size}` or
    /// `.u{size}` per operand signedness (arithmetic vs logical shift).
    ///
    /// Rust's `>>` on `i32` is arithmetic (sign-extends), on `u32` is logical
    /// (zero-extends). The IR must preserve this distinction so quant
    /// dequantization (Phase 7.1+) on signed INT8 storage produces correct
    /// sign extension.
    ///
    /// The shift count `rhs` is treated as `.u32` per PTX ISA §9.7.3.
    Shr {
        /// Destination register.
        dst: Register,
        /// Value being shifted.
        lhs: Operand,
        /// Shift count (PTX requires this operand to be 32-bit).
        rhs: Operand,
        /// Operand type — S32/S64 emit arithmetic shift, U32/U64 emit logical.
        ty: PtxType,
    },
    /// Bitwise NOT: `not{suffix} dst, src;` — suffix is `.b{size}` or `.pred`.
    ///
    /// Context-dispatched from the macro's unary `!`: integer types → bitwise
    /// flip, `Pred` → logical inversion.
    Not {
        /// Destination register.
        dst: Register,
        /// Source operand.
        src: Operand,
        /// Operand type — determines suffix via [`PtxType::reg_decl_type`].
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
                // Float multiply (including half-precision) has no `.lo`
                // suffix — that modifier is integer-only. Half-precision
                // defaults to `.rn` rounding per PTX ISA; bare `mul.f16`
                // is equivalent to `mul.rn.f16`.
                let mnemonic = match ty {
                    PtxType::F16 | PtxType::BF16 | PtxType::F32 | PtxType::F64 => {
                        format!("mul{}", ty.ptx_suffix())
                    }
                    // Integer multiply needs .lo (low N bits of 2N product).
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
            ArithOp::And { dst, lhs, rhs, ty } => {
                // Typeless on signedness: use reg_decl_type for `.b{size}` / `.pred`.
                let mnemonic = format!("and{}", ty.reg_decl_type());
                w.instruction(&mnemonic, &[dst as &dyn fmt::Display, lhs, rhs])
            }
            ArithOp::Or { dst, lhs, rhs, ty } => {
                let mnemonic = format!("or{}", ty.reg_decl_type());
                w.instruction(&mnemonic, &[dst as &dyn fmt::Display, lhs, rhs])
            }
            ArithOp::Xor { dst, lhs, rhs, ty } => {
                let mnemonic = format!("xor{}", ty.reg_decl_type());
                w.instruction(&mnemonic, &[dst as &dyn fmt::Display, lhs, rhs])
            }
            ArithOp::Shl { dst, lhs, rhs, ty } => {
                // Left shift is typeless on signedness.
                let mnemonic = format!("shl{}", ty.reg_decl_type());
                w.instruction(&mnemonic, &[dst as &dyn fmt::Display, lhs, rhs])
            }
            ArithOp::Shr { dst, lhs, rhs, ty } => {
                // Right shift preserves signed/unsigned distinction:
                // ptx_suffix gives `.s32` / `.u32` / `.s64` / `.u64`.
                let mnemonic = format!("shr{}", ty.ptx_suffix());
                w.instruction(&mnemonic, &[dst as &dyn fmt::Display, lhs, rhs])
            }
            ArithOp::Not { dst, src, ty } => {
                let mnemonic = format!("not{}", ty.reg_decl_type());
                w.instruction(&mnemonic, &[dst as &dyn fmt::Display, src])
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
    fn emit_mul_f16() {
        // Half-precision multiply: `mul.f16`, no `.lo` (integer-only).
        // PTX defaults `mul.f16` to `.rn` rounding.
        let mut w = PtxWriter::new();
        w.indent();
        let op = ArithOp::Mul {
            dst: reg(RegKind::H, 0, PtxType::F16),
            lhs: Operand::Reg(reg(RegKind::H, 1, PtxType::F16)),
            rhs: Operand::Reg(reg(RegKind::H, 2, PtxType::F16)),
            ty: PtxType::F16,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    mul.f16 %h0, %h1, %h2;\n");
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

    // --- Sprint 7.0: Bitwise operators ---

    #[test]
    fn emit_and_b32() {
        let mut w = PtxWriter::new();
        w.indent();
        ArithOp::And {
            dst: reg(RegKind::R, 0, PtxType::U32),
            lhs: Operand::Reg(reg(RegKind::R, 1, PtxType::U32)),
            rhs: Operand::Reg(reg(RegKind::R, 2, PtxType::U32)),
            ty: PtxType::U32,
        }
        .emit(&mut w)
        .unwrap();
        assert_eq!(w.finish(), "    and.b32 %r0, %r1, %r2;\n");
    }

    #[test]
    fn emit_and_pred() {
        // Sprint 7.0 AD3: unary `!` on bool dispatches to ArithOp::Not{ty:Pred},
        // but the three-operand and.pred variant must also be emittable for
        // Sprint 7.x short-circuit optimization pass and any future branch-free
        // logical combinator. Lock the format now.
        let mut w = PtxWriter::new();
        w.indent();
        ArithOp::And {
            dst: reg(RegKind::P, 0, PtxType::Pred),
            lhs: Operand::Reg(reg(RegKind::P, 1, PtxType::Pred)),
            rhs: Operand::Reg(reg(RegKind::P, 2, PtxType::Pred)),
            ty: PtxType::Pred,
        }
        .emit(&mut w)
        .unwrap();
        assert_eq!(w.finish(), "    and.pred %p0, %p1, %p2;\n");
    }

    #[test]
    fn emit_or_b64() {
        let mut w = PtxWriter::new();
        w.indent();
        ArithOp::Or {
            dst: reg(RegKind::Rd, 0, PtxType::U64),
            lhs: Operand::Reg(reg(RegKind::Rd, 1, PtxType::U64)),
            rhs: Operand::Reg(reg(RegKind::Rd, 2, PtxType::U64)),
            ty: PtxType::U64,
        }
        .emit(&mut w)
        .unwrap();
        assert_eq!(w.finish(), "    or.b64 %rd0, %rd1, %rd2;\n");
    }

    #[test]
    fn emit_xor_b32_with_immediate() {
        let mut w = PtxWriter::new();
        w.indent();
        ArithOp::Xor {
            dst: reg(RegKind::R, 0, PtxType::U32),
            lhs: Operand::Reg(reg(RegKind::R, 1, PtxType::U32)),
            rhs: Operand::ImmU32(0xFF),
            ty: PtxType::U32,
        }
        .emit(&mut w)
        .unwrap();
        assert_eq!(w.finish(), "    xor.b32 %r0, %r1, 255;\n");
    }

    #[test]
    fn emit_shl_b32() {
        // Left shift is typeless on signedness per AD2 — reg_decl_type
        // collapses S32/U32 to .b32.
        let mut w = PtxWriter::new();
        w.indent();
        ArithOp::Shl {
            dst: reg(RegKind::R, 0, PtxType::U32),
            lhs: Operand::Reg(reg(RegKind::R, 1, PtxType::U32)),
            rhs: Operand::ImmU32(4),
            ty: PtxType::U32,
        }
        .emit(&mut w)
        .unwrap();
        assert_eq!(w.finish(), "    shl.b32 %r0, %r1, 4;\n");
    }

    #[test]
    fn emit_shl_b32_signed_still_typeless() {
        // Signed S32 input must still emit shl.b32 (not shl.s32) — PTX does
        // not have signed shl variants.
        let mut w = PtxWriter::new();
        w.indent();
        ArithOp::Shl {
            dst: reg(RegKind::R, 0, PtxType::S32),
            lhs: Operand::Reg(reg(RegKind::R, 1, PtxType::S32)),
            rhs: Operand::ImmU32(2),
            ty: PtxType::S32,
        }
        .emit(&mut w)
        .unwrap();
        assert_eq!(w.finish(), "    shl.b32 %r0, %r1, 2;\n");
    }

    #[test]
    fn emit_shr_u32_logical() {
        // AD2 direct canary: unsigned right shift → shr.u32 (logical, zero-extend).
        // Paired with emit_shr_s32_arithmetic — these two tests together verify
        // the signed/unsigned dispatch that quant INT8 dequant depends on.
        let mut w = PtxWriter::new();
        w.indent();
        ArithOp::Shr {
            dst: reg(RegKind::R, 0, PtxType::U32),
            lhs: Operand::Reg(reg(RegKind::R, 1, PtxType::U32)),
            rhs: Operand::ImmU32(1),
            ty: PtxType::U32,
        }
        .emit(&mut w)
        .unwrap();
        assert_eq!(w.finish(), "    shr.u32 %r0, %r1, 1;\n");
    }

    #[test]
    fn emit_shr_s32_arithmetic() {
        // AD2 direct canary: signed right shift → shr.s32 (arithmetic, sign-extend).
        // If this test ever emits shr.u32 or shr.b32, the IR-level half of the
        // signedness dispatch has regressed and Phase 7.1+ INT8 dequant will
        // silently zero-extend negative packed values.
        let mut w = PtxWriter::new();
        w.indent();
        ArithOp::Shr {
            dst: reg(RegKind::R, 0, PtxType::S32),
            lhs: Operand::Reg(reg(RegKind::R, 1, PtxType::S32)),
            rhs: Operand::ImmU32(1),
            ty: PtxType::S32,
        }
        .emit(&mut w)
        .unwrap();
        assert_eq!(w.finish(), "    shr.s32 %r0, %r1, 1;\n");
    }

    #[test]
    fn emit_shr_u64_logical() {
        let mut w = PtxWriter::new();
        w.indent();
        ArithOp::Shr {
            dst: reg(RegKind::Rd, 0, PtxType::U64),
            lhs: Operand::Reg(reg(RegKind::Rd, 1, PtxType::U64)),
            rhs: Operand::ImmU32(4),
            ty: PtxType::U64,
        }
        .emit(&mut w)
        .unwrap();
        assert_eq!(w.finish(), "    shr.u64 %rd0, %rd1, 4;\n");
    }

    #[test]
    fn emit_not_b32() {
        let mut w = PtxWriter::new();
        w.indent();
        ArithOp::Not {
            dst: reg(RegKind::R, 0, PtxType::U32),
            src: Operand::Reg(reg(RegKind::R, 1, PtxType::U32)),
            ty: PtxType::U32,
        }
        .emit(&mut w)
        .unwrap();
        assert_eq!(w.finish(), "    not.b32 %r0, %r1;\n");
    }

    #[test]
    fn emit_not_pred() {
        // Logical not on predicate — emitted when unary `!` is applied to a
        // bool-typed expression (e.g. `!(a < b)`).
        let mut w = PtxWriter::new();
        w.indent();
        ArithOp::Not {
            dst: reg(RegKind::P, 0, PtxType::Pred),
            src: Operand::Reg(reg(RegKind::P, 1, PtxType::Pred)),
            ty: PtxType::Pred,
        }
        .emit(&mut w)
        .unwrap();
        assert_eq!(w.finish(), "    not.pred %p0, %p1;\n");
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
