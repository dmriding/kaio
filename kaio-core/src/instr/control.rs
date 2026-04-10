//! Control flow PTX operations.
//!
//! Contains comparison-to-predicate ([`SetP`](ControlOp::SetP)),
//! branching ([`BraPred`](ControlOp::BraPred), [`Bra`](ControlOp::Bra)),
//! and [`Ret`](ControlOp::Ret). Phase 3 will add `bar.sync` and
//! `shfl.sync` when shared memory / warp-level ops are needed.

use std::fmt;

use crate::emit::{Emit, PtxWriter};
use crate::ir::{Operand, Register};
use crate::types::PtxType;

/// Comparison operator for `setp` instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    /// Equal (`==`)
    Eq,
    /// Not equal (`!=`)
    Ne,
    /// Less than (`<`)
    Lt,
    /// Less than or equal (`<=`)
    Le,
    /// Greater than (`>`)
    Gt,
    /// Greater than or equal (`>=`)
    Ge,
}

impl CmpOp {
    /// PTX comparison operator string (e.g. `"ge"`, `"lt"`).
    pub fn ptx_str(&self) -> &'static str {
        match self {
            Self::Eq => "eq",
            Self::Ne => "ne",
            Self::Lt => "lt",
            Self::Le => "le",
            Self::Gt => "gt",
            Self::Ge => "ge",
        }
    }
}

/// Control flow PTX instruction variants.
#[derive(Debug, Clone)]
pub enum ControlOp {
    /// Set predicate from comparison: `setp.{cmp_op}{ty} pred, lhs, rhs;`
    ///
    /// Compares `lhs` and `rhs` and writes the result to a predicate register.
    /// Example: `setp.ge.u32 %p1, %r1, %r2;`
    SetP {
        /// Destination predicate register.
        dst: Register,
        /// Comparison operation.
        cmp_op: CmpOp,
        /// Left-hand operand (register or immediate).
        lhs: Operand,
        /// Right-hand operand (register or immediate).
        rhs: Operand,
        /// PTX type for the comparison.
        ty: PtxType,
    },
    /// Predicated branch: `@{pred} bra {target};` or `@!{pred} bra {target};`
    ///
    /// Branches to `target` label if `pred` is true (or false when negated).
    /// Uses `PtxWriter::line()` instead of `instruction()` because the
    /// `@pred mnemonic target;` format doesn't fit the comma-separated
    /// operand pattern.
    ///
    /// Examples:
    /// - `@%p1 bra $L__BB0_2;` — branch if pred is true
    /// - `@!%p1 bra IF_END_0;` — branch if pred is false (Phase 2 if/else)
    BraPred {
        /// Predicate register to test.
        pred: Register,
        /// Label name to branch to.
        target: String,
        /// When `true`, negate the predicate (`@!pred`). Deferred from
        /// Sprint 1.4, needed for Phase 2 if/else lowering where `setp`
        /// matches the source comparison and `@!pred bra` skips the
        /// then-block when the condition is false.
        negate: bool,
    },
    /// Unconditional branch: `bra {target};`
    ///
    /// Not used in `vector_add` but included for Phase 3 loop support.
    Bra {
        /// Label name to branch to.
        target: String,
    },
    /// Return from kernel: `ret;`
    Ret,
}

impl Emit for ControlOp {
    fn emit(&self, w: &mut PtxWriter) -> fmt::Result {
        match self {
            ControlOp::SetP {
                dst,
                cmp_op,
                lhs,
                rhs,
                ty,
            } => {
                let mnemonic = format!("setp.{}{}", cmp_op.ptx_str(), ty.ptx_suffix());
                w.instruction(&mnemonic, &[dst as &dyn fmt::Display, lhs, rhs])
            }
            ControlOp::BraPred {
                pred,
                target,
                negate,
            } => {
                let neg = if *negate { "!" } else { "" };
                w.line(&format!("@{neg}{pred} bra {target};"))
            }
            ControlOp::Bra { target } => w.instruction("bra", &[&target as &dyn fmt::Display]),
            ControlOp::Ret => w.instruction("ret", &[]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::RegKind;

    fn reg(kind: RegKind, index: u32, ptx_type: PtxType) -> Register {
        Register {
            kind,
            index,
            ptx_type,
        }
    }

    // --- nvcc golden comparisons ---

    #[test]
    fn emit_setp_ge_u32() {
        // nvcc line 36: setp.ge.u32 %p1, %r1, %r2
        let mut w = PtxWriter::new();
        w.indent();
        let op = ControlOp::SetP {
            dst: reg(RegKind::P, 1, PtxType::Pred),
            cmp_op: CmpOp::Ge,
            lhs: Operand::Reg(reg(RegKind::R, 1, PtxType::U32)),
            rhs: Operand::Reg(reg(RegKind::R, 2, PtxType::U32)),
            ty: PtxType::U32,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    setp.ge.u32 %p1, %r1, %r2;\n");
    }

    #[test]
    fn emit_bra_pred() {
        // nvcc line 37: @%p1 bra $L__BB0_2
        // nvcc uses tab whitespace; we use space — both valid PTX
        let mut w = PtxWriter::new();
        w.indent();
        let op = ControlOp::BraPred {
            pred: reg(RegKind::P, 1, PtxType::Pred),
            target: "$L__BB0_2".to_string(),
            negate: false,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    @%p1 bra $L__BB0_2;\n");
    }

    #[test]
    fn emit_bra_pred_negated() {
        // Phase 2 if/else: @!%p1 bra IF_END_0 — skip then-block when false
        let mut w = PtxWriter::new();
        w.indent();
        let op = ControlOp::BraPred {
            pred: reg(RegKind::P, 1, PtxType::Pred),
            target: "IF_END_0".to_string(),
            negate: true,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    @!%p1 bra IF_END_0;\n");
    }

    #[test]
    fn emit_ret() {
        // nvcc line 52: ret
        let mut w = PtxWriter::new();
        w.indent();
        ControlOp::Ret.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    ret;\n");
    }

    #[test]
    fn emit_bra_unconditional() {
        let mut w = PtxWriter::new();
        w.indent();
        let op = ControlOp::Bra {
            target: "LOOP".to_string(),
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    bra LOOP;\n");
    }

    // --- Dispatch and unit tests ---

    #[test]
    fn control_via_ptx_instruction() {
        use crate::ir::PtxInstruction;

        let mut w = PtxWriter::new();
        w.indent();
        let instr = PtxInstruction::Control(ControlOp::Ret);
        instr.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    ret;\n");
    }

    #[test]
    fn cmp_op_all_variants() {
        assert_eq!(CmpOp::Eq.ptx_str(), "eq");
        assert_eq!(CmpOp::Ne.ptx_str(), "ne");
        assert_eq!(CmpOp::Lt.ptx_str(), "lt");
        assert_eq!(CmpOp::Le.ptx_str(), "le");
        assert_eq!(CmpOp::Gt.ptx_str(), "gt");
        assert_eq!(CmpOp::Ge.ptx_str(), "ge");
    }
}
