//! Control flow and synchronization PTX operations.
//!
//! Contains comparison-to-predicate ([`SetP`](ControlOp::SetP)),
//! branching ([`BraPred`](ControlOp::BraPred), [`Bra`](ControlOp::Bra)),
//! [`Ret`](ControlOp::Ret), barrier synchronization
//! ([`BarSync`](ControlOp::BarSync)), and warp shuffle operations
//! ([`ShflSyncDown`](ControlOp::ShflSyncDown),
//! [`ShflSyncUp`](ControlOp::ShflSyncUp),
//! [`ShflSyncBfly`](ControlOp::ShflSyncBfly)).

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
    /// Set predicate from comparison ANDed with a source predicate:
    /// `setp.{cmp_op}.and{ty} pred, lhs, rhs, src_pred;`
    ///
    /// Computes `pred = (lhs CmpOp rhs) AND src_pred` in one instruction.
    /// Used for compact edge-tile bounds checking — combines a row check
    /// with an existing col-check predicate without a separate `and.pred`.
    /// Sprint 6.7 (multi-warp matmul_tc edge tiles) is the first user.
    /// Example: `setp.lt.and.u32 %p3, %r5, %r10, %p2;`
    SetPAnd {
        /// Destination predicate register.
        dst: Register,
        /// Comparison operation applied to `lhs`/`rhs`.
        cmp_op: CmpOp,
        /// Left-hand operand of the comparison.
        lhs: Operand,
        /// Right-hand operand of the comparison.
        rhs: Operand,
        /// PTX type for the comparison.
        ty: PtxType,
        /// Source predicate AND'd with the comparison result.
        src_pred: Register,
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
    /// Block-level barrier synchronization: `bar.sync {barrier_id};`
    ///
    /// All threads in the block must reach this instruction before any
    /// can proceed. Barrier 0 is the conventional default.
    /// Example: `bar.sync 0;`
    BarSync {
        /// Barrier identifier (0 is conventional for single-barrier use).
        barrier_id: u32,
    },
    /// Warp shuffle down: `shfl.sync.down.b32 dst, src, delta, c, membermask;`
    ///
    /// Each thread reads from the thread `delta` lanes below it within
    /// the warp. The `c` operand packs clamp width (see PTX ISA 8.7 S9.7.8).
    /// Example: `shfl.sync.down.b32 %r2, %r1, 1, 31, 0xFFFFFFFF;`
    ShflSyncDown {
        /// Destination register.
        dst: Register,
        /// Source register (value to share).
        src: Register,
        /// Delta (offset) — how many lanes down.
        delta: Operand,
        /// Pre-packed clamp/width value (encoding is caller's responsibility).
        c: u32,
        /// Member mask (0xFFFFFFFF = full warp).
        mask: u32,
    },
    /// Warp shuffle up: `shfl.sync.up.b32 dst, src, delta, c, membermask;`
    ///
    /// Each thread reads from the thread `delta` lanes above it.
    ShflSyncUp {
        /// Destination register.
        dst: Register,
        /// Source register.
        src: Register,
        /// Delta (offset) — how many lanes up.
        delta: Operand,
        /// Pre-packed clamp/width value.
        c: u32,
        /// Member mask.
        mask: u32,
    },
    /// Warp shuffle butterfly (XOR): `shfl.sync.bfly.b32 dst, src, lane_mask, c, membermask;`
    ///
    /// Each thread reads from the thread at `lane XOR lane_mask`.
    ShflSyncBfly {
        /// Destination register.
        dst: Register,
        /// Source register.
        src: Register,
        /// Lane mask for XOR operation.
        lane_mask: Operand,
        /// Pre-packed clamp/width value.
        c: u32,
        /// Member mask.
        mask: u32,
    },
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
            ControlOp::SetPAnd {
                dst,
                cmp_op,
                lhs,
                rhs,
                ty,
                src_pred,
            } => {
                let mnemonic = format!("setp.{}.and{}", cmp_op.ptx_str(), ty.ptx_suffix());
                w.instruction(&mnemonic, &[dst as &dyn fmt::Display, lhs, rhs, src_pred])
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
            ControlOp::BarSync { barrier_id } => w.line(&format!("bar.sync {barrier_id};")),
            ControlOp::ShflSyncDown {
                dst,
                src,
                delta,
                c,
                mask,
            } => w.line(&format!(
                "shfl.sync.down.b32 {dst}, {src}, {delta}, {c}, 0x{mask:08X};"
            )),
            ControlOp::ShflSyncUp {
                dst,
                src,
                delta,
                c,
                mask,
            } => w.line(&format!(
                "shfl.sync.up.b32 {dst}, {src}, {delta}, {c}, 0x{mask:08X};"
            )),
            ControlOp::ShflSyncBfly {
                dst,
                src,
                lane_mask,
                c,
                mask,
            } => w.line(&format!(
                "shfl.sync.bfly.b32 {dst}, {src}, {lane_mask}, {c}, 0x{mask:08X};"
            )),
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
    fn emit_setp_and_lt_u32() {
        // Sprint 6.7 edge-tile: setp.lt.and.u32 %p3, %r5, %r10, %p2
        let mut w = PtxWriter::new();
        w.indent();
        let op = ControlOp::SetPAnd {
            dst: reg(RegKind::P, 3, PtxType::Pred),
            cmp_op: CmpOp::Lt,
            lhs: Operand::Reg(reg(RegKind::R, 5, PtxType::U32)),
            rhs: Operand::Reg(reg(RegKind::R, 10, PtxType::U32)),
            ty: PtxType::U32,
            src_pred: reg(RegKind::P, 2, PtxType::Pred),
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    setp.lt.and.u32 %p3, %r5, %r10, %p2;\n");
    }

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

    // --- Phase 3: Barrier + Shuffle ---

    #[test]
    fn emit_bar_sync() {
        let mut w = PtxWriter::new();
        w.indent();
        ControlOp::BarSync { barrier_id: 0 }.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    bar.sync 0;\n");
    }

    #[test]
    fn emit_shfl_sync_down() {
        let mut w = PtxWriter::new();
        w.indent();
        let op = ControlOp::ShflSyncDown {
            dst: reg(RegKind::R, 2, PtxType::U32),
            src: reg(RegKind::R, 1, PtxType::U32),
            delta: Operand::ImmU32(1),
            c: 31,
            mask: 0xFFFFFFFF,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(
            w.finish(),
            "    shfl.sync.down.b32 %r2, %r1, 1, 31, 0xFFFFFFFF;\n"
        );
    }

    #[test]
    fn emit_shfl_sync_up() {
        let mut w = PtxWriter::new();
        w.indent();
        let op = ControlOp::ShflSyncUp {
            dst: reg(RegKind::R, 2, PtxType::U32),
            src: reg(RegKind::R, 1, PtxType::U32),
            delta: Operand::ImmU32(1),
            c: 0,
            mask: 0xFFFFFFFF,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(
            w.finish(),
            "    shfl.sync.up.b32 %r2, %r1, 1, 0, 0xFFFFFFFF;\n"
        );
    }

    #[test]
    fn emit_shfl_sync_bfly() {
        let mut w = PtxWriter::new();
        w.indent();
        let op = ControlOp::ShflSyncBfly {
            dst: reg(RegKind::R, 2, PtxType::U32),
            src: reg(RegKind::R, 1, PtxType::U32),
            lane_mask: Operand::ImmU32(1),
            c: 31,
            mask: 0xFFFFFFFF,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(
            w.finish(),
            "    shfl.sync.bfly.b32 %r2, %r1, 1, 31, 0xFFFFFFFF;\n"
        );
    }

    #[test]
    fn shfl_sync_down_with_register_delta() {
        let mut w = PtxWriter::new();
        w.indent();
        let op = ControlOp::ShflSyncDown {
            dst: reg(RegKind::R, 3, PtxType::U32),
            src: reg(RegKind::R, 0, PtxType::U32),
            delta: Operand::Reg(reg(RegKind::R, 4, PtxType::U32)),
            c: 31,
            mask: 0xFFFFFFFF,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(
            w.finish(),
            "    shfl.sync.down.b32 %r3, %r0, %r4, 31, 0xFFFFFFFF;\n"
        );
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
