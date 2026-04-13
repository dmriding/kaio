//! The PTX instruction enum — the central IR node for kernel bodies.

use super::operand::Operand;
use super::register::Register;
use crate::instr::{ArithOp, ControlOp, MemoryOp, TensorCoreOp};
use crate::types::PtxType;

/// A single PTX instruction in a kernel body.
///
/// The `Arith`, `Memory`, and `Control` variants wrap category-specific
/// enums defined in [`crate::instr`]. Those enums are currently uninhabited
/// (empty) — sprints 1.2, 1.3, and 1.4 will add real variants. Until then,
/// these `PtxInstruction` variants exist but cannot be constructed.
#[derive(Debug, Clone)]
pub enum PtxInstruction {
    /// Arithmetic operation (populated in Sprint 1.2).
    Arith(ArithOp),
    /// Memory operation (populated in Sprint 1.3).
    Memory(MemoryOp),
    /// Control flow operation (populated in Sprint 1.4).
    Control(ControlOp),
    /// Warp-collective tensor-core operation (populated in Sprint 6.2).
    ///
    /// Currently supports `mma.sync.m16n8k16` for fp16/bf16 inputs with
    /// fp32 accumulation (Ampere+, SM 8.0+). See [`crate::instr::tensor_core`]
    /// and [`crate::fragment`].
    TensorCore(TensorCoreOp),
    /// Register-to-register or immediate-to-register move.
    ///
    /// Also used for special register reads:
    /// `mov.u32 %r0, %tid.x;`
    Mov {
        /// Destination register.
        dst: Register,
        /// Source operand (register, immediate, or special register).
        src: Operand,
        /// PTX type of the move.
        ty: PtxType,
    },
    /// Type conversion between registers.
    ///
    /// `cvt.dst_ty.src_ty %dst, %src;`
    Cvt {
        /// Destination register (must match `dst_ty`'s reg kind).
        dst: Register,
        /// Source register (must match `src_ty`'s reg kind).
        src: Register,
        /// Destination PTX type.
        dst_ty: PtxType,
        /// Source PTX type.
        src_ty: PtxType,
    },
    /// A label target for branches.
    Label(String),
    /// A comment in the emitted PTX (for debugging / readability).
    Comment(String),
}
