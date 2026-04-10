//! Control flow PTX operations.
//!
//! Currently uninhabited (empty enum). Sprint 1.4 will add variants
//! (`SetP`, `Bra`, `BraPred`, `Ret`). When that happens:
//!
//! 1. Add the variants to [`ControlOp`]
//! 2. Update the [`Emit`](crate::emit::Emit) impl from `match *self {}` to real match arms
//! 3. Update any [`PtxInstruction`](crate::ir::PtxInstruction) match sites

/// Control flow PTX instruction variants.
#[derive(Debug, Clone)]
pub enum ControlOp {}
