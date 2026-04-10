//! Memory PTX operations.
//!
//! Currently uninhabited (empty enum). Sprint 1.3 will add variants
//! (`LdParam`, `LdGlobal`, `StGlobal`, `CvtaToGlobal`). When that happens:
//!
//! 1. Add the variants to [`MemoryOp`]
//! 2. Update the [`Emit`](crate::emit::Emit) impl from `match *self {}` to real match arms
//! 3. Update any [`PtxInstruction`](crate::ir::PtxInstruction) match sites

/// Memory PTX instruction variants.
#[derive(Debug, Clone)]
pub enum MemoryOp {}
