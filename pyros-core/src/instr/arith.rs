//! Arithmetic PTX operations.
//!
//! Currently uninhabited (empty enum). Sprint 1.2 will add variants
//! (`Add`, `Mad`, `MulWide`). When that happens:
//!
//! 1. Add the variants to [`ArithOp`]
//! 2. Update the [`Emit`](crate::emit::Emit) impl from `match *self {}` to real match arms
//! 3. Update any [`PtxInstruction`](crate::ir::PtxInstruction) match sites
//!    that had unreachable empty arms

/// Arithmetic PTX instruction variants.
#[derive(Debug, Clone)]
pub enum ArithOp {}
