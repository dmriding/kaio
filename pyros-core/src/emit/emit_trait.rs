//! The `Emit` trait and empty implementations for all IR nodes.
//!
//! Sprint 1.5 will fill in the real emission logic. Until then, every
//! impl returns `Ok(())` (or uses an unreachable empty match for
//! uninhabited types).

use std::fmt;

use super::writer::PtxWriter;
use crate::instr::{ArithOp, ControlOp, MemoryOp};
use crate::ir::{PtxInstruction, PtxKernel, PtxModule};

/// Trait for emitting PTX text from an IR node.
///
/// Every IR type implements this. The writer handles indentation and
/// formatting; each `Emit` impl is responsible for producing the
/// content of its node type.
pub trait Emit {
    /// Write this node's PTX representation to the writer.
    fn emit(&self, w: &mut PtxWriter) -> fmt::Result;
}

// --- Empty impls (Sprint 1.5 fills these in) ---

impl Emit for PtxModule {
    fn emit(&self, _w: &mut PtxWriter) -> fmt::Result {
        // Sprint 1.5: emit .version, .target, .address_size, then each kernel
        Ok(())
    }
}

impl Emit for PtxKernel {
    fn emit(&self, _w: &mut PtxWriter) -> fmt::Result {
        // Sprint 1.5: emit .visible .entry, params, .reg declarations, body
        Ok(())
    }
}

impl Emit for PtxInstruction {
    fn emit(&self, _w: &mut PtxWriter) -> fmt::Result {
        // Sprint 1.5: dispatch to variant-specific emission
        Ok(())
    }
}

// Uninhabited enums — exhaustive empty match, unreachable but compiles.
// When Sprint 1.2/1.3/1.4 adds variants, these matches will break
// (intentionally) and force the implementer to write real emission logic.

impl Emit for ArithOp {
    fn emit(&self, _w: &mut PtxWriter) -> fmt::Result {
        match *self {}
    }
}

impl Emit for MemoryOp {
    fn emit(&self, _w: &mut PtxWriter) -> fmt::Result {
        match *self {}
    }
}

impl Emit for ControlOp {
    fn emit(&self, _w: &mut PtxWriter) -> fmt::Result {
        match *self {}
    }
}
