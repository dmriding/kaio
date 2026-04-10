//! PTX instruction category enums.
//!
//! Each category is a separate module with an enum that gets populated
//! in its corresponding sprint:
//! - [`ArithOp`] — Sprint 1.2
//! - [`MemoryOp`] — Sprint 1.3
//! - [`ControlOp`] — Sprint 1.4
//!
//! Sprint 1.4 will also add a `special` module with helper functions
//! for thread/block/grid index register access.

pub mod arith;
pub mod control;
pub mod memory;

pub use arith::{ArithOp, MadMode};
pub use control::ControlOp;
pub use memory::MemoryOp;
