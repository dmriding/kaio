//! PTX text emission — the `Emit` trait and `PtxWriter`.
//!
//! Every IR node implements [`Emit`], which writes PTX text through
//! [`PtxWriter`]'s formatting helpers. The writer handles indentation,
//! operand formatting, and label generation.

pub mod emit_trait;
pub mod writer;

pub use emit_trait::Emit;
pub use writer::PtxWriter;
