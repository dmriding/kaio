//! # kaio-core
//!
//! PTX intermediate representation and code emission for the KAIO GPU
//! kernel authoring framework. This crate provides the Layer 1 foundation:
//! Rust types modelling PTX programs, instruction emitters, and a writer
//! that renders valid `.ptx` text output.
//!
//! ## Modules
//!
//! - [`types`] — PTX type system and Rust-to-PTX type mapping
//! - [`ir`] — Intermediate representation (modules, kernels, instructions)
//! - [`instr`] — Instruction category enums (arithmetic, memory, control)
//! - [`emit`] — PTX text emission (`Emit` trait + `PtxWriter`)

#![warn(missing_docs)]

pub mod emit;
pub mod instr;
pub mod ir;
pub mod types;
