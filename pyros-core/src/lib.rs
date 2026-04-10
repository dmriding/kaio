//! # pyros-core
//!
//! PTX intermediate representation and code emission for the PYROS GPU
//! kernel authoring framework. This crate provides the Layer 1 foundation:
//! Rust types modelling PTX programs, instruction emitters, and a writer
//! that renders valid `.ptx` text output.
//!
//! **Status:** Phase 1 scaffolding. No public API yet.

#![warn(missing_docs)]
