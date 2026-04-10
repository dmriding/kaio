# Changelog

All notable changes to PYROS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Virtual workspace with three crates: `pyros` (umbrella), `pyros-core` (PTX IR + emission), `pyros-runtime` (CUDA driver wrapper)
- `pyros-core` type system: `PtxType` (7 variants), `RegKind` (5 register kinds), `GpuType` sealed trait for Rust↔PTX type mapping
- `pyros-core` IR tree: `PtxModule`, `PtxKernel`, `PtxParam`, `Register`, `RegisterAllocator`, `PtxInstruction`, `Operand`, `SpecialReg`
- `pyros-core` emit scaffold: `Emit` trait, `PtxWriter` (indent-aware PTX text builder)
- Instruction category stubs: `ArithOp`, `MemoryOp`, `ControlOp` (uninhabited — populated in upcoming sprints)
- `rust-toolchain.toml` pinning Rust 1.94.1
- `cudarc` 0.19 workspace dependency with `driver + std + dynamic-loading + cuda-12080` features
- Phase 1 sprint log at `docs/development/PHASE_1_LOG.md`

### Changed
- Restructured from single crate at root to virtual workspace
- Bumped version to 0.0.2
- Updated docs: MSRV 1.94, edition 2024, cudarc repo URL fix

## [0.0.1] — 2026-04-10

### Added
- Name reservation crate with metadata, README, dual MIT/Apache-2.0 license
- Project design docs: index.md, implementation.md, phases.md, success-criteria.md
