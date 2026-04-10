# Changelog

All notable changes to PYROS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

Updated at phase completion, not per sprint. Per-sprint detail lives in
[docs/development/sprints/](docs/development/sprints/).

## [Unreleased]

### Added
- Virtual workspace: `pyros` (umbrella), `pyros-core` (PTX IR + emission), `pyros-runtime` (CUDA driver wrapper via cudarc 0.19)
- PTX type system mapping Rust primitives to PTX types, with sealed `GpuType` trait
- PTX intermediate representation: modules, kernels, parameters, registers, instructions, operands
- Virtual register allocator with 5-counter model matching nvcc register naming conventions
- `Emit` trait and `PtxWriter` for indent-aware PTX text generation
- Arithmetic instruction emitters (`add`, `mad.lo`, `mul.wide`) validated byte-for-byte against nvcc 12.8 output on RTX 4090
- nvcc reference golden PTX for `vector_add` (sm_89)
- cudarc smoke test confirming GPU communication on RTX 4090

### Changed
- Restructured from single crate to virtual workspace
- PTX ISA version corrected from 7.8 to 8.7 (CUDA 12.8)
- Register declarations use `.b32`/`.b64` (untyped) matching nvcc convention

## [0.0.1] — 2026-04-10

### Added
- Name reservation crate with metadata, README, dual MIT/Apache-2.0 license
- Project design docs: index.md, implementation.md, phases.md, success-criteria.md
