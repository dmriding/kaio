# Changelog

All notable changes to PYROS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

Updated at phase completion. Per-sprint detail lives in
[docs/development/sprints/](docs/development/sprints/).

## [Unreleased] — Phase 1 Complete

### Added
- **PTX code generation** (`pyros-core`): IR types modelling complete PTX programs,
  instruction emitters for arithmetic (add, mad, mul.wide), memory (ld.param,
  ld.global, st.global, cvta.to.global), and control flow (setp, bra, ret).
  Emit trait + PtxWriter produce valid PTX text from an IR tree.
- **CUDA runtime wrapper** (`pyros-runtime`): PyrosDevice for GPU context management,
  GpuBuffer<T> for typed device memory, PyrosModule/PyrosFunction for PTX loading
  and kernel launch via cudarc 0.19.
- **End-to-end `vector_add`**: kernel constructed via Rust IR, emitted to PTX,
  loaded into the CUDA driver, launched on RTX 4090 — produces correct results
  for both single-block (3 elements) and multi-block (10,000 elements).
- **Validation**: all PTX instruction emitters verified byte-for-byte against
  nvcc 12.8 output. ptxas offline verification passes. cudarc smoke test confirms
  host↔device data transfer.
- Virtual workspace with umbrella `pyros` crate re-exporting `pyros-core` + `pyros-runtime`
- 53 host-side tests + 9 GPU-gated tests, 82.8% line coverage
- Per-sprint architectural decision records in `docs/development/sprints/`

### Changed
- PTX ISA version corrected from 7.8 to 8.7 (CUDA 12.8)
- Register declarations use `.b32`/`.b64` (untyped) matching nvcc convention

## [0.0.1] — 2026-04-10

### Added
- Name reservation crate with metadata, README, dual MIT/Apache-2.0 license
- Project design docs: index.md, implementation.md, phases.md, success-criteria.md
