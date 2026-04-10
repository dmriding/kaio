# Changelog

All notable changes to KAIO will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

Updated at phase completion. Per-sprint detail lives in
[docs/development/sprints/](docs/development/sprints/).

## [Unreleased] — Phase 2 Complete

### Added — Phase 2: Proc Macro DSL
- **`#[gpu_kernel]` proc macro** (`kaio-macros`): full pipeline from Rust function
  syntax to executable PTX. Parse (syn → KernelIR) → lower (IR → PTX instructions)
  → codegen (build_ptx with OnceLock + typed launch wrapper).
- **Supported Rust subset**: arithmetic (`+`, `-`, `*`, `/`, `%`, `+=`, `-=`,
  `*=`, `/=`), comparisons (`<`, `<=`, `>`, `>=`, `==`, `!=`), `if`/`else` with
  `@!pred` predicated branches, array indexing with cvta.to.global caching, `let`
  bindings with type inference, type casts via `as`.
- **19 built-in functions**: thread/block/grid intrinsics (`thread_idx_x`,
  `block_idx_x`, `block_dim_x`, `grid_dim_x`), math functions (`sqrt`, `rsqrt`,
  `abs`, `min`, `max`, `sin`, `cos`), synthesized transcendentals (`exp`, `log`,
  `tanh` via PTX `ex2`/`lg2`/`rcp` primitives).
- **Launch wrapper generation**: typed function signatures matching kernel params,
  automatic grid/block calculation, argument marshaling via `PushKernelArg`.
- **`KAIO_DUMP_PTX`** environment variable: set to `1` to write `.ptx` files to
  disk during compilation for inspection.
- **4 E2E kernels validated on RTX 4090**: `vector_add`, `saxpy`, `fused_relu`,
  `fused_gelu` — all produce correct results against CPU reference implementations.
- **10 trybuild compile-fail tests** (CF1–CF10): unsupported types, non-unit
  return, macro invocations, heap allocation, non-built-in function calls, missing
  `block_size`, non-power-of-2 `block_size`, `block_size` > 1024, lifetime
  parameters, `loop` keyword.
- **`kaio` umbrella crate**: prelude module re-exporting macro + runtime, 
  `gpu_builtins` IDE stub module for autocomplete.
- 168 host tests + 5 GPU tests across workspace

### Fixed — Phase 2
- **cvta.to.global register scope across if/else branches**: conversion registers
  were scoped inside branches, causing undefined-register errors in subsequent
  code. Fix: eager emission during parameter loading, before any control flow.
- **PTX float division requires `.approx`/`.rn` modifier**: bare `div.f32` is
  invalid PTX. Fix: type-aware modifier selection in `ArithOp::Div` emission
  (`.approx` for f32, `.rn` for f64).

### Changed — Phase 2
- Renamed project from PYROS to KAIO across entire codebase (commit `50a3ab0`)

### Added — Phase 1: PTX Foundation
- **PTX code generation** (`kaio-core`): IR types modelling complete PTX programs,
  instruction emitters for arithmetic (add, mad, mul.wide), memory (ld.param,
  ld.global, st.global, cvta.to.global), and control flow (setp, bra, ret).
  Emit trait + PtxWriter produce valid PTX text from an IR tree.
- **CUDA runtime wrapper** (`kaio-runtime`): KaioDevice for GPU context management,
  GpuBuffer<T> for typed device memory, KaioModule/KaioFunction for PTX loading
  and kernel launch via cudarc 0.19.
- **End-to-end `vector_add`**: kernel constructed via Rust IR, emitted to PTX,
  loaded into the CUDA driver, launched on RTX 4090 — produces correct results
  for both single-block (3 elements) and multi-block (10,000 elements).
- **Validation**: all PTX instruction emitters verified byte-for-byte against
  nvcc 12.8 output. ptxas offline verification passes. cudarc smoke test confirms
  host↔device data transfer.
- Virtual workspace with umbrella `kaio` crate re-exporting `kaio-core` + `kaio-runtime`
- 53 host-side tests + 9 GPU-gated tests, 82.8% line coverage
- Per-sprint architectural decision records in `docs/development/sprints/`

### Changed — Phase 1
- PTX ISA version corrected from 7.8 to 8.7 (CUDA 12.8)
- Register declarations use `.b32`/`.b64` (untyped) matching nvcc convention

## [0.0.1] — 2026-04-10

### Added
- Name reservation crate with metadata, README, dual MIT/Apache-2.0 license
- Project design docs: index.md, implementation.md, phases.md, success-criteria.md
