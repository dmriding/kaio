# PYROS — Rust-Native GPU Kernel Authoring Framework

> πῦρ (pyr) — fire. The forge where GPU kernels are crafted.

## What Is PYROS?

PYROS is a Rust crate that lets developers write GPU compute kernels in Rust and compile them to PTX for execution on NVIDIA GPUs. It eliminates the need to write CUDA C++ or depend on Python-based tooling like OpenAI's Triton.

## Why PYROS Exists

The Rust ML ecosystem has no native way to author custom GPU kernels. Every project that needs a custom operation — fused activations, quantized inference, custom attention — hits the same wall: drop into CUDA C++, wrestle with FFI bindings, or accept the performance ceiling of pre-built ops.

Triton (OpenAI) solves this for Python, but:

- **Linux-only.** No official Windows support. The community PR was rejected. A fork (`triton-windows`) exists but is fragile and requires manual CUDA path management.
- **Python-locked.** Cannot be embedded in Rust, C++, or any non-Python runtime without dragging in a Python interpreter.
- **Slow compilation.** Kernels compile at runtime through Python → MLIR → LLVM → PTX. First launch can take 30+ seconds.
- **Silent failures.** Wrong index math produces garbage output with no error.

PYROS targets these gaps directly:

- **Cross-platform from day one.** Windows and Linux. `cargo build` just works.
- **Compile-time PTX emission.** Kernels are compiled during `cargo build` via proc macros. Zero cold-start.
- **Rust type safety.** Catch out-of-bounds indexing, dtype mismatches, and synchronization errors at compile time.
- **Embeddable anywhere.** Use from Rust natively, from C/C++ via FFI, from Python via PyO3.

## Architecture Overview

PYROS is structured in four layers, each building on the one below:

```
┌─────────────────────────────────────────┐
│  Layer 4: Block-Level Operations        │  ← tiled matmul, fused attention
│  (Triton-equivalent abstraction)        │
├─────────────────────────────────────────┤
│  Layer 3: Proc Macro DSL                │  ← #[gpu_kernel], user-facing API
│  (Rust syntax → PTX lowering)           │
├─────────────────────────────────────────┤
│  Layer 2: Runtime                       │  ← kernel launch, memory mgmt
│  (CUDA driver API via cudarc)           │
├─────────────────────────────────────────┤
│  Layer 1: PTX Codegen                   │  ← instruction emission, IR
│  (Rust structs → valid .ptx text)       │
└─────────────────────────────────────────┘
```

## Crate Structure

```
pyros/
├── pyros-core/        # PTX codegen + IR (Layer 1)
├── pyros-runtime/     # CUDA driver API bindings, kernel launch (Layer 2)
├── pyros-macros/      # #[gpu_kernel] proc macro (Layer 3)
├── pyros-ops/         # Block-level operations library (Layer 4)
├── pyros/             # Umbrella crate re-exporting everything
├── tests/             # Integration tests, golden-reference validation
├── benches/           # Benchmarks against cuBLAS / known-good implementations
└── examples/          # Standalone example kernels
```

## Target Hardware

- **Primary:** NVIDIA GPUs, SM 7.0+ (Volta and newer)
- **Development GPU:** RTX 4090 (SM 8.9, Ada Lovelace)
- **PTX ISA Target:** 7.x+ (with forward compatibility)
- **Platforms:** Windows 10/11, Linux (Ubuntu 22.04+)

## Documentation Index

| Document | Description |
|----------|-------------|
| [index.md](index.md) | This file — project overview and architecture |
| [implementation.md](implementation.md) | Technical implementation details per layer |
| [phases.md](phases.md) | Development phases, timelines, and deliverables |
| [success-criteria.md](success-criteria.md) | Quality gates and success metrics per phase |

## Key Dependencies

| Crate | Purpose |
|-------|---------|
| `cudarc` | Rust bindings to CUDA driver API (runtime layer foundation) |
| `syn` / `quote` / `proc-macro2` | Proc macro infrastructure (DSL layer) |

## Project Metadata

- **Owner:** Dave Riding / NetViper
- **License:** MIT OR Apache-2.0 (dual-licensed for ecosystem adoption)
- **Repository:** https://github.com/dmriding/pyros
- **Crates.io:** `pyros` — name reserved at `v0.0.1` (publish pending Phase 1 completion)
- **Rust Edition:** 2024
- **Minimum Rust Version:** 1.94 (pinned to match the dev environment during Phase 1; may relax to 1.85, the edition 2024 floor, in later phases)

## Design Principles

1. **Correctness over performance initially.** A correct kernel that's 80% as fast as cuBLAS is infinitely more useful than an optimized kernel that silently produces wrong results.
2. **Incremental complexity.** A user writing their first elementwise kernel should not need to understand tiling, shared memory, or warp-level programming.
3. **No hidden magic.** The generated PTX should be inspectable and understandable. Users should be able to see exactly what PYROS produces.
4. **Windows is not an afterthought.** Every feature is tested on Windows and Linux before merge.
5. **Forge-friendly architecture.** Modules are decomposable into independent sprint-sized units with clear interfaces and testable boundaries.
