# Phase 4 Master Plan — Tiled MatMul & Block-Level API

**Status:** Complete (v0.0.4, 2026-04-11)
**Depends on:** Phase 3 complete (commit `0691be8`)
**Reviewed by:** Codex 5.4 (readiness review), Claude Opus 4.6

## Goal

Deliver tiled matrix multiplication through KAIO's `#[gpu_kernel]` macro,
introduce the `kaio-ops` crate as a host-side library layer, and benchmark
against cuBLAS. This is the phase where KAIO transitions from "promising
custom-kernel DSL" to "practical GPU programming layer for real workloads."

Target user experience:

```rust
use kaio_ops::matmul;
use kaio::prelude::*;

let device = KaioDevice::new(0)?;
let a = device.alloc_from(&a_data)?;  // M x K, row-major f32
let b = device.alloc_from(&b_data)?;  // K x N, row-major f32
let mut c = device.alloc_zeros::<f32>(m * n)?;
matmul(&device, &a, &b, &mut c, m, n, k)?;
```

## Key Architectural Decisions

### 1. Launch Model: Grid Tuple for 2D Kernels

The current launch wrapper infers grid size from the last `u32` parameter
via `LaunchConfig::for_num_elems()`. This is fundamentally incompatible
with 2D matmul where grid shape depends on `(M, N, tile_size)`.

**Decision:** Keep 1D behavior for `block_size = N` (backward compatible).
Add `block_size = (X, Y)` tuple syntax for 2D kernels. 2D kernels generate
`launch()` that takes an explicit `grid: (u32, u32, u32)` tuple parameter.

```rust
#[gpu_kernel(block_size = 256)]        // 1D: existing, grid inferred
#[gpu_kernel(block_size = (16, 16))]   // 2D: caller passes grid tuple
```

**Note:** The original plan proposed an explicit `LaunchConfig` struct
parameter. During implementation, this was simplified to a plain
`(u32, u32, u32)` grid tuple appended to the `launch()` signature.
This is simpler and matches CUDA's grid-launch model more directly.

### 2. FMA Instruction

The matmul inner loop does `acc += a * b` thousands of times. Without
`fma.rn.f32`, that's a separate `mul.f32 + add.f32` per iteration — twice
the instruction count and worse numerical accuracy. FMA does it in one
instruction with a single rounding step.

**Decision:** Add `ArithOp::Fma` to kaio-core. Expose as `fma(a, b, c)`
builtin. This is Sprint 4.1 infrastructure, not deferred.

### 3. 2D Thread Indices

PTX supports `%tid.y`, `%ctaid.y`, `%ntid.y`, `%nctaid.y` — all variants
already exist in kaio-core's `SpecialReg` enum. They're just not wired
into the macro's builtin registry.

**Decision:** Expose `thread_idx_y()`, `block_idx_y()`, `block_dim_y()`,
`grid_dim_y()` in Sprint 4.1. No new IR types needed.

### 4. Matrix Semantics (Phase 4 Scope)

| Property | Phase 4 Scope |
|----------|--------------|
| Dtype | f32 only |
| Layout | Row-major, contiguous |
| Shapes | Non-square: A(M×K) × B(K×N) → C(M×N) |
| Edge tiles | Bounds checking (zero-fill in shared loads) |
| Transpose | Not supported |
| Batching | Not supported |
| Strides | Not supported (contiguous only) |

### 5. kaio-ops: Host-Side Library

New workspace member. Contains `#[gpu_kernel]` kernels internally. Users
call `kaio_ops::matmul()` without knowing about tiles or shared memory.
Fifth crate to publish to crates.io.

### 6. Performance Target

- **Original target:** ≥60% of cuBLAS `cublasSgemm` at 2048×2048 f32
- **Actual result:** 31% of cuBLAS at 4096×4096 (17.44 TFLOPS vs
  56.00 TFLOPS on RTX 4090). Documented honestly in `docs/benchmarks.md`.
- **Metric:** TFLOPS = 2 × M × N × K / median_time / 1e12
- **Methodology:** 5 warm-up launches, 20 timed, report median
- **Timing:** Kernel execution only (no alloc/transfer)
- **What shipped:** Naive matmul (Sprint 4.3) ≈ 8% of cuBLAS.
  Register-tiled (Sprint 4.6, 64×64 tiles, 4×4 per thread) reached
  31%. Remaining gap requires vectorized loads (LDG.128) and double
  buffering — deferred to Phase 6.

## Sprint Plan

| Sprint | Scope | Depends On |
|--------|-------|------------|
| [4.1](sprint_4_1.md) | FMA + 2D indices + 2D launch model | — |
| [4.2](sprint_4_2.md) | Multi-allocation shared memory | 4.1 |
| [4.3](sprint_4_3.md) | Naive tiled matmul kernel (correctness) | 4.1 + 4.2 |
| [4.4](sprint_4_4.md) | kaio-ops crate + host-side matmul API | 4.3 |
| [4.5](sprint_4_5.md) | Benchmark harness + cuBLAS baseline | 4.4 |
| [4.6](sprint_4_6.md) | Register tiling + optimization | 4.5 |
| [4.7](sprint_4_7.md) | PTX inspection + performance documentation | 4.6 |
| [4.8](sprint_4_8.md) | Polish + integration tests + publish | All |
| [4.9](sprint_4_9.md) | Adoption polish — examples + README | 4.8 |

## Dependency Graph

```
4.1 (FMA + 2D launch) → 4.2 (multi-shared-mem) → 4.3 (naive matmul)
                                                        ↓
                         4.4 (kaio-ops) → 4.5 (benchmarks) → 4.6 (register tiling)
                                                                    ↓
                                                              4.7 (PTX inspection)
                                                                    ↓
                                                              4.8 (polish)
                                                                    ↓
                                                              4.9 (adoption)
```

All sprints are sequential. 4.1 and 4.2 are infrastructure, 4.3 is proof
of concept, 4.4-4.5 are packaging and measurement, 4.6 is optimization,
4.7 is developer tooling, 4.8 is polish, 4.9 is adoption (examples,
README, patterns).

**Note:** Compile-time coalescing analysis (originally planned for 4.7)
is deferred to Phase 5+ when `block_load`/`block_store` abstractions
provide analyzable access patterns.

## Key Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| 2D launch regression | Breaks 1D kernels | Sprint 4.1 runs all Phase 3 tests |
| Shared memory bank conflicts | 10× perf cliff | Sprint 4.6 padding; benchmark detects |
| FMA rounding vs CPU reference | False test failures | 1e-3 tolerance for large matrices |
| Register tiling complexity | Correctness bugs | Test each config against naive output |
| 60% cuBLAS target | May need vectorized loads | Document actual numbers honestly |
| Edge tile bounds | Off-by-one bugs | Prime-number dimension test suite |

## Review Context

This plan was informed by two external reviews:
- [General review](../review1/review_2026_04_11.md) — code quality assessment
- [Phase 4 readiness review](../review1/phase4_readiness_review_2026_04_11.md) —
  transition analysis with must-do/should-do/can-defer gate structure

Key review recommendations incorporated:
- Launch model redesign before matmul (review 5.1) → Sprint 4.1
- Block semantics definition (review 5.2) → simplified to shared_mem + loops
- Library-first approach (review 5.3) → kaio-ops in Sprint 4.4
- Benchmark methodology (review 5.5) → Sprint 4.5
- Coalescing as heuristics (review 8.1) → deferred to Phase 5+ (Sprint 4.7 rescoped to PTX inspection)
- Naive before optimized (review 8.2) → Sprint 4.3 then 4.6
