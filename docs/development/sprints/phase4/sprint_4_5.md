# Sprint 4.5 — Benchmark Harness + cuBLAS Baseline

**Status:** Complete
**Commit:** (pending)
**Date:** 2026-04-11
**Depends on:** Sprint 4.4

## Context

Measure the naive tiled matmul's performance against cuBLAS sgemm to
establish a baseline for optimization. This sprint answers: "how far are
we from cuBLAS, and what should Sprint 4.6 focus on?"

## Completed Items

### Benchmark Harness — Done
`kaio-ops/tests/matmul_bench.rs` — ignored test with manual execution.
GPU-side timing via `stream.synchronize()` + wall clock. 5 warm-up,
20 measured, report median TFLOPS.

### cuBLAS Reference — Done
cudarc's `cublas` module as a dev-dependency. Row-major trick validated
against CPU reference at 64×64 before reporting any ratios.

### Benchmark Results — Done

| Size | KAIO TFLOPS | cuBLAS TFLOPS | Ratio |
|------|------------|--------------|-------|
| 256² | 0.16 | 1.72 | 9.6% |
| 512² | 0.92 | 10.78 | 8.5% |
| 1024² | 3.02 | 37.15 | 8.1% |
| 1024×2048×512 | 3.08 | 35.73 | 8.6% |
| 2048² | 4.04 | 52.93 | 7.6% |
| 4096² | 4.61 | 57.40 | 8.0% |

**Analysis:** Naive kernel achieves ~8% of cuBLAS. Expected for a
16×16 tiled kernel without register tiling, bank conflict avoidance,
or vectorized loads.

### Methodology Doc — Done
`docs/benchmarks.md` — methodology-first document with reproducible
setup, timing approach, input generation, and cuBLAS validation.

## Key Decisions

1. **Wall-clock + synchronize** over CUDA events — simpler, fair for
   both KAIO and cuBLAS on the same stream.
2. **Deterministic RNG** — fixed seeds (42, 137) for reproducibility.
3. **cuBLAS as dev-dependency** — no feature flag needed since
   dev-dependencies only compile for tests. Regular users unaffected.
4. **Methodology-first doc** — benchmarks.md describes how numbers are
   produced, not just the numbers.

## Tests Added

- `benchmark_matmul` — ignored benchmark test (+0 host, +1 GPU bench)

Total: 205 host tests + 41 GPU tests + 1 benchmark
