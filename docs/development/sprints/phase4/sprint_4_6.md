# Sprint 4.6 — Register Tiling Optimization

**Status:** Complete
**Commit:** (pending)
**Date:** 2026-04-11
**Depends on:** Sprint 4.5

## Context

Optimize the naive 16×16 tiled matmul (~8% of cuBLAS) using register
tiling. Each thread computes a 4×4 output subblock instead of 1 element,
increasing arithmetic intensity by 16×.

## Completed Items

### Register-Tiled Kernel — Done

Design parameters (hardcoded, documented as tuning targets):
- **BM = BN = 64** — block output tile size
- **BK = 16** — K tile size
- **TM = TN = 4** — per-thread output tile
- **Block = (16, 16) = 256 threads**
- **Shared:** tile_a 64×17, tile_b 16×65 (padded strides for bank conflicts)

Each thread:
1. Loads 4 elements each from A and B tiles (loop of 4 iterations)
2. Inner loop over BK=16: loads 4 values from each tile, does 16 FMAs
3. Writes 4×4 = 16 output elements with bounds checking

16 named accumulator registers (`acc_00` through `acc_33`), no arrays.
Output write fully unrolled, grouped by row for readability.

### Benchmark Results — Done

| Size | Naive TFLOPS | Opt TFLOPS | cuBLAS TFLOPS | vs cuBLAS | Speedup |
|------|-------------|-----------|--------------|-----------|---------|
| 256² | 0.18 | 0.14 | 1.71 | 8.3% | 0.8× |
| 512² | 1.06 | 1.00 | 10.96 | 9.1% | 0.9× |
| 1024² | 3.10 | 5.96 | 37.61 | 15.8% | 1.9× |
| 2048² | 4.04 | 13.32 | 43.14 | 30.9% | 3.3× |
| 4096² | 4.53 | 17.44 | 56.00 | 31.2% | 3.8× |

### Performance Analysis

**What worked:**
- 3.3-3.8× speedup over naive at large sizes
- Bank conflict padding effective (stride 17/65)
- Correctness maintained — same error tolerances as naive

**Why not 60%:**
- No vectorized loads (LDG.128) — each thread does scalar 32-bit loads
- No software pipelining (double buffering of tiles)
- BK=16 may be suboptimal — BK=8 with 8×8 register tile could be better
- Launch overhead dominates at small sizes (256-512)

**What would get to 60%:**
- Vectorized loads (requires DSL extension: `ld.global.v4.f32`)
- Larger register tiles (8×8 per thread = 64 outputs)
- BK tuning (smaller BK with more register reuse)
- These are Phase 5+ optimizations

### Kernel Selection Policy — Done

- `matmul()` uses optimized kernel by default for all sizes
- `matmul_naive()` available as `#[doc(hidden)]` pub for benchmarks
- No size-based kernel selection (optimized is slower for <1024 but
  the difference is small and consistency matters more than microperf)

## Key Decisions

1. **4×4 per thread, not 8×8** — 16 accumulators fit comfortably in
   registers (vs 64 for 8×8). Lower risk, proven first.

2. **Honest 31% result** — documented actual numbers. 60% target
   requires DSL extensions (vectorized loads) not available in Phase 4.

3. **No size-based dispatch** — keep it simple. One kernel for all sizes.

## Tests

All existing API tests pass with optimized kernel:
- api_matmul_tiny, api_matmul_64x64, api_matmul_non_square,
  api_matmul_non_aligned, api_matmul_oversized_buffers

Total: 205 host tests + 41 GPU tests + 1 benchmark
