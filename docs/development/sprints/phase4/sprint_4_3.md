# Sprint 4.3 — Naive Tiled MatMul Kernel

**Status:** Complete
**Commit:** (pending)
**Date:** 2026-04-11
**Depends on:** Sprint 4.1 (FMA, 2D launch), Sprint 4.2 (multi-alloc shared mem)

## Context

First real matrix multiplication through `#[gpu_kernel]`. Correctness-first
naive 16×16 tiled kernel. Exercises the full Phase 4 pipeline: 2D thread
blocks, two shared memory tiles, FMA inner loop, while-loop tiling over K,
and edge handling via zero-fill.

## Completed Items

### Tiled MatMul Kernel — Done

Classic tiled matmul `C = A × B` using `#[gpu_kernel(block_size = (16, 16))]`:
- Each thread block computes a 16×16 tile of output C
- Two shared tiles (`tile_a`, `tile_b`) for A and B fragments
- While loop tiles over K dimension: `(k + 15) / 16` iterations
- FMA inner loop: `acc = fma(tile_a[ty*16+i], tile_b[i*16+tx], acc)`
- Edge handling: zero-fill shared tiles, bounds-check output write
- Grid launch: `(n.div_ceil(16), m.div_ceil(16), 1)`

### Known Unknowns Resolved

1. **Integer division in kernel** — `(k + 15) / 16` works. `div.u32` emits
   correctly and produces correct tile counts.
2. **While loop + shared mem + bar_sync combo** — works correctly. First
   kernel to exercise this combination: load tiles → bar_sync → FMA loop
   → bar_sync → next iteration.
3. **2D shared indexing** — `tile_a[ty * 16 + i]` compound index expressions
   lower correctly.

### Accuracy Results

| Size | Max Abs Error | Max Rel Error |
|------|-------------|-------------|
| 2×3×4 (tiny) | exact | exact |
| 16×16 | 1.79e-7 | 3.35e-6 |
| 64×64 | 5.96e-7 | 1.02e-5 |
| 100×200×150 | 7.15e-7 | 1.56e-5 |
| 17×33×19 | 3.58e-7 | 1.19e-6 |
| 1024×1024 | 4.77e-6 | 1.53e-5 |

All well under 1e-3 absolute / 1e-4 relative thresholds. FMA's single
rounding step gives better accuracy than the CPU reference (separate
mul+add).

## Tests Added

- `matmul_tiny` — 2×3×4, hand-checkable exact values (+1 GPU)
- `matmul_16x16` — single tile, no edge handling (+1 GPU)
- `matmul_64x64` — multi-tile, aligned (+1 GPU)
- `matmul_non_square` — 100×200×150, non-aligned (+1 GPU)
- `matmul_non_aligned` — 17×33×19, prime-ish (+1 GPU)
- `matmul_1024x1024` — large stress test (+1 GPU)

Total: 203 host tests + 36 GPU tests
