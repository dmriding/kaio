# Sprint 4.1 — FMA + 2D Block Size + 2D Launch Model

**Status:** Complete
**Commit:** (pending)
**Date:** 2026-04-11
**Depends on:** Phase 3 complete

## Context

Infrastructure sprint for Phase 4. No matmul yet — just the primitives
it needs. Three deliverables that unblock tiled matrix multiplication:
FMA instruction, 2D block size, and 2D launch configuration.

## Completed Items

### FMA Instruction — Done
Added `ArithOp::Fma { dst, a, b, c, ty }` to kaio-core. Emits
`fma.rn.f32 dst, a, b, c;` — fused multiply-add with IEEE round-to-nearest.
Float-only (F32, F64). Essential for matmul inner loop: `acc = fma(a, b, acc)`
is one instruction with single rounding, vs separate mul+add.

Added `fma(a, b, c)` as a builtin function in kaio-macros and IDE stub
in `gpu_builtins.rs`. GPU-validated on RTX 4090.

### 2D Block Size Attribute — Done
Extended attribute parser to accept `block_size = (X, Y)` tuple syntax
alongside existing `block_size = N` scalar syntax. Backward compatible.

Validation:
- 2D: total threads (X × Y) ≤ 1024, both > 0
- 1D: power of 2, ≤ 1024 (unchanged)
- 2D does NOT require power-of-2 per dimension

`KernelConfig` gained `block_size_y: Option<u32>`. `LoweringContext`
gained `block_size_x` and `block_size_y` fields for future tile dimension
access in Sprint 4.3+.

### 2D Launch Wrapper — Done
When `block_size_y` is `Some`, the generated `launch()` function takes a
`grid: (u32, u32, u32)` parameter instead of inferring grid from the last
`u32` parameter. Block dims are hardcoded from the attribute to prevent
mismatches between declared and runtime block shapes:

```rust
let grid = (cols.div_ceil(16), rows.div_ceil(16), 1);
kernel::launch(&device, &mut out, rows, cols, grid).unwrap();
```

1D kernels (`block_size = N`) are completely unchanged — grid is still
inferred from the last `u32` via `LaunchConfig::for_num_elems()`.

### 2D Thread Indices — Already Existed
Discovery: `thread_idx_y()`, `block_idx_y()`, `block_dim_y()`, and
`grid_dim_y()` were already fully implemented in the builtin registry,
kaio-core special.rs helpers, and IDE stubs from Phase 1/2 forward-looking
design. No new work needed.

## Key Decisions

1. **Explicit LaunchConfig for 2D** — Users must compute grid dims.
   Can't infer 2D grid from kernel params alone (grid depends on
   matrix dimensions AND tile size — application logic).

2. **block_size_x/y on LoweringContext** — Added now even though Sprint
   4.1 doesn't use them. Sprint 4.3 matmul will need individual dimensions
   for tile size calculations. Avoids a refactor.

3. **FMA is f32-only for now** — uses `.rn` rounding (IEEE round-to-nearest).
   No mode parameter (unlike Mad which has .lo/.hi/.wide). All three
   arguments are validated as f32. f64 support deferred.

## Tests Added

- `emit_fma_f32` — PTX emission unit test (+1 kaio-core)
- `parse_block_size_2d` — tuple parsing (+1 kaio-macros)
- `parse_block_size_2d_asymmetric` — (32, 8) works (+1)
- `reject_block_size_2d_exceeds_1024` — (32, 64) = 2048 rejected (+1)
- `reject_block_size_2d_zero_y` — (16, 0) rejected (+1)
- `kernel_2d_write_indices` — 2D kernel GPU E2E, 32×48 (+1 GPU)
- `kernel_2d_non_aligned_dims` — 17×33 non-tile-aligned (+1 GPU)
- `kernel_fma_correctness` — FMA elementwise GPU test (+1 GPU)
- Updated cf04_unknown_call.stderr — added `fma` to builtin list

Total: 203 host tests + 27 GPU tests
