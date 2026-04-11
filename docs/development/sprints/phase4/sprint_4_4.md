# Sprint 4.4 — kaio-ops Crate + Host-Side MatMul API

**Status:** Complete
**Commit:** (pending)
**Date:** 2026-04-11
**Depends on:** Sprint 4.3

## Context

Package the tiled matmul kernel from Sprint 4.3 into a new `kaio-ops`
workspace crate with a clean host-side API. Users call
`kaio_ops::matmul()` without knowing about tiles, shared memory, or
launch configuration.

## Completed Items

### kaio-ops Crate — Done
New workspace member with:
- `pub fn matmul(device, a, b, c, m, n, k) -> Result<()>`
- Internal `#[gpu_kernel(block_size = (16, 16))]` tiled matmul kernel
- Depends only on `kaio` umbrella (not kaio-runtime directly)

### Dimension Validation — Done
Before launch, validates:
- All dimensions non-zero (specific error: "dimensions must be non-zero")
- A buffer ≥ M×K elements (specific: "A buffer too small: need N, got M")
- B buffer ≥ K×N elements
- C buffer ≥ M×N elements
- Returns `KaioError::InvalidConfig` with descriptive messages

### Launch Derivation — Done
`matmul()` owns the launch policy:
- Tile size: 16×16
- Block size: (16, 16) = 256 threads
- Grid: `(ceil(N/16), ceil(M/16), 1)`

### Oversized Buffers — Documented
Buffers may be larger than the logical matrix region. Only the first
M×K / K×N / M×N elements are used. This is documented in the function
doc comment and tested.

## Key Decisions

1. **kaio-ops is separate from kaio umbrella.** Users add it with
   `cargo add kaio-ops`. The umbrella is for kernel authors, kaio-ops
   is for kernel consumers.

2. **Depends only on kaio (umbrella).** The umbrella re-exports all
   needed runtime types via `kaio::prelude::*`. No direct `kaio-runtime`
   dependency.

3. **kaio-ops owns the canonical matmul kernel.** The test in
   `kaio/tests/matmul_macro.rs` stays as a macro pipeline regression
   test (different purpose — tests the codegen, not the product).

## Tests Added

- `api_matmul_tiny` — 2×3×4 hand-checkable (+1 GPU)
- `api_matmul_64x64` — aligned (+1 GPU)
- `api_matmul_non_square` — 100×200×150 (+1 GPU)
- `api_matmul_non_aligned` — 17×33×19 (+1 GPU)
- `api_matmul_oversized_buffers` — buffers larger than logical region (+1 GPU)
- `api_matmul_rejects_zero_m` — zero-dim validation (+1 host)
- `api_matmul_rejects_small_buffer` — buffer too small (+1 host)

Total: 205 host tests + 41 GPU tests
