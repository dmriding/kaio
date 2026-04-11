# Sprint 4.2 — Multi-Allocation Shared Memory

**Status:** Complete
**Commit:** (pending)
**Date:** 2026-04-11
**Depends on:** Sprint 4.1

## Context

Tiled matmul (Sprint 4.3) needs two shared memory tiles (`tile_a`,
`tile_b`). The existing shared memory addressing assumed a single
allocation starting at offset 0, which silently produced wrong results
when multiple allocations were present.

## Completed Items

### Named-Symbol Base Addressing — Done
Replaced the single-allocation `mul.u32 addr, idx, sizeof(T)` pattern
with named-symbol base addressing:

```
mov.u32 base, tile_b       // PTX resolves allocation's base address
mul.u32 byte_off, idx, 4   // byte offset within allocation
add.u32 addr, base, byte_off
ld.shared.f32 dst, [addr]
```

Each `.shared` declaration gets its own PTX symbol. `Operand::SharedAddr`
(from Sprint 3.3) loads the symbol's base address. PTX handles layout —
no manual offset computation needed.

**Design note:** Shared addresses are always 32-bit (U32). This is a
deliberate PTX/shared-space choice — `.shared` is per-SM SRAM with a
32-bit address space, distinct from 64-bit global memory.

### Reduction Shared Memory Fix — Done
Updated all 4 `ld.shared`/`st.shared` access points in the reduction
codegen (`block_reduce_sum`/`block_reduce_max`) to use named-symbol base
addressing for `_kaio_reduce_smem`. Previously used raw byte offsets
assuming offset 0 — would have overlapped user-declared shared arrays.

### Launch Config Block Size Fix — Done (bonus find)
Discovered that `LaunchConfig::for_num_elems()` from cudarc hardcodes
`block_dim = (1024, 1, 1)` regardless of the kernel's declared
`block_size`. This was a latent Phase 2 bug: kernels declared with
`block_size = 256` were launched with 1024 threads per block. The
reduction allocated `_kaio_reduce_smem` for 8 warps (256/32) but 32
warps (1024/32) tried to write — causing `CUDA_ERROR_ILLEGAL_ADDRESS`
when user shared memory was also present.

Fix: 1D launch wrapper now computes its own `LaunchConfig` using the
declared `block_size` instead of delegating to `for_num_elems()`.

### Multi-Allocation GPU Tests — Done
- **Two shared f32 arrays**: distinct write patterns, bar_sync, readback
- **Shared array + block_reduce_sum coexistence**: both work in same kernel
- **Three shared arrays**: cumulative addressing verification

## Key Decisions

1. **Named symbols over manual offsets** — PTX resolves allocation base
   addresses via symbol names. Simpler, order-independent, no manual
   offset bookkeeping.

2. **Unconditional base load** — `mov.u32 base, symbol` emitted for
   every access (even single-allocation kernels where base = 0). Adds
   2 instructions per access. Intentional simplicity tradeoff —
   optimization via base hoisting deferred to Sprint 4.6+.

3. **`_kaio_reduce_smem` is compiler-owned** — treated like any named
   shared allocation for addressing, but remains internal. Users cannot
   reference or collide with it.

## Tests Added

- `two_shared_arrays_no_aliasing` — 2 shared f32 arrays, distinct patterns (+1 GPU)
- `shared_array_plus_reduction` — shared_mem + block_reduce_sum coexist (+1 GPU)
- `three_shared_arrays_correctness` — 3 shared arrays sum correctly (+1 GPU)

Total: 203 host tests + 30 GPU tests
