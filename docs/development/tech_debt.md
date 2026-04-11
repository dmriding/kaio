# KAIO — Technical Debt Tracker

Items identified during development that are deferred but should be
addressed before v0.1.0 release. Organized by priority.

## High Priority (before Phase 5)

### block_reduce_* 2D support
Reductions derive thread identity from `TidX` only. In 2D kernels
(`block_size = (X, Y)`), multiple rows alias the same warp slots.
Currently rejected at compile time for 2D kernels (Sprint 4.2 review).
Fix: use linear tid = `thread_idx_x + block_dim_x * thread_idx_y`.
**Blocking:** any kernel combining 2D blocks with reductions.
**Added:** Sprint 4.2 | **Sprint:** Phase 4.6+ or Phase 5

### Host-level codegen regression tests
Critical fixes (launch config block_dim, shared addressing) are only
verified by ignored GPU tests — not part of CI. Add host-level tests:
- launch wrapper emits correct `block_dim` matching declared `block_size`
- shared memory lowering emits `Operand::SharedAddr` + `Add` pattern
- reduction lowering uses `_kaio_reduce_smem` named-symbol addressing
**Added:** Sprint 4.2 review | **Sprint:** Phase 4.8 or Phase 5

### fma() f64 support
`fma()` builtin validates all args as f32 only. The `ArithOp::Fma` variant
supports f64 in the IR but the builtin and IDE stub are f32-only.
**Added:** Sprint 4.1 | **Sprint:** when f64 kernels are needed

## Medium Priority

### Shared memory base register hoisting
`mov.u32 base, <symbol>` is emitted for every shared memory access, even
inside tight loops. Could be hoisted to emit once before the loop body.
Adds 2 instructions per access — acceptable for correctness but becomes
a performance concern in matmul inner loops.
**Added:** Sprint 4.2 | **Sprint:** Phase 4.6 optimization

### Shared memory helper abstraction
Two code paths generate the same `mov+mul+add` addressing pattern:
generic `compute_shared_address()` and reduction codegen in builtins.rs.
Extract a shared helper to prevent drift as the pattern evolves.
**Added:** Sprint 4.2 review | **Sprint:** when either path changes

### Windows CI
Docs and README claim Windows support. CI is Ubuntu-only. Add Windows
to the GitHub Actions matrix before release.
**Added:** Phase 3 housekeeping | **Sprint:** Phase 5

## Low Priority

### `&&` / `||` logical operators in kernel DSL
Currently not supported — users must use nested `if` statements.
Discovered in Sprint 4.1 when writing 2D kernel test.
**Added:** Sprint 4.1 | **Sprint:** TBD

### Compound assignment for shared memory
`sdata[idx] += val` is not supported — requires `sdata[idx] = sdata[idx] + val`.
**Added:** Phase 3 | **Sprint:** TBD

### ArithOp::Shr optimization for reductions
Reduction warp_id computation uses `div.u32 warp_id, tid, 32`. Could use
`shr.u32 warp_id, tid, 5` for better performance (shift vs divide).
**Added:** Phase 3 | **Sprint:** Phase 4.6 optimization
