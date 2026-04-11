# Sprint 5.1 — 2D Block Reductions (HARD GATE)

**Status:** Done
**Branch:** phase5
**Goal:** Enable `block_reduce_sum/max` in 2D kernels by computing
linear thread identity.

## Problem

`block_reduce_sum/max` were rejected at compile time for 2D kernels.
The reduction tree derived thread identity from `TidX` only — in 2D
blocks, threads in different rows but the same column got the same
`tid`, breaking warp ID and lane calculations.

This was the single blocker for Sprint 5.2 (standard attention).

## Fix

One function, one change: `lower_block_reduce()` in
`kaio-macros/src/lower/builtins.rs`.

For 2D kernels, compute `linear_tid = tidx + tidy * block_dim_x`
(row-major, x is fast-varying dimension). For 1D kernels, `tid = TidX`
unchanged — zero extra instructions, zero regression risk.

The decision between 1D and 2D codegen happens at macro expansion time
via `ctx.block_size_x.is_some()`, not at runtime.

### Changes

1. Removed the 2D rejection guard (`if ctx.block_size_y.is_some()`)
2. Added conditional `tid_tokens` construction before the main `quote!`
   block — 2D path emits 4 extra PTX instructions (mov TidX, mov TidY,
   mul, add), 1D path unchanged
3. Deleted `cf11_2d_reduce_rejected` compile-fail test (no longer errors)

### Verified

- `num_warps` and `_kaio_reduce_smem` sizing already use `ctx.block_size`
  (total = x * y), not `block_size_x`. No change needed.
- All downstream uses of `#tid` (warp_id, lane-0 check, first-warp
  check, thread-0 check, cross-warp indexing) flow from the single
  `#tid` variable — replacing its computation propagates automatically.

## New Tests

| Test | Block Size | What It Verifies |
|------|-----------|------------------|
| `block_reduce_sum_2d_16x16` | (16,16) | Sum of 1..256 via global load + reduce |
| `block_reduce_max_2d_16x16` | (16,16) | Max of 1..256 via global load + reduce |
| `block_reduce_sum_2d_asymmetric_32x8` | (32,8) | Non-square block geometry |
| `block_reduce_sum_2d_identity_based` | (16,16) | Value = tidx*100 + tidy — catches row aliasing |

## Regression

All existing 1D tests must pass unchanged:
- `reduce_sum_ones`, `reduce_max_one_high` (reduce_macro.rs)
- 10 softmax tests (softmax_macro.rs)
- `shared_array_plus_reduction` (shared_mem_macro.rs)

## Files

| File | Change |
|------|--------|
| `kaio-macros/src/lower/builtins.rs` | Remove guard, add linear tid |
| `kaio/tests/reduce_macro.rs` | +4 GPU tests |
| `kaio/tests/compile_fail/cf11_2d_reduce_rejected.rs` | Deleted |
| `kaio/tests/compile_fail/cf11_2d_reduce_rejected.stderr` | Deleted |
| `docs/development/tech_debt.md` | Marked resolved |

## Review Notes

- Opus 4.6: verify `num_warps` uses total block_size (confirmed)
- Codex 5.4: add asymmetric block test, identity-based test, document
  linearization convention, verify no raw TidX leaks in PTX
