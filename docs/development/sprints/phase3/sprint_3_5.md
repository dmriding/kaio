# Sprint 3.5 — Reduction Primitives (block_reduce_sum / block_reduce_max)

**Status:** Complete
**Commit:** `24efd87`
**Date:** 2026-04-11
**Depends on:** Sprint 3.4 (bar_sync, shfl_sync builtins)

## Context

Most complex sprint in Phase 3. Adds `block_reduce_sum(val)` and
`block_reduce_max(val)` as multi-instruction built-in expansions (~35 PTX
instructions each). These are the core building blocks for softmax
(Sprint 3.6).

## Decisions

### block_size in LoweringContext

**Context:** Reduction needs `num_warps = block_size / 32` as a compile-time
constant for `setp` instructions.

**Decision:** Added `block_size: Option<u32>` to LoweringContext, set from
`sig.config.block_size` in `generate_build_ptx()`. Reductions read
`ctx.block_size.unwrap()`.

### Single shared allocation reused

**Context:** Softmax calls both `block_reduce_max` and `block_reduce_sum`.
Each completes (including broadcast) before the next starts.

**Decision:** `_kaio_reduce_smem` allocated on first reduction call, reused
by subsequent calls. Tracked by `ctx.reduce_smem_allocated: bool`.

### Broadcast to all threads

**Context:** Softmax needs every thread to know `row_max` and `row_sum`.

**Decision:** After the cross-warp reduction, thread 0 writes final result
to `shared[0]`, all threads read after `bar.sync`. Every thread gets the
reduction result — no separate broadcast step needed by the caller.

### warp_id via div.u32

**Context:** No `ArithOp::Shr` in current instruction set.

**Decision:** `div.u32(tid, 32)` computes warp_id. Correct but slower than
`shr` on GPU hardware. PTX JIT may optimize. Phase 4 optimization: add
`ArithOp::Shr`, replace div-by-power-of-2 with shift.

### Unique labels via fresh_label

**Context:** Multiple reduction calls in one kernel need distinct labels
for conditional branches.

**Decision:** All internal labels (`REDUCE_WRITE_DONE`, `REDUCE_BROADCAST`,
`REDUCE_LOAD_DONE`, `REDUCE_T0_DONE`) use `ctx.fresh_label()` with monotonic
counter — guaranteed unique across calls.

## Scope

**In:**
- `block_reduce_sum(val: f32) -> f32` — sum across block with broadcast
- `block_reduce_max(val: f32) -> f32` — max across block with broadcast
- `lower_block_reduce()` helper (~200 lines of token generation)
- Auto-allocated `_kaio_reduce_smem` shared memory
- IDE stubs, cf04 stderr update
- GPU E2E: sum of ones = 256.0, max finding 999.0

**Out:**
- Sub-warp reductions (width < 32)
- u32 reduction type (f32 only for now)
- ArithOp::Shr optimization (Phase 4)

## Results

Completed.

**Files created:** 1
- `kaio/tests/reduce_macro.rs` (2 GPU E2E tests)

**Files modified:** 4
- `kaio-macros/src/lower/mod.rs` (block_size, reduce_smem_allocated fields)
- `kaio-macros/src/lower/builtins.rs` (lower_block_reduce, dispatch, error msg)
- `kaio-macros/src/codegen/ptx_builder.rs` (set ctx.block_size)
- `kaio/src/gpu_builtins.rs` (block_reduce_sum/max stubs)
- `kaio/tests/compile_fail/cf04_unknown_call.stderr` (updated error msg)

**Tests:** 198 host + 14 GPU, all passing on RTX 4090.

**Quality gates:** `cargo fmt --check` clean, `cargo clippy -- -D warnings`
zero warnings, all tests pass.
