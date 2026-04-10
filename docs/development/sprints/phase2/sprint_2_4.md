# Sprint 2.4 — Array Indexing + Memory Access

**Status:** Complete
**Commit:** `374530d`

## Context

High-risk sprint. Every kernel needs array indexing. A wrong `sizeof(T)`
or missing `cvta` produces garbage data, not a clear error. Standalone
address calculation tests (per Phase 2 master plan review) catch these
bugs before kernel-level testing.

## Decisions

### Address calculation matches Phase 1 E2E pattern

**Pattern:** `cvta.to.global → mul.wide (idx × sizeof(T)) → add.s64 → ld/st.global`

This is byte-for-byte the same pattern from `kaio-runtime/tests/vector_add_e2e.rs`.
No optimization, no divergence from the proven path.

### CvtaToGlobal cached, byte offsets not cached

**Context:** The E2E test reuses `rd_offset` across `a[idx]`, `b[idx]`,
`out[idx]` when the index is identical. Should the lowering do the same?

**Decision:** Cache `CvtaToGlobal` per pointer param (stored in
`ctx.global_addrs`), but do NOT cache byte offsets. Each index access
recomputes `mul.wide`. The PTX JIT does CSE (common subexpression
elimination) and will optimize redundant MulWide operations. This keeps
the lowering stateless with respect to index expressions and avoids the
complexity of detecting identical index expressions.

### Pointer lookup through ctx.locals with slice types

**Context:** The lowering for `a[idx]` needs to find the param register
for "a". Two options: (A) look up in `ctx.locals`, (B) separate `ctx.params` map.

**Decision:** (A) — look up in `ctx.locals`. Sprint 2.6 will populate
locals with the loaded param register and the full slice type
(`SliceRef(F32)` or `SliceMutRef(F32)`). The `KernelType::elem_type()`
method extracts the inner type. No separate `ParamInfo` struct needed.

### ImmU32 already exists in Operand

`Operand::ImmU32(u32)` was present since Sprint 1.1. The address
calculation uses `Operand::ImmU32(4)` for f32, `ImmU32(8)` for f64.
No new Operand variants needed.

### Mutability check on IndexAssign

**Decision:** `out[idx] = val` where `out` is `&[T]` (not `&mut [T]`)
produces a compile error: "cannot write to immutable slice parameter".
The check uses `KernelType::is_mut_slice()` from Sprint 2.1.

## Scope

**In:**
- `lower/memory.rs`: `compute_address`, `lower_index_read`, `lower_index_write`
- `LoweringContext.global_addrs` for cvta caching
- Wire `KernelExpr::Index` into `lower_expr`
- Wire `KernelStmt::IndexAssign` into `lower_stmt` with mutability check
- 10 new tests (7 in memory.rs, 4 integration in mod.rs — some overlap)

**Out:** Built-in functions (Sprint 2.5), launch wrapper (Sprint 2.6).

## Results

Completed as planned.

**Files created:** 1
- `kaio-macros/src/lower/memory.rs`

**Files modified:** 1
- `kaio-macros/src/lower/mod.rs` (`pub mod memory`, `global_addrs` field,
  Index in lower_expr, IndexAssign in lower_stmt, 4 integration tests)

**Tests:** 138 total (65 kaio-core + 74 kaio-macros), all passing.
- kaio-macros: +10 (3 standalone sizeof tests, 2 read/write, 1 cvta cache,
  1 index read integration, 1 index write integration, 1 immutability
  rejection, 1 scalar index rejection)

**Quality gates:** `cargo fmt --check` clean, `cargo clippy -- -D warnings`
zero warnings, all tests pass.
