# Sprint 4.7 — PTX Inspection & Performance Documentation

**Status:** Complete
**Commit:** (pending)
**Date:** 2026-04-11
**Depends on:** Sprint 4.6

## Context

The original 4.7 planned compile-time coalescing analysis assuming
`block_load`/`block_store` abstractions existed as explicit API
surfaces. They don't — the matmul uses raw `shared_mem!` and manual
index math. Building a static analyzer for arbitrary index expressions
is a research problem, not a sprint.

Rescoped to developer tooling: PTX statistics, source annotations
in PTX output, and a performance patterns guide.

## Completed Items

### PTX Statistics (`KAIO_PTX_STATS=1`) — Done

Added `KernelStats` struct and `stats()` method to `PtxKernel` in
kaio-core. Walks the instruction body and counts instruction types.
Reported via `eprintln!` when `KAIO_PTX_STATS=1` is set at runtime.

Counts:
- Arithmetic: FMA vs other (add, mul, sub, etc.)
- Memory: ld.global, st.global, ld.shared, st.shared
- Control: bar.sync, branches, setp, mov, cvt
- Registers by kind: r32, r64, f32, f64, pred
- Shared memory bytes from declarations

Output includes "(PTX structure, not runtime profile)" framing to
prevent users from overreading counts as performance truth.

### PTX Annotation Mode (`KAIO_PTX_ANNOTATE=1`) — Done

When set, injects `PtxInstruction::Comment` at statement boundaries
in generated PTX. Maps source constructs to their PTX output.

High-signal annotations only — structural boundaries, not every
trivial assignment:
- `shared_mem!` declarations
- `while` / `for` / `if` blocks
- Index assignments (`array[...] = ...`)
- `let` bindings
- `bar_sync()` calls

Implementation uses a helper function `annotation_tokens()` to avoid
copy-paste drift across match arms. The `_kaio_annotate` variable is
a bare identifier in `quote!` blocks (no `#` prefix) — it resolves
at runtime in the generated `build_ptx()`, same pattern as `kernel`
and `alloc`.

### Performance Guide (`docs/performance.md`) — Done

New document covering:
1. Memory coalescing — good/bad code examples
2. Shared memory bank conflicts — stride padding explanation
3. Register tiling — concept, tradeoffs, link to implementation
4. PTX inspection tools — all three env vars with usage examples
5. Annotated PTX excerpt — concrete example of what annotation
   mode produces
6. Naive vs optimized stats comparison — demonstrates why stats
   feature exists

### Benchmarks Update — Done

Added "Performance Status" section to `docs/benchmarks.md`:
- Current: 31% of cuBLAS sgemm — BM=BN=64, BK=16, TM=TN=4
- Planned: vectorized loads, double buffering, size-based dispatch

## Key Decisions

1. **Deferred coalescing analysis to Phase 5+** — requires
   `block_load`/`block_store` abstractions with analyzable access
   patterns. Analyzing arbitrary `array[complex_expression]` at
   macro expansion time is a research problem.

2. **High-signal annotations only** — dropped `Assign` and generic
   `Expr` annotations per review feedback. Bar_sync is the
   only bare expression annotated.

3. **Stats framed as structural, not performance** — doc comments
   and output explicitly state these are PTX structure counts,
   not runtime profiling data.

## Tests

+4 host tests in `kaio-core/src/ir/kernel.rs`:
- `stats_empty_kernel`
- `stats_counts_instruction_types`
- `stats_counts_registers_by_kind`
- `stats_counts_shared_bytes`

Total: 207 host tests + 41 GPU tests + 1 benchmark

## Files Modified

| File | Changes |
|------|---------|
| `kaio-core/src/ir/kernel.rs` | `KernelStats` struct, `stats()` method, 4 tests |
| `kaio-core/src/ir/mod.rs` | Export `KernelStats` |
| `kaio-macros/src/codegen/ptx_builder.rs` | `KAIO_PTX_STATS` + `KAIO_PTX_ANNOTATE` |
| `kaio-macros/src/lower/mod.rs` | `annotation_tokens()` helper, annotations in 7 arms |
| `docs/performance.md` | New: performance patterns guide |
| `docs/benchmarks.md` | Performance Status section |
| `docs/development/sprints/phase4/sprint_4_7.md` | This file |
| `docs/development/sprints/phase4/PHASE_4_LOG.md` | Status update |
| `docs/development/sprints/phase4/phase4_master_plan.md` | Renumber sprints |
