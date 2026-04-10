# Sprint 2.3 — Comparisons + If/Else + `@!pred`

**Status:** Complete
**Commit:** (pending)

## Context

Sprint 2.2 delivered arithmetic expression lowering and body parsing.
Sprint 2.3 adds comparisons and control flow — the `if idx < n { ... }`
pattern that every GPU kernel needs for bounds checking.

## Decisions

### @!pred via negate: bool (ADR from Phase 2 master plan)

**Context:** Sprint 1.4 deferred predicate negation as not needed for
`vector_add`'s inverted-comparison pattern. Phase 2 if/else lowering
needs to conditionally skip code blocks.

**Decision:** Add `negate: bool` to `ControlOp::BraPred`. When true,
emits `@!%p bra target;`. The SetP comparison matches the source code
operator (`setp.lt` for `<`), and `@!pred bra` skips the then-block
when the condition is false. This reads naturally in PTX and was agreed
during the Phase 2 master plan review.

**Implementation:** One field, one format character in the emit match
arm. All existing call sites pass `negate: false` — zero behavior change.

### Label allocation order in if/else

**Context:** The if/else lowering allocates `IF_END` first, then `IF_ELSE`
(if needed). This means in an if/else block, `IF_END_0` and `IF_ELSE_1`.

**Decision:** Accepted. The label numbers don't matter for correctness —
only uniqueness matters. The allocation order follows code structure:
we know we need an end label before we know if we need an else label.

### lower_stmt handles Let registration

**Context:** `KernelStmt::Let` must both lower the value expression AND
register the resulting register+type in `ctx.locals` so subsequent
statements can reference the variable.

**Decision:** `lower_stmt` for Let calls `lower_expr(value)`, then
inserts `(reg_ident, type)` into `ctx.locals`. Duplicate variable names
produce a compile error. The `locals` HashMap was already added to
`LoweringContext` in Sprint 2.2.

## Scope

**In:**
- pyros-core: `negate: bool` on BraPred, emit `@!pred`, update 3 call sites
- pyros-macros: `lower/compare.rs`, comparisons wired into `lower_expr`,
  `lower_stmt`/`lower_stmts` with Let + If/Else handling, `fresh_label()`
- 8 new tests (1 pyros-core + 3 compare + 4 lower_stmt)

**Out:** Array indexing (Sprint 2.4), built-in functions (Sprint 2.5),
Assign/IndexAssign statement lowering (Sprint 2.4).

## Results

Completed as planned.

**Files created:** 1
- `pyros-macros/src/lower/compare.rs`

**Files modified:** 4
- `pyros-core/src/instr/control.rs` (negate field + emit + 1 new test)
- `pyros-core/tests/common/mod.rs` (negate: false)
- `pyros-runtime/tests/vector_add_e2e.rs` (negate: false)
- `pyros-macros/src/lower/mod.rs` (compare module, comparisons in
  lower_expr, lower_stmt/lower_stmts, fresh_label, 6 new tests)

**Tests:** 128 total (65 pyros-core + 64 pyros-macros), all passing.
- pyros-core: +1 (emit_bra_pred_negated)
- pyros-macros: +8 (3 comparison lowering, 1 label uniqueness,
  1 let registration, 1 if simple, 1 if/else, 1 comparison in lower_expr)

**Quality gates:** `cargo fmt --check` clean, `cargo clippy -- -D warnings`
zero warnings, all tests pass.
