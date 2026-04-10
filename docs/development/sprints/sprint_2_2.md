# Sprint 2.2 — Arithmetic Expression Lowering

**Status:** Complete
**Commit:** `5468051`

## Context

Sprint 2.1 delivered the `pyros-macros` crate skeleton with attribute and
signature parsing. Sprint 2.2 builds the two key pieces needed for
expression codegen: new arithmetic instructions in `pyros-core`, and the
body parsing + expression lowering pipeline in `pyros-macros`.

This sprint establishes the "generate code that builds IR" pattern: the
macro's lowering pass produces `TokenStream` fragments that, when compiled,
call `pyros-core`'s `RegisterAllocator`, `PtxInstruction`, and `ArithOp`
constructors. Every subsequent sprint follows this same pattern.

## Decisions

### Mul vs MulWide — when to use which

**Context:** Phase 1 has `ArithOp::MulWide` (`mul.wide.{ty}`) for widening
multiply (32→64 bit). Phase 2 adds `ArithOp::Mul` for same-width multiply.
Both are valid PTX, but the lowering must choose correctly.

**Decision:** `BinOpKind::Mul` (the `*` operator in user code) ALWAYS lowers
to `ArithOp::Mul`, NEVER `MulWide`. `MulWide` is reserved for address offset
calculation in Sprint 2.4 (32-bit index × sizeof(T) → 64-bit byte offset).
A comment in `lower/arith.rs` documents this invariant.

### Integer Mul emits `mul.lo`

**Context:** PTX `mul` for integers produces a 2N-bit product. To get a
same-width result, you must specify `.lo` (low N bits) or `.hi` (high N bits).
Floats have no mode.

**Decision:** The `Emit` impl checks the type: floats emit `mul.f32` /
`mul.f64`, integers emit `mul.lo.s32` / `mul.lo.u32` / etc. This is handled
entirely in `pyros-core`'s emit logic — the macro's lowering doesn't need
to know about it.

### LoweringContext includes locals map from Sprint 2.2

**Context:** The original plan had a minimal `LoweringContext` with just
counters. But `lower_expr` needs to resolve `KernelExpr::Var("x")` to
the register `Ident` holding that variable's value. Without a locals map,
the expression evaluator can't function.

**Decision:** Add `locals: HashMap<String, (Ident, KernelType)>` to
`LoweringContext` immediately. For Sprint 2.2 tests, we pre-populate
the map with known variables. Sprint 2.6's codegen populates it from
parameter loading and let-binding lowering.

### lower_expr as the recursive orchestrator

**Context:** `lower_binop` takes pre-resolved register `Ident`s. Something
must walk the `KernelExpr` tree, resolve leaves to registers, and call
`lower_binop` with those registers.

**Decision:** `lower_expr` is the recursive expression evaluator, built in
this sprint. It handles: `Var` (lookup), `LitInt`/`LitFloat` (alloc + mov),
`BinOp` (recurse + lower_binop), `UnaryOp::Neg` (recurse + lower_neg),
`Paren` (recurse into inner). Unimplemented variants (`Index`, `BuiltinCall`,
`Cast`, comparisons) return descriptive "not yet implemented (Sprint X)" errors.

### Body parser maps syn exhaustively

**Context:** The body parser converts `syn::Expr` variants to `KernelExpr`.
How many syn variants should we handle?

**Decision:** Map ALL relevant syn variants now (Binary, Unary, Lit, Path,
Index, Call, Cast, Paren, If, Assign, Block) and reject all others with
specific error messages (ForLoop, While, Loop, Match, Closure, Return,
Unsafe, Macro, MethodCall, Struct, Tuple, Range, Try). This avoids
revisiting the parser in later sprints.

### Literal type defaults match Rust conventions

**Decision:** Unsuffixed integer literals default to `I32`, unsuffixed
float literals default to `F32`. Suffixed literals (`42u32`, `1.0f64`)
map to the specified type. This matches Rust's behavior (though Rust
defaults to `i32`/`f64` — we default floats to `f32` since GPU kernels
overwhelmingly use single precision).

### `#![allow(dead_code)]` at module level for body parser

**Context:** Every function in `parse/body.rs` triggers dead_code warnings
because none are called from the macro entry point yet (that happens in
Sprint 2.6). Individual `#[allow]` on each function would be noisy.

**Decision:** Module-level `#![allow(dead_code)]` with a comment referencing
Sprint 2.6 codegen. This is the cleanest approach for a module where ALL
functions are future API surface.

## Scope

**In:**
- pyros-core: Sub, Mul, Div, Rem, Neg ArithOp variants + Emit + 11 tests
- pyros-macros: KernelExpr, KernelStmt, BinOpKind, UnaryOpKind types
- pyros-macros: parse/body.rs — full syn-to-KernelIR conversion
- pyros-macros: lower/mod.rs — LoweringContext + lower_expr orchestrator
- pyros-macros: lower/arith.rs — lower_binop + lower_neg
- 33 new pyros-macros tests, 11 new pyros-core tests

**Out:** Comparison lowering (Sprint 2.3), if/else lowering (Sprint 2.3),
array indexing (Sprint 2.4), built-in functions (Sprint 2.5), launch wrapper
codegen (Sprint 2.6), type validation (Sprint 2.7).

## Results

Completed as planned.

**Files created:** 5
- `pyros-macros/src/kernel_ir/expr.rs`
- `pyros-macros/src/kernel_ir/stmt.rs`
- `pyros-macros/src/parse/body.rs`
- `pyros-macros/src/lower/mod.rs`
- `pyros-macros/src/lower/arith.rs`

**Files modified:** 4
- `pyros-core/src/instr/arith.rs` (5 new variants + Emit + 11 tests)
- `pyros-macros/src/kernel_ir/mod.rs` (add expr, stmt exports)
- `pyros-macros/src/parse/mod.rs` (add body module)
- `pyros-macros/src/lib.rs` (add `mod lower`)

**Tests:** 120 total (64 pyros-core + 56 pyros-macros), all passing.
- pyros-core: +11 (emit_sub_s32, emit_sub_f32, emit_mul_lo_s32, emit_mul_f32,
  emit_mul_lo_u32_with_immediate, emit_div_f32, emit_div_s32, emit_rem_u32,
  emit_neg_f32, emit_neg_s32, sub_via_ptx_instruction)
- pyros-macros: +33 (17 body parsing, 2 expr type, 4 arith lowering,
  8 lower_expr orchestrator, 2 BinOpKind classification)

**Quality gates:** `cargo fmt --check` clean, `cargo clippy -- -D warnings`
zero warnings, all tests pass.
