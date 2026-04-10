# Sprint 2.5 — Built-in Functions

**Status:** Complete
**Commit:** `89a7a61`

## Context

Sprint 2.4 delivered array indexing. Sprint 2.5 adds the built-in function
registry — thread/block index functions needed by every kernel, and math
functions needed for `fused_gelu` in Sprint 2.8.

## Decisions

### Thread/block builtins generate `special::` helper calls

**Context:** pyros-core already has `special::tid_x()` etc. from Phase 1.
How should the macro's lowering call them?

**Decision:** Generate `let (reg, instr) = special::tid_x(&mut alloc);
kernel.push(instr);`. This reuses the existing helpers with zero
duplication. The generated code uses a two-ident pattern (`reg` + `instr`)
matching the helper's return type.

### Math builtins use `.approx` for transcendentals

**Decision:** `sqrt`, `ex2`, `lg2`, `rcp` all emit with `.approx.f32`
modifier. This matches GPU fast-math behavior and is standard for ML
workloads. Phase 5+ can add a precision attribute for IEEE-compliant modes.

### `exp(x)` = `2^(x * log2(e))` — standard GPU synthesis

**Decision:** Two instructions: `mul(x, LOG2_E)` → `ex2(result)`. This is
the standard GPU approach used by nvcc and Triton. LOG2_E constant
(`1.442695f32`) is loaded via `mov` with `ImmF32`.

### `tanh(x)` = `(exp(2x) - 1) / (exp(2x) + 1)` — register reuse

**Context:** The `exp(2x)` result must be used in both the numerator
(`sub`) and denominator (`add`). Computing it twice would be 10+
instructions and could give different results if the register allocator
assigns different physical registers.

**Decision:** Compute `exp(2x)` once into a single register (`exp2x`),
then reference that register in both the `Sub` and `Add` instructions.
The test `lower_tanh_synthesized` explicitly verifies `Ex2` appears
exactly once in the generated code.

### Fma deferred — not needed for Sprint 2.8 kernels

**Context:** The Phase 2 master plan lists `ArithOp::Fma` with
`fma.rn{ty}`. None of the four E2E kernels (vector_add, saxpy,
fused_relu, fused_gelu) require it.

**Decision:** Deferred. Can be added in Phase 3+ when needed. Not an
omission — explicitly scoped out.

## Scope

**In:**
- pyros-core: 7 new ArithOp variants (Abs, Min, Max, Sqrt, Ex2, Lg2, Rcp)
- pyros-macros: `lower/builtins.rs` with 12 thread/block builtins +
  7 math builtins (sqrt, abs, min, max, exp, log, tanh)
- Wire `BuiltinCall` into `lower_expr`
- 18 new tests (7 pyros-core + 11 pyros-macros)

**Out:** `ArithOp::Fma` (deferred, not needed for Phase 2 E2E kernels).

## Results

Completed as planned.

**Files created:** 1
- `pyros-macros/src/lower/builtins.rs`

**Files modified:** 2
- `pyros-core/src/instr/arith.rs` (7 new variants + Emit + 7 tests)
- `pyros-macros/src/lower/mod.rs` (`pub mod builtins`, BuiltinCall in lower_expr)

**Tests:** 157 total (72 pyros-core + 85 pyros-macros), all passing.
- pyros-core: +7 (abs, min, max, sqrt, ex2, lg2, rcp emit tests)
- pyros-macros: +11 (thread_idx_x, block_idx_x, block_dim_x, sqrt, abs,
  min_max, exp synthesis, log synthesis, tanh synthesis with register
  reuse check, unknown builtin rejection, arg count rejection)

**Quality gates:** `cargo fmt --check` clean, `cargo clippy -- -D warnings`
zero warnings, all tests pass.
