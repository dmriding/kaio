# Sprint 3.1 — Loops + Compound Assignment

**Status:** Complete
**Date:** 2026-04-11
**Depends on:** Phase 2 complete (commit `50a3ab0`)

## Context

First sprint of Phase 3. Adds `for`/`while` loop support and compound
assignment (`+=`, `-=`, `*=`, `/=`) to the `#[gpu_kernel]` macro. These
are prerequisites for every subsequent Phase 3 sprint — reductions and
softmax all use loops and accumulators.

## Scope

**In:**
- `KernelStmt::For` and `KernelStmt::While` IR variants
- Parse `syn::Expr::ForLoop` with `start..end` range patterns
- Parse `syn::Expr::While` with condition + body
- Compound assignment desugaring (`+=`, `-=`, `*=`, `/=`)
- `KernelStmt::Assign` lowering (currently returns "not yet implemented")
- `For` loop lowering: counter init → label → bounds check → body →
  increment → back-edge
- `While` loop lowering: label → condition eval → branch → body → back-edge
- Parse + lower tests, compile-fail test updates, GPU E2E tests

**Out:**
- `loop` (stays rejected — no `break`/`continue`)
- `step_by()`, `break`, `continue`, inclusive ranges (`..=`)
- Shared memory, barriers, shuffles (Sprint 3.2+)

## Key Discoveries

### syn 2 compound assignment representation

**Critical finding:** In syn 2, `x += 1` is NOT `Expr::Assign` or a
separate `Expr::AssignOp`. It is `Expr::Binary` with `BinOp::AddAssign`.
Confirmed by checking syn 2 source (`op.rs`):

```rust
pub enum BinOp {
    Add(Token![+]),
    Sub(Token![-]),
    // ...
    AddAssign(Token![+=]),   // ← compound assignment lives here
    SubAssign(Token![-=]),
    MulAssign(Token![*=]),
    DivAssign(Token![/=]),
    RemAssign(Token![%=]),
    // ...
}
```

This means `x += 1` flows into `parse_expr_stmt` as a plain expression,
hits the `_ =>` catch-all, enters `parse_expr`, matches `Expr::Binary`,
calls `convert_binop`, and currently errors on the `_ =>` fallthrough
("unsupported binary operator").

**Desugaring strategy:** Handle compound assignment in `parse_expr_stmt`
BEFORE the catch-all. Check if an `Expr::Binary` has a compound assignment
operator, and if so, desugar:

```
x += expr     →  Assign { name: "x", value: BinOp(Var("x"), Add, expr) }
arr[i] += expr → IndexAssign { array, index, value: BinOp(Index(array, i), Add, expr) }
```

The LHS determines the statement type: `Expr::Path` → Assign,
`Expr::Index` → IndexAssign. The operator maps: `AddAssign` → `Add`,
`SubAssign` → `Sub`, `MulAssign` → `Mul`, `DivAssign` → `Div`,
`RemAssign` → `Rem`.

### Assign lowering strategy

`KernelStmt::Assign` lowering (line 363 in `lower/mod.rs`) currently
returns an error. In PTX, registers are mutable — `add.s32 %r0, %r0, 1`
overwrites `%r0` in place. But in the KAIO codegen model, each lowered
expression creates a new register:

```rust
let _kaio_r5 = alloc.alloc(PtxType::S32);
kernel.push(PtxInstruction::Arith(ArithOp::Add { dst: _kaio_r5, ... }));
```

For `x = new_expr`, we need to:
1. Lower `new_expr` → produces result in a fresh register `_kaio_rN`
2. Emit `Mov` from `_kaio_rN` to the register currently stored in
   `ctx.locals["x"]`
3. Keep the locals mapping unchanged (same register identity)

This works because in the generated PTX, `mov.s32 %r_x, %r_N` copies the
new value into x's register, and subsequent loop iterations reuse `%r_x`.

**Alternative considered:** Re-map locals to point to the new register.
Rejected because loop bodies need the variable identity to persist across
iterations — the same PTX register must be used on each pass.

**Note on compound self-assignment:** `x += x` desugars to
`x = x + x` → `Add { dst: fresh, lhs: existing_reg, rhs: existing_reg }`
then `Mov { dst: existing_reg, src: fresh }`. The Add reads existing_reg
twice BEFORE the Mov overwrites it — correct by construction. Pathological
cases like `x = x + (x += 1)` (nested compound assignment in expressions)
are impossible because compound assignment desugaring happens at the
statement level in `parse_expr_stmt`, never recursively within expression
trees. `parse_expr` never sees `AddAssign` — it would error if it did.

### For loop lowering pattern

```ptx
// for i in start..end { body }
mov.s32    %counter, start       // init
LOOP_START_N:
setp.ge.s32 %p, %counter, end   // bounds check
@%p bra    LOOP_END_N;           // exit if done
// body (with %counter as 'i')
add.s32    %counter, %counter, 1 // increment
bra        LOOP_START_N;         // back-edge
LOOP_END_N:
```

In codegen terms:
1. Lower `start` expr → `start_reg`
2. Lower `end` expr → `end_reg` (only once, before loop)
3. Allocate counter register, emit `Mov` from `start_reg`
4. Register counter as loop var in `ctx.locals`
5. Emit LOOP_START label
6. Allocate predicate, emit `SetP { cmp_op: Ge, ... }`
7. Emit `BraPred { negate: false, target: LOOP_END }`
8. Lower body statements
9. Emit `Add` to increment counter (in-place: dst = counter, lhs = counter)
10. Emit `Bra { target: LOOP_START }`
11. Emit LOOP_END label
12. Remove loop var from `ctx.locals` (scoped to loop body)

**Counter type:** The loop variable type is inferred from the range
bounds. If `start` is `U32` and `end` is `U32`, the counter is `U32`.
If `start` is a bare `0` (defaulting to `I32`) and `end` is a `U32`
variable, we need a type reconciliation strategy. For now: use the end
bound's type, and emit a `Cvt` for the start if types differ. This matches
the common pattern `for i in 0..n` where `n: u32`.

**Adopted approach:** Use the end bound's type as the counter type, with
literal coercion for the start bound. If `start` is an unsuffixed integer
literal (`LitInt` with empty suffix, e.g., `0`, `1`, `10`) and `end` has
a known type, coerce the literal's type to match `end`. This covers the
overwhelmingly common pattern `for i in 0..n` where `n: u32` without
requiring users to write `0u32..n`. Only unsuffixed literals are coerced —
suffixed literals (`0i32..n_u32`) and variables (`start_var..end_var`) with
mismatched types still produce a compile error. The coercion happens during
For lowering: check if `start` is `LitInt` with default type, and if so,
re-emit it with the end's type.

### While loop lowering pattern

```ptx
LOOP_START_N:
// evaluate condition → %pred
@!%pred bra LOOP_END_N;          // exit if false
// body
bra        LOOP_START_N;         // back-edge
LOOP_END_N:
```

Simpler than `for` — no counter, no increment. Just condition + body.

### Variable scoping in loops

Loop variables (`for i in ...`) should be scoped to the loop body. After
the loop, `i` should not be accessible. Implementation: save the locals
state before entering the loop, restore after. Or simpler: just remove
the loop variable from locals after the loop body is lowered.

For `while` loops, variables declared inside the body with `let` are also
loop-scoped. But `let mut x = 0; while x < n { x += 1; }` declares `x`
OUTSIDE the loop — it persists. This already works correctly because `let`
adds to `ctx.locals` at the point of declaration, and Assign updates the
value.

### let mut handling

Current `parse_let` extracts the variable name from `Pat::Ident` but
ignores the `mutability` field. This is correct — PTX registers are always
mutable. `let mut x = 0;` and `let x = 0;` produce the same IR. The
`mut` keyword is only needed so Rust's syntax rules allow `x += 1` later.

However, for correctness at the KAIO level, we should track mutability
and reject reassignment of non-`mut` variables. This would catch bugs
in user code. **Decision:** Defer mutability checking to Sprint 3.8
(polish). For now, all variables are implicitly mutable in KAIO kernels.

## Implementation Plan

### Step 1: KernelStmt variants (**done**)

Add `For` and `While` to `KernelStmt` in `kernel_ir/stmt.rs`.

Files: `kaio-macros/src/kernel_ir/stmt.rs`

### Step 2: Parse for/while loops

In `parse/body.rs`:

1. Handle `Expr::ForLoop` in `parse_expr_stmt`:
   - Extract loop variable from `Pat::Ident`
   - Extract range from inner `Expr::Range` (must be `HalfOpen`)
   - Parse start + end via `parse_expr`
   - Recurse body via `parse_body`
   - Return `KernelStmt::For`

2. Handle `Expr::While` in `parse_expr_stmt`:
   - Parse condition via `parse_expr`
   - Recurse body via `parse_body`
   - Return `KernelStmt::While`

3. Remove for/while rejection in `parse_expr` (lines 238-244).
   Update `loop` message to remove "available in Phase 3".

Files: `kaio-macros/src/parse/body.rs`

### Step 3: Parse compound assignment

In `parse_expr_stmt`, add a handler BEFORE the `_ =>` catch-all:

1. Match `Expr::Binary` where `op` is `AddAssign|SubAssign|MulAssign|
   DivAssign|RemAssign`
2. Extract LHS: if `Expr::Path` → variable name, if `Expr::Index` →
   array + index
3. Map compound op to base op: `AddAssign → Add`, etc.
4. Parse RHS via `parse_expr`
5. Construct desugared value: `BinOp(lhs_expr, base_op, rhs_expr)`
6. Return `Assign` or `IndexAssign` accordingly

Helper function: `desugar_compound_op(op: &BinOp) -> Option<BinOpKind>`
returns `Some(base_op)` for compound operators, `None` for others.

Files: `kaio-macros/src/parse/body.rs`

### Step 4: Lower Assign

In `lower/mod.rs`, replace the error at line 363 with:

1. Look up variable in `ctx.locals` → get `(existing_reg, ty)`
2. Lower the value expression → `(new_val_reg, val_ty, val_tokens)`
3. Emit `Mov { dst: existing_reg, src: Operand::Reg(new_val_reg), ty }`
4. Return combined tokens

The key: `existing_reg` stays the same in locals. The `Mov` copies the
new value into the existing register.

Files: `kaio-macros/src/lower/mod.rs`

### Step 5: Lower For

In `lower/mod.rs`, add `KernelStmt::For` to `lower_stmt`:

1. Lower `end` expr first → `(end_reg, end_ty)`
2. Lower `start` expr → `(start_reg, start_ty)`. If start is an unsuffixed
   literal and types differ, re-lower with end's type (literal coercion).
   If types still differ (both non-literal), error.
3. Allocate counter register with end's type, emit Mov from start_reg
4. Register counter var in `ctx.locals` with end's type
5. Generate LOOP_START and LOOP_END labels
6. Emit LOOP_START label
7. Allocate predicate register, emit `SetP { Ge, counter, end_reg }`
8. Emit `BraPred { pred, target: LOOP_END, negate: false }`
9. Lower body statements
10. Emit `Add { dst: counter, lhs: counter, rhs: Imm(1), ty }`
    (in-place increment)
11. Emit `Bra { target: LOOP_START }`
12. Emit LOOP_END label
13. Remove loop var from `ctx.locals`

Files: `kaio-macros/src/lower/mod.rs`

### Step 6: Lower While

In `lower/mod.rs`, add `KernelStmt::While` to `lower_stmt`:

1. Generate LOOP_START and LOOP_END labels
2. Emit LOOP_START label
3. Lower condition → `(pred_reg, _, cond_tokens)`
4. Emit `BraPred { pred: pred_reg, target: LOOP_END, negate: true }`
5. Lower body statements
6. Emit `Bra { target: LOOP_START }`
7. Emit LOOP_END label

Files: `kaio-macros/src/lower/mod.rs`

### Step 7: Tests

**Parse tests** (in `parse/body.rs` tests module):
- `parse_for_loop` — `for i in 0..n { out[i] = 0.0; }`
- `parse_while_loop` — `while x > 0 { x = x - 1; }`
- `parse_compound_assign_variable` — `x += 1` → Assign with BinOp
- `parse_compound_assign_index` — `arr[i] += val` → IndexAssign with BinOp
- `reject_for_inclusive_range` — `for i in 0..=n {}` → error
- `reject_for_iterator` — error on non-range patterns
- Update `reject_for_loop` → rename to test for-loops with bad patterns
- Update `reject_while_loop` → remove (while is now supported)
- Keep `reject_loop` — update assertion text

**Lower tests** (in `lower/mod.rs` tests module):
- `lower_assign` — reassignment emits Mov to existing register
- `lower_for_loop` — generated code contains LOOP_START, SetP, BraPred,
  Add increment, Bra back-edge, LOOP_END
- `lower_while_loop` — generated code contains LOOP_START, condition,
  BraPred with negate:true, Bra back-edge, LOOP_END
- `lower_nested_loops` — labels are unique across nesting levels

**GPU E2E tests** (new file: `kaio/tests/loops_macro.rs`):
- `sum_0_to_n` — for loop summing integers, verify against n*(n-1)/2
- `while_converge` — while loop halving until < threshold
- `strided_accumulate` — while loop with `i += stride`, tests compound
  assignment with index access

**Compile-fail updates:**
- `cf09_loop.rs` — keep `loop {}` rejection, update expected stderr
  (remove "available in Phase 3")
- Consider new CF: `for i in vec.iter() {}` → clear error about ranges

### Step 8: Quality gates

- `cargo fmt --check` — clean
- `cargo clippy --workspace -- -D warnings` — zero warnings
- `cargo test --workspace` — all host tests pass
- `cargo test --workspace -- --ignored` — GPU tests pass on RTX 4090

## Risks

| Risk | Mitigation |
|------|------------|
| syn `ForLoop` pat/expr destructuring | Tested in step 7 parse tests; only `Pat::Ident` + `Expr::Range(HalfOpen)` accepted |
| Loop var type inference mismatch | Unsuffixed literal start coerced to end's type. Suffixed/variable mismatches error. Covers `0..n` without friction. |
| Assign Mov ordering with compound assignment | Compound assignment desugared at parse time — lowering sees plain `Assign` with `BinOp` value, reads old value before writing new |
| Nested loop label collision | `fresh_label` uses monotonic counter — already proven unique in if/else tests |

## Dependencies

None — this sprint touches only kaio-macros. No kaio-core changes needed
(all required PTX instructions already exist: Mov, Add, SetP, BraPred,
Bra, Label).

## Results

Completed as planned.

**Bug found and fixed:** `let mut i = tid;` caused `i` and `tid` to share
the same PTX register. When `i += stride` overwrote the register, `tid`
was corrupted. Fix: `let` lowering now allocates a fresh register and
emits a `Mov` copy when the value expression is a bare variable reference
(empty TokenStream). This prevents aliasing between the new binding and
the source variable — critical for mutable loop counters.

**Files created:** 1
- `kaio/tests/loops_macro.rs` (3 GPU E2E tests)

**Files modified:** 4
- `kaio-macros/src/kernel_ir/stmt.rs` (For, While variants)
- `kaio-macros/src/kernel_ir/types.rs` (is_integer method)
- `kaio-macros/src/parse/body.rs` (for/while parsing, compound assignment
  desugaring, `desugar_compound_op` helper, updated rejection messages,
  10 new parse tests)
- `kaio-macros/src/lower/mod.rs` (Assign lowering, For lowering, While
  lowering, `coerce_literal_type` helper, let-binding copy-on-alias fix,
  6 new lower tests)
- `kaio/tests/compile_fail/cf09_loop.stderr` (updated error message)

**Tests:** 181 total (70 kaio-core + 109 kaio-macros + 2 other host),
all passing. 8 GPU tests (5 prior + 3 new), all passing on RTX 4090.

**Quality gates:** `cargo fmt --check` clean, `cargo clippy -- -D warnings`
zero warnings, all tests pass.
