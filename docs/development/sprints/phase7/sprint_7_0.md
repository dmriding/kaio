# Sprint 7.0 — DSL completeness + Phase 6 closeout + Phase 7 scaffold

**Status:** Complete (v0.2.1, 2026-04-14 — combined release with Sprint 6.10)
**Branch:** `phase7` off `main` (post v0.2.1)
**Master plan:** [phase7_master_plan.md](phase7_master_plan.md)

## Context

Phase 6 shipped v0.2.1 (Sprint 6.10 closeout merged). Two DSL gaps
blocked Phase 7 quantized-kernels work from starting cleanly:

1. **No bitops.** Quant dequantization is `(packed >> shift) & mask`
   on `u32` / `u64` storage. The `#[gpu_kernel]` parser already
   accepted `&` `|` `^` `<<` `>>` `!`, but lowering rejected all of
   them with "operator lowering not yet implemented."
2. **No short-circuit `&&` / `||`.** Users coming from Rust will write
   `if i < n && arr[i] > 0` as the natural bounds-guarded access
   pattern. Without short-circuit semantics, this errors outright — or
   worse, could silently OOB-read on the last thread if a bitwise
   fallback were used.

Docs were also out of sync: `docs/phases.md` still framed Phase 6 as
"Post-v0.1 Roadmap / not a commitment," predating v0.2.0's ship. Phase
7 had no scaffolding.

## Goal

Ship v0.2.1 (combined with Sprint 6.10) with:

- Full bitwise operator support in `#[gpu_kernel]` (binary `&` `|` `^`
  `<<` `>>`, unary `!`, compound `&=` `|=` `^=` `<<=` `>>=`)
- Short-circuit `&&` / `||` — Rust-faithful semantics
- Phase 6 formally closed in docs; Phase 7 master plan + this sprint
  log present
- Zero regressions on existing host + GPU test suites

## Deliverables

### D1 — `kaio-core` bitwise `ArithOp` variants

Added six variants to `ArithOp`:

- `And`, `Or`, `Xor`, `Not` — typeless on signedness; emit
  `.b{size}` or `.pred` via `PtxType::reg_decl_type`.
- `Shl` — typeless (`.b{size}`). Bit-width-only; PTX does not
  distinguish signed left shift.
- `Shr` — preserves signed/unsigned distinction. `i32 >> n` emits
  `shr.s32` (arithmetic / sign-extend). `u32 >> n` emits `shr.u32`
  (logical / zero-extend). This is the AD2 canary at the IR level;
  quant INT8 dequant on signed packed values depends on it.

Tests (11 new unit tests + 1 ptxas_verify integration):
- Emit-string assertions per variant.
- Explicit signed vs unsigned `Shr` round-trip
  (`emit_shr_s32_arithmetic`, `emit_shr_u32_logical`).
- Bitwise on `Pred` (`and.pred`, `not.pred`) for future use by the
  D4 expression-position short-circuit materialization.
- `ptxas_verify_bitops` kernel exercising every new variant.

**Commit:** `9e9a681` — `wip(phase7): D1 — kaio-core bitwise ArithOp
variants (And/Or/Xor/Shl/Shr/Not)`

### D2 — Macro lowering for bitwise binops + unary NOT

- `BinOpKind::is_bitwise()` and `is_logical()` predicates.
- `lower::arith::lower_bitop` dispatches `BitAnd` / `BitOr` / `BitXor`
  / `Shl` / `Shr` to the matching `ArithOp` variant. Signed vs
  unsigned `Shr` is resolved from `lhs_ty`.
- `lower::arith::lower_not` handles both integer (bitwise) and bool
  (logical) cases via `lower_expr`'s type propagation (AD3).
- Bitwise operator on non-integer operands produces a clear
  compile-time error.

Tests (6 new macro-level, 5 new GPU round-trip):
- Macro-level: bitwise op dispatch, Shr signedness preservation (both
  `i32` and `u32` cases), `!bool` → Pred, `!u32` → b32.
- GPU: `bitops_all` (AND/OR/XOR end-to-end), `shift_left`,
  `shr_logical_u32_zero_extends` (AD2 signature canary logical half —
  `0xFFFFFFFF >> 1 == 0x7FFFFFFF`), `shr_arithmetic_i32_sign_extends`
  (AD2 signature canary arithmetic half — `-2 >> 1 == -1`),
  `not_bitwise_u32_flips_bits`.

**Commit:** `ca79596` — `wip(phase7): D2 — macro lowering for bitwise
operators and unary NOT`

### D3 — Compound bitwise assignment

Five new entries in `desugar_compound_op`: `BitAndAssign` → `BitAnd`,
`BitOrAssign` → `BitOr`, `BitXorAssign` → `BitXor`, `ShlAssign` →
`Shl`, `ShrAssign` → `Shr`.

Because the parser already desugars `arr[i] += val` to `IndexAssign {
value: BinOp(Index(...), Add, val) }`, bitwise compound assigns fall
out for free once `desugar_compound_op` recognizes them — no new IR
statement type, no new lowering code.

Tests (6 new GPU round-trip):
- Indexed: `|=`, `&=`, `^=`, `<<=`, `>>=` against known inputs.
- Scalar: `acc |= mask; acc &= other_mask;` chain, verifying scalar
  compound bitwise works in addition to indexed.

**Commit:** `0977ee1` — `wip(phase7): D3 — compound bitwise assignment
on scalar and indexed lvalues`

### D4 — Short-circuit `&&` / `||`

Two lowering paths sharing a builder (`kaio-macros/src/lower/logical.rs`):

- **If-condition position** — branch-direct. Detected in the
  `KernelStmt::If` arm when the condition is `BinOp { And | Or, .. }`.
  Emits sequential `setp` + conditional `bra` directly to the
  `skip_label` supplied by the if-lowering. No intermediate `p_out`
  register. For `&&`, short-circuit on LHS false jumps to skip.
  For `||`, short-circuit on LHS true jumps to a local TAKE label
  past the RHS eval (fall-through is also a take).
- **Expression position** — materialized. Reached via the normal
  `BinOp` arm when `op.is_logical()`. Allocates a fresh `.pred`
  register; emits `mov.pred p_out, p_lhs; @!p_lhs bra DONE_k;
  <rhs>; mov.pred p_out, p_rhs; DONE_k:`. The materialized predicate
  flows to downstream consumers (stored, compared, passed to another
  op).

Both paths preserve Rust's short-circuit semantics: if LHS determines
the result (`a == false` for `&&`, `a == true` for `||`), RHS is
never evaluated. This is why `if i < n && arr[i] > 0` is safe as a
bounds-guarded access — the `arr[i]` read only fires when `i < n`.

Tests (6 new GPU round-trip, 4 new codegen regression):
- AD4 signature canary: `logical_and_bounds_guard_no_oob` — launches
  with `i == n` on the last thread, verifies `arr[n]` is not OOB-read.
- Early-success `||` with a flag controlling whether LHS is true.
- Expression-position materialization with all four `(LHS, RHS)`
  permutations.
- Nested `if (a && b) || c { ... }` — label allocation stress.
- Expression-position nested `let m = (a && b) || (c && d);` —
  four-way label interleaving.
- Codegen regression: materialized path emits `LOGICAL_DONE` label +
  `PtxType::Pred` Mov; branch-direct path emits `BraPred` only, with
  no `PtxType::Pred` Mov.

**Commit:** `f37dc11` — `wip(phase7): D4 — short-circuit && and || with
Rust-faithful semantics`

### D6 — Phase 6 closeout + Phase 7 scaffold

Pure docs.

- `docs/phases.md` — Phase 6 promoted from "Post-v0.1 Roadmap" to a
  full `## Phase 6: Tensor Cores & Async Copies ✅` section matching
  Phase 4 / 5 format (Status / Deliverables / Sprint Breakdown / Key
  Decisions / Performance). Phase 7 section added with sprint outline
  pointing at `phase7_master_plan.md`.
- `docs/development/sprints/phase7/phase7_master_plan.md` — created,
  mirrors the Phase 5 / Phase 6 master plan template.
- `docs/development/sprints/phase7/sprint_7_0.md` — this document.
- `docs/success-criteria.md` — Phase 6 section added (post-hoc) with
  measured outcomes. Phase 7 placeholder section added.
- `CHANGELOG.md` — `[Unreleased]` from v0.2.1 promoted to `[0.2.1]`
  proper, new `[Unreleased]` section created for Sprint 7.0 bullets.
  Duplicate `[0.2.0]` header artifact removed.
- `docs/development/tech_debt.md` — three DSL items marked
  **RESOLVED Sprint 7.0**: `&&` / `||` logical operators, compound
  assignment for shared memory (bitwise variants), `ArithOp::Shr /
  Shl / And / Or` bitops.

## Results

### Test counts

| Suite | Before | After | Delta |
|-------|--------|-------|-------|
| `kaio-core` lib | 140 | 151 | +11 bitops emit unit tests |
| `kaio-macros` lib | 122 | 134 | +2 predicate (`is_bitwise` / `is_logical`) + 6 `lower_bitop` / `lower_not` + 4 codegen regression canaries |
| `kaio-ops` lib | 24 | 24 | unchanged |
| `kaio-core` ptxas_verify (`--ignored`) | 6 | 7 | +`ptxas_verify_bitops` |
| `kaio` GPU tests (`--ignored`) | baseline | +17 | 5 bitops + 6 compound bitops + 6 short-circuit |

### Gates

- `cargo fmt --all --check` — clean.
- `cargo clippy --workspace --all-targets -- -D warnings` — clean.
- `cargo test --workspace` — all host tests green.
- `cargo test --workspace -- --ignored` on RTX 4090 sm_89 — all GPU
  tests green including new 7.0 additions.
- `cargo test -p kaio-core --test ptxas_verify -- --ignored` — 7/7.
- `matmul_tc_bench` at 4096² — async median in the expected 2.1–2.4ms
  band (no regression).

### Scope folded / deferred

- No quant-specific IR primitives — per 7.0 non-goals. Phase 7.1+.
- `MemoryOp::LdGlobalB128` remains unused; natural home is INT8 / INT4
  matmul (7.1+).
- Short-circuit optimization pass (prove-both-sides-safe →
  `and.pred` / `or.pred`) not written. Additive, backward-compatible
  future work.
- 14 Phase 6 sprint files did not receive retroactive `✅ COMPLETE`
  blocks per plan (PHASE_6_LOG.md already carries completion status;
  per-file blocks are busywork).

## Commits

| Commit | Scope |
|--------|-------|
| `9e9a681` | D1 — kaio-core bitwise `ArithOp` variants |
| `ca79596` | D2 — macro lowering for bitwise operators + unary NOT |
| `0977ee1` | D3 — compound bitwise assignment |
| `f37dc11` | D4 — short-circuit `&&` / `\|\|` |
| _(D6 commit)_ | docs closeout + Phase 7 scaffold |
| _(final)_ | `feat(phase7): Sprint 7.0` — version bump + release notes |

## Carry-forward

- **Sprint 7.1 (INT8 dequantize-matmul)** — next up. See
  [phase7_master_plan.md](phase7_master_plan.md) §2 for the design
  starting point. Open question at 7.1 kickoff: does the
  `mma.sync.m16n8k16.s8.s8.s32` shape behave as expected on Ampere+,
  skipping the dequant-to-f16 step?
- Env-var hygiene for the remaining 3 test helpers reading
  `KAIO_SM_TARGET` (`build_vector_add_ptx`, `build_shared_mem_ptx`,
  `build_ld_global_b128_ptx`). Their callers don't mutate env so no
  correctness issue — minor consistency cleanup, low priority.
- Short-circuit optimization pass (flagged by owner during planning):
  when both sides of `&&` / `||` are provably pure (no side effects,
  no OOB-dependent array access), emit `and.pred` / `or.pred` instead
  of branches. Additive, no semantic change.
