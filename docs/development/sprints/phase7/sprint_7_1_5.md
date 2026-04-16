# Sprint 7.1.5 — Warp + block reductions in the DSL

**Status:** ✅ Complete
**Branch:** `phase7-rest` off `main` (shared across Phase 7 remaining sprints;
no independent crates.io release — Phase 7 closes with an aggregate release).
**Release target:** None this sprint. Part of the eventual Phase 7 aggregate
release when sprints 7.1.5 → 7.4 are all complete.

## Context

Sprint 7.1 shipped `matmul_int8` (v0.3.0) — the quant headline. 7.1.5 is the
standalone "bridge" sprint between quant milestones. It lands the warp-level
reduction primitives that an external convergence review flagged as the single
biggest expressiveness gap in the DSL after matmul: without warp reductions,
users can't write softmax / layer-norm / RMS-norm / loss functions in pure
DSL. Block-level reductions are already shipped (Phase 3,
`block_reduce_sum/max(f32)`), but the warp layer underneath them is not
exposed, and `min` is missing at both warp and block level.

Sequencing rationale: strict numeric-order ship preference. 7.1.5 ships before
7.2 INT4 even though it doesn't block 7.2 directly (INT4 matmul is IR-authored
in `kaio-ops`, not DSL-authored). Closes a real expressiveness gap before
more quant work stacks on top.

## Scope

**This sprint adds:**
- `warp_reduce_sum(f32) -> f32`
- `warp_reduce_max(f32) -> f32`
- `warp_reduce_min(f32) -> f32`
- `block_reduce_min(f32) -> f32`

**Explicitly OUT of scope:**
- Type generalization beyond f32 (i32/u32/f16/bf16 reductions are a future sprint)
- `shfl.sync.idx` IR variant (not needed for reductions)
- Atomics, dynamic shared memory (adjacent convergence items, standalone sprints)
- New public showcase example (`examples/softmax/` already demonstrates block_reduce)

## Deliverables

- **D1** — IR audit, no-op confirm. `ShflSync {Down, Up, Bfly}` already present
  in `kaio-core/src/instr/control.rs`; no new IR variants needed. Sprint stub
  lands here.
- **D2.0** — Pre-refactor PTX snapshot canary (captured from main-tip state
  BEFORE any refactor starts, otherwise it tests against itself).
- **D2** — Factor `emit_warp_tree_reduce` helper out of `lower_block_reduce`.
  Parameterized by shuffle variant (Down for block_reduce byte-identical;
  Bfly for warp_reduce all-lanes-get-result), combine op, and PTX type.
  Snapshot canary must stay green.
- **D3** — `warp_reduce_{sum,max,min}` DSL builtins + whole-warp-multiple
  compile-time guard (total threads must be a multiple of 32, checked against
  the product of all block dimensions).
- **D4** — `block_reduce_min` as an additional mode on `lower_block_reduce`.
- **D5** — GPU + trybuild tests (warp_reduce bit-exact, 64-thread two-warp
  independence, block_reduce_min across block sizes, compile-fail fixtures
  for the guard at 16 / (4,4) / (8,2) / 48 / wrong-type).
- **D6** — Rustdoc, CHANGELOG `[Unreleased]`, sprint log results section,
  master plan row → complete.

## D1 — IR audit result

**Confirmed, no changes needed in `kaio-core`.** The three existing `ShflSync`
variants (Down, Up, Bfly) cover everything this sprint needs:
- `ShflSyncDown` stays load-bearing for the existing block_reduce path
  (reduce-to-lane-0 pattern; subsequent phases handle cross-warp + broadcast).
- `ShflSyncBfly` is the right fit for standalone `warp_reduce_*` — butterfly
  with halving lane_mask converges every lane to the full warp reduction in
  5 rounds, so all lanes get the result and no broadcast phase is needed.
- `ShflSyncUp` unused in this sprint, remains available for future patterns.

No `shfl.sync.idx` variant is needed for reductions; any future broadcast /
arbitrary-permutation use case can add it as a separate IR deliverable.

## Results

### Correctness — all green

- **Snapshot canary** (new): `kaio-macros/tests/snapshots/block_reduce_sum_f32.tokens.txt`
  captured from main-tip state pre-D2; D2's helper extraction produced a
  byte-identical TokenStream (AD2/AD3 hygiene held).
- **Existing `_kaio_reduce_smem` symbol canary:** unchanged, still green.
- **`kaio/tests/reduce_macro.rs`** — 18 GPU tests pass bit-exact on
  sm_89. Coverage: 6 existing `block_reduce_sum/max` tests (regression
  protection), 5 new `block_reduce_min` tests at block sizes 32 / 64 /
  128 / 256 / 512, 5 `warp_reduce_*` tests covering sum / max / min
  across all-ones / ascending / single-hot / alternating-sign / lane-
  index patterns, 1 two-warp independence canary (block_size = 64
  with warp 0 fed `lane as f32` and warp 1 fed `lane * 2.0`; asserts
  warp 0 sees 496.0 and warp 1 sees 3040.0 independently), and 1 2D
  `block_size = (8, 4)` warp_reduce test proving product-based linear
  tid works.
- **`kaio/tests/compile_fail/`** — 4 new trybuild fixtures all fail
  with the expected errors:
  - `cf_warp_reduce_small_block.rs` (`block_size = 16`): whole-warp
    guard fires.
  - `cf_warp_reduce_2d_small_product.rs` (`(4, 4)`): 2D product 16
    fails the guard.
  - `cf_warp_reduce_partial_warp_48.rs` (`(8, 6)` = 48): non-multiple
    of 32 fails the guard even though ≥ 32. This is the round-3
    specific catch.
  - `cf_warp_reduce_wrong_type.rs`: `i32` passed where `f32` required
    — caught by `check_f32` with a precise span pointing at the call.
- **Per-warp semantics and partial-warp guard are exactly as specified
  in the approved plan.** The `.stderr` for `cf04_unknown_call` was
  refreshed to list the 4 new builtin names.

### Scope — exactly as planned

- No new IR variants in `kaio-core`.
- No new showcase example (existing `examples/softmax/` already
  demonstrates block_reduce; warp_reduce coverage via tests only).
- No release this sprint — Phase 7 closes with an aggregate release
  after 7.2 / 7.3 / 7.4.
- f32-only. Integer / half-precision variants deferred.
- `shfl.sync.idx` variant deferred.

### Commits

| # | Commit | Scope |
|---|---|---|
| 1 | `c39ba45` | D1 — sprint stub + IR audit (no-op on kaio-core) |
| 2 | `2f28dfa` | D2.0 — pre-refactor TokenStream snapshot canary |
| 3 | `3edb5c7` | D2 — factor `emit_warp_tree_reduce` helper (byte-identical) |
| 4 | `b850ee7` | D3 + D4 — warp_reduce + block_reduce_min + whole-warp-multiple guard |
| 5 | `c5d2548` | D5 — 12 new GPU tests + 4 trybuild compile-fail fixtures |
| 6 | _this_ | D6 — docs, CHANGELOG `[Unreleased]`, master plan / phases marker |

### Follow-ups noted for future sprints

- Integer-typed reductions (`warp_reduce_sum(i32)` etc.) — follow-up
  when a quant kernel needs them in DSL rather than IR-authored code.
  The helper's type parameterization means this is a small extension,
  not a rewrite.
- Half-precision reductions (f16 / bf16) — follow-up.
- `shfl.sync.idx` IR variant — not needed for reductions, but useful
  for broadcast / general permutation patterns; standalone sprint when
  a kernel requires it.
- Atomics — standalone sprint when histogram / scatter-add / gradient
  accumulation needs it.
- Dynamic shared memory — runtime-sized tiles for auto-tuning.
- Per-warp softmax / layer-norm showcase example — would demonstrate
  `warp_reduce_*` visually well; defer to a polish pass or fold into
  7.3 (quant + attention).
- Optional: relax the TokenStream snapshot canary into pattern
  assertions if it becomes noisy across future refactors (AD7).
