# Sprint 7.1.5 — Warp + block reductions in the DSL

**Status:** In progress
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

_To be filled in at sprint completion._
