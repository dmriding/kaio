# Phase 9 — Attention Backward & Kernel Deepening: Sprint Index

Quick-reference index for Phase 9 sprints. Each sprint gets a dedicated
doc in this directory with the post-delivery outline (context, what
shipped, tests, what didn't change, follow-ups).

Master plan: [phase9_master_plan.md](phase9_master_plan.md)

## Sprint Status

| Sprint | Scope | Status | Headline |
|---|---|---|---|
| 9.1 | bf16 TC matmul family (`matmul_tc_bf16` + optional async / auto-tuner / candle bindings) | 📝 Planned | — |
| 9.2 | FlashAttention backward (`attention_flash_bwd` + causal, candle bridge integration) | 📝 Planned | — |
| 9.3 | `ldmatrix.sync.aligned` IR primitive + `matmul_tc` fragment-A loader rewire | 📝 Planned | — |
| v0.5.0 | Phase 9 aggregate release | 📝 Planned | After 9.2 ships |

## Branch

`phase9` — long-running branch off `main` for all Phase 9 sprints.
No independent crates.io release per sprint; Phase 9 closes with an
aggregate v0.5.0 release after 9.2 lands. 9.1 and 9.3 ship inside the
phase but do not bump versions on their own.

## Key References

- **Master plan:** [phase9_master_plan.md](phase9_master_plan.md)
- **Phases roadmap:** [`../../phases.md`](../../../phases.md) Phase 9
- **Performance tracking:** [`../../performance.md`](../../../performance.md)
  §"Path to higher throughput" — the sync-vs-async gap that 9.3
  targets.
- **Sprint 7.4d bwd precedent:** [`../phase7/sprint_7_4d.md`](../phase7/sprint_7_4d.md)
  — the matmul_tc forward-reuse backward pattern that 9.1 reuses for
  bf16 candle bindings, and that 9.2 supersedes for attention with new
  PTX kernels.

## Phase 9 ops shipped so far

*Updated as sprints land.*

| Op | Variant | Sprint | Input / output types | Notes |
|---|---|---|---|---|

## kaio-candle additions (`kaio-candle` — standalone crate)

*Updated as sprints land.*

| Op | Sprint | Trait | Kernel |
|---|---|---|---|
