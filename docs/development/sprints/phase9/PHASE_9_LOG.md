# Phase 9 — Attention Backward & Kernel Deepening: Sprint Index

Quick-reference index for Phase 9 sprints. Each sprint gets a dedicated
doc in this directory with the post-delivery outline (context, what
shipped, tests, what didn't change, follow-ups).

Master plan: [phase9_master_plan.md](phase9_master_plan.md)

## Sprint Status

| Sprint | Scope | Status | Headline |
|---|---|---|---|
| 9.1 | bf16 TC matmul family (`matmul_tc_bf16` + optional async / auto-tuner / candle bindings) | ✅ Complete (2026-05-14) | bf16 sync ≈ 55–60 median TF at 4096³; SC-2 split-bound gate green (per-iter bf16/f16 median ≈ +0.9% within ±3%, worst ≈ +2–9% within ±15%); 25/25 D5 correctness tests green. [sprint_9_1.md](sprint_9_1.md) |
| 9.1.1 | bf16 async TC matmul (`matmul_tc_bf16_async`) — cp.async-pipelined sibling | ✅ Complete (2026-05-15) | bf16_async at 4096³ on RTX 4090 sm_89: SC-2 perf-parity gate green vs f16_async (median +0.72% within ±3%, worst +1.55% within ±15%); 25/25 D5 correctness tests green; D6 cvt-free hot-path gate green. [sprint_9_1_1.md](sprint_9_1_1.md) |
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
| `matmul_tc_bf16` | sync | 9.1 | `bf16 × bf16 → f32` | SM 8.0+, K%16==0, edge-tile predication on M/N. ≈ 91.8% of cuBLAS sgemm at 4096³ on sm_89; SC-2 split-bound gate (per-iter bf16/f16 median ±3% + worst ±15%) green. |
| `matmul_tc_bf16_async` | async | 9.1.1 | `bf16 × bf16 → f32` | SM 8.0+, K%16==0, edge-tile predication on M/N. cp.async-pipelined A staging (double-buffered, size=16 issue); cross-product of (f16 async × bf16 sync). SC-2 split-bound gate (per-iter bf16_async/f16_async median +0.72% within ±3%, worst +1.55% within ±15%) green at 4096³ on sm_89. |

## kaio-candle additions (`kaio-candle` — standalone crate)

*Updated as sprints land.*

| Op | Sprint | Trait | Kernel |
|---|---|---|---|
