# Phase 2 — Proc Macro DSL: Sprint Index

Quick-reference index for Phase 2 sprints. Each sprint will get a dedicated
doc in [sprints/](sprints/) with full reasoning traces for every decision.

Master plan: [phase2_master_plan.md](phase2_master_plan.md)

## Sprint Status

| Sprint | Scope | Status | Commit | Tests |
|---|---|---|---|---|
| [2.1](sprints/sprint_2_1.md) | Macro skeleton + fn signature parsing | Done | `b3d254f` | +23 = 76 |
| [2.2](sprints/sprint_2_2.md) | Expression lowering: arithmetic | Done | `5468051` | +44 = 120 |
| [2.3](sprints/sprint_2_3.md) | Comparisons + if/else + `@!pred` | Done | `1652d43` | +9 = 128 |
| [2.4](sprints/sprint_2_4.md) | Array indexing + memory access | Done | `374530d` | +10 = 138 |
| [2.5](sprints/sprint_2_5.md) | Built-in functions (thread/block, math) | Done | `89a7a61` | +18 = 157 |
| [2.6](sprints/sprint_2_6.md) | Launch wrapper + full pipeline | Done | `a89eb2b` | +5 = 162 |
| [2.7](sprints/sprint_2_7.md) | Type validation + compile-fail tests | Done | `8dcfa52` | +1 = 163 |
| [2.8](sprints/sprint_2_8.md) | E2E kernel tests | Done | `8012f3a` | +5 GPU = 168+5 |

## Key References

- **Master plan:** [phase2_master_plan.md](phase2_master_plan.md) — architecture,
  design decisions, generated code shape, kernel IR, new instructions
- **Success criteria:** [../success-criteria.md](../success-criteria.md) Phase 2
  section — functional criteria, compile-fail cases, coverage targets,
  numerical accuracy thresholds
- **Implementation spec:** [../implementation.md](../implementation.md) Layer 3 —
  user-facing API design, supported Rust subset, macro expansion strategy
- **Phase 1 reference:** [PHASE_1_LOG.md](PHASE_1_LOG.md) — completed foundation
