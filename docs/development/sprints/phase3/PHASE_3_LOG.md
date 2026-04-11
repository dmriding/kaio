# Phase 3 — Loops, Reductions & Softmax: Sprint Index

Quick-reference index for Phase 3 sprints. Each sprint gets a dedicated
doc in this directory with full reasoning traces for every decision.

Master plan: [phase3_master_plan.md](phase3_master_plan.md)

## Sprint Status

| Sprint | Scope | Status | Commit | Tests |
|---|---|---|---|---|
| [3.1](sprint_3_1.md) | Loops (`for`/`while`) + compound assignment (`+=`) | Done | `3c3b291` | +16 host, +3 GPU = 181+8 |
| [3.2](sprint_3_2.md) | Shared memory + barrier + shuffle instructions (kaio-core) | Done | `d82cc1f` | +7 unit, +2 integ = 188 total |
| [3.3](sprint_3_3.md) | Shared memory in macro DSL (`shared_mem![]`) | Done | `218162d` | +7 = 190 total |
| [3.4](sprint_3_4.md) | Barrier + shuffle built-in functions | Done | `6fde908` | +9 = 198+12 GPU |
| [3.5](sprint_3_5.md) | Reduction primitives (`block_reduce_sum/max`) | Done | `24efd87` | +2 GPU = 198+14 GPU |
| [3.6](sprint_3_6.md) | Softmax kernel | Done | `b9c9473` | +5 GPU = 198+19 GPU |
| [3.7](sprint_3_7.md) | PyTorch validation + accuracy suite | Done | `01f0610` | +5 GPU = 198+24 GPU |
| [3.8](sprint_3_8.md) | Polish + coverage + docs | Done | `0691be8` | +2 = 200+24 GPU |

## Key References

- **Master plan:** [phase3_master_plan.md](phase3_master_plan.md) — architecture
  decisions, new instructions, IR variants, sprint details
- **Success criteria:** [../../success-criteria.md](../../success-criteria.md)
  Phase 3 section — functional criteria, numerical accuracy, coverage targets
- **Phases roadmap:** [../../phases.md](../../phases.md) Phase 3 — deliverables,
  sprint breakdown, key risks
- **Phase 2 reference:** [../phase2/PHASE_2_LOG.md](../phase2/PHASE_2_LOG.md) —
  completed proc macro DSL

## Dependency Graph

```
3.1 (loops + +=) ─────────────┐
                               ├──→ 3.3 → 3.4 → 3.5 → 3.6 → 3.7 → 3.8
3.2 (shared mem + shfl core) ─┘
```

Sprints 3.1 and 3.2 can execute in parallel (no shared dependencies).
