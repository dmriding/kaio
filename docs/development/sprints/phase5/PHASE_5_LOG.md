# Phase 5 — Fused Attention & Community Release: Sprint Index

Quick-reference index for Phase 5 sprints. Each sprint gets a dedicated
doc in this directory with full reasoning traces for every decision.

Master plan: [phase5_master_plan.md](phase5_master_plan.md)

## Sprint Status

| Sprint | Scope | Status | Commit | Tests |
|---|---|---|---|---|
| [5.1](sprint_5_1.md) | 2D reductions + DSL fixes | Done | (pending) | +4 GPU tests |
| [5.2](sprint_5_2.md) | Standard attention — forward pass | Done | (pending) | +7 GPU tests |
| [5.3](sprint_5_3.md) | Masking + validation | Pending | — | — |
| [5.4](sprint_5_4.md) | FlashAttention — online softmax (stretch) | Pending | — | — |
| [5.5](sprint_5_5.md) | Auto-tuner | Pending | — | — |
| [5.6](sprint_5_6.md) | CI/CD + platform | Pending | — | — |
| [5.7](sprint_5_7.md) | v0.1.0 prep | Pending | — | — |
| [5.8](sprint_5_8.md) | Community launch | Pending | — | — |

## Key References

- **Master plan:** [phase5_master_plan.md](phase5_master_plan.md) — architecture
  decisions, sprint details, risks
- **Success criteria:** [../../success-criteria.md](../../success-criteria.md)
  Phase 5 section
- **Phases roadmap:** [../../phases.md](../../phases.md) Phase 5 — deliverables
- **Phase 4 reference:** [../phase4/PHASE_4_LOG.md](../phase4/PHASE_4_LOG.md)
- **Tech debt:** [../../tech_debt.md](../../tech_debt.md)

## Dependency Graph

```
5.1 (2D reductions — HARD GATE) -> 5.2 (standard attention) -> 5.3 (masking + validation)
                                                                      |
                                                                 5.4 (FlashAttention — stretch)
                                                                      |
                                                                 5.5 (auto-tuner)

5.6 (CI/CD) -- start immediately after 5.1
5.7 (v0.1.0) -- depends on 5.1-5.6
5.8 (launch) -- depends on 5.7
```

All sprints 5.1-5.3 are sequential. 5.4 is a stretch goal.
5.6 runs in parallel with 5.2+.
