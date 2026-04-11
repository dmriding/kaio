# Phase 4 — Tiled MatMul & Block-Level API: Sprint Index

Quick-reference index for Phase 4 sprints. Each sprint gets a dedicated
doc in this directory with full reasoning traces for every decision.

Master plan: [phase4_master_plan.md](phase4_master_plan.md)

## Sprint Status

| Sprint | Scope | Status | Commit | Tests |
|---|---|---|---|---|
| [4.1](sprint_4_1.md) | FMA + 2D indices + 2D launch model | Done | (pending) | +5 host, +3 GPU = 203+27 |
| [4.2](sprint_4_2.md) | Multi-allocation shared memory | Done | (pending) | +3 GPU = 203+30 |
| [4.3](sprint_4_3.md) | Naive tiled matmul kernel | Planned | — | — |
| [4.4](sprint_4_4.md) | kaio-ops crate + host-side API | Planned | — | — |
| [4.5](sprint_4_5.md) | Benchmark harness + cuBLAS baseline | Planned | — | — |
| [4.6](sprint_4_6.md) | Register tiling + optimization | Planned | — | — |
| [4.7](sprint_4_7.md) | Coalescing heuristics + inspection | Planned | — | — |
| [4.8](sprint_4_8.md) | Polish + integration tests + docs | Planned | — | — |

## Key References

- **Master plan:** [phase4_master_plan.md](phase4_master_plan.md) — architecture
  decisions, launch model, matrix semantics, sprint details
- **Success criteria:** [../../success-criteria.md](../../success-criteria.md)
  Phase 4 section — functional criteria, performance targets
- **Phases roadmap:** [../../phases.md](../../phases.md) Phase 4 — deliverables
- **Phase 3 reference:** [../phase3/PHASE_3_LOG.md](../phase3/PHASE_3_LOG.md)
- **Codex readiness review:** [../review1/phase4_readiness_review_2026_04_11.md](../../development/review1/phase4_readiness_review_2026_04_11.md)

## Dependency Graph

```
4.1 (FMA + 2D launch) → 4.2 (multi-shared-mem) → 4.3 (naive matmul)
                                                        ↓
                         4.4 (kaio-ops) → 4.5 (benchmarks) → 4.6 (register tiling)
                                                                    ↓
                                                              4.7 (coalescing)
                                                                    ↓
                                                              4.8 (polish)
```

All sprints are sequential.
