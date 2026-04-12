# Phase 6 — Tensor Cores + Async Copies: Sprint Index

Quick-reference index for Phase 6 sprints. Each sprint gets a dedicated
doc in this directory with full reasoning traces for every decision.

Master plan: [phase6_master_plan.md](phase6_master_plan.md)

## Sprint Status

| Sprint | Scope | Status | Commit | Tests |
|---|---|---|---|---|
| [6.1](sprint_6_1.md) | fp16/bf16 types + conversions | ✅ Complete | `a1dd450` | 219 host + 102 GPU (+11 host, +2 GPU) |
| [6.2](sprint_6_2.md) | mma.sync + cp.async in kaio-core, typed fragments, gate test | ✅ Complete | `71603c2` | 247 host + 104 GPU (+28 host, +2 GPU) |
| [6.3](sprint_6_3.md) | Tensor-core matmul (IR API, m16n8k16 SM 8.0+, internal) | ✅ Complete | `09cc304` | 262 host + 108 GPU (+15 host, +4 GPU) |
| [6.4](sprint_6_4.md) | cp.async double-buffered matmul (SM 8.0+, internal) | ✅ Complete | `be0e708` | 264 host + 112 GPU (+2 host, +4 GPU) |
| [6.5](sprint_6_5.md) | TC auto-tuner (`matmul_auto_tc`) + `load_module` migration (first public API) | ✅ Complete | `076fcfc` | 268 host + 122 GPU (+9 host, +5 GPU) |
| [6.6](sprint_6_6.md) | TC attention (optional) | Pending | — | — |
| [6.7](sprint_6_7.md) | Benchmarks + performance docs | Pending | — | — |
| [6.8](sprint_6_8.md) | Polish + v0.2.0 publish | Pending | — | — |

## Key References

- **Master plan:** [phase6_master_plan.md](phase6_master_plan.md) — architecture
  decisions, sprint details, risks
- **Success criteria:** [../../success-criteria.md](../../success-criteria.md)
- **Phases roadmap:** [../../phases.md](../../phases.md) Phase 6
- **Phase 5 reference:** [../phase5/PHASE_5_LOG.md](../phase5/PHASE_5_LOG.md)
- **Tech debt:** [../../tech_debt.md](../../tech_debt.md)

## Dependency Graph

```
6.1 (fp16/bf16) -> 6.2 (mma.sync + cp.async) -> 6.3 (TC matmul)
                                                       |
                                                  6.4 (double buffer)
                                                       |
                                                  6.5 (auto-tuner)
                                                       |
                                                  6.6 (TC attention — optional)
                                                       |
                                                  6.7 (benchmarks)
                                                       |
                                                  6.8 (publish)
```

All sprints are sequential. Sprint 6.6 (TC attention) is optional
for v0.2.0 — matmul alone proves tensor-core integration.
