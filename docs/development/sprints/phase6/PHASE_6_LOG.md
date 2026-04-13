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
| [6.6](sprint_6_6.md) | Fused TC attention + causal variant (internal API) | ✅ Complete | `5c3ae53` | 275 host + 133 GPU (+7 host, +11 GPU) |
| [6.7](sprint_6_7.md) | Multi-warp 64×64 TC matmul + edge tiles + cuBLAS bench (79.9% sync, 85.1% async at 4096²) + matmul_tc/_async promoted to stable pub | ✅ Complete | `a3d5ca3` | 279 host + 148 GPU |
| 6.7-post | Codex post-review: size-heuristic cache-miss default + doc drift | ✅ Complete | `78b886c` | 280 host + 148 GPU (+1 host) |
| [6.8](sprint_6_8.md) | Showcase examples (fused SiLU-gate, GELU comparison, RMSNorm) | ✅ Complete | `6509cae` | 280 host + 148 GPU (examples are standalone binaries, not counted) |
| [6.7b](sprint_6_7b.md) | Bank-conflict padding + D10 hoist (async 92.5% / sync 82.3% cuBLAS sgemm 4096²); LdGlobalB128 IR primitive landed as unused-future-anchor | ✅ Complete | `6c8e177` (Gate A) + `449dbee` (Gate B) + `d8cd0a0` (feat) + `9038182` (closeout) | 286 host + 148 GPU |
| [6.9](sprint_6_9.md) | v0.2.0 publish prep — version bumps, example path+version deps, CHANGELOG 0.2.0 section, rustdoc polish on promoted TC matmul APIs | In progress | — | 286 host + 148 GPU |
| [6.9](sprint_6_9.md) | Polish + v0.2.0 publish | Pending | — | — |

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
                                                  6.6 (TC attention — internal)
                                                       |
                                                  6.7 (multi-warp + edge tiles + bench + promotion ✅)
                                                       |
                                                  6.8 (showcase examples ✅)
                                                       |
                                                  6.7b (padding + D10 hoist — async 92.5% ✅)
                                                       |
                                                  6.9 (publish v0.2.0 — in progress)
```

All sprints are sequential. Sprint 6.6 (TC attention) was originally
marked optional for v0.2.0 but shipped — matmul and attention
together prove tensor-core integration end-to-end. Sprint 6.8
(showcase examples) was reordered ahead of 6.7b (vectorized loads)
on 2026-04-12: at 79.9/85.1% cuBLAS sgemm the 6.7 perf story is
already strong enough to launch, and examples are what makes a
v0.2.0 release discoverable. 6.7b becomes a v0.2.x perf bump.
