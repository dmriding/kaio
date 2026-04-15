# Phase 7 — Quantization + Attention: Sprint Index

Quick-reference index for Phase 7 sprints. Each sprint gets a dedicated
doc in this directory with full reasoning traces for every decision.

Master plan: [phase7_master_plan.md](phase7_master_plan.md)

## Sprint Status

| Sprint | Scope | Status | Headline |
|---|---|---|---|
| [7.0](sprint_7_0.md) | Planning + tooling foundations | ✅ Complete | Phase 7 master plan locked |
| [7.0.5](sprint_7_0_5.md) | Follow-up hardening | ✅ Complete | — |
| [7.1](sprint_7_1.md) | `matmul_int8` (W8A8 symmetric) | ✅ Complete | 80–94 TOPS at 4096³ on RTX 4090 |
| [7.1.5](sprint_7_1_5.md) | Warp + block reductions in the DSL | ✅ Complete | — |
| [7.2](sprint_7_2.md) | `matmul_int4` (W4A16 GPTQ-style) | ✅ Complete | 49–58 TOPS at 4096³; 95–116% of cuBLAS sgemm |
| [7.3](sprint_7_3.md) | Fused tri-output QKV projection (`qkv_project_int8` + `qkv_project_int4`) | ✅ Complete | 3.0× decode over 3× standalone; prefill ship-narrow |
| 7.3.5 | Design S+½P optimization (2 W slots, barriers 7→4) | 📝 Planned | Target: recover `prefill_m2048` from 0.85× to ≥1.15× |
| 7.4 | `kaio-candle` bridge crate | 📝 Planned | Candle CustomOp bindings for quant ops |
| v0.4.0 | Phase 7 aggregate release | 📝 Planned | After 7.4 ships |

## Branch

`phase7-rest` — long-running branch off `main` for all Phase 7 sprints
after 7.1's release. No independent crates.io release per sprint; Phase 7
closes with an aggregate v0.4.0 release after 7.4.

## Key References

- **Master plan:** [phase7_master_plan.md](phase7_master_plan.md)
- **Phases roadmap:** [../../phases.md](../../phases.md) Phase 7

## Phase 7 ops shipped so far

| Op | Variant | Sprint | Input / output types | Notes |
|---|---|---|---|---|
| `matmul_int8` | W8A8 | 7.1 | `i8 × i8 → f32` | Single scalar scale. `m16n8k32.s8.s8.s32` direct. |
| `matmul_int4` | W4A16 | 7.2 | packed `u32 × f16 → f32` | GPTQ-style, group_size=128, `m16n8k16.f16.f16.f32` DEQUANT-F16. |
| `qkv_project_int8` | W8A16 | 7.3 | `i8 × f16 → f16` | Fused tri-output, scalar per-projection scales. |
| `qkv_project_int4` | W4A16 | 7.3 | packed `u32 × f16 → f16` | Fused tri-output, f16 group scales. |
