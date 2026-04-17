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
| [7.3.5](sprint_7_3_5.md) | Design S+½P optimization (2 W slots, barriers 7→4) | ✅ Complete | INT8 shipped S+½P (1.15× `prefill_m2048` like-for-like vs Design-S); INT4 measured at 1.05× vs 3× standalone, retained Design S |
| [7.4a](sprint_7_4a.md) | `kaio-candle` bridge crate — forward ops | ✅ Complete | 5 forward `CustomOp` bindings (matmul_tc, matmul_tc_async, matmul_int4, attention_tc + causal); 15 bit-exact GPU tests |
| [7.4b-part1](sprint_7_4b_part1.md) | `kaio-candle::matmul_int8` binding | ✅ Complete | CustomOp2 W8A8 binding with spread-scale bit-exact tests; 20 GPU tests total |
| [7.4b-part2](sprint_7_4b_part2.md) | `kaio-candle` direct-call pattern + fused QKV bindings | ✅ Complete | New direct-call bridge pattern; qkv_project_int{4,8} with 12 new GPU tests; 32 total |
| [7.4c](sprint_7_4c.md) | `kaio-candle` — event-based stream sync | ✅ Complete | Replaced cuCtxSynchronize with join()-based event sync; CUDA Graph capture partially unblocked |
| 7.4d | Backward PTX kernel skeleton plan | 📝 Planned | Scoping sprint for backward kernels across all 8 bridge ops |
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
| `qkv_project_int8` | W8A16 | 7.3 + 7.3.5 | `i8 × f16 → f16` | Fused tri-output, scalar per-projection scales. Design S+½P (7.3.5): 2 W slots ping-pong, barriers 4/K-tile. |
| `qkv_project_int4` | W4A16 | 7.3 | packed `u32 × f16 → f16` | Fused tri-output, f16 group scales. Design S (7.3.5 S+½P port measured at 1.05× `prefill_m2048`, retained Design S). |

## candle bridge (`kaio-candle` — standalone crate, not a workspace member)

| Op | Sprint | Trait | Kernel |
|---|---|---|---|
| `matmul_tc` | 7.4a | `CustomOp2` | `kaio_ops::matmul_tc` |
| `matmul_tc_async` | 7.4a | `CustomOp2` | `kaio_ops::matmul_tc_async` |
| `matmul_int4` | 7.4a | `CustomOp3` | `kaio_ops::matmul_int4` |
| `attention_tc` | 7.4a | `CustomOp3` | `kaio_ops::attention_tc` |
| `attention_tc_causal` | 7.4a | `CustomOp3` (`causal` field) | `kaio_ops::attention_tc_causal` |
| `matmul_int8` | 7.4b-part1 | `CustomOp2` (`scale: f32` field) | `kaio_ops::matmul_int8` |
| `qkv_project_int8` | 7.4b-part2 | Direct-call (4 inputs → 3 f16 outputs) | `kaio_ops::qkv_project_int8` |
| `qkv_project_int4` | 7.4b-part2 | Direct-call (7 inputs → 3 f16 outputs) | `kaio_ops::qkv_project_int4` |
