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
| 9.1.2 | bf16 auto-tuner cache (`matmul_auto_tc_bf16` + `tune_matmul_tc_bf16`) — 2-way dispatch between bf16 sync and async | ✅ Complete (2026-05-16) | Cache coexistence invariant locked: f16-TC and bf16-TC entries share the same JSON file disambiguated by `kernel` field; 6 GPU dispatch / fallback / correctness tests + 2 host unit tests green. Latent `CacheEnvGuard` parallel race fixed in both tuner test files. [sprint_9_1_2.md](sprint_9_1_2.md) |
| 9.1.3 | kaio-candle bf16 forward bindings (`matmul_tc_bf16` + `matmul_tc_bf16_async` via `CustomOp2`) | ✅ Complete (2026-05-18) | Bf16 forwards bridged into candle mirroring the f16 binding shape; 12 GPU roundtrip tests green (6 bit-exact + 4 rejection + 2 SC-2 negative-backward). Forward-only — backward in 9.1.4. `.backward()` on a graph containing either op returns an explicit `Err` naming 9.1.4 + workaround paths, not the generic `BackwardNotSupported` default. Bridge primitives are dtype-generic so no `bridge.rs` changes were needed. [sprint_9_1_3.md](sprint_9_1_3.md) |
| 9.1.4 | kaio-candle bf16 backward bindings (forward-reuse) | ✅ Complete (2026-05-18) | Bf16 backward shipped via the same forward-reuse pattern used by f16 (mirror of Sprint 7.4d): `dA = grad @ B^T`, `dB = A^T @ grad`, no new PTX. 8 new gradient-correctness GPU tests green (4 per binding, mirroring the f16 coverage). Dual-tolerance `rel < 1e-2 \|\| abs < 1e-3` (identical to f16) held without empirical fallback. 2 negative-backward tests from 9.1.3 deleted in same commit as the bwd impl that lifted their contracts. Closes 9.1.x as 4 sub-sprints (9.1.5 consolidated into 9.1.3 + 9.1.4 mid-phase; cross-product is complete). [sprint_9_1_4.md](sprint_9_1_4.md) |
| 9.2 | FlashAttention backward (`attention_flash_bwd` + causal, candle bridge integration) | ✅ Complete (2026-06-12) | The v0.5.0 hard gate. `_with_stats` forwards (save row logsumexp `L = m + log(l)`; out zero-diff vs shipped fwd) + three new backward kernels (D-preprocess, dK/dV, dQ — block-per-row, loop-nest swap, no atomics, 18–22 regs/thread on sm_89) + first kaio-candle flash binding (fwd + bwd, f32 end-to-end, `CustomOp3`). CPU f64 analytical oracle with FD self-check as primary gate; seq=1 closed-form gate + full `{32,64,128}²` matrix + wide-range-dO cancellation cases green. candle bwd recomputes L (no saved-intermediate channel in CustomOp3; forward determinism verified bit-exact). bwd/fwd 2.5× at seq 128 → ~10.6× at 2048 (block-per-row property; tiled rework is the named perf follow-up). [sprint_9_2.md](sprint_9_2.md) |
| 9.3 | `ldmatrix.sync.aligned` IR primitive + `matmul_tc` fragment-A loader rewire | ✅ Complete (2026-06-12) | `TensorCoreOp::LdMatrix` (m8n8 b16, x2/x4, ±trans; min_sm 75 — the first sub-Ampere tensor-core op, ISA-audited + ptxas-probed) with register-type validation, emission/module-validate/ptxas-verify coverage at sm_75+sm_80. New fragment-A loader (4 ALU + 1 `ldmatrix.x4` per stripe vs ~9 ALU + 4 `ld.shared`) proven bit-identical to the shipped loader by a GPU contract gate. Loader rewire **measured at the ±3% noise floor** (4096³ interleaved medians 102.75/104.49/102.82% over 3 release runs; bank-conflict pattern unchanged at 32-B stride) → default stays `ld.shared`, ldmatrix ships built-and-parked behind `FragALoaderKind` + hidden `matmul_tc_ldmatrix`, guarded by a permanent A/B bench (bit-exact pre-gate + one-sided non-regression gates, all green). XOR-swizzle is the named flip-on lever. [sprint_9_3.md](sprint_9_3.md) |
| v0.5.0 | Phase 9 aggregate release | ✅ Complete (2026-06-12) | Workspace crates → 0.5.0, kaio-candle → 0.2.0 (kaio-py unchanged). Aggregates 9.1–9.3 plus the previously-unreleased Sprint 8.0.5 / 8.1 entries; coverage badge refreshed (94.67%); bf16 parity section added to performance.md; release review completed pre-publish |

## Branch

`phase9` — long-running branch off `main` for all Phase 9 sprints.
No independent crates.io release per sprint; Phase 9 closes with an
aggregate v0.5.0 release after 9.2 lands. 9.1 and 9.3 ship inside the
phase but do not bump versions on their own.

## Key References

- **Master plan:** [phase9_master_plan.md](phase9_master_plan.md)
- **Phases roadmap:** [`phases.md`](../../../phases.md) Phase 9
- **Performance tracking:** [`performance.md`](../../../performance.md)
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
| `matmul_tc_bf16` | sync | 9.1 | `bf16 × bf16 → f32` | SM 8.0+, K%16==0, edge-tile predication on M/N. Median ≈ 91.8% of same-run cuBLAS sgemm at 4096³ on sm_89 in the sprint-gate runs; SC-2 split-bound gate (per-iter bf16/f16 median ±3% + worst ±15%) green. |
| `matmul_tc_bf16_async` | async | 9.1.1 | `bf16 × bf16 → f32` | SM 8.0+, K%16==0, edge-tile predication on M/N. cp.async-pipelined A staging (double-buffered, size=16 issue); cross-product of (f16 async × bf16 sync). SC-2 split-bound gate (per-iter bf16_async/f16_async median +0.72% within ±3%, worst +1.55% within ±15%) green at 4096³ on sm_89. |
| `matmul_auto_tc_bf16` + `tune_matmul_tc_bf16` | auto-tuned | 9.1.2 | `bf16 × bf16 → f32` | SM 8.0+, K%16==0. 2-way dispatch cache between `matmul_tc_bf16` (sync) and `matmul_tc_bf16_async` (async). Shares the f16 auto-tuner's on-disk JSON cache (`~/.cache/kaio/tune_cache.json` or `KAIO_TUNE_CACHE` override); entries disambiguated by `kernel` field. Cache-miss fallback inherits the f16 3072 threshold (separate `ASYNC_FALLBACK_MAX_DIM_THRESHOLD_BF16` symbol so it can drift independently). |
| `attention_flash_with_stats` (+ causal) | FA fwd + saved logsumexp | 9.2 | `f32 → f32` (+ `[seq_len]` f32 stats) | Kernel copy of the shipped flash forward plus one tail store (`L = m + log(l)` per query row). Output zero-diff vs the plain forward; shipped kernels untouched. Deterministic (bit-identical stats across runs, test-locked). |
| `attention_flash_bwd` (+ causal) | FA backward | 9.2 | `f32 → f32` (dQ/dK/dV) | Three kernels (D-preprocess + dK/dV + dQ), block-per-row, loop-nest swap, no atomics, 18–22 regs/thread on sm_89. Primary gate: CPU f64 analytical oracle (FD-self-checked) at `rel < 1e-2 \|\| abs < 1e-3`; seq=1 closed-form gate. `out`/`stats` provenance is a documented caller contract. |
| `TensorCoreOp::LdMatrix` + `load_fragment_a_m16n8k16_ldmatrix` | IR primitive + loader (kaio-core; not a public op) | 9.3 | shared `.b16` tiles → mma fragment registers | `ldmatrix.sync.aligned.m8n8.{x2,x4}{.trans}.shared.b16`; min_sm 75 (first sub-Ampere tensor-core entry). Loader bit-identical to the `ld.shared` path (GPU contract gate). Built-and-parked in `matmul_tc` per measurement (noise-floor uplift at 32-B row stride); default flip awaits the XOR-swizzle follow-up. |

## kaio-candle additions (`kaio-candle` — standalone crate)

*Updated as sprints land.*

| Op | Sprint | Trait | Kernel |
|---|---|---|---|
| `matmul_tc_bf16` | 9.1.3 fwd / 9.1.4 bwd | `CustomOp2` (`MatmulTcBf16Op`) | `kaio_ops::matmul_tc_bf16` (sync). Forward + backward via forward-reuse (`dA = grad @ B^T`, `dB = A^T @ grad`, no new PTX). |
| `matmul_tc_bf16_async` | 9.1.3 fwd / 9.1.4 bwd | `CustomOp2` (`MatmulTcBf16AsyncOp`) | `kaio_ops::matmul_tc_bf16_async` (cp.async). Forward + backward via forward-reuse (same pattern as sync sibling). |
| `attention_flash` | 9.2 fwd + bwd | `CustomOp3` (`AttentionFlashOp`) | `kaio_ops::attention_flash` fwd; backward via the three dedicated bwd kernels (preprocess + dkdv + dq), with L recomputed per bwd call via `attention_flash_with_stats`. f32 end-to-end, single-head self-attention only. |
| `attention_flash_causal` | 9.2 fwd + bwd | `CustomOp3` (`AttentionFlashOp`, `causal: true`) | Same shape as the plain sibling with the causal mask predicate in both fwd and bwd kernels. |
