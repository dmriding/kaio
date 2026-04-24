# Sprint 8.0.5 — Bench coverage extension

**Status:** ✅ Complete (2026-04-24)
**Branch:** `8.0.5` (PR to `main` pending)

---

## Context

Sprint 8.0 landed RFC-0001 pointer syntax and v0.4.1. At that point `cargo xtask bench` covered the matmul family only: `matmul_tc_bench` (f16 TC sync + async), `matmul_int8_bench`, `matmul_int4_bench`. The remaining public kernel families — fused QKV projections, attention (standard + FlashAttention), and the norm / activation showcase kernels — were correctness-tested under the GPU `--ignored` suite but not performance-gated.

Sprint 8.0.5 closes that gap without changing any kernel, macro, runtime, or public API — pure measurement coverage + docs.

## What shipped

### xtask wiring + new bench harnesses

- `qkv_project_bench` (pre-existing file at `kaio-ops/tests/qkv_project_bench.rs`) wired into the xtask `BENCHES` array. INT4 fused-vs-3×-standalone ratio + INT8 absolute TOPS.
- `attention_tc_bench` — new. Invokes `attention_tc` + `attention_tc_causal` at shapes `n64 / n128 / n256 / n384`. These are constrained by the kernel's `seq_k ≤ 384` shared-memory scores buffer cap — the TC path is a short-sequence fused kernel by design.
- `attention_flash_bench` — new. Invokes `attention_flash` + `attention_flash_causal` at shapes `n128 / n512 / n1024 / n2048`. Online softmax has no score-matrix cap, so flash serves as the long-sequence complement to the TC path.
- `norm_activation_bench` — new. Unified harness for six showcase kernels split into two regimes:
  - Reductions (`rms_norm`, `layer_norm`, `softmax`): single-block (`block_reduce_*` is block-scope), capped at `n = 256`. Reported as a launch-overhead reference, not throughput.
  - Elementwise (`fused_silu_gate`, `gelu_exact`, `gelu_fast`): multi-block. Swept across `{256K, 1M, 4M}` for a bandwidth curve.

Kernel bodies in `norm_activation_bench` are copied from the corresponding example crates, each prefixed with a `// SYNC: copied from examples/<crate>/src/main.rs — keep in sync` comment. The examples remain the source of truth; the bench copies pick up pointer syntax (`*const [T]` / `*mut [T]`, RFC-0001) from the migrated example sources.

### Docs

- `docs/performance.md`: three new sections — §Fused QKV Projection Performance, §Attention Performance, §Norm + Activation Kernel Performance. Each table uses the same worst-of-10 protocol applied across all earlier published numbers.
- `docs/performance.md` §Bench coverage today + roadmap: updated from "Sprint 8.0.5 will extend coverage" to the landed seven-bench roster.
- `docs/benchmarks.md` §Bench coverage: refreshed to list all seven harnesses. Added a methodology sentence clarifying that published numbers represent the sprint in which they first landed — re-runs on the same hardware in later sprints have been within run-to-run variance.

### Pointer-syntax proof-of-life

Sprint 8.0 shipped a correctness smoke test at `kaio/tests/macro_pointer_smoke.rs` exercising mixed `&[T]` / `*const [T]` / `&mut [T]` / `*mut [T]` kernel parameters. Sprint 8.0.5 adds the performance-under-measurement complement: `norm_activation_bench.rs` contains six kernels in pointer syntax, producing the headline numbers in `performance.md` §Norm + Activation.

## Tests

- 3 new `#[test] #[ignore]` bench tests added (attention_tc, attention_flash, norm_activation); the pre-existing `qkv_project_bench` had 2 tests.
- All 7 bench harnesses smoke-run green via `cargo xtask bench <name>`.
- 10 consecutive `cargo xtask bench` runs on RTX 4090 sm_89 captured to `target/bench/run_{1..10}.log` and aggregated for the worst/median/best tables in `performance.md`.
- Existing test counts unchanged (no new non-bench tests; no correctness regressions).

## What didn't change

- Zero changes to public API, runtime, macro codegen, PTX IR, or any existing kernel source.
- No version bump — bench coverage is additive and doesn't merit a 0.4.2 release.
- Existing matmul performance.md tables (Sprint 8.0's 10-run numbers for TC sync/async and INT8/INT4) preserved as the published floor. Sprint 8.0.5 re-runs on the same hardware were within run-to-run variance (see the methodology note in `benchmarks.md`).

## Execution-time API findings worth noting

Three bench-scope adjustments surfaced from reading the kernel sources during execution:

1. **KAIO's public attention surface is single-head self-attention only.** `attention_tc`, `attention_flash`, and their causal variants take either `(seq_q, seq_k, d_k, d_v)` with `seq_q` tile constraints or a single `seq_len`. There is no cross-attention / decode (`seq_q = 1, seq_k = N`) path as a public op. The bench uses symmetric self-attention shapes.
2. **`attention_tc` caps `seq_k ≤ 384`** (shared-memory scores buffer). The two attention benches split by kernel design intent: TC at short sequences, flash at long sequences.
3. **Reduction showcase kernels are single-block, `n ≤ 256`.** Multi-block versions would require the `#[gpu_kernel]` DSL to gain cross-block reduction primitives. Out of scope for this sprint; called out as a future Ops Track item in `performance.md`.

## Follow-ups

- **Multi-block reduction kernels** (`rms_norm`, `layer_norm`, `softmax` at hidden-dim ≥ 1024) — Ops Track when cross-block reduction primitives ship.
- **cuDNN MHA reference for attention benches** — tracked tech debt; currently `cudarc` 0.19 does not expose `cudnnMultiHeadAttnForward` cleanly.
- **Phase 8.1 — PyO3 scaffold** per `docs/phases.md`.
- **Phase 9 — FlashAttention backward + kernel deepening** (bf16 TC, `ldmatrix.sync`).
