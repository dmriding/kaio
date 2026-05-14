# Sprint 9.1.1 — bf16 async tensor-core matmul (cp.async-pipelined sibling)

**Status:** ✅ Complete (2026-05-15)
**Branch:** `phase9` (PR to `main` pending Phase 9 close)

---

## Context

First of the planned 9.1.x follow-on sub-sprints flagged in Sprint
9.1's "What's next" section. The cross-product is well-defined:

```
              f16              bf16
sync     matmul_tc        matmul_tc_bf16        (Sprint 9.1)
async    matmul_tc_async  matmul_tc_bf16_async  (Sprint 9.1.1, this sprint)
```

bf16 inference workloads at large shapes (LLM decode at seqlen ≥ 4096
with bf16 weights) lose memory/compute overlap on the K-loop without
an async-pipelined bf16 variant. The f16 async-vs-sync margin on
RTX 4090 at 4096³ is in the 4–8% TFLOPS range per the Sprint 6.7c
bench, and bf16 inherits the same pipeline structure so the same
margin is available — gating users who pick bf16 from claiming it
without this sprint.

The deliverable is the **kernel + 25-test correctness suite + bench
+ public API + docs.** No auto-tuner (sub-sprint 9.1.2), no candle
bindings (sub-sprints 9.1.3–9.1.5).

## What shipped

### Infrastructure prep (C0)

Two private helpers in `kaio-ops` promoted to `pub(crate)` so the
new module can call them:

- `emit_warp_quadrant_mma_bf16` in `matmul_tc_bf16_kernel.rs` —
  the per-warp 8-mma loop that uses `TensorCoreOp::MmaSyncBf16` and
  the bf16 fragment loaders. Cross-module within the bf16 path; same
  accumulator layout and fragment-loader hoist as the sync sibling.
- `emit_mw_load_tile_a_64x16_async` in `matmul_tc_async_kernel.rs` —
  the cooperative `cp.async.ca.shared.global` size=16 A-tile loader.
  Precision-agnostic at the byte level (cp.async issues 16-byte
  transfers with no dtype tag), so it serves both f16 async and bf16
  async without modification.

Visibility promotion only — no signature or behaviour changes. Rustdoc
addenda added to both helpers documenting the contract (C0) and the
cross-precision callers (added at C1 once the new module landed, per
the plan's generic-then-extended rustdoc convention).

### `matmul_tc_bf16_async_kernel` module skeleton (C1)

New file `kaio-ops/src/matmul_tc_bf16_async_kernel.rs` (~750 LOC)
containing `build_matmul_tc_bf16_async_module` plus 5 host validation
tests. Module structure is a near-mirror of `matmul_tc_async_kernel`
(the f16 async path) with two substitutions:

- A and B kernel params declared as `PtxType::BF16` instead of `F16`.
- mma op call goes through `emit_warp_quadrant_mma_bf16` (the
  C0-promoted bf16 helper) instead of `emit_warp_quadrant_mma` (the
  f16 sibling).

Everything else clones verbatim — same 64×64 block tile, same 4-warp
32×32 quadrant layout, same Sprint 6.7b padded Tile B col-stride
(36 B), same D10 fragment-loader `(group_id, tig)` hoist, same
double-buffered cp.async pipeline (preamble + in-loop next-iter
issue), same edge-tile predication on M and N, same `emit_warp_quadrant_store`
output. Per the plan's IR-enforcement-boundary lesson from 9.1's
post-delivery cleanup, the bf16 module always uses `MmaSyncBf16` (the
dedicated IR variant), never the generic `MmaSync` with bf16 dtype
tags.

The 5 host validation tests:
- `build_matmul_tc_bf16_async_module_produces_valid_structure` —
  asserts the emitted PTX contains the bf16-tagged mma mnemonic × 8,
  exactly 2 `cp.async.ca.shared.global` issue sites (preamble +
  in-loop, matching the 2 `cp.async.commit_group`s), a `cp.async.wait_group`,
  edge-tile predicate, padded Tile B sizing, 32 predicated
  `st.global.f32` outputs, and NO f16-tagged mma mnemonic.
- `build_matmul_tc_bf16_async_module_declares_requested_sm_target` —
  round-trips `sm_70` and `sm_89` PTX target headers.
- `matmul_tc_bf16_async_module_rejects_sm_70_via_validate` — asserts
  `ValidationError::SmTooLow { required: 80, actual: 70, ... }` with
  the permissive `||` feature-string pattern matching the existing
  `matmul_tc_async` test.
- `matmul_tc_bf16_async_module_validates_at_sm_80_and_above` —
  sm_80 / sm_89 / sm_90 all pass `PtxModule::validate()`.
- `d4_gate_no_cvt_in_bf16_mma_hot_path` — D6 cvt-free gate ported
  from Sprint 9.1: walks the `K_LOOP:` body and fails if any `cvt.*`
  appears between an `ld.shared.b32` fragment load and the next
  `mma.sync.bf16`. Catches accidental f16↔bf16 conversion inserted
  somewhere on the fragment-load → mma hot path. Matters more for
  the async kernel than the sync kernel because the async kernel has
  more in-loop arithmetic (buffer toggle math, has_next predicate,
  next-iter cp.async issue), more PTX surface for a cvt to hide on.

### Public host API + 25-test correctness suite (C2)

Public `kaio_ops::matmul_tc_bf16_async` host fn added to
`matmul_tc_bf16_async_kernel.rs`; `pub use` line added to `lib.rs`.
The host fn signature mirrors `matmul_tc_async` (the f16 async host
fn) with `GpuBuffer<bf16>` for A and B; grid dim spelled via
`n.div_ceil(64)` / `m.div_ceil(64)` to handle non-multiple-of-64
shapes through edge-tile predication.

Full 25-test correctness suite at
`kaio-ops/tests/matmul_tc_bf16_async_correctness.rs` mirroring 9.1's
D5 grid:

| Shape class           | Magnitudes run                                            | Reference        | Count |
|-----------------------|-----------------------------------------------------------|------------------|-------|
| Small (32³, 64³)      | small / medium / large / tiny_product / min_normal        | dense f64        | 10    |
| Medium (256³, 512³)   | small / medium / large / tiny_product / min_normal        | dense f64        | 10    |
| Large 2048³           | small + one large smoke                                   | sampled-cell f64 | 2     |
| Large 4096³           | small (also bench shape)                                  | sampled-cell f64 | 1     |
| Non-square / odd-N    | small only (edge-tile predication)                        | dense f64        | 2     |

Data generators and assertions are imported from the existing
`kaio-ops/tests/common/mod.rs` (`patterned_bf16_*` family,
`assert_bf16_close_d5*` family, `cpu_matmul_bf16xbf16_f64`,
`sample_cells`, `sampled_cell_f64_reference`). The test file owns
its own per-flavor launch + runner wrappers (`launch_bf16_async`,
`require_gpu_ampere`, `run_dense*`, `run_sampled`) — the same
per-flavor pattern the existing `matmul_tc_bf16_correctness.rs`
follows. Same tolerances as 9.1: standard `rel < 1e-2 || abs < 1e-3`
for the small/medium/large magnitudes, relative-only `rel < 1e-1`
plus a nonzero-output assertion for the tiny_product and min_normal
canary classes.

### bf16-async vs f16-async bench + SC-2 split-bound gate (C3)

New `kaio-ops/tests/matmul_tc_bf16_async_bench.rs` with a perf-parity
gate at 4096³, methodology identical to 9.1's SC-2 but with the
reference variable switched: bf16_async vs **f16_async** (not
bf16_sync vs f16_sync as in 9.1). The same precision-isolation logic
applied one slot over in the cross-product — answers "did we pay for
bf16 in the async staging path?"

Methodology: 10 interleaved alternating-order runs of bf16_async and
f16_async at 4096³ (5× f16-async-first, 5× bf16-async-first to cancel
the intra-iter thermal bias). Per-iter `bf16_async_TFLOPS /
f16_async_TFLOPS × 100` ratio is the gated quantity. Two independent
bounds, both must hold:

- **Median ratio bound ±3%** — structural-kernel gate. Tight,
  noise-robust, measures whether the bf16 async kernel has a real
  perf delta from f16 async at the structural level (the kernels are
  byte-identical at the IR level except for the mma operand dtype
  tag, so the bound should be easy to clear).
- **Worst ratio bound ±15%** — catastrophic-tail gate. Generous to
  OS noise; catches genuinely pathological tail behaviour.

Debug-build guard skips the hard assertion in `cfg!(debug_assertions)`.
Canonical reproduction: `cargo xtask bench matmul_tc_bf16_async_bench`.

Same-run interleaving cancels thermal drift and toolchain/driver
variations on its own; the bench prints the same-run f16_async
median TFLOPS at 4096³ for future drift comparison (informational
only, not a gate — see the bench file's docstring for the rationale
behind the no-historical-anchor design).

The new bench is registered in `xtask/src/main.rs` alongside the
existing `matmul_tc_bf16_bench` so `cargo xtask bench` resolves it
without an explicit `cargo test` invocation.

### Public-shipping docs (C4)

CHANGELOG entry under the "Unreleased — Phase 9" section
(consolidating the Sprint 9.1 and 9.1.1 entries since both ship in
the same release cycle). README feature-table row updated to cover
both sync and async bf16 variants together. kaio-ops README
matrix-multiplication table row added for `matmul_tc_bf16_async`.
No version bump (per the master plan, Phase 9 ships as one
aggregate v0.5.0 release; mid-phase sprints don't bump).

### Sprint outline + phase log (C5)

This document. Plus a 9.1.1 row in `PHASE_9_LOG.md`'s sprint status
table and ops-shipped table.

## Tests

### SC-1: correctness

25/25 D5 grid tests passed in 0.37 s on RTX 4090 sm_89
(`cargo test --release -p kaio-ops --test matmul_tc_bf16_async_correctness -- --ignored`).
Shape × magnitude coverage identical to 9.1; no new test data
generators or reference computations introduced (the bf16 math
doesn't change between sync and async, only the kernel does).

### SC-2: perf-parity

`cargo xtask bench matmul_tc_bf16_async_bench` at 4096³ on
RTX 4090 sm_89:

- **Median ratio:** 100.72% (delta +0.72%; bound ±3%) — well inside
  the structural-kernel envelope.
- **Worst ratio:** 101.55% (delta +1.55%; bound ±15%) — comfortably
  inside the catastrophic-tail envelope.
- Same-run f16_async median: 65.04 TFLOPS at 4096³.
- bf16_async median in the per-shape table: 59.65 TFLOPS at 4096³.

Per-iter ratios across the 10 interleaved alternating-order runs:
99.70%, 99.71%, 100.47%, 100.56%, 100.61%, 100.72%, 100.76%, 101.18%,
101.25%, 101.55%. Tight cluster confirming the two kernels are
structurally equivalent with only the mma operand dtype tag differing.

### SC-3: cvt-free hot path

The D6 host-side gate (`d4_gate_no_cvt_in_bf16_mma_hot_path`) parses
the emitted PTX K-loop body and asserts zero `cvt.*` instructions
between any `ld.shared.b32` fragment load and the next `mma.sync.bf16`.
Passed at C1 and unchanged through C5.

### Host-side validation (5 tests)

The 5 module-build tests in `matmul_tc_bf16_async_kernel.rs::tests`
all pass without GPU access — exercised in CI's `cargo test --workspace`
sweep. Covers PTX structural shape, SM target round-trip, validate-
time SM rejection (with permissive `||` feature-string pattern matching
the existing `matmul_tc_async` test convention), and validate
acceptance at sm_80 / sm_89 / sm_90.

## What didn't change

- IR (`kaio-core`): zero changes. `TensorCoreOp::MmaSyncBf16` from
  9.1 covers the new kernel's mma needs; no new IR variants. The
  `PtxModule::validate` enforcement boundary added in 9.1's
  post-delivery cleanup keeps the bf16 mma dtype-tag rejection
  in-force for the new module without any extension.
- bf16 fragment types (`FragmentA_BF16`, `FragmentB_BF16`) — reused
  as-is from 9.1; already `pub` in `kaio-core::fragment`.
- f16 sync / f16 async / bf16 sync kernels — only rustdoc addenda
  on two helpers (`emit_warp_quadrant_mma_bf16` in the bf16 sync
  module, `emit_mw_load_tile_a_64x16_async` in the f16 async module).
  No signature changes, no behaviour changes; their existing test
  surfaces are unchanged.
- Shared correctness infrastructure in `kaio-ops/tests/common/mod.rs`:
  unchanged. The 25-test bf16-async suite imports the same data
  generators and assertions the bf16 sync suite uses.
- `cargo xtask showcase` — no new showcase. The bf16 async path is
  exercised through the auto-tuner in a future sub-sprint; standalone
  showcase doesn't pull its weight as a separate example.
- Version: still 0.4.1 across the workspace. Phase 9 ships at
  v0.5.0 aggregate after 9.2 lands.

## Follow-ups

- **9.1.2 — `matmul_auto_tc_bf16`.** 2-way auto-tuner cache between
  the sync and async bf16 variants. Now has two candidates to
  dispatch between; no longer degenerate. Scheduled next in the
  9.1.x chain.
- **9.1.3 / 9.1.4 / 9.1.5 — candle bf16 bindings.** Forward,
  backward (via forward-reuse — no saved intermediates needed,
  since matmul bwd recomputes via two extra forward calls
  `grad_a = grad_out @ b^T`, `grad_b = a^T @ grad_out`), and
  async-candle. Independent of 9.1.2; can ship in parallel if needed.

None of these block 9.2 (FA backward, v0.5.0 hard gate) or 9.3
(`ldmatrix.sync.aligned`). The chain continues as scheduling permits.

