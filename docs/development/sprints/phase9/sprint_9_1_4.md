# Sprint 9.1.4 — kaio-candle bf16 backward bindings (forward-reuse)

**Status:** ✅ Complete (2026-05-18)
**Branch:** `phase9` (PR to `main` pending Phase 9 close)

---

## Context

Sprint 9.1.3 landed the bf16 candle forward bindings
(`matmul_tc_bf16` + `matmul_tc_bf16_async` as `CustomOp2`),
forward-only — the `bwd()` override on both modules returned an
explicit `Err` naming Sprint 9.1.4 + workaround paths.

9.1.4 fulfills that promise. Mirror of the Sprint 7.4d pattern (f16
candle bwd via forward-reuse): both bf16 modules now compute
`dA = grad @ B^T` and `dB = A^T @ grad` using the same forward
kernel for backward — no new PTX. The 2 SC-2 negative-backward tests
from 9.1.3 were deleted in the same commits as the bwd impls that
lifted their contracts (atomic transition: tests that asserted a
temporary contract leave with the contract).

After 9.1.4 the bf16 candle surface has full parity with f16:
forward + backward, sync + async.

**9.1.x scope consolidation (note):** the original master-plan row
for 9.1 listed four extensions ("bf16 async → bf16 auto-tuner →
kaio-candle bf16 fwd → kaio-candle bf16 bwd"). A since-superseded
follow-up line in `sprint_9_1_1.md` mentioned a hypothetical "9.1.5
= async-candle" sub-sprint, but 9.1.3's mid-phase decision to bind
both sync AND async forwards in one sprint made 9.1.5 redundant —
the sync/async × fwd/bwd cross-product is complete after
9.1.3 + 9.1.4. 9.1.x closes here as 4 sub-sprints; no 9.1.5.

## What shipped

### `matmul_tc_bf16` backward + 4 gradient tests (C0)

`kaio-candle/src/matmul_tc_bf16.rs` — replaced the explicit-`Err`
`bwd()` body with the forward-reuse impl. The f32 upstream gradient
is downcast to bf16 (`grad_res.to_dtype(DType::BF16)?`), used as the
left input to two more `matmul_tc_bf16` calls
(`grad_a = grad @ B^T`, `grad_b = A^T @ grad`), and the f32 outputs
are cast back to bf16 to match candle's gradient-accumulator
dtype-matching requirement (per `backprop.rs:672`). Added `DType` to
the imports (small compile trap caught during plan review).

Module-level rustdoc + struct rustdoc + free-function rustdoc all
refreshed to drop the "Forward-only in Sprint 9.1.3" markers
atomically with the bwd impl (no doc-vs-code drift window).

Numerical-approximation note added to the bwd docstring: bf16's
7-bit mantissa is lower precision than f16's 10-bit, so per-element
quantization noise from the round-trip cast is higher in absolute
terms; bf16's 8-bit exponent gives values representable at scales
where f16 would overflow or underflow. The dual-tolerance gradient
check (`rel < 1e-2 || abs < 1e-3`, identical to f16) covers the
shapes tested in `candle_gpu_roundtrip.rs`.

`kaio-candle/tests/candle_gpu_roundtrip.rs` — deleted
`matmul_tc_bf16_backward_errors_explicitly` (same commit as the bwd
impl; contract lifted, test goes with contract). Added new bf16
gradient-check helpers (`gradient_check_matmul_bf16` and
`gradient_check_matmul_bf16_weighted`, mirroring the f16 helpers
with bf16 host data + bf16 candle ops — they are NOT precision-
generic; per the plan, parallel helpers are cleaner than trying to
generalize). Plus 4 sync tests:
`matmul_tc_bf16_backward_{32x32x32, 128x128x128, 64x32x128, weighted_64x32x128}`.

All 4 sync tests passed in 0.17s on RTX 4090 sm_89 — tolerance held
without needing the empirical fallback flagged in D1.

### `matmul_tc_bf16_async` backward + 4 gradient tests (C1)

Same shape as C0, applied to `matmul_tc_bf16_async.rs`. The bwd
impl uses the async forward kernel for both forward and backward
(consistent perf, matching the f16 async sibling's pattern). Module-
level rustdoc + struct rustdoc + free-function rustdoc all refreshed
to drop forward-only markers, with the precision details
cross-referenced to `MatmulTcBf16Op::bwd` (avoiding duplicate prose).

Deleted `matmul_tc_bf16_async_backward_errors_explicitly` in the
same commit. Added 4 async tests
(`matmul_tc_bf16_async_backward_{32x32x32, 128x128x128, 64x32x128, weighted_64x32x128}`)
reusing the bf16 helpers from C0 with `use_async: true`.

All 4 async tests passed in 0.22s on RTX 4090 sm_89.

### Crate-level `Backward support` docs refresh (C2)

`kaio-candle/src/lib.rs`:

- Per-op entries: both bf16 entries bumped from "Forward-only
  (backward in Sprint 9.1.4)" markers to "**Backward supported.**
  _(Sprint 9.1.3 fwd, 9.1.4 bwd)_".
- `## Backward support` section: dropped the temporary scaffolding
  about "bf16 forwards are forward-only in Sprint 9.1.3"; now says
  all four matmul TC ops (matmul_tc, matmul_tc_bf16,
  matmul_tc_async, matmul_tc_bf16_async) implement `CustomOp2::bwd()`
  via the forward-reuse pattern.
- Numerical-approximation section: split into f16-specific and
  bf16-specific paragraphs. The bf16 paragraph documents the precision
  tradeoff (lower mantissa, better exponent range) and the dual-
  tolerance gradient check. Numerical-only framing — the note avoids
  prescriptive "use bf16 when X" training-strategy claims that the
  crate itself cannot substantiate.

### Public docs (C3)

- `CHANGELOG.md` — `[Unreleased] — Phase 9` heading bumped to include
  Sprint 9.1.4; new top-of-Added entry covering both bwd impls + the
  8 new gradient-correctness tests.
- `README.md` — three current-state mentions updated from "10 forward
  ops + 2 backward" to "10 forward ops + 4 backward" (limitations
  paragraph, crate table row, Candle integration section). Two
  historical Phase 7 status entries intentionally left at their
  original counts — Phase 7 shipped 8 forwards + 2 backwards; Phase 9
  added 2 more forwards (9.1.3) + 2 more backwards (9.1.4).
- `kaio-candle/README.md` line 9 — updated enumeration to say all four
  matmul TC variants support backward.
- `kaio-candle/Cargo.toml` description — same update; "All four matmul
  TC variants (f16+bf16, sync+async) support backward (autograd)."
  Surfaces on crates.io at next publish.

### Sprint outline + phase log (C4)

This document. Plus a 9.1.4 row in `PHASE_9_LOG.md`'s sprint status
table, and notes-column updates on the existing 9.1.3 rows in the
kaio-candle additions table to reflect that bwd now ships
(no new rows — the table tracks the bindings, not the per-method
adds).

No version bump (per the master plan, Phase 9 ships as one aggregate
v0.5.0 release after 9.2 lands).

## Tests

### SC-1: gradient correctness on RTX 4090 sm_89

8 new gradient tests passed via
`cd kaio-candle && cargo test --features cuda -- --ignored matmul_tc_bf16_backward matmul_tc_bf16_async_backward`:

- 4 sync (`matmul_tc_bf16_backward_*`) — 0.17s
- 4 async (`matmul_tc_bf16_async_backward_*`) — 0.22s

Dual-tolerance assertion (`rel < 1e-2 || abs < 1e-3`) held without
the empirical fallback flagged in D1. bf16's lower mantissa
precision didn't push the round-trip beyond f16's tolerance at the
tested shapes (32³ / 64×32×128 / 128³ / weighted-loss 64×32×128).

### SC-2: no regression in existing f16 backward tests

The 8 existing f16 gradient tests (`matmul_tc_backward_*`,
`matmul_tc_async_backward_*`, weighted variants) remain green.
Workspace tests + kaio-candle no-CUDA build all green; no
regression.

### SC-3: quality gates

From `kaio-candle/`: `cargo fmt -- --check`,
`cargo clippy --features cuda --all-targets -- -D warnings`,
`cargo build --no-default-features`, `cargo build --features cuda`,
`cargo doc --features cuda --no-deps`, `cargo test --features cuda`
— all green. From workspace root: `cargo fmt --all -- --check`,
`cargo clippy --all-targets -- -D warnings`, `cargo test --workspace`
— all green. No new bench (no perf claim).

## What didn't change

- `kaio-candle/src/bridge.rs`: zero changes. Bridge primitives are
  dtype-generic; bf16 bwd reuses the same `slice_ref_from_storage`,
  `buffer_ref_from_slice_readonly`, sync helpers as forward.
- `kaio-ops` side: zero changes. The bf16 kernels themselves shipped
  in 9.1 and 9.1.1; 9.1.4 only adds the candle-side `bwd()` method
  that orchestrates two existing forward calls.
- f16 candle bindings (`matmul_tc`, `matmul_tc_async`) and their
  bwd impls: zero changes.
- f16 gradient-check helpers in the test file: zero changes. The new
  bf16 helpers are parallel structures, not precision-generic — per
  the plan's D2 (generalizing the existing helpers was tempting but
  would have risked f16-test regression for no real upside; parallel
  helpers are clean and cheap).
- `matmul_auto_tc_bf16` candle binding: still not added. The f16 side
  doesn't bind `matmul_auto_tc` either; symmetric posture preserved.
- `cargo xtask showcase`: no new showcase. bf16 autograd usage is
  exercised through user/candle code, not standalone showcase.
- Version: still 0.4.1 workspace / 0.1.1 kaio-candle. Phase 9 ships
  at the aggregate v0.5.0 after 9.2 lands.

## Follow-ups

- **Test file size proxy fired this sprint.**
  `kaio-candle/tests/candle_gpu_roundtrip.rs` is now ~1741 LOC with
  a ~430 LOC bf16 section. The 9.1.3 plan's D3 split-tripwire (1800
  LOC file OR ~400 LOC bf16 section) fires on the section proxy. Per
  the 9.1.4 plan's R5 decision, defer split with a named trigger:
  "split into `candle_gpu_roundtrip_f16.rs` +
  `candle_gpu_roundtrip_bf16.rs` when either (a) the bf16 section
  exceeds ~600 LOC from genuine density, OR (b) any reviewer flags
  the section as hard to bisect." Underlying signal is bisect-cleanness;
  LOC is a proxy and the section structure is still clean
  (helpers + tests grouped by sync/async/category).
- **f16 async rejection-test gap** (carried over from 9.1.3
  followups). The bf16 path ships symmetric `_rejects_*` coverage
  for both sync and async; the f16 async path is still missing
  `matmul_tc_async_rejects_*`. Small followup (~50 LOC, 2 tests);
  Phase 9 hygiene-cleanup line item, not a sub-sprint.
- **Precision-generic gradient_check helpers.** If a future precision
  variant gets added (e.g., int8 candle bridging with autograd, or
  fp8), it may be worth refactoring `gradient_check_matmul[_bf16]`
  + `_weighted` into a precision-generic shape. Not done in 9.1.4
  per D2's call — premature generalization would have risked f16 test
  regression. Re-evaluate when there's a third consumer.
- **Adversarial post-execution review (9.1.x close).** The per-sprint
  plan-review pass ran on the 9.1.4 plan in-cycle. The separate
  **adversarial** review for the full 9.1.x bf16 chain
  (9.1.2 + 9.1.3 + 9.1.4) is queued at 9.1.x close (i.e., now,
  before 9.2 begins). Findings come back as a small correction
  pass if anything substantive surfaces.

9.1.x is now closed as 4 sub-sprints. Next is 9.2 (FlashAttention
backward — the v0.5.0 hard gate) once the 9.1.x adversarial pass
clears.
