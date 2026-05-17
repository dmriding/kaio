# Sprint 9.1.3 — kaio-candle bf16 forward bindings

**Status:** ✅ Complete (2026-05-18)
**Branch:** `phase9` (PR to `main` pending Phase 9 close)

---

## Context

Sprint 9.1 shipped `kaio_ops::matmul_tc_bf16` (sync). Sprint 9.1.1
shipped `kaio_ops::matmul_tc_bf16_async` (cp.async-pipelined). Sprint
9.1.2 shipped the bf16 auto-tuner. After those three, the `kaio-ops`
side of the bf16 path had full feature parity with f16 on the public
matmul surface — but candle users still had to drop to direct
`kaio-ops` calls and manage `GpuBuffer`s by hand.

9.1.3 mirrors the existing f16 candle binding shape onto the bf16
forwards. After this sprint, candle users get
`kaio_candle::matmul_tc_bf16` and `kaio_candle::matmul_tc_bf16_async`
exactly the way they get `matmul_tc` / `matmul_tc_async` today:
`DType::BF16` in, `DType::F32` out, same rank/contiguity/offset
gates, same `Arc<KaioDevice>` bridge.

The deliverable is **two new candle binding modules + 12 GPU
roundtrip tests + crate-level rustdoc refresh + CHANGELOG / README /
Cargo.toml description updates.** Forward-only — backward arrives in
Sprint 9.1.4 via the same forward-reuse pattern used by
`MatmulTcOp::bwd`.

## What shipped

### `matmul_tc_bf16` candle binding (C0)

New file `kaio-candle/src/matmul_tc_bf16.rs` (~170 LOC) — mirror of
`kaio-candle/src/matmul_tc.rs` with bf16 substitutions:

- `half::f16` → `half::bf16`
- `kaio_ops::matmul_tc` → `kaio_ops::matmul_tc_bf16`
- `MatmulTcOp` → `MatmulTcBf16Op`
- Op name string `"kaio::matmul_tc"` → `"kaio::matmul_tc_bf16"`
- `bwd()` override returns an explicit `Err` naming Sprint 9.1.4:
  `"matmul_tc_bf16 backward is sprint 9.1.4; not yet implemented —
  use kaio_ops::matmul_tc_bf16 directly or downcast to f16"`. Better
  than the default `BackwardNotSupported` because it names the sprint
  that fills the gap + concrete workarounds.

Bridge primitives (`slice_ref_from_storage`,
`buffer_ref_from_slice_readonly`, sync helpers) are dtype-generic
over the input `T`, so no `bridge.rs` changes were needed.
`lib.rs` gained one `#[cfg(feature = "cuda")] mod matmul_tc_bf16;`
+ `pub use` pair.

### `matmul_tc_bf16_async` candle binding (C1)

New file `kaio-candle/src/matmul_tc_bf16_async.rs` (~165 LOC) —
mirror of `matmul_tc_async.rs` with the same bf16 substitutions but
targeting the cp.async kernel. Module-level rustdoc cross-references
`MatmulTcBf16Op` for the bf16 precision contract (avoids duplicate
prose). `lib.rs` gained the second `mod` + `pub use` pair.

### Crate-level rustdoc refresh (C2)

`kaio-candle/src/lib.rs` crate-level docs updated:

- Status header simplified from `## Status — v0.1.0 (Sprint 7.4a–7.4d)`
  to `## Status — v0.1.0 (initial release)`. The dated sprint range
  parses awkwardly when extended with new sub-sprints; per-op sprint
  annotations are clearer.
- `Bridges 8 ops` → `Bridges 10 ops`.
- Two new entries in the CustomOp-based list:
  - `matmul_tc_bf16` — bf16 × bf16 → f32, marked
    "Forward-only (backward in Sprint 9.1.4) _(Sprint 9.1.3)_".
  - `matmul_tc_bf16_async` — same, cp.async variant.
- "Backward support" prose tightened to be explicit about which ops
  have backward (matmul_tc and matmul_tc_async only — the bf16
  forwards do NOT, and calling `.backward()` returns an explicit error
  pointing at 9.1.4).

### bf16 roundtrip test suite (C3)

`kaio-candle/tests/candle_gpu_roundtrip.rs` extended with a new bf16
section (12 tests, all `#[ignore]` for `--ignored` GPU sweep):

| Class | Tests | Asserts |
|---|---|---|
| Bit-exact sync | 3 (64³, 256³, 1024³) | candle-routed output bytes match direct-call output bytes for the same input bytes |
| Bit-exact async | 3 (64³, 256³, 1024³) | Same, against `matmul_tc_bf16_async` |
| Rejection sync | 2 (non-contiguous, non-zero offset) | Bridge surfaces clean rejection messages |
| Rejection async | 2 (same) | Same for the async binding |
| SC-2 negative-bwd | 2 (one per binding) | First `.backward()` returns explicit `Err` with substring anchors `"sprint 9.1.4"` AND `"not yet implemented"` |

The bf16 section is preceded by an explicit cross-precision
isolation contract comment: bf16 tests share `Arc<KaioDevice>` and
Allocator state with the f16 tests above, but no mutable state
beyond device creation. A future contributor reading only the bf16
section doesn't have to grep the f16 section to know the invariant.

The bf16 rejection tests also fill an asymmetry from the f16 path:
the existing `matmul_tc_async_rejects_*` tests were absent. Bf16
ships symmetric coverage for both bindings; bringing f16-async up to
matching coverage is a 9.1.x followup.

### Public docs (C4)

- `CHANGELOG.md` — `[Unreleased] — Phase 9` heading bumped to include
  Sprint 9.1.3; new top-of-Added entry covering both bindings.
- `README.md` — three current-state mentions updated from "8 forward
  ops" to "10 forward ops" (limitations paragraph, crate table row,
  Candle integration section). Two historical Phase 7 status sections
  intentionally left unchanged — Phase 7 shipped 8; Phase 9 added 2.
- `kaio-candle/README.md` — "Ships eight ops" → "Ships ten ops" with
  full enumeration; status header simplified to match lib.rs.
- `kaio-candle/Cargo.toml` — `description` field updated from "8 GPU
  ops" to "10 GPU ops" with the two new bf16 entries enumerated.
  Surfaces on crates.io; accuracy matters at publish time even
  though this sprint doesn't publish.

### Sprint outline + phase log (C5)

This document. Plus a 9.1.3 row in `PHASE_9_LOG.md`'s sprint status
table, and two rows in the previously-empty kaio-candle additions
table (one per binding).

No version bump (per the master plan, Phase 9 ships as one aggregate
v0.5.0 release after 9.2 lands; mid-phase sprints don't bump).

## Tests

### SC-1: bit-exact correctness + rejection paths

10 of the 12 new C3 tests scope to SC-1: 6 bit-exact shape tests
(3 shapes × 2 bindings) + 4 rejection tests (2 per binding).
Bit-exactness is the load-bearing assertion — the bridge introduces
no reinterpretation; same kernel + same bits → same output bits.

Verified locally on RTX 4090 sm_89 via
`cd kaio-candle && cargo test --features cuda -- --ignored matmul_tc_bf16`:
all 12 bf16 tests pass in 0.15s. Existing f16 tests
(`matmul_tc_*_bit_exact_*`, `matmul_tc_rejects_*`) remain green —
no regression.

### SC-2: forward-only is loud at first `.backward()`

The 2 remaining C3 tests scope to SC-2. Each builds a
`Var`-tracked bf16 input, runs forward, calls `.backward()`, and
asserts the returned `candle_core::Error::Msg` substring-matches
both `"sprint 9.1.4"` AND `"not yet implemented"`. The trigger is at
FIRST `.backward()` traversal, not delayed iteration-N — the user
mental model is "if I built the graph, backward should be valid";
failing at iteration 1 beats failing at iteration 1000.

This pattern improves on candle's default `BackwardNotSupported`,
which also errors but generically. The override names the sprint
that fills the gap + the workaround paths.

### SC-3: quality gates

From `kaio-candle/`: `cargo fmt -- --check`,
`cargo clippy --features cuda --all-targets -- -D warnings`,
`cargo build --no-default-features`, `cargo build --features cuda`,
`cargo doc --features cuda --no-deps`, `cargo test --features cuda`
— all green. From workspace root: `cargo fmt --all -- --check`,
`cargo clippy --all-targets -- -D warnings`, `cargo test --workspace`
— all green. No new bench (forward-only, no perf claim).

**kaio-candle is intentionally outside the root workspace** per
`Cargo.toml:12` (cudarc rejects `dynamic-loading` +
`dynamic-linking` as simultaneously active). Build commands target
`kaio-candle/` explicitly via `cd kaio-candle && cargo build ...`;
the `-p kaio-candle` flag from root fails with "cannot specify
features for packages outside of workspace."

## What didn't change

- `kaio-candle/src/bridge.rs`: zero changes. The bridge primitives
  (`slice_ref_from_storage<T: CudaDType>`,
  `buffer_ref_from_slice_readonly<T>`, sync helpers) are
  dtype-generic; bf16 just becomes a new `T`.
- `kaio-ops` side: zero changes. The bf16 entries
  (`matmul_tc_bf16`, `matmul_tc_bf16_async`) already shipped in
  9.1 and 9.1.1.
- f16 candle bindings (`matmul_tc`, `matmul_tc_async`): only
  internal-to-bf16-doc cross-references. No signature changes, no
  behaviour changes, no test surface changes.
- `matmul_auto_tc_bf16` candle binding: deliberately not added. The
  f16 side doesn't bind `matmul_auto_tc` either (candle users on the
  raw entries get explicit shape-vs-variant choice); maintaining
  symmetry over adding unnecessary surface.
- Test infrastructure (`tests/common/mod.rs`, etc.): unchanged. The
  new bf16 section uses inline patterned-data generation matching
  the existing f16 helpers' style.
- `cargo xtask showcase`: no new showcase. Candle binding usage
  belongs in candle-side example code, not in `kaio-ops`'s showcase
  surface.
- Version: still 0.4.1 across the workspace; kaio-candle stays at
  0.1.1. Phase 9 ships at the aggregate.

## Follow-ups

- **9.1.4 — kaio-candle bf16 backward.** Same forward-reuse pattern
  used by `MatmulTcOp::bwd` (no new PTX, just two extra forward
  calls per backward: `grad_a = grad @ b^T`, `grad_b = a^T @ grad`).
  Replaces both bf16 modules' explicit-`Err` `bwd()` impls with
  actual gradient computation. The 2 SC-2 negative-backward tests
  from this sprint convert to positive gradient-correctness tests.
- **9.1.5 — async-candle.** Per the original master-plan chain,
  but the scope is fuzzy: 9.1.3 already shipped both sync AND async
  bf16 forwards, 9.1.4 will ship both sync AND async bf16
  backwards. There may be nothing left for 9.1.5 to ship. Decide at
  9.1.4 close.
- **f16 async rejection-test gap.** This sprint added
  `matmul_tc_bf16_async_rejects_*` tests but did NOT add the
  symmetric `matmul_tc_async_rejects_*` for f16. The f16 async path
  has the same bridge rejection logic and the tests would be a
  one-helper extension. Cheap followup; not 9.1.3 scope.
- **README.md / docs/phases.md Phase 7 status entries.** Two
  historical Phase 7 entries still say "8 forward ops" — accurate at
  Phase 7's close, now describing pre-Phase-9 state. They should
  remain as historical record unless a future phase-roadmap refresh
  rewrites the entire Phase column to current-cumulative state.

None of these block 9.2 (FA backward, v0.5.0 hard gate) or 9.3
(`ldmatrix.sync.aligned`).
