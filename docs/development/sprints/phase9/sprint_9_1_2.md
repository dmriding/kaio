# Sprint 9.1.2 — `matmul_auto_tc_bf16` (2-way bf16 auto-tuner cache)

**Status:** ✅ Complete (2026-05-16)
**Branch:** `phase9` (PR to `main` pending Phase 9 close)

---

## Context

Sprint 9.1 shipped `matmul_tc_bf16` (sync). Sprint 9.1.1 shipped
`matmul_tc_bf16_async` (cp.async-pipelined). 9.1.2 is the third leg
of the bf16 tuner slot in the cross-product: a 2-way auto-tuner cache
that dispatches between the two bf16 candidates per shape, mirroring
the f16 auto-tuner (`matmul_auto_tc` + `tune_matmul_tc`) from Sprint
6.5. After 9.1.2 the bf16 path has full feature parity with f16 on
the public matmul surface — raw sync + raw async + auto-tuned
dispatch.

The deliverable is the **bf16 variant enum + tune entry point +
auto-dispatch entry point + 2 in-module unit tests + 6 GPU integration
tests + parallel-safe `CacheEnvGuard` upgrade (latent test-infra bug
fix) + public docs + sprint outline.** No new IR work, no new bench
file (the two candidates were already benched in 9.1 and 9.1.1).

## What shipped

### Bf16 auto-tuner in `tuner.rs` (C0)

`kaio-ops/src/tuner.rs` extended in-place rather than splitting into
a new module. The cache infrastructure (`TuneCache`, `TuneResult`,
`cache_path`, `load_cache`, `save_cache`, `sm_target`, `WARMUP`,
`ITERS`) was designed multi-precision from Sprint 6.5 (D6 locked in
`kernel`-field disambiguation; existing regression test
`cache_matmul_and_matmul_tc_entries_coexist` proved the model). A
separate module would have forced extracting shared helpers, which
is more churn than the in-place clone saved.

New surface:

- `MatmulTcBf16Variant` enum (TensorCore / TensorCoreAsync) with
  `as_str` / `from_str` / `all` methods. Variant strings are
  precision-tagged (`"tensor_core_bf16"` / `"tensor_core_bf16_async"`)
  so cache JSON files containing both f16 and bf16 entries remain
  immediately legible when inspected by hand. The `kernel` field
  disambiguates the cache key; the self-identifying variant strings
  are an ergonomic choice for log/tool output.
- `check_tc_bf16_eligibility` — pre-dispatch gate cloned from the
  f16 sibling with bf16-named error messages. The SM-too-low path
  suggests "convert inputs to f32 and use `matmul_auto`" rather than
  "use `matmul_auto_tc` (f16 TC) instead", because f16 TC also
  requires Ampere+. A rustdoc maintainer note flags this as
  load-bearing so a future contributor "improving" the message
  doesn't accidentally introduce the misleading f16 TC suggestion.
- `launch_matmul_tc_bf16` — typed dispatch helper over
  `&GpuBuffer<bf16>` / `&GpuBuffer<f32>`.
- `bench_matmul_tc_bf16_variant` — 3-warmup + 10-iter median
  measurement, cloned shape from the f16 bench helper.
- `pub fn tune_matmul_tc_bf16(device, m, n, k) -> Result<String>` —
  benchmark both variants and cache the faster one under
  `(kernel="matmul_tc_bf16", sm_target, [m, n, k])`.
- `pub fn matmul_auto_tc_bf16(device, a, b, c, m, n, k) -> Result<()>`
  — auto-dispatch entry point. Rustdoc documents the per-dispatch
  cache-lookup cost (the resolver calls `load_cache()` on every
  call, not first-call-only) and points small-shape users to the
  raw `matmul_tc_bf16` / `matmul_tc_bf16_async` entries if they
  know their shape ahead of time.
- `ASYNC_FALLBACK_MAX_DIM_THRESHOLD_BF16: u32 = 3072` — separate
  symbol from the f16 constant so the bf16 threshold can drift
  independently. Inherits the f16 value backed by Sprint 6.7's
  measured curve plus Sprint 9.1 (bf16_sync ≈ f16_sync within ±10%
  at every measured size 256³–4096³) and Sprint 9.1.1 (bf16_async
  within +0.72% of f16_async at 4096³). The auto-tuner overrides the
  fallback per-shape on first `tune_matmul_tc_bf16` call.
- `cache_miss_default_bf16` + `resolve_matmul_tc_bf16_variant` —
  same shape as the f16 fall-through; if `sm_target(device)` errors
  the resolver short-circuits to the size heuristic rather than
  propagating (matches the f16 pattern).

Two in-module unit tests added to `tc_tuner_tests`:

- `cache_miss_default_bf16_matches_threshold_3072` — pure host;
  validates `cache_miss_default_bf16(256, 256, 256) → TensorCore`,
  `cache_miss_default_bf16(3072, 256, 256) → TensorCoreAsync`, etc.,
  matching the f16 unit test pattern.
- `cache_matmul_tc_and_matmul_tc_bf16_entries_coexist` — pure host;
  the load-bearing architectural test. Writes one f16 TC entry and
  one bf16 TC entry for the same `(sm_target, dims)` tuple and
  verifies both persist with the correct lookup behaviour. Sibling
  of the Sprint 6.5 `cache_matmul_and_matmul_tc_entries_coexist`
  test that proved scalar-vs-TC coexistence; this one extends to
  f16-TC-vs-bf16-TC.

Plus a one-line fix to the stale `MatmulTcVariant` rustdoc at
`tuner.rs:56-58` which still said `M%16 = N%8 = K%16 = 0` — pre-
Sprint-6.7 constraints. Sprint 6.7 Gate C unconstrained M and N;
only `K % 16 == 0` remains. Same area, free fix.

`kaio-ops/src/lib.rs` updated with two new `pub use` lines and a
new bf16 row in the crate-level docs (the `#![warn(missing_docs)]`
soft gate makes the crate-level overview part of the public
surface).

### Integration test suite + `CacheEnvGuard` latent-bug fix (C1)

New file `kaio-ops/tests/tuner_tc_bf16_test.rs` (~280 LOC) with 6
`#[ignore]`'d GPU tests mirroring the f16 tuner integration suite:

| Test | Purpose |
|------|---------|
| `matmul_auto_tc_bf16_handles_non_divisible_dims` | M=17, N=9, K=16 accepted (edge-tile predication) + correctness vs `cpu_matmul_bf16xbf16_f64` |
| `matmul_auto_tc_bf16_rejects_k_not_multiple_of_16` | K=24 rejected; error message names `matmul_auto_tc_bf16` (the bf16 entry point) |
| `matmul_auto_tc_bf16_rejects_zero_dim` | Boundary: any zero dim rejected with `InvalidConfig` |
| `tune_matmul_tc_bf16_returns_valid_variant` | Tune succeeds at 64³; return string parses back to one of `"tensor_core_bf16"` / `"tensor_core_bf16_async"` |
| `matmul_auto_tc_bf16_produces_correct_output` | Tune first, then dispatch via `matmul_auto_tc_bf16`; correctness vs `cpu_matmul_bf16xbf16_f64` |
| `matmul_auto_tc_bf16_falls_back_without_cache` | Fresh `CacheEnvGuard`, small shape (16×8×16) → fallback picks `TensorCore` (sync), dispatch runs without error |

C1 also fixed a latent test-infrastructure bug in the existing
`tuner_tc_test.rs`'s `CacheEnvGuard`. The original comment claimed
"test binaries run single-threaded by default" — this is incorrect.
Cargo's test harness runs tests within a test binary in parallel by
default. Different test binaries are separate processes (so
cross-binary races don't exist), but within-binary tests share env-
var state and race on `KAIO_TUNE_CACHE` mutations. The existing f16
tests had been passing despite the race because the window between
`set_var` and the kernel's `load_cache()` call was small.

The fix replaces the wishful comment + bare env mutation with a
process-local `std::sync::OnceLock<std::sync::Mutex<()>>` that
serialises env-var mutations across the lifetime of any
`CacheEnvGuard` instance within the same binary. Applied to both
the existing `tuner_tc_test.rs` (latent bug fix) and the new
`tuner_tc_bf16_test.rs`. Poison-tolerant — if a prior test panicked
while holding the lock, subsequent tests still acquire it (test
isolation matters more than poison correctness here).

### Public docs (C2)

- `CHANGELOG.md` — top of Unreleased Added section; heading bumped
  to "Phase 9 (Sprints 9.1, 9.1.1, 9.1.2)".
- `README.md` — bf16 row extended to list `matmul_auto_tc_bf16`
  alongside the raw variants.
- `kaio-ops/README.md` — Auto-tuned variants subsection extended
  with a `matmul_auto_tc_bf16` bullet plus `tune_matmul_tc_bf16` in
  the tuning entry-points list. Notes that f16-TC and bf16-TC cache
  entries share the same JSON file and are disambiguated by the
  `kernel` field.

### Sprint outline + phase log (C3)

This document. Plus a 9.1.2 row in `PHASE_9_LOG.md`'s sprint status
table and an entry in the ops-shipped table for the auto-tuner
pair.

No version bump (per the master plan, Phase 9 ships as one aggregate
v0.5.0 release after 9.2 lands; mid-phase sprints don't bump).

## Tests

### SC-1a: host correctness

Both new in-module unit tests in `tuner.rs`'s `tc_tuner_tests`
green under `cargo test --workspace`:

- `cache_miss_default_bf16_matches_threshold_3072`
- `cache_matmul_tc_and_matmul_tc_bf16_entries_coexist`

The existing `cache_matmul_and_matmul_tc_entries_coexist` (Sprint
6.5) and the other existing host gates remain green — no regression
in the f16 tuner or the cache invariants.

### SC-1b: GPU correctness

6 `#[ignore]` integration tests in `tuner_tc_bf16_test.rs` green
under `cargo test --workspace -- --ignored` on RTX 4090 sm_89. The
existing `tuner_tc_test.rs` and the existing
`matmul_tc_bf16_correctness.rs` / `matmul_tc_bf16_async_correctness.rs`
suites remain green — no regression in the f16 tuner or the raw bf16
kernel paths.

### SC-2: cache coexistence invariant

`cache_matmul_tc_and_matmul_tc_bf16_entries_coexist` (host) is the
load-bearing test. It proves f16 and bf16 entries persist in the
same JSON cache file without collision, complementing the existing
`cache_matmul_and_matmul_tc_entries_coexist` from Sprint 6.5. A
failure would have falsified the entire multi-precision tune-cache
architecture, not just 9.1.2's extension — flagged as such in the
plan-doc rollback section.

### SC-3: hygiene

The standard 10-item pre-push gate is green: fmt, clippy with
`-D warnings`, host tests, GPU ignored sweep, `cargo xtask showcase`,
`cargo doc --no-deps --workspace`, docs ↔ code parity, no internal-
doc refs in the commit series, no AI / reviewer-round / personal-
name leaks in commit messages or public-shipping content, no new
bench (none expected; 9.1.2 is correctness + cache coexistence,
not perf).

## What didn't change

- IR (`kaio-core`): zero changes. The bf16 auto-tuner dispatches
  between two already-shipped raw kernels; no new IR variants, no
  validate-time changes.
- Raw bf16 kernels (`matmul_tc_bf16`, `matmul_tc_bf16_async`):
  zero changes. The auto-tuner is purely a dispatch layer over the
  existing public entries.
- Cache schema: `TuneCache` version stays at 1; bf16 entries are
  parsed by the existing v1 deserializer. The `kernel`-field model
  was the multi-precision seam from Sprint 6.5.
- F16 tuner public surface (`matmul_auto_tc`, `tune_matmul_tc`):
  no changes. The bf16 functions are new entries alongside, not
  modifications of the f16 ones.
- Bench surface: no new `cargo xtask bench` entry. The Sprint 9.1
  and 9.1.1 benches are the performance record for the candidate
  kernels themselves; the auto-tuner inherits that performance.
- `cargo xtask showcase`: no new showcase. The auto-tuner is
  exercised through user code (or candle binding in 9.1.3+);
  standalone showcase doesn't pull its weight as a separate example.
- Version: still 0.4.1 across the workspace. Phase 9 ships at
  v0.5.0 aggregate after 9.2 lands.

## Follow-ups

- **9.1.3 / 9.1.4 / 9.1.5 — kaio-candle bf16 bindings.** Forward,
  backward (via forward-reuse — no saved intermediates needed,
  since matmul bwd recomputes via two extra forward calls
  `grad_a = grad_out @ b^T`, `grad_b = a^T @ grad_out`), and
  async-candle. Follows 9.1.2 per the phase master plan chain. The
  candle bridge's choice between calling `matmul_tc_bf16` /
  `matmul_tc_bf16_async` raw vs `matmul_auto_tc_bf16` is a 9.1.3
  design decision (the current f16 candle bridge in
  `kaio-candle/src/matmul_tc.rs` uses the raw entries, not the auto
  variant).
- **Pre-committed `tuner::common` extraction trigger.** If a future
  sprint adds a third precision-specific tuner section to
  `tuner.rs`, the first commit of that sprint extracts a
  `tuner::common` submodule containing the shared cache
  infrastructure ahead of the new precision-specific code. The
  9.1.2 in-place clone brought `tuner.rs` from 846 → ~1100 lines;
  a third precision section would push it past the 1500-line
  monolith-watch threshold.
- **Bf16 sync-vs-async calibration bench (deferred).** The 3072
  fallback threshold inherits from the f16 measured curve; bf16
  sync-vs-async wasn't directly measured at multiple sizes. The
  structural argument and the existing 9.1 + 9.1.1 data make the
  inherit safe as an initial default, and the auto-tuner overrides
  the fallback per-shape on first call. Revisit only if 9.1.3-5's
  candle bridge exposes a hot path where the un-tuned fallback
  materially affects user experience.

None of these block 9.2 (FA backward, v0.5.0 hard gate) or 9.3
(`ldmatrix.sync.aligned`). The chain continues as scheduling
permits.
