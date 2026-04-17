# Sprint 6.5 — Tensor-Core Auto-Tuner + `load_module` Migration

**Status:** Done
**Branch:** phase6
**Parent:** `5d82bb5` (6.4 + review fallout)

## Goal

Deliver the first user-facing Phase 6 surface — a tensor-core auto-
tuner — and, while doing so, centralize kernel loading through
`PtxModule::validate()` so the Phase 6 PTX path stops relying on ad-
hoc `device.info()` checks sprinkled per-kernel.

The two deliverables are intentionally coupled: the auto-tuner
needs SM + divisibility eligibility gating before it can measure
anything, and `load_module` already gives that validation for free.
Migrating kernels to `load_module` while the tuner is new avoids
re-plumbing the same code path twice in a later sprint.

## What shipped

### `kaio-ops::matmul_auto_tc` (new — first Phase 6 public API)

- `matmul_auto_tc(device, a: &GpuBuffer<f16>, b: &GpuBuffer<f16>, c: &mut GpuBuffer<f32>, m, n, k)`
  — dispatches between `matmul_tc` (sync) and `matmul_tc_async`
  (`cp.async` double-buffered) based on cached benchmark data.
  Conservative default on cache miss: `TensorCore` (sync) — matches
  6.4's timing observation that async is ~7% slower at 1 warp/block.
  An inline code comment at the fallback site flags the 6.7 revisit
  trigger (multi-warp will likely invert this default).
- `tune_matmul_tc(device, m, n, k)` — benchmarks both variants,
  caches the faster one to the shared tuner cache file
  (`~/.cache/kaio/tune_cache.json`), returns the winning variant's
  string name.
- **Narrow-contract docs** (per review feedback): rustdoc + README +
  CHANGELOG announce f16-only, SM 8.0+ only, temporary
  M%16=N%8=K%16=0 constraint, and explicit framing that production
  performance lands in Sprint 6.7's multi-warp restructure — not in
  6.5. Avoids the "first tensor-core matmul API" misreading.
- Pre-dispatch eligibility gate in `check_tc_eligibility`: SM 8.0+
  AND divisibility. Rejected-early errors name **both** real
  fallback options (pad/convert inputs, OR switch to f32
  `matmul_auto`) — not just "use matmul_auto," which would only be
  actionable for users willing to change their buffer types.

### `load_module` migration (`matmul_tc` + `matmul_tc_async`)

- `build_matmul_tc_ptx() -> String` → `build_matmul_tc_module(sm: &str) -> PtxModule`.
  Same refactor for the async kernel. `KAIO_SM_TARGET` env-var
  reading + sm_80 flooring removed from both TC build functions.
- Host APIs `matmul_tc` and `matmul_tc_async` now derive module
  SM from `device.info().compute_capability` at call time, build
  the `PtxModule`, and call `device.load_module(&module)`.
  `PtxModule::validate()` catches sub-Ampere targets cleanly with
  `ValidationError::SmTooLow`, bubbling as
  `KaioError::Validation(...)` — distinct from `KaioError::PtxLoad`
  (which stays reserved for genuine ptxas / driver failures).
- Ad-hoc `device.info().compute_capability < 8` checks in both host
  APIs are **deleted**. The `KaioError::InvalidConfig` path for
  pre-Ampere is gone from direct callers; the tuner's eligibility
  pre-check keeps the `InvalidConfig` surface for users calling
  through `matmul_auto_tc`, since that's the public API and needs
  the actionable fallback message.

### Visibility

- `tune_matmul_tc` and `matmul_auto_tc` are stable `pub fn` in
  `kaio-ops/src/lib.rs` (no `#[doc(hidden)]`) — the first Phase 6
  user-facing surface.
- `matmul_tc` and `matmul_tc_async` remain `#[doc(hidden)] pub use`
  with their TEMP-comment promotion markers intact (6.7 lifts
  divisibility + promotes them alongside the API announcement).

### Shared test helpers

- **NEW** `kaio-ops/tests/common/mod.rs` — extracts
  `patterned_f16_data`, `cpu_matmul_f16xf16_f32`, and
  `assert_close_with_k_scaled_tol` out of the three TC test files
  that were duplicating them (6.3 api, 6.4 api, and the new 6.5
  tuner test). Three copies was tolerable; four (when
  `tuner_tc_test.rs` was added) would have been the breaking point.

### Tests

- 6 new host unit tests for module/validation behavior:
  `build_matmul_tc_module_produces_valid_structure` (refactored
  from the old `_ptx` version, now instruction-centric per review
  feedback), `build_matmul_tc_module_declares_requested_sm_target`,
  `matmul_tc_module_rejects_sm_70_via_validate`,
  `matmul_tc_module_validates_at_sm_80_and_above`, and the matching
  `_async` pair.
- 3 new host unit tests in the tuner module:
  `matmul_tc_variant_as_str_from_str_roundtrip`,
  `tune_result_json_roundtrip_matmul_tc_variant`, and
  `cache_matmul_and_matmul_tc_entries_coexist` (review feedback — the
  last one locks D6's shared-cache-file claim into a regression
  gate: scalar `matmul` and `matmul_tc` entries for the same
  `(sm_target, dims)` tuple coexist without colliding).
- 5 new GPU integration tests in
  `kaio-ops/tests/tuner_tc_test.rs`: `tune_matmul_tc_returns_valid_variant`,
  `matmul_auto_tc_produces_correct_output`,
  `matmul_auto_tc_falls_back_without_cache`,
  `matmul_auto_tc_rejects_non_divisible_m`,
  `matmul_auto_tc_rejects_zero_dim`. Each uses an isolated
  temp-file tuner cache via a `CacheEnvGuard` RAII helper so tests
  don't leak state into each other.

## Gate results

| Test suite | Result |
|---|---|
| `matmul_tc_api` (6.3 correctness, 4 tests) | ✅ all pass |
| `matmul_tc_async_api` (6.4 correctness, 4 tests) | ✅ all pass |
| `tuner_tc_test` (6.5 gate, 5 tests) | ✅ all pass |
| `tuner_test` (Phase 5 scalar tuner, 6 tests) | ✅ all pass |
| `mma_sync_fragment` (6.2 gate) | ✅ pass |
| `cp_async_roundtrip` (6.2 primitive) | ✅ pass |
| Host unit + structural (268 total) | ✅ all pass |
| `ptxas_verify` (5 tests, sm_89) | ✅ all pass |

Error floor on `matmul_auto_tc`'s correctness test is identical to
6.3/6.4 (~1e-6 against a 1e-2 tolerance on 32×32 × 32×16) — no
regression introduced by the migration.

## Behavioral change: `KAIO_SM_TARGET` no longer affects TC kernels

**Breaking change for direct callers of `matmul_tc` /
`matmul_tc_async`.**

Pre-6.5: the TC kernels floored the emitted module's target SM at
`sm_80` regardless of `KAIO_SM_TARGET`, and an ad-hoc
`device.info().compute_capability` check rejected pre-Ampere
hardware.

Post-6.5: the TC kernels **ignore `KAIO_SM_TARGET` entirely** and
derive the module's target SM from `device.info()` at call time.
`PtxModule::validate()` then rejects pre-Ampere cleanly via
`ValidationError::SmTooLow`, which surfaces as
`KaioError::Validation(...)` rather than
`KaioError::InvalidConfig(...)`.

Both TC kernels are `#[doc(hidden)] pub use`, so documented user
impact is zero, but the changelog flags it as a Breaking Change
for completeness. The scalar `#[gpu_kernel]` proc-macro path
**continues to honor `KAIO_SM_TARGET`** — that asymmetry is
intentional (macro-generated kernels genuinely run on any SM 7.0+
and users on Volta/Turing need the env override) and documented
in README + CHANGELOG.

## Sanity timing observation (env-gated, carry-forward from 6.4)

Ran `KAIO_SPRINT_6_4_TIMING=1` on `tc_async_matmul_medium_64_64_64`
with the release build after the migration to confirm the timing
datapoint didn't shift:

| Kernel | ms/iter | ratio |
|---|---|---|
| `matmul_tc` (sync) | 0.250 | 1.00× |
| `matmul_tc_async` | 0.269 | 1.07× |

Identical to pre-migration numbers. `matmul_auto_tc`'s fallback
default (`TensorCore` sync) is correctly aligned to this datapoint
for 1-warp workloads. Sprint 6.7's multi-warp restructure is where
this expected to invert.

## Architectural decisions made

From the plan (all folded in as-shipped):

- **D1 — 2-way TC tuner, not "3-way unified".** `matmul_auto_tc`
  sits as a sibling to `matmul_auto`, not a replacement. Two
  type-distinct auto-dispatch entry points (f32 and f16) — three
  matmul variants exposed across two APIs.
- **D2 — Module target SM from `device.info()`.** The change that
  makes `load_module` validation meaningful. `KAIO_SM_TARGET`
  removed from TC build functions.
- **D3 — `MatmulTcVariant` mirrors `MatmulVariant`.** Same
  `TuneResult` struct, same cache file, `kernel="matmul_tc"` field
  disambiguates from scalar entries.
- **D4 — Pre-dispatch eligibility gate** (SM 8.0+ + divisibility),
  with an error message naming both real fallback options per
  review feedback. Conservative default comment at the fallback site
  per review feedback.
- **D5 — Phase 5 benchmark shape (3 warm-up + 10 timed iters,
  median)** unchanged.
- **D6 — Shared cache file**, `kernel` field disambiguates —
  coexistence locked in by the new host regression test.
- **D7 — `matmul_auto_tc` public, raw kernels hidden** — narrow-
  contract docs requirement folded into rustdoc + README +
  CHANGELOG to address the "first TC API" positioning risk review
  flagged.
- **D8 — Host SM-validation regression tests** for both modules —
  prove *our* modules are shaped correctly for validation to catch
  the right things, not that `validate()` itself works.
- **D9 — `tests/common/mod.rs` extraction** + cache-coexistence
  unit test.
- **D10 — `KaioError::Validation` routing** unchanged; tuner uses
  `InvalidConfig` for pre-dispatch rejection, kernel-level
  validation bubbles `Validation`.
- **D11 — `format!("sm_{major}{minor}")`** for the device →
  module-target translation. `KAIO_SM_TARGET` asymmetry explicitly
  documented.

## Bugs caught during execution

None. The migration was mechanical and the new tuner followed the
Phase 5 pattern closely enough that correctness was never at risk —
the host regression tests caught any behavioral drift before the
GPU tests ran.

## Tech-debt rollup

Updated `load_ptx(&str)` validation-bypass entry in `tech_debt.md`:

- 6.5 migrated the three tuner-dispatched matmul kernels
  (`matmul`, `matmul_tc`, `matmul_tc_async`… correction below).
  Actually: `matmul_tc` and `matmul_tc_async` are migrated; the
  scalar `matmul` still goes through the `#[gpu_kernel]` proc-
  macro codegen path which emits `load_ptx(&str)`.
- Remaining work: migrate `kaio-macros/src/codegen/launch_wrapper.rs`
  to emit `load_module(&PtxModule)` calls. This is the last
  production caller of `load_ptx(&str)`; once it's migrated,
  `load_ptx(&str)` can be `#[deprecated]` with a migration guide
  (which is currently blocked because the macro emits
  user-visible warnings if `load_ptx` is deprecated).
- The scalar `#[gpu_kernel]` path also still respects
  `KAIO_SM_TARGET` with default `sm_70`, which is correct for
  scalar kernels that run on any SM 7.0+.

New tech-debt entry queued by review feedback (6.4 fallout) on
`ptxas_verify` env-var mutation — no status change this sprint.

## Verification

All gates green before commit:

- `cargo fmt --all --check` ✓
- `cargo clippy --workspace --all-targets -- -D warnings` ✓
- `cargo test --workspace` — 268 host tests pass (up from 259)
- `cargo test --workspace -- --ignored` (sm_89) — 122 GPU tests
  pass (up from 117)
- `cargo test -p kaio-core --test ptxas_verify -- --ignored` —
  5/5 pass (unchanged)
- `cargo doc --workspace --no-deps` — clean

## Files

| File | Change |
|------|--------|
| `kaio-ops/src/matmul_tc_kernel.rs` | `build_matmul_tc_ptx` → `build_matmul_tc_module(sm: &str) -> PtxModule`. Host API uses `device.load_module`. Ad-hoc SM check deleted. Structural test refactored to emit manually + stay instruction-centric. 4 new validation regression tests. |
| `kaio-ops/src/matmul_tc_async_kernel.rs` | Same refactor pattern. 4 new validation regression tests. |
| `kaio-ops/src/tuner.rs` | **+** `MatmulTcVariant` enum, `check_tc_eligibility` pre-check, `launch_matmul_tc`, `bench_matmul_tc_variant`, `tune_matmul_tc`, `matmul_auto_tc`, `resolve_matmul_tc_variant`. Conservative-default inline comment at the fallback site (review feedback). 3 new host tests including cache-coexistence (review feedback). |
| `kaio-ops/src/lib.rs` | `pub use tuner::{..., matmul_auto_tc, tune_matmul_tc, ...}`. Module rustdoc lists `matmul_auto_tc` under "Operations" with the narrow-contract framing. |
| `kaio-ops/tests/common/mod.rs` | **NEW** — shared helpers extracted from three duplicate copies. |
| `kaio-ops/tests/matmul_tc_api.rs` | Imports from `common`. |
| `kaio-ops/tests/matmul_tc_async_api.rs` | Imports from `common`. Timing block unchanged. |
| `kaio-ops/tests/tuner_tc_test.rs` | **NEW** — 5 GPU integration tests including cache-isolation via `CacheEnvGuard`. |
| `CHANGELOG.md` | 6.5 bullets + **Breaking Changes** section for `KAIO_SM_TARGET` behavior change on TC kernels. |
| `README.md` | 6.5 ✅, `matmul_auto_tc` row added to Supported Kernel Features (with "TC / fp16 only, temporary divisibility constraint" annotation), env-variables note on the TC vs scalar asymmetry. |
| `docs/development/sprints/phase6/PHASE_6_LOG.md` | 6.5 row complete. |
| `docs/development/sprints/phase6/sprint_6_5.md` | **NEW** — this doc. |
| `docs/development/tech_debt.md` | `load_ptx(&str)` entry updated with 6.5 migration status. |

## Review input

- **Planning review:** green light with two folds — conservative-default
  inline comment at the fallback site (D4) pointing at the 6.4
  timing datapoint and 6.7 revisit trigger; noted the SM-format-
  string edge case as a non-action.
- **Adversarial review:** green light with five folds — softened
  pre-check error message (D4) to name both fallback options;
  narrow-contract docs requirement (D7) to frame `matmul_auto_tc`
  as a Sprint 6.5 preview surface rather than a mature API; added
  host-only cache-coexistence test (D9); explicit Breaking Changes
  bullet for `KAIO_SM_TARGET` asymmetry (D11); structural tests
  kept instruction-centric.

## Carry-forward to 6.6 and 6.7

Sprint 6.6 (TC attention, optional):
- Inherits the `load_module` pattern directly — new kernels would
  be built via `build_*_module(sm)` from day one.

Sprint 6.7 (multi-warp restructure + divisibility relaxation):
- `matmul_tc` and `matmul_tc_async` graduate from `#[doc(hidden)] pub`
  to stable `pub`. TEMP comments in `kaio-ops/src/lib.rs` removed.
- `matmul_auto_tc`'s eligibility check loses the divisibility
  clause (only SM 8.0+ remains). Signature unchanged.
- 6.5's cache entries stay valid — tuner key is
  `(variant, sm_target, dims)`, unchanged by kernel internals.
- The conservative-default fallback (`TensorCore` sync) revisited
  — multi-warp should make async competitive or faster.

Tech debt (unchanged priorities post-6.5):
- `#[gpu_kernel]` macro migration to `load_module`. After this
  lands, `load_ptx(&str)` can be `#[deprecated]`.
- `store_fragment_c` register-stride variant.
- `group_id` / `thread_id_in_group` hoisting in fragment helpers.
- ArithOp bitops (shr/shl/and/or).
- Pure scalar f16 GPU kernel execution test.
- `ptxas_verify` env-var mutation hygiene (new in 6.4 fallout).
