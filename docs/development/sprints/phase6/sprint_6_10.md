# Sprint 6.10 — Close Open Threads

**Status:** Executing (2026-04-13)
**Branch:** `cleanup/open-threads`
**Parent:** v0.2.0 release, main branch tip

---

## Goal

Close three open threads carried forward from Phases 4 through 6 before Phase 7 (quantization) opens. Priority: make what we have rock solid. No user-visible features, no perf work, no Phase 7 scope.

**Deliverables (execution order D2 → D1a → D1b → D3):**

- **D2** — Four host-level codegen regression tests (launch block_dim, shared memory lowering, reduction named symbol, SM threading canary)
- **D1a** — Migrate `#[gpu_kernel]` macro from `load_ptx(&str)` to `load_module(&PtxModule)`; migrate 4 non-macro test call sites; rebuild PtxModule per launch (no cache)
- **D1b** — Add `#[deprecated]` to `KaioDevice::load_ptx(&str)` with migration guide
- **D3** — Remove `unsafe { std::env::set_var("KAIO_SM_TARGET", ...) }` from ptxas_verify tests by parameterizing the PTX build helpers

---

## Benchmark checkpoints

Per plan: diagnostic bisection tool. Log median async % vs cuBLAS at 4096² after each commit.

| Checkpoint | Run 1 | Run 2 | Run 3 | Median async | Median sync | Notes |
|------------|-------|-------|-------|--------------|-------------|-------|
| **Pre-sprint baseline** | 112.1% | 109.8% | 107.1% | **109.8%** | **103.1%** | 2026-04-13. cuBLAS running at 47-59 TFLOPS this session (thermal / driver state). Use these numbers as "this machine, today" reference; the published 102.3% v0.2.0 median is the cross-session baseline. |
| **D2 landed** | 100.4% | 143.4% | 113.4% | **113.4%** | **116.9%** | 2026-04-13, commit `b49cbad`. Wide ratio variance driven by cuBLAS wobbling 45-58 TFLOPS — KAIO async itself was 47/65/66 TFLOPS (run 1 looks like thermal outlier). D2 touched only `kaio-macros` host tests — no runtime kernel path change, no mechanism for perf regression. Noise, not signal. |
| **D1a landed** | 100.9% | 103.4% | 113.4% | **103.4%** | **104.5%** | 2026-04-13, commit `7fad5f2`. KAIO async 4096² median: **2.31ms / 59.48 TFLOPS** vs baseline 2.32ms / 59.20 TFLOPS — within noise, no regression. Small-matrix 256² async: 0.38→0.39ms (+0.01ms, expected `OnceLock<String>` cache-removal cost, not a regression). Trust-boundary fix is in without perf cost at the headline size. |
| **D1b landed** | 100.9% | 108.0% | 108.5% | **108.0%** | **94.8%** | 2026-04-13, commit `b789346`. Attribute-only change (no runtime path). KAIO async 4096² median: 2.12ms / 65.05 TFLOPS — within run-to-run noise of D1a (2.31ms). Ratio drift is cuBLAS wobble. |
| **D3 landed** | 105.7% | 109.5% | 123.4% | **109.5%** | **94.8%** | 2026-04-13, commit `bf13542`. Test refactor only (no runtime path). KAIO async 4096² median: 2.15ms / 64.04 TFLOPS. Within noise of D1b. Run 3 ratio spike (123.4%) driven by cuBLAS having a bad run (51.90 TFLOPS vs session median 60+). |

**Action thresholds:**
- Single run drops >5pp from prior checkpoint median → rerun 3×, confirm, stop and diagnose before next D
- Variance widens noticeably → possible non-determinism, worth a look
- Small-matrix drop after D1a → expected, log don't investigate

---

## D2 — Host codegen regression tests

**Status:** ✅ Complete

### Tests added

1. ✅ `launch_wrapper_emits_correct_block_dim_1d` — passing
2. ✅ `launch_wrapper_emits_correct_block_dim_2d` — passing (split from #1 into 1D+2D for focused assertions)
3. ✅ `shared_memory_lowering_emits_shared_addr_pattern` — passing
4. ✅ `reduction_lowering_uses_named_symbol` — passing
5. ⏸️ `launch_wrapper_threads_compute_capability_into_module_build` — `#[ignore]`d pending D1a. Serves as the written-down spec for D1a's acceptance (remove `#[ignore]`, verify passes).

Commits:
- `c18ffd5` — launch_wrapper block_dim canaries + SM-threading stub
- `b49cbad` — shared memory + reduction lowering canaries

### Mutation verification

Tests 1–4 share the same `TokenStream.to_string().contains(pattern)` mechanism. One mutation check was performed on test 3 (shared memory, most complex fixture — parses full kernel body through `parse_body` + `generate_kernel_module`):

- **Mutation:** in `kaio-macros/src/lower/memory.rs:166`, changed `Operand::SharedAddr(...)` to `Operand::MUTATION_TEST(...)` in the emitted token stream.
- **Result:** test failed with the expected diagnostic ("expected Operand::SharedAddr(...) in shared-memory lowering output, but did not find it" — plus the first 800 chars of the output for debugging).
- **Revert + rerun:** all 4 tests green.

Tests 1, 2, 4 use identical assertion mechanics (substring check on TokenStream output). Validated by parity — the same failure mode would fire in all four.

### Gate status

- ✅ All four active tests pass on non-GPU host
- ✅ Each test has a regression canary comment
- ✅ One test verified to fail under regression mutation; others covered by parity
- ✅ Tests assert semantic structure (substring patterns), not cosmetic output
- ✅ No impact on existing tests — workspace-wide `cargo test` clean

---

## D1a — Macro loader migration

**Status:** ✅ Complete

### Target files migrated

- ✅ `kaio-macros/src/codegen/ptx_builder.rs` — `build_ptx() -> String` renamed to `build_module(sm: &str) -> PtxModule`. Inner flow unchanged up to the `PtxWriter.finish()` point; now returns the `PtxModule` directly. `KAIO_SM_TARGET` env var preserved as caller-override escape hatch.
- ✅ `kaio-macros/src/codegen/mod.rs` — `static PTX_CACHE: OnceLock<String>` removed entirely. Macro-generated `launch()` rebuilds `PtxModule` per call (microseconds of host overhead, invisible at 4096² kernel size).
- ✅ `kaio-macros/src/codegen/launch_wrapper.rs` — `launch()` now reads `device.info().compute_capability`, formats as `sm_XX`, calls `build_module(&sm)` and `device.load_module(&ptx_module)?`. Old `PTX_CACHE.get_or_init(build_ptx)` + `device.load_ptx(ptx)?` path gone.

### Non-macro test call sites migrated

- ✅ `kaio-runtime/tests/vector_add_e2e.rs` — `build_vector_add_ptx() -> String` renamed to `build_vector_add_module() -> PtxModule`; both tests (`vector_add_small`, `vector_add_large`) use `device.load_module(&ptx_module)`. `emit_ptx_debug()` helper added for failure-path PTX dump.
- ✅ `kaio/tests/cp_async_roundtrip.rs` — same pattern. `build_cp_async_roundtrip_module() -> PtxModule`.
- ✅ `kaio/tests/mma_sync_fragment.rs` — same pattern. `build_mma_gate_module() -> PtxModule`.

`kaio-runtime/src/device.rs:116` intentionally untouched — that's `load_module()` calling `load_ptx()` as a private implementation detail after `validate()`.

### SM-threading canary activated

Test `launch_wrapper_threads_compute_capability_into_module_build` (D2 Test 5) was `#[ignore]`'d during D2 as a spec-stub. D1a removes the `#[ignore]` and the test passes — structural verification that the macro:
- reads `device.info().compute_capability`
- calls `load_module`
- does NOT emit `load_ptx`

### Gate status

- ✅ Full workspace `cargo test` clean (host) — 140 + 122 + 24 + all other crate tests pass
- ✅ Full workspace `cargo test -- --ignored` clean on the reference 4090 dev box — every GPU-gated test including the two long-running Gate C pathological-shape tests (74s and 69s) passes
- ✅ `cargo check --workspace --tests` clean — no compile errors
- ✅ All three example kernels (`fused_silu_gate`, `gelu_comparison`, `rms_norm`) — `cargo run --release` PASS correctness:
  - `fused_silu_gate`: max_abs_err 1.49e-8, 182.2 μs median
  - `gelu_comparison`: exact PASS (2.38e-7) 184.0 μs, fast PASS (2.38e-7) 177.8 μs
  - `rms_norm`: max_abs_err 2.38e-7, 214.2 μs median
- ✅ No user-visible API change — `#[gpu_kernel]` attribute usage unchanged, no new imports required at the user's call site

Commit: `7fad5f2` — migrate macro + test call sites to load_module(&PtxModule)

---

## D1b — Deprecation

**Status:** ✅ Complete

### Work

- ✅ Added `#[deprecated(since = "0.2.1", note = "use load_module(&PtxModule) — runs PtxModule::validate() for readable SM-mismatch errors")]` on `KaioDevice::load_ptx(&str)`
- ✅ Added migration-guide rustdoc with before/after example pointing at `load_module`
- ✅ Added `#[allow(deprecated)]` on the internal `self.load_ptx(&ptx_text)` call at [device.rs:117](../../../kaio-runtime/src/device.rs#L117) (inside `load_module()` — private implementation detail that can't be migrated without circular dependency)

### Gate status

- ✅ `cargo build --workspace` clean — no unexpected deprecation warnings in our own code
- ✅ `cargo build --workspace --tests --examples` clean — neither tests nor user-facing examples surface the warning
- ✅ `cargo doc --workspace --no-deps` clean — migration-guide rustdoc renders
- ✅ All workspace tests pass (host + GPU)

Commit: `b789346` — deprecate load_ptx(&str) with migration-guide rustdoc

---

## D3 — env-var hygiene

**Status:** ✅ Complete

### Target files migrated

- ✅ `kaio-core/tests/common/mod.rs` — three builders (`build_mma_sync_ptx`, `build_mma_sync_shared_ptx`, `build_cp_async_ptx`) now take `sm: &str` as an explicit argument. Internal `std::env::var("KAIO_SM_TARGET")...` reads removed.
- ✅ `kaio-core/tests/ptxas_verify.rs` — three tests (`ptxas_verify_mma_sync`, `ptxas_verify_mma_sync_shared`, `ptxas_verify_cp_async`) now pass `sm` directly to the builder. All three `unsafe { std::env::set_var(...) }` calls removed.

### Audit note

Other test helpers in `common/mod.rs` also read `KAIO_SM_TARGET` internally:

- `build_vector_add_ptx` (line ~172 pre-sprint)
- `build_shared_mem_ptx` (line ~253 pre-sprint)
- `build_ld_global_b128_ptx` (line ~516 pre-sprint)

Their callers (`ptxas_verify_vector_add`, `ptxas_verify_shared_mem`, `ptxas_verify_ld_global_b128`) do NOT call `set_var` — so no hygiene issue exists today. Left intact to keep the sprint scope focused on the `set_var` callers specifically; parameterizing the other three for consistency is a minor follow-up, not a correctness concern.

### Gate status

- ✅ All 6 `ptxas_verify_*` tests pass (`cargo test -p kaio-core --test ptxas_verify -- --ignored --nocapture`)
- ✅ No `unsafe { set_var(...) }` calls remain in `ptxas_verify.rs`
- ✅ `KAIO_SM_TARGET` user-facing env var still works (read by `sm_target()` helper in test, still read by `build_module` in macro codegen) — unchanged user surface
- ✅ Full workspace `cargo test` clean

Commit: `bf13542` — parameterize ptxas_verify builders, remove set_var hygiene issue

---

## Sprint summary

**Status: all deliverables complete.** D2 → D1a → D1b → D3 executed in planned order, each with independent commits providing clean rollback points. No regressions at the headline 4096² async bench across all four checkpoints (kernel time stable at 2.12–2.32ms median per run, well within run-to-run variance). Small-matrix (256²) timing nudged +0.01ms after D1a as expected from the `OnceLock<String>` cache removal — the documented cost-model change.

**Trust-boundary fix delivered:** `#[gpu_kernel]` macro-authored kernels now flow through `PtxModule::validate()` before ptxas sees the PTX. A user-authored kernel using Ampere-gated features (`mma.sync`, `cp.async`) on a sub-Ampere target now surfaces a structured `KaioError::Validation` instead of a cryptic ptxas error.

**Deprecation in place:** `KaioDevice::load_ptx(&str)` is `#[deprecated(since = "0.2.1")]` with a migration-guide rustdoc. Public API preserved (not removed), so no breaking change.

**CI coverage expanded:** 4 host-side codegen regression tests live in `kaio-macros` (block_dim 1D/2D, shared-memory lowering, reduction named symbol, SM threading canary). One mutation verified. CI can now catch macro codegen regressions without a GPU runner.

**Test hygiene:** no `unsafe { std::env::set_var(...) }` calls remain in `ptxas_verify.rs`. Three `build_*_ptx` helpers now take SM target as an explicit argument.

**Commits on branch (in order):**

1. `18bb94a` update gitignore
2. `c83320d` add Sprint 6.10 log skeleton
3. `c18ffd5` D2 partial: launch_wrapper block_dim canaries + SM-threading stub
4. `b49cbad` D2: shared memory + reduction lowering canaries
5. `c4c7977` log: D2 complete, checkpoint bench captured
6. `7fad5f2` D1a: migrate macro + test call sites to load_module(&PtxModule)
7. `1a797db` log: D1a complete, checkpoint bench captured
8. `b789346` D1b: deprecate load_ptx(&str) with migration-guide rustdoc
9. `bf13542` D3: parameterize ptxas_verify builders, remove set_var hygiene issue

Plus a final CHANGELOG + tech_debt update commit (next step).
