# Sprint 6.10 — Close Open Threads

**Status:** Executing (2026-04-13)
**Branch:** `cleanup/open-threads`
**Parent:** v0.2.0 release, main branch tip
**Plan:** `C:\Users\david\.claude\plans\sprint_6_10_close_open_threads.md`
**Review rounds:** CC draft → Opus 4.6 → Codex 5.4 → Dave self-review (all accepted)

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
| **D1a landed** | 100.9% | 103.4% | 113.4% | **103.4%** | **104.5%** | 2026-04-13, commit `7fad5f2`. KAIO async 4096² median: **2.31ms / 59.48 TFLOPS** vs baseline 2.32ms / 59.20 TFLOPS — within noise, no regression. Small-matrix 256² async: 0.38→0.39ms (+0.01ms, expected `OnceLock<String>` cache-removal cost; per plan NOT a regression). Trust-boundary fix is in without perf cost at the headline size. |
| D1b landed | | | | | | Expected: no delta (attribute only) |
| D3 landed | | | | | | Expected: no delta (test refactor) |

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
- ✅ Full workspace `cargo test -- --ignored` clean (GPU on Dave's 4090) — every GPU-gated test including the two long-running Gate C pathological-shape tests (74s and 69s) passes
- ✅ `cargo check --workspace --tests` clean — no compile errors
- ✅ All three example kernels (`fused_silu_gate`, `gelu_comparison`, `rms_norm`) — `cargo run --release` PASS correctness:
  - `fused_silu_gate`: max_abs_err 1.49e-8, 182.2 μs median
  - `gelu_comparison`: exact PASS (2.38e-7) 184.0 μs, fast PASS (2.38e-7) 177.8 μs
  - `rms_norm`: max_abs_err 2.38e-7, 214.2 μs median
- ✅ No user-visible API change — `#[gpu_kernel]` attribute usage unchanged, no new imports required at the user's call site

Commit: `7fad5f2` — migrate macro + test call sites to load_module(&PtxModule)

---

## D1b — Deprecation

**Status:** Not started

### Work

- `#[deprecated(note = "use load_module(&PtxModule) instead")]` on `KaioDevice::load_ptx(&str)`
- Rustdoc migration guide with before/after example

### Results

_To be filled in_

---

## D3 — env-var hygiene

**Status:** Not started

### Target files

- `kaio-core/tests/common/mod.rs` — parameterize `build_mma_sync_ptx`, `build_cp_async_ptx` to take SM target argument
- `kaio-core/tests/ptxas_verify.rs` — three tests (`ptxas_verify_mma_sync`, `ptxas_verify_mma_sync_shared`, `ptxas_verify_cp_async`) — pass SM target explicitly, remove `set_var` calls

### Results

_To be filled in_

---

## Sprint summary

_To be filled in at merge._
