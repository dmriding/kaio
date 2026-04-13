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
| D1a landed | | | | | | Expected: no delta at 4096² (per-launch PtxModule rebuild cost is microseconds). Small-matrix numbers (256² / 512²) may drop — expected cost-model change, not a regression. |
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

**Status:** Not started

### Target files

- `kaio-macros/src/codegen/ptx_builder.rs` — `build_ptx() -> String` becomes `build_module(sm: &str) -> PtxModule`
- `kaio-macros/src/codegen/mod.rs` — remove `static PTX_CACHE: OnceLock<String>` (rebuild per launch)
- `kaio-macros/src/codegen/launch_wrapper.rs` — read `device.info().compute_capability`, format as `sm_XX`, call `device.load_module(&module)?`

### Non-macro test call sites

- `kaio-runtime/tests/vector_add_e2e.rs:189, 238`
- `kaio/tests/cp_async_roundtrip.rs:205`
- `kaio/tests/mma_sync_fragment.rs:201`

### Results

_To be filled in_

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
