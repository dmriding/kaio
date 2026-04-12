# Changelog

All notable changes to KAIO will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

Updated at phase completion. Per-sprint detail lives in
[docs/development/sprints/](docs/development/sprints/).

## [Unreleased] — Phase 6: Tensor Cores & Async Copies

Branch: `phase6`. In progress toward v0.2.0.

### Added — Sprint 6.1 (fp16/bf16 types)
- `PtxType::F16` / `PtxType::BF16` with `.f16` / `.bf16` PTX suffixes.
- `RegKind::H` (`%h`) / `RegKind::Hb` (`%hb`) register classes with
  independent counters from `%r`/`%f`.
- `cvt` rounding modes generalized for float-to-float (incl. half)
  conversions — emits `.rn` consistently.
- `half` crate as kaio-core's first dependency; cudarc `f16` feature
  enabled. `GpuBuffer<f16>` / `GpuBuffer<bf16>` roundtrip on the device.
- `half::f16` and `half::bf16` implement `GpuType`.

### Added — Sprint 6.2 (mma.sync + cp.async + fragments)
- **`TensorCoreOp::MmaSync`** with `MmaShape::M16N8K16` — emits
  `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` (and the `bf16`
  variant). Ampere+ (SM 8.0+) only.
- **Typed fragment containers** — `FragmentA` (4 × `.b32` packed half2),
  `FragmentB` (2 × `.b32` packed half2), `FragmentC` (4 × `.f32`). Pure
  register bags with `pub` fields; helpers (`alloc_a/b/c`,
  `load_fragment_*_global_*`, `store_fragment_c_global_row`) live as
  free functions in `kaio-core::fragment`.
- **`RegisterAllocator::alloc_packed_half2`** — explicit allocator for
  the `.b32` registers `mma.sync` expects for A/B fragments.
- **`MemoryOp::CpAsync*`** variants: `CpAsyncCaSharedGlobal`
  (`cp.async.ca.shared.global [shared], [global], N` for N ∈ {4,8,16}),
  `CpAsyncCommitGroup`, `CpAsyncWaitGroup { n }`. Size validated at
  construction via `MemoryOp::new_cp_async_ca`.
- **`PtxModule::validate()`** + `ValidationError::SmTooLow` — rejects
  kernels at emit time when they use features (`mma.sync`, `cp.async`)
  requiring a higher target SM than the module declares. Surfaced via
  `kaio-runtime::KaioDevice::load_module`.
- **`KaioDevice::load_module(&PtxModule)`** — preferred entrypoint for
  in-memory modules; validates before handing text to the driver.
- **Gate test** `mma_sync_m16n8k16_fragment_gate` — single mma.sync on
  known-value 16×16 × 16×8 inputs, bit-exact assertion (passes on RTX 4090).
- **Smoke test** `cp_async_ca_roundtrip_4_floats` — primitive global →
  shared → global roundtrip via `cp.async.ca` + `commit_group` +
  `wait_group 0` (passes on RTX 4090).
- ptxas verification for `mma.sync` and `cp.async` at `sm_80+`.
- `KernelStats`: `mma`, `cp_async`, `cp_async_commit`, `cp_async_wait`.

### Changed — Phase 6
- Phase 6 master plan corrected: `mma.sync.m16n8k16` requires SM 8.0+
  (Ampere), not SM 7.0+ (Volta). Earlier Volta/Turing shapes (`m8n8k4`,
  `m16n8k8`) are out of scope.

### Added — Sprint 6.3 (IR-level tensor-core matmul)
- **`kaio_ops::matmul_tc`** — first IR-authored tensor-core matmul in
  KAIO. Computes `C = A × B` with f16 × f16 → f32, `m16n8k16.f16.f32`.
  **Internal (`#[doc(hidden)]`)** for now — requires `M % 16 == 0`,
  `N % 8 == 0`, `K % 16 == 0`. Promotes to a public API in Sprint 6.7
  once edge-tile handling lifts the divisibility constraint.
- **Shared-memory fragment loaders** in `kaio-core::fragment`:
  `load_fragment_a_m16n8k16_shared_row` and
  `load_fragment_b_m16n8k16_shared_col` — free-function siblings to
  the Sprint 6.2 global-source loaders, with a caller-supplied
  `row_stride_bytes` / `col_stride_bytes` parameter for non-native
  tile strides.
- 4 GPU correctness tests (tiny, small, rect, medium) — bit-close to
  CPU reference (max error ~1e-7 against a tolerance of 1e-2).
- 6 host-only `validate_dims_tc` tests + 1 PTX-structure test + 1
  ptxas-verify test for shared-source loaders.

### Fixed — Sprint 6.3
- **`ld.global.f16` / `st.shared.f16` are not valid PTX.** Sprint 6.1's
  `MemoryOp::LdGlobal`/`StGlobal`/`LdShared`/`StShared`/`LdParam`
  emitted the wrong type modifier for half-precision loads/stores —
  PTX ISA §8.7.9 lists `f16` and `bf16` as invalid types for `ld`/`st`.
  The correct form loads a 16-bit half into an `.f16` register via
  `ld.global.b16`. Added `PtxType::ptx_memory_suffix()` that collapses
  `F16`/`BF16` to `.b16`. Register declarations and `cvt` still use
  `.f16`. Found by running the first kernel that actually executed an
  f16 load/store on the GPU.
- **`store_fragment_c_m16n8k16_global_row` row stride was hardcoded.**
  The 6.2 helper assumed a standalone 16×8 D matrix (32-byte row
  stride). For a 16×8 tile inside a larger M×N output (the matmul
  case), the stride needs to be `N * 4` bytes. Added a
  `row_stride_bytes: u32` parameter. Callers: `matmul_tc` emits the
  D-store inline (runtime-valued stride), Sprint 6.2's gate test
  passes `32`.

### Changed — Sprint 6.3
- **API change (Sprint 6.2 caller):** `store_fragment_c_m16n8k16_global_row`
  now takes a `row_stride_bytes: u32` parameter. Pre-6.3 callers
  pass `32` to match old behavior.

### Added — Sprint 6.4 (cp.async double-buffered matmul)
- **`kaio_ops::matmul_tc_async`** — double-buffered variant of
  `matmul_tc` that stages A tiles via `cp.async.ca.shared.global`
  (16 B per thread) while B stays synchronous. Pipeline preamble
  issues `A[0]` + commit; K loop waits on current A, issues next A
  before `mma.sync`, recomputes buffer offsets. Same dimension
  constraints as `matmul_tc` (M%16 = N%8 = K%16 = 0) and same
  SM 8.0+ requirement. **Internal (`#[doc(hidden)]`)** for now;
  promotes to public API in Sprint 6.7 alongside `matmul_tc`.
- Shared memory layout uses two buffers per tile: `tile_a[1024]`
  (2 × 512 B, `.align 16` for cp.async destination alignment) and
  `tile_b[512]` (2 × 256 B). Per-block shared = 1.5 KB.
- `kaio-ops::matmul_tc_kernel::emit_load_b_tile` and
  `validate_dims_tc` promoted from private to `pub(crate)` so the
  async sibling module can reuse them (visibility bump only,
  no behavioral change).
- 4 GPU correctness tests (tiny, small, rect, medium) — bit-close
  to CPU reference with identical error floor as 6.3 (max error
  ~1e-7 against a 1e-2 tolerance).
- 2 host tests — `buffer_offsets_toggle` (pure toggle math) and
  `build_matmul_tc_async_ptx_produces_valid_structure`
  (instruction-centric PTX check: cp.async mnemonics, shared-decl
  sizing, exactly-one mma, exactly-two commit_groups).
- Optional env-gated (`KAIO_SPRINT_6_4_TIMING=1`) timing log on the
  medium test — records async vs sync `ms/iter` for Sprint 6.7's
  multi-warp restructure to improve against. Not a benchmark, not
  in CI output. Current baseline: async is ~7% slower than sync at
  1 warp / block — as predicted, overlap gains require 6.7.

### Added — Sprint 6.5 (TC auto-tuner + `load_module` migration)
- **`kaio_ops::matmul_auto_tc`** — **first Phase 6 user-facing API**.
  Tensor-core auto-tuner for f16 × f16 → f32 matmul. Dispatches
  between `matmul_tc` (sync) and `matmul_tc_async` (`cp.async`
  double-buffered) based on cached benchmark data. Conservative
  default on cache miss: sync variant (matches 6.4's observation
  that async is ~7% slower at 1 warp/block; Sprint 6.7 will likely
  invert this default).
  - **Narrow contract, deliberately temporary.** This is a preview
    surface landing in 6.5 to unblock users who want TC dispatch
    today; production performance (60%+ cuBLAS) ships in Sprint
    6.7's multi-warp restructure — *not* here.
  - **Hardware:** NVIDIA Ampere or newer (SM 8.0+). Pre-Ampere
    callers get a clean `KaioError::InvalidConfig` naming both real
    fallback options (pad/convert inputs, or use the f32
    `matmul_auto` path if f16 precision is not required).
  - **Shape:** `M % 16 == 0 && N % 8 == 0 && K % 16 == 0` —
    **temporary**. Sprint 6.7 will relax via edge-tile handling.
  - **Performance:** single-warp-per-block under the hood.
    Correctness-validated, not yet at the Phase 6 target.
- **`kaio_ops::tune_matmul_tc`** — benchmarks both TC variants at
  the given dimensions, caches the faster one to the shared tuner
  cache file (`~/.cache/kaio/tune_cache.json` or the `KAIO_TUNE_CACHE`
  override). Entries are keyed by `(kernel="matmul_tc", sm_target,
  dims)`, coexisting with scalar `matmul` cache entries.
- **`matmul_tc` and `matmul_tc_async` migrated to `load_module`.**
  Both kernels now emit a `PtxModule` via
  `build_matmul_tc_module(sm)` / `build_matmul_tc_async_module(sm)`
  (replacing the older `build_*_ptx() -> String`) and call
  `device.load_module(&module)`. `PtxModule::validate()` inside
  `load_module` catches sub-Ampere targets cleanly with
  `ValidationError::SmTooLow`, surfaced as `KaioError::Validation`.
  The ad-hoc `device.info().compute_capability` checks in both host
  APIs are deleted.
- New shared test helpers at `kaio-ops/tests/common/mod.rs`
  (`patterned_f16_data`, `cpu_matmul_f16xf16_f32`,
  `assert_close_with_k_scaled_tol`) consolidate what 6.3 / 6.4 /
  6.5 test files were duplicating.
- Host regression tests for `load_module` behavior: four new unit
  tests per kernel asserting sub-Ampere targets produce
  `ValidationError::SmTooLow` with `required=80` and the correct
  feature name; plus a cache-coexistence unit test proving
  `matmul` and `matmul_tc` entries don't collide on shared cache
  keys.

### Breaking — Sprint 6.5
- **`KAIO_SM_TARGET` no longer affects tensor-core kernels.** Both
  `matmul_tc` and `matmul_tc_async` (and therefore `matmul_auto_tc`)
  now derive the emitted module's target SM from
  `device.info().compute_capability` at call time rather than from
  the `KAIO_SM_TARGET` env var. The env var continues to be honored
  by scalar `#[gpu_kernel]` kernels where cross-SM testing and
  pre-Ampere support genuinely matter; for tensor-core kernels it
  was always "lie to the kernel about the GPU," which is the
  problem `load_module` validation was built to solve.
  - **User impact:** near-zero — both kernels are still
    `#[doc(hidden)]` until Sprint 6.7, and public-API users only
    reach them through `matmul_auto_tc` which derives SM from the
    device regardless.
  - **Migration:** if you were setting `KAIO_SM_TARGET` specifically
    to target a TC kernel, stop — the kernel will pick the right
    SM based on the actual device. For scalar kernels, the env var
    still works as before.

### Added — Sprint 6.6 (fused TC attention, internal preview)
- **`kaio_ops::attention_tc`** — first fused tensor-core scaled
  dot-product attention in KAIO. `f16 Q × f16 K × f16 V → f32 out`
  via two back-to-back `mma.sync.m16n8k16` instructions with an
  intra-kernel `cvt.rn.f16.f32` bridge between the f32 softmax
  output and the f16 input to the second matmul — the architectural
  contract every production TC-attention implementation (FlashAttention
  v2, xFormers, FasterTransformer) depends on, validated here on
  RTX 4090 bit-close to the CPU reference. `#[doc(hidden)] pub use`
  — **internal preview only** until Phase 7's FlashAttention-TC
  lands and `attention_auto_tc` becomes the real user-facing
  dispatcher (matches the `matmul_auto_tc` pattern from Sprint 6.5).
  - Narrow contract: SM 8.0+ (Ampere or newer), `seq_q % 16 == 0`,
    `seq_k % 16 == 0`, `d_k % 16 == 0`, `d_v % 8 == 0`,
    `seq_k ≤ 384`, `d_k ≤ 128`, `d_v ≤ 128`. Divisibility and
    `seq_k` cap lifted at Phase 7; the seq_k ceiling exists because
    the full softmax scores matrix lives in shared memory (Phase 7's
    online softmax eliminates it).
  - Correctness-first, **not fast**. Single-warp-per-block kernel
    targeting the 6.3 bring-up philosophy — deliberately slow at
    realistic sizes, restructured for throughput at Sprint 6.7.
- **`kaio_ops::attention_tc_causal`** — standard decoder causal-
  masked variant, same signature. Build-time `causal: bool` flag
  drives PTX emission (zero-runtime-cost branching; two distinct
  modules from one Rust builder). Applies `-3.4e38` mask between
  matmul1 and softmax via `setp.gt.u32` + `selp.f32` per-lane
  branchless select. Global-coordinate math regression-gated by
  a dedicated `row0_self_only` canary test.
- **`kaio_core::instr::ArithOp::Selp`** — new IR variant emitting
  `selp{ty} dst, a, b, p` (PTX ISA §9.7.8.1). Branchless conditional
  assignment; required by the 6.6b causal mask and generally useful
  for any lane-predicated data-flow that should avoid warp divergence.
- Per-binary `#[allow(dead_code)]` on `kaio-ops/tests/common/mod.rs` —
  shared helpers file is compiled per test binary and any one binary
  sees different subsets as "used"; the allow silences the false
  positives without hiding genuinely-unused code.
- Eleven new GPU correctness tests on RTX 4090: five shapes for
  `attention_tc`, five shapes for `attention_tc_causal`, plus the
  causal row-0 canary. Five new host unit tests locking down module
  shape (mma count, cvt presence, mask-op presence for causal),
  `sm_70` rejection via `PtxModule::validate()` for both variants,
  and a shared-memory budget regression test (worst-case
  `SharedDecl` sum + alignment ≤ 46 KB). Test counts: 275 host / 133 GPU.

## [0.1.0] — Phase 5: Fused Attention & Community Release

### Added — Phase 5
- **2D block reductions**: `block_reduce_sum/max` now work in 2D kernels
  via linear thread identity (`tidx + tidy * block_dim_x`).
- **Standard attention**: `attention()`, `attention_causal()` — single-head
  scaled dot-product attention with optional causal masking. Three-kernel
  decomposition (Q*K^T, softmax, P*V).
- **FlashAttention**: `attention_flash()`, `attention_flash_causal()` —
  O(d_k) memory per query, no materialized attention matrix. BLOCK_M=1
  design with online softmax and running output rescaling.
- **Auto-tuner**: `tune_matmul()`, `tune_attention()` benchmark kernel
  variants and cache results as JSON. `matmul_auto()`, `attention_auto()`
  dispatch to the best cached variant with deterministic fallback.
- **Windows CI**: GitHub Actions matrix (Ubuntu + Windows). Doc build job.
- **DSL friction report**: 5 documented friction points from attention
  implementation (no `&&`/`||`, 1D grid inference, no `sqrt()`, no compound
  shared assign, no `-inf` literal).
- 24 attention GPU tests + 7 tuner tests.

### Fixed — Phase 5
- 2D block reductions: previously rejected at compile time, now work via
  linear thread identity.

### Changed — Phase 5
- `kaio-ops` now depends on `serde` + `serde_json` (for tuner cache).

## [0.0.4] — Phase 4: Tiled MatMul & Block-Level API

### Added — Phase 4
- **`kaio-ops` crate** with `matmul()` — tiled matrix multiplication on GPU.
  Host-side API: `matmul(&device, &a, &b, &mut c, m, n, k)`.
- **FMA instruction**: `fma(a, b, c)` builtin for f32 fused multiply-add.
- **2D thread blocks**: `#[gpu_kernel(block_size = (16, 16))]` tuple syntax
  with 2D grid launch model (`grid: (u32, u32, u32)`).
- **2D thread indices**: `thread_idx_y()`, `block_idx_y()`, `block_dim_y()`,
  `grid_dim_y()`.
- **Multi-allocation shared memory**: multiple `shared_mem!` calls per kernel,
  each tracked independently via named PTX symbols.
- **Register-tiled matmul kernel**: 64x64 block tiles, 4x4 per thread,
  bank conflict padding (stride 17/65). 31% of cuBLAS sgemm on RTX 4090.
- **Benchmark harness** vs cuBLAS sgemm: deterministic inputs, 5 warm-up,
  20 measured, median TFLOPS. Results in `docs/benchmarks.md`.
- **`KAIO_PTX_STATS=1`**: instruction/register/shared-mem statistics at
  kernel compile time.
- **`KAIO_PTX_ANNOTATE=1`**: source-construct annotations in emitted PTX.
- **`docs/performance.md`**: coalescing patterns, bank conflict avoidance,
  register tiling guide, PTX inspection tool documentation.
- 207 host tests + 41 GPU tests + 1 benchmark.

### Fixed — Phase 4
- **Launch config block_dim mismatch**: cudarc's `for_num_elems()` hardcoded
  block_dim=1024 regardless of declared `block_size`. 1D launch wrapper now
  computes its own LaunchConfig.
- **Shared memory multi-allocation overlap**: `compute_shared_address()` assumed
  offset 0 for all allocations. Fixed with named-symbol base addressing
  (`Operand::SharedAddr`).
- **FMA builtin argument validation**: previously only validated first argument.
  Now all 3 args are checked for f32 type.
- **`block_reduce_*` in 2D kernels**: reductions derive thread identity from
  TidX only, which is wrong for 2D blocks. Now rejected at compile time with
  clear error message (fix planned for Phase 5).

## [0.0.3] — Phase 3 Complete

### Added — Phase 3: Loops, Reductions & Softmax
- **`for`/`while` loop support** in `#[gpu_kernel]`: `for i in start..end`
  with unsuffixed literal coercion (`0..n` where `n: u32`), `while condition`,
  compound assignment (`+=`, `-=`, `*=`, `/=`), loop variable scoping.
- **Shared memory** (`shared_mem![T; N]`): declare block-scoped SRAM buffers
  in kernel code, accessed via `sdata[idx]` with automatic `ld.shared`/
  `st.shared` dispatch. 32-bit addressing distinct from 64-bit global memory.
- **Barrier synchronization** (`bar_sync()`): `bar.sync 0` for block-level
  thread synchronization.
- **Warp shuffle** (`shfl_sync_down/up/bfly(val, delta, width)`): warp-level
  data exchange via `shfl.sync.{down,up,bfly}.b32`. Full-warp (width=32) only.
- **Block-level reductions** (`block_reduce_sum(val)`, `block_reduce_max(val)`):
  ~35-instruction expansion using warp shuffle tree + shared memory cross-warp
  reduction + broadcast to all threads. Auto-allocated `_kaio_reduce_smem`.
- **Softmax kernel** validated on RTX 4090: row-wise numerically-stable softmax
  using while loops, `block_reduce_max`, `block_reduce_sum`, `exp()`. 10 GPU
  tests including accuracy suite (< 1e-5 error vs CPU reference), edge cases
  (all-zeros, uniform, large values, negative, mixed-sign).
- **`Operand::SharedAddr`** in kaio-core for loading named shared allocation
  base addresses.
- **`SharedDecl`** struct + `PtxKernel::shared_decls` for shared memory
  declarations in PTX kernel preamble.
- **6 new kaio-core instruction variants**: `LdShared`, `StShared` (MemoryOp),
  `BarSync`, `ShflSyncDown`, `ShflSyncUp`, `ShflSyncBfly` (ControlOp).
- `KAIO_SM_TARGET` environment variable: set PTX target (default `sm_70`
  for maximum GPU compatibility, override with e.g. `KAIO_SM_TARGET=sm_89`).
- 200 host tests + 24 GPU tests across workspace.

### Fixed — Phase 3
- **Cvt rounding modifier**: `cvt.f32.u32` now correctly emits `cvt.rn.f32.u32`
  (round to nearest for int→float), `cvt.rzi.u32.f32` (round toward zero for
  float→int, matching Rust `as` semantics). Previously rejected by ptxas.
- **Let-binding register aliasing**: `let mut i = tid` now copies to a fresh
  register instead of aliasing. Prevents mutation of source variable when the
  new binding is reassigned (e.g., `i += stride` no longer corrupts `tid`).
- **Variable shadowing**: `let mut i = tid;` can now be used multiple times in
  the same kernel (e.g., once per while loop). Previously errored with
  "variable already defined".

### Changed — Phase 3
- SM target default changed from hardcoded `sm_89` to configurable via
  `KAIO_SM_TARGET` env var (default `sm_70`).

### Added — Phase 2: Proc Macro DSL
- **`#[gpu_kernel]` proc macro** (`kaio-macros`): full pipeline from Rust function
  syntax to executable PTX. Parse (syn → KernelIR) → lower (IR → PTX instructions)
  → codegen (build_ptx with OnceLock + typed launch wrapper).
- **Supported Rust subset**: arithmetic (`+`, `-`, `*`, `/`, `%`, `+=`, `-=`,
  `*=`, `/=`), comparisons (`<`, `<=`, `>`, `>=`, `==`, `!=`), `if`/`else` with
  `@!pred` predicated branches, array indexing with cvta.to.global caching, `let`
  bindings with type inference, type casts via `as`.
- **19 built-in functions**: thread/block/grid intrinsics (`thread_idx_x`,
  `block_idx_x`, `block_dim_x`, `grid_dim_x`), math functions (`sqrt`, `rsqrt`,
  `abs`, `min`, `max`, `sin`, `cos`), synthesized transcendentals (`exp`, `log`,
  `tanh` via PTX `ex2`/`lg2`/`rcp` primitives).
- **Launch wrapper generation**: typed function signatures matching kernel params,
  automatic grid/block calculation, argument marshaling via `PushKernelArg`.
- **`KAIO_DUMP_PTX`** environment variable: set to `1` to write `.ptx` files to
  disk during compilation for inspection.
- **4 E2E kernels validated on RTX 4090**: `vector_add`, `saxpy`, `fused_relu`,
  `fused_gelu` — all produce correct results against CPU reference implementations.
- **10 trybuild compile-fail tests** (CF1–CF10): unsupported types, non-unit
  return, macro invocations, heap allocation, non-built-in function calls, missing
  `block_size`, non-power-of-2 `block_size`, `block_size` > 1024, lifetime
  parameters, `loop` keyword.
- **`kaio` umbrella crate**: prelude module re-exporting macro + runtime, 
  `gpu_builtins` IDE stub module for autocomplete.
- 168 host tests + 5 GPU tests across workspace

### Fixed — Phase 2
- **cvta.to.global register scope across if/else branches**: conversion registers
  were scoped inside branches, causing undefined-register errors in subsequent
  code. Fix: eager emission during parameter loading, before any control flow.
- **PTX float division requires `.approx`/`.rn` modifier**: bare `div.f32` is
  invalid PTX. Fix: type-aware modifier selection in `ArithOp::Div` emission
  (`.approx` for f32, `.rn` for f64).

### Changed — Phase 2
- Renamed project from PYROS to KAIO across entire codebase (commit `50a3ab0`)

### Added — Phase 1: PTX Foundation
- **PTX code generation** (`kaio-core`): IR types modelling complete PTX programs,
  instruction emitters for arithmetic (add, mad, mul.wide), memory (ld.param,
  ld.global, st.global, cvta.to.global), and control flow (setp, bra, ret).
  Emit trait + PtxWriter produce valid PTX text from an IR tree.
- **CUDA runtime wrapper** (`kaio-runtime`): KaioDevice for GPU context management,
  GpuBuffer<T> for typed device memory, KaioModule/KaioFunction for PTX loading
  and kernel launch via cudarc 0.19.
- **End-to-end `vector_add`**: kernel constructed via Rust IR, emitted to PTX,
  loaded into the CUDA driver, launched on RTX 4090 — produces correct results
  for both single-block (3 elements) and multi-block (10,000 elements).
- **Validation**: all PTX instruction emitters verified byte-for-byte against
  nvcc 12.8 output. ptxas offline verification passes. cudarc smoke test confirms
  host↔device data transfer.
- Virtual workspace with umbrella `kaio` crate re-exporting `kaio-core` + `kaio-runtime`
- 53 host-side tests + 9 GPU-gated tests, 82.8% line coverage
- Per-sprint architectural decision records in `docs/development/sprints/`

### Changed — Phase 1
- PTX ISA version corrected from 7.8 to 8.7 (CUDA 12.8)
- Register declarations use `.b32`/`.b64` (untyped) matching nvcc convention

## [0.0.1] — 2026-04-10

### Added
- Name reservation crate with metadata, README, dual MIT/Apache-2.0 license
- Project design docs: index.md, implementation.md, phases.md, success-criteria.md
