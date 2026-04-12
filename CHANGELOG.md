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
