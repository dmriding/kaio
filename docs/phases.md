# KAIO — Development Phases

## Overview

KAIO is developed in phases, each producing a shippable milestone. Phases are designed to be sprint-friendly: decomposable into independent sprints with clear inputs, outputs, and testable boundaries.

Each phase builds on the previous. No phase is started until the prior phase meets its success criteria (see `success-criteria.md`).

---

## Phase 1: PTX Foundation ✅

**Goal:** Emit valid, executable PTX from Rust code.

**Status:** Complete

**Deliverables:**

- `kaio-core` crate with PTX IR types and instruction emission
- `kaio-runtime` crate with basic device management and kernel launch
- One working end-to-end kernel: `vector_add`
- PTX output passes `ptxas --verify` on both Windows and Linux
- Kernel executes on a real GPU and produces correct output

**What Gets Built:**

- PTX IR structs: `PtxModule`, `PtxKernel`, `PtxInstruction`, `PtxRegister`, `PtxParam`
- Instruction emitters for Priority 1 (arithmetic) and Priority 2 (memory) instructions
- Special register access (`%tid.x`, `%ctaid.x`, `%ntid.x`)
- Basic control flow (`setp`, predicated branches, `ret`)
- `PtxWriter` that produces formatted `.ptx` text output
- Runtime: device enumeration, memory alloc/free, host↔device transfer, kernel launch
- `vector_add` kernel built manually via the IR API (no macro yet)

**Sprint Breakdown:**
| Sprint | Scope | Agent Work |
|--------|-------|------------|
| 1.1 | PTX type system + register model | Struct definitions, type mapping |
| 1.2 | Arithmetic instruction emitters | `add`, `sub`, `mul`, `div`, `mad`, `fma` for all types |
| 1.3 | Memory instruction emitters | `ld.global`, `st.global`, `ld.param`, `mov`, `cvt` |
| 1.4 | Control flow + special registers | `setp`, `bra`, `ret`, `%tid`, `%ctaid`, `%ntid` |
| 1.5 | PtxWriter + module emission | Formatting, label generation, `.ptx` file output |
| 1.6 | Runtime: device + memory | `cudarc` wrapper, `GpuBuffer<T>`, transfers |
| 1.7 | Runtime: kernel launch | PTX loading, launch config, `vector_add` E2E |
| 1.8 | Testing + validation | `ptxas` verification, correctness checks, coverage |

**Key Risk:** `cudarc` may not expose all CUDA driver API features needed. Mitigation: wrap raw FFI calls for missing features.

---

## Phase 2: Proc Macro DSL ✅

**Goal:** Users write GPU kernels in Rust syntax using `#[gpu_kernel]`.

**Status:** Complete

**Deliverables:**

- `kaio-macros` crate with `#[gpu_kernel]` attribute macro
- Macro parses a subset of Rust (arithmetic, comparisons, `if/else`, `let`, array indexing)
- Macro emits PTX via `kaio-core` at compile time
- Macro generates launch wrapper function
- Working kernels: `vector_add`, `saxpy`, `fused_gelu`, `fused_relu`
- Compile-fail tests for invalid kernel signatures (via `trybuild`)

**What Gets Built:**

- `syn`-based parser for the supported Rust subset
- AST → KAIO IR lowering
- Type inference and validation within kernel bodies
- Launch wrapper code generation (grid/block computation, argument marshaling)
- Built-in function registry (`thread_idx_x()`, `block_idx_x()`, `sqrt()`, `exp()`, etc.)
- Math function emitters (Priority 5 instructions: `ex2`, `lg2`, `rcp`, `sqrt`, `sin`, `cos`)

**Sprint Breakdown:**
| Sprint | Scope | Agent Work |
|--------|-------|------------|
| 2.1 | Macro skeleton + function parsing | `syn` visitor, parameter extraction |
| 2.2 | Expression lowering: arithmetic | Binary ops → PTX arithmetic instructions |
| 2.3 | Expression lowering: comparisons + control flow | `if/else` → `setp` + branch |
| 2.4 | Array indexing + memory access | `a[idx]` → address calc + `ld.global`/`st.global` |
| 2.5 | Built-in functions | Thread/block index functions, math functions |
| 2.6 | Launch wrapper generation | Grid/block calc, argument marshaling, `quote!` output |
| 2.7 | Type validation + error messages | Compile-fail tests, clear diagnostics |
| 2.8 | End-to-end kernel tests | `vector_add`, `saxpy`, `fused_gelu`, `fused_relu` |

**Key Risk:** Proc macro debugging is painful. Mitigation: extensive `trybuild` test suite, `KAIO_DUMP_PTX` for inspecting output.

---

## Phase 3: Loops, Reductions & Softmax ✅

**Goal:** Support loops in kernel DSL and implement the first block-level reduction operations.

**Status:** Complete (commit `0691be8`, 2026-04-11)

**Deliverables:**

- `for` and `while` loop support in `#[gpu_kernel]`
- Shared memory declaration and access (`shared_mem![T; N]`)
- `bar_sync()` (thread block synchronization)
- Warp-level shuffle operations (`shfl_sync_down/up/bfly`)
- Reduction primitives: `block_reduce_sum`, `block_reduce_max`
- Working kernel: `softmax` (row-wise, single block per row)
- Softmax output validated within f32 tolerance against CPU reference
- 200 host tests + 24 GPU tests across workspace

**Sprint Breakdown (actual):**
| Sprint | Scope | Key Deliverable |
|--------|-------|-----------------|
| 3.1 | Loops + compound assignment | `for`/`while` lowering, `+=`/`-=`/`*=`/`/=` |
| 3.2 | Shared memory + shuffle (kaio-core) | `LdShared`, `StShared`, `ShflSync*`, `BarSync` instructions |
| 3.3 | Shared memory in macro DSL | `shared_mem![f32; 256]` syntax, 32-bit addressing |
| 3.4 | Barrier + shuffle built-ins | `bar_sync()`, `shfl_sync_down/up/bfly()` in `#[gpu_kernel]` |
| 3.5 | Reduction primitives | `block_reduce_sum`, `block_reduce_max` (~35 PTX instructions each) |
| 3.6 | Softmax kernel | Row-wise numerically-stable softmax on RTX 4090 |
| 3.7 | Accuracy validation suite | 10 GPU tests incl. edge cases, < 1e-5 error vs CPU reference |
| 3.8 | Polish + coverage + docs | Cvt rounding fix, variable shadowing fix, `KAIO_SM_TARGET` config |

**Key Decisions:** Sprint logs with full reasoning traces in
[docs/development/sprints/phase3/](development/sprints/phase3/).

---

## Phase 4: Tiled MatMul & Block-Level API ✅

**Goal:** Implement tiled matrix multiplication through KAIO's `#[gpu_kernel]` macro
and introduce `kaio-ops` as a host-side operations library.

**Status:** Complete (v0.0.4, 2026-04-11)

**Deliverables:**

- `kaio-ops` crate with `matmul()` host-side API
- FMA instruction + 2D thread blocks + 2D grid launch model
- Multi-allocation shared memory (named PTX symbols)
- Naive tiled matmul (16x16, ~8% of cuBLAS) → register-tiled (64x64,
  4x4 per thread, 31% of cuBLAS sgemm on RTX 4090)
- Benchmark harness vs cuBLAS with documented methodology
- PTX inspection tools: `KAIO_PTX_STATS`, `KAIO_PTX_ANNOTATE`
- Performance guide (`docs/performance.md`)
- 207 host tests + 41 GPU tests + 1 benchmark

**Sprint Breakdown (actual):**
| Sprint | Scope | Key Deliverable |
|--------|-------|-----------------|
| 4.1 | FMA + 2D indices + 2D launch | `fma()`, `block_size = (X, Y)`, 2D grid |
| 4.2 | Multi-allocation shared memory | Named-symbol addressing, launch fix |
| 4.3 | Naive tiled matmul | 16x16 tiles, correctness baseline |
| 4.4 | kaio-ops crate + host API | `matmul()` public API |
| 4.5 | Benchmark harness + cuBLAS | Deterministic timing, TFLOPS comparison |
| 4.6 | Register tiling optimization | 64x64, 4x4/thread, 31% cuBLAS |
| 4.7 | PTX inspection + performance docs | Stats, annotations, performance guide |
| 4.8 | Polish + integration tests + publish | CHANGELOG, README, version bump |

**Key Decisions:** Sprint logs with full reasoning traces in
[docs/development/sprints/phase4/](development/sprints/phase4/).

**Performance:** 31% of cuBLAS at large sizes. Remaining gap requires
vectorized loads (LDG.128) and double buffering — planned for Phase 5+.
Coalescing analysis deferred to Phase 5+ when block_load/block_store
abstractions provide analyzable access patterns.

---

## Phase 5: Fused Attention & Community Release

**Goal:** Implement fused multi-head attention, add it to `kaio-ops`,
build an auto-tuning framework, and ship v0.1.0 to crates.io.

**Status:** Complete (v0.1.0, 2026-04-12)

**Deliverables:**

- 2D block reduction support (hard gate — resolved Sprint 5.1)
- Standard attention: Q×K^T → scale → mask → softmax → ×V
- FlashAttention-style online softmax (stretch goal)
- `kaio_ops::attention()` validated against CPU reference
- Auto-tuner: block/tile size grid search with JSON cache
- CI pipeline: Windows + Linux matrix
- v0.1.0 published to crates.io
- Blog post / r/rust announcement

**Sprint Breakdown (actual):**
| Sprint | Scope | Key Deliverable |
|--------|-------|-----------------|
| 5.1 | 2D reductions (hard gate) | Linear tid for 2D blocks, unblocks attention |
| 5.2 | Standard attention — forward pass | Q×K^T → scale → softmax → ×V, single-head |
| 5.3 | Masking + validation | Causal mask, CPU reference validation, DSL friction report |
| 5.4 | FlashAttention — online softmax (stretch) | Tiled attention, O(n) memory |
| 5.5 | Auto-tuner | Block/tile size grid search, benchmark runner |
| 5.6 | CI/CD + platform | Windows CI, Linux verification |
| 5.7 | v0.1.0 prep | API stability review, cargo doc, publish |
| 5.8 | Community launch | Blog post, r/rust, benchmarks |

**Key Decisions:** Standard attention first (5.2-5.3), FlashAttention
second (5.4) — same pattern as naive → optimized matmul in Phase 4.
FlashAttention is a stretch goal; Phase 5 ships with standard attention
if FlashAttention is not stable. Sprint logs in
[docs/development/sprints/phase5/](development/sprints/phase5/).

**Key Risk:** FlashAttention (online softmax with running correction,
tiled Q/K/V, output rescaling) is the hardest sprint in the project.
Mitigation: standard attention baseline validates correctness before
optimization; 5.4 is internally split into online softmax in isolation,
then full tiled attention.

---

## Phase 6: Tensor Cores & Async Copies ✅

**Goal:** Add fp16/bf16 type support, tensor core instructions
(`mma.sync`), and async memory copies (`cp.async`) to `kaio-core`.
Implement a tensor-core matmul in `kaio-ops`. Target 60%+ of cuBLAS
sgemm (up from 31% with scalar FMA).

**Status:** Complete (v0.2.0, 2026-04-13; polish landed in v0.2.1,
2026-04-14)

**Deliverables:**

- `PtxType::F16` / `PtxType::BF16` + `RegKind::H` / `RegKind::Hb`
  (packed-half2 register allocator for `mma.sync` fragments)
- `TensorCoreOp::MmaSync` with `MmaShape::M16N8K16` — emits
  `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` (Ampere+ only)
- `MemoryOp::CpAsync*` variants + `PtxModule::validate()` with
  `SmTooLow` rejection at emit time for SM-gated features
- `KaioDevice::load_module(&PtxModule)` as the preferred runtime
  entrypoint; raw `load_ptx(&str)` deprecated in v0.2.1 (still
  supported for external PTX research use cases)
- `kaio_ops::matmul_tc` / `matmul_tc_async` / `matmul_auto_tc` —
  tensor-core matmul with auto-tuner cache and size-heuristic
  cache-miss default
- Three standalone showcase examples (`examples/fused_silu_gate`,
  `gelu_comparison`, `rms_norm`) that build from a fresh clone
- Host-level codegen regression tests (6.10 D2) for macro lowering
  invariants, running on CI without a GPU
- **Performance:** 82.3% sync / 92.5% async of cuBLAS sgemm at 4096²
  on RTX 4090 (async past the 90% stretch target — see
  `docs/performance.md` for the full matrix and apples-to-apples
  disclaimer)

**Sprint Breakdown (actual):**
| Sprint | Scope | Key Deliverable |
|--------|-------|-----------------|
| 6.1 | fp16/bf16 types | `PtxType::F16/BF16`, `RegKind::H/Hb`, `cvt` rounding, `half` crate integration |
| 6.2 | mma.sync + cp.async in kaio-core | `TensorCoreOp`, `CpAsync`, typed fragments, standalone mma.sync correctness test |
| 6.3 | Tensor-core matmul (IR API) | Basic `mma.sync` matmul, manual loads, SM 8.0+ (m16n8k16) |
| 6.4 | Double-buffered matmul | `cp.async` pipeline on top of 6.3 |
| 6.5 | Integration + auto-tuner | 3-way dispatch (scalar / TC / TC+async), `PtxModule::validate()` gate |
| 6.6 | TC attention + causal mask | `mma.sync` in FlashAttention inner loops, `selp` for masking |
| 6.7 | Multi-warp + edge tiles + benchmarks | 64×64 block tile, M/N divisibility lifted, 79.9% sync / 85.1% async of cuBLAS at 4096²; `matmul_tc[_async]` promoted to stable pub |
| 6.7b | Bank-conflict padding + D10 hoist | Col-stride pad 32→36 B, fragment-loader `(group_id, tig)` hoist, async 92.5% / sync 82.3% (LDG.128 IR primitive landed as unused-future-anchor) |
| 6.8 | Showcase examples for v0.2.0 | Three standalone Cargo projects under `examples/` |
| 6.9 | Polish + v0.2.0 publish | CHANGELOG, README, version bump, crates.io |
| 6.10 | Close open threads | D2 host-level codegen regression tests, D1a macro migration to `load_module`, D1b `load_ptx` deprecation, D3 test-helper env-var hygiene (v0.2.1) |

**Key Decisions:** IR kernels are internal performance primitives (not
user-facing); fragments are typed containers, not raw register arrays;
cp.async is pluggable on top of the sync kernel, not foundational. Full
reasoning traces in [docs/development/sprints/phase6/](development/sprints/phase6/).

---

## Phase 7: Quantized Kernels & Candle Bridge ✅

**Goal:** INT8 / INT4 dequantize-matmul for efficient inference, fused
QKV projections for transformer decode, and a candle bridge crate
(`kaio-candle`) for Rust ML framework integration.

**Status:** Complete (v0.4.0, 2026-04-18)

**Deliverables:**

- `kaio_ops::matmul_int8` — W8A8 symmetric i8 × i8 → f32, 80–94 TOPS
  at 4096³ on RTX 4090 sm_89
- `kaio_ops::matmul_int4` — W4A16 GPTQ-style packed INT4, 49–58 TOPS
  at 4096³ (95–116% of cuBLAS sgemm)
- `kaio_ops::qkv_project_int8` / `qkv_project_int4` — fused tri-output
  QKV projections, ~3× decode over standalone calls
- `kaio-candle` bridge crate — 8 forward ops + 2 backward
  (matmul_tc, matmul_tc_async), event-based stream sync (CUDA Graph
  compatible), 41 GPU tests
- DSL completeness: bitwise operators, short-circuit `&&`/`||`,
  compound bitwise assignment, warp/block reductions
- DSL-vs-compiled-Rust documentation clarification (issue #13),
  RFC-0001 pointer-syntax draft published

**Sprint Breakdown (actual):**

| Sprint | Scope | Status |
|--------|-------|--------|
| 7.0 | DSL completeness (bitops + short-circuit + compound bitwise assign) | Complete (v0.2.1) |
| 7.0.5 | Pre-7.1 ergonomics fast-track — debug-build note, error-span audit, debugging guide | Complete (v0.2.2) |
| 7.1 | INT8 dequantize-matmul — `matmul_int8`, W8A8 symmetric, 80–94 TOPS at 4096³ | Complete (v0.3.0) |
| 7.1.5 | Warp + block reductions — `warp_reduce_sum/max/min`, `block_reduce_min` | Complete |
| 7.2 | INT4 dequantize-matmul — `matmul_int4`, W4A16 GPTQ-style, 49–58 TOPS at 4096³ | Complete |
| 7.3 | Fused tri-output QKV projection — `qkv_project_int8` + `qkv_project_int4` | Complete |
| 7.3.5 | Design S+½P optimization — INT8 K-loop ping-pong, barriers 7→4 | Complete |
| 7.4a | `kaio-candle` bridge — 5 forward `CustomOp` bindings, 15 GPU tests | Complete |
| 7.4b | `kaio-candle` — `matmul_int8` binding + direct-call pattern for QKV ops | Complete |
| 7.4c | `kaio-candle` — event-based stream sync (replaces `cuCtxSynchronize`) | Complete |
| 7.4d | `kaio-candle` — matmul_tc + matmul_tc_async backward (autograd) | Complete |

**Key Decisions:** Sprint logs with full reasoning traces in
[docs/development/sprints/phase7/](development/sprints/phase7/).

---

## Phase 8: PyO3 Bindings

**Goal:** Thin Python wrapper around `kaio-ops` functions. Python users
gain access to tensor-core matmul, fused attention, and quantized
matmul on Windows without Triton's Linux-only constraint.

**Status:** Planned

**Depends on:** Phase 7 complete (v0.4.0, 2026-04-18).

---

## Phase 8.5: Pointer Syntax (RFC-0001)

**Goal:** Accept `*mut [T]` / `*const [T]` in `#[gpu_kernel]`
signatures as the primary parameter syntax. `&mut [T]` / `&[T]`
retained as permanent ergonomic sugar.

Resolves the DSL-vs-compiled-Rust communication gap raised in
issue #13: `*mut [T]` signals "device pointer, no aliasing contract"
using standard Rust syntax. ~20-line parser change in
`parse_kernel_signature` (`Type::Ptr` arm alongside `Type::Reference`).

See [RFC-0001](rfcs/rfc-0001-pointer-syntax.md) for the full design.

**Status:** Planned (RFC drafted, docs clarification shipped in v0.4.0)

**Depends on:** Phase 8 complete.

---

## Phase 9: Attention Backward & Kernel Deepening

**Goal:** FlashAttention backward pass (new PTX kernels — softmax
recomputation, tiled Q/K/V backward, causal mask in reverse) and
further kernel improvements (bf16 TC matmul variant, `ldmatrix.sync`
for additional TC headroom).

**Status:** Planned

**Depends on:** Phase 8.5 complete.

---

## Unscoped — Community-Driven

- **Multi-GPU:** Kernel launch across multiple devices. Requires
  NCCL-style communication primitives. Deferred until there's demand.
- **AMD ROCm:** HIP compatibility path rather than a native GCN/CDNA
  emitter. The tooling gap is large and the Windows story is
  nonexistent. Community contributions welcome.
