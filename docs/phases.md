# KAIO — Development Phases

## Overview

KAIO is developed in five phases, each producing a shippable milestone. Phases are designed to be Forge-friendly: decomposable into independent sprints with clear inputs, outputs, and testable boundaries.

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

**Forge Sprint Breakdown:**
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

**Forge Sprint Breakdown:**
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

## Phase 4: Tiled MatMul & Block-Level API

**Goal:** Implement tiled matrix multiplication and formalize the block-level programming model.

**Duration:** 3-4 weeks

**Deliverables:**
- `kaio-ops` crate with block-level abstractions
- `block_load` / `block_store` — coalesced global ↔ shared memory transfers
- `block_dot` — tiled matrix multiply via shared memory
- Tiled matmul kernel benchmarked against cuBLAS
- Performance within 60% of cuBLAS for large matrices (correctness is the priority)
- Memory coalescing analysis at compile time (warn on uncoalesced patterns)

**What Gets Built:**
- Block abstraction: `Block<T, const SIZE: usize>` type representing a tile in shared memory
- Tiling logic: automatic decomposition of large operations into block-sized tiles
- Double buffering: overlap computation with memory loads (pipeline optimization)
- Memory coalescing checker: analyze access patterns at compile time
- `block_dot` using the classic shared-memory tiled matmul algorithm
- Benchmark harness comparing against cuBLAS `sgemm`

**Forge Sprint Breakdown:**
| Sprint | Scope | Agent Work |
|--------|-------|------------|
| 4.1 | Block type + block_load/block_store | Type design, coalesced load codegen |
| 4.2 | Block arithmetic operations | Element-wise ops on Block types |
| 4.3 | Tiled matmul — naive | Shared memory tiling, correctness first |
| 4.4 | Tiled matmul — optimized | Double buffering, register tiling |
| 4.5 | Memory coalescing analysis | Compile-time access pattern checker |
| 4.6 | cuBLAS benchmark harness | Automated comparison, regression tracking |
| 4.7 | Block-level API polish | Documentation, ergonomic refinements |
| 4.8 | Integration tests + benchmarks | Full validation suite |

**Key Risk:** Performance gap with cuBLAS. Mitigation: 60% target is realistic for a first implementation; optimization is iterative post-launch.

---

## Phase 5: Fused Attention & Community Release

**Goal:** Implement fused multi-head attention and publish KAIO to crates.io.

**Duration:** 3-4 weeks

**Deliverables:**
- Fused multi-head attention kernel (FlashAttention-style)
- Attention kernel validated against PyTorch `F.scaled_dot_product_attention`
- Auto-tuning: grid search over block sizes and tiling configurations
- `kaio` published to crates.io as v0.1.0
- Documentation: README, API docs, tutorial kernels, architecture guide
- Blog post / r/rust announcement

**What Gets Built:**
- Fused attention: Q×K^T → scale → mask → softmax → ×V in one kernel
- Online softmax (numerically stable, no full materialization of attention matrix)
- Auto-tuner: compile multiple kernel variants, benchmark, select best
- `cargo doc` documentation for all public APIs
- Tutorial examples: vector_add, saxpy, gelu, softmax, matmul, attention
- CI pipeline: Windows + Linux, test + clippy + fmt

**Forge Sprint Breakdown:**
| Sprint | Scope | Agent Work |
|--------|-------|------------|
| 5.1 | Fused attention — forward pass | Q×K^T → scale → softmax → ×V |
| 5.2 | Online softmax implementation | Numerically stable streaming softmax |
| 5.3 | Masking support | Causal mask, padding mask |
| 5.4 | Auto-tuner framework | Config grid, benchmark runner, selection |
| 5.5 | Documentation | API docs, README, tutorials |
| 5.6 | CI/CD pipeline | GitHub Actions: Windows + Linux matrix |
| 5.7 | Crates.io publication prep | Metadata, license, dry-run publish |
| 5.8 | Community launch | Blog post, r/rust, benchmarks |

**Key Risk:** FlashAttention is algorithmically complex. Mitigation: start with standard attention (materialize full attention matrix), optimize to FlashAttention incrementally.

---

## Post-v0.1 Roadmap (Not Scoped Yet)

These are future directions, not commitments:

- **Backward pass / autodiff:** Enable training, not just inference
- **Quantized kernels:** INT8/INT4 for efficient inference
- **Multi-GPU:** Kernel launch across multiple devices
- **AMD ROCm support:** Emit GCN/CDNA ISA instead of PTX
- **PyO3 bindings:** Use KAIO kernels from Python
- **Tensor core ops:** `mma` instructions for fp16/bf16 matmul
- **Async memory copies:** `cp.async` for Ampere+ architectures
