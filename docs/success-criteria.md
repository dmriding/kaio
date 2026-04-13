# KAIO — Success Criteria

## Universal Quality Gates

These gates apply to ALL phases. No phase is considered complete unless every gate passes.

### Code Quality

| Gate | Requirement | How to Verify |
|------|-------------|---------------|
| **`cargo fmt`** | All code formatted before each commit | `cargo fmt --check` returns 0 |
| **`cargo clippy`** | Zero warnings across all crates | `cargo clippy --workspace -- -D warnings` returns 0 |
| **Test Coverage** | Minimum 60% line coverage across the workspace | `cargo llvm-cov --workspace` reports ≥60% |
| **Documentation** | All public APIs have doc comments | `#![deny(missing_docs)]` compiles cleanly |
| **No `unsafe` without justification** | Every `unsafe` block has a `// SAFETY:` comment explaining the invariant | Manual review |

### Platform

| Gate | Requirement |
|------|-------------|
| **Windows** | All tests pass on Windows 10/11 with NVIDIA GPU |
| **Linux** | All tests pass on Ubuntu 22.04+ with NVIDIA GPU |
| **CI** | Automated test runs (GitHub Actions, Ubuntu-only for now; Windows planned for release) |

### Process

| Gate | Requirement |
|------|-------------|
| **`cargo fmt`** | Run before every commit, enforced by pre-commit hook |
| **Commit messages** | Conventional commits format: `feat:`, `fix:`, `test:`, `docs:`, `refactor:` |
| **No dead code** | `#![deny(dead_code)]` compiles cleanly (with allowances for future API surface) |

---

## Phase 1: PTX Foundation

### Functional Criteria

| # | Criterion | Validation Method |
|---|-----------|-------------------|
| 1.1 | PTX IR can represent a complete `vector_add` kernel | Unit test: build IR, assert all nodes present |
| 1.2 | Emitted PTX passes `ptxas --verify` without errors | Integration test: emit PTX, shell out to `ptxas` |
| 1.3 | Emitted PTX passes `ptxas --verify` on **both Windows and Linux** | Cross-platform CI or manual verification |
| 1.4 | `vector_add` kernel executes on GPU and produces correct output | Integration test: `[1,2,3] + [4,5,6] = [5,7,9]` |
| 1.5 | Correct results for all supported dtypes: `f32`, `f64`, `i32`, `u32` | Parameterized integration tests |
| 1.6 | Device enumeration returns correct GPU name and compute capability | Integration test against known hardware |
| 1.7 | Host → device → host round-trip preserves data exactly | Integration test: send 10K elements, read back, assert equal |
| 1.8 | OOM error is caught and returned as `KaioError::OutOfMemory` | Integration test: attempt allocation larger than device memory |

### Coverage Targets

| Crate | Minimum Line Coverage |
|-------|----------------------|
| `kaio-core` | 70% (instruction emitters are highly testable) |
| `kaio-runtime` | 60% (some paths require GPU, harder to test in CI) |
| **Workspace total** | **≥60%** |

### Performance Criteria (Phase 1)

None. Phase 1 is about correctness, not performance. If `vector_add` runs and produces the right answer, Phase 1 is done.

---

## Phase 2: Proc Macro DSL

### Functional Criteria

| # | Criterion | Validation Method |
|---|-----------|-------------------|
| 2.1 | `#[gpu_kernel]` compiles a valid `vector_add` kernel | Integration test: macro expansion → launch → verify |
| 2.2 | `#[gpu_kernel]` compiles `saxpy`: `y = a * x + y` | Integration test: known input/output pairs |
| 2.3 | `#[gpu_kernel]` compiles `fused_gelu` activation | Integration test: compare against CPU GELU reference |
| 2.4 | `#[gpu_kernel]` compiles `fused_relu` activation | Integration test: compare against CPU ReLU reference |
| 2.5 | Invalid kernel signatures produce clear compile-time errors | `trybuild` compile-fail tests (minimum 10 cases) |
| 2.6 | Error messages include the source location within the kernel body | `trybuild` output inspection |
| 2.7 | `if/else` branches produce correct predicated code | Integration test: kernel with conditional logic |
| 2.8 | Type casting (`as`) produces correct `cvt` instructions | Integration test: `f32` → `i32`, `i32` → `f32`, etc. |
| 2.9 | `KAIO_DUMP_PTX=1` writes `.ptx` files to disk | Integration test: check file existence and validity |
| 2.10 | Math functions produce results within tolerance of CPU reference | Integration test: `sqrt`, `exp`, `log`, `tanh` against `f32` stdlib |

### Compile-Fail Test Cases (minimum set)

| # | Invalid Code | Expected Error |
|---|-------------|----------------|
| CF1 | Kernel parameter is `String` | "unsupported type: String is not a GPU-compatible type" |
| CF2 | Kernel returns a value | "GPU kernels must return `()`" |
| CF3 | Kernel uses `println!` | "macro invocations are not supported in GPU kernels" |
| CF4 | Kernel uses `Vec::new()` | "heap allocation is not supported in GPU kernels" |
| CF5 | Kernel calls a non-built-in function | "function calls are not yet supported in GPU kernels" |
| CF6 | Missing `block_size` attribute | "`block_size` is required in `#[gpu_kernel(...)]`" |
| CF7 | `block_size` not power of 2 | "`block_size` must be a power of 2" |
| CF8 | `block_size` exceeds 1024 | "`block_size` cannot exceed 1024" |
| CF9 | Kernel parameter uses lifetime | "lifetime parameters are not supported in GPU kernels" |
| CF10 | Kernel uses `loop` | "`loop` is not supported in GPU kernels — use `for` or `while`" |

### Coverage Targets

| Crate | Minimum Line Coverage |
|-------|----------------------|
| `kaio-core` | 70% |
| `kaio-runtime` | 60% |
| `kaio-macros` | 65% (proc macro logic is testable via expansion) |
| **Workspace total** | **≥60%** |

### Numerical Accuracy Criteria

| Function | Max Absolute Error vs CPU `f32` | Max Relative Error |
|----------|--------------------------------|-------------------|
| `sqrt` | 1e-6 | 1e-5 |
| `exp` | 1e-5 | 1e-4 |
| `log` | 1e-5 | 1e-4 |
| `tanh` | 1e-5 | 1e-4 |
| `sin` / `cos` | 1e-5 | 1e-4 |
| `gelu` | 1e-4 | 1e-3 |

---

## Phase 3: Loops, Reductions & Softmax

### Functional Criteria

| # | Criterion | Validation Method |
|---|-----------|-------------------|
| 3.1 | `for` loops compile and execute correctly | Integration test: sum of 0..N |
| 3.2 | `while` loops compile and execute correctly | Integration test: iterative convergence |
| 3.3 | Shared memory allocation works for declared arrays | Integration test: write to shared, barrier, read back |
| 3.4 | `bar.sync` prevents data races in shared memory | Integration test: producer-consumer pattern |
| 3.5 | `block_reduce_sum` produces correct results | Integration test: known sums, compare to CPU |
| 3.6 | `block_reduce_max` produces correct results | Integration test: known max values |
| 3.7 | `softmax` kernel produces results within tolerance of PyTorch | Integration test: random matrices, compare to `torch.softmax` |
| 3.8 | `softmax` handles edge cases: all zeros, all same value, very large values | Integration test: specific edge case inputs |
| 3.9 | Shared memory usage reported at compile time | Build output includes "shared memory: N bytes" |
| 3.10 | Warning emitted if shared memory exceeds SM limit | Compile-time diagnostic test |

### Numerical Accuracy Criteria

| Kernel | Max Absolute Error vs PyTorch | Input Size |
|--------|-------------------------------|------------|
| `softmax` (small) | 1e-5 | 128 × 128 |
| `softmax` (medium) | 1e-4 | 1024 × 1024 |
| `softmax` (large) | 1e-3 | 4096 × 4096 |

### Coverage Targets

| Crate | Minimum Line Coverage |
|-------|----------------------|
| `kaio-core` | 70% |
| `kaio-runtime` | 60% |
| `kaio-macros` | 65% |
| **Workspace total** | **≥65%** |

Note: Coverage target increases by 5% from Phase 2. Each phase should improve coverage, not just maintain it.

---

## Phase 4: Tiled MatMul & Block-Level API

**Status:** Complete (v0.0.4)

### Functional Criteria

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 4.1 | FMA instruction + 2D thread blocks + 2D launch model | Done | `fma()`, `block_size = (X,Y)`, grid tuple |
| 4.2 | Multi-allocation shared memory (named PTX symbols) | Done | `shared_mem!` with named-symbol addressing |
| 4.3 | Tiled `matmul` produces correct results for square matrices | Done | Naive 16×16 + register-tiled 64×64 |
| 4.4 | Tiled `matmul` produces correct results for non-square matrices | Done | (M×K) × (K×N) for various M, K, N |
| 4.5 | Tiled `matmul` handles non-tile-aligned dimensions | Done | Prime-number dimensions tested |
| 4.6 | `kaio-ops` crate with `matmul()` host API | Done | First publish at v0.0.4 |
| 4.7 | PTX inspection tools | Done | `KAIO_PTX_STATS`, `KAIO_PTX_ANNOTATE` |
| 4.8 | Benchmark harness vs cuBLAS | Done | Deterministic timing, TFLOPS comparison |

**Rescoped items** (intentionally deferred, not abandoned):
- `block_load`/`block_store` abstractions — matmul uses `shared_mem!` +
  explicit indexing instead. Deferred to when analyzable access patterns
  are needed for coalescing analysis.
- Compile-time coalescing warnings — requires `block_load`/`block_store`.
- Compile-time bank conflict analysis — deferred; padding documented in
  `docs/performance.md`.

### Performance Criteria

| Metric | Original Target | Actual Result | Notes |
|--------|----------------|---------------|-------|
| Matmul TFLOPS (4096×4096, f32) | ≥60% of cuBLAS | 31% (17.44 vs 56.00) | Register-tiled, scalar PTX |
| Naive baseline | 10-20% of cuBLAS | ~8% | 16×16 tiles |
| Speedup over naive | — | 3.8× | At large sizes |

31% is good enough for custom ops. Reaching 60%+ requires vectorized
loads (LDG.128) and double buffering — planned for Phase 6 (tensor cores).

### Coverage Targets

| Crate | Minimum Line Coverage |
|-------|----------------------|
| `kaio-core` | 70% |
| `kaio-runtime` | 60% |
| `kaio-macros` | 65% |
| `kaio-ops` | 65% |
| **Workspace total** | **≥65%** |

---

## Phase 5: Fused Attention & Community Release

### Functional Criteria

| # | Criterion | Status | Validation Method |
|---|-----------|--------|-------------------|
| 5.1 | 2D block reductions work (hard gate) | Done | GPU tests: square, asymmetric, identity-based |
| 5.2 | Standard attention produces correct output | Done | 7 GPU tests vs CPU reference |
| 5.3 | Causal masking works correctly | Done | 6 causal tests + direct mask verification |
| 5.4 | FlashAttention uses O(n) memory (stretch goal) | Done | 9 flash tests, flash_matches_standard validation |
| 5.5 | Auto-tuner selects optimal block/tile size | Done | 7 tuner tests (tune, auto, fallback, d_k guard) |
| 5.6 | CI runs on Windows + Linux | Done | GitHub Actions matrix (Ubuntu + Windows) |
| 5.7 | `cargo publish --dry-run` succeeds for all crates | Done | v0.1.0 published to crates.io |
| 5.8 | DSL friction points documented for Phase 6 | Done | Sprint 5.3 friction report (5 points) |

**Note:** Validation is against CPU reference implementation, not
PyTorch. CPU reference is standard matmul + softmax + matmul in f32.
Tolerance scales with magnitude at longer sequence lengths (chained
matmuls accumulate FP error).

### Performance Criteria

| Metric | Target |
|--------|--------|
| Standard attention vs CPU | Correct output, GPU speedup over CPU |
| FlashAttention memory | O(seq_len) not O(seq_len^2) (stretch goal) |
| Cold start (first kernel launch) | <100ms (PTX cached via OnceLock) |

### Publication Checklist

| # | Item | Status |
|---|------|--------|
| P1 | Crate name `kaio` claimed on crates.io | Done — v0.0.4 (all 5 crates published) |
| P2 | License: MIT OR Apache-2.0 (dual license) | Done |
| P3 | `Cargo.toml` metadata: description, repository, keywords, categories | Done (Sprint 4.8) |
| P4 | `CHANGELOG.md` covers all phases | Done through Phase 5 |
| P5 | `CONTRIBUTING.md` exists with dev setup instructions | Done |
| P6 | GitHub Actions CI: Windows + Linux matrix | Done (Sprint 5.6) |
| P7 | All `#![warn(missing_docs)]` passes | Done (all 5 crates) |
| P8 | r/rust post drafted and reviewed | Deferred (Sprint 5.8 not executed) |
| P9 | Blog post (optional but recommended) | Deferred |
| P10 | Benchmark comparison table in README | Done (Sprint 4.9) |

### Coverage Targets

| Crate | Minimum Line Coverage |
|-------|----------------------|
| `kaio-core` | 75% |
| `kaio-runtime` | 65% |
| `kaio-macros` | 70% |
| `kaio-ops` | 70% |
| **Workspace total** | **≥70%** |

---

## Phase 6: Tensor Cores & Async Copies

### Functional Criteria

| # | Criterion | Status | Validation Method |
|---|-----------|--------|-------------------|
| 6.1 | `PtxType::F16` / `PtxType::BF16` + packed-half2 register allocator | Done | Unit + GPU roundtrip tests |
| 6.2 | `mma.sync.m16n8k16` emits valid PTX, produces correct output on known-value fragments | Done | Standalone gate test (bit-exact vs hand-computed expected D) |
| 6.3 | `cp.async.ca.shared.global` emits valid PTX, SM 8.0+ gate enforced | Done | ptxas_verify + `PtxModule::validate()` rejection tests |
| 6.4 | Tensor-core matmul produces correct output vs CPU reference | Done | `matmul_tc_api` test suite (16×8×16 through 1023×1023×1024) |
| 6.5 | Auto-tuner dispatches to tensor-core variant when eligible | Done | 3-way dispatch tests (scalar / TC / TC+async) |
| 6.6 | TC attention + causal mask produces output within tolerance vs CPU reference | Done | Attention GPU tests |
| 6.7 | Tensor-core matmul reaches ≥60% of cuBLAS sgemm at 4096² (stretch: 70%) | Done, past both | Multi-warp: 79.9% sync / 85.1% async |
| 6.7b | Bank-conflict padding + D10 hoist lifts perf to ≥87% sync / ≥86% async | Partial | async 92.5% ✅ (past 90% stretch); sync 82.3% (LDG.128 reverted, banked as future-anchor IR) |
| 6.8 | Three showcase examples build from a fresh clone | Done | Standalone `cargo run` under `examples/` |
| 6.9 | All 5 crates published at v0.2.0 | Done | crates.io: kaio, kaio-core, kaio-macros, kaio-ops, kaio-runtime |
| 6.10 | Host-level codegen regression tests catch macro invariants without a GPU | Done | 4 inline tests in `kaio-macros::codegen::mod.rs` + 1 activated SM-threading canary |

### Performance Criteria

| Metric | Target | Actual Result | Notes |
|--------|--------|---------------|-------|
| TC matmul (4096², fp16 × fp16 → fp32, sync) | ≥60% of cuBLAS sgemm | 82.3% | Well past original 60% target |
| TC matmul (4096², fp16 × fp16 → fp32, async) | ≥60% of cuBLAS sgemm | 92.5% | Past 90% stretch — single-kernel reproducible ceiling |

See `docs/performance.md` for the full matrix and the apples-to-apples
disclaimer (KAIO is fp16 × fp16 → fp32 accumulation; cuBLAS reference
is sgemm — the project's existing benchmark path, not an equivalent-
precision comparison).

### Coverage Targets

| Crate | Minimum Line Coverage |
|-------|----------------------|
| `kaio-core` | 75% |
| `kaio-runtime` | 65% |
| `kaio-macros` | 70% |
| `kaio-ops` | 70% |
| **Workspace total** | **≥70%** |

---

## Phase 7: Quantized Kernels & Training Integration

### Functional Criteria (planned; fills in as sprints land)

| # | Criterion | Status | Validation Method |
|---|-----------|--------|-------------------|
| 7.0 | `#[gpu_kernel]` supports bitwise operators with preserved signed/unsigned `Shr` | Done | Unit + macro codegen + GPU round-trip with `i32 -2 >> 1 == -1` and `u32 0xFFFFFFFF >> 1 == 0x7FFFFFFF` assertions |
| 7.0 | Short-circuit `&&` / `||` with Rust-faithful semantics | Done | `logical_and_bounds_guard_no_oob` GPU signature canary + nested cases |
| 7.1 | `kaio_ops::matmul_int8` produces bit-exact output vs CPU reference | Planned | — |
| 7.2 | `kaio_ops::matmul_int4` produces output within fp16 tolerance vs CPU reference | Planned | — |
| 7.3 | Quantized attention (QKV projection quantized) produces output within LLM-inference tolerance | Planned | — |
| 7.4 | `kaio-candle` crate publishes with `CustomOp` bindings for core operations | Planned | — |

### Performance Criteria

To be defined at each sprint's planning stage. Quant performance is
scheme-dependent (INT8 vs INT4, per-tensor vs per-tile scale) and
hardware-dependent — target benchmarks set in sprint plans, not the
phase-level criteria document.

### Coverage Targets

Inherits Phase 6 targets. Phase 7 kernels added to `kaio-ops` must
bring their own test coverage; the workspace total gate stays at
≥70%.

---

## Summary: Coverage Progression

| Phase | Workspace Target | Rationale |
|-------|-----------------|-----------|
| Phase 1 | 60% | Foundation — establishing test patterns |
| Phase 2 | 60% | Macro code is harder to cover, maintaining baseline |
| Phase 3 | 65% | Mature enough to push coverage up |
| Phase 4 | 65% | Complex algorithms, maintaining target |
| Phase 5 | 70% | Release-quality coverage for public crate |
| Phase 6 | 70% | Sustained release-quality coverage — tensor-core primitives |
| Phase 7 | 70% | Sustained release-quality coverage — quantized kernels |

## Summary: Clippy & Fmt Requirements

- `cargo fmt` — enforced before every commit via pre-commit hook
- `cargo clippy --workspace -- -D warnings` — zero warnings, no exceptions
- These are non-negotiable across all phases
- Both must pass before marking any sprint as complete
