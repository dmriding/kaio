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
| 5.2 | Standard attention produces correct output | Pending | Compare to CPU reference (< 1e-3 abs error) |
| 5.3 | Causal masking works correctly | Pending | Verify masked positions produce correct output |
| 5.4 | FlashAttention uses O(n) memory (stretch goal) | Pending | Memory measurement vs standard attention |
| 5.5 | Auto-tuner selects optimal block/tile size | Pending | Grid search produces correct output |
| 5.6 | CI runs on Windows + Linux | Pending | GitHub Actions matrix |
| 5.7 | `cargo publish --dry-run` succeeds for all crates | Pending | Dry-run all 5 crates |
| 5.8 | DSL friction points documented for Phase 6 | Pending | Friction report from attention implementation |

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
| P4 | `CHANGELOG.md` covers all phases | Done through Phase 4 |
| P5 | `CONTRIBUTING.md` exists with dev setup instructions | Done |
| P6 | GitHub Actions CI: Windows + Linux matrix | Pending (Sprint 5.6) |
| P7 | All `#![warn(missing_docs)]` passes | Done (all 5 crates) |
| P8 | r/rust post drafted and reviewed | Pending (Sprint 5.8) |
| P9 | Blog post (optional but recommended) | Pending (Sprint 5.8) |
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

## Summary: Coverage Progression

| Phase | Workspace Target | Rationale |
|-------|-----------------|-----------|
| Phase 1 | 60% | Foundation — establishing test patterns |
| Phase 2 | 60% | Macro code is harder to cover, maintaining baseline |
| Phase 3 | 65% | Mature enough to push coverage up |
| Phase 4 | 65% | Complex algorithms, maintaining target |
| Phase 5 | 70% | Release-quality coverage for public crate |

## Summary: Clippy & Fmt Requirements

- `cargo fmt` — enforced before every commit via pre-commit hook
- `cargo clippy --workspace -- -D warnings` — zero warnings, no exceptions
- These are non-negotiable across all phases
- Forge agents must run both before marking any sprint as complete
