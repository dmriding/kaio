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
| **CI** | Automated test runs on both platforms (from Phase 5, manual before that) |

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
| CF10 | Kernel uses `loop` | "`loop` is not supported — use `for` or `while` (available in Phase 3)" |

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

### Functional Criteria

| # | Criterion | Validation Method |
|---|-----------|-------------------|
| 4.1 | `block_load` correctly loads contiguous data from global to shared memory | Integration test: load, barrier, verify in shared |
| 4.2 | `block_store` correctly writes from shared/registers to global memory | Integration test: compute in shared, store, verify on host |
| 4.3 | Tiled `matmul` produces correct results for square matrices | Integration test: A×B = C, compare to CPU reference |
| 4.4 | Tiled `matmul` produces correct results for non-square matrices | Integration test: (M×K) × (K×N) for various M, K, N |
| 4.5 | Tiled `matmul` handles non-tile-aligned dimensions | Integration test: M, K, N not divisible by tile size |
| 4.6 | Performance within 60% of cuBLAS `sgemm` for 2048×2048 f32 | Benchmark: `kaio_matmul` vs `cublas_sgemm` |
| 4.7 | Memory coalescing warnings emitted for known-bad patterns | Compile-time diagnostic test |
| 4.8 | No shared memory bank conflicts in generated matmul PTX | `ncu` profiling or compile-time analysis |

### Performance Criteria

| Metric | Target | How to Measure |
|--------|--------|---------------|
| Matmul TFLOPS (2048×2048, f32) | ≥60% of cuBLAS | `benches/matmul_benchmark.rs` |
| Matmul TFLOPS (4096×4096, f32) | ≥60% of cuBLAS | `benches/matmul_benchmark.rs` |
| Memory throughput (block_load) | ≥70% of theoretical bandwidth | `ncu` profiling |

These are v0.1 targets. Performance optimization continues post-release.

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

| # | Criterion | Validation Method |
|---|-----------|-------------------|
| 5.1 | Fused attention produces correct output for standard multi-head attention | Integration test: compare to PyTorch `F.scaled_dot_product_attention` |
| 5.2 | Causal masking works correctly | Integration test: verify masked positions are -inf before softmax |
| 5.3 | Auto-tuner selects a valid configuration | Integration test: tuner runs without crash, selected config produces correct output |
| 5.4 | All tutorial examples compile and run without errors | CI: `cargo test --examples` |
| 5.5 | `cargo doc` generates complete documentation with no broken links | CI: `cargo doc --no-deps` + link checker |
| 5.6 | `cargo publish --dry-run` succeeds | CI: dry-run publish |
| 5.7 | README includes: overview, install instructions, minimal example, benchmark results | Manual review |
| 5.8 | All examples from README actually compile and run | CI: extract and test README code blocks |

### Performance Criteria

| Metric | Target |
|--------|--------|
| Fused attention vs unfused PyTorch (forward only) | ≥1.5x speedup |
| Fused attention vs cuDNN attention | Track but no minimum (aspirational) |
| Cold start (first kernel launch) | <100ms (no JIT compilation, PTX is precompiled) |

### Publication Checklist

| # | Item | Status |
|---|------|--------|
| P1 | Crate name `kaio` claimed on crates.io | ☐ |
| P2 | License: MIT OR Apache-2.0 (dual license) | ☐ |
| P3 | `Cargo.toml` metadata: description, repository, keywords, categories | ☐ |
| P4 | `CHANGELOG.md` covers all phases | ☐ |
| P5 | `CONTRIBUTING.md` exists with dev setup instructions | ☐ |
| P6 | GitHub Actions CI: Windows + Linux matrix | ☐ |
| P7 | All `#![deny(missing_docs)]` passes | ☐ |
| P8 | r/rust post drafted and reviewed | ☐ |
| P9 | Blog post (optional but recommended) | ☐ |
| P10 | Benchmark comparison table in README | ☐ |

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
