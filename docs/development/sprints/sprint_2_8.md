# Sprint 2.8 — End-to-End Kernel Tests

**Status:** Complete
**Commit:** `8012f3a`

## Context

The final sprint of Phase 2. Four kernels through the `#[gpu_kernel]` macro,
compiled to PTX, loaded into CUDA, launched on RTX 4090, results verified
against CPU reference.

## Bugs Found and Fixed

### 1. cvta.to.global register scope across if/else branches

**Bug:** When the first access to a pointer param was inside a conditional
branch (e.g., `if idx < n { out[idx] = ... }`), the `cvta.to.global`
register was only defined in one branch. The other branch referenced an
uninitialized register → CUDA_ERROR_LAUNCH_FAILED.

**Root cause:** Lazy cvta emission in `lower/memory.rs` cached the result
in `ctx.global_addrs`, but the first emission happened inside a branch.

**Fix:** Move cvta eagerly into `lower/params.rs` — emit `cvta.to.global`
for ALL pointer params during param loading, before any control flow. This
matches nvcc behavior. Pre-populate `ctx.global_addrs` in param lowering.

**Affected kernels:** fused_relu (nested if/else), fused_gelu (if with
complex body).

### 2. PTX float division requires rounding modifier

**Bug:** `div.f32` is invalid PTX — float division requires `.approx`,
`.full`, or a rounding modifier (`.rn`, `.rz`, `.rm`, `.rp`). Integer
division is fine without a modifier.

**Root cause:** `ArithOp::Div` emit used `format!("div{}", ty.ptx_suffix())`
for all types, but PTX ISA mandates a modifier for float division.

**Fix:** Type-aware emit: `div.approx.f32` for f32 (fast-math),
`div.rn.f64` for f64 (round-to-nearest), plain `div{ty}` for integers.

**Affected kernels:** fused_gelu (uses tanh → exp → div in synthesis).

## Kernels Tested

| Kernel | What it exercises | Status |
|---|---|---|
| `vector_add` (3 + 10K elements) | Basic arithmetic + indexing | Pass |
| `saxpy` (1024 elements) | Scalar * array + array | Pass |
| `fused_relu` (1024 elements) | Nested if/else + float comparison | Pass |
| `fused_gelu` (1024 elements) | Complex arithmetic + tanh synthesis | Pass (max abs error < 1e-4) |

## Results

**Files created:** 4
- `pyros/tests/vector_add_macro.rs`
- `pyros/tests/saxpy_macro.rs`
- `pyros/tests/fused_relu_macro.rs`
- `pyros/tests/fused_gelu_macro.rs`

**Files modified:** 2
- `pyros-macros/src/lower/params.rs` (eager cvta emission for pointer params)
- `pyros-core/src/instr/arith.rs` (float div.approx/div.rn modifier)

**Tests:** 168 total host + 5 GPU, all passing.
- Host: 163 (70 pyros-core + 90 pyros-macros + 1 compile_fail + 2 integration)
- GPU: 5 (vector_add_small, vector_add_large, saxpy, fused_relu, fused_gelu)

**Quality gates:** `cargo fmt --check` clean, `cargo clippy -- -D warnings`
zero warnings, all host + GPU tests pass.
