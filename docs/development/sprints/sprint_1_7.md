# Sprint 1.7 — Runtime Launch + vector_add E2E

**Commit:** pending
**Status:** Complete

## Context

Sprint 1.7 is the Phase 1 success gate. It adds PTX module loading and
kernel launch, then runs vector_add on real GPU data. This proves the
core thesis: Rust IR → PTX text → GPU execution.

## Decisions

### Enable cudarc `nvrtc` feature — not what we expected

**Context:** Sprint 1.0 explicitly excluded the `nvrtc` feature because
"PYROS emits PTX, doesn't compile C kernels." We discovered that
`CudaContext::load_module()` and the `Ptx` type are gated behind
`#[cfg(feature = "nvrtc")]`.

**Analysis:** The `Ptx::from_src(text)` constructor and `load_module`
use only the CUDA driver API (`cuModuleLoadData`), not the NVRTC
compiler. The `nvrtc` feature just compiles the Rust module containing
these types. With `dynamic-loading`, the NVRTC shared library is only
dlopened if `compile_ptx()` is called — which we never do. Verified:
`cargo build --workspace` succeeds without NVRTC installed.

**Decision:** Enable `nvrtc`. Updated the Cargo.toml comment to explain
why and clarify we're not doing C kernel compilation.

### Skip LaunchPending abstraction

**Context:** The master plan specified `LaunchPending<'a>` wrapping
cudarc's `LaunchArgs`. After reading cudarc's actual API, this is
unnecessary abstraction.

**Analysis:** cudarc's `LaunchArgs` already IS a builder with `.arg()`
chaining and `.launch(cfg)`. Wrapping it adds a layer of indirection,
lifetime management, and maintenance burden — all for an API that
Phase 2's macro will replace entirely with typed safe wrappers.

**Decision:** Ship thin wrappers only (PyrosModule, PyrosFunction) and
use cudarc's launch builder directly in the E2E test. The unsafe launch
is explicit and visible. No premature abstraction.

### stream() visibility — pub(crate) vs pub

**Context:** `PyrosDevice::stream()` was `pub(crate)` from Sprint 1.6.
The E2E integration test (in `tests/`) is outside the crate boundary
and can't access `pub(crate)` methods.

**Decision:** Made `stream()` public. Users need stream access for
launch operations in Phase 1. The method is well-documented as "Phase 2
macro will hide this."

### PushKernelArg trait import

**Context:** cudarc's `LaunchArgs::arg()` method comes from the
`PushKernelArg` trait, not inherent methods. Without importing the
trait, `.arg()` isn't available.

**Decision:** Import `cudarc::driver::PushKernelArg` in the E2E test.
For Phase 2, the macro-generated wrappers will handle this internally
so users never need to import cudarc traits directly.

### PTX debug output on failure

**Context:** If the CUDA driver rejects the PTX or the kernel produces
wrong results, the error message alone may be cryptic. The PTX source is
the key diagnostic artifact.

**Decision:** `unwrap_or_else` handlers on `load_ptx` and `launch` print
the full PTX text via `eprintln!` before panicking. If something breaks
at 2am, the test output shows exactly what PTX was generated.

## Scope

**In:** cudarc `nvrtc` feature enablement, PyrosModule, PyrosFunction,
PyrosDevice::load_ptx, stream() made public, LaunchConfig re-export,
2 E2E GPU tests (small 3-element + large 10k-element).

**Out:** LaunchPending abstraction, typed launch wrappers, compute_grid_1d
helper, f64/i32/u32 dtype tests, ptxas verification, coverage.

## Results

**PHASE 1 SUCCESS GATE PASSED.**

`vector_add` — constructed entirely from Rust IR, emitted to PTX text,
loaded into the CUDA driver, launched on the RTX 4090 — produces correct
results for both 3 elements and 10,000 elements.

Two compile fixes during implementation:
1. `stream()` changed from `pub(crate)` to `pub` (integration test boundary)
2. `use cudarc::driver::PushKernelArg` added (trait import for `.arg()`)

**Quality gates:**
- `cargo build --workspace`: clean
- `cargo test --workspace`: 52 host tests pass
- `cargo test -p pyros-runtime -- --ignored`: 9 GPU tests pass (7 device + 2 E2E)
- `cargo fmt --all --check`: clean
- `cargo clippy --workspace --all-targets -- -D warnings`: clean

**Files created:** 2 (module.rs, vector_add_e2e.rs)
**Files modified:** 3 (Cargo.toml, device.rs, lib.rs)
