# Sprint 2.6 — Launch Wrapper + Full Pipeline

**Status:** Complete
**Commit:** (pending)

## Context

Sprints 2.1-2.5 built all the lowering pieces. Sprint 2.6 connects them:
the `#[gpu_kernel]` macro now produces a working
`mod kernel_name { fn build_ptx(), pub fn launch() }`.

This is the second high-risk sprint — everything comes together here.

## Decisions

### PushKernelArg re-export through pyros-runtime

**Context:** cudarc's `.arg()` method is a trait method on `PushKernelArg`.
The trait must be in scope for the generated launch code to compile.

**Decision:** Add `pub use cudarc::driver::PushKernelArg;` to
`pyros-runtime/src/lib.rs`. The generated module imports it via
`use pyros::runtime::PushKernelArg;`. Users never need to depend on
cudarc directly.

### Grid config: last u32 param convention

**Context:** `LaunchConfig::for_num_elems(n)` needs a u32 element count.
Which parameter is the count?

**Decision:** Use the last scalar `u32` parameter. This works for all four
Sprint 2.8 kernels which follow the `fn kernel(slices..., n: u32)` pattern.

**Known limitation (Phase 2):** For kernels with multiple u32 params
(e.g., `rows: u32, cols: u32`), the last one is used, which may not be
correct. Custom grid config support deferred to Phase 3.

### Import paths: pyros::core:: and pyros::runtime::

**Context:** The generated code uses `pyros::core::ir::*` etc. The umbrella
crate does `pub use pyros_core as core; pub use pyros_runtime as runtime;`.

**Verification:** `cargo check` confirms these paths resolve correctly in
the generated module. The `pub use ... as` re-export creates proper module
paths for downstream crates.

### IDE stub functions in gpu_builtins.rs

**Decision:** Real Rust functions that panic at runtime, replaced by the
macro at compile time. Provides rust-analyzer autocomplete + docs + type
checking. Generic stubs for `abs<T>`, `min<T>`, `max<T>`; concrete `f32`
stubs for transcendentals.

## Scope

**In:**
- Full codegen pipeline: `codegen/mod.rs`, `codegen/ptx_builder.rs`,
  `codegen/launch_wrapper.rs`
- Parameter lowering: `lower/params.rs`
- Cast lowering: `lower/cast.rs`
- User-facing: `pyros/src/prelude.rs`, `pyros/src/gpu_builtins.rs`
- PushKernelArg re-export from pyros-runtime
- Replace placeholder in lib.rs with real pipeline

**Out:** Type validation (Sprint 2.7), E2E GPU tests (Sprint 2.8).

## Results

Completed as planned.

**Files created:** 7
- `pyros-macros/src/codegen/mod.rs`
- `pyros-macros/src/codegen/ptx_builder.rs`
- `pyros-macros/src/codegen/launch_wrapper.rs`
- `pyros-macros/src/lower/params.rs`
- `pyros-macros/src/lower/cast.rs`
- `pyros/src/gpu_builtins.rs`
- `pyros/src/prelude.rs`

**Files modified:** 4
- `pyros-macros/src/lib.rs` (full pipeline replacing placeholder)
- `pyros-macros/src/lower/mod.rs` (cast + params modules, Cast in lower_expr)
- `pyros/src/lib.rs` (prelude + gpu_builtins modules)
- `pyros-runtime/src/lib.rs` (PushKernelArg re-export)

**Tests:** 162 total (72 pyros-core + 90 pyros-macros), all passing.
- pyros-macros: +5 (3 param lowering, 2 cast lowering)

**Quality gates:** `cargo fmt --check` clean, `cargo clippy -- -D warnings`
zero warnings, all tests pass.
