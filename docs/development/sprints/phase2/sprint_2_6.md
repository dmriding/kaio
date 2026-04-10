# Sprint 2.6 — Launch Wrapper + Full Pipeline

**Status:** Complete
**Commit:** `a89eb2b`

## Context

Sprints 2.1-2.5 built all the lowering pieces. Sprint 2.6 connects them:
the `#[gpu_kernel]` macro now produces a working
`mod kernel_name { fn build_ptx(), pub fn launch() }`.

This is the second high-risk sprint — everything comes together here.

## Decisions

### PushKernelArg re-export through kaio-runtime

**Context:** cudarc's `.arg()` method is a trait method on `PushKernelArg`.
The trait must be in scope for the generated launch code to compile.

**Decision:** Add `pub use cudarc::driver::PushKernelArg;` to
`kaio-runtime/src/lib.rs`. The generated module imports it via
`use kaio::runtime::PushKernelArg;`. Users never need to depend on
cudarc directly.

### Grid config: last u32 param convention

**Context:** `LaunchConfig::for_num_elems(n)` needs a u32 element count.
Which parameter is the count?

**Decision:** Use the last scalar `u32` parameter. This works for all four
Sprint 2.8 kernels which follow the `fn kernel(slices..., n: u32)` pattern.

**Known limitation (Phase 2):** For kernels with multiple u32 params
(e.g., `rows: u32, cols: u32`), the last one is used, which may not be
correct. Custom grid config support deferred to Phase 3.

### Import paths: kaio::core:: and kaio::runtime::

**Context:** The generated code uses `kaio::core::ir::*` etc. The umbrella
crate does `pub use kaio_core as core; pub use kaio_runtime as runtime;`.

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
- User-facing: `kaio/src/prelude.rs`, `kaio/src/gpu_builtins.rs`
- PushKernelArg re-export from kaio-runtime
- Replace placeholder in lib.rs with real pipeline

**Out:** Type validation (Sprint 2.7), E2E GPU tests (Sprint 2.8).

## Results

Completed as planned.

**Files created:** 7
- `kaio-macros/src/codegen/mod.rs`
- `kaio-macros/src/codegen/ptx_builder.rs`
- `kaio-macros/src/codegen/launch_wrapper.rs`
- `kaio-macros/src/lower/params.rs`
- `kaio-macros/src/lower/cast.rs`
- `kaio/src/gpu_builtins.rs`
- `kaio/src/prelude.rs`

**Files modified:** 4
- `kaio-macros/src/lib.rs` (full pipeline replacing placeholder)
- `kaio-macros/src/lower/mod.rs` (cast + params modules, Cast in lower_expr)
- `kaio/src/lib.rs` (prelude + gpu_builtins modules)
- `kaio-runtime/src/lib.rs` (PushKernelArg re-export)

**Tests:** 162 total (72 kaio-core + 90 kaio-macros), all passing.
- kaio-macros: +5 (3 param lowering, 2 cast lowering)

**Quality gates:** `cargo fmt --check` clean, `cargo clippy -- -D warnings`
zero warnings, all tests pass.
