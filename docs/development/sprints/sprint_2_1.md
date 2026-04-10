# Sprint 2.1 — Macro Skeleton + Function Parsing

**Status:** Complete
**Commit:** `b3d254f`

## Context

First sprint of Phase 2. Establishes the `pyros-macros` crate, the parsing
layer, and the kernel IR type system. After this sprint, `#[gpu_kernel]`
compiles (with a placeholder expansion) and all signature/attribute validation
works.

## Decisions

### syn features — `full` + `extra-traits`

**Context:** syn 2 has feature flags controlling which AST types are available.
We need `ItemFn`, `Type`, `Pat`, `Expr` (for later sprints), and `Debug`
impls on AST types for development.

**Decision:** Enable `full` (all AST types) and `extra-traits` (Debug, Eq,
Hash impls). The `full` feature adds ~2s to proc-macro compile time but
avoids incremental feature additions across sprints. `extra-traits` aids
debugging during development — can be removed before release if compile
time matters.

### KernelType — mirror Rust types, not PTX types

**Context:** Should the macro's internal type enum mirror Rust types
(`i32`, `u32`) or PTX types (`S32`, `U32`)?

**Decision:** Mirror Rust types (`KernelType::I32`, `KernelType::U32`).
The mapping to PTX types (`PtxType::S32`, `PtxType::U32`) happens during
lowering (Sprint 2.2+). This keeps the parsing layer focused on Rust
semantics and the lowering layer focused on PTX semantics. The
`ptx_type_token()` method provides the bridge.

### Slice types — dedicated variants vs generic Reference

**Context:** Should `&[f32]` be `KernelType::SliceRef(F32)` or
`KernelType::Ref(Slice(F32))`?

**Decision:** Dedicated `SliceRef`/`SliceMutRef` variants. GPU kernels
only accept slice references as pointer parameters — bare references
(`&f32`), mutable references (`&mut f32`), and non-slice references
are all rejected. Having dedicated variants makes validation trivial
(pattern match vs nested checks) and documents the intent clearly.

### Attribute parsing — syn's `Parse` trait

**Context:** How to parse `#[gpu_kernel(block_size = 256)]`?

**Decision:** Implement `syn::parse::Parse` for a private `GpuKernelAttrs`
struct. This uses syn's standard parsing infrastructure, handles commas
and whitespace correctly, and extends naturally when we add more attributes
later (e.g., `target = "sm_89"` in Phase 3+).

### Error style — messages match success-criteria.md CF1-CF10

**Context:** What error messages should parsing produce?

**Decision:** Error messages are written to match the exact wording in
`docs/success-criteria.md` compile-fail cases CF1-CF10 as closely as
possible. This ensures Sprint 2.7's trybuild tests pass without message
adjustments. Every `syn::Error::new_spanned()` call carries the originating
token's span for accurate source location.

### Placeholder expansion — empty module

**Context:** What should the macro expand to before codegen is implemented?

**Decision:** Return an empty `mod kernel_name {}`. This means
`#[gpu_kernel(block_size = 256)] fn vector_add(...)` compiles but the
module has no `launch()` function yet. Code that calls
`vector_add::launch()` won't compile, which is correct — that comes
in Sprint 2.6.

### dead_code allowances — targeted, with sprint references

**Context:** KernelParam fields, KernelConfig fields, and KernelType
methods are defined now but not used until Sprint 2.2+. Clippy's
`-D warnings` gate rejects these.

**Decision:** Add `#[allow(dead_code)]` with comments referencing which
sprint will use them (e.g., `// Fields used in Sprint 2.6 codegen`).
This passes clippy, documents intent, and the allowances can be removed
as each sprint activates the code. Matches Phase 1's approach of building
ahead for the next sprint's needs.

## Scope

**In:** Cargo.toml, lib.rs entry point, parse/attrs.rs, parse/signature.rs,
kernel_ir/types.rs, workspace + umbrella wiring, 23 unit tests, zero
clippy warnings.

**Out:** Body parsing, expression/statement types, lowering, codegen,
validation, prelude, gpu_builtins stubs.

## Results

Completed as planned.

**Files created:** 7
- `pyros-macros/Cargo.toml`
- `pyros-macros/src/lib.rs`
- `pyros-macros/src/parse/mod.rs`
- `pyros-macros/src/parse/attrs.rs`
- `pyros-macros/src/parse/signature.rs`
- `pyros-macros/src/kernel_ir/mod.rs`
- `pyros-macros/src/kernel_ir/types.rs`

**Files modified:** 3
- `Cargo.toml` (workspace members)
- `pyros/Cargo.toml` (pyros-macros dependency)
- `pyros/src/lib.rs` (re-export gpu_kernel)

**Tests:** 76 total (53 existing Phase 1 + 23 new pyros-macros), all passing.
- 8 attribute parsing tests (valid configs, rejection of invalid/missing)
- 11 signature parsing tests (vector_add, all scalar types, f64 slices,
  rejection of return types/generics/unsupported types/lifetimes/self/
  async/unsafe/non-slice references)
- 4 kernel type tests (size_bytes, ptx_type_token, slice properties,
  display names)

**Quality gates:** `cargo fmt --check` clean, `cargo clippy -- -D warnings`
zero warnings, all tests pass.
