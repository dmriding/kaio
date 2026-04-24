# Sprint 8.0 — Pointer syntax (RFC-0001)

**Status:** ✅ Complete (v0.4.1, 2026-04-24)
**Branch:** `phase8` (merged to `main` via PR #15, commit `e8b7ae6`)
**Closes:** [#13](https://github.com/dmriding/kaio/issues/13)

---

## Context

Issue #13 was raised by the rust-cuda author and flagged a DSL-vs-compiled-Rust communication gap: `#[gpu_kernel]` kernels used `&mut [T]` / `&[T]` in parameter signatures, but the DSL lowers directly to PTX — no `noalias` is ever emitted — so the aliasing contract Rust readers project onto `&mut [T]` does not structurally apply. Kernel correctness depends on disjoint-access patterns, not on the type system.

No actual UB was introduced (confirmed in the issue thread), but the syntax was a credibility mismatch for a framework that positions itself as "GPU kernels in Rust." RFC-0001 resolved the concern by accepting `*mut [T]` / `*const [T]` as the primary kernel-parameter syntax — standard Rust syntax that correctly signals "device pointer, no aliasing contract." `&[T]` / `&mut [T]` were retained as permanent sugar (identical PTX lowering).

This closed the issue with a rust-cuda author watching and absorbed the originally-scheduled Phase 8.5 into Sprint 8.0 so the syntax lands before PyO3 widens KAIO's user-facing surface in 8.1+.

## What shipped

### Parser

- New `Type::Ptr` match arm in `kaio-macros/src/parse/signature.rs`, parallel to the existing `Type::Reference` arm. `type_ptr.mutability.is_some()` → `KernelType::SliceMutRef`, else `KernelType::SliceRef`.
- Shared `parse_slice_elem_type` helper used by **both** the Ptr and Reference arms, producing symmetric diagnostics across the two forms.
- Dedicated error messages for accidental-user cases: fixed-size arrays (`*mut [f32; 256]`, `&mut [f32; 256]`), pointer/reference-to-scalar (`*mut f32`, `&mut f32`), nested pointers/references (`*const *mut [f32]`, `&mut &[f32]`). Previously some of these fell through to the generic catch-all.
- Error-message enumerations updated to list all four supported slice forms:
  `&[T], &mut [T], *const [T], *mut [T]`.

### Tests

- **10 new parser unit tests** in `kaio-macros/src/parse/signature.rs`: 6 pointer-form acceptance/rejection tests, 3 reference-symmetry diagnostics, and 1 updated existing test for the new wording.
- **1 macro-level smoke test** in `kaio/tests/macro_pointer_smoke.rs` — GPU-ignored, mixed-form kernel (`&[T]`, `*const [T]`, `&mut [T]`, `*mut [T]` all in one signature), asserts end-to-end correctness across parser → codegen → launch wrapper → GPU execution.
- Existing parser tests and the full GPU test suite (41 `--ignored` tests) pass unchanged — zero behaviour change for valid reference-form signatures.
- Trybuild compile-fail fixtures updated in the same commit as the parser wording change.

### Examples

All 6 showcase example crates with direct `#[gpu_kernel]` use migrated to pointer-form as the primary signature (7 kernel functions total):

- `examples/fused_silu_gate` — `fused_silu_gate`
- `examples/gelu_comparison` — `gelu_exact`, `gelu_fast`
- `examples/int8_dequant` — `dequant_i8`
- `examples/layer_norm` — `layer_norm`
- `examples/rms_norm` — `rms_norm`
- `examples/softmax` — `softmax`

Each example's README kernel snippet was migrated in the same commit as its `src/main.rs`. Host-side launch invocations (`&GpuBuffer<T>` / `&mut GpuBuffer<T>`) are unchanged — the macro's launch-wrapper codegen maps both `SliceRef` and `SliceMutRef` variants identically regardless of source-form.

### Documentation

- Root `README.md`: Quick Start, "Patterns → bounds-checked element-wise", "Patterns → shared memory tiling", and the "DSL is a Rust subset" paragraph all lead with pointer form. RFC-0001 reference rewritten from "planned" to "shipped."
- `kaio/README.md`: saxpy tutorial snippet migrated.
- `kaio/src/prelude.rs`: prelude-level doctest migrated.
- `kaio-macros/src/lib.rs`: `#[gpu_kernel]` rustdoc gained explicit primary-vs-sugar framing in the first paragraph of the signature-syntax section, with the RFC-0001 link.
- `docs/performance.md`: pedagogy snippets (coalescing, shared-memory padding) migrated.
- `docs/rfcs/rfc-0001-pointer-syntax.md`: status `Draft` → `Implemented`, migration path rewritten as implementation timeline, implementation-notes section added.
- `CHANGELOG.md`: `[Unreleased] > Added` entry promoted to `[0.4.1] — 2026-04-24 — Sprint 8.0: Pointer Syntax`.
- `docs/phases.md`: Phase 8.5 marked absorbed-as-Sprint-8.0 ✅, Phase 9 dependency updated, CUDA graph capture added as a roadmap candidate.

### Release

v0.4.1 across the six publishable crates:

- `kaio` 0.4.0 → 0.4.1
- `kaio-core` 0.4.0 → 0.4.1
- `kaio-macros` 0.4.0 → 0.4.1
- `kaio-ops` 0.4.0 → 0.4.1
- `kaio-runtime` 0.4.0 → 0.4.1
- `kaio-candle` 0.1.0 → 0.1.1 (standalone, not a workspace member)

All 9 example `Cargo.lock` files refreshed. Coverage badge updated from 93.65% to 88.39% (measured-lines grew faster than covered as new parser paths and compile-fail fixtures landed).

### README positioning

Outside the strict RFC-0001 scope, the 0.4.1 cut refreshed the project-level framing:

- Sharpened one-line opener; elevated "No CUDA toolkit required."
- Added Triton to the comparison table (Windows + Linux story).
- Added a "What this is not" section (no autograd, no multi-GPU, not cuBLAS-level, API may change).
- Worst-of-10 bench framing introduced with full distribution tables (TC sync/async + INT8 + INT4) — report the worst observed median as the floor, not the best or median, with methodology disclosed.

## What didn't change

- **PTX IR** (`kaio-core`) — zero touch.
- **Codegen** (`kaio-macros/src/lower/`, `kaio-macros/src/codegen/`) — zero touch. Pointer and reference forms share the same `KernelType::SliceRef` / `SliceMutRef` variants; everything below the parser is oblivious.
- **Runtime** (`kaio-runtime`) — zero touch.
- **Host-side launch wrapper** — `SliceRef` → `&GpuBuffer<T>`, `SliceMutRef` → `&mut GpuBuffer<T>` in both cases.
- **`&[T]` / `&mut [T]` reference forms** — continue to work as permanent sugar; no deprecation planned.
- **kaio-ops internal kernels** — out-of-scope migration; internal performance primitives are not user-facing tutorials. Will be addressed when those files are touched for other reasons in Phase 9.

## Known limitations

- Pointer forms are accepted as parameter syntax only; there is no pointer arithmetic surface inside kernel bodies (same as the reference form, which only supports indexed access via `a[idx]`).
- The RFC-0001 "primary form" framing is a documentation-level convention. The parser treats both forms identically; choosing between them is a readability call, not a correctness one.

## Follow-ups

- **Sprint 8.0.5** — bench coverage extension for the attention + QKV + norm/activation kernel families under the unified `cargo xtask bench` harness. Closes the "KAIO benches every kernel it ships" gap flagged in `performance.md` §Bench coverage today + roadmap.
- **Phase 8.1+** — PyO3 scaffold and bindings per `docs/phases.md` §Phase 8.
- **Phase 9** — FlashAttention backward + kernel deepening (bf16 TC, `ldmatrix.sync`, additional `mma` shapes).
