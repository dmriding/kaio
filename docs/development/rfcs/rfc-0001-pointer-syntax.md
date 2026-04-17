# RFC-0001: Raw pointer syntax for kernel parameters

**Status:** Draft
**Date:** 2026-04-18
**Motivation:** [Issue #13](https://github.com/dmriding/kaio/issues/13) (juntyr)

## Summary

Accept `*mut [T]` and `*const [T]` as kernel parameter types in
`#[gpu_kernel]` signatures, alongside the existing `&mut [T]` and
`&[T]`. The raw pointer forms become the recommended primary syntax;
the reference forms are retained as permanent ergonomic sugar.

## Motivation

KAIO's `#[gpu_kernel]` macro uses Rust syntax but is a DSL — the
kernel body is parsed into KAIO's own IR and lowered directly to PTX.
It never reaches rustc's backend, so Rust's aliasing guarantees
(`&mut T` implies `noalias`) do not structurally apply.

This creates a communication gap: a Rust developer reading
`out: &mut [f32]` in a kernel signature will project Rust's uniqueness
guarantee onto it, but thousands of GPU threads access the same buffer
concurrently. Correctness depends on the kernel author writing disjoint
access patterns, not on compiler enforcement.

Using `*mut [T]` communicates the reality: this is a device pointer
with no aliasing contract. The kernel author is responsible for access
safety, just as they would be with `*mut T` in host Rust. This aligns
KAIO's DSL syntax with the mental model GPU programmers already have —
device pointers are shared across threads, and correctness comes from
the access pattern, not from the type system. Rust's `*mut` is the
standard way to express "pointer without aliasing guarantees," which
is exactly what a GPU kernel parameter is.

This concern was raised by @juntyr in issue #13, who noted that while
there is no actual UB (the DSL bypass means no `noalias` is emitted),
the `&mut [T]` syntax is a credibility mismatch for a project that
positions itself as "GPU kernels in Rust."

## Design

### Accepted parameter forms

| Syntax | Meaning | PTX lowering |
|---|---|---|
| `*mut [T]` | Read-write device buffer | `.param .u64` (unchanged) |
| `*const [T]` | Read-only device buffer | `.param .u64` (unchanged) |
| `&mut [T]` | Ergonomic sugar for `*mut [T]` | `.param .u64` (unchanged) |
| `&[T]` | Ergonomic sugar for `*const [T]` | `.param .u64` (unchanged) |
| `T` (scalar) | Pass-by-value scalar | `.param .{u32,f32,...}` (unchanged) |

The PTX output is identical regardless of which form is used. The
distinction is entirely at the Rust syntax level — a documentation and
intent signal, not a codegen difference.

### Migration path

1. **v0.4.0 (this release):** Document the DSL-vs-compiled-Rust
   distinction in README, `#[gpu_kernel]` rustdoc, and this RFC.
   No syntax changes.

2. **Future minor release:** Accept `*mut [T]` / `*const [T]` in
   kernel signatures. Update all examples and docs to use pointer
   syntax as the primary form. `&mut` / `&` continue to work — no
   deprecation, permanent sugar.

### Host-side launch wrapper

The generated launch wrapper currently takes `&GpuBuffer<T>` (for
`&[T]` params) and `&mut GpuBuffer<T>` (for `&mut [T]` params). For
pointer-syntax kernel params, the launch wrapper accepts the same
host types — the wrapper is host Rust where `&mut` aliasing guarantees
DO apply (one Rust-level `&mut GpuBuffer` holds the unique reference).

The kernel-side `*mut` signals "no aliasing contract on-device." The
host-side `&mut GpuBuffer` signals "unique Rust-level ownership of the
allocation." These are complementary, not contradictory.

### Implementation complexity

Parser change is ~20 lines in `parse_kernel_signature` — a new
`Type::Ptr` match arm alongside the existing `Type::Reference` arm,
with identical lowering to `.param .u64`. This is a small parser
change, not a compiler rewrite.

### Alternative considered: `&aliased mut [T]`

juntyr suggested a custom `&aliased mut [T]` syntax that only works in
kernel signatures. This would require a custom token parser (not
standard `syn`) and introduce a KAIO-specific keyword that doesn't
exist in Rust. Rejected in favor of `*mut [T]` which is standard Rust
syntax with well-understood semantics.

### Alternative considered: per-thread sub-slice rewriting

rust-cuda (juntyr's project) takes the approach of structurally
splitting `&mut [T]` into per-thread sub-slices at macro expansion
time, so the aliasing contract actually holds per thread. This works
for disjoint per-thread access patterns but doesn't compose with
reductions, atomics, scatter writes, or shared memory — all of which
KAIO kernels use routinely. Not viable as the primary API design.

## Resolved questions

1. **Host-side wrapper type:** `&mut GpuBuffer<T>` on the host side,
   unchanged. The host wrapper is real Rust where `&mut` aliasing
   guarantees apply — one `&mut GpuBuffer` holds the unique reference
   to the allocation. The kernel-side `*mut` and host-side `&mut` are
   complementary: different contracts at different levels of the stack.

2. **Reference form deprecation:** Permanent silent sugar, no
   deprecation planned. The reference forms (`&mut [T]`, `&[T]`)
   produce identical PTX output. Deprecating them would create churn
   for every existing kernel without any correctness or performance
   benefit. Documentation and examples will use pointer forms as
   primary; reference forms remain accepted syntax.

3. **Doc-alias for `*mut [T]`:** Not needed. `*mut [T]` is standard
   Rust syntax that every developer who has written `unsafe` code
   understands. The `#[gpu_kernel]` rustdoc explains the DSL context.

## References

- [Issue #13](https://github.com/dmriding/kaio/issues/13) — original report
- [rust-cuda](https://github.com/juntyr/rust-cuda) — juntyr's prior art with per-thread sub-slicing
