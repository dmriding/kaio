# Sprint 6.1 — fp16/bf16 Types + Registers + Conversions

**Status:** Done
**Branch:** phase6
**Commit:** `a1dd450`
**Goal:** Add the type-level groundwork for tensor-core work —
`PtxType::F16`/`BF16`, register kinds `%h`/`%hb`, cvt rounding for
float-to-float conversions, and host-side `GpuBuffer<f16>` support.
No compute yet; the compute instructions land in Sprint 6.2.

## Problem

Phase 5 shipped v0.1.0 with scalar fp32/fp64 kernels reaching 31% of
cuBLAS sgemm. The next performance step is tensor cores, which consume
fp16/bf16 inputs with fp32 accumulation. Before we could emit
`mma.sync`, we needed the primitive types threaded through the IR:
PtxType variants, register class prefixes, cvt instruction rounding
modes, and runtime buffer support.

## Fix

### Types (`kaio-core/src/types.rs`)

Added two variants to `PtxType`:

```rust
PtxType::F16  // .f16  — requires SM 5.3+
PtxType::BF16 // .bf16 — requires SM 8.0+
```

`BF16` is **type-level only** this sprint — no execution gating. The
auto-tuner in Sprint 6.5 will enforce SM requirements at dispatch time.

`size_bytes()`, `ptx_suffix()`, `reg_decl_type()`, and `reg_kind()`
cover both. `reg_decl_type()` keeps the typed form (`.f16`/`.bf16`)
rather than collapsing to `.b16`, matching nvcc.

### Registers (`kaio-core/src/types.rs`, `kaio-core/src/ir/register.rs`)

Added two `RegKind` variants:

```rust
RegKind::H   // %h  — f16 single value
RegKind::Hb  // %hb — bf16 single value
```

Independent counters — `%h0`..`%hN` and `%hb0`..`%hbN` don't collide
with each other or with `%r`/`%f`. Extended allocator counter array
from `[u32; 5]` to `[u32; 7]`.

These are the `.f16`/`.bf16` single-value register classes for scalar
arithmetic and cvt. Fragment registers used by `mma.sync` (Sprint 6.2)
are **b32-packed half2**, which live in `%r` — a separate concern.

### CVT Rounding (`kaio-core/src/emit/emit_trait.rs`)

Generalized the cvt rounding match to treat `F16` and `BF16` as floats
for modifier selection. Matrix:

| Direction | Modifier |
|---|---|
| int → float (any, incl. half) | `.rn` |
| float → int (any, incl. half) | `.rzi` (matches Rust `as` truncation) |
| float → float (incl. f16↔f32, f16↔bf16) | `.rn` |
| int → int or same-type | (none) |

KAIO emits `.rn` for **all** float-to-float conversions for consistency
and PTX validity, even where the conversion is exact (f16→f32). PTX
requires a rounding modifier on narrowing and cross-half conversions;
emitting it unconditionally avoids special-casing and matches nvcc.

### Host-side (`kaio-core/Cargo.toml`, `kaio-runtime/Cargo.toml`)

Added `half` crate (v2.x) to kaio-core — the first kaio-core dep.
Rationale: `half::f16`/`half::bf16` are the ecosystem standard;
cudarc already depends on `half`; rolling our own newtypes would
fight the ecosystem. Also enabled cudarc's `f16` feature so
`DeviceRepr` is implemented for `half::f16` / `half::bf16`,
making `GpuBuffer<f16>` and `GpuBuffer<bf16>` roundtrip with no
extra glue.

`GpuType` implementations added for `half::f16` and `half::bf16`,
giving the kernel macro and IR API a uniform type mapping.

## New Tests

11 host tests + 2 GPU tests (total: 219 host + 102 GPU).

### Host (`kaio-core`)

- `ptx_type_size_bytes` — F16/BF16 = 2 bytes
- `ptx_type_suffix` — `.f16`, `.bf16`
- `ptx_type_reg_decl_type` — typed half declarations
- `ptx_type_reg_kind` — half → H, bf16 → Hb
- `reg_kind_prefix` — `%h`, `%hb`
- `gpu_type_impls` — `half::f16`/`bf16` round-trip through `GpuType::PTX_TYPE`
- `f16_bf16_register_allocation` — independent counters
- `emit_ld_global_f16`, `emit_st_global_f16`, `emit_ld_shared_bf16` — load/store paths
- `emit_cvt_f32_to_f16`, `emit_cvt_f16_to_f32`, `emit_cvt_int_to_f16`,
  `emit_cvt_f16_to_int`, `emit_cvt_bf16_to_f32` — cvt rounding for
  half types
- `emit_reg_declarations_with_f16` — register declarations emit
  `.reg .f16 %h<N>;` / `.reg .bf16 %hb<N>;`
- `emit_kernel_f16_flow` — end-to-end: params → ld.global.f16 →
  cvt.rn.f32.f16 → add.f32 → cvt.rn.f16.f32 → st.global.f16

### GPU (`kaio/tests/f16_buffer.rs`)

- `f16_buffer_roundtrip` — host → device → host bit-exact for `half::f16`
- `bf16_buffer_roundtrip` — same for `half::bf16`

Both `#[ignore]` (GPU required).

## Design Decisions

### half crate as kaio-core dep — accepted

kaio-core had zero dependencies before this sprint. Adding `half`
was non-trivial architecturally. Accepted because:

- `half::f16` / `half::bf16` are the de-facto ecosystem types
- cudarc already requires it for DeviceRepr on half types
- Rolling our own newtypes would force every downstream crate to
  convert at the boundary, killing ergonomics

### BF16 execution gating deferred

BF16 requires SM 8.0+ at execution time. This sprint provides only
type-level support — allocation and emission compile fine at any SM.
Actual execution-path gating (SM detection + dispatch) lands in
Sprint 6.5 (auto-tuner), consistent with the scalar/TC/TC+async
three-way dispatch model.

### cvt.rn emitted for exact float-to-float conversions

f16 → f32 is exact, but PTX requires a rounding modifier on cross-width
float conversions. Emitting `.rn` unconditionally for all float-to-float
cvt is consistent and matches nvcc. Zero-cost at the hardware level.

### RegKind models PTX register classes directly

`RegKind::H` and `RegKind::Hb` are distinct enum variants (not a
`Half(HalfKind)` shape) because they produce different PTX prefixes
(`%h` vs `%hb`) and declarations (`.f16` vs `.bf16`). Modelling the
PTX register class directly rather than a higher-level numeric family
keeps the allocator and emitter simple. Revisit in 6.2 if fragments
push the model to its limits.

## Files

| File | Change |
|------|--------|
| `kaio-core/Cargo.toml` | Add `half = "2"` |
| `kaio-core/src/types.rs` | PtxType::F16/BF16, RegKind::H/Hb, GpuType impls |
| `kaio-core/src/ir/register.rs` | Allocator counter array `[u32;5]` → `[u32;7]` |
| `kaio-core/src/ir/kernel.rs` | KernelStats: `registers_h`, `registers_hb` fields |
| `kaio-core/src/emit/emit_trait.rs` | Generalized cvt rounding for halves |
| `kaio-core/src/instr/memory.rs` | Half load/store tests (golden not needed — generic emit path) |
| `kaio-runtime/Cargo.toml` | Enable cudarc `f16` feature |
| `kaio/Cargo.toml` | Re-export `half::f16`/`half::bf16` for ergonomics |
| `kaio/tests/f16_buffer.rs` | NEW — GpuBuffer roundtrip tests |

## Review Notes

- **Adversarial review:** 12-point review. Biggest gap was load/store
  roundtrip coverage for the half register classes (fixed — added
  the three `emit_ld_*/emit_st_*` tests). Also caught the need for
  the end-to-end `emit_kernel_f16_flow` emitter test, and suggested
  the cvt rounding rationale be documented in code comments.
- **Planning review:** kaio-core zero-dep change explicitly
  acknowledged and accepted (`half` is the right call). Confirmed
  the BF16 "type-level only this sprint" deferral plan.

## Follow-ups carried into 6.2

- `mma.sync` and `cp.async` instruction emission
- Typed fragment containers (FragmentA/B/C) — A/B use b32-packed
  `%r` registers, NOT `%h`, because that's how mma.sync encodes half2
- Standalone single-instruction validation test before tiled matmul
- `RegisterAllocator::alloc_packed_half2()` helper for fragment regs
