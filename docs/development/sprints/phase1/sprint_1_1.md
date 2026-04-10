# Sprint 1.1 — Types + IR Skeleton + Register Allocator

**Commit:** `b2dac92`
**Status:** Complete

## Context

Sprint 1.1 creates the type system, IR tree, register allocator, and
emit scaffold inside `pyros-core`. This is the load-bearing foundation —
every subsequent sprint (1.2–1.8) builds directly on these types. Nothing
downstream starts until 1.1 compiles and passes tests.

## Decisions

### PtxType variants — include F16/BF16 or defer?

**Context:** `docs/implementation.md` lists `f16` and `bf16` in the type
mapping table. Including them now means they're available for future
kernels. Excluding means one fewer set of enum variants, RegKind
extensions, and GpuType impls to maintain until needed.

**Decision:** Defer F16/BF16 to Phase 3+. Rationale:
- No Phase 1 kernel uses half precision
- F16/BF16 need new RegKind values (`%h`, `%hb`) that don't exist in the
  5-counter model
- Half-precision types may require the `half` crate dependency
- Adding later is a non-breaking change (new enum variants, new trait impls)

Marked with an explicit "do NOT add" comment to prevent Forge agents from
scope-creeping.

### Register allocator — 5 counters vs 7 (per PTX type)

**Context:** `docs/implementation.md` §1.2 suggested separate register
prefixes for signed vs unsigned integers (`%r` for i32, `%ru` for u32,
`%rd` for i64). But real `nvcc` output uses `%r` for ALL 32-bit integers
and `%rd` for ALL 64-bit integers — the sign distinction is carried by
the instruction, not the register name.

**Options:**
- **(A) 7 counters** (one per PtxType). Each type gets a unique prefix.
  Problem: `%r0` could be declared as both `.reg .s32 %r0` and
  `.reg .u32 %r0` — name collision.
- **(B) 5 counters** (one per RegKind — R, Rd, F, Fd, P). Signed and
  unsigned share the prefix. Matches nvcc convention. The PTX type is
  stored on the Register struct for `.reg` declaration emission.

**Decision:** (B) 5 counters. Matches nvcc output, avoids name collisions,
and users inspecting PYROS-emitted PTX will recognize the convention.

### Register declarations — typed (.s32) vs untyped (.b32)

**Context:** After running `nvcc --ptx` on a real vector_add kernel, we
discovered nvcc uses `.reg .b32 %r<N>` (untyped bit-container) for
integer registers, not `.reg .s32 %r<N>` or `.reg .u32 %r<N>`. Floats
and predicates keep typed declarations (`.f32`, `.f64`, `.pred`).

**Decision:** Match nvcc convention. Added `PtxType::reg_decl_type()`
which returns `.b32`/`.b64` for integers and keeps typed suffixes for
floats/predicates. Sprint 1.5 will use this method when emitting
register declaration lines.

### PTX version — 7.8 vs 8.7

**Context:** The master plan assumed `.version 7.8`. Running `nvcc --ptx`
with CUDA 12.8 showed the actual PTX ISA version is 8.7. CUDA 12.8 maps
to PTX ISA 8.7, not 7.8.

**Decision:** Changed `PtxModule::new()` default to `"8.7"`. Saved the
nvcc output as a golden reference file at
`pyros-core/tests/golden/nvcc_vector_add_sm89.ptx`.

### Operand immediates — single i32 vs separate signed/unsigned

**Context:** Dave (via review feedback) flagged that `Operand::Imm32(i32)`
would force casting through `i32` when passing unsigned values. The nvcc
output has `mul.wide.u32 %rd5, %r1, 4` where `4` should be unsigned.

**Decision:** Split into `ImmI32(i32)`, `ImmU32(u32)`, `ImmI64(i64)`,
`ImmU64(u64)`. Avoids casting friction in Sprint 1.2 arith emitters.
Both signed and unsigned variants render the same way via Display, but
the type distinction matters for correctness of the IR representation.

### Empty enums as stubs for instruction categories

**Context:** `ArithOp`, `MemoryOp`, `ControlOp` need to exist as types
referenced by `PtxInstruction` variants, but their actual variants are
added in sprints 1.2, 1.3, 1.4 respectively.

**Options:**
- **(A) Placeholder variant** — `ArithOp { _Placeholder }`. Constructible
  but meaningless. Requires `#[allow(dead_code)]` and runtime `unreachable!()`.
- **(B) Empty enum** — `pub enum ArithOp {}`. Uninhabited in Rust — can't
  construct a value. The `PtxInstruction::Arith(ArithOp)` variant exists
  but is unreachable. `Emit` impl uses `match *self {}` (exhaustive empty
  match).

**Decision:** (B) Empty enum. Cleaner — no dead code, no runtime panics,
compiler enforces unreachability. When Sprint 1.2 adds variants, `match
*self {}` breaks intentionally, forcing the implementer to write real
emission logic. Documented handoff notes on each stub enum.

### Emit trait returns fmt::Result, not PyrosError

**Context:** `PyrosError` lives in `pyros-runtime`. If `Emit::emit()`
returned `Result<(), PyrosError>`, `pyros-core` would need to depend on
`pyros-runtime` — creating a circular dependency (runtime depends on core).

**Decision:** `Emit::emit()` returns `std::fmt::Result`. PTX emission
writes into a `String` — the only failure mode is OOM, which `fmt::Result`
handles. This keeps `pyros-core` zero-dependency and decoupled from the
runtime error type.

### PtxWriter::instruction — &[&dyn Display] vs macro

**Context:** The `instruction` method signature `&[&dyn Display]` creates
friction at call sites — you need explicit `&` references and the compiler
sometimes struggles with heterogeneous trait object coercion.

**Decision:** Ship `&[&dyn Display]` for the 1.1 scaffold. Note for
Sprint 1.5: consider a `ptx_instr!()` macro for ergonomics when the
method gets heavy use. Not worth the complexity for the scaffold — want
to see the friction in practice first before optimizing.

## Scope

**In:** PtxType (7 variants), RegKind (5 kinds), GpuType (7 impls),
PtxModule, PtxKernel, PtxParam, Register, RegisterAllocator, Operand
(8 variants), SpecialReg (12 variants), PtxInstruction (7 variants),
ArithOp/MemoryOp/ControlOp stubs, Emit trait, PtxWriter, empty Emit
impls, 20 unit tests.

**Out:** F16/BF16, instruction logic, real Emit output, runtime code,
golden files, external deps.

## Results

Completed as planned. Two minor fixes:
- `cargo fmt` reformatted function signatures and `matches!` assertions
- Doc link `RegisterAllocator::into_allocated` needed full path qualification

**Tests:** 20 passed (later 1 more added for `reg_decl_type` during nvcc
validation = 21 at nvcc fix commit). **Files:** 16 created, 1 modified.
