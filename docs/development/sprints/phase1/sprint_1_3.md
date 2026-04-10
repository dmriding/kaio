# Sprint 1.3 — Memory Instructions

**Commit:** `23db945`
**Status:** Complete

## Context

Sprint 1.3 adds the four memory operations used in `vector_add`:
parameter loads, global memory loads/stores, and address space conversion.
These are the first instructions with PTX formatting quirks — bracket
syntax for addresses and reversed operand order for stores.

## Decisions

### Register vs Operand for memory instruction operands

**Context:** Sprint 1.2's ArithOp used `Operand` for source operands to
support immediates (`mul.wide.u32 %rd5, %r1, 4` — the `4` is a literal).
Should MemoryOp follow the same pattern?

**Analysis:** In PTX, memory instructions always reference registers for
addresses and values. You can't `ld.global` from an immediate address
(there's no `ld.global.f32 %f0, [0x1234]` in practice) and you can't
`st.global` an immediate value (you'd `mov` to a register first, then
store). The only non-register operand is `LdParam`'s parameter name,
which is a `String`, not an `Operand`.

**Decision:** All memory operands use `Register` (not `Operand`).
Enforcing this at the type level prevents constructing invalid IR like
"store immediate 3.14 to global memory." `LdParam` uses `String` for
the parameter name since it references the kernel signature, not a
register.

### Bracket formatting — in Emit or in Display?

**Context:** PTX uses bracket syntax for memory addresses: `[%rd8]`,
`[vector_add_param_0]`. This could be handled by:
1. Adding a `Display` variant that wraps registers in brackets
2. A helper method on `Register` like `.bracketed()` returning a wrapper type
3. Formatting the brackets inline in each MemoryOp Emit match arm

**Decision:** Option 3 — inline formatting with `format!("[{reg}]")`.
The bracket syntax is memory-instruction-specific. It doesn't belong
in `Register::Display` (which should always return `%rd8`, not `[%rd8]`
— other instructions don't use brackets). Adding a wrapper type for a
string concat is over-engineering. The inline format is clear, local,
and each match arm shows exactly what PTX it produces.

### Store operand order — struct order vs PTX order

**Context:** PTX stores have reversed operand order: `st.global.f32
[addr], src` — address first, value second. This is opposite to loads
(`ld.global.f32 dst, [addr]`) and arithmetic (`add.f32 dst, lhs, rhs`)
where the destination comes first.

**Options:**
- **(A)** Struct fields match PTX syntax: `StGlobal { addr_first, src_second }`.
  Easy emit but confusing semantics — "addr_first" is the memory destination,
  not the register destination.
- **(B)** Struct fields match semantics: `StGlobal { addr, src }` where `addr`
  is "store AT this address" and `src` is "store THIS value." Emit impl
  handles the PTX order reversal.

**Decision:** (B) — struct fields match semantics. A developer reading
`StGlobal { addr: r5, src: f0 }` understands "store f0 at r5" without
knowing PTX operand ordering. The Emit impl has a comment marking the
reversal. A dedicated test (`st_global_operand_order`) validates the
emitted PTX has `[addr]` before `src` by checking string positions.

### CvtaToGlobal — explicit type parameter or hardcoded?

**Context:** `cvta.to.global.u64` always uses `.u64` because our module
is `.address_size 64`. Should we make the address type a field for
forward compatibility (32-bit address spaces)?

**Decision:** Hardcoded `.u64` in the emitter, no type field. 32-bit
address space support is extremely unlikely for KAIO (no modern GPU
uses 32-bit addressing). Adding a field "for future flexibility" would
be dead weight now and create a decision point at every call site for
something that's always `.u64`. If needed later, it's a one-field-add
+ one-match-arm change.

### LdParam — String vs &str vs PtxParam reference

**Context:** `LdParam` needs to reference a kernel parameter. Options:
1. `param_name: String` — owned name, copies from `PtxParam::name()`
2. `param_name: &'a str` — borrowed, but adds a lifetime to MemoryOp
   and everything that contains it (PtxInstruction, PtxKernel, PtxModule)
3. `param: &'a PtxParam` — direct reference, even worse lifetime infection

**Decision:** `String`. A few bytes of heap allocation per `ld.param`
instruction is irrelevant for a code generator. Lifetime-free IR types
are dramatically easier to work with — no borrow checker fights when
building instruction sequences, no issues with the allocator owning
registers while instructions reference them. Worth the allocation cost.

## Scope

**In:** MemoryOp (LdParam, LdGlobal, StGlobal, CvtaToGlobal), Emit impl,
PtxInstruction::Memory dispatch, 7 inline tests, 5 nvcc golden matches.

**Out:** Shared memory ops, cache modifiers, vector loads, ld.const,
ld.local, atomics, any changes to ArithOp or ControlOp.

## Results

Completed as planned. No deviations.

**Tests:** 34 total (27 existing + 7 new), all passing.
**Files modified:** 2 (instr/memory.rs, emit/emit_trait.rs).

**nvcc golden matches (byte-for-byte):**
- `ld.param.u64 %rd1, [vector_add_param_0];` ✓
- `ld.param.u32 %r2, [vector_add_param_3];` ✓
- `cvta.to.global.u64 %rd4, %rd1;` ✓
- `ld.global.f32 %f1, [%rd8];` ✓
- `st.global.f32 [%rd10], %f3;` ✓
