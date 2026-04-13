# Sprint 1.2 — Arithmetic Instructions

**Commit:** `279c799`
**Status:** Complete

## Context

Sprint 1.2 is the first sprint that emits real PTX text. It populates
`ArithOp` with the three arithmetic operations used in `vector_add`
and writes `Emit` impls validated byte-for-byte against nvcc 12.8 output.
This establishes the pattern all future instruction sprints (1.3, 1.4)
will follow.

## Decisions

### Which arithmetic operations to include

**Context:** PTX has ~15 arithmetic instructions. `docs/implementation.md`
§1.3 Priority 1 lists: add, sub, mul, div, rem, mad, fma, neg, abs,
min, max. The master plan scoped Sprint 1.2 to "the minimum set for
vector_add."

**Analysis of nvcc vector_add output:**
- `mad.lo.s32` — index computation (`blockIdx.x * blockDim.x + threadIdx.x`)
- `mul.wide.u32` — byte offset (`idx * sizeof(f32)`, 32→64 widening)
- `add.s64` — pointer + offset (address arithmetic, 3 occurrences)
- `add.f32` — the actual vector addition

**Decision:** Three variants only: `Add`, `Mad`, `MulWide`. Everything
else (Sub, Mul, Div, Fma, Neg, Abs, Min, Max) deferred until a kernel
needs them. Each is a one-variant + one-match-arm addition when the time
comes.

### MadMode — include Hi and Wide or just Lo?

**Context:** PTX `mad` instruction has three modes: `lo` (low 32 bits of
product), `hi` (high 32 bits), `wide` (full 64-bit product from 32-bit
inputs). Only `mad.lo.s32` appears in vector_add.

**Decision:** Enum has only `Lo`. Adding `Hi`/`Wide` later is trivial
(new variant + match arm). Didn't stub them because empty enum variants
with no construction site would trigger dead code warnings under the
project's strict clippy config.

### Source operands — Register or Operand?

**Context:** Should `ArithOp::Add { lhs, rhs }` use `Register` (only
register-to-register ops) or `Operand` (registers, immediates, special
registers)?

**Evidence from nvcc:** `mul.wide.u32 %rd5, %r1, 4` — the `4` is an
immediate operand. If sources were `Register`, we couldn't represent this.

**Decision:** Source operands are `Operand`. Only `dst` is `Register`
(destinations are always registers in PTX). This is consistent across
all three variants.

### Where to put the ArithOp Emit impl

**Context:** Two options for Emit impl placement:
1. In `emit/emit_trait.rs` alongside all other Emit impls — centralized
2. In `instr/arith.rs` alongside the ArithOp type — co-located

**Decision:** Co-located in `instr/arith.rs`. Rationale:
- Each instruction category will grow its own emission logic; keeping it
  next to the type definition means adding a variant and its emission is
  a single-file change
- Avoids `emit_trait.rs` growing into a monolith as more instruction
  categories land
- Work on Sprint 1.3 (memory) only touches `instr/memory.rs` — no
  merge conflicts with arith work

Consequence: removed the `ArithOp` import from `emit_trait.rs` (stale
after the move).

### PtxInstruction dispatch — wire Arith now or wait for 1.5?

**Context:** `PtxInstruction::Emit` was a blanket `Ok(())` for all
variants. Sprint 1.2 adds real ArithOp emission. Should we wire the
dispatch now or leave it for Sprint 1.5?

**Decision:** Wire it now. Without dispatch, there's no way to test
ArithOp emission through the `PtxInstruction` layer (test 6 validates
this). Also switched `Memory`/`Control` arms from `Ok(())` to
`match *op {}` (uninhabited empty match) — this is safer because if
someone accidentally constructs a MemoryOp before Sprint 1.3, the
compiler will error instead of silently emitting nothing.

### &dyn Display coercion in instruction() call sites

**Context:** `PtxWriter::instruction()` takes `&[&dyn Display]`. ArithOp
match arms pass a mix of `&Register` (dst) and `&Operand` (lhs, rhs).
These are different concrete types.

**Concern:** Rust might not auto-coerce heterogeneous concrete references
to `&dyn Display` in a slice literal.

**Result:** It works. Rust coerces `&Register` and `&Operand` to
`&dyn Display` in the slice context when the first element has an
explicit `as &dyn fmt::Display` cast. The remaining elements coerce
automatically. Validated at compile time — no explicit casts needed on
source operands.

## Scope

**In:** ArithOp (Add, Mad, MulWide), MadMode (Lo), Emit impl, PtxInstruction
Arith dispatch, 7 inline tests, 4 nvcc golden matches.

**Out:** Sub, Mul, Div, Rem, Fma, Neg, Abs, Min, Max, MadMode::Hi/Wide,
rounding modes, saturate modifiers, MemoryOp, ControlOp.

## Results

Completed as planned. One minor fix: stale `ArithOp` import in
`emit_trait.rs` removed (the Emit impl moved to `arith.rs`).

**Tests:** 27 total (20 existing + 7 new). **Files modified:** 3.

**nvcc golden matches (byte-for-byte):**
- `add.f32 %f3, %f2, %f1;` ✓
- `add.s64 %rd6, %rd4, %rd5;` ✓
- `mad.lo.s32 %r1, %r3, %r4, %r5;` ✓
- `mul.wide.u32 %rd5, %r1, 4;` ✓
