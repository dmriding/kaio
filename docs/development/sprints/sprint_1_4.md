# Sprint 1.4 — Control Flow + Special Registers

**Commit:** pending
**Status:** Complete

## Context

Sprint 1.4 is the last instruction category sprint. After this, every
PTX instruction needed for `vector_add` exists in the IR. Sprint 1.5
will orchestrate them into a complete `.ptx` module.

## Decisions

### CmpOp — include all 6 variants or just Ge?

**Context:** Only `setp.ge.u32` appears in vector_add. The master plan
scoped Sprint 1.2's ArithOp to "minimum for vector_add" (only 3 of ~15
possible operations). Should CmpOp follow the same philosophy?

**Decision:** Include all 6 (Eq, Ne, Lt, Le, Gt, Ge). Unlike arithmetic
operations where each variant has different semantics, register types,
and emission logic, comparison operators are trivially a single `&str`
mapping. Adding one is identical to adding all six. Every non-trivial
kernel needs comparisons beyond `>=`, and leaving them out would mean
touching this code again for the first `if x < threshold` kernel.

### SetP operands — Register or Operand?

**Context:** Should `setp` source operands be `Register` (register-only
comparisons) or `Operand` (allows comparing against immediates)?

**Analysis:** PTX supports `setp.ge.u32 %p1, %r1, 42` — comparing a
register against an immediate. This is common for bounds checking
(e.g., `if idx < N` where N is a compile-time constant). Restricting
to `Register` would force a `mov` for every constant comparison.

**Decision:** `Operand`. Consistent with ArithOp's convention. The
Phase 2 macro will generate constant comparisons without needing an
extra `mov` instruction.

### BraPred formatting — instruction() vs line()

**Context:** PTX predicated branches have the format `@%p1 bra target;`
which has a prefix (`@%p1`) before the mnemonic. This doesn't fit
`PtxWriter::instruction()`'s `mnemonic op1, op2;` pattern — there's
no comma-separated operand list and the predicate guard is syntactically
distinct from a regular operand.

**Options:**
- **(A)** Stuff predicate into mnemonic: `instruction("@%p1 bra", &[&target])`.
  Works but semantically wrong — the predicate is separate from the instruction.
- **(B)** Use `line()` with explicit formatting: `line(&format!("@{pred} bra {target};"))`.
  Clear, explicit, handles the special syntax directly.
- **(C)** Add `PtxWriter::predicated_instruction()` method. Over-engineering for
  Sprint 1.4 — only one predicated instruction exists.

**Decision:** (B) — `line()` with manual formatting. The format is explicit in
the match arm, and `line()` handles indentation. If Phase 3 introduces more
predicated instructions (predicated stores, predicated moves), we can add
a proper `predicated_instruction()` method then.

### Bra (unconditional) — include or defer?

**Context:** `vector_add` doesn't use unconditional branches. Phase 3
loop lowering will need them (`for` → counter + `bra` back to loop top).

**Decision:** Include. One variant, one match arm, one test. The
implementation cost is effectively zero, and omitting it means touching
`control.rs` again for the first loop kernel. "Don't make me come back
for something trivial" is the right heuristic here.

### Label emission — Sprint 1.4 or 1.5?

**Context:** `PtxInstruction::Label` exists from Sprint 1.1 but its
`Emit` impl returns `Ok(())`. Labels need special indentation — they
print at column 0 (`EXIT:`) while instructions are indented (`    ret;`).
This requires either PtxWriter API changes or temporary indent manipulation.

**Decision:** Defer to Sprint 1.5. Labels are a formatting concern, not
an instruction-category concern. Sprint 1.4 tests only verify ControlOp
instruction-level output — they don't need label emission. Sprint 1.5
already plans to implement Mov/Cvt/Label/Comment emission alongside the
full kernel orchestration, where indentation management is a natural fit.

### Special register helpers — function API vs builder

**Options:**
- **(A)** Free functions: `special::tid_x(&mut alloc) -> (Register, PtxInstruction)`.
  Caller pushes instruction and keeps register. No hidden state.
- **(B)** Builder method: `kernel.read_special(SpecialReg::TidX, &mut alloc)` that
  auto-pushes the instruction and returns the register. Less boilerplate
  but hides the instruction push.

**Decision:** (A) — free functions. Explicit is better. The caller sees
both the register and the instruction. Nothing is hidden. The `(Register,
PtxInstruction)` tuple pattern is slightly verbose but fully transparent —
exactly what a GPU kernel framework should prioritize over convenience.

### `&target as &dyn fmt::Display` for Bra emit

**Context:** `ControlOp::Bra { target: String }` — in the emit match arm,
`target` is `&String`. Passing to `instruction(&[...])` needs `&dyn Display`.
`&String` coerces to `&dyn Display` (String implements Display), but
Claude Desktop review flagged a potential `&&str` double-reference issue
if using `target.as_str()`.

**Decision:** Use `&target as &dyn fmt::Display` explicitly. Avoids the
double-reference edge case entirely. Compiler confirms it works.

## Scope

**In:** CmpOp (6 variants), ControlOp (SetP, BraPred, Bra, Ret), Emit impl,
PtxInstruction::Control dispatch, instr/special.rs (12 helpers), 8 tests
(6 in control.rs, 2 in special.rs), 3 nvcc golden matches.

**Out:** Label/Comment emission, Mov/Cvt emission, predicate negation,
bar.sync, shfl.sync, any changes to ArithOp or MemoryOp.

## Results

Completed as planned. One minor auto-format fix:
- `cargo fmt` collapsed `Bra` match arm from 3 lines to 1 (single-expression body)

`emit_trait.rs` now has zero instruction-category imports — all Emit impls
co-located with their types. Clean separation achieved.

**Tests:** 42 total (34 existing + 8 new), all passing.
**Files modified:** 3 (instr/control.rs, instr/mod.rs, emit/emit_trait.rs).
**Files created:** 1 (instr/special.rs).

**nvcc golden matches (byte-for-byte):**
- `setp.ge.u32 %p1, %r1, %r2;` ✓
- `@%p1 bra $L__BB0_2;` ✓
- `ret;` ✓
