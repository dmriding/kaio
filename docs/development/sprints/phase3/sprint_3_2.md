# Sprint 3.2 — Shared Memory + Barrier + Shuffle Instructions

**Status:** Complete
**Date:** 2026-04-11
**Depends on:** Phase 2 complete (commit `50a3ab0`). Independent of Sprint 3.1.

## Context

Adds PTX instruction primitives for shared memory, barrier synchronization,
and warp shuffle to kaio-core. Sprint 3.3+ will consume these from the
macro layer. Pure kaio-core sprint — no macro changes.

## Decisions

### shfl.sync operand order — confirmed via ptxas

**Context:** The mask position in shfl.sync varies between PTX ISA versions.
Older specs had the mask at the end, newer specs moved it.

**Decision:** Tested against ptxas with PTX ISA 8.7 / SM 8.9 (RTX 4090).
Confirmed operand order: `shfl.sync.down.b32 dst, src, delta, c, membermask;`
— mask is LAST. All three variants (down, up, bfly) verified with ptxas
before writing emitter code.

### shfl.sync `c` parameter stored as pre-packed u32

**Context:** The `c` operand encodes clamp width in a packed format. The
encoding formula varies by shuffle mode.

**Decision:** kaio-core stores `c: u32` as-is and emits it literally.
The encoding logic (`((32 - width) << 8) | (width - 1)` etc.) is the
macro layer's responsibility (Sprint 3.4). This gives clean separation
of concerns — kaio-core is a PTX emitter, not a semantic encoder.

### SharedDecl uses .b8 with byte count

**Context:** Shared memory can be declared with typed syntax (`.f32 name[count]`)
or untyped (`.b8 name[byte_count]`).

**Decision:** Use `.b8` with total byte count, matching nvcc output convention.
Format: `.shared .align {align} .b8 {name}[{size_bytes}];`. Alignment is
explicit (4 for f32, 8 for f64).

### Emit patterns — w.line() for non-standard syntax

**Context:** `bar.sync` and `shfl.sync` have operand formats that don't fit
`w.instruction()`'s comma-separated Display args pattern cleanly (hex mask,
non-register operands mixed with literals).

**Decision:** Use `w.line()` with format strings, consistent with how
`BraPred` is already implemented.

## Scope

**In:**
- `MemoryOp::LdShared`, `MemoryOp::StShared` + Emit + tests
- `ControlOp::BarSync`, `ShflSyncDown/Up/Bfly` + Emit + tests
- `SharedDecl` struct, `PtxKernel::shared_decls` field + preamble emission
- Integration test: hand-built shared memory kernel
- ptxas verification for shared memory + shfl.sync kernel

**Out:**
- Cvt rounding modifier fix (pre-existing issue, discovered during testing)
- Macro layer integration (Sprint 3.3)
- shfl.sync `c` encoding formulas (Sprint 3.4)

## Results

Completed as planned.

**Bug discovered (not fixed — out of scope):** `PtxInstruction::Cvt` does not
emit rounding modifiers (e.g., `.rn` for round-to-nearest). `cvt.f32.u32`
should be `cvt.rn.f32.u32`. This is a pre-existing issue from Phase 1 that
hasn't surfaced because the macro layer's `cast::lower_cast()` may handle
it differently. Noted for Sprint 3.8 polish.

**Files modified:** 5
- `kaio-core/src/ir/kernel.rs` (SharedDecl struct, shared_decls field, add_shared_decl)
- `kaio-core/src/ir/mod.rs` (re-export SharedDecl)
- `kaio-core/src/emit/emit_trait.rs` (emit shared decls in kernel preamble)
- `kaio-core/src/instr/memory.rs` (LdShared, StShared + Emit + 2 tests)
- `kaio-core/src/instr/control.rs` (BarSync, ShflSyncDown/Up/Bfly + Emit + 5 tests)
- `kaio-core/tests/common/mod.rs` (build_shared_mem_ptx helper)
- `kaio-core/tests/vector_add_emit.rs` (shared memory emission test)
- `kaio-core/tests/ptxas_verify.rs` (ptxas verification for shared mem kernel)

**Tests:** 188 total (77 kaio-core + 109 kaio-macros + 2 integration),
all passing. ptxas verification passes for both vector_add and shared_mem
kernels on SM 8.9.

**Quality gates:** `cargo fmt --check` clean, `cargo clippy -- -D warnings`
zero warnings, all tests pass.
