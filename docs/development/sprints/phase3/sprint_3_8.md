# Sprint 3.8 — Polish + Coverage + Docs

**Status:** Complete
**Commit:** `0691be8`
**Date:** 2026-04-11
**Depends on:** All prior Phase 3 sprints

## Context

Final Phase 3 sprint. Fix known bugs, add success criteria items, update
CHANGELOG, prepare for Phase 4.

## Completed Items

### Variable shadowing — Fixed
Removed duplicate-name error check in `lower_stmt` for `Let`. HashMap
`insert()` overwrites silently. Added comment explaining why: each `let`
allocates a fresh register via `lower_expr`, and the old register becomes
unreferenced. Softmax kernel can now use `let mut i = tid;` multiple times.

### Cvt rounding modifier — Fixed
Added rounding modifier logic to `PtxInstruction::Cvt` emission:
- int → float: `.rn` (round to nearest even)
- float → int: `.rzi` (round toward zero, Rust `as` semantics)
- float → float: `.rn`
- int → int: no modifier

Unblocks `as f32` / `as u32` casts in kernels. 2 new kaio-core tests.

### SM target configuration — Done
`KAIO_SM_TARGET` env var in generated `build_ptx()`. Default `sm_70` for
maximum compatibility (Volta+, 7 years of GPUs). Users override with
`KAIO_SM_TARGET=sm_89 cargo test` for their hardware.

### Shared memory compile-time reporting — Done
If shared memory exceeds 48KB, emits warning via `eprintln!` in generated
`build_ptx()`.

### CHANGELOG — Done
Phase 3 completion entry with all deliverables, fixes, and changes.
