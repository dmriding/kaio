# Sprint 3.8 — Polish + Coverage (65%) + Docs

**Status:** Planning
**Date:** 2026-04-11
**Depends on:** All prior Phase 3 sprints

## Known Items

### Variable shadowing in kernels
**Discovered in Sprint 3.6.** `let mut i = tid;` used three times in the
softmax kernel causes "variable already defined" error. Rust allows
shadowing, but KAIO's `lower_stmt` for `Let` rejects duplicate names in
`ctx.locals`. Fix: allow re-insertion (overwrite) in locals for `let`
bindings, or remove the duplicate check entirely — PTX registers are
freshly allocated regardless.

### Cvt rounding modifier
**Discovered in Sprint 3.2.** `PtxInstruction::Cvt` emits `cvt.f32.u32`
without the required `.rn` rounding modifier. ptxas rejects it. Fix: add
rounding modifier to Cvt emission for int↔float conversions. Affects
`as f32` / `as u32` casts in kernel code.

### Compile-time shared memory reporting
**From master plan (success criteria 3.9, 3.10).** Emit diagnostic with
total shared memory bytes. Warn if exceeds 48KB (SM 8.9 default).

### Coverage target: 65%
Phase 3 target is 65% workspace line coverage (up from 60% in Phase 2).
Run `cargo llvm-cov`, identify gaps, add tests.

### SM target configuration
**From Dave's notes.** Change hardcoded `sm_89` to `KAIO_SM_TARGET` env
var with default `sm_70`. One-line change in `ptx_builder.rs`. See
memory: `project_sm_target_config.md`.
