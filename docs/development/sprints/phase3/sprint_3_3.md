# Sprint 3.3 — Shared Memory in Macro DSL

**Status:** Complete
**Date:** 2026-04-11
**Depends on:** Sprint 3.1 (loops), Sprint 3.2 (kaio-core instructions)

## Context

Wires shared memory into the `#[gpu_kernel]` macro. Users write
`let sdata = shared_mem![f32; 256];` and access it via `sdata[i]`.

## Decisions

### Parse shared_mem! in parse_let, not parse_expr

**Context:** `let sdata = shared_mem![f32; 256]` is a let binding with an
`Expr::Macro` initializer. Need to intercept before normal init parsing.

**Decision:** Detect `Expr::Macro` with path `shared_mem` in `parse_let`
via early-return before `parse_expr`. Returns `KernelStmt::SharedMemDecl`
with the variable name from the let binding. Bare `shared_mem![]` without
let gets a clear error: "must be assigned to a variable".

### Operand::SharedAddr for named allocation addresses

**Context:** Need `mov.u32 %r, sdata;` to load a shared allocation's base
address. The existing `Operand` enum has no variant for named addresses.

**Decision:** Added `Operand::SharedAddr(String)` to kaio-core. Displays
as the bare name. Verified via ptxas that `mov.u32 %r, sdata;` is valid PTX
for loading shared memory base addresses. Minimal change: 1 variant, 1
Display arm. Enables correct multi-allocation addressing for Sprint 3.5.

### 32-bit shared memory addressing

**Context:** Global memory uses 64-bit addressing (mul.wide + add.s64).
Shared memory is per-SM SRAM with 32-bit addresses.

**Decision:** `compute_shared_address` uses `mul.lo.u32` (byte offset) +
`add.u32` (base + offset) — all U32. Documented with explicit comment
explaining the 32-bit vs 64-bit distinction to prevent future "fixes".

### Shared arrays tracked separately from locals

**Context:** Shared memory buffers are not pointer parameters — they don't
have registers in `ctx.locals`.

**Decision:** New `ctx.shared_arrays: HashMap<String, (KernelType, usize)>`
field. Index reads/writes check `shared_arrays` first, then fall back to
`locals`. Shared memory is always mutable — no `&mut` check needed.

## Scope

**In:**
- `KernelStmt::SharedMemDecl` IR variant
- Parse `shared_mem![T; N]` in let bindings
- Lower SharedMemDecl → `kernel.add_shared_decl()` tokens
- Shared index read/write dispatch (ld.shared/st.shared)
- `compute_shared_address` with 32-bit addressing
- `Operand::SharedAddr(String)` in kaio-core
- `shared_mem!` macro stub for IDE autocomplete
- Parse + lower tests

**Out:**
- GPU E2E test (deferred to Sprint 3.4 — needs `bar_sync()` builtin)
- Compile-time shared memory size reporting (Sprint 3.8)

## Results

Completed as planned.

**Files modified:** 8
- `kaio-core/src/ir/operand.rs` (SharedAddr variant + Display)
- `kaio-core/src/emit/emit_trait.rs` (emit_mov_shared_addr test)
- `kaio-macros/src/kernel_ir/stmt.rs` (SharedMemDecl variant)
- `kaio-macros/src/parse/body.rs` (parse shared_mem!, Span import,
  improved Expr::Macro error for shared_mem, 3 parse tests)
- `kaio-macros/src/lower/mod.rs` (shared_arrays field, SharedMemDecl
  lowering, Index/IndexAssign shared dispatch)
- `kaio-macros/src/lower/memory.rs` (compute_shared_address,
  lower_shared_index_read/write, 3 memory tests)
- `kaio-macros/src/codegen/ptx_builder.rs` (SharedDecl import)
- `kaio/src/gpu_builtins.rs` (shared_mem! macro stub)

**Tests:** 190 total (78 kaio-core + 107 kaio-macros + 5 integration),
all passing. +1 kaio-core test (SharedAddr emit), +6 kaio-macros tests
(3 parse + 3 memory lowering).

**Quality gates:** `cargo fmt --check` clean, `cargo clippy -- -D warnings`
zero warnings, all tests pass.
