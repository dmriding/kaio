# Phase 1 — PTX Foundation: Sprint Log

Running log of plan-vs-reality for each sprint in Phase 1.
Each sprint has a **Plan** section (captured before execution) and a
**Results** section (captured after execution).

---

## Sprint 1.0 — Workspace Restructure

### Plan

Restructure the repo from a single umbrella crate at the workspace root
to a virtual workspace with three member crates. This is prerequisite
infrastructure — no functional code, just layout.

**Scope:**
- Convert root `Cargo.toml` from `[workspace] + [package]` to `[workspace]` only
- Move umbrella `pyros` crate to `pyros/` subdirectory
- Create `pyros-core/` and `pyros-runtime/` stub crates
- Add `rust-toolchain.toml` pinning Rust 1.94.1
- Pin `cudarc = 0.19` in workspace deps with `driver + std + dynamic-loading + cuda-12080`
- Fix doc drift: MSRV 1.75→1.94, cudarc repo URL coreylowman→chelsea0x3b

**Key decisions:**
- Virtual workspace (not umbrella-at-root) to match docs/index.md layout
- `resolver = "3"` for edition 2024 default
- cudarc `cuda-12080` feature required for FFI binding header selection (discovered during build — not in original plan)
- NVRTC feature intentionally off — PYROS emits PTX, doesn't compile C kernels
- License/README copies in `pyros/` for crate tarball self-containment

### Results

**Completed as planned.** One deviation:

- **cudarc CUDA version feature** — the original plan had `features = ["driver", "std", "dynamic-loading"]` which panicked at build time. cudarc 0.19.4's `build.rs` requires one `cuda-XXXXX` feature for FFI header selection. Added `cuda-12080`. Documented in a comment in root `Cargo.toml`.

**Quality gates:** build, fmt, clippy, doc, `cargo package -p pyros --allow-dirty` — all clean. Umbrella crate still packages the same 8 files.

**Commit:** `8e59f66 restructure into virtual workspace for phase 1 kickoff`

**Files created:** 7 (rust-toolchain.toml, pyros/Cargo.toml, pyros/README.md, pyros/LICENSE-MIT, pyros/LICENSE-APACHE, pyros-core/Cargo.toml + lib.rs, pyros-runtime/Cargo.toml + lib.rs)
**Files modified:** 3 (Cargo.toml, docs/index.md, docs/implementation.md)
**Files moved:** 1 (src/lib.rs → pyros/src/lib.rs)

---

## Sprint 1.1 — Types + IR Skeleton + Register Allocator

### Plan

Create the type system, IR tree nodes, register allocator, operand types,
and the Emit trait + PtxWriter scaffold inside `pyros-core`. This is the
load-bearing foundation — nothing downstream starts until 1.1 compiles
and tests pass.

**Scope:**
- `types.rs` — `PtxType` (7 variants), `RegKind` (5 kinds), `GpuType` sealed trait with impls for 7 Rust types
- `ir/` — `PtxModule`, `PtxKernel`, `Register`, `RegisterAllocator` (5-counter model), `PtxParam` (Scalar/Pointer), `Operand` (8 variants including separate signed/unsigned immediates), `SpecialReg` (12 variants), `PtxInstruction` (7 variants)
- `instr/` — `ArithOp`, `MemoryOp`, `ControlOp` as uninhabited empty enums (stub pattern for 1.2/1.3/1.4 to populate)
- `emit/` — `Emit` trait, `PtxWriter` scaffold, empty Emit impls for all IR nodes
- Unit tests for types, register allocator, params, PtxWriter

**Key decisions:**
- F16/BF16 intentionally omitted from `PtxType` — deferred to Phase 3+
- Operand has separate `ImmI32`/`ImmU32`/`ImmI64`/`ImmU64` to avoid casting friction in Sprint 1.2 arith
- Empty enums as stubs — uninhabited types that compile but can't be constructed. `match *self {}` in Emit impls breaks intentionally when variants are added (forces implementer to write real emission)
- `PtxWriter::instruction` uses `&[&dyn Display]` — noted as ergonomic friction, `ptx_instr!()` macro deferred to Sprint 1.5
- `Emit::emit` returns `fmt::Result` (not `PyrosError`) to keep pyros-core decoupled from pyros-runtime's error type
- `RegisterAllocator` tracks all allocations in a Vec for `.reg` declaration emission
- `RegKind::counter_index()` is `pub(crate)` — internal implementation detail

**File tree:**
```
pyros-core/src/
├── lib.rs              (modified)
├── types.rs            (created)
├── ir/
│   ├── mod.rs, module.rs, kernel.rs, register.rs
│   ├── param.rs, operand.rs, instruction.rs
├── instr/
│   ├── mod.rs, arith.rs, memory.rs, control.rs
└── emit/
    ├── mod.rs, writer.rs, emit_trait.rs
```

### Results

**Completed as planned.** No deviations from scope.

**Minor fixes during execution:**
- `cargo fmt` reformatted `PtxWriter::instruction` signature (multi-line → single-line) and `matches!` assertions in param tests — auto-applied
- Doc link `RegisterAllocator::into_allocated` in kernel.rs couldn't resolve — fixed to use full path `super::register::RegisterAllocator::into_allocated`

**Quality gates:**
| Gate | Result |
|---|---|
| `cargo build -p pyros-core` | clean |
| `cargo test -p pyros-core` | **19 passed**, 0 failed |
| `cargo fmt --all --check` | clean |
| `cargo clippy --workspace --all-targets -- -D warnings` | clean |
| `cargo doc -p pyros-core --no-deps` | clean (0 warnings) |

**Test breakdown:**
- types: 5 tests (size_bytes, ptx_suffix, reg_kind, reg_prefix, gpu_type_impls)
- ir/register: 4 tests (sequential indices, independent counters, allocation order, display)
- ir/param: 2 tests (scalar, pointer)
- emit/writer: 8 tests (line, indent/dedent, blank, instruction formatting, no-operands, label uniqueness, dedent saturation)

**Files created:** 16
**Files modified:** 1 (pyros-core/src/lib.rs)

**Commit:** pending

---

## Sprint 1.2 — Arithmetic Instructions

### Plan

_To be written at sprint start._

---

## Sprint 1.3 — Memory Instructions

### Plan

_To be written at sprint start._

---

## Sprint 1.4 — Control Flow + Special Registers

### Plan

_To be written at sprint start._

---

## Sprint 1.5 — PtxWriter + Full Module Emission

### Plan

_To be written at sprint start._

---

## Sprint 1.6 — Runtime Device + Buffers

### Plan

_To be written at sprint start._

---

## Sprint 1.7 — Runtime Launch + vector_add E2E

### Plan

_To be written at sprint start._

---

## Sprint 1.8 — Testing + Coverage + Docs Polish

### Plan

_To be written at sprint start._
