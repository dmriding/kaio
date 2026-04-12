# Sprint 6.2 — mma.sync + cp.async + Fragment Model

**Status:** In progress
**Branch:** phase6
**Parent commit:** `a1dd450` (Sprint 6.1)

## Goal

Add the first tensor-core instruction (`mma.sync`) and the async
memory pipeline primitives (`cp.async`) to kaio-core, plus the typed
fragment structs that Sprint 6.3's IR-level matmul kernel will use to
drive mma.sync without passing raw register arrays around.

This is the sprint the whole phase hinges on. Fragment-register mapping
is the single most bug-prone part of tensor-core programming. The
standalone single-instruction validation test shipped here is what
prevents silent corruption in the tiled matmul work that follows.

## Scope

- **6.2a** — `TensorCoreOp::MmaSync` + fragment types + standalone GPU
  gate test. This half must be green (emit + ptxas + GPU correctness)
  before any work on 6.2b.
- **6.2b** — `MemoryOp::CpAsync*` variants + ptxas-verify tests + a
  primitive global→shared→global smoke test. No pipelining (that is 6.4).

Do not debug both simultaneously. One correctness gate at a time.

## Architectural Decisions

### D1 — TensorCoreOp as a new instruction category

New enum `TensorCoreOp` in `kaio-core/src/instr/tensor_core.rs`, parallel
to `ArithOp`/`MemoryOp`/`ControlOp`. One variant for 6.2:

```rust
pub enum TensorCoreOp {
    MmaSync {
        d: FragmentC, a: FragmentA, b: FragmentB, c: FragmentC,
        shape: MmaShape, a_ty: PtxType, b_ty: PtxType, c_ty: PtxType, d_ty: PtxType,
    },
}
pub enum MmaShape { M16N8K16 }
```

New `PtxInstruction::TensorCore(TensorCoreOp)` variant.

### D2 — Fragment structs as typed register bags

In `kaio-core/src/fragment.rs`:

```rust
pub struct FragmentA { pub regs: [Register; 4] }  // 4 × .b32 packed half2
pub struct FragmentB { pub regs: [Register; 2] }  // 2 × .b32 packed half2
pub struct FragmentC { pub regs: [Register; 4] }  // 4 × .f32 accumulator/output
```

`pub` fields by design — these are pure register containers with no
invariant to protect. Counts are compile-time constants fixed by the
PTX ISA. A future second shape gets a sibling type (`FragmentA_m16n8k8`),
not a generic.

Free functions (not methods) for allocation and load/store:
`fragment::alloc_a/b/c`, `load_fragment_a_m16n8k16_global_row`,
`load_fragment_b_m16n8k16_global_col`,
`store_fragment_c_m16n8k16_global_row`. Shared-source variants in 6.3
will be sibling functions, not method overloads.

### D3 — Fragment A/B use `.b32` packed registers (`%r`), not `.f16` (`%h`)

Load-bearing PTX detail. `mma.sync.m16n8k16.f16.f16` packs two fp16
values per 32-bit operand register. Fragment A/B storage uses `%r`
(`.b32`), not `%h`. FragmentC uses `%f` (`.f32`) as expected.

New helper `RegisterAllocator::alloc_packed_half2() -> Register` that
returns a `%r` register with `ptx_type` tagged `U32` and a doc comment
explaining "two fp16 values packed into 32 bits — tensor-core fragment
storage." No new RegKind variant — these really are `.b32` at the PTX
level.

### D4 — m16n8k16 only (SM 8.0+)

Volta (SM 7.0) uses `m8n8k4`, Turing (SM 7.5) uses `m16n8k8`, Ampere+
(SM 8.0) adds `m16n8k16`. Each has a different fragment layout.
Phase 6 supports only `m16n8k16` — Ampere, Ada, Hopper (covers the
RTX 30/40 series and A100/H100). Master plan corrected.

### D5 — cp.async in `MemoryOp`

```rust
MemoryOp::CpAsyncCaSharedGlobal { dst_shared: Register, src_global: Register, size_bytes: u8 } // 4/8/16
MemoryOp::CpAsyncCommitGroup
MemoryOp::CpAsyncWaitGroup { n: u8 }
```

Semantically these are async memory ops. Acknowledged that commit/wait
are pipeline-state ops — if Sprint 6.4's double-buffering pressure
makes that placement feel wrong, they move into a new `PipelineOp`
category at that point. No pre-emptive refactor.

### D6 — Standalone fragment GPU test: the gatekeeper

`kaio/tests/mma_sync_fragment.rs`, `#[ignore]`. Single warp (32 threads).

Inputs:
- A: 16×16, `a[i,k] = i * 16 + k` as f16 (integers 0..255, exact)
- B: 16×8, `b[k,j] = j + 1` as f16 (column-varying 1..8 — catches
  B-fragment column-index errors that all-ones would miss)
- C: 16×8 fp32 zeros

Expected: `d[i,j] = (j+1) * (256i + 120)`, bit-exact.

Body: per-thread compute of `groupID = tid / 4`, `threadID_in_group =
tid % 4`; load fragments per NVIDIA PTX ISA §9.7.13.5.8.1 mapping;
emit one `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`; store
D per output layout.

Diagnostic caveat: this test exercises fragment mapping + addressing +
packing + operand ordering + output store mapping. A failure could be
anywhere in that path, not just mma.sync emission. That is acceptable
for the gate.

### D7–D10 (workflow)

- `sprint_6_1.md` written retroactively (this prep commit)
- `project_sm_target_config.md` memory updated to RESOLVED
- nvcc reference PTX is a debugging tool, not an upfront gate
- Tolerance: bit-exact `assert_eq!` on output bit pattern

### D11 — No FragmentAllocator wrapper

Three free allocation functions. No abstraction theatre until a second
use case appears.

### D12 — `PtxModule::validate()` — narrow target-capability check

New `ValidationError` in `kaio-core::ir::module` with variant
`SmTooLow { required, actual, feature }`. `PtxModule::validate()`
walks kernel bodies, collects min SM from any TensorCore op and
cp.async op, parses the target string (`"sm_89"` → `89`), returns
`Err` if insufficient. Called from `kaio-runtime::KaioDevice::load_ptx`
before cudarc compiles, so users get a clean error, not a ptxas
message from the bowels of the driver.

Scope is strictly target-capability. Not a semantic analysis pass,
not an operand linter. Future capability checks extend the same
pattern (cp.async.bulk → 9.0+, etc.).

### D13 — Two-commit sprint

1. **Prep commit** (docs only): this file, sprint_6_1.md retroactive,
   PHASE_6_LOG.md row update, phase6_master_plan.md SM corrections.
2. **Code commit**: TensorCoreOp, fragments, cp.async, stats, validate,
   tests, CHANGELOG, README.

Bisect-friendly.

## Verification Sequence

1. `cargo fmt --all --check`
2. `cargo clippy --workspace --all-targets -- -D warnings`
3. `cargo test -p kaio-core` — host unit tests
4. `cargo test -p kaio-core --test ptxas_verify -- --ignored` — ptxas validates mma.sync + cp.async with `KAIO_SM_TARGET=sm_80`
5. `cargo test -p kaio --test mma_sync_fragment -- --ignored` — **THE GATE**
6. `cargo test -p kaio --test cp_async_roundtrip -- --ignored` — primitive smoke test
7. `cargo test --workspace` + `cargo test --workspace -- --ignored` — full suite
8. `cargo doc --workspace --no-deps` — no `missing_docs` warnings

## Success Criteria

1. `PtxInstruction::TensorCore` variant; all existing tests still green
2. `TensorCoreOp::MmaSync` emits valid, semantically correct PTX
   (correct mnemonic, shape, layout, type signature, operand ordering),
   accepted by ptxas with `sm_80+` and validated by the gate test.
   nvcc output is calibration, not a byte-exact gate.
3. cp.async variants emit valid PTX; ptxas accepts with `sm_80`
4. Fragment structs typed (pub register bags), no raw register tuples at call sites
5. **Gate test passes**: single mma.sync produces bit-exact correct D
   from known A@B+C on RTX 4090 (sm_89)
6. cp.async roundtrip test passes on RTX 4090 — primitive smoke test
   (emission, operand order, size encoding, commit/wait once). Does
   NOT prove pipeline overlap or double-buffer safety — those are 6.4.
7. `PtxModule::validate()` rejects emit-time SM mismatches with a
   clear error (host unit test)
8. sprint_6_1.md and sprint_6_2.md both exist; PHASE_6_LOG.md accurate
9. `cargo doc` clean; fmt/clippy/test green

## Review Inputs

- **Opus 4.6** (planning review): flagged B test input gap (all-ones
  hides column errors → changed to `j+1`), recommended free-function
  fragment helpers over methods, raised `validate()` for SM gating.
- **Codex 5.4** (planning review): softened success criterion #2 from
  "byte-exact nvcc parity" to "semantically correct PTX accepted by
  ptxas + gate test", recommended fragment.rs internal sectioning,
  bounded `validate()` scope to target-capability-only.
- **Dave:** committed to pub fragment fields (no accessor), workflow
  preference for commit-try-iterate over upfront nvcc verification.

## Results

_(populated after execution)_
