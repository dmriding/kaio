# Phase 6 Master Plan — Tensor Cores + Async Copies

**Status:** In progress (6.1–6.7 complete; 6.7b–6.9 pending)
**Depends on:** Phase 5 complete (v0.1.0, commit `bbc1c4d`)
**Current branch tip:** `a3d5ca3` (Sprint 6.7 complete — multi-warp TC matmul + promotion)

See [PHASE_6_LOG.md](PHASE_6_LOG.md) for per-sprint commit hashes
and test counts.

## Goal

Add fp16/bf16 type support, tensor core instructions (`mma.sync`),
and async memory copies (`cp.async`) to kaio-core. Implement a
tensor-core matmul in kaio-ops. Target 60%+ of cuBLAS sgemm
(up from 31% with scalar FMA).

## Critical Architectural Constraints

**1. IR kernels are internal performance primitives, not user-facing
authoring tools.** The `#[gpu_kernel]` DSL remains the primary
interface. IR-level tensor-core kernels live inside kaio-ops only.
Users are not expected to write mma kernels directly. If this
boundary blurs, KAIO becomes "DSL for toy kernels, IR for real
work" — that kills usability.

**2. Tensor cores are warp-collective.** `mma.sync` is executed by an
entire warp (32 threads) cooperatively. Register "fragments" are
distributed across warp threads with a rigid layout defined by
NVIDIA. This is fundamentally incompatible with the per-thread
scalar DSL.

**3. cp.async is pluggable, not foundational.** The tensor-core
kernel must work synchronously without cp.async (Sprint 6.3).
cp.async double buffering is layered on top (Sprint 6.4) as an
optimization for Ampere+ (SM 8.0+).

## Key Architectural Decisions

### 1. fp16/bf16 Types (Sprint 6.1)

Add `PtxType::F16` and `PtxType::BF16` to kaio-core. New register
kinds `H` (%h) and `Hb` (%hb). Type conversion instructions
(`cvt.rn.f16.f32`, `cvt.f32.f16`). Host-side: `half` crate for
`f16` type in `GpuBuffer<f16>`.

Keep type conversions explicit — no implicit fp16↔fp32 coercion.

### 2. mma.sync + Fragment Model (Sprint 6.2a)

For `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`:
- Fragment A (16×16 fp16): 8 fp16 values per thread (4 registers)
- Fragment B (16×8 fp16): 4 fp16 values per thread (2 registers)
- Fragment C/D (16×8 fp32): 4 fp32 values per thread (4 registers)

**Typed fragment structs** (`FragmentA`, `FragmentB`, `FragmentC`),
not raw register arrays. Prevents mapping bugs and accidental misuse.

**Standalone fragment test required:** Load known values, run one
mma.sync, validate against hand-computed results. Don't attempt a
full tiled matmul until single-instruction test passes.

### 3. cp.async Instructions (Sprint 6.2b)

Async global→shared memory copy for Ampere+ (SM 8.0+).
`cp.async.ca.shared.global`, `cp.async.commit_group`,
`cp.async.wait_group`. Added to `MemoryOp`.

Conceptually separate from mma.sync — don't debug fragment mapping
and async pipeline simultaneously.

### 4. Tensor-Core Matmul (Sprints 6.3-6.4)

Written via the IR API (PtxKernel, PtxInstruction), NOT the proc
macro DSL.

**Sprint 6.3 (SM 8.0+):** Basic tensor-core matmul using the
`m16n8k16.f16.f16.f32` shape. Start with manual shared memory
loads, not ldmatrix. Correctness first.

*SM threshold note:* The `m16n8k16` shape is Ampere-only. Volta
(SM 7.0) uses `m8n8k4` and Turing (SM 7.5) uses `m16n8k8`, each
with a different fragment layout. Phase 6 targets Ampere+ only.
Earlier shapes are out of scope.

**Sprint 6.4 (SM 8.0+):** Add cp.async double buffering. Kernel
works without cp.async; async pipeline is layered on top.

### 5. Three-Variant Auto-Tuner (Sprint 6.5)

Explicit eligibility rules:
- scalar: any SM
- tensor core without cp.async: SM 8.0+ AND fp16 inputs (m16n8k16 shape)
- tensor core with cp.async: SM 8.0+ AND fp16 inputs

No implicit assumptions, no runtime failure as discovery mechanism.

### 6. Performance Target

- Phase 4 baseline: 31% of cuBLAS (scalar FMA)
- Sprint 6.7 multi-warp result: **79.9% sync / 85.1% async** at 4096² ✅
  (well past the 60% target and 70% stretch)
- Sprint 6.7b target: 90%+ via vectorized loads (LDG.128) + bank-conflict
  padding
- cuBLAS uses tensor cores + vectorized loads + multi-stage pipeline

### 7. SM Version Requirements

| Feature | Minimum SM | Notes |
|---------|-----------|-------|
| mma.sync m16n8k16 (fp16) | SM 8.0 (Ampere) | Shape used by Phase 6. Volta (SM 7.0) uses `m8n8k4`; Turing (SM 7.5) uses `m16n8k8` — both out of scope. |
| cp.async | SM 8.0 (Ampere) | Async global→shared copy |
| mma.sync (bf16) | SM 8.0 (Ampere) | bf16 support |

## Sprint Breakdown

| Sprint | Scope | Key Deliverable |
|--------|-------|-----------------|
| 6.1 | fp16/bf16 types + conversions | PtxType::F16/BF16, RegKind::H/Hb, cvt, `half` crate |
| 6.2 | mma.sync + cp.async in kaio-core | TensorCoreOp, CpAsync, typed fragments, standalone mma test |
| 6.3 | Tensor-core matmul (IR API) | Basic mma.sync matmul, manual loads, SM 8.0+ (m16n8k16) |
| 6.4 | Double-buffered matmul | cp.async pipeline, SM 8.0+ |
| 6.5 | Integration + auto-tuner | 3-way dispatch (scalar / TC / TC+async) |
| 6.6 | TC attention (optional) | mma.sync in FlashAttention inner loops — skip if unstable |
| 6.7 | Multi-warp + edge tiles + benchmarks + promotion | 64×64 block tile, M/N divisibility lifted, 79.9% sync / 85.1% async cuBLAS sgemm at 4096², matmul_tc[_async] promoted to stable pub |
| 6.7b | Vectorized loads + bank-conflict padding | LDG.128, swizzle/padding, chasing 90%+ at 4096² |
| 6.8 | Showcase examples for v0.2.0 | Three standalone Cargo projects under `examples/` — fused SiLU-gate, GELU comparison, single-block RMSNorm. See [`sprint_6_8.md`](sprint_6_8.md). |
| 6.9 | Polish + v0.2.0 publish | CHANGELOG, README, version bump |

## Dependency Graph

```
6.1 (fp16/bf16) -> 6.2 (mma.sync + cp.async) -> 6.3 (TC matmul)
                                                       |
                                                  6.4 (double buffer)
                                                       |
                                                  6.5 (auto-tuner)
                                                       |
                                                  6.6 (TC attention — optional)
                                                       |
                                                  6.7 (benchmarks)
                                                       |
                                                  6.8 (showcase examples)
                                                       |
                                                  6.9 (publish)
```

## Success Criteria

1. kaio-core supports F16, BF16 types with register allocation
2. mma.sync emits valid PTX that passes `ptxas --verify`
3. cp.async emits valid PTX for SM 8.0+
4. Tensor-core matmul produces correct output vs CPU reference
5. Tensor-core matmul reaches 60%+ of cuBLAS at 4096×4096
6. Auto-tuner dispatches to tensor-core variant when available
7. Precision differences (fp16 vs fp32) documented clearly
8. All 5 crates published at v0.2.0

## Key Risks

1. **Fragment register mapping** — rigid layout defined by NVIDIA.
   Wrong mapping = silent corruption. Mitigation: standalone
   single-instruction test before full kernel.
2. **Mixed precision** — fp16 inputs → fp32 accumulation. Need
   appropriate test tolerances and user documentation.
3. **SM gating** — tensor-core path must not launch on SM < 8.0.
   Phase 6's `mma.sync.m16n8k16` and `cp.async` both require Ampere+
   (SM 8.0+). Earlier shapes (Volta `m8n8k4`, Turing `m16n8k8`) are
   out of scope. Both `matmul_tc` and `matmul_tc_async` enforce this
   via `PtxModule::validate()` through `KaioDevice::load_module`
   (centralized in Sprint 6.5 — the previous ad-hoc `device.info()`
   checks were removed). See tech_debt.md for the remaining macro-
   codegen migration off `load_ptx(&str)`.
4. **IR/DSL divergence** — if IR becomes "where real performance
   lives," DSL becomes "demo layer." Keep IR internal to kaio-ops.

## Review Context

- Opus 4.6: standalone fragment test, cudarc half verification,
  SM gating for three variants, manual loads before ldmatrix
- Codex 5.4: IR as internal-only, fragment abstractions not raw
  registers, cp.async pluggable not foundational, TC attention
  optional, precision documentation
