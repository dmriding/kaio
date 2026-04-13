# Phase 7 Master Plan — Quantized Kernels & Training Integration

**Status:** In progress (7.0 complete; 7.1 planning pending)
**Depends on:** Phase 6 complete (v0.2.1, 2026-04-14)
**Current branch tip:** Sprint 7.0 complete on `phase7`. See PHASE_7_LOG
(to be created at 7.1 kickoff) for per-sprint commit hashes.
**Sprint order (as planned):** 7.0 → 7.1 → 7.2 → 7.3 → 7.4. May fold /
split as each sprint is scoped; 7.0 is the only fixed-scope sprint at
plan time.

## Goal

Bring quantized matmul (INT8 → INT4, potentially 4/6-bit LLM formats)
into `kaio-ops` as first-class `matmul_int8` / `matmul_int4`
operations, and land a `kaio-candle` bridge so downstream users can
use KAIO kernels through candle's `CustomOp` trait without manually
wiring the runtime.

Quant dequantize-matmul is the operation every local LLM runner
currently reaches for through CUDA C++ in GGUF / GPTQ / AWQ — Rust
implementations fall back to scalar compute or FFI. Phase 7 closes
that gap through pure Rust + PTX.

## Critical Architectural Constraints

**1. Quant kernels are IR-authored, not DSL-authored.** Like
tensor-core matmul in Phase 6, the dequantize-matmul inner loops live
in `kaio-ops` as IR-level kernels. The `#[gpu_kernel]` DSL is the
primary authoring surface for *end users*; `kaio-ops` is where
performance-critical primitives live.

**2. Sign preservation is correctness, not optimization.** INT8 and
INT4 quant schemes typically use signed packed storage. The arithmetic
right shift that reconstructs a signed value (`(packed >> shift) as
i8`) must use `shr.s32` — not `shr.u32`. Getting this wrong produces
silently wrong weights on negative values with no loud failure. Sprint
7.0 D1 (AD2) locks this in at the IR + macro + GPU-roundtrip level.

**3. Dequant must fuse with matmul.** Reading packed values into
registers and dequantizing in a separate pass (load-dequant → store
to shared → load from shared → matmul) destroys the point of
quantization. The dequant step must fuse with the tensor-core inner
loop — unpack → dequantize → feed mma.sync in registers, no
round-trip through shared or global.

**4. Training integration layers on top of the forward kernels, not
beneath them.** `kaio-candle` is a bridge crate in the same sense
that PyTorch extensions are — the framework (candle) handles the
computation graph and autograd; KAIO provides the forward and backward
CUDA kernels. Not autodiff, not a framework competitor.

## Key Architectural Decisions

### 1. DSL completeness (Sprint 7.0)

Bitwise operators (`&`, `|`, `^`, `<<`, `>>`, `!`) and short-circuit
logical operators (`&&`, `||`) were missing from the `#[gpu_kernel]`
DSL through Phase 6. 7.1+ quant work needs them on day one — dequant
is `(packed >> shift) & mask`; bounds-guarded array access is
`if i < n && arr[i] > 0`.

7.0 lands all of them with Rust-faithful semantics. `Shr` preserves
signed/unsigned distinction; `&&` / `||` short-circuit the RHS. Docs
closeout for Phase 6 rides along (`phases.md` promotion, Phase 7
scaffold, CHANGELOG).

### 2. INT8 dequantize-matmul (Sprint 7.1 — planned)

Target scheme: **symmetric INT8 per-tile scale** (simplest viable
scheme for a first implementation; GPTQ/AWQ come later).

Storage: `W_q: i8` weight tensor + `scale: f32` per-tile scale factor.

Fused kernel:
- Load packed INT8 weights (4× INT8 = one `u32` register via
  `MemoryOp::LdGlobalB128` — already landed in Phase 6.7b as
  unused-future-anchor).
- Unpack with `shr.s32` + `and.b32` (AD2-preserved signedness).
- Dequantize: `w = (w_q as f32) * scale`.
- Feed into existing `mma.sync.m16n8k16.f16.f16.f32` after `cvt.f16.f32`.

**Open question (for 7.1 planning):** does the `mma.sync` shape accept
signed INT8 operands directly? `mma.sync.m16n8k16.s8.s8.s32` exists on
Ampere+ (SM 8.0+) — if so, we may skip the dequant-to-f16 step
entirely for INT8 × INT8 matmul and only dequantize the output. Needs
investigation at 7.1 kickoff.

### 3. INT4 dequantize-matmul (Sprint 7.2 — planned)

Target scheme: **GPTQ-style symmetric INT4 with group quantization**.

Storage: packed 4-bit weights (`8 × INT4 = one u32`), group scales
(`f16` per 64 or 128 weights), optional zero points.

Dequant is more complex than INT8:
- Unpack 8 INT4 values from one `u32` via 8 different shift-and-mask
  operations.
- Signedness handling: INT4 is 4 bits; sign extension after
  unpacking is `(x << 28) >> 28` — another AD2 canary target
  (if signedness collapses, INT4 weights near the sign boundary go
  silently wrong).

Likely uses `mma.sync.m16n8k16.f16.f16.f32` after dequant → f16
(INT4 × INT4 is not directly supported by the m16n8k16 shape).

### 4. Quant + attention integration (Sprint 7.3 — planned)

Attention's QKV projections are the primary beneficiary of quantized
matmul (they dominate LLM inference cost). 7.3 wires the 7.1 / 7.2
quant matmul into the Phase 6 FlashAttention inner loop:

- Dequant-on-the-fly QKV projection
- Retains attention score accumulation in fp32
- Retains softmax + V multiplication in fp16 / fp32

### 5. `kaio-candle` bridge crate (Sprint 7.4 — planned)

New crate in the workspace: `kaio-candle`. Depends on both `kaio` and
`candle-core`. Implements `candle_core::CustomOp1` / `CustomOp2` /
`CustomOp3` for each exposed `kaio-ops` operation.

**Scope decision at 7.4 planning:** forward-only first, then backward
kernels in a follow-up sprint. `candle-nn` users get the forward
perf win immediately; training users wait for backward.

## Sprint Breakdown

| Sprint | Scope | Key Deliverable |
|--------|-------|-----------------|
| 7.0 | DSL completeness + Phase 6 closeout + Phase 7 scaffold | Bitops, short-circuit `&&`/`\|\|`, compound bitwise assign, phases.md Phase 6 promotion, phase7 master plan (this file) — v0.2.1 (combined with Sprint 6.10 in the same release) |
| 7.1 | INT8 dequantize-matmul | `kaio_ops::matmul_int8` symmetric INT8 × f16 → f32, perf target TBD at sprint kickoff |
| 7.2 | INT4 dequantize-matmul | `kaio_ops::matmul_int4` GPTQ-style packed 4-bit + group scales |
| 7.3 | Quant + attention integration | Dequant-on-the-fly QKV projection wired into FlashAttention |
| 7.4 | `kaio-candle` bridge crate | Forward-kernel `CustomOp` bindings for core `kaio-ops` operations |

## Dependency Graph

```
7.0 (DSL completeness) -> 7.1 (INT8 matmul) -> 7.2 (INT4 matmul)
                                                    |
                                                    v
                                               7.3 (quant + attention)
                                                    |
                                                    v
                                               7.4 (candle bridge)
```

7.0 blocks 7.1 because dequant is literally impossible without
bitops. 7.2 benefits from 7.1's infrastructure (dequant kernel
skeleton, tile-scale handling). 7.3 needs 7.1 or 7.2 landed. 7.4 is
additive — depends on whichever forward kernels exist at its
sprint kickoff.

## Success Criteria

1. `#[gpu_kernel]` DSL supports bitwise operators, short-circuit
   logical operators, and bitwise compound assignment with
   Rust-faithful semantics. Preserved signed vs unsigned distinction
   on right shifts, verified end-to-end. (7.0 ✅)
2. `kaio_ops::matmul_int8` produces bit-exact output vs a CPU
   reference dequant+matmul across a measured test suite. (7.1)
3. `kaio_ops::matmul_int4` produces output within fp16 tolerance vs
   a CPU reference dequant+matmul. Correct sign extension on INT4
   values near the sign boundary. (7.2)
4. Quantized attention (QKV projection quantized, softmax + V in
   fp16) produces output within LLM-inference-grade tolerance vs
   the fp16 reference attention. (7.3)
5. `kaio-candle` crate publishes to crates.io with forward-kernel
   `CustomOp` bindings for `matmul_tc`, `matmul_int8`, `matmul_int4`,
   and `attention_tc`. (7.4)

## Key Risks

1. **Signed/unsigned shift collapse (R-SHR).** If the parse →
   kernel_ir → lower → emit chain ever loses the distinction between
   `i32` and `u32` at the shift site, INT8/INT4 dequant silently
   produces wrong weights. Mitigated by 7.0's AD2 canary tests at
   every layer (IR emit, macro codegen regression, GPU round-trip
   with `i32 -2 >> 1 == -1` assertion).
2. **INT4 unpacking correctness.** 8 values per `u32` with sign
   extension is complex; one wrong shift count silently produces
   wrong results. Mitigation: CPU reference + exhaustive boundary-
   value tests in 7.2 (all 16 possible INT4 values per position).
3. **Dequant fusion register pressure.** Fusing unpack + dequant +
   mma.sync in the inner loop may push per-thread register count
   over the occupancy-tier threshold (64 regs/thread on sm_89 for
   128-thread CTAs). Mitigation: ptxas --verbose register-count
   check in 7.1 and 7.2 acceptance, split-kernel dispatch if
   occupancy drops.
4. **INT8 mma.sync shape discovery at 7.1 kickoff.** The
   `m16n8k16.s8.s8.s32` shape needs verification — if it doesn't
   behave as expected, 7.1 falls back to the dequant-to-f16 path
   which is slower but still shippable.
5. **candle API stability.** `candle-core::CustomOp` may change
   across candle versions. Mitigation (at 7.4 planning): pin a
   specific candle version; decide between supporting a single
   recent version or maintaining compat across a range.

## Review Context

Phase 7 scaffolding landed in Sprint 7.0 ahead of any
quantization-specific planning. The sprint-level plans for 7.1–7.4
will each receive the full planning + adversarial review cycle as
they kick off. This master plan establishes the guardrails (constraints
+ sign-preservation canary + fusion requirement) so sprint-level plans
can reference them without re-litigating.
