# Phase 7 Master Plan — Quantized Kernels & Training Integration

**Status:** In progress (7.0 complete v0.2.1; 7.0.5 complete v0.2.2;
7.1 INT8 dequantize-matmul complete v0.3.0; 7.1.5 / 7.2 / 7.3 / 7.4
planned)
**Depends on:** Phase 6 complete (v0.2.1, 2026-04-14)
**Per-sprint logs:** see
[`sprint_7_0.md`](sprint_7_0.md),
[`sprint_7_0_5.md`](sprint_7_0_5.md),
[`sprint_7_1.md`](sprint_7_1.md) for commit hashes and results. Older
sprints may reference a `PHASE_7_LOG` file that was never created —
the per-sprint logs serve that role.
**Sprint order (as planned):** 7.0 → 7.0.5 → 7.1 → 7.1.5 → 7.2 → 7.3 →
7.4. May fold / split as each sprint is scoped.

> **Note:** This document is the long-form Phase 7 planning reference.
> It captures architecture, risks, and sprint sequencing as they were
> understood at planning time. Authoritative state for shipped work
> lives in the per-sprint log files above and in
> [`docs/phases.md`](../../../phases.md). If this plan disagrees with
> a sprint log, the sprint log wins.

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

**Resolved (Sprint 7.1 D1):** `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`
works as expected on Ampere+ (SM 8.0+). Path FAST landed for v0.3.0:
INT8 flows directly into the tensor core, scale is applied only to the
s32 accumulator post-accumulation. DEQUANT-F16 fallback was never needed.

**Shape correction (Sprint 7.1 D1 pre-work):** Earlier drafts of this
doc referenced `mma.sync.m16n8k16.s8.s8.s32`. That was wrong. The INT8
mma shape on sm_80+ is `m16n8k32` (K = 32, not K = 16 — twice the
K-tile of the f16 path). The `.row.col` layout qualifiers are
mandatory in the instruction string, not optional. Sprint 7.1's plan
doc captures the full string.

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
| 7.0.5 | Pre-7.1 ergonomics fast-track | Debug-build performance note (A2), proc-macro error-span audit + 3 fixes (A4), consolidated `docs/debugging.md` (B3) — v0.2.2 |
| 7.1 | INT8 dequantize-matmul | `kaio_ops::matmul_int8` symmetric INT8 × f16 → f32, perf target TBD at sprint kickoff. **Folds in integer-arithmetic DSL support** — mixed-width `u8×u8→u16` / `i8×i8→i32` is a 7.1 prerequisite, not a separate deliverable |
| 7.1.5 (planned) | Warp + block reductions in DSL | Convergence-doc D1/D2 — `warp_reduce_sum/max/min`, `block_reduce_sum/max` as `#[gpu_kernel]` builtins. Standalone between-quant-milestones sprint |
| 7.2 | INT4 dequantize-matmul | `kaio_ops::matmul_int4` GPTQ-style packed 4-bit + group scales |
| 7.3 | Quant + attention integration | Dequant-on-the-fly QKV projection wired into FlashAttention |
| 7.4 | `kaio-candle` bridge crate | Forward-kernel `CustomOp` bindings for core `kaio-ops` operations |

## Dependency Graph

```
7.0 (DSL completeness) -> 7.0.5 (ergonomics) -> 7.1 (INT8 matmul)
                                                    |
                                                    +-> 7.1.5 (reductions, optional between quant milestones)
                                                    |
                                                    v
                                               7.2 (INT4 matmul)
                                                    |
                                                    v
                                               7.3 (quant + attention)
                                                    |
                                                    v
                                               7.4 (candle bridge)
```

7.0 blocks 7.1 because dequant is literally impossible without
bitops. 7.0.5 is a narrow pre-7.1 ergonomics sprint — not blocking on
correctness, but on the adoption-friction argument described below.
7.2 benefits from 7.1's infrastructure (dequant kernel skeleton,
tile-scale handling). 7.3 needs 7.1 or 7.2 landed. 7.4 is additive —
depends on whichever forward kernels exist at its sprint kickoff.

## Adoption-ergonomics sequencing

Sprint 7.0 ship coincided with external feedback converging on a thesis:
*trust and ergonomics gate adoption more than feature count*. The
feedback flagged a set of potential items across trust-and-first-
impressions (module caching, debug-mode guardrails, error-message
quality, first-run path), discoverability (feature matrix, kernel
programming model doc, debugging guide, example quality), runtime
ergonomics (higher-level buffer API, launch syntax, struct args), and
kernel primitives (warp / block reductions, atomics, integer
arithmetic, dynamic shared memory).

The thesis has real weight. The individual items need to be weighted
against actual codebase state and sprint cost. This plan treats the
feedback as *signal* — informs priority, does not dictate scope. Here
is where each flagged item lands:

### Folded into Sprint 7.0.5 (pre-7.1 ergonomics fast-track)
- **Debug-build performance note** — one-time stderr warning in `KaioDevice::new` when running in a debug build. Prevents the common "benchmarked in debug, bounced" adoption failure.
- **Proc-macro error spans** — audit of 63 error sites; 3 identified as potentially improvable. Implementation showed the fixes are defensive (unreachable paths and call-site-equivalent fallbacks) — the honest finding is that the macro's span handling is already quite good. Documented as a finding rather than a deliverable win.
- **Consolidated debugging guide** — new `docs/debugging.md` aggregating scattered env-var docs, `compute-sanitizer` usage, tolerance guidance, troubleshooting flowchart.
- **`cargo xtask` repo tooling** — `cargo xtask showcase` / `bench` / `all` from the repo root. Load-bearing for first-impression UX. Emerged during the sprint as the cargo-native pivot away from any justfile-as-first-impression approach.
- **docs.rs explicit target list** — preserves Windows visibility on docs.rs after its 2026-05-01 default-target change.

### Folded into 7.1 INT8 matmul scope (not a separate item)
- **Integer arithmetic in the DSL** — mixed-width `u8×u8→u16` and `i8×i8→i32` are INT8 matmul prerequisites. Treating as a separate sprint-level deliverable would falsely imply 7.1 can happen without them.

### Planned as standalone sprint (7.1.5 or 7.2.5)
- **Warp-level reductions** (`warp_reduce_sum`, `warp_reduce_max`, `warp_reduce_min` + raw `shfl_sync_*`)
- **Block-level reductions** in DSL (builds on warp reductions + shared memory + `bar_sync`)

Both are substantial enough to deserve their own sprint. Sequenced between quant milestones rather than inside them to keep the quant story cohesive and the reductions sprint focused.

### Deferred, with rationale
- **Persistent module cache** — Sprint 6.10 D1a *intentionally* removed `OnceLock<String>` PTX caching so user kernels flow through `PtxModule::validate()` on every launch. Reintroducing a cache is not "add a cache" — it's "design a cache that doesn't re-open the trust-boundary gap." Needs a dedicated design sprint. Post-7.1.
- **Feature / compatibility matrix doc** — living docs rot. Ride along with each sprint's DSL additions; pair with authoritative `trybuild` compile-fail fixtures. Not a single-sprint deliverable.
- **Kernel programming model doc** — partly covered by existing `docs/implementation.md`; expand incrementally.
- **Example-quality upgrade (tutorial-style commentary)** — continuous improvement, not a single-sprint deliverable.
- **Higher-level buffer API (`GpuTensor`)** — post-7.1. Natural interop point for the 7.4 candle bridge. Keep thin, don't build a tensor library.
- **Atomics** — real feature but not on the quant critical path. Standalone sprint when a quant / histogram / scatter-add kernel needs it.
- **Dynamic shared memory** — fold into 7.1 or 7.1.5 if the INT8 matmul tile sizes need runtime configuration (probably at auto-tuner time, not day-one INT8).

### Rejected
- **Builder-pattern launch syntax** — feature-negative. The current `kernel::launch(&device, &x, &mut y, n)?` IS the minimal form. Adding `.grid()` + `.block()` + `.args()` + `.run()` for the same action is more boilerplate, not less. Grid/block dims are already declared in the `#[gpu_kernel(block_size = N)]` attribute; re-specifying them at launch site is error-prone duplication.
- **`repr(C)` struct kernel arguments** — Phase 8+ at earliest. Deeper macro change than the current parameter-passing model; evaluate complexity vs payoff later.
- **Adopting the full set of ergonomics feedback upfront** — too long a delay before quant. Three narrow items (debug-mode note, error spans, debugging guide) fit in a pre-7.1 sprint (7.0.5); more than that is an ergonomics phase masquerading as a sprint.

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
4. **INT8 mma.sync shape discovery at 7.1 kickoff. — RESOLVED.**
   `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` works on sm_80+
   as expected. Path FAST shipped in v0.3.0 (Sprint 7.1); DEQUANT-F16
   fallback was specced but never activated.
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
