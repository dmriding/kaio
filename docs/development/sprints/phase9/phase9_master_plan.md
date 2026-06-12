# Phase 9 Master Plan — Attention Backward & Kernel Deepening

**Status:** Planned.
**Depends on:** Phase 7 (v0.4.0) + Phase 8 (PyO3 scaffold landed; pointer
syntax delivered as Sprint 8.0 prelude).
**Per-sprint logs:** see the `sprint_9_*.md` files in this directory and
the per-sprint status row in [`PHASE_9_LOG.md`](PHASE_9_LOG.md) for
commit hashes and shipped outcomes.
**Sprint order (planned):** 9.1 → 9.2 → 9.3.
**Release target:** Phase 9 aggregate → **v0.5.0** (workspace crates;
`kaio-candle` to 0.2.0; `kaio-py` unchanged unless touched). No
mid-phase publishes unless a sprint independently warrants one.

> **Note:** This document is the long-form Phase 9 planning reference.
> It captures architecture, risks, and sprint sequencing as they were
> understood at planning time. Authoritative state for shipped work
> lives in the per-sprint log files above and in
> [`docs/phases.md`](../../../phases.md). If this plan disagrees with a
> sprint log, the sprint log wins.

## Goal

Phase 9 does two things:

1. **FlashAttention backward pass.** Ship `attention_flash_bwd` +
   `attention_flash_causal_bwd` through new PTX kernels (softmax
   recomputation from saved statistics, tiled Q/K/V backward, causal
   mask in reverse). Wire them into `kaio-candle` as the Tier-2
   backward scheduled out of Sprint 7.4d. This unblocks single-head
   FlashAttention backward and establishes the candle autograd
   integration pattern for new-PTX-kernel bwd ops. Multi-head
   orchestration stays out of scope.
2. **Kernel deepening for the existing TC matmul family:**
   - `matmul_tc_bf16` / `matmul_tc_bf16_async` — second precision
     variant of the existing TC matmul family, using the already-landed
     IR bf16 `mma.sync` path.
   - `ldmatrix.sync.aligned` — warp-collective fragment loader that
     replaces the hand-rolled fragment-A loader in `matmul_tc`. The
     remaining structural lever for closing the sync-vs-async gap
     documented in [`docs/performance.md`](../../performance.md)
     §"Path to higher throughput".

Combined, Phase 9 closes the training story for the candle bridge and
closes the precision / sync-path-throughput story for inference.

## Critical Architectural Constraints

**1. Attention backward requires new PTX kernels — not forward-kernel
reuse.** Unlike matmul (`dA = grad @ B^T`, `dB = A^T @ grad` reuses
forward), FlashAttention backward has no forward-reuse identity. It
needs bespoke kernels that recompute softmax from saved statistics,
accumulate `dQ`, `dK`, `dV` against the recomputed `P` matrix, and
apply causal masking in reverse. Sprint 7.4d explicitly deferred this
to Phase 9 because forward-reuse was not available.

**2. Forward kernel must expose row-wise softmax statistics for backward
to function without a 2× compute penalty.** Standard FA2 convention:
save the logsumexp `L_i = m_i + log(l_i)` as one `f32` per query row
(not `m` and `l` separately). Backward recomputes `P_ij = exp(S_ij −
L_i)` from `L` and `S` (which is rebuilt from `Q`, `K`). This keeps
backward memory at `O(seq_len)` per-head and avoids storing the `P`
matrix.

**3. bf16 is a fresh kernel family, not a template swap.** The IR side
of bf16 `mma.sync` is already wired (`tensor_core.rs` carries an
`mma.sync.m16n8k16.f32.bf16.bf16.f32` emit path; `RegKind::Hb` exists).
What is new in Phase 9: picking bf16 as the operand dtype in the
`kaio-ops` TC matmul fragment-load path, handling the host-side dtype
via `half::bf16`, and threading bf16 through the benchmark harness.
The kernel file largely mirrors `matmul_tc_kernel.rs` — bf16 fragment
loaders, bf16 `cvt` on the load side, same `.f32` accumulator. Same
multi-warp 64×64 block tile, same bank-conflict-padded Tile B, same
Sprint 6.7b D10 fragment-loader hoist.

**4. `ldmatrix.sync.aligned` is a warp-collective op — new IR primitive,
same op family as `mma.sync`.** Unlike `cp.async` (a memory op),
`ldmatrix` is a tensor-core-adjacent primitive that cooperatively loads
a fragment-shaped slice from shared memory directly into the register
layout `mma.sync` expects, across all 32 threads of a warp. Adding it
means (a) new `TensorCoreOp::LdMatrix` variant, (b) a validation gate
at emit time (SM 7.5+ baseline; specific shape/dtype combinations
permit different forms), (c) ptxas-verify coverage at the IR level
before any `kaio-ops` kernel changes. Only then do we wire it into
`matmul_tc`'s fragment-A (and potentially fragment-B) loader.

**5. Candle bridge bwd for attention is the first kaio-candle bwd op
with a new PTX kernel behind it.** Previous bwd ops (Sprint 7.4d)
reused forward kernels. Sprint 9.2 exercises the full
`CustomOp::bwd()` integration path including stream sync between
multiple new PTX launches inside a single bwd call. Test coverage
must include gradient correctness against a CPU analytical reference,
not just shape / type parity.

**6. FA backward inherits the f32 scope of the existing flash forward.**
Current `attention_flash` / `attention_flash_causal` are `f32` Q/K/V
→ `f32` out by construction (Sprint 5.4). Phase 9 bwd is also
`f32`-only. A bf16/f16 FA variant (forward + backward) is a legitimate
future extension but explicitly out of scope here — it would double
the kernel count and force dtype-generic fragment handling before the
correctness baseline is proven.

**7. FA backward is self-attention only in 9.2.** The
`attention_flash_bwd` signature takes a single `seq_len`, which means
Q/K/V all have the same sequence dimension — this is self-attention,
not cross-attention. Cross-attention bwd (distinct `seq_q` vs
`seq_kv`) is deferred to a future sprint if demand materializes; it
requires a different kernel structure because the causal-mask-in-
reverse predicate changes shape.

## Key Architectural Decisions

### 1. Sprint sequencing

Three themes, three sprints. Recommended ordering:

- **9.1 bf16 TC matmul** — warm-up; lowest risk, fastest to ship, proves
  Phase 9 velocity.
- **9.2 FlashAttention backward** — main event; biggest deliverable,
  highest user value.
- **9.3 `ldmatrix.sync.aligned`** — closing perf sprint; improves the
  existing `matmul_tc` sync path without changing public API.

Rationale:

- bf16 is the lowest-risk sprint because the IR path exists and the
  kernel is structurally a near-copy of `matmul_tc`. It ships a
  user-visible feature in under a week and gives `kaio-candle`
  something to add a bf16 binding to (small follow-on in the same
  sprint). If the sprint goes long, the surface area is small enough
  that scope can be narrowed (sync-only, no async variant) without
  breaking the phase.
- FA bwd is the hardest sprint in the phase by a wide margin. It lands
  in the middle so it has both: (a) momentum from a completed bf16
  sprint, (b) calendar headroom for any escape hatches to play out.
- ldmatrix closes the phase because it is an optimization of an existing
  shipped op (low risk to the public surface), and its benefit
  compounds with whatever bf16 gains arrived in 9.1 — ldmatrix improves
  both f16 and bf16 TC sync paths, so landing it after bf16 means one
  optimization sprint lifts two kernels.

**Alternative orderings considered:**

- **FA bwd first, bf16 / ldmatrix after.** Tempting because FA bwd is
  the highest-user-value deliverable. Rejected: FA bwd has the highest
  slip risk; starting the phase with its hardest sprint burns the
  escape-hatch budget on the first deliverable.
- **ldmatrix first.** Would lift f16 numbers immediately so bf16
  inherits the uplift. Rejected: ldmatrix uplift on the sync path is
  not yet measured; the most-mechanical sprint should not be gated on
  a speculative result.

### 2. bf16 TC matmul API shape (9.1)

**Decision:** separate public ops `matmul_tc_bf16` /
`matmul_tc_bf16_async`, not an overload of `matmul_tc`. Matches the
`matmul_int8` / `matmul_int4` separation precedent from Phase 7.

**Rationale:** Rust's lack of function overloading makes a single-name
approach either generic-over-dtype (large refactor, not in scope) or
runtime-dispatch (loses compile-time type safety on the
`GpuBuffer<half::f16>` vs `GpuBuffer<half::bf16>` distinction).
Separate ops keep each kernel self-contained and its error paths
obvious.

**Auto-tuner:** `matmul_auto_tc_bf16` is a 2-way selector between
`matmul_tc_bf16` (sync) and `matmul_tc_bf16_async` (async) only. No
scalar bf16 fallback; bf16 on pre-Ampere GPUs returns `SmTooLow`
through the existing validation path. Cache file keyed separately from
the f16 tuner cache so bf16 tuning does not contaminate f16 cache
entries. The auto-tuner is itself an optional 9.1 extension —
deferred if either candidate kernel slips.

**Candle bridge:** `kaio_candle::MatmulTcBf16Op` and
`MatmulTcBf16AsyncOp` follow the `CustomOp2` pattern. Forward + bwd
via forward-reuse (same pattern as 7.4d) — small enough that 9.1 can
ship both inside one sprint, or split to a 9.1.5 if validation time
runs long.

### 3. FA-backward forward-stats save pattern (9.2)

**Decision:** add a `_with_stats` variant of each existing flash
forward. The existing API stays identity-compatible.

- `attention_flash_with_stats(q, k, v, out, stats: &mut GpuBuffer<f32>,
  seq_len, d_k)` — identical to `attention_flash()` except the forward
  kernel also writes one `f32` per query row to `stats` (the logsumexp
  `L_i = m_i + log(l_i)`).
- `attention_flash_causal_with_stats(...)` — same, causal variant.
- `attention_flash()` / `attention_flash_causal()` public functions
  stay exactly as shipped in Sprint 5.4. No API break, no perf
  regression for users that do not need bwd.

The bwd function signature takes the saved stats as input:

```rust
pub fn attention_flash_bwd(
    device: &KaioDevice,
    grad_out: &GpuBuffer<f32>,
    q: &GpuBuffer<f32>,
    k: &GpuBuffer<f32>,
    v: &GpuBuffer<f32>,
    out: &GpuBuffer<f32>,
    stats: &GpuBuffer<f32>,       // saved L from forward
    dq: &mut GpuBuffer<f32>,
    dk: &mut GpuBuffer<f32>,
    dv: &mut GpuBuffer<f32>,
    seq_len: u32,
    d_k: u32,
) -> Result<()>;
```

(and `attention_flash_bwd_causal` variant with the same shape.)

**Alternative rejected:** saving `(m, l)` separately (two `f32` per
row). FA2 found `L = m + log(l)` is all that is needed for bwd; halves
saved-stats memory and simplifies the bwd recomputation to a single
per-row scalar.

**Alternative rejected:** no saved stats, bwd fully recomputes. ~2×
compute cost on backward (forward rerun inside bwd). For a sprint
whose whole point is training integration, the tradeoff sits clearly
on the side of one extra `seq_len * f32` buffer.

**Candle bridge fwd semantics.** `AttentionFlashOp::fwd()` in
kaio-candle calls the `_with_stats` variant unconditionally for every
forward invocation inside a candle graph. Stats buffer is allocated
and filled whether or not bwd ends up running. One extra `seq_len *
f32` allocation per forward is a rounding error vs the activation
memory the forward already produces (`seq_len * d_k * f32` for `out`
is at minimum `d_k`× larger; at `d_k = 128` it is 128× larger). The
dispatch-on-requires_grad alternative adds a branch and two code paths
for a saving the user will never notice.

**Stats buffer shape.** The invariant is "one `f32` per (batch_item,
head, query_row)", not "one `f32` per query row." In 9.2's single-head
self-attention scope that collapses to `[seq_len]` flat, but code and
docs phrase the contract generally so a future multi-head / batched
variant does not have to rename the API or reshape the buffer.

**Stats ownership across candle's CustomOp.** The `_with_stats` fwd →
`_bwd` handoff is trivial when both calls are direct Rust — bwd takes
the stats buffer as an explicit argument. Inside candle's autograd
graph it is not trivial: `CustomOp::bwd()` receives inputs / output /
grad_output from candle, and it is not guaranteed that kaio-candle can
freely stash a custom saved intermediate that bwd can retrieve later.
Options the 9.2 sprint plan must pick between, **before any kernel
code is written:**

1. **Recompute `L` inside `::bwd()`** — walk the forward kernel once
   more at bwd time to rebuild stats. Doubles the forward compute
   inside bwd but requires no saved-state plumbing. Simplest; the
   right default if candle's CustomOp API has no saved-intermediate
   story.
2. **Return stats as a hidden internal output of `::fwd()`** — candle
   gives bwd the fwd outputs, so if stats rides along as a second
   output tensor it flows back. Requires the op to be a multi-output
   CustomOp (candle's trait API permitting — audit required).
3. **Stash on op state** — put stats on the `AttentionFlashOp` struct
   between fwd and bwd. Candle may build / use / discard op instances
   at any granularity, so this needs explicit lifetime + thread-safety
   analysis; risky.

The first commit of 9.2's candle bridge work decides between these
options after a direct read of whatever `candle-core` version is
audited under R-KAIO-CANDLE-API-DRIFT. The decision is captured in the
9.2 sprint plan, not left as an implementation detail.

### 4. FA-backward tile structure (9.2)

**Decision:** FA2-style two-kernel bwd. `flash_attn_bwd_dkdv_kernel`
iterates outer-over-K-blocks / inner-over-Q-blocks to accumulate `dK`,
`dV`. `flash_attn_bwd_dq_kernel` iterates outer-over-Q-blocks /
inner-over-K-blocks to accumulate `dQ`. No atomic-add single-pass
variant in 9.2.

**Rationale:** two kernels with opposite loop nests is the standard
FA2 approach and avoids atomic contention on either `dQ` or `dK`/`dV`.
Each kernel is self-contained and independently testable. Single-pass
atomic-add is a potential follow-up optimization if perf justifies it
— but first we need a correct baseline.

**Mask handling:** causal-mask-in-reverse is the same predicate as
forward (`j ≤ q_row`) applied inside both bwd kernels. Non-causal bwd
skips the mask check entirely.

**BLOCK_M for bwd:** start at 16 (BLOCK_M = 1 from forward is too small
for bwd throughput; each bwd kernel has more per-block work than fwd).
Re-evaluate against roofline at the correctness-first milestone.
Capture the knob as a tunable constant, not a runtime arg.

**Register-pressure gate.** The dQ kernel's inner loop recomputes
`S_ij = Q_i · K_j^T` per K-block, holding simultaneously: the Q tile,
a partial S tile, the dQ accumulator, and rescaling scalars from the
saved `L`. At `d_k = 128` this combination may push per-thread
register count past Ampere's 128-reg/thread occupancy cliff (drops to
1 block/SM, loses latency hiding). Same class of issue Sprint 7.3.5
hit with INT4 S+½P. Quality gate in 9.2: `ptxas -v` register count
checked at each correctness milestone; if a bwd kernel exceeds ~100
regs/thread, rework tile sizes or register reuse before the perf
milestone. Applies to both bwd kernels but dQ is the higher-risk one.

### 5. `ldmatrix.sync.aligned` IR primitive (9.3)

**Decision:** new `TensorCoreOp::LdMatrix` variant. Co-located with
`MmaSync` because it is a warp-collective op that feeds `MmaSync` and
shares the fragment-register layout semantics. Not a `MemoryOp` — the
fragment-register distribution and warp-collective nature belong on
the tensor-core side of the IR.

**Variants to support in 9.3:**

- `ldmatrix.sync.aligned.m8n8.x4.shared.b16` — load four 8×8 b16
  matrices → four b32 registers per thread. Matches `FragmentA` layout
  for `m16n8k16.f16`. Replaces the current hand-rolled fragment-A
  loader.
- `ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16` — transposed
  2-matrix form. May or may not wire into fragment-B this sprint;
  decision gated on register-pressure analysis once the primary
  fragment-A replacement is landed.

**Validation gate.** `LdMatrix` tentatively adds `.min_sm() = 75`
(Turing+). The exact minimum-SM value and the set of permitted
form × dtype combinations (f16 vs bf16; `.trans` qualifier; `.x1` /
`.x2` / `.x4` width) must be verified against the current PTX ISA
docs plus a `ptxas --verify` probe on each emitted variant at 9.3a
kickoff. If bf16 `ldmatrix` requires SM 8.0+ while f16 is available on
SM 7.5+, the `.min_sm()` value becomes variant-dependent — the
validation gate has to reflect that.

**Alternative considered:** compose from existing primitives
(`LdShared` per-thread with reshuffle via `shfl.sync`). Rejected: not
semantically equivalent at the warp-cooperative level; software-
emulated ldmatrix is uncompetitive on perf; the whole point of the
primitive is hardware-level register-layout dance.

### 6. Candle bridge updates

**9.1:** `MatmulTcBf16Op` / `MatmulTcBf16AsyncOp` forward + bwd
(forward-reuse pattern — same as Sprint 7.4d).

**9.2:** `AttentionFlashOp::bwd()` + `AttentionFlashCausalOp::bwd()`.
First kaio-candle bwd ops with new PTX behind them. Each bwd call
launches the two bwd kernels (dkdv + dq) in sequence then returns
gradient tensors. Stream sync inside a single bwd call reuses the
event-based pattern from Sprint 7.4c.

**9.3:** no bridge changes. ldmatrix is a kernel-internal optimization;
public TC matmul API does not change.

**Deferred to post-Phase-9:** `attention_tc` / `attention_tc_causal`
bwd. These are short-seq inference ops; training users route to
`attention_flash` (no `seq_k ≤ 384` cap). Adding bwd to `attention_tc`
is symmetry rather than user demand; tracked behind an issue if user
demand materializes.

### 7. Bench coverage for FA backward

No cuDNN MHA reference (same constraint as forward —
`cudarc` 0.19 does not expose `cudnnMultiHeadAttnBackward` cleanly).
The FA bwd benches report:

- Absolute bwd latency vs forward at matched `(seq_len, d_k)` shapes.
- bwd/fwd ratio (expected ≈ 2.5–3× for a well-tuned implementation).
- 10-run worst / median / best protocol, same as every other table in
  `docs/performance.md`.

**Correctness oracle stack** (defined in R-FA-BWD-NUMERICS):

1. **`seq_len = 1` closed-form gate.** Trivial smoke before anything
   else. At `seq_len = 1`, `O = V` (softmax of one element is `1.0`)
   and `dS = 0` because softmax is constant. Therefore `dV =
   grad_out`, `dQ = 0`, `dK = 0`. Gates the bwd wiring before any
   tiled-softmax-recomputation complexity.
2. **CPU f64 analytical bwd reference.** Naive no-tiling attention
   backward in f64 on host, compared against the GPU kernel within
   `rel_err < 1e-2 || abs_err < 1e-3` at `seq_len ∈ {32, 64, 128}`,
   `d_k ∈ {32, 64, 128}`. Bit-close is the target; FA bwd has an
   inherent reduction-order rearrangement vs naive, so bitwise
   identity is not achievable nor desired.
3. **Finite-difference smoke checks.** 1–2 tiny shapes
   (`seq_len = 8, d_k = 16`) as a sanity layer on top of the
   analytical oracle. Not the primary gate.

### 8. Version target + v0.5.0 phase gate

**Phase aggregate:** v0.5.0 on phase close. Workspace crates (`kaio` /
`kaio-core` / `kaio-macros` / `kaio-ops` / `kaio-runtime`) bump to
0.5.0; `kaio-candle` to 0.2.0; `kaio-py` unchanged unless touched.

**Intra-phase:** Sprints 9.1 / 9.2 / 9.3 accumulate `[Unreleased]`
changelog entries. No mid-phase publish. Dedicated adversarial review
on the v0.5.0 commit series per the project's version-bump rule.

**Required for v0.5.0 (hard gate):**

- `attention_flash_with_stats` + `attention_flash_causal_with_stats`
  forward variants with saved `L`.
- `attention_flash_bwd` + `attention_flash_causal_bwd` kernels with
  CPU f64 analytical correctness.
- `kaio_candle::AttentionFlashOp::bwd` + causal variant integrated
  into candle autograd.
- Performance documentation (bwd/fwd ratio + absolute ms in
  `docs/performance.md`).
- Standard documentation hygiene (changelog, README, phase log,
  coverage refresh).

**Optional for v0.5.0** (ship if clean, defer if not):

- bf16 TC matmul family — full or partial (see 9.1 scope below).
- `ldmatrix` optimization — IR primitive alone is acceptable; loader
  rewire optional.
- kaio-candle bf16 bindings.
- bf16 auto-tuner cache.

If 9.2 lands cleanly but 9.3 or the bf16 family runs long, v0.5.0
ships with what's landed. The phase is named "Attention Backward &
Kernel Deepening"; the attention-backward half is non-negotiable, the
kernel-deepening half is scoped by calendar.

## Sprint Breakdown

| Sprint | Scope | Key Deliverable | Gate |
|---|---|---|---|
| 9.1 | bf16 TC matmul family | **Minimum shippable slice:** `matmul_tc_bf16` sync + correctness + bench. **Extensions, each gated on the previous landing cleanly:** bf16 async → bf16 auto-tuner → kaio-candle bf16 fwd → kaio-candle bf16 bwd (forward-reuse). If any extension pushes the sprint past one review cycle over estimate, that extension is deferred to the Unscheduled list and 9.2 starts | Optional for v0.5.0 |
| 9.2 | FlashAttention backward | `attention_flash_with_stats` + causal variant (fwd adds L export); `attention_flash_bwd` + causal (two new PTX kernels: dkdv + dq); candle bridge `AttentionFlashOp::bwd` + causal; CPU f64 analytical correctness suite + closed-form `seq_len=1` gate + FD smoke checks | **Required for v0.5.0** |
| 9.3 | `ldmatrix.sync.aligned` | **9.3a (MVS):** new `TensorCoreOp::LdMatrix` IR variant + validation + ptxas-verify tests. **9.3b (extension):** rewire `matmul_tc` fragment-A loader; measure sync-path effect; bench refresh. 9.3a and 9.3b ship as one sprint if clean; split at kickoff if scope feels heavy | 9.3a optional for v0.5.0; 9.3b also optional — IR primitive is useful even without the kernel rewire |
| v0.5.0 | Phase 9 aggregate release | Workspace crates to 0.5.0; `kaio-candle` to 0.2.0; `kaio-py` unchanged unless touched. Coverage refresh; adversarial review pass | Gate: 9.2 must have landed; 9.1/9.3 ship what landed |

Optional sub-sprints (reserved, unscheduled):

- **9.1 deferred extensions** — bf16 async / auto-tuner / candle
  bindings individually park to the Unscheduled list if 9.1 pushes
  over estimate. Each is independently resumable post-v0.5.0 without
  revisiting the sprint plan.
- **9.2.5** — single-pass atomic-add dQ bwd kernel if two-kernel perf
  is meaningfully worse than a naive FA2 estimate. Post-v0.5.0.
- **9.3.5** — `ldmatrix` for fragment-B (transposed form). Gated on
  9.3b outcome + register-pressure budget. Post-v0.5.0.

## Dependency Graph

```
9.1 (bf16 TC matmul) -> 9.2 (FA backward) -> 9.3 (ldmatrix)
     |                       |                    |
     | (optional fwd-reuse   | (AttentionFlash    | (matmul_tc
     |  bwd in same sprint)  |  ::bwd lands in    |  fragment-A
     |                       |  kaio-candle)      |  loader rewire;
     v                       v                    |  no public API
     matmul_tc_bf16          attention_flash      |  change)
     + candle bwd            _bwd + candle bwd    v
                                                 no kaio-candle
                                                 changes
```

9.2 does not technically depend on 9.1 — the FA bwd kernels are
independent of the bf16 TC matmul family. Sequencing 9.1 → 9.2 is
about phase momentum (warm-up → main event), not hard dependency. If
9.1 hits an unexpected blocker and 9.2 is unblocked, 9.2 can move
first without replan.

9.3 depends on 9.1 landing if we want the ldmatrix uplift to apply to
both kernel families simultaneously (both matmul_tc families share
fragment-loader structure). If 9.3 runs first, the bf16 sprint
inherits the improved loader. Either order is admissible; the
recommended order means one optimization sprint lifts both kernels at
once.

## Success Criteria

1. `matmul_tc_bf16` produces output within bf16 tolerance (typically
   `rel_err < 1e-2`) vs a CPU f64 reference matmul across ≥ 3 shape
   classes (small / medium / large). Tensor-core compute density at
   4096³ matches f16 within measurement noise. (9.1)
2. `attention_flash_with_stats` produces outputs and saved-`L` values
   identical (within fp tolerance) to the existing `attention_flash`
   out values; the L buffer decoded via the property test
   `(exp(S - L)).sum() == 1` per row holds across shapes. (9.2)
3. `attention_flash_bwd` + causal variant produce `dQ`, `dK`, `dV`
   that match the CPU f64 analytical attention backward reference
   (primary oracle per R-FA-BWD-NUMERICS) within `rel_err < 1e-2 ||
   abs_err < 1e-3` for `seq_len ∈ {32, 64, 128}`, `d_k ∈ {32, 64,
   128}`. The `seq_len = 1` closed-form gate passes as a prerequisite
   (`dV = grad_out`, `dQ = 0`, `dK = 0`). Finite-difference smoke
   checks pass on a 1–2-shape tiny subset but are not the primary
   correctness gate. (9.2)
4. `kaio_candle::AttentionFlashOp::bwd()` integrates into a candle
   autograd graph without panic or dtype mismatch. Round-trip test:
   forward → `sum_all` → backward → compare against the CPU f64
   analytical oracle. (9.2)
5. `ldmatrix.sync.aligned` IR primitive passes ptxas-verify across all
   emitted variants on sm_75 and sm_80. (9.3)
6. `matmul_tc` sync-path worst-of-10 ratio at 4096³ vs cuBLAS sgemm
   shows measurable improvement above the bench noise floor after
   ldmatrix rewire, with ≥ 3 percentage points as the stretch target.
   Ship gate is non-regression (sync path does not get slower at any
   measured shape) + correctness; uplift magnitude is an outcome, not
   a gate. (9.3)
7. v0.5.0 publishes cleanly: the five workspace crates (`kaio` /
   `kaio-core` / `kaio-macros` / `kaio-ops` / `kaio-runtime`) at
   0.5.0; `kaio-candle` at 0.2.0; `kaio-py` unchanged unless phase 9
   added a user-requested op. Coverage badge refreshed, CHANGELOG
   moved from `[Unreleased]`, and adversarial review sign-off on the
   publish commit series. (Phase close)

## Key Risks

1. **R-FA-BWD-STATS — forward-kernel save of L underflows / overflows.**
   `L = m + log(l)` with `l = Σ exp(S - m)` can blow up or collapse in
   the tail of long sequences. FA2 addresses this by keeping `l ≥ 1`
   invariant (the first valid `s` always contributes `exp(s - m) =
   1`). Mitigation: test `_with_stats` vs a CPU reference computing
   `L` in f64; assert `rel_err < 1e-5` on `L` across all shapes in the
   test matrix.

2. **R-FA-BWD-NUMERICS — flash bwd numerical drift.** Unlike matmul
   bwd's clean `dA = grad @ B^T`, FA bwd recomputes P from saved L and
   accumulates gradient products with careful rescaling. One wrong
   sign or missed rescale produces results that are "close" to correct
   but systematically off in a way only a precise oracle catches.

   **Correctness oracle stack:** the primary oracle is a CPU f64
   analytical attention backward (not finite differences) because FD
   on attention is noisy (double numerical derivative over an
   exp-and-sum) and slow at any interesting shape. Stack, in order:

   1. `seq_len = 1` closed-form gate (`dV = grad_out`, `dQ = 0`,
      `dK = 0`).
   2. CPU f64 analytical bwd reference at `seq_len ∈ {32, 64, 128}`,
      `d_k ∈ {32, 64, 128}`.
   3. Finite-difference smoke checks at one or two tiny shapes
      (`seq_len = 8, d_k = 16`) as a sanity layer.

   Mitigation cadence: land the CPU analytical reference as the first
   commit of 9.2; validate each gradient accumulator (dq, dk, dv,
   L-retrieval) separately before wiring everything together.

3. **R-BF16-PRECISION-VISIBILITY — bf16 numerics masquerade as f16 in
   the bench table.** bf16 has wider exponent range but smaller
   mantissa than f16; numerical behavior at denorm / overflow
   boundaries differs. A bf16 matmul that looks correct in
   large-shape bench tests can be silently wrong in the tail of
   small-magnitude inputs. Mitigation: dedicated bf16 correctness
   test suite with inputs spanning {small / medium / large}
   magnitudes plus near-denorm and near-overflow inputs. Mirror the
   existing f16 correctness surface for `matmul_tc` and add the
   bf16-specific magnitude classes.

4. **R-LDMATRIX-UPLIFT-UNCERTAIN — ldmatrix may not move the sync-path
   number as much as hoped.** The sync path's remaining gap vs async
   is partly shared-memory contention, partly global-load throughput.
   ldmatrix solves the former but not the latter. Mitigation: land
   the IR primitive and wire it into fragment-A regardless of measured
   uplift; the primitive is useful even if 9.3 does not hit a
   dramatic bench number because it unlocks future optimizations.
   Set 9.3's ship gate on correctness + non-regression rather than a
   headline number that may not materialize.

5. **R-PHASE-SCOPE-DRIFT — attention bwd sprint expands under review.**
   FA bwd has many valid extensions (atomic-add single-pass, bf16
   version, multi-head generalization, dropout-in-fwd-reversed-in-bwd,
   …). Mitigation: 9.2 ships single-head f32 fwd+bwd with saved-L
   stats and no other changes. Anything else gets flagged into 9.2.5
   or post-Phase-9 issues.

6. **R-FA-BWD-STATS-OWNERSHIP — candle may not support arbitrary saved
   intermediates between fwd and bwd.** kaio-candle's existing bwd
   ops (Sprint 7.4d matmul_tc) did not need saved state — the
   forward-reuse pattern uses only inputs. FA bwd needs the saved `L`
   from fwd. If candle's `CustomOp::bwd()` API only exposes inputs /
   output / grad_output and has no hook for arbitrary saved tensors,
   the saved-stats plan silently breaks. Mitigation: the first commit
   of 9.2's candle bridge work is a direct audit of candle-core's
   `CustomOp2`/`CustomOp3` bwd signatures plus any multi-output or
   saved-state API; the decision is captured in the 9.2 sprint plan
   between (a) recompute `L` inside `bwd()` — simplest, slight perf
   hit; (b) multi-output fwd returning stats as hidden output;
   (c) stash on op state with explicit lifetime analysis. No kernel
   code work begins before this decision. The recompute path
   (option a) is the recommended default if the audit is unclear — it
   avoids saved-state coupling to candle internals.

7. **R-KAIO-CANDLE-API-DRIFT — candle's CustomOp bwd signature may
   have evolved since Sprint 7.4d.** Mitigation: the first commit of
   9.2 audits the currently-pinned `candle-core` version's
   `CustomOp2::bwd` / `CustomOp3::bwd` trait signatures. Do **not**
   `cargo update -p candle-core` as part of this audit — if the
   audit shows drift is needed, the update is a separate, subsequent
   commit isolated from any other 9.2 work so dependency churn does
   not mix with the hard correctness work.

## Explicit non-goals (Phase 9 will NOT do)

- **Multi-head attention.** Phase 9 keeps single-head self-attention.
  Multi-head is a separate orchestration layer on top.
- **Attention backward for `attention_tc`.** Short-seq inference op;
  training users route to `attention_flash`. Tracked behind an issue
  if user demand materializes.
- **Training-grade mixed-precision loss scaling / gradient clipping.**
  These belong in the framework (candle), not in KAIO kernels.
  `kaio-candle` provides the gradient compute; the training loop
  provides the numerics.
- **Hopper `wgmma` / SM 9.0 kernels.** Out of scope. Phase 9 stays on
  Ampere (SM 8.0+) plus Turing (SM 7.5 for ldmatrix forms that support
  it).
- **Multi-GPU, ROCm, PyPI publish.** All explicitly community-driven
  per [`docs/phases.md`](../../../phases.md).

## Review Context

This master plan establishes the architectural guardrails (constraints
+ saved-stats ownership audit + sign-preservation analogues for the
new precision variant + bwd-kernel register-pressure gate) so the
sprint-level plans for 9.1 / 9.2 / 9.3 can reference them without
re-litigating. The sprint-level plans each receive their own full
planning + adversarial review cycle as they kick off.
