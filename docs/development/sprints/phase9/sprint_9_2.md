# Sprint 9.2 — FlashAttention backward

**Status:** ✅ Complete (2026-06-12)
**Branch:** `phase9` (PR to `main` pending Phase 9 close)
**Gate:** Required for v0.5.0 (the phase's hard gate).

---

## Context

Unlike the matmul family (`dA = grad @ B^T` reuses the forward kernel),
attention backward has no forward-reuse identity — it needs dedicated
kernels that rebuild the softmax from saved statistics and accumulate
`dQ`, `dK`, `dV`. Sprint 7.4d explicitly deferred this. 9.2 ships it
in three layers: stats-saving forward variants in kaio-ops, the
backward kernels themselves, and the first kaio-candle binding for the
flash family (forward + backward together — no flash binding existed
before this sprint).

Backward identities (standard formulation, single-head self-attention,
`d_v == d_k`): with `P_ij = exp(S_ij − L_i)` rebuilt from the saved
row logsumexp `L`, the kernels compute `dV = Pᵀ·dO`,
`D_i = Σ_d dO·O`, `dS = P ∘ (dO·Vᵀ − D)`, `dQ = scale·dS·K`,
`dK = scale·dSᵀ·Q`. At `seq_len = 1` this collapses to the closed form
`dV = dO`, `dQ = dK = 0`, which serves as the wiring gate.

## What shipped

### CPU f64 analytical oracle (C0)

`kaio-ops/tests/attention_flash_bwd.rs` opens with an all-f64
analytical attention backward and validates it against f64 central
finite differences (plain, causal, and wide-dynamic-range upstream
gradients) plus the `seq_len = 1` closed form — the oracle proves
itself before judging any GPU output. Host-only; runs in the default
suite.

### `_with_stats` forward variants (C1)

`attention_flash_with_stats` + `attention_flash_causal_with_stats` —
kernel copies of the shipped forwards plus one tail store:
`stats[q_row] = m + log(l)`, the per-row softmax logsumexp (`m`, `l`
are block-uniform reduction results at kernel end). Shipped forwards
untouched. Contract: one f32 per (batch, head, query row) — collapses
to a flat `[seq_len]` buffer here; validation requires exactly that
and nothing more. Gates, all green on RTX 4090 sm_89:

- `out` matches the plain forward at `max_abs_diff == 0` (contract is
  "must not change numerical output"; the zero-diff form holds while
  both kernels come from the same toolchain).
- L vs f64 logsumexp at `abs < 1e-5 || rel < 1e-5` (dual — L sits near
  zero where pure relative error is undefined). Held with no
  calibration needed, max errors ~1e-7.
- Rowsum property `Σ_j exp(S_ij − L_i) = 1` per row.
- Determinism canary: two identical runs produce bit-identical stats
  (the candle backward's correctness rests on exactly this).

This was the first kernel consumer of the DSL `log()` builtin; its
macro lowering now interpolates `f32::consts::LN_2` as a named
constant so generated kernels stay clean under
`clippy::approx_constant` (separate prep commit).

### Backward kernels (C2–C4)

Three kernel shapes, five kernel functions, all block-per-row with
256 threads — the same proven structure as the forward, with the
loop nest swapped between the two main kernels so every output row
has exactly one owning block (no atomics):

- `flash_attn_bwd_preprocess_kernel` — `D[i] = Σ_d dO·O`, one
  block-reduce per row, mask-independent (serves both variants).
- `flash_attn_bwd_dkdv_kernel` (+ causal) — one block per **key** row,
  streams query rows in 256-tiles; per-thread `P_ij`/`dS_ij` into two
  shared tiles, then the forward's serial accumulation pattern into
  per-dim `dK`/`dV` registers.
- `flash_attn_bwd_dq_kernel` (+ causal) — one block per **query** row,
  streams key rows; single `dS` tile, same accumulation shape.

Public API: `attention_flash_bwd(device, grad_out, q, k, v, out,
stats, dq, dk, dv, seq_len, d_k)` + `_causal` sibling — validates
dims, allocates the internal `D` scratch (`seq_len × f32`), launches
preprocess → dkdv → dq. Documented provenance contract: `out`/`stats`
must come from the matching `_with_stats` call on the same inputs and
mask mode — lengths are checkable, provenance is not.

Register pressure (ptxas -v, sm_89): preprocess 18, dkdv 22/22,
dq 22/21, `_with_stats` 22/22 registers/thread — far under the ~100
budget the tiled alternative would have risked. Shared memory ≤ 2 KB
per block.

### kaio-candle bindings (C5–C6)

New `kaio-candle/src/attention_flash.rs`: `AttentionFlashOp`
(`CustomOp3`, `causal: bool`) + wrappers `attention_flash` /
`attention_flash_causal`. f32 end-to-end — no dtype casts anywhere.
Strictly single-head self-attention: Q/K/V all `[seq_len, d_k]`;
cross-attention shapes (which `attention_tc` accepts) are loudly
rejected with a pointer back to `attention_tc`.

`bwd()` is the first kaio-candle backward with dedicated PTX behind
it. candle's `CustomOp3` has no fwd→bwd saved-intermediate channel
(audited against the pinned 0.10.2: single-output `cuda_fwd`, no
multi-output mechanism), so the backward recovers L by re-running the
`_with_stats` forward — stateless, and bit-identical to saved stats
by the C1 determinism guarantee. The forward therefore calls the
plain kernel (stats written in fwd would have nowhere to go). Direct
kaio-ops callers keep the efficient path by holding the stats buffer
themselves. Implementation notes: storage guards from
`Tensor::storage_and_layout()` are held across all launches (the
readonly buffer borrows must outlive the GPU work), and gradient
tensors are built with `Tensor::from_storage(…, BackpropOp::none())`.

### Bench + docs (C7–C9)

`benchmark_attention_flash_backward` added to
`attention_flash_bench.rs`; `docs/performance.md` gains a backward
section with the two-tier cost contract — bwd-only (caller holds
stats) vs bwd + recompute (the candle autograd cost; the gap is
exactly one forward). 10-run worst/median at `d_k = 128`: bwd/fwd
2.6/2.5× at `seq = 128` rising to 10.8/10.6× at `seq = 2048` (plain).
The growth is a property of the block-per-row backward (each backward
block does ~2× the forward block's serial work; the forward gains
occupancy efficiency at scale that the heavier backward blocks
cannot match) — documented as such, with the tiled rework named as
the follow-up lever. Crate docs, READMEs, CHANGELOG and the
kaio-candle crates.io description updated to the 12-op / 6-backward
surface.

## Tests

All on RTX 4090 sm_89, `rel < 1e-2 || abs < 1e-3` vs the f64 oracle
unless noted:

- **kaio-ops** (`attention_flash_bwd.rs`): 4 host-only oracle
  self-checks; 26 GPU tests — 6 `_with_stats` L-gate shapes
  (incl. seq 1, 17×19, 257×32), 4 dkdv + 4 dq per-kernel oracle
  checks, seq=1 closed-form gate (plain + causal), full
  `{32,64,128}²` matrix, non-aligned + tile-boundary shapes,
  wide-dynamic-range `dO` (the `dS = P·(dP − D)` cancellation path),
  FD smoke at (8, 16).
- **kaio-candle** (`candle_attention_flash.rs`, new file): 19 GPU
  tests — 5 bit-exact forwards vs direct kaio-ops, 5 rejections
  (dtype, contiguity, rank-3, seq mismatch, d_v mismatch), 8
  gradient checks through real `.backward()` graphs (sum + weighted
  loss × plain/causal × shapes to 128²), seq=1 graph closed form.
- Full workspace + kaio-candle suites green; no regression in the
  78-test combined GPU sweep.
- A post-delivery review pass added 3 path-equivalence tests
  (candle suite now 22 GPU tests): the candle `.backward()`
  gradients must be bit-exact against the direct kaio-ops
  `_with_stats` → `attention_flash_bwd` pipeline on the same input
  bits. The two paths were previously pinned only transitively
  through the shared oracle; the direct pin is an
  orchestration-bug tripwire (e.g. a wrong buffer wired as `out`
  or `stats` in the binding's backward call).

Quality gates per commit: `cargo fmt --all -- --check`, `cargo clippy
--all-targets -- -D warnings`, `cargo test --workspace`, kaio-candle
fmt + clippy + no-CUDA build + `cargo doc`; GPU sweeps at each
kernel-bearing commit. One sequencing note for the audit trail: the
C4 commit's chained gate command short-circuited before `cargo fmt`
ran (clippy and tests had passed separately), so a formatting-only
follow-up commit landed later in the sprint and every gate was
re-verified individually at close.

## What didn't change

- Shipped `attention_flash` / `attention_flash_causal` kernels and
  public functions: byte-identical (the `_with_stats` variants are
  separate kernel functions).
- `kaio-candle/src/bridge.rs`: zero changes — all helpers were
  already dtype-generic.
- `attention_tc` family, matmul bindings, tuner: untouched.
- Versions: workspace 0.4.1 / kaio-candle 0.1.1 (v0.5.0 is the
  aggregate phase release).
- `candle-core` pin: stays `=0.10.2`.

## Follow-ups

- **Backward perf (9.2.5 candidates, post-v0.5.0):** FA2-style tiled
  backward (BLOCK_M > 1) to flatten the bwd/fwd ratio growth at large
  seq; a logsumexp-only recompute kernel (skips the V-accumulation
  half of the recompute forward); op-state stats stash in the candle
  binding (eliminates the recompute entirely but makes the op
  stateful — needs double-`.backward()` semantics analysis).
- **Out of scope by design** (tracked, not planned): multi-head
  orchestration, cross-attention backward (different mask-predicate
  shape), f16/bf16 flash variants, `attention_tc` backward.

9.2 closed the v0.5.0 hard gate. Sprint 9.3 (`ldmatrix`) followed the
same day; the release review + version-bump pass closed the phase as
v0.5.0 (2026-06-12).
