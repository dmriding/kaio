# Sprint 6.6 — Tensor-Core Scaled Dot-Product Attention (`attention_tc`)

**Status:** ✅ Complete
**Branch:** `phase6`
**Parent:** `64b2bca` (Sprint 6.5 + PHASE_6_LOG hash fill)
**Plan file:** `C:\Users\david\.claude\plans\calm-inventing-dragon.md`
**Master plan:** [`phase6_master_plan.md`](phase6_master_plan.md)

---

## Intent

Prove the **fused** TC-attention pipeline end-to-end before Sprint 6.7's
multi-warp restructure and Phase 7's FlashAttention-TC replace the kernel
body. Sprint 6.6 is the correctness-architecture checkpoint for attention
that Sprint 6.3 was for matmul: ship a deliberately slow single-warp
kernel that exercises every load-bearing contract (fragment layouts,
shared-memory tile conventions, intra-kernel f32→f16 bridge, warp-shuffle
softmax, causal masking with global-coordinate math) so that future
performance work builds on validated foundations instead of inventing
them under pressure.

The master plan marks 6.6 **optional** for v0.2.0. The decision to ship
it anyway was load-bearing: (a) the 6.3 parallel — 6.3 locked in TC
matmul contracts before 6.4 added cp.async and 6.5 added the tuner, same
logic applies to attention before Phase 7; (b) attention is the kernel
Rust ML users most need, so the correctness proof matters for the
positioning story; and (c) the **bridge seam** (f32 softmax output →
`cvt.rn.f16.f32` → f16 input to the second `mma.sync` within a single
warp's lifetime) is novel versus anything shipped in 6.3–6.5 and
must be validated before 6.7 can optimize it.

## Scope delivered

1. **`attention_tc` kernel** — fused `Q · Kᵀ · inv_sqrt_dk → softmax
   → cvt → probs · V → out` with two back-to-back `mma.sync.m16n8k16`
   instructions, row-serial warp-shuffle softmax in between, and an
   intra-kernel `cvt.rn.f16.f32` on the probs. `#[doc(hidden)] pub use`
   — internal preview until Phase 7.
2. **`attention_tc_causal` variant** — same kernel with a build-time
   `causal: bool` flag driving a `setp.gt.u32` + `selp.f32 -3.4e38,
   %score, %p` mask block between matmul1 and softmax. Zero runtime-
   branch cost. Two distinct PTX modules from one Rust builder. Also
   `#[doc(hidden)] pub use`.
3. **`Selp` instruction in `kaio-core::ArithOp`** — minimal IR
   addition (`selp{ty} dst, a, b, p`) required by the causal mask.
   One variant + emit + `emit_selp_f32` unit test.
4. **Shared test helpers** — `cpu_attention_f16xf16_f32` and
   `assert_close_attention` in `kaio-ops/tests/common/mod.rs`, with
   the load-bearing `abs < 5e-3 OR rel < 2e-2` tolerance documented.
5. **Regression gates** — five new host unit tests cover the sprint's
   structural claims (module build, two-`mma.sync` shape, causal
   mask ops presence, `sm_70` rejection via `PtxModule::validate()`
   for both variants, shared-memory budget ≤ 46 KB for both).

### Out of scope

Per plan: FlashAttention-TC (Phase 7, own sprint), multi-warp
restructure (Sprint 6.7), lifting the `seq_k ≤ 384` cap (algorithmic
change — Flash or multi-block tiling), public `attention_auto_tc`
(Phase 7 when a second candidate provides the real dispatch
decision), benchmarks vs cuBLAS / FlashAttention v2,
bf16/non-standard mma shapes, macro-codegen `load_module` migration.

## Three-gate bring-up

Plan Opus + Codex reviews both flagged this sprint's stacked novelty
(TC matmul #1 + scale + optional causal mask + row-serial softmax +
f32→f16 bridge + TC matmul #2) as a debugging swamp risk. Solution:
structure implementation as three internally-sequenced correctness
gates with isolated dev entrypoints that are deleted before the
sprint commit bundle lands.

### Gate A — matmul1 correctness (commit `845cdde`)

Reduced kernel `attention_tc_gate_a` writes `Q·Kᵀ·inv_sqrt_dk` to a
global scores buffer. No softmax, no mask, no second matmul. Three
GPU tests (16×16×16, 32×32×32, 64×128×64) against the host-side
scaled-scores reference. **All three green on first GPU try** —
validated: runtime-stride `FragmentA` loader (`emit_load_fragment_a_runtime_stride`
— needed because `d_k` is a kernel parameter, not a compile-time
constant), Q staged once per block in 16×d_k row-major shared with
stride = `d_k * 2`, K staged per-mma-iteration as 16×8 column-major
shared (reusing the `emit_stage_k_chunk` pattern established by
matmul_tc), single-warp grid `(seq_q / 16, 1, 1)`, scale factor
folded in post-mma with a single `mul.f32`.

### Gate B — softmax + cvt bridge (commit `cc8b4c1`)

Extended dev kernel `attention_tc_gate_b` adds scores staging into
`scores_tile` shared (f32 row-major), row-serial softmax, `cvt.rn.f16.f32`,
and probs staging into `probs_tile` shared, with a dev-path copy
of `probs_tile` → global `probs` output. Three GPU tests at the
same shapes against the host softmax reference. **All three green
on first GPU try**.

Softmax algorithm per row: (1) max-reduce — warp-strided local scan
initialized to `-INF`, bfly reductions with masks 16/8/4/2/1 and
member-mask `0xFFFFFFFF`; (2) `ex2.approx.f32 ex, (diff * log2(e))`
with exp values stored back into `scores_tile` for phase 3 to reread,
sum-reduce via the same bfly pattern; (3) `rcp.approx.f32 inv_sum`,
multiply, `cvt.rn.f16.f32`, store to `probs_tile`. For `seq_k < 32`
lanes above `seq_k` never enter the strided loop; their `-INF` max
and `0` sum are identity elements for bfly reductions so the row
result stays correct at the smallest 16×16 shape.

Key IR-level finding: `shfl.sync.bfly.b32` on f32-kind registers
emits cleanly and ptxas treats them as `.b32` bit-exact — no
bit-cast helper needed. `Ex2` and `Rcp` kaio-core ops cover the
softmax math at f32 with `.approx` semantics that match the scalar
attention path's numerical contract.

### Gate C — full fused kernel (commit `d181623`)

Final `attention_tc` kernel: drops the Gate B probs-copy-to-global
and adds the second matmul — `probs_tile` as A (row-major shared,
16 × seq_k, loaded via `emit_load_fragment_a_runtime_stride` with
stride = `seq_k * 2`), V staged per-mma-iteration as 16×8
column-major shared (new helper `emit_stage_v_chunk` — V's global
slice is 16 rows × 8 cols row-major, opposite shape from K's 8×16
row-major slice, so the helper's flat→(row, col) mapping differs),
`mma.sync` accumulating into an output FragmentC, and
`emit_store_fragment_c_to_global_out` mirroring matmul_tc's inline
store with f32 row stride = `d_v * 4`.

Five shapes tested against the full CPU attention reference with
`abs < 5e-3 OR rel < 2e-2` tolerance: smallest 16×16×16×8, 32³×32,
64×128×64×64, 64×384×64×64 hard-cap probe, and 16×32×16×8 (the
row-0 canary's shape). **All five green on first GPU try**. The
hard-cap probe proves the published `seq_k ≤ 384` contract is
honest — the shared-memory budget actually fits at worst-case
`(seq_k=384, d_k=128, d_v=64)` combination.

## 6.6b causal variant (commit `59ba3f0`)

`build_attention_tc_module` gains a `causal: bool` flag parameter.
When `true`, emits a mask block between matmul1's scale and the
`scores_tile` store: per-lane, four `setp.gt.u32` + `selp.f32
-3.4e38, %scored, %p` pairs — one per FragmentC scalar register.
Each scalar's `(local_row, local_col)` within the 16×8 output chunk
maps to global coordinates `(block_row + group_id{,+8}, n_chunk*8 +
2*tig{,+1})`. Mask predicate `global_col > global_row` → replace
with `-3.4e38`, matching scalar `attention_causal`'s convention
(softmax's max-subtract then underflows cleanly to zero without NaN).

Six GPU tests: five shapes matching the non-causal suite plus a
dedicated `attention_tc_causal_row0_self_only` canary that
constructs V with distinct per-row magnitudes and asserts the
output row 0 exactly matches V row 0 — gates against the off-by-one
trap (`c > r` vs `c >= r`) the scalar path also has a regression
test for. **All six green on first GPU try**.

## 6.6c regression gates (commit `335a777`)

Five new host unit tests in `attention_tc_kernel::tests` lock down
the structural claims without a GPU:

- **`attention_tc_module_builds_for_sm_89`** — module emits
  `.entry attention_tc`, exactly two `mma.sync.aligned.m16n8k16`
  calls (matmul1 + matmul2), and `cvt.rn.f16.f32` bridge.
- **`attention_tc_causal_module_contains_mask_ops`** — causal
  variant emits `setp.gt.u32` + `selp.f32`, still exactly two
  `mma.sync` calls. Locks the build-time flag's wiring.
- **`attention_tc_module_rejects_sm_70_via_validate`** — inherits
  Sprint 6.5's `load_module` idiom: `PtxModule::validate()` rejects
  `sm_70` cleanly with `ValidationError::SmTooLow { required: 80,
  actual: 70 }` before any driver interaction.
- **`attention_tc_causal_module_rejects_sm_70_via_validate`** —
  same for the causal variant.
- **`attention_tc_module_shared_bytes_under_ceiling`** — Codex-fold
  regression gate: builds both variants, sums the module's
  `SharedDecl` bytes, asserts total stays under the 46 KB ceiling
  (2 KB margin vs the 48 KB hardware static shared limit). Actual
  total: 41,472 B = 40.5 KB. Asserts the sum matches the module-
  level `DECLARED_SHARED_BYTES` constant (catches silent decl drift).
  Covers both causal and non-causal variants — both declare the
  same five regions identically.

Also a compile-time `const _: () = assert!(DECLARED_SHARED_BYTES ≤
SHARED_MEMORY_CEILING_BYTES)` inside `attention_tc_kernel.rs` that
would fail the build if a developer adds a shared decl that pushes
the total over 46 KB.

## Architecture

### Shared-memory layout (internal contract, Phase 7 must preserve)

| Region        | Layout       | Size (max)   | Role                                       |
|---------------|--------------|--------------|---------------------------------------------|
| `tile_q`      | row-major f16 | 4,096 B (16×128) | Q once per block, reused across all n_chunks |
| `k_chunk`     | col-major f16 | 256 B (16×8)    | K per matmul1 inner iteration               |
| `scores_tile` | row-major f32 | 24,576 B (16×384) | Matmul1 output → softmax input            |
| `probs_tile`  | row-major f16 | 12,288 B (16×384) | Softmax output → matmul2 A fragment       |
| `v_chunk`     | col-major f16 | 256 B (16×8)    | V per matmul2 inner iteration               |
| **Total**     |              | **41,472 B ≈ 40.5 KB** | Worst-case `(seq_k=384, d_k=128, d_v=64)` |

Worst-case utilization is 86% of the 48 KB hardware static shared
limit; the `seq_k ≤ 384` cap is exactly what makes this fit with
~2 KB margin. Phase 7's FlashAttention-TC eliminates `scores_tile`
and `probs_tile` via online softmax, lifting the seq_k constraint.

### Grid and warp

Grid: `(seq_q / 16, 1, 1)`. Block: `(32, 1, 1)` — single warp per
block. Each block owns one 16-row output slab and the full
`seq_k × d_v` computation for those rows. At `seq_q = 1024` this
launches 64 blocks of 32 threads = 2048 threads total vs an RTX
4090's 128 SMs × ~48 resident warps — terrible occupancy, expected
for a correctness-first kernel. Sprint 6.7's multi-warp restructure
targets this.

### Narrow contract (temporary, lifted at Phase 7)

- **Types:** f16 Q/K/V → f32 output. No bf16 variant yet.
- **Hardware:** NVIDIA Ampere or newer (SM 8.0+).
- **Shapes:** `seq_q % 16 == 0`, `seq_k % 16 == 0`, `d_k % 16 == 0`,
  `d_v % 8 == 0`, `seq_k ≤ 384`, `d_k ≤ 128`, `d_v ≤ 128`.

### Bridge seam (D8 — the sprint centerpiece)

The architectural contract this sprint exists to validate is the
`f32 accumulator → softmax (f32) → cvt.rn.f16.f32 → f16 input to
next mma.sync` inside a single warp's lifetime with coherent
shared-memory layout for both intermediate f32 scores and the f16
probs that feed matmul2. Every Sprint 6.6 decision (single-warp
grid, separate scores/probs shared regions, row-serial warp-shuffle
softmax, build-time causal flag) exists to isolate this seam from
unrelated failure modes — Gates A/B/C specifically target
pre-bridge / bridge-only / post-bridge failures respectively.

## Correctness

- **275 host tests pass** (up from 268 at Sprint 6.5) — +7 from
  6.6: 5 new `attention_tc_kernel::tests` + 1 `emit_selp_f32`
  (kaio-core) + 1 `cpu_reference_is_deterministic` (attention_tc_api).
- **133 GPU tests pass** (up from 122 at Sprint 6.5) — +11 from
  6.6: 5 non-causal + 5 causal + 1 `row0_self_only` canary.
- All Phase 5 + earlier Phase 6 tests untouched and passing
  (mma_sync_fragment, cp_async_roundtrip, matmul_tc_api,
  matmul_tc_async_api, tuner_tc_test, attention_api).
- Clippy clean with `-D warnings`. Rustfmt clean. `cargo doc
  --workspace --no-deps` clean.

## Plan-mode review folds (both accepted)

### Opus 4.6 (pass 1)
- **D5 V/K tile iteration** pinned explicitly as 16×8 per-mma
  tiles; budget recomputed to 41,472 B at worst case with
  pre-implementation headroom verification.
- **D11 tolerance `OR` is intentional** — documented in
  `assert_close_attention`'s rustdoc as load-bearing for near-zero
  outputs; tighten `abs_err` first if flakes appear.
- **D11 test matrix** expanded to 5 shapes × 2 causal variants =
  10 correctness tests plus the row-0 canary.

### Codex 5.4 (pass 2)
- **Scope split into three internal bring-up gates** — the primary
  process recommendation. Adopted as Gate A → Gate B → Gate C with
  dev-only entrypoints deleted at the final sprint commit.
- **`Q=K=V identity test` dropped** — the claim
  `softmax(QQᵀ/√d_k)·Q ≈ Q` doesn't hold in general. The CPU-
  reference comparison at 5 shapes × 2 variants is the actual
  correctness gate.
- **Shared-memory budget regression test added** as
  `attention_tc_module_shared_bytes_under_ceiling` — pure host,
  regression-gates the cap against future kaio-core alignment-
  policy changes.
- **Validation error wording simplified** in `validate_attention_tc_dims`
  to "use the existing f32 `attention` / `attention_flash` path
  instead" — avoids overclaiming f32-vs-f16 as the only difference.
- **Structural tests stay instruction-centric** (mnemonic presence
  + counts) rather than exact-PTX equality, per the Sprint 6.5
  precedent.
- **D8 bridge seam promoted to sprint centerpiece** in the plan
  doc and reiterated here — every other decision supports or
  isolates it.

## Carry-forward to Phase 7

- `attention_tc` and `attention_tc_causal` promote from
  `#[doc(hidden)] pub use` to stable `pub` when Phase 7's
  FlashAttention-TC arrives and lifts the divisibility + seq_k
  constraints via online-softmax tiling.
- `attention_auto_tc` becomes a real user-facing dispatcher at
  that point — mirrors `matmul_auto_tc` from Sprint 6.5. Standard
  TC attention handles small seq_k (fused path from 6.6) and
  flash TC attention handles arbitrary seq_k.
- `emit_load_fragment_a_runtime_stride`, `emit_stage_q`,
  `emit_stage_k_chunk`, `emit_stage_v_chunk`, `emit_softmax_rows`,
  `emit_store_fragment_c_to_scores_tile`,
  `emit_store_fragment_c_to_global_out` are all `pub(crate)` helpers
  that a Phase 7 multi-warp restructure can reuse or graduate into
  kaio-core depending on what the Phase 7 sprint determines.
- Generic `softmax_f32` / `warp_reduce_f32_max` /
  `warp_reduce_f32_sum` helpers in kaio-core — extract candidate
  if Phase 7's FlashAttention-TC duplicates the shuffle-reduction
  pattern non-trivially.
- `Selp` is a general-purpose f32 selection primitive; non-
  attention use cases will find it when they appear.

## Commits

| Commit    | Gate                                     |
|-----------|------------------------------------------|
| `845cdde` | Gate A — matmul1-only dev kernel         |
| `cc8b4c1` | Gate B — softmax + cvt bridge + probs dev |
| `d181623` | Gate C — full fused non-causal attention_tc |
| `59ba3f0` | 6.6b causal variant + Selp IR addition    |
| `335a777` | 6.6c regression tests                    |
| _(final)_ | 6.6d docs + dev-entrypoint cleanup       |
