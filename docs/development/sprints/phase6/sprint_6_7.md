# Sprint 6.7 — Multi-warp TC matmul + edge tiles + benchmark + promotion

**Status:** ✅ Complete
**Branch:** `phase6`
**Parent:** `8aa77cb` (Sprint 6.6 + 6.8 prep)
**Master plan:** [`phase6_master_plan.md`](phase6_master_plan.md)

---

## Intent

Convert Phase 6's correctness foundation (Sprints 6.1–6.6) into a
measurable production claim. Three things converged in this sprint:

1. **Performance** — restructure the per-block tile from single-warp
   16×8 (Sprint 6.3 / 6.4) to multi-warp 64×64 so each block does
   meaningful work and SM occupancy becomes real.
2. **API contract** — lift the temporary `M%16 = N%8 = K%16 = 0`
   divisibility constraint via runtime edge-tile predication.
3. **Public surface** — promote `matmul_tc` and `matmul_tc_async` from
   `#[doc(hidden)]` to stable `pub use`, and ship a real cuBLAS
   benchmark so `matmul_auto_tc` graduates from "Sprint 6.5 preview"
   to a measurable production claim before the v0.2.0 publish in 6.9.

The 6.5 README + CHANGELOG explicitly framed `matmul_auto_tc` as
"production performance lands in Sprint 6.7's multi-warp restructure."
Sprint 6.7 cashes that check.

---

## Headline result

**RTX 4090 (sm_89), 4096²:**

- `matmul_tc` (sync): **46.53 TFLOPS, 79.9% of cuBLAS sgemm**.
- `matmul_tc_async` (cp.async double-buffered): **49.56 TFLOPS, 85.1% of cuBLAS sgemm**.
- cuBLAS sgemm reference: 58.24 TFLOPS.

Past both the 50% gate (D11) and the 60% stretch from the original
plan. The 6.7b vectorized-loads sprint chases the remaining ~15
percentage points toward 90%+. See
[`docs/performance.md`](../../performance.md#tensor-core-matmul-performance-sprint-67)
for the full table 256–4096 and the apples-to-apples disclaimer.

---

## Three-gate cadence

Mirrors Sprint 6.6's stacked-novelty mitigation: structure
implementation as three sequenced correctness gates with focused
test surfaces.

### Gate A — Multi-warp `matmul_tc` (sync) + edge handling baked in (commit `e2c8bf3`)

**Path B reshape (mid-sprint).** The original plan staged
edge handling in Gate C; Gate A would temporarily break 3 of 4
existing GPU tests (16×8×16, 32×16×32, 128×8×16) until Gate C lifted
divisibility. Reshape: bake edge-tile handling into Gate A from day
one so all existing tests pass on the multi-warp restructure. Gate C
becomes "pathological-shape coverage + lift validate divisibility"
instead of "introduce edge tiles."

**Implementation:**
- Block dim `(32, 4, 1)` = 128 threads / 4 warps. `lane_id = tid_x`,
  `warp_id = tid_y`. Grid `(N.div_ceil(64), M.div_ceil(64), 1)`.
- 64×64 output block tile, each warp owns 32×32 sub-quadrant via 8 ×
  `mma.sync.m16n8k16` arranged as a 2 m_stripes × 4 n_stripes grid.
- Per-warp quadrant mapping: w0 rows[0,32) cols[0,32), w1 rows[0,32)
  cols[32,64), w2 rows[32,64) cols[0,32), w3 rows[32,64) cols[32,64).
- Shared layout (sync): `tile_a` 64×16 fp16 row-major (2 KB) +
  `tile_b` 16×64 fp16 col-major (2 KB) = **4 KB/block** (well under
  the 48 KB ceiling).
- **Edge handling pattern:** pre-zero shared tiles cooperatively at
  kernel start (128 threads × 4 b32 stores per tile = 4 KB total).
  Per-K-iter A and B loads use `bra-pred` to skip OOB threads' load
  + store pairs entirely; OOB shared slots stay zero. Output stores:
  per-fragment 4 stores each gated by `setp.lt.and.u32` combining
  row + col bounds in one PTX instruction, then `@p st.global.f32`.
  32 predicated stores per warp output.

**New kaio-core IR (additive, no breakage):**
- `MemoryOp::LdGlobalPred { dst, addr, ty, pred, negate }` — `@[!]p ld.global`.
- `MemoryOp::StGlobalPred { addr, src, ty, pred, negate }` — `@[!]p st.global`.
- `ControlOp::SetPAnd { dst, cmp_op, lhs, rhs, ty, src_pred }` —
  `setp.cmp.and.ty` for predicate composition in one PTX instruction.

**Two bugs caught during execution:**
1. **`mov.f16 %h0, 0` is invalid PTX** — no f16 immediate-zero form
   in the PTX ISA. Initial code used `mov.f16` with `ImmU32(0)` then
   `ImmF32(0.0)` — both rejected by ptxas (`Unexpected instruction
   types specified for 'mov'`). Fix: pre-zero shared
   at kernel start cooperatively, then bra-skip around the OOB B load
   + store pair. The shared slot stays zero from pre-zero; no f16
   zero immediate needed. Same pattern applied to A for consistency.
2. **Output store address double-counted `block_row` / `block_col`.**
   Initial code passed `rd_d_block_base` (= `rd_d + block_row*N*4 +
   block_col*4`) plus a row offset computed from `warp_block_row`
   (= `block_row + warp_row_quad*32`) — `block_row` was added twice.
   Fix: pass `rd_d` directly to `emit_warp_quadrant_store`;
   `warp_block_row` already includes the absolute `block_row`.
   Caught by the medium 64×64×64 test producing garbage (~26784 vs
   expected ~0.09 at output index (0, 40) in warp 1's quadrant).

**Gate test surface:** All 4 existing matmul_tc_api tests pass
(16×8×16, 32×16×32, 128×8×16, 64×64×64). Plus a new
`tc_matmul_multi_warp_quadrant_canary_64_64_64` with per-warp-
distinguishable inputs (A row-folded, B col-folded) and per-cell
analytic spot-checks inside each warp's 32×32 quadrant — catches
per-warp routing bugs that uniform inputs would mask.

### Gate B — Multi-warp `matmul_tc_async` (cp.async double-buffered) (commit `a2bbe78`)

Same multi-warp restructure on `matmul_tc_async_kernel.rs`. cp.async
pipeline structure unchanged from Sprint 6.4 — just bigger payloads
(2 KB per A buffer instead of 512 B; 2 KB per B buffer instead of
256 B).

**Helper reuse:** Gate A's `emit_mw_load_tile_a_64x16`,
`emit_mw_load_tile_b_16x64`, `emit_warp_quadrant_mma`,
`emit_warp_quadrant_store`, and `emit_pre_zero_shared_tiles`
promoted to `pub(crate)`. Load helpers parameterized with
`label_suffix: &str` so multi-call sites (preamble + per-iter)
generate unique bra-skip labels (e.g. `A_SKIP_TILE_LOAD_PRE` vs
`A_SKIP_TILE_LOAD_ITER`).

**New helper `emit_mw_load_tile_a_64x16_async`:** 128 threads × 1
`cp.async.ca` (size = 16) issue each = 2 KB per A buffer. Per-thread
row = `flat_tid / 2`, col_byte = `(tid % 2) * 16`. Edge: bra-skip on
OOB row. Same alignment contract as Sprint 6.4's single-warp variant
(shared dst + global src both 16-byte aligned via `BK = 16` and
`K % 16 == 0`).

**Sync vs async timing observation at 64×64×64** (env-gated
`KAIO_SPRINT_6_4_TIMING=1`, 50 iters median):
- sync: 0.735 ms/iter
- async: 0.793 ms/iter (1.08× sync)

Same 1.08× ratio as Sprint 6.4's single-warp 1.07×. At 64×64×64
there are only 4 K-tile iterations — too few for cp.async to
amortize pipeline overhead. The multi-warp benchmark across
256–4096 (post-Gate-C) shows the inversion at 4096² where async
pulls ahead by 6.5%.

**Zero bugs.** Gate B landed clean on first GPU try.

### Gate C — Lift M/N divisibility + pathological tests (commit `760b5f5`)

Edge-tile predication is already in Gate A. Gate C: (1) extend test
surface to pathological dims; (2) remove M%16 / N%8 from
`validate_dims_tc` and `check_tc_eligibility`. K%16=0 stays —
mma.sync.m16n8k16 K-tile is structural and the kernel does not
edge-pad K within an iteration.

**12 new pathological GPU tests** (6 per kernel):
- Sub-tile (mostly OOB lanes): 7×5×16, 15×7×16
- Off-by-one against mma boundary (16/8): 17×9×16, 33×17×16
- Mid-range mixed-divisibility: 100×50×64
- Large off-by-one against 1024 boundary: 1023×1023×1024

**K-shape note:** sub-tile shapes use K=16
not K=3. K=3 would force zero-padding across an entire K-tile,
which the kernel intentionally does not do (K-tile granularity is
structural, not edge-handled). Sub-tile stress lives in M and N
where users actually hit edge cases.

**Tuner test surface updates:**
- Removed `matmul_auto_tc_rejects_non_divisible_m` — constraint gone.
- Added `matmul_auto_tc_handles_non_divisible_dims` (positive test
  at 17×9×16 with CPU reference comparison, max_abs_err < 1e-2).
- Added `matmul_auto_tc_rejects_k_not_multiple_of_16` (K=24 still
  rejected with informative error).

**Performance note:** the 1023×1023×1024 GPU test takes ~67s,
dominated by the host CPU reference loop (O(M·N·K) ≈ 1e9 ops),
**not GPU work**. The kernel itself completes in milliseconds.
Tracked as low-priority tech debt (sampled CPU ref or checksum at
scale).

---

## D6: tuner conservative-default flip

**Sprint 6.5's conservative default** was `MatmulTcVariant::TensorCore`
(sync) — informed by Sprint 6.4's single-warp datapoint where async
was 1.07× slower than sync.

**Sprint 6.7 multi-warp benchmark inverts the relationship at large
shapes:**

| Size | sync TFLOPS | async TFLOPS | async/sync |
|------|------------:|-------------:|-----------:|
| 256³  | 0.05  | 0.04  | 0.86× |
| 512³  | 0.38  | 0.33  | 0.87× |
| 1024³ | 3.01  | 2.72  | 0.90× |
| 2048³ | 17.60 | 17.15 | 0.97× |
| 4096³ | **46.53** | **49.56** | **1.07×** |

At 4096² async wins by 6.5%. The multi-warp restructure gives
cp.async enough compute parallelism to hide pipeline latency. Below
2048² sync wins by tiny margins, but TC matmul is so far from cuBLAS
in that regime (3–8%) that the default barely matters there.

**Decision:** flip the cache-miss default to `MatmulTcVariant::TensorCoreAsync`.
Per-shape cache hits override the default; this only affects the
first-call path before tuning is run. Rationale: TC matmul is for
the large-shape regime, and async wins there. Inline comment in
`resolve_matmul_tc_variant` updated to point at this measurement.

---

## D7: promotion checklist (all 4 items green)

1. ✅ **All existing correctness tests green** — every Phase 5/6
   matmul_tc[_async] integration test passes after multi-warp
   restructure (no regression on the divisible shapes).
2. ✅ **Edge-tile tests green** — Gate C suite (sub-tile,
   off-by-one, mid, large) all pass. Divisibility constraint
   removed from validate_dims_tc + check_tc_eligibility.
3. ✅ **Benchmark table generated** — `matmul_tc_bench.rs` runs at
   all 5 sizes, captured in `docs/performance.md`. Measured 4096²
   ratio: **79.9% (sync) / 85.1% (async)** of cuBLAS sgemm — well
   past the 50% gate.
4. ✅ **Tuner fallback default reviewed post-Gate B** — D6 above.

**Promotion executed:** `kaio-ops/src/lib.rs` — dropped
`#[doc(hidden)]` and TEMP comments from `matmul_tc` (line 60) and
`matmul_tc_async` (line 69). Stable `pub use`. Kept `#[doc(hidden)]`
on `attention_tc` + `attention_tc_causal` (lines 71–78) per P2 —
Phase 7 only (FlashAttention-TC lifts the seq_k≤384 + divisibility
constraints; `attention_auto_tc` becomes the real dispatcher then).

**Rustdoc updates** (per framing guardrails — graduation, not
finality):
- `matmul_auto_tc` rustdoc in `tuner.rs`: dropped "Sprint 6.5 preview"
  framing, dropped temporary divisibility text, added measured
  performance numbers and the apples-to-apples disclaimer reference.
- `kaio-ops/src/lib.rs` Operations section: same treatment.
- README "Supported Kernel Features" matmul_auto_tc row: dropped
  "preview" + divisibility, added measured % cuBLAS at 4096² with
  link to performance.md.

---

## Architecture summary (the contract Phase 7 must preserve)

### Block + warp layout

```
Block dim:  (32, 4, 1)  = 128 threads = 4 warps
Grid dim:   (N.div_ceil(64), M.div_ceil(64), 1)
lane_id:    tid_x ∈ [0, 32)
warp_id:    tid_y ∈ [0, 4)
flat_tid:   warp_id * 32 + lane_id ∈ [0, 128)

Warp quadrant mapping:
  warp_row_quad = warp_id / 2  ∈ {0, 1}
  warp_col_quad = warp_id % 2  ∈ {0, 1}

  warp 0: rows [ 0, 32) cols [ 0, 32)
  warp 1: rows [ 0, 32) cols [32, 64)
  warp 2: rows [32, 64) cols [ 0, 32)
  warp 3: rows [32, 64) cols [32, 64)

Per-warp work per K-tile: 8 mma.sync.m16n8k16
  arranged as 2 m_stripes (16 rows each) × 4 n_stripes (8 cols each)
```

### Shared-memory layout

| Region    | Variant | Per-buffer | Buffers | Total | Layout                 |
|-----------|---------|-----------:|--------:|------:|------------------------|
| `tile_a`  | sync    | 2,048 B    | 1       | 2 KB  | row-major fp16, row stride 32 B |
| `tile_b`  | sync    | 2,048 B    | 1       | 2 KB  | col-major fp16, col stride 32 B |
| `tile_a`  | async   | 2,048 B    | 2       | 4 KB  | row-major fp16, align 16 (cp.async.ca) |
| `tile_b`  | async   | 2,048 B    | 2       | 4 KB  | col-major fp16, align 4 (sync staging path) |

Sync footprint: **4 KB/block**. Async footprint: **8 KB/block**. Both
well under the 48 KB ceiling.

### Edge-tile handling pattern

1. Pre-zero shared tiles cooperatively at kernel start (128 threads
   × 4 b32 stores per tile sync, × 8 b32 async). One `bar.sync` after.
2. Per-K-iter A load: `setp.lt.u32 p_row, block_row + row, M`;
   `@!p bra A_SKIP_TILE_LOAD_<suffix>`; 4× (`ld.global.b32` +
   `st.shared.b32`); label.
3. Per-K-iter B load: same pattern with col bound, label
   `B_SKIP_TILE_LOAD_<suffix>`. Async A path uses `cp.async.ca size=16`
   inside the bra-skip body.
4. Output stores: per-fragment 4 stores, each gated by
   `setp.lt.and.u32` combining row + col bounds in one PTX
   instruction, then `@p st.global.f32`. 32 predicated stores per
   warp output.

OOB shared slots stay zero from pre-zero. mma reads zero, accumulator
unchanged. Standard CUTLASS / Triton / cuDNN approach.

### Public-API contract

Both `matmul_tc` and `matmul_tc_async`:
- `f16 × f16 → f32` with fp32 accumulation.
- SM 8.0+ (Ampere or newer). Pre-Ampere returns
  `KaioError::Validation(SmTooLow)` via `PtxModule::validate()`.
- `M`, `N` may be any positive value.
- `K % 16 == 0` required (mma K-tile is structural).

`matmul_auto_tc` adds tuner-dispatched variant selection per
`(sm_target, [m, n, k])` cache key. Cache miss falls back to
`TensorCoreAsync` (D6).

---

## Quality gates

All passing at commit-time:

- `cargo fmt --all --check` ✓
- `cargo clippy --workspace --all-targets -- -D warnings` ✓
- `cargo test --workspace` — **279 host tests pass** (up from 275 at
  6.6 — +4 from kaio-core IR additions for the new edge-tile primitives;
  Gate C swapped tests within tuner module without count change)
- `cargo test --workspace -- --ignored` on RTX 4090 sm_89:
  - `matmul_tc_api`: **11/11** GPU tests (4 divisible + 1 quadrant
    canary + 6 pathological)
  - `matmul_tc_async_api`: **10/10** GPU tests (4 divisible + 6
    pathological)
  - `tuner_tc_test`: **6/6** (4 unchanged + handles_non_divisible_dims
    + rejects_k_not_multiple_of_16)
  - All Phase 5 + 6.6 GPU tests untouched and passing
- `cargo test -p kaio-core --test ptxas_verify -- --ignored` — **5/5**
- `cargo test -p kaio-ops --test matmul_tc_bench -- --ignored
  --nocapture` — runs cleanly, produces the TFLOPS table above
- `cargo doc --workspace --no-deps` — clean

---

## Files touched

| File | Change |
|------|--------|
| `kaio-core/src/instr/memory.rs` | **+** `MemoryOp::LdGlobalPred`, `StGlobalPred`. Emit + 3 unit tests. |
| `kaio-core/src/instr/control.rs` | **+** `ControlOp::SetPAnd`. Emit + 1 unit test. |
| `kaio-ops/src/matmul_tc_kernel.rs` | Full multi-warp rewrite. New constants (`BM_BLOCK=64`, `BN_BLOCK=64`, `WARPS_PER_BLOCK=4`, etc.). New helpers `emit_mw_load_tile_a_64x16`, `emit_mw_load_tile_b_16x64`, `emit_warp_quadrant_mma`, `emit_warp_quadrant_store`, `emit_pre_zero_shared_tiles` — all `pub(crate)` for async-kernel reuse. Old `emit_load_a_tile` / `emit_load_b_tile` removed (no callers after Gate B). `validate_dims_tc` lifts M%16, N%8 — keeps K%16. Gate A quadrant canary helper. Structural test updated for 8 mma per K-iter, 2048 B tile sizes, predicate-composition emit, 32 predicated stores. |
| `kaio-ops/src/matmul_tc_async_kernel.rs` | Full multi-warp rewrite. Reuses Gate A helpers via `pub(crate)` imports. New `emit_mw_load_tile_a_64x16_async` for cp.async.ca size=16 with bra-skip on OOB row. Pre-zero shared via shared helper (8 KB total — both buffers of both tiles). Block dim (32, 4, 1), grid div_ceil-by-64. Output store via shared `emit_warp_quadrant_store`. Structural test updated: 8 mma, 4096 B per tile (2 buffers × 2048), 32 predicated stores. |
| `kaio-ops/src/tuner.rs` | `check_tc_eligibility`: dropped TC_M_STEP / TC_N_STEP constants + checks. Kept TC_K_STEP. `resolve_matmul_tc_variant`: D6 default flip to `TensorCoreAsync`, inline-comment rationale updated. `matmul_auto_tc` rustdoc rewritten — drop preview framing, add measured perf, point at performance.md disclaimer. `tune_matmul_tc` initial best_variant default also flipped to `TensorCoreAsync`. |
| `kaio-ops/src/lib.rs` | **D7 promotion:** `matmul_tc` + `matmul_tc_async` no longer `#[doc(hidden)]`. Operations section rewritten with measured perf headline. KEPT `#[doc(hidden)]` on `attention_tc` + `attention_tc_causal` per P2. |
| `kaio-ops/tests/matmul_tc_api.rs` | **+** 6 Gate C pathological tests + 1 Gate A quadrant canary. |
| `kaio-ops/tests/matmul_tc_async_api.rs` | **+** 6 Gate C pathological tests. |
| `kaio-ops/tests/tuner_tc_test.rs` | Swapped `rejects_non_divisible_m` for `handles_non_divisible_dims` (positive) + added `rejects_k_not_multiple_of_16` (negative for the structural K constraint). |
| `kaio-ops/tests/matmul_tc_bench.rs` | **NEW** — TC sync + TC async + cuBLAS sgemm at 256/512/1024/2048/4096. 5 warmup + 20 timed median. `#[ignore]`-gated. Apples-to-apples disclaimer in eprintln header (D8). |
| `CHANGELOG.md` | Sprint 6.7 entry under [Unreleased] — added + breaking sections. Framing guardrails (graduation, not finality). |
| `README.md` | Supported Kernel Features matmul_auto_tc row update. Phase 6 progress section: 6.7 ✅ + 6.7b queued. |
| `docs/performance.md` | NEW Tensor-Core Matmul Performance section with measured table, apples-to-apples disclaimer, why-small-sizes-underperform explanation, path to 90%+. |
| `docs/benchmarks.md` | TC bench reproduction command + Sprint 6.7 results table. |
| `docs/development/sprints/phase6/sprint_6_7.md` | **NEW** — this doc. |
| `docs/development/sprints/phase6/PHASE_6_LOG.md` | 6.7 row filled in. |
| `docs/development/sprints/phase6/phase6_master_plan.md` | 6.7 row updated, 6.7b row inserted. |
| `docs/development/tech_debt.md` | D10 fragment-loader hoist deferred to 6.7b. Added large-bench CPU-ref runtime caveat. |

---

## Planning decisions and mid-sprint folds

### Initial scope (P1/P2/P3)
- P1: 50% gate, 60% stretch (split into 6.7 + 6.7b). Confirmed.
  Result: blew past at 79.9% / 85.1% — the conservative gate was
  almost too conservative. Gap to 90%+ in 6.7b is meaningful but
  smaller than the original P1 framing implied.
- P2: matmul-only promotion; defer attention_tc[_causal] to Phase 7.
  Confirmed and held.
- P3: cuBLAS sgemm only + apples-to-apples disclaimer. Confirmed and
  held.

### Structural corrections during planning
- **D1 arithmetic bug** in the original plan: 4 warps × 16×16 = 1024
  elements, not 4096. Fixed to 32×32 per warp via 8 mma per K-tile
  (the layout that actually shipped).
- Confirmed: 64×64 / 4-warp over 128×64 / 8-warp for this sprint;
  D4 OOB-padding precision-safe (standard CUTLASS pattern); D10
  fragment-loader parameterization should live in kaio-core if done.
- Confirmed Gate A doesn't need further splitting.

### Gate C deferral protocol
- Hardened: if Gate C defers, deferral is documented in 3 places and
  6.7c becomes hard sequencing block before 6.7b / 6.8 / 6.9. (Did
  not fire — Gate C landed cleanly.)

### Adversarial-review folds
- Edge-tile test surface needs sub-tile cases (1, 7, 15-class).
  Folded in as 7×5×16 + 15×7×16.
- Gate C should be non-blocking to promotion if it threatens the
  multi-warp performance story. Spelled out in promotion checklist
  item 2.
- D10 hoisting timing — only do if low-diff and non-disruptive.
  Made conditional, then deferred to 6.7b after Gate-A measurement
  showed ptxas's CSE handles the redundant div/rem within
  `emit_warp_quadrant_mma` cleanly.
- Promotion criteria should be a checklist, not just sprint
  completion. Added 4-item checklist (D7 above).
- README/CHANGELOG framing should be "graduation, not finality."
  Applied verbatim guardrails.
- D8 cuBLAS-disclaimer phrasing — used verbatim in
  `matmul_tc_bench.rs` header and `docs/performance.md`.

### Mid-sprint: Path B reshape
- After Gate B planning revealed the test-surface mismatch (Path A
  would `#[ignore]`-disable 3 of 4 existing matmul_tc_api tests
  until Gate C), execution paused for a second-opinion check.
  Outcome: bake edge handling into Gate A so the multi-warp
  restructure ships green-on-everything from day one. Gate C
  becomes "pathological coverage + lift validate," not "introduce
  edge tiles."

---

## Carry-forward to 6.7b / 6.8 / 6.9 / Phase 7

- **6.7b (NEW, queued):** vectorized loads (`MemoryOp::LdGlobalB128`
  + emit + ptxas_verify), bank-conflict padding for `tile_a` /
  `tile_b`, possibly `ldmatrix`. D10 fragment-loader hoist
  (parameterize `(group_id, thread_id_in_group)` on
  `load_fragment_*_shared_*`) lands here as well — same surface
  area, lower marginal disruption when the loaders are already
  being touched. Target: push past 90% of cuBLAS sgemm at 4096².
- **6.8 (showcase examples — pre-planned):** unchanged. Three
  standalone examples under `examples/` (fused SiLU-gate, GELU
  comparison, RMSNorm).
- **6.9 (publish v0.2.0):** unchanged. Includes `matmul_auto_tc` +
  promoted `matmul_tc` + `matmul_tc_async` as stable production
  APIs at the 6.7b measured number.
- **Phase 7:** FlashAttention-TC, `attention_auto_tc` real
  dispatcher, `attention_tc[_causal]` promotion to stable `pub`
  (waits on FlashAttention-TC lifting seq_k ≤ 384), bf16 inputs,
  larger mma shapes on Hopper+.

---

## Tech debt

- **D10 fragment-loader hoist** (parameterize
  `load_fragment_a_m16n8k16_shared_row` and
  `load_fragment_b_m16n8k16_shared_col` with optional
  `(group_id, thread_id_in_group)` to avoid recomputing div/rem on
  each call). Deferred to 6.7b — Gate-A measurement suggests
  ptxas's CSE handles the 6 calls per K-tile cleanly, and the
  parameterization is more naturally folded into 6.7b's
  fragment-loader work for vectorized loads.
- **Large-bench CPU reference runtime** — 1023×1023×1024 GPU test
  takes ~67s, dominated by the host CPU reference loop. Consider
  sampled CPU ref or checksum at scale. Low-priority.
- **`ptxas_verify` env-var mutation hygiene** (Sprint 6.4 fallout)
  — unchanged this sprint.
- **Macro-codegen `load_module` migration** — unchanged this sprint.
  After it lands, `load_ptx(&str)` can be `#[deprecated]`.

---

## Commits

| Commit    | Gate / scope                                             |
|-----------|----------------------------------------------------------|
| `e2c8bf3` | Gate A — multi-warp `matmul_tc` (sync) + edge handling   |
| `a2bbe78` | Gate B — multi-warp `matmul_tc_async` (cp.async)         |
| `760b5f5` | Gate C — pathological tests + lift M/N divisibility      |
| _(final)_ | Sprint completion: bench, D6 flip, D7 promotion, docs    |
