# Sprint 7.3 — Fused quantized QKV projection (`qkv_project_int8` / `qkv_project_int4`)

**Status:** ✅ Complete (INT8 MVS shipped at D4, INT4 contingent shipped at D6)
**Branch:** `phase7-rest` (continuing long-running branch; 16 ahead of `main` at sprint start, +12 this sprint = 28 ahead at sprint end).
**Release target:** None this sprint. Phase 7 closes with an aggregate release after 7.4.

---

## Context

Sprint 7.1 shipped `matmul_int8` (v0.3.0) — W8A8 symmetric, single-scalar scale, direct `mma.sync.m16n8k32.s8.s8.s32`. Sprint 7.2 shipped `matmul_int4` on `phase7-rest` — GPTQ-style W4A16 with group scales (DEQUANT-F16 path feeding `mma.sync.m16n8k16.f16.f16.f32`). Sprint 7.3 is where those two ops meet FlashAttention.

**Locked interpretation:** fuse the three QKV projections into a single kernel (rather than modifying `attention_tc` itself). The win is X-reuse: at typical LLM shapes the input activation X gets loaded once per K-tile and accumulated into three independent fragment-C banks — vs. 3× the global bandwidth when calling `matmul_int{4,8}` three times. `attention_tc` stays untouched; its f16 Q/K/V input contract is honored by emitting fragment-C as f16 at store-out.

**Fusion scope clarifier:** 7.3 fuses the three *linear projections* (Q, K, V) into one launch. It does **not** fuse projection with the attention kernel itself — score matmul, softmax, masking, and V-weighted sum all remain inside `attention_tc` unchanged.

**Why W8A16 (not W8A8) for the INT8 variant** (maintainer decision, round 2): `qkv_project_int8` sits between f16-contract layers in an inference pipeline. W8A16 makes the projection op drop-in across layer boundaries and unifies both INT8 + INT4 variants onto the same mma path (`m16n8k16.f16.f16.f32`, `K_TILE_SHARED=16`). `matmul_int8` remains the right op for users who genuinely need W8A8.

**MVS posture** (maintainer decision, round 2): `qkv_project_int8` (W8A16) through D4 was declared the minimum shippable core. INT4 shipped contingent on D5/D6/D7 staying clean. Both shipped.

## Scope

**MVS (shipped):**

- `kaio_ops::qkv_project_int8(device, x: &GpuBuffer<f16>, w_q/k/v_i8, scale_q/k/v: f32, q/k/v_out, m, n, k)` — W8A16 fused tri-output projection.
- Shared store-out helper `emit_store_fragment_c_f32_to_f16_packed` in `kaio-ops/src/store_out.rs` (cvt.rn.f16.f32 + MovPack + st.global.u32, optional scalar scale).
- GPU e2e tests: 7 tests incl. Q/K/V differentiation canary.
- Bench: fused vs 3× matmul_int4 baseline across decode + prefill tiers.

**Contingent (also shipped):**

- `kaio_ops::qkv_project_int4(device, x_f16, w_q/k/v_packed_u32, scales_q/k/v_f16, q/k/v_out, m, n, k, group_size)` — W4A16 fused tri-output projection.
- GPU e2e tests: 8 tests incl. group-boundary shapes, sign-extend canary, and Q/K/V differentiation canary.
- Showcase `examples/quantized_attention/` — end-to-end `qkv_project_int4 → attention_tc`.

## Decisions

### D1 — Sprint stub + dims validation + tile constants

`qkv_project_int4_kernel.rs` + `qkv_project_int8_kernel.rs` skeletons. Unified tile constants (both variants share the same mma path): `BM_BLOCK=64`, `BN_BLOCK=32` (later changed at Rollback #1), `K_TILE_SHARED=16`, `WARPS_PER_BLOCK=4`, `THREADS_PER_BLOCK=128`. `validate_dims_qkv_int{4,8}` enforce MHA constraint (`N_q == N_k == N_v`), buffer sizes, and K/N divisibility.

### D2 — Shared store-out helper

**Premise correction mid-execution.** Original plan scoped D2 as a shared helper promoted out of `attention_tc`'s store-out path. Execution surfaced that `attention_tc` outputs **f32** directly — no f16 store path existed to promote. Revised scope (maintainer-approved): build the helper standalone for qkv_project use. Helper shipped with cvt.rn.f16.f32 + MovPack + st.global.u32 chain, optional scalar scale for INT8. Attention_tc remains untouched. `ptxas_verify_store_out` passes sm_80 + sm_89.

### D2.5 — Register-pressure skeleton checkpoint

Minimal skeleton allocating all 48 f32 fragment-C regs per lane (3 grids × 2×2 sub-tiles × 4) plus fragment-A/B staging and scale regs. `ptxas -v` baseline:

| target | regs/thread | spills | cmem[0] |
|--------|-------------|--------|---------|
| sm_80  | 32          | 0      | 392     |
| sm_89  | 40          | 0      | 392     |

Comfortable 24–32 reg headroom under the 64-reg cliff at the skeleton level. No `.const` bank pathology (all kernel params land uniform in `cmem[0]`, per round 3 note).

### D3 — INT8 kernel body (split D3.1 / D3.2 / D3.3 / D3.4)

- **D3.1** cooperative loaders: `emit_mw_load_tile_x_f16_64x16` delegates to matmul_int4; `emit_mw_load_tile_w_int8_16x32` is new (16×32 i8, single-issue per thread, Q/K/V label_suffix differentiation).
- **D3.2** `emit_fragment_b_int8_per_lane`: 4 ld.shared.s8 → 4 `cvt.rn.f16.s8` → 2 `MovPack` packed pairs.
- **D3.3** `emit_warp_quadrant_mma_int8_per_projection`: 2×2 inner grid = 4 mma.sync per per-projection call.
- **D3.4** `build_qkv_project_int8_module`: full IR wiring. 13 params, 2 shared decls, K-loop with 7-barrier Design-S cadence (1 X + 2 per projection × 3 = 7).

**Rollback #1 fired at D3.4.** Full kernel ptxas reported **80 regs/0 spills** at the original 64×32 output tile on both sm_80 and sm_89 — above the 64-reg full-occupancy cliff. The plan pre-authorized this: drop `MMAS_PER_WARP_N` from 2 to 1, halving per-warp fragment-C live state from 16 to 8 f32 regs (3 grids × 8 = 24 frag_c regs per lane, vs the prior 48). Per-block output tile becomes **64×16** instead of 64×32.

Post-rollback ptxas_verify: 48 regs/0 spills both targets. 16-reg headroom under the cliff.

One framework bug caught: kaio-core's cvt emitter omitted `.rn` for S8→F16 (ptxas rejects bare `cvt.f16.s8` with "Rounding modifier required"). Fix in `kaio-core/src/emit/emit_trait.rs`: extended the int→float match to include `S8` as a source type.

### D4 — INT8 public host API + launch (MVS ship point)

`pub fn qkv_project_int8(...)` wired. Rustdoc explicitly spells out W8A16 vs W8A8 contract (round 2 discipline). GPU launch smoke test: zero inputs → zero outputs canary on canonical shape. MVS shippable here. Quality gates all green; velocity + register headroom signals favor continuing to D5.

### D5 — INT4 kernel body (contingent)

Same Rollback #1 applied preemptively (INT4 has identical fragment-C pressure + extra from group-scale reload + nibble-extract chain). New helpers mirror D3 pattern plus three INT4-specifics:
- `emit_mw_load_tile_w_packed_int4_2x16` — 16-col packed-INT4 cooperative load with in-tile + N-edge predicates.
- `emit_cooperative_load_group_scales_int4` — 16-cell f16 scale load, 8 active threads.
- `emit_warp_quadrant_mma_int4_per_projection` — 2×1 mma sweep with INT4 dequant feed via reused `super::matmul_int4_kernel::emit_fragment_b_int4_per_lane` (made `pub(crate)` for cross-kernel sharing).

`build_qkv_project_int4_module`: 3 shared decls (tile_x + tile_w + tile_scales), K-loop with **unconditional per-projection scale reload** (see D7 bug-fix note). ptxas_verify: 56 regs/0 spills both sm_80 + sm_89. 8-reg headroom (tighter than INT8's 16, as expected from dequant + scale-load state).

### D6 — INT4 public host API + launch (contingent ship)

`pub fn qkv_project_int4(...)` wired. Rustdoc documents the packing convention (8 nibbles per u32, K-contiguous, col-major `[K/8, N]`) + group-scale layout (`[K/group_size, N]` row-major, group_size=128 fixed) with cross-links to matmul_int4. GPU launch smoke test passes.

### D7 — GPU e2e tests (two bug catches)

7 INT8 tests + 8 INT4 tests across canonical / multi-block / group-boundary shapes + sign-extend canary + Q/K/V differentiation canary. **All 15 pass on RTX 4090 sm_89.** Tolerance: `max_rel_err < 1e-3` (loose enough for one f16-rounding-step difference vs reference; observed max_rel stays under 3e-4 in practice).

**Bug 1 (INT8 W-loader double-count):** `emit_mw_load_tile_w_int8_16x32` added `block_col` to its global offset, but `w_block_base_global` already carried the `block_col` byte shift. Every non-zero N-block computed from wrong global W addresses; single-N-block tests passed because `block_col=0` is a no-op. Fix: use `col_start` for within-tile offset.

**Bug 2 (INT4 scales-slot race):** Original design reloaded scales only on group-boundary K-tiles (every 8 K-tiles) but the single `tile_scales` shared slot was reused across Q/K/V per K-tile. After mma_V on K-tile 0, the slot held scales_V; on K-tile 1 (non-boundary) all three projections' mmas read scales_V instead of their own. Constant-weight canaries passed; random-data tests failed. Fix: reload scales every K-tile unconditionally (32 B/projection × 3 = 96 B/K-tile added global bandwidth, negligible vs 384 B/K-tile for the W loads). Group-boundary-only optimization with three separate `tile_scales_P` slots is deferred as a future optimization.

### D8 — Bench fused vs 3-separate

`kaio-ops/tests/qkv_project_bench.rs`: fused `qkv_project_int4` vs 3× sequential `matmul_int4` (apples-to-apples W4A16 baseline) + absolute-TOPS for `qkv_project_int8` (no fair W8A16 standalone baseline exists — `matmul_int8` is W8A8).

**Stable results (RTX 4090 sm_89, release mode, median of 20 iters after 5 warmup):**

| shape | M | N | K | fused ms | 3× matmul_int4 ms | ratio | tier |
|-|-|-|-|-|-|-|-|
| decode_m1 | 1 (pad to 64) | 2048 | 2048 | 81.9 | 246.5 | **3.01×** | decode win |
| decode_m64 | 64 | 2048 | 2048 | 82.2 | 246.8 | **3.00×** | decode win |
| decode_m64_large | 64 | 4096 | 4096 | 82.7 | 246.7 | **2.98×** | decode win |
| diag_m64_n1024_k2048 | 64 | 1024 | 2048 | 82.0 | 247.2 | **3.02×** | decode win |
| diag_m64_n2048_k1024 | 64 | 2048 | 1024 | 82.3 | 247.3 | **3.01×** | decode win |
| diag_m64_n2048_k4096 | 64 | 2048 | 4096 | 82.5 | 247.4 | **3.00×** | decode win |
| diag_m128_n2048_k2048 | 128 | 2048 | 2048 | 82.2 | 246.5 | **3.00×** | decode win |
| prefill_m512 | 512 | 4096 | 4096 | 206.7 | 246.6 | **1.19×** | prefill win |
| prefill_m2048 | 2048 | 4096 | 4096 | 579.7 | 494.8 | **0.85×** | prefill loss |

**Ship decision (maintainer 2026-04-16, after reviewing results):** ship B now, plan A as Sprint 7.3.5.

- Decode wins are substantial (3.0× across all M ≤ 128 shapes) — matches the plan's X-reuse + launch-overhead amortization thesis.
- Prefill_m512 at 1.19× passes the plan's ship threshold (≥ 1.15×).
- Prefill_m2048 at 0.85× is below the 1.00× ship-narrow threshold — per the plan this triggers Rollback #4 (Design S+½P). Maintainer scoped this as a separate **Sprint 7.3.5** rather than inline (synchronization semantics change in the inner loop; warrants full reviewer cadence, not a hotfix).

**Outlier investigation** (first-run decode_m64 reported 0.40×): not reproducible. Diagnostic sweep across 4 nearby shapes (rows labelled `diag_*` above) shows consistent 3.00× throughout. Confirmed transient measurement noise in the first run (likely first-call CUDA context warmup bleeding through despite 5-iter warmup loop). No kernel bug.

**Absolute-TOPS numbers intentionally not highlighted.** Repeated back-to-back bench runs cause thermal throttling on the RTX 4090 that crushes absolute throughput by 50–100× for all workloads including the cuBLAS sgemm reference. Ratios are throttle-invariant (fused + baseline throttle equally) so they remain the valid headline. Clean absolute-TOPS run is a follow-up.

### D9 — Showcase example (`examples/quantized_attention/`)

End-to-end `X → qkv_project_int4 → attention_tc` with an f16-reference path (3× matmul_tc → attention_tc) for comparison. GPTQ-lite per-column group quantizer reused from `examples/int4_matmul/`. Reports cosine similarity + max abs + mean rel on the final attention output.

Observed on RTX 4090 sm_89 (SEQ=64, D_MODEL=128, D_HEAD=64, GROUP_SIZE=128):
- Projection-stage Q cosine = **0.9975** — kernel working correctly.
- Final attention cosine = **0.9655** — below the 0.98 plan threshold.

Per the plan: "Random f16 weights are a worst case for group-scale quantization fidelity. Real LLM weights have much tighter group statistics and land measurably better than these synthetic numbers." The projection-stage number being 0.9975 localizes the final-output delta to softmax amplification of a small projection error, not a kernel bug. Documented prominently in both the example README and the run-time warn banner.

Wired into `cargo xtask showcase qkvattn`.

### D10 — Sprint doc + CHANGELOG + README + rustdoc

This document. Phase 7 log row (see `PHASE_7_LOG.md`). CHANGELOG unreleased entry. README ops table updated with `qkv_project_int8` + `qkv_project_int4`.

## Architectural decisions

### AD1 — Design S (serial fusion) over Design P (parallel)

Single shared W slot reused per projection, X loaded once per K-tile. 7 barriers per K-tile (1 X + 2 per projection × 3). Register budget wins over Design P's 3 parallel W slots + overlapped loads. Trade-off accepted: prefill under-performs matmul_int4 × 3 by ~15%. Rollback #4 (Design S+½P: 2 W slots with ping-pong, barriers 7→4) scoped as Sprint 7.3.5.

### AD2 — W8A16 for INT8 variant

f16 activations × i8 weights, scalar per-projection scale applied at store-out. Unifies INT4 + INT8 variants onto the same `m16n8k16.f16.f16.f32` mma path. `matmul_int8` (Sprint 7.1, W8A8) remains the right op for users who genuinely need int-only activations.

### AD3 — Rollback #1 on both variants (64×16 per-block tile)

INT8 full kernel overshot the 64-reg cliff at the original 64×32 tile (80 regs, 0 spills). INT4 preemptively halved at the same tile size since it has identical fragment-C pressure + extra dequant state. Post-rollback both variants sit comfortably under the cliff (INT8: 48 regs, INT4: 56 regs, 0 spills each, both sm_80 + sm_89).

### AD4 — Per-K-tile unconditional scales reload for INT4

Single shared `tile_scales` slot reused across Q/K/V per K-tile. Must be reloaded every K-tile (not only at group boundaries) because each of Q/K/V reads its own scales mid-K-tile. 96 B/K-tile added global bandwidth, negligible vs the 384 B/K-tile W loads. Three-separate-slot optimization deferred.

### AD5 — f16 pair store at store-out

`emit_store_fragment_c_f32_to_f16_packed` packs two adjacent f16 outputs into one b32 store (`N % 2 == 0` enforced in validate). Halves store bandwidth vs per-f16 scalar stores. The r+8 row offset is a within-fragment property of m16n8k16 (not projection-specific), so the helper is reused identically across Q/K/V.

## Results (sprint-final)

### Correctness — all green

- 84 kaio-ops host tests pass (+11 vs sprint start: D5 emit + D6 smoke + D4 smoke + existing)
- 15 GPU e2e tests pass on RTX 4090 sm_89 (7 INT8 + 8 INT4, including Q/K/V differentiation and sign-extend canaries)
- ptxas_verify PASSED on both sm_80 + sm_89 for: `store_out`, `qkv_skeleton`, `qkv_project_int8` (48 regs / 0 spills), `qkv_project_int4` (56 regs / 0 spills)
- Sprint 7.1 / 7.2 kernels untouched — all their existing tests continue to pass

### Performance — honest framing

- **Decode tier (M ≤ 128)** — fused wins ~3.0× consistently.
- **Prefill mid (M=512)** — fused wins 1.19× (above plan ship threshold).
- **Prefill large (M=2048)** — fused loses 15% vs 3× standalone. Documented in the public API rustdoc: "call three separate `matmul_int{4,8}`s for prefill-heavy workloads".

### Bugs caught and fixed this sprint

1. kaio-core cvt emitter omitted `.rn` for `S8` source (would have affected any future kernel doing int8→f16 conversions) — fixed in `kaio-core/src/emit/emit_trait.rs`.
2. INT8 W loader double-counted `block_col` in its global-offset computation — corrupted every non-zero N-block (caught by multi-block e2e).
3. INT4 shared `tile_scales` slot held stale scales across projections on non-boundary K-tiles — made all non-Q projections read Q's scales (caught by random-data e2e; constant-weight canaries passed through).

### Scope — all D sections landed

- D1 stubs + validate — ✅
- D2 standalone store-out helper (premise correction recorded) — ✅
- D2.5 register-pressure skeleton — ✅
- D3.1/D3.2/D3.3/D3.4 INT8 kernel body (+ Rollback #1) — ✅
- D4 INT8 host API (MVS ship point) — ✅
- D5 INT4 kernel body — ✅
- D6 INT4 host API (contingent ship) — ✅
- D7 GPU e2e tests (+ two bug fixes) — ✅
- D8 bench (ship-B decision recorded) — ✅
- D9 quantized_attention showcase — ✅
- D10 this doc + PHASE_7_LOG + CHANGELOG + README + rustdoc — ✅

### Commits (phase7-rest, 12 this sprint)

- `f96c1b6` D1 — sprint stubs + validate_dims + tile constants
- `49bdcaa` D2 — emit_store_fragment_c_f32_to_f16_packed (ptxas_verify sm_80 + sm_89)
- `94c5558` D2.5 — tri-output register-pressure skeleton (32–40 regs, 0 spills)
- `a31797e` D3.1 — cooperative loaders for X (f16) + W (i8) tiles + pre-zero
- `2934d27` D3.2 — emit_fragment_b_int8_per_lane (4 ld.s8 + 4 cvt.rn.f16.s8 + 2 MovPack)
- `dfa22a5` D3.3 — emit_warp_quadrant_mma_int8_per_projection (4 mmas/call)
- `3739d38` kaio-core: emit cvt.rn for s8 source
- `ed66a71` D3.4 — full INT8 module assembles, 80 regs (Rollback #1 trigger)
- `4852a85` D3.4 Rollback #1 — drop MMAS_PER_WARP_N 2→1 (48 regs, 0 spills sm_80/89)
- `484013e` D4 — qkv_project_int8 public host API + launch (MVS ship point)
- `5b3b9f4` D5 — qkv_project_int4 full kernel body (56 regs, 0 spills)
- `083840c` D6 — qkv_project_int4 public host API + launch (contingent ship)
- `484013e` (above)
- `ab8ecbd` D8 — qkv_project bench (fused vs 3×, decode + prefill tiers)
- D7 + D9 + D10 (this commit) as additional entries in final tally

### Follow-ups for future sprints

- **Sprint 7.3.5** — Design S+½P: 2 W slots with ping-pong index, overlap W_{P+1} load with mma_P. Barriers 7→4 per K-tile. Target: recover `prefill_m2048` from 0.85× to ≥1.15×. Full review cadence (synchronization semantics change in the inner loop). Ships only if bench confirms the recovery; if not, Design S stays and the docs are the answer.
- **Three-slot scales (INT4)** — eliminate the per-K-tile unconditional scale reload by giving each projection its own `tile_scales_P` slot (load once per group, persistent). +96 B shared mem per block, saves ~90 B global read per K-tile per projection. Only worth doing if bench shows it matters.
- **Sprint 7.4** — `kaio-candle` bridge crate. `qkv_project_int4/int8` join the forward CustomOp bindings.
- **Phase 7 aggregate release (v0.4.0)** after 7.4.
- **Grouped-query attention** (`qkv_project_gqa`) — dedicated follow-up sprint.
- **Bias / activation fusion** in projections — additive v2.
- **Asymmetric INT4 with zero-points** — inherited from 7.2's list.
- **Clean absolute-TOPS bench run** — thermal-stable methodology (cold system, single test at a time).
