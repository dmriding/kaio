# Sprint 7.3.5 — Design S+½P optimization for fused QKV projection

**Status:** ✅ Complete (INT8 shipped S+½P; INT4 measured, retained Design S)
**Branch:** `phase7-ship` (merging to `main`); `phase7-rest` retained as archive carrying the INT4 S+½P measured-data commits.
**Release target:** Phase 7 aggregate release after 7.4.

---

## Context

Sprint 7.3 shipped fused `qkv_project_int{4,8}` (Design S — serial fusion, one shared W slot reused per projection, 7 `bar.sync` per K-tile). The D8 bench found:

- Decode (M ≤ 128): fused ≈ 3.0× over 3× standalone.
- Prefill mid (M = 512): 1.19×.
- **Prefill large (M = 2048): 0.85×** — fused loses 15%.

Per the 7.3 plan's escape ladder, this triggers Design S+½P: **two shared W slots with a ping-pong index**, overlapping the cooperative load of `W_{P+1}` with the mma compute of `W_P`. Barriers drop from **7 → 4 per K-tile**. Target: recover `prefill_m2048` from 0.85× to ≥ 1.15×.

## Scope

- Rewrite the inner K-loop of `qkv_project_int{4,8}` to the 2-slot ping-pong pattern.
- Keep the 15 GPU e2e tests from 7.3 green (7 INT8 + 8 INT4).
- Add S+½P-specific correctness gates: a **slot-mapping canary** (distinguishable per-projection constants; catches deterministic ping-pong mis-wiring) and a **determinism stress** (100× same-shape bit-exact across runs; catches the barrier-misplacement race class this design introduces).
- Rerun the D8 bench. Ship criterion: `prefill_m2048 ≥ 1.15×`, decode tier ≥ 2.5×.

## Out of scope

- Any tile-shape change beyond the W-slot pipelining.
- `cp.async` (explicitly deferred; see Results / follow-ups).
- New public API surface.
- INT4 three-slot scales optimization (independent follow-up).

---

## What shipped

### INT8 `qkv_project_int8` → Design S+½P

- Inner K-loop rewritten: 2-slot `tile_w` ping-pong, 64 B bank-phase pad between slots (576 B stride avoids SMEM bank-port phase-alignment across concurrent LDSM/STS during overlap), `frag_A` register-hoisted once per K-tile and reused across Q/K/V epochs, predicated overlap-skip on the last K-tile.
- Barrier cadence: **4 per K-tile + 2 setup** (pre-zero, pre-load) — down from 7 per K-tile.
- Scales: unchanged from 7.3 (INT8 uses scalar per-projection scales applied at store-out).
- `ptxas_verify`: **sm_80 64 regs, sm_89 56 regs, 0 spills both.** Natural allocation 66 regs; sm_80 compresses 2 under the cliff without spilling.
- Correctness: **11 GPU e2e tests green** on sm_89 (7 existing + slot-mapping canary at K=32 / K=48 + determinism stress at short / prefill shapes, bit-exact across 100 runs each).

### INT4 `qkv_project_int4` → retained at Design S

The S+½P port was implemented, measured, and retained on branch but **not merged**. The prefill_m2048 ship gate was missed.

`ptxas_verify` on the INT4 S+½P kernel: sm_80 64 regs, sm_89 64 regs, 0 spills. Natural 88 regs — ptxas compressed 24 down without spilling, a much more aggressive compression than INT8's 2. Correctness gates all passed: 12/12 e2e (8 existing + slot canary K=128 / K=256 + determinism stress short/prefill 100× bit-exact).

**Bench (INT4 S+½P vs 3× standalone `matmul_int4`, RTX 4090 sm_89, median of 20 iters, 3 independent runs):**

| shape | Design S (7.3 archived) | S+½P (three runs) | median | decision |
|---|---|---|---|---|
| decode_m1 | 3.01× | 3.96× / 4.17× / 4.05× | 4.05× | (not gated) |
| decode_m64 | 3.00× | 4.01× / 3.99× / 4.00× | 4.00× | (not gated) |
| decode_m64_large | 2.98× | 3.70× / 3.59× / 3.71× | 3.70× | (not gated) |
| prefill_m512 | 1.19× | 1.83× / 1.82× / 1.94× | 1.83× | (not gated) |
| **prefill_m2048** | **0.85×** | **1.05× / 1.09× / 1.01×** | **1.05×** | **below ship gate** |

S+½P beats Design S at every shape. At `prefill_m2048` the design recovers from 0.85× to 1.05× (+24 percentage points) but misses the 1.15× ship threshold and the 1.10× ship-narrow floor. The plan's 5-tier outcomes table lands the INT4 result in **"Measured, not shipped"** (1.00×–1.10×) — a pre-declared tier, not a fallback.

**Interpretation.** Barrier reduction worked directionally but non-uniformly: `prefill_m2048` gained the least of any shape (+24% vs decode's +35% and prefill_m512's +54%). Pattern suggests barrier reduction exposed a memory-latency bottleneck at M=2048 / K=4096 / 256 K-tiles — the synchronous `ld.global → st.shared` inner loop stalls waiting on loads that were previously hidden behind barrier serialization. This is `cp.async` territory and a candidate for a future sprint; see follow-ups.

---

## Architectural decisions

### AD1 — Two-W-slot ping-pong with bank-phase pad

64 B padding between slots (576 B stride — non-multiple of 128). Prevents bank-phase alignment between slot0 and slot1 that would cause cross-warp SMEM bank-port contention when LDSM reads one slot while STS writes the other. Cost: 64 B shared memory. Both variants fit inside 3136 B total (under sm_89's 100 KB/SM).

### AD2 — `frag_A` register-hoisted per K-tile (tile_x overwrite invariant)

`frag_A` is loaded from shared once at K-tile start and reused across all three projections' mma epochs. This authorises the X_next cooperative load in Epoch 3 to overwrite `tile_x` while mma_V runs on the hoisted registers. Violating the invariant — re-reading `tile_x` after the hoist — races the overlap load against the reload and produces nondeterministic output. Enforced via emit-site comments in both kernels and the determinism stress test.

### AD3 — INT4 scales register-hoisted from global (Design invariant #2)

The INT4 S+½P port removes the `tile_scales` shared slot entirely; each lane loads its own column's f16 scale via `ld.global.f16` at K-tile start, passed as a register to the dequant helper. Categorically eliminates the 7.3 bug 3 class (scales coherence under pipelined loads — no shared intermediate exists to race). Unpredicated OOB load is safe because OOB columns have `tile_w` pre-zeroed and dequant produces `0 × scale = 0` regardless. Applies to the retained-on-branch INT4 S+½P code only; the Design-S INT4 kernel that ships is unchanged from 7.3.

### AD4 — Branch split: `phase7-ship` vs `phase7-rest`

`phase7-ship` merges to `main` with INT8 S+½P + INT4 Design-S. `phase7-rest` is retained as-is on the remote, carrying the INT4 S+½P port and its correctness gates as measured-data for a potential future `cp.async` contingency sprint. No merge-in-history of a tried-and-reverted arc; production kernel state matches the narrative in docs.

### AD5 — Correctness gate split: slot canary + determinism stress

Two distinct failure modes under S+½P need distinct tests. **Slot-mapping canary** (W_Q=1, W_K=2, W_V=3 with scale=1 — expected per-element outputs exactly `{K, 2K, 3K}`) catches deterministic ping-pong mis-wiring that random-data e2e can pass by coincidence. **Determinism stress** (same input replayed 100×, bit-exact assertion from run 1) catches the barrier-misplacement race class whose resolution varies with warp scheduling across launches. Both are needed; neither subsumes the other.

---

## Results

### Correctness — all green

- 84 kaio-ops host tests pass
- 11 INT8 GPU e2e tests pass (7 from 7.3 + 2 slot canaries + 2 determinism stress)
- 8 INT4 GPU e2e tests pass (7.3 suite; Design S kernel unchanged)
- `ptxas_verify` PASSED both sm_80 + sm_89: INT8 S+½P (64 / 56 regs, 0 spills), INT4 Design S (56 / 56 regs, 0 spills)
- `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo doc` all clean

### Performance

**INT8 S+½P (abs TOPS baseline capture, RTX 4090 sm_89).** INT8 has no fair 3×-standalone reference (`qkv_project_int8` is W8A16; `matmul_int8` is W8A8 with different arithmetic), so the INT8 shipping decision rests on correctness + abs-TOPS no-regression rather than a ratio gate.

| shape | S+½P abs TOPS |
|---|---|
| decode_m1 | 3.3 |
| decode_m64 | 3.3 |
| decode_m64_large | 10.1 |
| prefill_m512 | 34.8 |
| prefill_m2048 | 45.1 |

**INT4 Design S (unchanged on `main`).** Decode 3.0× median, prefill_m2048 0.85× — same shipping story as 7.3. Users needing prefill-heavy INT4 still call three separate `matmul_int4`s, as documented in the rustdoc.

### Scope — all D sections landed

- D1 register-budget skeleton (sm_80 32 / sm_89 40 regs, 0 spills) — ✅
- D2 INT8 S+½P kernel body — ✅
- D3 INT8 correctness gates (slot canary + determinism stress) — ✅
- D4 baseline reproducibility + INT8 abs-TOPS reference capture — ✅
- D5 INT4 S+½P kernel body (on `phase7-rest` archive) — ✅
- D6 INT4 correctness gates + bench decision (Measured, not shipped) — ✅
- D7 this doc + PHASE_7_LOG + CHANGELOG + README — ✅

---

## Follow-ups for future sprints

- **`cp.async` contingency.** The S+½P result localises the remaining gap at `prefill_m2048` to memory-latency dominance after barrier reduction succeeds. `cp.async` overlaps the `ld.global → st.shared` with mma compute at the hardware level (vs S+½P's macro-level overlap via the 2-slot ping-pong). Worth a fresh sprint, scoped explicitly against memory-latency-dominant regimes. The `phase7-rest` archive carries the S+½P baseline it can build on.
- **INT4 three-slot scales optimization** — independent of S+½P; eliminates the per-K-tile scale reload by giving each projection its own persistent `tile_scales_P` slot. Worth a bench-driven evaluation independently of the `cp.async` question.
- **Grouped-query attention** (`qkv_project_gqa`) — separate sprint in the Phase 7 tail.
- **Clean absolute-TOPS bench methodology** — cold-system single-test runs to avoid cross-variant thermal state contamination.
