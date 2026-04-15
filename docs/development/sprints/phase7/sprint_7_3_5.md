# Sprint 7.3.5 — Design S+½P: two-W-slot ping-pong for fused QKV projection

**Status:** 📝 Planned (not yet reviewed)
**Branch:** `phase7-rest` (continuing)
**Release target:** None — rolls into Phase 7 aggregate release after 7.4.

---

## Context

Sprint 7.3 shipped fused `qkv_project_int{4,8}` with Design S (serial
fusion, one shared W slot reused per projection). The D8 bench showed:

- **Decode tier (M ≤ 128)**: fused wins ~3.0× over 3× standalone — huge.
- **Prefill mid (M = 512)**: 1.19× — passes ship threshold.
- **Prefill large (M = 2048)**: **0.85×** — fused *loses* by 15%.

Per the Sprint 7.3 plan's D4.5 escape ladder, this triggers Design
S+½P: two shared W slots with ping-pong index, overlapping the
cooperative load of `W_{P+1}` with the mma compute of `W_P`. Barriers
drop from **7 to 4 per K-tile**. Target: recover `prefill_m2048` from
0.85× to ≥ 1.15×.

Scoped as a separate sprint rather than inline because it changes
synchronization semantics in the inner loop — warrants the full
reviewer cadence (CC + Opus + Codex + Gemini maintainer review), not a
hotfix on a working kernel.

## Scope

- Extend `qkv_project_int{4,8}_kernel.rs` to support a 2-W-slot
  shared-memory layout.
- Rewrite the K-loop inner sequence to overlap W load + mma.
- Keep correctness with the same 15 GPU e2e tests from 7.3 (no new
  correctness tests expected).
- Rerun the D8 bench. Ship criterion: `prefill_m2048` recovers to
  ≥ 1.15× over 3× standalone without regressing decode tier.
- If the bench target misses, Design S stays and the sprint closes as
  "measured, rejected" — the docs from 7.3 remain the answer for
  prefill-heavy users.

## Out of scope

- Any kernel structural changes beyond the W-slot pipelining (no new
  tile dimensions, no register-budget refactor, no dequant changes).
- INT4 group-scale three-slot optimization (separate follow-up; cheap
  relative to the Design S+½P rework and independent).
- New public API surface — same `qkv_project_int{4,8}` signatures.
- New bench shapes beyond the 9 already defined in 7.3 D8.

## Design sketch (pre-review)

### Shared memory layout

- `tile_x` — 2048 B (unchanged).
- `tile_w[0]` — 512 B (unchanged).
- `tile_w[1]` — 512 B (**new** — second W slot).
- `tile_scales` — unchanged (INT4 only).
- Total shared budget: 2048 + 1024 + 32 = **3104 B** for INT4, or
  2048 + 1024 = **3072 B** for INT8. Well under sm_89's 100 KB/SM
  shared capacity.

### K-loop inner sequence (per K-tile)

```text
pre-loop: load W_Q into tile_w[0], load X, bar.sync (X + W_Q visible)

for k_tile in 0..num_k_tiles:
  load W_K into tile_w[(k_tile+1) % 2]     # overlap — writes to NEXT slot
  mma_Q using tile_w[k_tile % 2]           # reads CURRENT slot
  bar.sync                                 # sync: W_K load visible, mma_Q done

  load W_V into tile_w[(k_tile+0) % 2]     # overlap — writes back to Q slot
  mma_K using tile_w[(k_tile+1) % 2]       # reads the slot we just wrote
  bar.sync                                 # sync: W_V load visible, mma_K done

  (if not last K-tile)
  load W_Q_next into tile_w[(k_tile+1) % 2]# overlap — writes back to K slot
  mma_V using tile_w[k_tile % 2]           # reads Q slot (has W_V)
  bar.sync                                 # sync: W_Q_next visible, mma_V done

  (next K-tile: load X_next overlapping with above, bar.sync on X visible)
```

**Barriers per K-tile:** 1 X-sync + 3 projection-epoch syncs = **4**
(vs 7 in Design S). Three W loads still happen per K-tile, just
overlapped with mma work.

Ping-pong indexing: `tile_w[(k_tile + P_index) % 2]` with a per-
projection static phase `P_index ∈ {0, 1, 0}` for `{Q, K, V}`.

### Register pressure impact

Adding a second W slot's scales register is +1 register (for INT4;
INT8 has no scales state). Overlapping load/mma adds per-K-tile
address-math registers (+3 to +5 estimate based on matmul_tc_async's
pipelining pattern). Expected post-Design-S+½P:

- INT8: 48 regs → ~51 regs (within the 64-reg cliff)
- INT4: 56 regs → ~60 regs (tight — D2.5-style skeleton checkpoint
  needed before the full rewrite)

If either variant overshoots, Rollback #2 (INT8-only S+½P, INT4 stays
Design S) is the named fallback.

## Decisions

### D1 — Register-budget skeleton (INT4 first)

Port `qkv_skeleton.rs` to the 2-W-slot pattern: allocate everything
at peak, run `ptxas -v`, confirm N ≤ 64 / 0 spills. If INT4 clears the
budget, proceed to the full kernel; if not, apply Rollback #2 and
continue with INT8-only S+½P.

### D2 — INT8 S+½P kernel body + ptxas_verify

Two W slots, ping-pong, overlapped load/mma. Reuse existing
cooperative W loader unchanged — the only change is which shared slot
the load writes to.

### D3 — INT8 e2e regression

Re-run the existing 7 INT8 e2e tests against the S+½P kernel.
Expected: all pass; correctness is not what S+½P changes.

### D4 — INT8 D8 bench re-run

Run the 9-shape sweep, confirm `prefill_m2048` is ≥ 1.15×. If miss,
document as "measured, rejected" and close with Design S remaining in
production. If hit, proceed to INT4.

### D5 — INT4 S+½P kernel body (contingent on INT8 landing)

Same pattern applied to the INT4 kernel.

### D6 — INT4 e2e regression + bench

Same shape as D3 + D4 for INT4.

### D7 — Sprint doc, CHANGELOG update, rustdoc perf table refresh

## Reviewer cadence

Full CC + Opus + Codex + Gemini + maintainer review before execution,
per Dave's standing workflow. Barrier semantics changes in the inner
loop are the exact class of thing that benefits most from the multi-
reviewer pass.

## Success criterion (plan-locked)

**Single number:** `prefill_m2048` ratio goes from **0.85×** to
**≥ 1.15×** on RTX 4090 sm_89 with no correctness regression.
