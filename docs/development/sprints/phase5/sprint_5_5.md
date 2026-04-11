# Sprint 5.5 — Auto-Tuner

**Status:** Done
**Branch:** phase5
**Goal:** Benchmark kernel variants, cache results, dispatch
automatically.

## What Was Built

`kaio-ops/src/tuner.rs` — auto-tuner framework:
- `tune_matmul()` / `tune_attention()` / `tune_attention_causal()`
  — explicit benchmark entry points
- `matmul_auto()` / `attention_auto()` / `attention_auto_causal()`
  — dispatch from cache, fall back to default
- JSON cache at `~/.cache/kaio/tune_cache.json` (override with
  `KAIO_TUNE_CACHE` env var)

## Key Decisions

- **Internal enums** (`MatmulVariant`, `AttentionVariant`) for
  exhaustive dispatch. Strings only for serialization/public return.
- **Cache version field** (`{"version": 1, ...}`) — mismatch or
  parse failure → treat as empty, overwrite.
- **Duplicate entry behavior** — overwrite existing matching key.
- **Variant filtering** — flash attention skipped when d_k > 256.
  tune_attention() succeeds with "standard" even if flash ineligible.
- **`*_auto()` is pure dispatch** — read cache, choose variant,
  dispatch. No benchmarking, no file writes, no side effects.
  Cache miss → silently fall back to default.
- **Cache key** includes op mode: `"matmul"`, `"attention"`,
  `"attention_causal"` are separate keys. SM target from
  `device.info()?.compute_capability`.
- **Benchmark timing** — full operation (dispatch + GPU + sync),
  3 warm-up, 10 timed, median.

## Tests

| Test | Type | Status |
|------|------|--------|
| `tune_matmul_returns_variant` | GPU | Pass |
| `matmul_auto_produces_correct_output` | GPU | Pass |
| `tune_attention_returns_variant` | GPU | Pass |
| `attention_auto_produces_correct_output` | GPU | Pass |
| `tune_cache_roundtrip` | Host | Pass |
| `auto_falls_back_no_cache` | GPU | Pass |
| `tune_attention_skips_flash_when_dk_too_large` | GPU | Pass |

## Files

| File | Change |
|------|--------|
| `kaio-ops/Cargo.toml` | Added serde, serde_json deps |
| `kaio-ops/src/tuner.rs` | New — tuner framework |
| `kaio-ops/src/lib.rs` | Export tuner functions |
| `kaio-ops/tests/tuner_test.rs` | New — 7 tests |
| `docs/development/sprints/phase5/PHASE_5_LOG.md` | Updated |
