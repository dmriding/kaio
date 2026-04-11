# Sprint 4.9 — Adoption Polish

**Status:** In Progress
**Branch:** phase4.9-adoption
**Goal:** Make KAIO frictionless for first-time users — runnable
examples, README rewrite with positioning/limitations/patterns,
crates.io landing page improvements.

## Context

Codex reviewed the repo from an outsider's perspective and identified
7 adoption barriers. The technical work is solid (207 host tests, 31%
cuBLAS matmul) but the "can I run this right now?" experience is poor.
This sprint adds no new features — it makes what exists easy to try.

## Deliverables

1. **4 runnable examples** (`cargo run --example`)
   - `vector_add` — simplest possible GPU kernel
   - `saxpy` — scalar parameters
   - `reduction` — shared memory + block_reduce_sum with real data
   - `matmul` — kaio-ops high-level API

2. **README.md rewrite** — new sections:
   - "When to Use KAIO" — positioning vs Candle/Burn/cudarc/raw CUDA
   - "Examples" table with star marker on vector_add
   - "Patterns" — 3 copy-paste skeletons (5-8 lines each)
   - "Limitations" — honest list including DSL gaps
   - "Gotchas" — 3 common pitfalls
   - "Feedback" — actionable CTA
   - Quick Start now prints output + shows expected terminal output

3. **kaio/README.md** — crates.io landing page expanded (<150 lines)

## Review Notes

- Opus 4.6: reduction example must use real data (not synthetic
  constants), patterns must stay 5-8 lines, crates.io README < 150 lines
- Codex 5.4: star-mark vector_add as entry point, add gotchas section,
  framing sentence for "When to Use KAIO", failure expectation note
  after quickstart

## Files

| File | Change |
|------|--------|
| `kaio/examples/vector_add.rs` | New |
| `kaio/examples/saxpy.rs` | New |
| `kaio/examples/reduction.rs` | New |
| `kaio-ops/examples/matmul.rs` | New |
| `README.md` | Rewrite |
| `kaio/README.md` | Expand |
| `docs/development/sprints/phase4/PHASE_4_LOG.md` | Add 4.9 row |

## Test Impact

No new tests. Examples are binaries (`fn main()`), not test harness.
Existing 207 host + 43 GPU + 1 bench unchanged.
