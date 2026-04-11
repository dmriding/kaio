# Sprint 3.6 — Softmax Kernel

**Status:** Complete
**Commit:** `b9c9473`
**Date:** 2026-04-11
**Depends on:** Sprint 3.5 (reductions)

## Context

Phase 3 capstone. Implements row-wise softmax using all Phase 3 features:
while loops, compound assignment (+=), block_reduce_max, block_reduce_sum,
exp(), and compound index expressions (input[row_offset + i]).

No new language features added — pure test code validating that everything
composes correctly in a real GPU algorithm.

## Decisions

### Variable shadowing not supported — use unique names

**Context:** The master plan's softmax kernel used `let mut i = tid;` three
times (once per while loop). Rust allows shadowing, but the KAIO lowering
rejects duplicate variable names.

**Decision:** Use unique loop variables (`i1`, `i2`, `i3`) in the kernel.
Variable shadowing support is a Sprint 3.8 polish item or Phase 4 feature.

### Single-row testing (launch wrapper limitation)

**Context:** The launch wrapper uses the last u32 parameter as element count.
For softmax, this is `row_len`, not `num_rows`.

**Decision:** Test single-row softmax only. Multi-row requires custom
LaunchConfig (deferred to Phase 4+).

## Scope

**In:**
- Softmax kernel using loops + reductions + exp + compound index expressions
- CPU reference function for validation
- 5 GPU E2E tests: correctness, uniform, all-zeros, large values, sums-to-one

**Out:**
- Multi-row softmax (needs custom LaunchConfig)
- Variable shadowing support
- PyTorch comparison at multiple sizes (Sprint 3.7)

## Results

All 5 softmax tests pass on RTX 4090:

- `softmax_small_row` — 128 elements [0..127], max abs error < 1e-5 vs CPU
- `softmax_uniform` — all 1.0 → uniform 1/128, within 1e-5
- `softmax_all_zeros` — all 0.0 → uniform 1/128, within 1e-5
- `softmax_large_values` — [1000..1127], no inf/nan, error < 1e-4
- `softmax_sums_to_one` — output sums to 1.0 within 1e-4

**Compound index expression `input[row_offset + i]` works correctly.**
The recursive `lower_expr` design handles arbitrary expressions as array
indices — the BinOp(Add) is lowered to a temporary register which becomes
the index for the ld.global address computation.

**Files created:** 1
- `kaio/tests/softmax_macro.rs` (5 GPU E2E tests)

**Tests:** 198 host + 19 GPU, all passing.

**Quality gates:** `cargo fmt --check` clean, `cargo clippy -- -D warnings`
zero warnings, all tests pass.
