# Sprint 5.3 — Causal Masking + Validation + DSL Friction Report

**Status:** Done
**Branch:** phase5
**Goal:** Add causal masking for autoregressive attention, expand
validation suite, document DSL friction from Sprints 5.2-5.3.

## What Was Built

`kaio_ops::attention_causal()` — causal masked attention. Same
three-kernel pipeline as `attention()` with a causal mask kernel
injected between Q*K^T and softmax.

### Causal Mask Kernel

`apply_causal_mask`: sets S[i,j] = -FLT_MAX where j > i. Uses
-3.402823e+38 (not -inf — DSL has no `f32::NEG_INFINITY`). Softmax
numerically stable: exp(-3.4e38 - max) ≈ 0 regardless of max.

Separate kernel for composability — unmasked `attention()` unchanged.
Extra kernel launch cost negligible at these sizes.

## Tests Added

| Test | What it verifies |
|------|------------------|
| `attention_causal_tiny` (4×4) | Basic causal correctness |
| `attention_causal_16x16` | Tile boundary with mask |
| `attention_causal_non_aligned` (seq=17, dk=19) | Non-aligned dims, 17×17 mask |
| `attention_causal_medium` (64×64) | Larger size |
| `attention_causal_row0_self_only` | Row 0 attends only to position 0, no NaN |
| `causal_mask_direct` | V=identity → out=P, verify lower-triangular weights, rows sum to 1 |

## DSL Friction Report (Sprints 5.2-5.3)

Per master plan, this sprint documents DSL friction encountered
during attention implementation. These feed Phase 6 DSL improvements.

### 1. No `&&`/`||` logical operators
**Where it hurts:** Bounds checking in tiled matmul requires 4 levels
of nested `if` instead of `if row < m && col < n`. The causal mask
kernel needs `if row < seq_len { if col < seq_len { if col > row { ... }}}`.
**Severity:** Medium. Verbose but not blocking.
**Fix complexity:** Parser + lowering change. Would need to emit
short-circuit branch logic (setp + bra for &&, two-path merge for ||).

### 2. 1D grid inference mismatch
**Where it hurts:** The softmax kernel needs one block per row
(grid.x = num_rows), but 1D auto-grid infers `ceil(N/block_size)`.
For seq_len < block_size, this gives 1 block instead of seq_len.
**Workaround:** `block_size = (256, 1)` makes the kernel 2D so it
takes an explicit grid tuple. Functionally identical to 1D.
**Severity:** Low. Workaround is clean. Only affects kernels where
grid size != ceil(N/block_size).
**Possible fix:** A `grid_size` attribute or explicit grid parameter.

### 3. No `sqrt()` in scalar host context
**Where it hurts:** `1.0 / sqrt(d_k as f32)` must be pre-computed
on the host and passed as a parameter. Can't do `let scale = 1.0 / sqrt(head_dim as f32);` inside the kernel on a u32-derived value.
**Severity:** Low. One extra parameter.

### 4. No compound shared memory assignment
**Where it hurts:** `tile[i] += val` must be written as
`tile[i] = tile[i] + val`. Softmax accumulation uses this.
**Severity:** Low. Pattern is well-known, just verbose.

### 5. No `f32::NEG_INFINITY` or `-inf` literal
**Where it hurts:** Causal mask needs a sentinel "negative infinity"
value. Used `-3.402823e+38` (-FLT_MAX) which works numerically but
looks fragile. A future contributor might try to "fix" it.
**Severity:** Low. Comment in code prevents accidental changes.

## Files

| File | Change |
|------|--------|
| `kaio-ops/src/attention_kernel.rs` | Added `apply_causal_mask` + `attention_causal()` |
| `kaio-ops/src/lib.rs` | Export `attention_causal` |
| `kaio-ops/tests/attention_api.rs` | +6 causal tests |
| `docs/development/sprints/phase5/PHASE_5_LOG.md` | Updated |
