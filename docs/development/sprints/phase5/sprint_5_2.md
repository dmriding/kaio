# Sprint 5.2 — Standard Attention Forward Pass

**Status:** Done
**Branch:** phase5
**Goal:** Implement standard single-head attention as correctness
baseline for FlashAttention (Sprint 5.4).

## What Was Built

`kaio_ops::attention()` — standard (materialized) scaled dot-product
attention: `out = softmax(Q * K^T / sqrt(d_k)) * V`

Three-kernel decomposition:
1. `qk_scaled_matmul` — naive 16×16 tiled matmul with transposed K
   indexing + scale by 1/sqrt(d_k)
2. `row_softmax` — Phase 3 softmax pattern, one block per row
3. `matmul()` — reused existing kaio-ops matmul (no new kernel)

Single-head, single-sequence, f32 only, d_v == d_k. No masking
(Sprint 5.3). Materializes O(seq_len^2) intermediate buffers.

## Key Decisions

**Transposed K indexing:** K stored as (seq_len, d_k) row-major.
For Q*K^T, tile_b loads K[col_global * d_k + inner + ty] instead
of B[inner_row * N + col_global]. No host-side transpose needed.

**Softmax grid fix:** 1D auto-grid infers ceil(seq_len/256) blocks
from the last u32 param — gives 1 block for seq_len < 256, but we
need seq_len blocks (one per row). Fixed by using
`block_size = (256, 1)` which requires an explicit grid tuple.
Functionally identical to 1D (tidy always 0).

**Reused matmul():** P*V is a standard matmul —
`matmul(device, &probs, v, out, seq_len, d_k, seq_len)`. No new
kernel needed.

## DSL Friction Points (for Phase 6)

1. **No `&&`/`||` operators** — nested `if` for bounds checks.
   Verbose but not blocking. 4 nested `if` blocks in qk_scaled_matmul.
2. **1D grid inference** — can't control grid for 1D kernels.
   Workaround: `block_size = (256, 1)` for explicit grid. Would be
   cleaner with a `grid_size` attribute or parameter.
3. **No `sqrt()` in scalar context** — had to pass `inv_sqrt_dk` as
   pre-computed host parameter. Would be nice to have scalar math on
   host-computed values.

## Tests

| Test | seq_len | d_k | Status |
|------|---------|-----|--------|
| `attention_tiny` | 4 | 4 | Pass |
| `attention_16x16` | 16 | 16 | Pass |
| `attention_non_aligned` | 17 | 19 | Pass |
| `attention_medium` | 64 | 64 | Pass |
| `attention_identity` | 8 | 8 | Pass (Q==K==V) |
| `attention_rejects_zero_seq_len` | 0 | — | Pass |
| `attention_rejects_zero_dk` | — | 0 | Pass |

Tolerance: abs_err < 1e-3 || rel_err < 1e-2.

## Files

| File | Change |
|------|--------|
| `kaio-ops/src/attention_kernel.rs` | New — 2 kernels + host API |
| `kaio-ops/src/lib.rs` | Added attention export |
| `kaio-ops/tests/attention_api.rs` | New — 7 tests |
| `docs/development/sprints/phase5/PHASE_5_LOG.md` | Updated |

## Review Notes

- Opus 4.6: verify matmul arg ordering (M=seq_len, N=d_k, K=seq_len),
  use relative error for medium+ sizes, PTX dump to check K^T addressing
- Codex 5.4: document memory-heavy nature, distinguish from FlashAttention,
  add structured Q==K==V test
