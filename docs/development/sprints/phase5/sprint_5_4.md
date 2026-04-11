# Sprint 5.4 — FlashAttention (Online Softmax + Tiled Attention)

**Status:** Done
**Branch:** phase5
**Goal:** Implement FlashAttention — attention without materializing
the O(seq_len^2) attention matrix.

## What Was Built

Both sub-steps landed:

**(a) Online softmax** — proven via the FlashAttention kernel's inner
loop. Running (m, l) state maintained across K/V tiles with correct
rescaling. Validated against standard attention output.

**(b) FlashAttention kernel** — `flash_attn_kernel` and
`flash_attn_causal_kernel`. BLOCK_M = 1 design (one query position
per block, 256 threads). O(d_k + 256) memory per block.

### BLOCK_M = 1 Design

Avoids per-row reductions (KAIO DSL limitation). All 256 threads
cooperate on one query position:

1. **Phase 1:** Each thread computes one attention score (dot product
   over d_k), writes to shared memory
2. **Phase 2:** Online softmax — block_reduce_max, rescale existing
   accumulators by exp(m_old - m_new), block_reduce_sum
3. **Phase 3:** First d_k threads accumulate P*V from shared memory
   (sequential over 256 keys per thread)

**Invariant after tile t:**
- m_t = max(all scores seen)
- l_t = sum(exp(score_j - m_t))
- O_t[d] = sum(exp(score_j - m_t) * V[j, d])
- Final output: O_t / l_t

### Grid Fix

Same issue as Sprint 5.2 softmax: 1D auto-grid infers from last u32
param (d_k), giving 1 block. Fixed with `block_size = (256, 1)` for
explicit grid tuple `(seq_len, 1, 1)`.

## Key Decisions

- **Separate flash APIs** — `attention_flash()` and
  `attention_flash_causal()` alongside standard attention. Standard
  attention remains the shipping default and correctness oracle.
- **Two kernel functions** — causal variant is separate for clarity.
  One `if` difference in Phase 1 (mask j > q_row).
- **d_k <= 256 runtime check** — clear error instead of silent
  corruption.
- **-FLT_MAX sentinels** — OOB and masked positions set to
  -3.402823e+38. Assumption: every query row has at least one valid
  key (causal self-attention guarantees the diagonal).

## Performance Notes

Flash wins on memory (O(d_k) vs O(seq_len^2)), but V accumulation is
sequential per thread (256 iterations per tile). Not optimized for
peak throughput — this is a correctness proof. Future: BLOCK_M > 1
with per-row reductions would improve parallelism.

## Tests

| Test | Status |
|------|--------|
| `flash_attention_tiny` (4×4) | Pass |
| `flash_attention_16x16` | Pass |
| `flash_attention_non_aligned` (17×19) | Pass |
| `flash_attention_medium` (64×64) | Pass |
| `flash_attention_causal_medium` | Pass |
| `flash_matches_standard` (32×32) | Pass |
| `flash_causal_first_rows` | Pass |
| `flash_all_in_one_tile` (8×8) | Pass |
| `flash_last_tile_partial` (257×32) | Pass |

All 22 attention tests pass (13 standard + 9 flash).

## Files

| File | Change |
|------|--------|
| `kaio-ops/src/attention_kernel.rs` | +2 kernels, +2 host APIs, +1 validation fn |
| `kaio-ops/src/lib.rs` | Export flash APIs |
| `kaio-ops/tests/attention_api.rs` | +9 flash tests |
| `docs/development/sprints/phase5/PHASE_5_LOG.md` | Updated |
