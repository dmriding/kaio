# Sprint 7.4d — matmul_tc + matmul_tc_async backward

**Status:** ✅ Complete
**Branch:** `phase7-wrap`
**Release target:** bundled into Phase 7 aggregate release (v0.4.0).

---

## Context

7.4a–c shipped 8 forward ops + event-based sync. The `CustomOp2::bwd()` method on `MatmulTcOp` and `MatmulTcAsyncOp` defaulted to `BackwardNotSupported`. Sprint 7.4d implements backward for these two ops, proving the candle autograd integration pattern.

## What shipped

Analytical backward for `matmul_tc` and `matmul_tc_async`: `dA = grad @ B^T`, `dB = A^T @ grad`. Reuses the existing forward kernels — zero new PTX. Numerically approximate (f32 gradient downcast to f16 for TC kernel reuse).

### Changes

- `MatmulTcOp::bwd()` — computes both input gradients via two `matmul_tc()` forward calls with transposed inputs. Casts f32 `grad_res` to f16 before matmul, casts f32 output gradients to f16 before returning (candle's gradient accumulation requires dtype matching).
- `MatmulTcAsyncOp::bwd()` — same implementation using `matmul_tc_async()` for consistent kernel variant in both directions.
- `.t()?.contiguous()?` materializes transposed layouts (bridge rejects non-contiguous inputs).
- lib.rs + README updated with backward support status, tier rationale, and memory/precision caveats.

### Tier rationale (why only these two?)

| Tier | Ops | Status | Reason |
|---|---|---|---|
| 1 | `matmul_tc`, `matmul_tc_async` | ✅ Shipped | Pure bridge code — two forward calls per backward, no new kernels |
| 2 | `attention_tc`, `attention_tc_causal` | Phase 8 | FlashAttention backward requires new PTX kernels |
| 3 | `matmul_int4`, `matmul_int8`, `qkv_project_int{4,8}` | Deferred | Quantized inference ops — frozen weights, no backprop in practice |

### Tests

- 6 gradient-correctness tests added: numerical finite-difference checks against analytical backward.
- 3 shape variants per op: small square (32x32x32), medium square (128x128x128), non-square (64x32 x 32x128).
- Scalar loss via `sum_all()`, small-magnitude inputs near zero for f16 stability.
- Dual tolerance: `rel_err < 1e-2 || abs_err < 1e-3`.
- **Total kaio-candle GPU tests: 39** (was 33).

### What didn't change

- Zero new PTX kernels. Backward reuses forward kernels.
- Zero API changes to the public surface. All 8 ops have the same call signatures.
- All 33 existing GPU tests pass unchanged.

---

## Known limitations

- **Numerically approximate:** f32 upstream gradient is downcast to f16 before the tensor-core matmul, and output gradients are cast back to f16. The double f16 cast is a known precision approximation — negligible for a single backward pass, may compound in deep networks. This is an initial autograd integration, not a final mixed-precision training stack.
- **Memory overhead:** backward materializes two transposed tensors in VRAM (`.contiguous()` = allocation + copy) plus the casted gradient. Peak backward memory ≈ 2–3x forward input size.

## Follow-ups

- **Phase 7 close** — `phase7-wrap → phase7-ship → main` merge, v0.4.0 aggregate release.
- **kaio 0.3.1 patch + kaio-candle 0.1.0 publish** — `dynamic-linking` feature needed on crates.io.
- **Attention backward (Phase 8)** — FlashAttention backward requires new tiled PTX kernels with softmax recomputation.
