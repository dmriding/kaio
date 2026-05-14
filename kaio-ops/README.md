# kaio-ops

Pre-built GPU operations for [KAIO](https://github.com/dmriding/kaio).

Users don't need to write kernels for the ops listed here — just allocate
inputs on a `KaioDevice` and call the function. All ops emit and JIT-load
PTX through the same compiler the rest of KAIO uses, so the codegen path
is identical to user-authored `#[gpu_kernel]` code.

Constraint shorthand: **SM 8.0+** means Ampere or newer (RTX 3000 / A100
and up); **any SM** means the op falls back to a scalar kernel that runs
on Volta+ (RTX 2000 / V100 and up).

## Matrix multiplication

| Op | Inputs / output | Min SM | Divisibility | Notes |
|----|-----------------|--------|--------------|-------|
| `matmul` | f32 × f32 → f32 | any | none | Scalar tiled kernel (64×64 block, 4×4 register tile). Edge-tile predication on all axes. |
| `matmul_tc` | f16 × f16 → f32 | 8.0+ | `K % 16 == 0` | Synchronous tensor-core path. 4-warp 32×32 quadrant, 8× `mma.sync.m16n8k16` per K-tile, bank-conflict-padded shared Tile B. |
| `matmul_tc_async` | f16 × f16 → f32 | 8.0+ | `K % 16 == 0` | Async sibling of `matmul_tc` — overlaps shared-tile loads with compute via `cp.async`. |
| `matmul_tc_bf16` | bf16 × bf16 → f32 | 8.0+ | `K % 16 == 0` | bf16 tensor-core sync matmul (Sprint 9.1). Same layout as `matmul_tc` with `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` on the hot path. |
| `matmul_tc_bf16_async` | bf16 × bf16 → f32 | 8.0+ | `K % 16 == 0` | bf16 tensor-core async matmul (Sprint 9.1.1). cp.async-pipelined sibling of `matmul_tc_bf16` — cross-product of (f16 async × bf16 sync). Same staging contracts as `matmul_tc_async` with `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` on the hot path. |
| `matmul_int8` | i8 × i8 → f32 | 8.0+ | `K % 32 == 0` | W8A8 symmetric dequant-matmul with a single scalar scale applied post-accumulation. Uses `mma.sync.m16n8k32.s8.s8.s32`. |
| `matmul_int4` | packed-s4 × f16 → f32 | 8.0+ | `K % 128 == 0`, `group_size = 128` | W4A16 GPTQ-style: 4-bit signed weights, f16 activations, f16 per-group scales. Dequant-to-f16 path through `mma.sync.m16n8k16`. Reference quant op — not a drop-in for external GPTQ/GGUF formats. |

### Auto-tuned variants

```rust
use kaio_ops::{tune_matmul, matmul_auto, matmul_auto_tc};

tune_matmul(&device, 1024, 1024, 1024)?;        // benchmark once
matmul_auto(&device, &a, &b, &mut c, m, n, k)?; // picks best variant from cache
```

- `matmul_auto` — dispatches across the scalar `matmul` family.
- `matmul_auto_tc` — dispatches across the f16 tensor-core family
  (`matmul_tc` vs `matmul_tc_async`, etc.).
- `tune_matmul` / `tune_matmul_tc` — explicit tuning entry points;
  results land in the on-disk JSON cache reused by `matmul_auto*`.

## Attention

| Op | Variant | Min SM | Notes |
|----|---------|--------|-------|
| `attention` | scalar, materialized | any | Standard scaled dot-product attention. Companion: `attention_causal` for causal masking. |
| `attention_flash` | scalar, FlashAttention | any | O(d_k) memory — no attention matrix materialization. `d_k ≤ 256`. Companion: `attention_flash_causal`. |

Auto-tuned: `attention_auto`, `attention_auto_causal`,
`tune_attention`, `tune_attention_causal`.

```rust
use kaio_ops::attention_flash;

attention_flash(&device, &q, &k, &v, &mut out, seq_len, d_k)?;
```

## Fused tri-output QKV projection

Single launch producing three `GpuBuffer<f16>` outputs ready for the
attention path. Saves 2× global activation reads vs three standalone
matmul calls. Per-block tile `64 × 16` (the D3.4 rollback that resolved
the register-pressure trigger).

| Op | Weight format | Min SM | Divisibility |
|----|---------------|--------|--------------|
| `qkv_project_int8` | i8 weights, f16 activations, scalar per-projection scales (W8A16) | 8.0+ | `K % 16 == 0`, `N % 2 == 0` |
| `qkv_project_int4` | packed-INT4 weights, f16 activations, f16 group scales (W4A16, group_size 128) | 8.0+ | `K % 128 == 0`, `N % 2 == 0` |

```rust
use kaio_ops::qkv_project_int8;

qkv_project_int8(&device, &x, &w_q, &w_k, &w_v, s_q, s_k, s_v,
                 &mut q_out, &mut k_out, &mut v_out, m, n, k)?;
```

## Requirements

- NVIDIA GPU (SM 7.0+); tensor-core / quantized ops require SM 8.0+
- CUDA driver installed
- Rust 1.94+

## Performance

Published numbers live in
[`docs/performance.md`](../docs/performance.md) at the root of the
KAIO repository. Per-op perf is updated at phase boundaries, not on
every sprint — mid-phase performance claims for sprints in flight are
deliberately deferred to the next aggregate release.

Apples-to-apples disclaimers:

- Tensor-core matmul is **f16 × f16 → f32**, vs cuBLAS sgemm
  **f32 × f32 → f32**. The KAIO ops are accumulator-precision-equivalent
  to cuBLAS sgemm but trade input precision for tensor-core throughput.
- The bf16 tensor-core matmul compares to cuBLAS sgemm at the same
  caveat; a bf16-vs-`cublasGemmEx-bf16` reference is tracked in
  `docs/development/tech_debt.md`.

## License

MIT OR Apache-2.0
