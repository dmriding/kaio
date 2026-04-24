# rms_norm — single-block RMSNorm

RMSNorm is the normalization every LLaMA-family model uses — cheaper
than LayerNorm (no mean subtraction, no bias term) with comparable
training stability. LLaMA / Mistral / Qwen / Gemma / Yi all ship it.

```rust
#[gpu_kernel(block_size = 256)]
fn rms_norm(x: *const [f32], weight: *const [f32], out: *mut [f32], n: u32, eps: f32) {
    let tid = thread_idx_x();
    let mut val = 0.0f32;
    if tid < n {
        val = x[tid];
    }
    let sq = val * val;

    // Block-wide sum of squares via the reduction builtin.
    let sum_sq = block_reduce_sum(sq);

    let inv_rms = 1.0f32 / sqrt(sum_sq / (n as f32) + eps);

    if tid < n {
        out[tid] = val * inv_rms * weight[tid];
    }
}
```

## What it computes

```
rms     = sqrt(mean(x²) + eps)
out[i]  = (x[i] / rms) * weight[i]
```

## Single-block limitation

This example operates over `hidden_dim = 256`, which fits in a single
thread block at `block_size = 256`. **Real LLaMA RMSNorm runs over
`hidden_dim = 4096`** — 16 blocks at `block_size = 256`, which needs
cross-block reduction.

Multi-block RMSNorm lands when either:

1. KAIO's `#[gpu_kernel]` macro gains a cross-block reduction
   primitive (atomic accumulation into a scratch buffer, or a
   two-kernel split baked into a single builtin), **or**
2. KAIO ships `kaio_ops::rms_norm` as a pre-built operation that
   wraps the two-kernel version.

Both are post-v0.2.0 work. The single-block case is genuinely useful
as-is for attention-head-sized normalizations (`head_dim = 64` / 128)
and as a teaching example for the `block_reduce_sum` + sqrt + divide
pattern.

## Running

```sh
cargo run --release
```

Requires an NVIDIA GPU with an installed driver. No CUDA toolkit needed.

## Output

```
=== rms_norm ===
Input size:        256 elements  (single-block — see README)
Correctness:       PASS  (max_abs_err = X.XXe-XX)
Median latency:    XX.X μs  (of 100 timed runs, 5 warm-ups skipped)
```

Correctness is checked against an f64 CPU reference (double-precision
sum of squares, double-precision sqrt, cast back to f32) with
`max_abs_err < 1e-4`. The tolerance is looser than the pointwise
examples because the `block_reduce_sum → sqrt → divide` chain
accumulates more f32 rounding than a single element-wise op.
