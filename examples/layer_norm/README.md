# layer_norm — single-block LayerNorm

LayerNorm is the classic transformer normalization used by BERT, GPT-2,
T5, and many encoder-decoder stacks. Unlike RMSNorm, it subtracts the
mean and applies both a learned scale (`gamma`) and bias (`beta`).

```rust
#[gpu_kernel(block_size = 256)]
fn layer_norm(x: *const [f32], gamma: *const [f32], beta: *const [f32], out: *mut [f32], n: u32, eps: f32) {
    let tid = thread_idx_x();

    let mut val = 0.0f32;
    if tid < n {
        val = x[tid];
    }

    let sum = block_reduce_sum(val);
    let mean = sum / (n as f32);

    let mut centered = 0.0f32;
    if tid < n {
        centered = val - mean;
    }
    let var_sum = block_reduce_sum(centered * centered);
    let inv_std = 1.0f32 / sqrt(var_sum / (n as f32) + eps);

    if tid < n {
        out[tid] = centered * inv_std * gamma[tid] + beta[tid];
    }
}
```

## What it computes

```
mean    = sum(x) / n
var     = sum((x - mean)^2) / n
inv_std = 1 / sqrt(var + eps)
out[i]  = ((x[i] - mean) * inv_std) * gamma[i] + beta[i]
```

## Why it matters

LayerNorm is one of the most recognizable ML primitives: it is the
default normalization in classic transformer stacks and still shows up
in encoders, diffusion models, and smaller attention blocks even where
RMSNorm has replaced it in frontier LLMs.

This example is also a clean demonstration of KAIO's current reduction
surface: two block-wide sums plus element-wise affine math, all inside
one ordinary Rust kernel.

## Single-block limitation

This example intentionally handles only the case `n <= block_size`
(`256` here), so the entire normalization fits in one block.

That is a limitation of the example, not a claim about KAIO's long-term
LayerNorm capability. Larger hidden sizes need cross-block reduction or
a multi-kernel split, which is outside the current example's scope.

## Running

```sh
cargo run --release
```

Requires an NVIDIA GPU with an installed driver. No CUDA toolkit needed.

## Output

```
=== layer_norm ===
Input size:        256 elements  (single-block — see README)
Correctness:       PASS  (max_abs_err = X.XXe-XX)
Median latency:    XX.X μs  (of 100 timed runs, 5 warm-ups skipped)
```

Correctness is checked against an f64 CPU reference for the mean,
variance, reciprocal standard deviation, and final affine transform.
The reported latency is the median wall-clock time across 100
synchronized launches after 5 warm-up runs.
