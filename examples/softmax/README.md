# softmax — single-block softmax

Softmax is the normalization at the heart of attention: it turns a row
of scores into a probability distribution by exponentiating and
normalizing.

```rust
#[gpu_kernel(block_size = 256)]
fn softmax(input: *const [f32], output: *mut [f32], n: u32) {
    let tid = thread_idx_x();

    let mut local_max = -3.402823e38f32;
    if tid < n {
        local_max = input[tid];
    }
    let row_max = block_reduce_max(local_max);

    let mut exp_val = 0.0f32;
    if tid < n {
        exp_val = exp(input[tid] - row_max);
    }
    let row_sum = block_reduce_sum(exp_val);

    if tid < n {
        output[tid] = exp_val / row_sum;
    }
}
```

## What it computes

```
row_max = max(x)
row_sum = sum(exp(x - row_max))
out[i]  = exp(x[i] - row_max) / row_sum
```

Subtracting `row_max` is the standard stability trick: it prevents the
exponential from overflowing on large positive inputs while leaving the
final probabilities unchanged.

## Why it matters

Softmax is one of the load-bearing primitives in transformer attention.
Even when more advanced fused attention kernels replace it in production,
row-wise softmax remains the clearest example of reduction-heavy GPU
math: max reduction, sum reduction, then normalization.

## Single-block limitation

This example intentionally handles only the case `n <= block_size`
(`256` here), so one block owns the whole row.

That is a limitation of the example, not a claim about KAIO's long-term
softmax capability. Larger rows need striding, tiling, or multi-block
coordination, which is outside this example's scope.

## Running

```sh
cargo run --release
```

Requires an NVIDIA GPU with an installed driver. No CUDA toolkit needed.

## Output

```
=== softmax ===
Input size:        256 elements  (single-block — see README)
Correctness:       PASS  (max_abs_err = X.XXe-XX)
Median latency:    XX.X μs  (of 100 timed runs, 5 warm-ups skipped)
```

Correctness is checked against an f64 CPU reference for the max,
exponentials, sum, and final normalization. The reported latency is the
median wall-clock time across 100 synchronized launches after 5 warm-up
runs.
