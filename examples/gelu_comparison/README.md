# gelu_comparison — exact vs fast GELU, side by side

GELU is the activation in BERT, GPT-2/3, and most encoder-decoder
transformers. There are two common forms in production code: an
**exact** tanh-based approximation that matches the Gaussian CDF to
~1e-4, and a **fast** sigmoid-based approximation that trades
precision (~5e-3) for fewer ops.

This example writes both as KAIO kernels and measures them against
each other. It's the teaching moment for the kernel-variant workflow
— write two, measure both, pick the winner.

```rust
// Exact (tanh):  0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
#[gpu_kernel(block_size = 256)]
fn gelu_exact(x: *const [f32], out: *mut [f32], n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        let xi = x[idx];
        let c: f32 = 0.7978845608028654f32; // sqrt(2/π)
        let inner = c * (xi + 0.044715f32 * xi * xi * xi);
        out[idx] = 0.5f32 * xi * (1.0f32 + tanh(inner));
    }
}

// Fast (sigmoid):  x / (1 + exp(-1.702 * x))
#[gpu_kernel(block_size = 256)]
fn gelu_fast(x: *const [f32], out: *mut [f32], n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        let xi = x[idx];
        out[idx] = xi / (1.0f32 + exp(-1.702f32 * xi));
    }
}
```

## When to pick which

- **Exact:** research, new-model training, anywhere the precision
  actually matters. What PyTorch's `nn.GELU(approximate='tanh')` runs.
- **Fast:** inference, large-batch serving, mobile-class accelerators
  where a few extra ops cost more than ~5e-3 of activation drift. Used
  in some BERT inference stacks.

The numeric gap between the two is a choice you make per deployment,
not a bug in either implementation.

## The bandwidth-bound teaching moment

You will likely notice both variants run at **nearly identical speed**
on an RTX 4090 despite the fast variant doing fewer ops. That is not a
measurement bug — it is the single most important lesson in GPU
programming, and it is why this example exists.

These kernels are **bandwidth-bound**, not compute-bound. Each element
requires one global-memory load and one global-memory store regardless
of which formula runs in between. Modern GPU ALUs and tensor cores
finish the math long before the memory subsystem can feed them new
data, so shaving a tanh call saves you nothing wall-clock.

The implication for ML workloads is large: **kernel fusion —
combining multiple operations into one kernel to reduce memory
round-trips — matters far more than arithmetic optimization.** That
is why the `fused_silu_gate` sibling example matters (three elementwise
ops fused, one load + one store instead of six), why FlashAttention
was a paradigm shift (fuse attention into one kernel, never
materialize the QKᵀ matrix), and why production inference stacks obsess
over operator fusion rather than activation-function micro-tweaks.

If the fast variant runs on a compute-bound workload (small tensors,
aggressive re-use, or a truly compute-heavy surrounding op) the
savings return. Everywhere else, fuse.

## Running

```sh
cargo run --release
```

Requires an NVIDIA GPU with an installed driver. No CUDA toolkit needed.

## Output

```
=== gelu_comparison ===
Input size:        1048576 elements

Exact (tanh):      PASS  (max_abs_err = X.XXe-XX)  — XX.X μs
Fast (sigmoid):    PASS  (max_abs_err = X.XXe-XX)  — XX.X μs

Fast is X.X% of exact's time.
```
