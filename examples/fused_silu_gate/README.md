# fused_silu_gate — SiLU-gate in 7 lines of Rust

The gated activation inside every LLaMA / Mistral / Qwen feedforward
block. `llama.cpp`, vLLM, and TensorRT-LLM all ship hand-written CUDA
for this. With KAIO you write it once, in Rust, and the `#[gpu_kernel]`
macro lowers it to PTX at compile time.

```rust
#[gpu_kernel(block_size = 256)]
fn fused_silu_gate(x: *const [f32], gate: *const [f32], out: *mut [f32], n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        let xi = x[idx];
        let sig = 1.0f32 / (1.0f32 + exp(-xi));
        out[idx] = xi * sig * gate[idx];
    }
}
```

## What it computes

```
out[i] = x[i] * sigmoid(x[i]) * gate[i]
       = x[i] * (1 / (1 + exp(-x[i]))) * gate[i]
```

## Why it matters

The SwiGLU / SiLU-gate pattern is the modern-transformer default for
feedforward activations — it's what replaced ReLU in the LLaMA family
and has been adopted across most open-weights models released since
2023. Production inference stacks fuse the three element-wise ops into
one kernel because the memory traffic, not the math, is the bottleneck.

## Running

```sh
cargo run --release
```

Requires an NVIDIA GPU with an installed driver. No CUDA toolkit needed.

## Output

```
=== fused_silu_gate ===
Input size:        1048576 elements
Correctness:       PASS  (max_abs_err = 7.45e-07)
Median latency:    XX.X μs  (of 100 timed runs, 5 warm-ups skipped)
```

Correctness is checked against an f64 CPU reference (the Rust math
library's `exp` rounded to f32) with `max_abs_err < 1e-5`.
