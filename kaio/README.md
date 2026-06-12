# KAIO

**Rust-native GPU kernel authoring framework.**

Write GPU compute kernels in Rust, compile to PTX, run on NVIDIA GPUs.
No CUDA C++, no Python, no CUDA toolkit required. Windows + Linux.

## Quick Start

```rust
use kaio::prelude::*;

#[gpu_kernel(block_size = 256)]
fn saxpy(x: *const [f32], y: *mut [f32], alpha: f32, n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        y[idx] = alpha * x[idx] + y[idx];
    }
}

fn main() -> Result<()> {
    let device = KaioDevice::new(0)?;
    let n = 1024u32;

    let x = device.alloc_from(&vec![1.0f32; n as usize])?;
    let mut y = device.alloc_from(&vec![2.0f32; n as usize])?;

    saxpy::launch(&device, &x, &mut y, 2.5f32, n)?;

    let result = y.to_host(&device)?;
    println!("result: {:?}", &result[..8]);
    // prints: result: [4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5]
    Ok(())
}
```

Requires an NVIDIA GPU with driver installed. No CUDA toolkit needed.

## When to Use KAIO

KAIO is not a replacement for ML frameworks like Candle or Burn.
It is the layer you use when you need more control than they provide.

- Write custom GPU kernels your framework doesn't support
- Control GPU memory explicitly — deterministic VRAM, buffer reuse
- Ship GPU binaries without Python or Triton in your dependency chain
- Run on Windows (Triton is Linux-only)
- Prototype GPU code in Rust without learning CUDA C++

## Features

| Feature | Syntax | Status |
|---------|--------|--------|
| Arithmetic | `+`, `-`, `*`, `/`, `%`, `+=`, `-=`, `*=`, `/=` | Supported |
| Comparisons | `<`, `<=`, `>`, `>=`, `==`, `!=` | Supported |
| Control flow | `if`/`else`, `for`, `while` | Supported |
| Array access | `a[idx]` (global memory) | Supported |
| Shared memory | `shared_mem![f32; 256]` | Supported |
| Synchronization | `bar_sync()` | Supported |
| Warp shuffle | `shfl_sync_down/up/bfly()` | Supported |
| Reductions | `block_reduce_sum()`, `block_reduce_max()` | Supported |
| Type casts | `x as f32` | Supported |
| Math builtins | `sqrt`, `exp`, `log`, `tanh`, `abs`, `min`, `max` | Supported |
| FMA | `fma(a, b, c)` | Supported |
| 2D blocks | `block_size = (16, 16)`, `thread_idx_y()` | Supported |
| Tiled matmul | `kaio_ops::matmul()` (scalar f32 baseline) | Supported |
| Tensor-core matmul | `kaio_ops::matmul_tc()` / `matmul_tc_bf16()` + `cp.async` siblings — f16/bf16 → f32, SM 8.0+ | Supported |
| Quantized matmul | `kaio_ops::matmul_int8()` (W8A8), `matmul_int4()` (W4A16) + fused QKV projections | Supported |
| Attention | `kaio_ops::attention()`, `attention_causal()` | Supported |
| FlashAttention | `kaio_ops::attention_flash()` — O(d_k) memory | Supported |
| FlashAttention backward | `kaio_ops::attention_flash_bwd()` + causal / `_with_stats` forwards | Supported |
| Auto-tuner | `kaio_ops::tune_matmul()`, `matmul_auto()`, `matmul_auto_tc()`, `matmul_auto_tc_bf16()` | Supported |

## Architecture

| Crate | Description |
|-------|-------------|
| `kaio` | Umbrella crate — re-exports everything via `prelude` |
| `kaio-macros` | `#[gpu_kernel]` proc macro |
| `kaio-core` | PTX IR, instruction emitters, zero external dependencies |
| `kaio-runtime` | CUDA driver wrapper via cudarc |
| `kaio-ops` | Pre-built GPU operations (matmul, attention, auto-tuner) |

## Limitations

- NVIDIA only (SM 7.0+) — no AMD, no Intel
- Performance is size-dependent: tensor-core matmul meets or beats
  cuBLAS sgemm at 4096³ on RTX 4090 (worst-of-10: 107% sync / 115%
  async, precision caveats disclosed) but lags heavily at small
  sizes; the scalar path tops out at ~31% of cuBLAS — see the
  repository's `docs/performance.md`
- DSL subset of Rust — no closures, traits, generics, or `&&`/`||`
- FlashAttention requires d_k <= 256
- No autograd in the core crates (the `kaio-candle` bridge provides
  backward for the matmul TC family and FlashAttention); no multi-GPU
- API may change

## Status

**Phase 9 complete** — tensor-core matmul family (f16 + bf16, sync +
`cp.async`, auto-tuned), quantized INT8/INT4 matmul + fused QKV
projections, FlashAttention forward + backward, candle bridge
(`kaio-candle`), `ldmatrix` IR primitive. v0.5.0.

See the [repository](https://github.com/dmriding/kaio) for full
documentation, runnable examples, copy-paste patterns, and development
logs.

## License

Licensed under either of Apache-2.0 or MIT at your option.
