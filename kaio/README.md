# KAIO

**Rust-native GPU kernel authoring framework.**

Write GPU compute kernels in Rust, compile to PTX, run on NVIDIA GPUs.
No CUDA C++, no Python, no CUDA toolkit required. Windows + Linux.

## Quick Start

```rust
use kaio::prelude::*;

#[gpu_kernel(block_size = 256)]
fn saxpy(x: &[f32], y: &mut [f32], alpha: f32, n: u32) {
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
| Tiled matmul | `kaio_ops::matmul()` (31% of cuBLAS) | Supported |

## Architecture

| Crate | Description |
|-------|-------------|
| `kaio` | Umbrella crate — re-exports everything via `prelude` |
| `kaio-macros` | `#[gpu_kernel]` proc macro |
| `kaio-core` | PTX IR, instruction emitters, zero external dependencies |
| `kaio-runtime` | CUDA driver wrapper via cudarc |
| `kaio-ops` | Pre-built GPU operations (matmul, more planned) |

## Limitations

- NVIDIA only (SM 7.0+) — no AMD, no Intel
- Not cuBLAS-level performance (matmul reaches 31%)
- DSL subset of Rust — no closures, traits, generics, or `&&`/`||`
- Block reductions are 1D only
- No autograd, no multi-GPU
- Pre-1.0 — API will change

## Status

**Phase 4 complete** — tiled matmul (31% of cuBLAS sgemm on RTX 4090),
`kaio-ops` crate, 2D thread blocks, FMA, PTX inspection tools.

See the [repository](https://github.com/dmriding/kaio) for full
documentation, runnable examples, copy-paste patterns, and development
logs.

## License

Licensed under either of Apache-2.0 or MIT at your option.
