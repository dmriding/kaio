# KAIO

**Rust-native GPU kernel authoring framework.**

KAIO (καίω — to kindle, to ignite) lets developers write GPU compute
kernels in Rust and lower them to PTX for execution on NVIDIA GPUs.
A Rust alternative to OpenAI's Triton — Windows + Linux, automatic PTX
generation, Rust type safety, no CUDA C++ toolchain required.

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
```

The `#[gpu_kernel]` macro lowers `saxpy` to PTX and generates a typed
`saxpy::launch()` function. PTX is generated once at first call and
cached for the process lifetime.

## Features

| Feature | Syntax | Status |
|---------|--------|--------|
| Arithmetic | `+`, `-`, `*`, `/`, `%`, `+=`, `-=`, `*=`, `/=` | ✅ |
| Comparisons | `<`, `<=`, `>`, `>=`, `==`, `!=` | ✅ |
| Control flow | `if`/`else`, `for`, `while` | ✅ |
| Array access | `a[idx]` (global memory) | ✅ |
| Shared memory | `shared_mem![f32; 256]` | ✅ |
| Synchronization | `bar_sync()` | ✅ |
| Warp shuffle | `shfl_sync_down/up/bfly()` | ✅ |
| Reductions | `block_reduce_sum()`, `block_reduce_max()` | ✅ |
| Type casts | `x as f32` | ✅ |
| Math builtins | `sqrt`, `exp`, `log`, `tanh`, `abs`, `min`, `max` | ✅ |
| Thread indices | `thread_idx_x()`, `block_idx_x()`, `block_dim_x()` | ✅ |
| FMA | `fma(a, b, c)` | ✅ |
| 2D blocks | `block_size = (16, 16)`, `thread_idx_y()` | ✅ |
| Tiled matmul | `kaio_ops::matmul()` (31% of cuBLAS) | ✅ |

## Architecture

| Crate | Description |
|-------|-------------|
| `kaio` | Umbrella crate — re-exports everything via `prelude` |
| `kaio-macros` | `#[gpu_kernel]` proc macro |
| `kaio-core` | PTX IR, instruction emitters, zero external dependencies |
| `kaio-runtime` | CUDA driver wrapper via cudarc |
| `kaio-ops` | Pre-built GPU operations (matmul, more planned) |

## Status

**Phase 4 complete** — tiled matmul (31% of cuBLAS sgemm on RTX 4090),
`kaio-ops` crate, 2D thread blocks, FMA, PTX inspection tools.
This is early development software. API will change.

See the [repository](https://github.com/dmriding/kaio) for the full
roadmap, CHANGELOG, and development logs.

## License

Licensed under either of Apache-2.0 or MIT at your option.
