# KAIO

[![Crates.io](https://img.shields.io/crates/v/kaio.svg)](https://crates.io/crates/kaio)
[![Documentation](https://docs.rs/kaio/badge.svg)](https://docs.rs/kaio)
[![Build Status](https://github.com/dmriding/kaio/actions/workflows/ci.yml/badge.svg)](https://github.com/dmriding/kaio/actions)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/dmriding/kaio)
[![Rust](https://img.shields.io/badge/rust-1.94+-orange.svg)](https://www.rust-lang.org/)

**Rust-native GPU kernel authoring framework.**

KAIO (from the Greek καίω — _to kindle, to ignite_) lets you write GPU
compute kernels in Rust and compile them to PTX for execution on NVIDIA
GPUs. Cross-platform (Windows + Linux), compile-time PTX emission, and
Rust's type safety — no CUDA C++, no Python, no runtime JIT.

## The Problem

The Rust ML ecosystem can't keep up with Python. Every time a new model
architecture drops with a custom operation — a novel attention variant, a
fused activation, a custom quantization kernel — frameworks like
[candle](https://github.com/huggingface/candle) and
[burn](https://github.com/tracel-ai/burn) can't support it until someone
writes the GPU function. Today, that means writing CUDA C++, fighting FFI
bindings, and giving up on Windows. Most Rust developers don't know CUDA.
The few who do are already overwhelmed. Models pile up in the
"unsupported" column.

Meanwhile, Python developers write a
[Triton](https://github.com/triton-lang/triton) kernel in an afternoon
and move on. Triton doesn't support Windows, requires Python, and
JIT-compiles at runtime — but it works, and Rust has no equivalent.

**KAIO is that equivalent.**

```rust
use kaio::prelude::*;

#[gpu_kernel(block_size = 256)]
fn fused_gelu(x: &[f32], out: &mut [f32], n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        let v = x[idx];
        out[idx] = 0.5 * v * (1.0 + tanh(0.7978846 * (v + 0.044715 * v * v * v)));
    }
}
```

Write a GPU kernel in Rust. `cargo build`. It runs on your GPU. That's it.

## Why KAIO?

- **No CUDA C++.** Write kernels in Rust syntax you already know.
- **No Python.** No Triton, no conda environments, no JIT warm-up.
- **Windows + Linux.** Triton is Linux-only. KAIO works everywhere
  `cargo build` works.
- **Compile-time PTX.** Kernels compile during `cargo build` via proc
  macros. Zero cold-start latency.
- **Type safe.** Catch dtype mismatches, undefined variables, and
  invalid kernel signatures at compile time — not as silent GPU
  corruption at runtime.
- **Standalone.** Not tied to any ML framework. Use with candle, burn,
  mistral.rs, or your own project.

## Quick Start

Add KAIO to your project:

```sh
cargo add kaio
```

Write a kernel and launch it:

```rust
use kaio::prelude::*;

#[gpu_kernel(block_size = 256)]
fn saxpy(x: &[f32], y: &mut [f32], alpha: f32, n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        y[idx] = alpha * x[idx] + y[idx];
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = KaioDevice::new(0)?;
    let n = 1024u32;

    let x = device.alloc_from(&vec![1.0f32; n as usize])?;
    let mut y = device.alloc_from(&vec![2.0f32; n as usize])?;

    saxpy::launch(&device, &x, &mut y, 2.5f32, n)?;

    let result = y.to_host(&device)?;
    // result[i] == 2.5 * 1.0 + 2.0 == 4.5
    Ok(())
}
```

The `#[gpu_kernel]` macro compiles `saxpy` to PTX at build time and
generates a typed `saxpy::launch()` function with correct argument
marshaling. No unsafe in user code.

## Supported Kernel Features

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
| Tiled matmul | `block_load`, `block_dot` | 🚧 Phase 4 |
| Fused attention | FlashAttention-style | 🚧 Phase 5 |

## Architecture

KAIO is structured in four layers:

```
┌─────────────────────────────────────────┐
│  Layer 4: Block-Level Operations        │  tiled matmul, fused attention
├─────────────────────────────────────────┤
│  Layer 3: #[gpu_kernel] Proc Macro      │  Rust syntax → PTX at build time
├─────────────────────────────────────────┤
│  Layer 2: Runtime (kaio-runtime)        │  device memory, kernel launch
├─────────────────────────────────────────┤
│  Layer 1: PTX Codegen (kaio-core)       │  IR types, instruction emitters
└─────────────────────────────────────────┘
```

| Crate | Description |
|-------|-------------|
| `kaio` | Umbrella crate — re-exports everything via `prelude` |
| `kaio-macros` | `#[gpu_kernel]` proc macro |
| `kaio-core` | PTX IR, instruction emitters, zero external dependencies |
| `kaio-runtime` | CUDA driver wrapper via [cudarc](https://github.com/coreylowman/cudarc) |

## Target Hardware

- **GPUs:** NVIDIA SM 7.0+ (Volta, Turing, Ampere, Ada Lovelace, Hopper)
- **Platforms:** Windows 10/11, Linux (Ubuntu 22.04+)
- **Tested on:** RTX 4090 (SM 8.9)

## Building

```sh
cargo build --workspace
cargo test --workspace                  # host tests (no GPU needed)
cargo test --workspace -- --ignored     # GPU tests (requires NVIDIA GPU)
KAIO_DUMP_PTX=1 cargo test              # inspect generated PTX
```

Requires Rust 1.94+ (pinned via `rust-toolchain.toml`). No CUDA toolkit
needed to build — KAIO uses dynamic loading via the NVIDIA display driver.

## How It Works

The `#[gpu_kernel]` macro:

1. **Parses** your Rust function body into a kernel IR
2. **Lowers** expressions to PTX instruction sequences
3. **Generates** a `build_ptx()` function that constructs the IR at
   runtime (first call only, cached via `OnceLock`)
4. **Emits** PTX text through `PtxWriter`
5. **Wraps** everything in a typed `launch()` function that handles
   device memory and kernel dispatch

The generated PTX is validated against `nvcc` output and passes
`ptxas --verify` on every tested kernel.

## IR API (Advanced)

You can also build kernels directly via the Layer 1 IR API for maximum
control:

```rust
use kaio_core::emit::{Emit, PtxWriter};
use kaio_core::ir::*;
use kaio_core::types::PtxType;

let mut alloc = RegisterAllocator::new();
let mut kernel = PtxKernel::new("my_kernel");
kernel.add_param(PtxParam::pointer("data", PtxType::F32));
// ... build instructions ...

let mut module = PtxModule::new("sm_89");
module.add_kernel(kernel);
let mut w = PtxWriter::new();
module.emit(&mut w).unwrap();
println!("{}", w.finish()); // valid PTX assembly
```

See
[kaio-runtime/tests/vector_add_e2e.rs](kaio-runtime/tests/vector_add_e2e.rs)
for a complete end-to-end example.

## Roadmap

- [x] **Phase 1** — PTX codegen + runtime (IR → PTX → GPU execution)
- [x] **Phase 2** — `#[gpu_kernel]` proc macro (arithmetic, control
  flow, memory access, math builtins)
- [x] **Phase 3** — Loops, shared memory, reductions, softmax
- [ ] **Phase 4** — Tiled matrix multiplication, block-level API
- [ ] **Phase 5** — Fused attention, auto-tuning, crates.io v0.1.0

See [docs/phases.md](docs/phases.md) for detailed plans and
[CHANGELOG.md](CHANGELOG.md) for per-sprint progress.

## License

Licensed under either of

- Apache License, Version 2.0
  ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license
  ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
