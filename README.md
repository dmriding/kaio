# KAIO

[![Crates.io](https://img.shields.io/crates/v/kaio.svg)](https://crates.io/crates/kaio)
[![Documentation](https://docs.rs/kaio/badge.svg)](https://docs.rs/kaio)
[![Build Status](https://github.com/dmriding/kaio/actions/workflows/ci.yml/badge.svg)](https://github.com/dmriding/kaio/actions)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/dmriding/kaio)
[![Rust](https://img.shields.io/badge/rust-1.94+-orange.svg)](https://www.rust-lang.org/)

**Rust-native GPU kernel authoring framework.**

KAIO (from the Greek kaio — _to kindle, to ignite_) lets you write GPU
compute kernels in Rust and compile them to PTX for execution on NVIDIA
GPUs. Cross-platform (Windows + Linux), automatic PTX generation, and
Rust's type safety — no CUDA C++, no Python, no CUDA toolkit required.

## Quick Start

```sh
cargo add kaio
```

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
    Ok(())
}
```

```
$ cargo run
result: [4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5]
```

Requires an NVIDIA GPU with driver installed. No CUDA toolkit needed.

## When to Use KAIO

KAIO is not a replacement for ML frameworks like
[Candle](https://github.com/huggingface/candle) or
[Burn](https://github.com/tracel-ai/burn). It is the layer you use
when you need more control than they provide.

Use KAIO when you need to:

- **Write a custom GPU kernel** that your framework doesn't support
  (novel attention variant, fused activation, quantization op)
- **Control GPU memory explicitly** — deterministic VRAM usage, buffer
  reuse, zero-copy transfers
- **Ship a GPU binary** without Python, conda, or Triton in the
  dependency chain
- **Run on Windows** — Triton is Linux-only; KAIO works everywhere
  `cargo build` works
- **Prototype GPU code fast** in a language you already know (Rust)
  without learning CUDA C++

|                             | KAIO       | cudarc        | Candle / Burn | Raw CUDA |
| --------------------------- | ---------- | ------------- | ------------- | -------- |
| Write kernels in Rust       | Yes        | No (load PTX) | No            | No       |
| Automatic PTX generation    | Yes        | No            | N/A           | No       |
| Windows support             | Yes        | Yes           | Partial       | Yes      |
| No CUDA toolkit needed      | Yes        | Yes           | Varies        | No       |
| Type-safe kernel signatures | Yes        | No            | N/A           | No       |
| ML framework integration    | Standalone | Standalone    | Built-in      | Manual   |

## The Problem

The Rust ML ecosystem can't keep up with Python. Every time a new model
architecture drops with a custom operation — a novel attention variant, a
fused activation, a custom quantization kernel — frameworks like
[candle](https://github.com/huggingface/candle) and
[burn](https://github.com/tracel-ai/burn) can't support it until someone
writes the GPU function. Today, that means writing CUDA C++, fighting FFI
bindings, and giving up on Windows.

Meanwhile, Python developers write a
[Triton](https://github.com/triton-lang/triton) kernel in an afternoon
and move on. Triton doesn't support Windows, requires Python, and
JIT-compiles at runtime — but it works, and Rust has no equivalent.

**KAIO is that equivalent.**

## Why KAIO?

- **No CUDA C++.** Write kernels in Rust syntax you already know.
- **No Python.** No Triton, no conda environments, no Python runtime.
- **Windows + Linux.** Triton is Linux-only. KAIO works everywhere
  `cargo build` works.
- **Automatic PTX.** The `#[gpu_kernel]` macro lowers your Rust code to
  PTX. Generated once at first call, cached for the process lifetime.
- **Type safe.** Catch dtype mismatches, undefined variables, and
  invalid kernel signatures at compile time — not as silent GPU
  corruption at runtime.
- **Standalone.** Not tied to any ML framework. Use with candle, burn,
  mistral.rs, or your own project.

## Examples

### Getting-started (single-file, in-tree)

Clone the repo and run:

| Example        | Description                      | Command                                  |
| -------------- | -------------------------------- | ---------------------------------------- |
| **vector_add** | Start here — minimal GPU kernel  | `cargo run --example vector_add -p kaio` |
| saxpy          | Scalar parameter passing         | `cargo run --example saxpy -p kaio`      |
| reduction      | Shared memory + block reduction  | `cargo run --example reduction -p kaio`  |
| matmul         | Matrix multiply via kaio-ops API | `cargo run --example matmul -p kaio-ops` |

### Showcase (standalone Cargo projects)

Each of these lives in [`examples/`](examples/) as its own standalone
Cargo project (its own `Cargo.toml`, its own `[workspace]` detach).
Run with `cargo run --release` from inside each directory — no need
to build the KAIO workspace first.

| Example              | Kernel                                        | Why it matters                                      |
| -------------------- | --------------------------------------------- | --------------------------------------------------- |
| **fused_silu_gate**  | `x * sigmoid(x) * gate`                      | The gated activation in every LLaMA feedforward block |
| **gelu_comparison**  | Exact (tanh) vs fast (sigmoid) GELU           | Kernel-variant workflow + bandwidth-bound teaching moment |
| **rms_norm**         | Single-block RMSNorm                          | LLaMA-family normalization (replaces LayerNorm)    |

Each example runs correctness (against an f64 CPU reference) and
reports median wall-clock latency over 100 timed runs.

## Patterns

Copy these skeletons, fill in your logic.

**Bounds-checked element-wise:**

```rust
#[gpu_kernel(block_size = 256)]
fn my_kernel(input: &[f32], output: &mut [f32], n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        output[idx] = input[idx] * 2.0; // your logic here
    }
}
```

**Shared memory tiling:**

```rust
#[gpu_kernel(block_size = 256)]
fn tiled(data: &[f32], out: &mut [f32], n: u32) {
    let tid = thread_idx_x();
    let idx = tid + block_idx_x() * block_dim_x();
    let tile = shared_mem![f32; 256];
    if idx < n { tile[tid] = data[idx]; }
    bar_sync();
    if idx < n { out[idx] = tile[tid]; } // read from shared
}
```

**Block reduction:**

```rust
#[gpu_kernel(block_size = 256)]
fn reduce(input: &[f32], out: &mut [f32], n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    let val = if idx < n { input[idx] } else { 0.0f32 };
    let sum = block_reduce_sum(val);
    if thread_idx_x() == 0u32 { out[block_idx_x()] = sum; }
}
```

## Supported Kernel Features

| Feature         | Syntax                                                     | Status    |
| --------------- | ---------------------------------------------------------- | --------- |
| Arithmetic      | `+`, `-`, `*`, `/`, `%`, `+=`, `-=`, `*=`, `/=`            | Supported |
| Comparisons     | `<`, `<=`, `>`, `>=`, `==`, `!=`                           | Supported |
| Control flow    | `if`/`else`, `for`, `while`                                | Supported |
| Array access    | `a[idx]` (global memory)                                   | Supported |
| Shared memory   | `shared_mem![f32; 256]`                                    | Supported |
| Synchronization | `bar_sync()`                                               | Supported |
| Warp shuffle    | `shfl_sync_down/up/bfly()`                                 | Supported |
| Reductions      | `block_reduce_sum()`, `block_reduce_max()`                 | Supported |
| Type casts      | `x as f32`                                                 | Supported |
| Math builtins   | `sqrt`, `exp`, `log`, `tanh`, `abs`, `min`, `max`          | Supported |
| Thread indices  | `thread_idx_x()`, `block_idx_x()`, `block_dim_x()`         | Supported |
| FMA             | `fma(a, b, c)`                                             | Supported |
| 2D blocks       | `block_size = (16, 16)`, `thread_idx_y()`                  | Supported |
| Tiled matmul    | `kaio_ops::matmul()` ([31% of cuBLAS](docs/benchmarks.md)) | Supported |
| Attention       | `kaio_ops::attention()`, `attention_causal()`              | Supported |
| FlashAttention  | `kaio_ops::attention_flash()` — O(d_k) memory             | Supported |
| Auto-tuner      | `kaio_ops::tune_matmul()`, `matmul_auto()`                | Supported |
| TC matmul       | `kaio_ops::matmul_tc()` / `matmul_tc_async()` / `matmul_auto_tc()` — f16×f16→f32, SM 8.0+, [~80–85% cuBLAS sgemm at 4096²](docs/performance.md) | Supported |

## Limitations

KAIO is early-stage software. Being honest about what it can't do:

- **NVIDIA only** — SM 7.0+ (Volta through Hopper). No AMD, no Intel.
- **Not cuBLAS-level performance** — matmul reaches 31% of cuBLAS.
  Good enough for custom ops, not for replacing vendor libraries.
- **No autograd / backward pass** — inference and custom compute only,
  not training.
- **DSL subset** — the kernel DSL supports a subset of Rust. No
  closures, traits, generics, method calls, or string operations.
- **No `&&` / `||` operators** — use nested `if` statements instead.
- **No compound shared memory assignment** — `sdata[i] += val` is not
  supported; write `sdata[i] = sdata[i] + val`.
- **FlashAttention d_k limit** — `attention_flash()` requires
  d_k <= 256 (one thread per output dimension).
- **No multi-GPU** — single device only.
- **API will change** — this is pre-1.0 software.

## Architecture

KAIO is structured in four layers:

```
+-------------------------------------------+
|  Layer 4: Block-Level Operations          |  tiled matmul, fused attention
+-------------------------------------------+
|  Layer 3: #[gpu_kernel] Proc Macro        |  Rust syntax -> PTX automatically
+-------------------------------------------+
|  Layer 2: Runtime (kaio-runtime)          |  device memory, kernel launch
+-------------------------------------------+
|  Layer 1: PTX Codegen (kaio-core)         |  IR types, instruction emitters
+-------------------------------------------+
```

| Crate          | Description                                                             |
| -------------- | ----------------------------------------------------------------------- |
| `kaio`         | Umbrella crate — re-exports everything via `prelude`                    |
| `kaio-macros`  | `#[gpu_kernel]` proc macro                                              |
| `kaio-core`    | PTX IR, instruction emitters, zero external dependencies                |
| `kaio-runtime` | CUDA driver wrapper via [cudarc](https://github.com/coreylowman/cudarc) |
| `kaio-ops`     | Pre-built GPU operations (matmul, more planned)                         |

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

- [x] **Phase 1** — PTX codegen + runtime (IR -> PTX -> GPU execution)
- [x] **Phase 2** — `#[gpu_kernel]` proc macro (arithmetic, control
  flow, memory access, math builtins)
- [x] **Phase 3** — Loops, shared memory, reductions, softmax
- [x] **Phase 4** — Tiled matmul (31% of cuBLAS), `kaio-ops` crate,
  2D blocks, FMA, PTX inspection tools
- [x] **Phase 5** — Fused attention, FlashAttention, auto-tuning,
  crates.io v0.1.0
- [ ] **Phase 6** — Tensor cores (`mma.sync` fp16/bf16), async copies
  (`cp.async`), 60-70% cuBLAS target
  - [x] **6.1** — fp16/bf16 type system (`PtxType::F16/BF16`, `%h`/`%hb`,
    cvt rounding, `GpuBuffer<f16>`)
  - [x] **6.2** — `mma.sync.m16n8k16` + `cp.async` + typed fragments
    (Ampere+ / SM 8.0+); single-instruction gate test passes bit-exact
    on RTX 4090
  - [x] **6.3** — IR-level tensor-core matmul (internal API, requires
    M%16=N%8=K%16=0); shared-memory fragment loaders; four correctness
    tests pass on RTX 4090. Performance restructuring deferred to 6.7.
  - [x] **6.4** — `cp.async` double-buffered variant `matmul_tc_async`
    (internal API, same constraints as 6.3). A-tile staged via
    `cp.async.ca`; B stays synchronous. Four correctness tests bit-
    close-to-zero on RTX 4090. Overlap gains wait on 6.7's multi-warp
    restructure.
  - [x] **6.5** — Tensor-core auto-tuner `kaio_ops::matmul_auto_tc`
    (first Phase 6 public API — f16×f16→f32, SM 8.0+, dispatches
    between sync and `cp.async` variants). All internal TC kernel
    loads migrated to `device.load_module(&PtxModule)` so
    `PtxModule::validate()` catches SM mismatches cleanly instead of
    ad-hoc per-kernel checks. Narrow contract: temporary
    `M%16=N%8=K%16=0` constraint, production performance targets land
    in Sprint 6.7.
  - [x] **6.6** — Fused tensor-core scaled dot-product attention
    (`attention_tc` / `attention_tc_causal`, internal preview). Two
    back-to-back `mma.sync.m16n8k16` with warp-shuffle softmax and
    intra-kernel `cvt.rn.f16.f32` bridge. 11 GPU correctness tests
    pass on RTX 4090 (5 non-causal + 5 causal + row-0 canary) at
    `seq_k ≤ 384`, `d_k ≤ 128`, SM 8.0+. `#[doc(hidden)]` until
    Phase 7's FlashAttention-TC lifts the constraints and
    `attention_auto_tc` arrives as the user-facing dispatcher.
  - [x] **6.7** — Multi-warp 64×64 TC matmul restructure (4 warps ×
    32×32 quadrant via 8 mma per K-tile), edge-tile predication lifts
    M and N divisibility (K%16=0 stays — mma K-tile is structural),
    cuBLAS sgemm benchmark at 256–4096. Measured **79.9% (sync)** /
    **85.1% (async)** of cuBLAS sgemm at 4096² on RTX 4090. `matmul_tc`
    + `matmul_tc_async` promoted from `#[doc(hidden)]` to stable `pub`.
    See [docs/performance.md](docs/performance.md) for the full table
    and the apples-to-apples disclaimer.
  - [ ] **6.7b** — Vectorized loads (LDG.128) + bank-conflict padding,
    chasing the remaining headroom toward 90%+.
- [ ] **Phase 7** — Quantized kernels (INT8/INT4), training integration
  (`kaio-candle` bridge)
- [ ] **Phase 8** — PyO3 bindings (Python access to kaio-ops)

See [docs/phases.md](docs/phases.md) for detailed plans and
[CHANGELOG.md](CHANGELOG.md) for per-sprint progress.

## Common Pitfalls

- **Always bounds-check array writes.** `if idx < n` before every
  global memory access — out-of-bounds GPU writes corrupt memory
  silently.
- **Shared memory must fit within block limits.** Default is 48KB per
  block. `shared_mem![f32; 12288]` = 48KB = the limit.
- **Thread indexing errors are the most common bug.** Double-check your
  row/col math, especially with 2D blocks. Off-by-one in a kernel
  doesn't panic — it writes to the wrong address.

## Feedback

If something is confusing, awkward, or broken —
[open an issue](https://github.com/dmriding/kaio/issues). Even small
friction matters. This project is actively developed and feedback
directly shapes what gets built next.

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
