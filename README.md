# KAIO

[![Crates.io](https://img.shields.io/crates/v/kaio.svg)](https://crates.io/crates/kaio)
[![Documentation](https://docs.rs/kaio/badge.svg)](https://docs.rs/kaio)
[![Build Status](https://github.com/dmriding/kaio/actions/workflows/ci.yml/badge.svg)](https://github.com/dmriding/kaio/actions)
[![Coverage](https://img.shields.io/badge/coverage-93.65%25-brightgreen)](#test-coverage)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/dmriding/kaio)
[![Rust](https://img.shields.io/badge/rust-1.94+-orange.svg)](https://www.rust-lang.org/)

**Write custom GPU kernels in Rust — no CUDA C++, no Python, no toolkit.** KAIO compiles a Rust-subset DSL to PTX at build time and JIT-loads it on the GPU at launch. Works on Windows and Linux with just the NVIDIA display driver installed.

KAIO (from the Greek καιω — _to burn, to ignite_) is for Rust engineers
who need custom GPU kernels today — fused attention variants,
quantization ops, novel activations — and would otherwise be writing
CUDA C++ because their framework doesn't support them.

## Key highlights

- **No CUDA toolkit required** — just the NVIDIA display driver. Build
  in CI on a standard GitHub runner; host tests pass without a GPU.
  This is KAIO's single biggest differentiator vs CUDA C++, `rust-cuda`,
  and Triton.
- **Meets or beats cuBLAS sgemm at 4096²** — on RTX 4090 across 10
  consecutive benchmark runs, KAIO tensor-core matmul (async, fp16
  inputs with fp32 accumulation) hits **58.74 TFLOPS at worst** /
  65.12 TFLOPS median — 115% / 112% of cuBLAS sgemm under the same
  conditions. Worst-case framing: these are the floors, not the peaks.
  [Full distribution →](docs/performance.md)
- **Windows and Linux native.** No WSL2, no Triton's Linux-only
  runtime, no Python. `cargo build` works everywhere.
- **Pure-Rust kernel authorship.** The `#[gpu_kernel]` proc macro lowers
  Rust to a PTX IR module at compile time; at launch the module is
  validated against the current GPU's SM target, emitted to PTX text, and
  handed to the CUDA driver for JIT compilation. Type-safe kernel
  signatures catch dtype mismatches at compile time, not as silent GPU
  corruption at runtime.

## Try KAIO in 30 seconds

Clone the repo, run one command, see real ML kernels build and execute on your GPU:

```sh
git clone https://github.com/dmriding/kaio.git
cd kaio
cargo xtask showcase
```

You'll see `fused_silu_gate`, `gelu_comparison`, `rms_norm`, `layer_norm`, `softmax`, `int8_dequant`, and `int8_matmul` compile, launch, verify correctness against a CPU reference, and report median latency. The seven examples span activations, normalizations, reductions, and the quantize → matmul pipeline: the canonical transformer-primitive arc plus the W8A8 headline op.

Want the performance pitch instead? `cargo xtask bench` runs the tensor-core matmul benchmark against cuBLAS sgemm across five sizes. Or `cargo xtask all` for both. `cargo xtask --help` for the full tooling surface.

Requires an NVIDIA GPU with an installed display driver (NVIDIA 525 or newer — any standard Game Ready or Studio driver works). No CUDA toolkit install needed.

## Quick Start

Requires an NVIDIA GPU (SM 7.0+, Volta or newer) with an installed
display driver — NVIDIA 525 or newer (any standard Game Ready or Studio
driver works; no Tesla/TCC-specific drivers needed).

```sh
cargo add kaio
```

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
    Ok(())
}
```

```
$ cargo run
result: [4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5]
```

## The real pitch — fused ML kernels

SAXPY is for learning the DSL. The actual value looks like this:

```rust
use kaio::prelude::*;

// Gated SiLU — the feedforward activation in every LLaMA / Mistral /
// Qwen block. llama.cpp, vLLM, and TensorRT-LLM all ship hand-written
// CUDA for it. With KAIO it's 7 lines of Rust, lowered to a PTX IR
// module at compile time and JIT-loaded at launch.
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

Run it from the repo root with no directory changes:

```
$ cargo xtask showcase silu

=== fused_silu_gate ===
Input size:        1048576 elements
Correctness:       PASS  (max_abs_err = 1.49e-8)
Median latency:    188.8 μs  (of 100 timed runs, 5 warm-ups skipped)
```

Or run all six showcases in sequence with `cargo xtask showcase`:
[fused SiLU-gate](examples/fused_silu_gate/),
[exact vs fast GELU](examples/gelu_comparison/),
[single-block RMSNorm](examples/rms_norm/),
[single-block LayerNorm](examples/layer_norm/),
[single-block softmax](examples/softmax/),
[INT8 dequantization](examples/int8_dequant/). Each is a complete
standalone project with correctness + timing (with its own `Cargo.toml`
so you can copy the directory out of the repo as a reference for your
own kernel); the GELU comparison's README explains why kernel fusion
matters more than arithmetic optimization for ML workloads (the
bandwidth-bound teaching moment).

## When to use KAIO

Reach for KAIO when:

- **Your framework can't support a custom op** (novel attention, fused
  activation, quantization) and you don't want to drop into CUDA C++
  for one kernel.
- **You need GPU inference on Windows** without WSL2 or Triton's
  Linux-only runtime.
- **Your CI runs on standard GitHub runners** without GPU or CUDA
  toolkit access. KAIO's host tests pass without a GPU; only the
  `#[ignore]`-gated integration tests need one. Flip the matrix on
  later.
- **You need deterministic VRAM usage, explicit buffer reuse, or
  zero-copy transfers** that high-level frameworks abstract away.
- **You're prototyping GPU code** in a language you already know
  (Rust) without learning CUDA C++.

KAIO is not a replacement for [Candle](https://github.com/huggingface/candle)
or [Burn](https://github.com/tracel-ai/burn). It is the layer you use
when you need more control than they provide.

|                             | KAIO       | cudarc        | Candle / Burn | Triton (Python) | Raw CUDA |
| --------------------------- | ---------- | ------------- | ------------- | --------------- | -------- |
| Write kernels in Rust       | Yes        | No (load PTX) | No            | No (Python)     | No       |
| Automatic PTX generation    | Yes        | No            | N/A           | Yes (runtime)   | No       |
| Windows support             | Yes        | Yes           | Partial       | **No**          | Yes      |
| No CUDA toolkit needed      | Yes        | Yes           | Varies        | No              | No       |
| Compile-time kernel codegen | Yes        | N/A           | N/A           | No (runtime JIT)| Yes      |
| Type-safe kernel signatures | Yes        | No            | N/A           | No              | No       |
| ML framework integration    | candle (via `kaio-candle`) | Standalone | Built-in | PyTorch      | Manual   |

## What this is not

- **Not compiled Rust.** `#[gpu_kernel]` bodies use Rust syntax but are
  parsed into KAIO's own IR and lowered directly to PTX. rustc's
  backend (LLVM, MIR, borrow checker) never sees the kernel body. You
  cannot call Rust functions declared outside the kernel from inside
  it.
- **Not CUDA bindings.** KAIO generates PTX itself. It does not wrap
  cuDNN, cuBLAS, CUTLASS, or any CUDA C++ library. The comparison to
  cuBLAS sgemm in this README is a *measurement reference*, not a
  dependency.
- **Not a full ML framework.** No autograd (beyond the handful of
  `kaio-candle` backward bindings), no model zoo, no training loop.
  KAIO is the layer you use when
  [Candle](https://github.com/huggingface/candle) or
  [Burn](https://github.com/tracel-ai/burn) don't have the op you need.

## Performance

Performance is optimized for large ML workloads (transformer-scale
shapes). Small sizes are launch-overhead dominated and will lag
cuBLAS — see the apples-to-apples notes below.

Numbers below are **worst observed** across 10 consecutive
`cargo xtask bench` runs on RTX 4090 sm_89, release build, warm GPU
(median of 20 timed iterations per run, then minimum across runs).
Under-promise framing: every run will do at least this well.

**Tensor-core matmul (f16 × f16 → f32):**

| Size  | TC sync worst | TC async worst | cuBLAS sgemm worst | sync vs cuBLAS | async vs cuBLAS |
|-------|--------------:|---------------:|-------------------:|---------------:|----------------:|
| 256³  | 0.09 TF       | 0.09 TF        | 1.71 TF            | 5.3%           | 5.3%            |
| 512³  | 0.72 TF       | 0.53 TF        | 10.74 TF           | 6.7%           | 4.9%            |
| 1024³ | 5.14 TF       | 4.98 TF        | 31.86 TF           | 16.1%          | 15.6%           |
| 2048³ | 27.11 TF      | 27.90 TF       | 43.12 TF           | 62.9%          | 64.7%           |
| **4096³** | **54.63 TF** | **58.74 TF** | **51.05 TF**        | **107.0%**     | **115.1%**      |

**Quantized matmul at 4096³ (same bench, worst-of-10):**

| Kernel                   | Worst TOPS | Median TOPS | Best TOPS |
|--------------------------|-----------:|------------:|----------:|
| `matmul_int8` (W8A8)     | 84.07      | 92.58       | 93.38     |
| `matmul_int4` (W4A16)    | 52.02      | 57.52       | 58.04     |

**Apples-to-apples disclaimer:** KAIO tensor-core matmul uses fp16 inputs
with fp32 accumulation; cuBLAS sgemm is f32 in / f32 out. The INT8 and
INT4 columns compare against cuBLAS sgemm because `cublasGemmEx` INT8
is not cleanly exposed by `cudarc` 0.19 — weight bandwidth alone
(0.5 B/weight for INT4 vs 4 B for sgemm) dominates these ratios.
Compare these numbers sprint-over-sprint for regression detection; the
"vs cuBLAS sgemm" column is a project-local baseline, not a
precision-identity claim. See [docs/performance.md](docs/performance.md)
for the full distribution (min / median / max across all sizes).

## The problem KAIO solves

The Rust ML ecosystem can't keep up with Python. Every time a new
model architecture drops with a custom operation — a novel attention
variant, a fused activation, a custom quantization kernel — frameworks
like [candle](https://github.com/huggingface/candle) and
[burn](https://github.com/tracel-ai/burn) can't support it until
someone writes the GPU function. Today, that means writing CUDA C++,
fighting FFI bindings, and giving up on Windows.

Meanwhile, Python developers write a
[Triton](https://github.com/triton-lang/triton) kernel in an afternoon
and move on. Triton doesn't support Windows, requires Python, and
JIT-compiles at runtime — but it works, and Rust has no equivalent.

**KAIO is that equivalent.**

## Patterns

Copy these skeletons, fill in your logic.

**Bounds-checked element-wise:**

```rust
#[gpu_kernel(block_size = 256)]
fn my_kernel(input: *const [f32], output: *mut [f32], n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        output[idx] = input[idx] * 2.0; // your logic here
    }
}
```

**Shared memory tiling:**

```rust
#[gpu_kernel(block_size = 256)]
fn tiled(data: *const [f32], out: *mut [f32], n: u32) {
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
fn reduce(input: *const [f32], out: *mut [f32], n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    let val = if idx < n { input[idx] } else { 0.0f32 };
    let sum = block_reduce_sum(val);
    if thread_idx_x() == 0u32 { out[block_idx_x()] = sum; }
}
```

## Feature summary

| Feature                                  | Notes                                                                      |
| ---------------------------------------- | -------------------------------------------------------------------------- |
| `#[gpu_kernel]` proc macro               | Rust → PTX IR at compile time; PTX emitted + JIT-loaded at launch. Type-safe launch wrapper auto-generated. |
| Shared memory + reductions + warp shuffles | `shared_mem![]`, `bar_sync()`, `block_reduce_sum/max/min`, `warp_reduce_sum/max/min`, `shfl_sync_*`. |
| 2D blocks, FMA, math builtins            | `block_size = (16,16)`, `fma`, `sqrt`, `exp`, `log`, `tanh`, `abs`, `min`, `max`. |
| Scalar tiled matmul                      | `kaio_ops::matmul` / `matmul_auto` — 31% of cuBLAS sgemm. Any SM.          |
| Fused attention + FlashAttention         | `kaio_ops::attention`, `attention_flash` (O(d_k) memory). Any SM.          |
| Tensor-core matmul                       | `kaio_ops::matmul_tc` / `matmul_tc_async` / `matmul_auto_tc` — f16 → f32, SM 8.0+, **worst-of-10 at 4096³ on RTX 4090: sync 107% / async 115% of cuBLAS sgemm**. |
| INT8 dequantize-matmul (W8A8)            | `kaio_ops::matmul_int8` — symmetric i8 × i8 → f32 with single-scalar scale, SM 8.0+, K%32==0. **Worst-of-10 at 4096³: 84.07 TOPS (median 92.58, best 93.38).** |
| INT4 dequantize-matmul (W4A16, GPTQ-style) | `kaio_ops::matmul_int4` — packed signed-INT4 weights × f16 activations → f32, f16 group scales (group_size=128), DEQUANT-F16 via `mma.sync.m16n8k16`, SM 8.0+, K%128==0. **Worst-of-10 at 4096³: 52.02 TOPS (median 57.52, best 58.04; 114% of cuBLAS sgemm worst).** |
| Fused tri-output QKV projection (INT8, W8A16) | `kaio_ops::qkv_project_int8` — f16 activations × i8 weights × per-projection scalar scales → three f16 outputs (Q, K, V). Decode tier ~3× faster than three standalone matmuls; prefill performance varies by shape. SM 8.0+, K%16==0, N%2==0. |
| Fused tri-output QKV projection (INT4, W4A16) | `kaio_ops::qkv_project_int4` — packed INT4 weights × f16 activations × f16 group scales → three f16 outputs. Decode tier ~3× faster than three standalone calls; for prefill-heavy workloads at M≥2048, three separate `matmul_int4` calls may be faster. SM 8.0+, K%128==0, group_size=128. |
| Auto-tuner + cache                       | `tune_matmul`, `matmul_auto`, `matmul_auto_tc` with JSON cache.            |
| PTX inspection                           | `KAIO_DUMP_PTX=1`, `KAIO_PTX_STATS=1`, `KAIO_PTX_ANNOTATE=1`.              |

See [docs.rs/kaio](https://docs.rs/kaio) for the full API surface and
the internal IR types.

## Project status and constraints

KAIO is pre-1.0 software. Current engineering constraints:

- **NVIDIA only.** SM 7.0+ (Volta, Turing, Ampere, Ada Lovelace,
  Hopper). No AMD, no Intel, no Apple Silicon.
- **Matmul performance is size-dependent.** Tensor-core matmul meets
  or beats cuBLAS sgemm at 4096² on RTX 4090 (worst-of-10: 107% sync /
  115% async) but lags heavily at ≤1024² (5–17%) because a 64×64
  multi-warp block tile doesn't fill the SM array until the grid is
  large. Scalar matmul tops out at 31% of cuBLAS. For small shapes
  prefer cuBLAS or the scalar path. [Details →](docs/performance.md)
- **Mostly inference.** The [`kaio-candle`](kaio-candle/) bridge ships
  8 forward ops; `matmul_tc` and `matmul_tc_async` support backward
  via mixed-precision autograd (gradients are computed in f16, matching
  the forward-pass precision). Attention and quantized-op backward are
  not yet implemented.
- **DSL is a Rust subset, not compiled Rust.** `#[gpu_kernel]` function
  bodies use Rust syntax but are parsed into KAIO's own IR and lowered
  directly to PTX. The kernel body **never reaches rustc's backend** —
  no LLVM, no MIR, no borrow checker. Kernel parameters are written as
  `*const [T]` / `*mut [T]` (primary, per
  [RFC-0001](docs/development/rfcs/rfc-0001-pointer-syntax.md)) to
  signal the on-device reality: thousands of GPU threads access the same
  buffer concurrently, and correctness depends on disjoint access
  patterns (e.g. `if idx < n` guards), not on compiler-enforced
  uniqueness. `&[T]` / `&mut [T]` are accepted as permanent sugar and
  lower identically. You cannot call Rust functions declared outside the
  kernel inside the kernel body. No closures, traits, generics, method
  calls, or string operations. Arithmetic, comparisons, bitwise
  operators (`&` `|` `^` `<<` `>>` `!`), short-circuit `&&` / `||`, and
  compound assignment (including bitwise `&=` / `|=` / `<<=` / etc.)
  all supported.
- **FlashAttention d_k limit.** `attention_flash()` requires
  d_k ≤ 256 (one thread per output dimension).
- **Single-device.** No multi-GPU support.
- **First-call PTX module load.** The `#[gpu_kernel]` macro generates
  PTX at Rust compile time, but CUDA-driver module loading still
  happens at first launch. First call pays the module-load latency;
  subsequent launches are dispatch-only. (The in-process PTX cache
  was removed in v0.2.1 to simplify the load path and run kernels
  through `PtxModule::validate()` on every launch.)
- **API will change** before 1.0. Breaking changes documented in
  CHANGELOG per release.

## Architecture

Four layers, bottom to top:

```
+-------------------------------------------+
|  Layer 4: Block-Level Operations          |  tiled matmul, fused attention, TC matmul
+-------------------------------------------+
|  Layer 3: #[gpu_kernel] Proc Macro        |  Rust syntax → PTX automatically
+-------------------------------------------+
|  Layer 2: Runtime (kaio-runtime)          |  device memory, kernel launch, SM validation
+-------------------------------------------+
|  Layer 1: PTX Codegen (kaio-core)         |  IR types, instruction emitters, fragments
+-------------------------------------------+
```

| Crate          | Description                                                             |
| -------------- | ----------------------------------------------------------------------- |
| `kaio`         | Umbrella crate — re-exports everything via `prelude`                    |
| `kaio-macros`  | `#[gpu_kernel]` proc macro                                              |
| `kaio-core`    | PTX IR, instruction emitters, fragment containers, zero external deps   |
| `kaio-runtime` | CUDA driver wrapper via [cudarc](https://github.com/coreylowman/cudarc) |
| `kaio-ops`     | Pre-built GPU operations (matmul, attention, TC matmul, auto-tuner)     |
| `kaio-candle`  | [candle](https://github.com/huggingface/candle) bridge — 8 forward ops + 2 backward (matmul_tc, matmul_tc_async), event-based stream sync. Standalone crate at [`kaio-candle/`](kaio-candle/) |

## Candle integration

The [`kaio-candle`](kaio-candle/) crate bridges KAIO's GPU kernels into
[candle](https://github.com/huggingface/candle)'s tensor graph. 8 forward
ops + 2 backward ops, event-based stream sync (CUDA Graph compatible).

```rust
use std::sync::Arc;
use kaio::prelude::KaioDevice;

let kd = Arc::new(KaioDevice::new(0)?);
let c = kaio_candle::matmul_tc(&kd, &a, &b)?;  // candle Tensor in, candle Tensor out
```

See [`kaio-candle/README.md`](kaio-candle/README.md) for the full op
surface and usage guide.

## Target hardware

- **GPUs:** NVIDIA SM 7.0+ (Volta, Turing, Ampere, Ada Lovelace,
  Hopper). Tensor-core kernels (`matmul_tc*`) require SM 8.0+ (Ampere
  or newer).
- **Platforms:** Windows 10/11 and Linux (Ubuntu 22.04+).
- **Driver:** NVIDIA 525+ (CUDA 12.0+ compatible). Standard Game Ready
  or Studio drivers work on consumer cards; Tesla/TCC drivers are not
  required and not needed for KAIO's dynamic-loading path.
- **Tested on:** RTX 4090 (sm_89) under Windows.

## Building

```sh
cargo build --workspace
cargo test --workspace                  # host tests (no GPU needed)
cargo test --workspace -- --ignored     # GPU tests (requires NVIDIA GPU)
KAIO_DUMP_PTX=1 cargo test              # inspect generated PTX
```

Requires Rust 1.94+ (pinned via `rust-toolchain.toml`). The version
floor reflects edition-2024 features and const-evaluation patterns used
in kernel tile-layout computation. No CUDA toolkit is needed to build
— KAIO resolves the NVIDIA driver at runtime via dynamic loading
(`nvcuda.dll` on Windows, `libcuda.so` on Linux).

## Debugging

When something goes wrong — launch errors, silent NaN, unexpectedly slow
performance — [`docs/debugging.md`](docs/debugging.md) is the single
entry point. It covers the env vars (`KAIO_DUMP_PTX`, `KAIO_PTX_STATS`,
`KAIO_PTX_ANNOTATE`, `KAIO_SM_TARGET`, `KAIO_TUNE_CACHE`,
`KAIO_SUPPRESS_DEBUG_WARNING`), the async-launch error model,
`compute-sanitizer` usage for silent-corruption diagnosis, tolerance
guidance for numerical verification, and a troubleshooting flowchart
for "did it compile → launch → produce right output?"

## Test coverage

**93.65% line coverage** across the 20,156-line workspace (1,280 lines
uncovered, mostly host-side parser error paths, the `xtask` repo-tooling
binary, and the unreachable-by-design host stubs for GPU builtins in
`kaio/src/gpu_builtins.rs`). Shipped kernel crates are well above the
workspace average — `kaio-ops/src/matmul_int8_kernel.rs` at 97.77%,
`matmul_tc_kernel.rs` at 97.74%, `matmul_tc_async_kernel.rs` at 99.40%,
`attention_tc_kernel.rs` at 98.82%. Measured on RTX 4090 sm_89 via
`cargo llvm-cov` with the host test suite and the full GPU-only
`--ignored` test suite merged:

```sh
cargo install cargo-llvm-cov           # one-time
cargo llvm-cov clean --workspace
cargo llvm-cov --workspace --no-report
cargo llvm-cov --workspace --no-report -- --ignored
cargo llvm-cov report --summary-only
```

The number is static (updated per release, not per CI run) because
the GPU-ignored tests require actual NVIDIA hardware and can't run on
standard GitHub Actions runners. See
[`docs/testing-strategy.md`](docs/testing-strategy.md) for the full
testing model (host tests, GPU integration tests, `ptxas_verify`
structural checks, and the `matmul_tc_bench` performance harness).

## How it works

The `#[gpu_kernel]` macro:

1. **Parses** your Rust function body into a kernel IR.
2. **Lowers** expressions to PTX instruction sequences.
3. **Generates** a `build_ptx()` function that constructs the IR at
   runtime (first call only, cached via `OnceLock`).
4. **Emits** PTX text through `PtxWriter`.
5. **Wraps** everything in a typed `launch()` function that handles
   device memory and kernel dispatch.

The generated PTX is validated against `nvcc` output and passes
`ptxas --verify` on every tested kernel.

## IR API (advanced)

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
- [x] **Phase 4** — Tiled matmul (31% of cuBLAS), `kaio-ops` crate, 2D
  blocks, FMA, PTX inspection tools
- [x] **Phase 5** — Fused attention + FlashAttention + auto-tuning,
  crates.io v0.1.0
- [x] **Phase 6** — Tensor cores (`mma.sync` fp16/bf16), async copies
  (`cp.async`), bank-conflict padding. Initial v0.2.0 publish shipped
  at 82.3% sync / 92.5% async of cuBLAS sgemm; subsequent warm-steady
  10-run worst-case is 107% / 115% (see Performance). Three standalone
  showcase examples. crates.io v0.2.0.
- [x] **Phase 7** — Quantized kernels (INT8/INT4, fused QKV
  projection), candle integration (`kaio-candle` bridge — 8 forward ops,
  2 backward ops, event-based stream sync).
- [ ] **Phase 8** — PyO3 bindings (Python access to `kaio-ops`).

See [CHANGELOG.md](CHANGELOG.md) for per-release detail and
[docs/phases.md](docs/phases.md) for deeper phase plans.

## Common pitfalls

- **Always bounds-check array writes.** `if idx < n` before every
  global memory access — out-of-bounds GPU writes corrupt memory
  silently.
- **Shared memory must fit within block limits.** Default is 48 KB
  per block. `shared_mem![f32; 12288]` = 48 KB = the limit.
- **Thread indexing errors are the most common bug.** Double-check
  your row / col math, especially with 2D blocks. Off-by-one in a
  kernel doesn't panic — it writes to the wrong address.

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
submitted for inclusion in the work by you, as defined in the
Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
