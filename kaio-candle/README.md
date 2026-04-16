# kaio-candle

[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/dmriding/kaio)
[![Rust](https://img.shields.io/badge/rust-1.94+-orange.svg)](https://www.rust-lang.org/)

**Candle bridge for [KAIO](https://github.com/dmriding/kaio) — forward-only `CustomOp` bindings that let you call KAIO's tensor-core GPU kernels directly on `candle_core::Tensor`.**

Ships five ops today: `matmul_tc`, `matmul_tc_async`, `matmul_int4`, `attention_tc`, `attention_tc_causal`. Backward kernels and the remaining quant ops (`matmul_int8`, `qkv_project_int{4,8}`) land in follow-up sprints.

## Status — v0.1.0 (Sprint 7.4a)

Forward-only. Inference / eval use cases only; training (autograd) is not yet supported — `bwd()` falls back to candle's `BackwardNotSupported`.

All ops are bit-exact verified against direct `kaio-ops` calls with the same input bits.

## Why a separate crate?

`kaio-candle` is **not** a member of the main KAIO workspace. cudarc rejects `dynamic-loading` + `dynamic-linking` as simultaneously active features:

- **Main KAIO** defaults to `dynamic-loading` — no CUDA toolkit required to build. Host tests pass on bare GitHub runners.
- **candle-core** with its `cuda` feature activates `dynamic-linking` — it links against `libcuda` at compile time.

Cargo unions features across a workspace build, so including `kaio-candle` in the main workspace would force every main-workspace build to also carry candle's `dynamic-linking`, breaking no-CUDA CI. The standalone crate keeps the two worlds apart.

Consumers who already build candle with the `cuda` feature see no new system requirement beyond what candle itself needs.

## Build

```sh
cd kaio-candle
cargo build --features cuda
```

The `cuda` feature is **required** for any actual bridge functionality. Without it, `kaio-candle` is an empty shell (matches candle-core's own opt-in `cuda` pattern) — attempting to call `kaio_candle::matmul_tc(...)` surfaces a "function not found" compile error pointing at the missing feature.

Build requirements with `cuda`:
- CUDA toolkit (candle-core's cudarc feature uses `dynamic-linking`).
- NVIDIA GPU with SM 8.0 or newer (Ampere, Ada, Hopper).

## Quickstart

```toml
# Cargo.toml
[dependencies]
kaio-candle = { version = "0.1", features = ["cuda"] }
kaio = "0.3"
candle-core = { version = "0.10", features = ["cuda"] }
half = "2"
```

```rust
use std::sync::Arc;
use candle_core::{Device, Tensor};
use half::f16;
use kaio::prelude::KaioDevice;

fn main() -> anyhow::Result<()> {
    let candle_dev = Device::new_cuda(0)?;
    let kaio_dev = Arc::new(KaioDevice::new(0)?);

    let m = 128usize;
    let k = 128usize;
    let n = 128usize;

    let a_host: Vec<f16> = (0..m * k).map(|i| f16::from_f32((i % 17) as f32 * 0.01)).collect();
    let b_host: Vec<f16> = (0..k * n).map(|i| f16::from_f32((i % 13) as f32 * 0.02)).collect();

    let a = Tensor::from_vec(a_host, (m, k), &candle_dev)?;
    let b = Tensor::from_vec(b_host, (k, n), &candle_dev)?;

    // f16[m,k] x f16[k,n] -> f32[m,n]
    let c = kaio_candle::matmul_tc(&kaio_dev, &a, &b)?;
    println!("output shape: {:?}", c.shape().dims());

    Ok(())
}
```

Two runnable examples ship in `examples/`:

```sh
cd kaio-candle
cargo run --release --features cuda --example matmul_tc_candle
cargo run --release --features cuda --example attention_tc_candle
```

## Op surface

| Op | Trait | Shapes | Dtype |
| --- | --- | --- | --- |
| `matmul_tc(kd, a, b)` | `CustomOp2` | `a: [M, K]`, `b: [K, N]` → `[M, N]` | f16 × f16 → f32 |
| `matmul_tc_async(kd, a, b)` | `CustomOp2` | same | f16 × f16 → f32 |
| `matmul_int4(kd, a, b_packed, scales)` | `CustomOp3` | `a: [M, K]`, `b_packed: [K/8, N]`, `scales: [K/128, N]` → `[M, N]` | f16 × u32 × f16 → f32 |
| `attention_tc(kd, q, k, v)` | `CustomOp3` | `q: [seq_q, d_k]`, `k: [seq_k, d_k]`, `v: [seq_k, d_v]` → `[seq_q, d_v]` | f16 × f16 × f16 → f32 |
| `attention_tc_causal(kd, q, k, v)` | `CustomOp3` | same | f16 × f16 × f16 → f32 |

`matmul_int4` is GPTQ-style: `group_size=128` is locked in by the kaio-ops kernel contract. `K` must be a multiple of 128, weights are packed 8 INT4 values per `u32`, one f16 scale per group of 128 elements.

`attention_tc` uses a shared-memory scores buffer capped at `seq_k ≤ 384`. FlashAttention-TC will lift this cap in a later sprint.

## Device lifetime

The `Arc<kaio::prelude::KaioDevice>` you construct and pass to `kaio-candle` wrapper functions is independent of the `candle_core::Device` you use for your tensors. Both retain the same CUDA primary context via `cuDevicePrimaryCtxRetain`; neither owns the other. Drop order between them is unconstrained.

Every wrapper call checks that the KAIO device and candle device share the same CUDA ordinal; a mismatch is a loud error.

## Candle version policy

`kaio-candle = 0.1` pins `candle-core = "=0.10.2"` exactly. This is deliberate:

- candle 0.10.2 is the current release at the time of publishing.
- The `CustomOp2` / `CustomOp3` surface has changed between candle minor versions in the past.
- cudarc feature conventions change with candle releases.

We re-pin `kaio-candle` against each new candle minor release. Use `kaio-candle 0.1.x` with `candle-core 0.10.x`; `kaio-candle 0.2` will target whichever candle minor is current when we publish.

## Known limitations (v0.1)

- **Non-contiguous tensors rejected.** Call `.contiguous()?` upstream.
- **Non-zero storage offset rejected** (e.g. from `.narrow(...)` / `.slice(...)`). Call `.contiguous()?` to compact.
- **Rank-2 only.** Multi-head attention callers must reshape `[heads, seq, d]` to `[heads * seq, d]` or call per-head with rank-2 slices. Wrappers error with a concrete reshape hint for higher-rank inputs.
- **CUDA Graph capture incompatible.** Wrappers call `cuCtxSynchronize` for stream safety, which is banned inside a stream-capture region. Returns `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`. Event-based stream plumbing in a later sprint unblocks Graph usage.
- **f32 output contract.** Wrappers return `DType::F32` tensors matching the kaio-ops accumulator. Cast via `.to_dtype(DType::F16)?` if you need f16 for downstream graph continuation.
- **No CPU fallback.** `cpu_fwd` returns a loud error rather than silently routing to `candle.matmul()`. KAIO's value is GPU-specific PTX; a silent CPU fallback would mask every perf claim.
- **Bench numbers vs direct-call gap.** Each bridge call issues two `cuCtxSynchronize` fences (one before, one after the kaio-ops launch). Tiny-shape decode is dominated by this overhead; prefill shapes less so. KAIO's published %-of-cuBLAS numbers are measured via direct kaio-ops calls, not through the bridge.

## License

Dual-licensed under MIT or Apache-2.0, at your option.
