# kaio-candle

[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/dmriding/kaio)
[![Rust](https://img.shields.io/badge/rust-1.94+-orange.svg)](https://www.rust-lang.org/)
[![candle HEAD compat](https://github.com/dmriding/kaio/actions/workflows/candle-head.yml/badge.svg)](https://github.com/dmriding/kaio/actions/workflows/candle-head.yml)

**Candle bridge for [KAIO](https://github.com/dmriding/kaio) — forward-only `CustomOp` bindings that let you call KAIO's tensor-core GPU kernels directly on `candle_core::Tensor`.**

Ships eight ops today: `matmul_tc`, `matmul_tc_async`, `matmul_int4`, `matmul_int8`, `attention_tc`, `attention_tc_causal`, `qkv_project_int8`, `qkv_project_int4`. Backward kernels land in 7.4c.

## Status — v0.1.0 (Sprint 7.4a + 7.4b)

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

Three runnable examples ship in `examples/`:

```sh
cd kaio-candle
cargo run --release --features cuda --example matmul_tc_candle
cargo run --release --features cuda --example matmul_int8_candle
cargo run --release --features cuda --example attention_tc_candle
```

## Op surface

| Op | Trait | Shapes | Dtype |
| --- | --- | --- | --- |
| `matmul_tc(kd, a, b)` | `CustomOp2` | `a: [M, K]`, `b: [K, N]` → `[M, N]` | f16 × f16 → f32 |
| `matmul_tc_async(kd, a, b)` | `CustomOp2` | same | f16 × f16 → f32 |
| `matmul_int4(kd, a, b_packed, scales)` | `CustomOp3` | `a: [M, K]`, `b_packed: [K/8, N]`, `scales: [K/128, N]` → `[M, N]` | f16 × u32 × f16 → f32 |
| `matmul_int8(kd, a, b, scale)` | `CustomOp2` | `a: [M, K]`, `b: [K, N]` → `[M, N]` | u8-as-i8 × u8-as-i8 → f32 (× f32 scale) |
| `attention_tc(kd, q, k, v)` | `CustomOp3` | `q: [seq_q, d_k]`, `k: [seq_k, d_k]`, `v: [seq_k, d_v]` → `[seq_q, d_v]` | f16 × f16 × f16 → f32 |
| `attention_tc_causal(kd, q, k, v)` | `CustomOp3` | same | f16 × f16 × f16 → f32 |
| `qkv_project_int8(kd, x, wq, wk, wv, sq, sk, sv)` | Direct-call | `x: [M, K]`, `wq/wk/wv: [K, N]` → `(Q, K, V)` each `[M, N]` | f16 × u8-as-i8 → **f16** |
| `qkv_project_int4(kd, x, wq, wk, wv, sq, sk, sv)` | Direct-call | `x: [M, K]`, `wq/wk/wv: [K/8, N]`, `sq/sk/sv: [K/128, N]` → `(Q, K, V)` each `[M, N]` | f16 × u32 × f16 → **f16** |

`matmul_int4` is GPTQ-style: `group_size=128` is locked in by the kaio-ops kernel contract. `K` must be a multiple of 128, weights are packed 8 INT4 values per `u32`, one f16 scale per group of 128 elements.

`matmul_int8` is W8A8 symmetric quant. Candle has no `DType::I8`, so the convention is `DType::U8` tensors whose bytes are interpreted as signed INT8 (`-128..=127`) by the kernel. The bridge reinterprets the storage via a same-layout transmute. `scale` is a scalar `f32` applied in the accumulator; a typical realistic value is `max_abs / 127`.

`attention_tc` uses a shared-memory scores buffer capped at `seq_k ≤ 384`. FlashAttention-TC will lift this cap in a later sprint.

`qkv_project_int8` and `qkv_project_int4` are **direct-call** functions (not `CustomOpN` — candle's trait maxes at 3 inputs and single output). They return `(Tensor, Tensor, Tensor)` with `DType::F16` output because the fused kernel performs the `f32→f16` conversion internally as part of the projection fusion. Gradient-tracked inputs are rejected with a loud error requiring `.detach()` — these ops are forward-only.

## Device lifetime

The `Arc<kaio::prelude::KaioDevice>` you construct and pass to `kaio-candle` wrapper functions is independent of the `candle_core::Device` you use for your tensors. Both retain the same CUDA primary context via `cuDevicePrimaryCtxRetain`; neither owns the other. Drop order between them is unconstrained.

Every wrapper call checks that the KAIO device and candle device share the same CUDA ordinal; a mismatch is a loud error.

## Candle version policy

`kaio-candle = 0.1` pins `candle-core = "=0.10.2"` exactly. This is deliberate:

- candle 0.10.2 is the current release at the time of publishing.
- The `CustomOp2` / `CustomOp3` surface has changed between candle minor versions in the past.
- cudarc feature conventions change with candle releases.

We re-pin `kaio-candle` against each new candle minor release. Use `kaio-candle 0.1.x` with `candle-core 0.10.x`; `kaio-candle 0.2` will target whichever candle minor is current when we publish.

A weekly GitHub Actions workflow (`.github/workflows/candle-head.yml`) builds kaio-candle against candle-core's git `main` branch once per Monday. If this badge goes red for more than two consecutive weeks, either the pin moves to the new candle minor or this section documents the divergence.

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
