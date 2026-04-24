# KAIO — Benchmark Methodology

This document describes **how** KAIO's performance numbers are
measured. Current **results** live in
[performance.md](performance.md); they are updated per sprint with
worst-of-N framing.

## What is measured

Kernel execution time only. Memory allocation, host-to-device
transfer, and device-to-host transfer are excluded. Both KAIO and
cuBLAS use the same device, stream, and input data.

## Timing

- `stream.synchronize()` before starting the timer (drain prior work)
- Launch kernel
- `stream.synchronize()` after launch (wait for completion)
- Wall-clock elapsed time via `std::time::Instant`

## Statistical approach

Each bench invocation:

- **Warm-up:** 5 launches (discarded)
- **Measurement:** 20 launches
- **Per-run report:** median of the 20 measurements
- **Metric (TFLOPS):** `2 × M × N × K / median_seconds / 1e12`
  (the factor of 2 is one multiply + one add per FMA)
- **Metric (TOPS, INT kernels):** same formula, interpreted as
  integer ops per second

For publication-quality numbers, the bench harness is run 10
consecutive times. The **worst observed median** across those 10
runs is reported as the floor (e.g., "at least X TFLOPS"); the
median-of-medians and best-of-medians are reported alongside for
distribution context. See `docs/performance.md` for the current
distribution tables.

## Input data

Deterministic pseudo-random f32 in `[-1, 1]`, fixed seeds (42 for A,
137 for B). Same inputs for KAIO and cuBLAS across all runs and all
invocations, so run-to-run variance reflects GPU state (thermal,
driver caching, boost clock) rather than workload variation.

## cuBLAS reference

cuBLAS sgemm via `cudarc`'s `cublas` module. Row-major matrices are
handled by the standard transpose trick: `C = A × B` in row-major
becomes `C^T = B^T × A^T` in column-major (swap operands, swap M/N).

cuBLAS correctness is validated against a CPU reference at 64×64
before any performance numbers are reported.

cuBLAS reference variance across runs is reported alongside KAIO's
(same 10-run protocol). The worst-over-worst ratio is the honest
apples-to-apples floor; median-over-median is the typical case.

## Environment

The reference machine for the published numbers:

- **GPU:** NVIDIA GeForce RTX 4090 (SM 8.9, Ada Lovelace)
- **Driver:** standard NVIDIA display driver (CUDA 12.8 runtime)
- **OS:** Windows 11
- **Rust:** 1.94.1
- **Build profile:** `--release`

No CUDA toolkit is required to build or run — only the display
driver. The CUDA 12.8 number reflects the runtime shipped with the
driver, not a separate toolkit install.

## How to reproduce

```sh
# All benchmarks under the unified harness:
cargo xtask bench

# Or a single bench target:
cargo xtask bench matmul_tc_bench
cargo xtask bench matmul_int8_bench
cargo xtask bench matmul_int4_bench
```

Requires an NVIDIA GPU with an installed display driver (NVIDIA 525
or newer). Each bench internally does its own warmup + 20 timed
iterations; for worst-of-N framing, invoke `cargo xtask bench` N
times in sequence and aggregate across invocations.

## Apples-to-apples framing

Current KAIO kernels compared against cuBLAS sgemm use different
dtypes and instructions (f16/INT8/INT4 inputs vs sgemm's f32/f32).
The comparison is a **project-local performance baseline** for
regression detection, not a precision-identity claim. Each result
table in `performance.md` carries its own apples-to-apples
disclaimer describing exactly what is being compared to what.

## Bench coverage

As of Sprint 8.0.5, `cargo xtask bench` covers seven benchmark
harnesses spanning the shipped high-level / public kernel families
plus the showcase kernels:

- `matmul_tc_bench` — f16 tensor-core matmul (sync + async) vs cuBLAS sgemm
- `matmul_int8_bench` — W8A8 symmetric INT8 matmul
- `matmul_int4_bench` — W4A16 GPTQ-style INT4 matmul
- `qkv_project_bench` — fused INT4 vs 3× standalone `matmul_int4`; INT8 absolute TOPS
- `attention_tc_bench` — `attention_tc` + `attention_tc_causal` (short-seq TC, `seq_k ≤ 384`)
- `attention_flash_bench` — `attention_flash` + `attention_flash_causal` (long-seq online softmax)
- `norm_activation_bench` — `rms_norm`, `layer_norm`, `softmax` (reductions) + `fused_silu_gate`, `gelu_exact`, `gelu_fast` (elementwise sweep)

Result tables live in `performance.md`. Methodology (5 warmup + 20
timed iterations per shape, worst-of-N framing across 10 consecutive
`cargo xtask bench` invocations) applies uniformly across all seven.

Internal `kaio-ops` primitives (fragment loaders, PTX-IR building
blocks) and test-only macro kernels are intentionally outside bench
coverage — they are correctness-tested via `cargo test --workspace`
but not performance-gated.
