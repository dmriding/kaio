# KAIO — Benchmark Methodology & Results

## Methodology

### What is measured

Kernel execution time only. Memory allocation, host-to-device transfer,
and device-to-host transfer are excluded. Both KAIO and cuBLAS use the
same device, stream, and input data.

### Timing

- `stream.synchronize()` before starting the timer (drain prior work)
- Launch kernel
- `stream.synchronize()` after launch (wait for completion)
- Wall-clock elapsed time via `std::time::Instant`

### Statistical approach

- **Warm-up:** 5 launches (discarded)
- **Measurement:** 20 launches
- **Reported:** Median of 20 measurements
- **Metric:** TFLOPS = 2 × M × N × K / median_seconds / 1e12

The factor of 2 accounts for one multiply and one add per output element
per K dimension step (FMA = 2 FLOPs).

### Input data

Deterministic pseudo-random f32 in [-1, 1], fixed seed (42 for A, 137
for B). Same inputs for KAIO and cuBLAS across all runs.

### cuBLAS reference

cuBLAS sgemm via cudarc's `cublas` module. Row-major matrices are
handled by the standard transpose trick: `C = A × B` in row-major
becomes `C^T = B^T × A^T` in column-major (swap operands, swap M/N).

cuBLAS correctness is validated against CPU reference at 64×64 before
any performance numbers are reported.

### Environment

All reported numbers are from a single machine configuration:
- **GPU:** NVIDIA GeForce RTX 4090 (SM 8.9, Ada Lovelace)
- **Driver:** CUDA 12.8
- **OS:** Windows 11
- **Rust:** 1.94.1

### How to reproduce

```sh
cargo test -p kaio-ops --test matmul_bench -- --ignored --nocapture
```

Requires NVIDIA GPU and CUDA toolkit (for cuBLAS).

---

## Results — Phase 4 Baseline (Naive 16×16 Tiled MatMul)

**Date:** 2026-04-11 | **GPU:** RTX 4090 | **KAIO kernel:** naive 16×16 tiled, FMA inner loop

| Size | KAIO (ms) | KAIO (TFLOPS) | cuBLAS (ms) | cuBLAS (TFLOPS) | Ratio |
|------|-----------|--------------|-------------|----------------|-------|
| 256×256×256 | 0.20 | 0.16 | 0.02 | 1.72 | 9.6% |
| 512×512×512 | 0.29 | 0.92 | 0.02 | 10.78 | 8.5% |
| 1024×1024×1024 | 0.71 | 3.02 | 0.06 | 37.15 | 8.1% |
| 1024×2048×512 | 0.70 | 3.08 | 0.06 | 35.73 | 8.6% |
| 2048×2048×2048 | 4.26 | 4.04 | 0.32 | 52.93 | 7.6% |
| 4096×4096×4096 | 29.84 | 4.61 | 2.39 | 57.40 | 8.0% |

**Analysis:** The naive tiled kernel achieves ~8% of cuBLAS across all
sizes. This is expected for a correctness-first implementation without
register tiling, bank conflict avoidance, or vectorized loads.

Peak KAIO throughput: 4.61 TFLOPS at 4096².
Peak cuBLAS throughput: 57.40 TFLOPS at 4096².

**Next:** Sprint 4.6 (register tiling + optimization) targets ≥60% of
cuBLAS at 2048×2048.
