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

## Results — Phase 4 Optimized (Register-Tiled MatMul)

**Date:** 2026-04-11 | **GPU:** RTX 4090 | **Kernels:** naive 16×16 + optimized 64×64 (4×4 reg tiling)

| Size | Naive (TFLOPS) | Optimized (TFLOPS) | cuBLAS (TFLOPS) | vs cuBLAS | Speedup |
|------|---------------|-------------------|----------------|-----------|---------|
| 256² | 0.18 | 0.14 | 1.71 | 8.3% | 0.8× |
| 512² | 1.06 | 1.00 | 10.96 | 9.1% | 0.9× |
| 1024² | 3.10 | 5.96 | 37.61 | 15.8% | 1.9× |
| 1024×2048×512 | 3.08 | 6.11 | 35.97 | 17.0% | 2.0× |
| 2048² | 4.04 | 13.32 | 43.14 | 30.9% | 3.3× |
| 4096² | 4.53 | 17.44 | 56.00 | 31.2% | 3.8× |

**Analysis:**
- Register tiling gives **3.3-3.8× speedup** over naive at large sizes
- Optimized kernel peaks at **17.44 TFLOPS** (31.2% of cuBLAS at 4096²)
- Small sizes (256-512) see no benefit — 64×64 tile overhead dominates
- Gap to 60% likely requires vectorized loads (LDG.128) and/or larger
  register tiles (8×8 per thread)

**Kernel design parameters (optimized):**
- BM=BN=64, BK=16, TM=TN=4
- Shared: tile_a 64×17 + tile_b 16×65 (bank conflict padding)
- 256 threads, 16 accumulators per thread

---

## Performance Status

**Current:** 31% of cuBLAS sgemm — BM=BN=64, BK=16, TM=TN=4, scalar loads

**Planned next levers:**
- Vectorized global loads (`LDG.128`) — 4× fewer load instructions
- Double buffering — overlap computation with next tile load
- Size-based dispatch — use naive kernel for small sizes where
  64×64 tile overhead dominates

These optimizations require DSL extensions (vectorized load intrinsics)
and are planned for Phase 5+. See [performance.md](performance.md)
for writing fast kernels with the current tooling.
