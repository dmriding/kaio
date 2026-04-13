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
# Scalar f32 matmul (Phase 4): naive + optimized vs cuBLAS sgemm
cargo test -p kaio-ops --test matmul_bench -- --ignored --nocapture

# Tensor-core f16 matmul (Sprint 6.7): sync + async vs cuBLAS sgemm
cargo test -p kaio-ops --test matmul_tc_bench -- --ignored --nocapture
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

**Scalar f32 matmul (Phase 4):** 31% of cuBLAS sgemm — BM=BN=64, BK=16,
TM=TN=4, scalar loads.

**Tensor-core f16 matmul (Sprints 6.7 + 6.7b — final Phase 6 result):**
**82.3% (sync) / 92.5% (async)** of cuBLAS sgemm at 4096² on RTX 4090.
Multi-warp 64×64 block tile (6.7) + Tile B col-stride padding +
fragment-loader `(group_id, tig)` hoist (6.7b). See the
**[Tensor-Core Matmul Performance section in performance.md](performance.md#tensor-core-matmul-performance-sprints-67--67b)**
for the full table across 256–4096, the apples-to-apples disclaimer
(KAIO uses fp16 inputs with fp32 accumulation; cuBLAS sgemm is f32),
and the analysis of why async lifted more than sync under 6.7b
(bank-conflict relief on fragment-B shared reads). 6.7b's async
result — 92.5% — is past the original 90% stretch target from one
sprint earlier.

---

## Results — Sprints 6.7 + 6.7b TC MatMul (final Phase 6 state)

**Date:** 2026-04-13 | **GPU:** RTX 4090 (sm_89) | **Kernels:**
multi-warp 64×64 sync (`matmul_tc`) + cp.async double-buffered
(`matmul_tc_async`), with 6.7b col-stride padding + fragment-loader
`(group_id, tig)` hoist | **Reference:** cuBLAS sgemm

| Size  | TC sync TFLOPS | TC async TFLOPS | cuBLAS TFLOPS | sync vs cuBLAS | async vs cuBLAS |
|-------|---------------:|----------------:|--------------:|---------------:|----------------:|
| 256³  | 0.05           | 0.05            | 1.77          | 2.9%           | 2.6%            |
| 512³  | 0.37           | 0.34            | 11.09         | 3.3%           | 3.1%            |
| 1024³ | 2.87           | 2.62            | 37.35         | 7.7%           | 7.0%            |
| 2048³ | 17.34          | 16.74           | 52.91         | 32.8%          | 31.6%           |
| 4096³ | **40.93**      | **45.96**       | **49.72**     | **82.3%**      | **92.5%**       |

Sprint 6.7 → Sprint 6.7b delta at 4096²: sync 79.9% → 82.3% (+2.4pp),
async 85.1% → **92.5%** (+7.4pp — past the 90% stretch target on
padding + hoist alone).

**Apples-to-apples disclaimer:** KAIO TC matmul uses fp16 inputs with
fp32 accumulation; cuBLAS sgemm is f32 inputs / f32 output. The
comparison is a project-local performance baseline, not a precision-
identity claim. See [performance.md](performance.md#apples-to-apples-disclaimer)
for the full framing.

**Analysis:**
- Multi-warp restructure (6.7) unlocked SM occupancy at large shapes
  — at 4096² the kernel launches 64×64 = 4,096 blocks × 128 threads
  ≈ 16 resident warps/SM on RTX 4090.
- Async pulls ahead of sync at 4096² (46.0 vs 40.9 TFLOPS, +12.4%
  under 6.7b — materially wider than 6.7's +6.5% gap) — cp.async
  pipeline gets enough K-iterations (256) to hide latency, and 6.7b's
  bank-conflict fix on Tile B fragment reads benefits async
  disproportionately because its global-load bandwidth was already
  saturated via cp.async direct bypass.
- Small sizes (256-1024) still lose to cuBLAS by large margins — too
  few blocks to fill the SM array, kernel launch overhead dominates.
  Use scalar `matmul` or stay on cuBLAS for small shapes.
- Path past 92.5% for async / 82.3% for sync: future sprints
  (LDG.128 IR primitive landed in 6.7b as well-formed unused IR for
  this purpose, plus `ldmatrix.sync.aligned` as the sync-path lever).

---

## Phase 4 baseline retained for comparison
