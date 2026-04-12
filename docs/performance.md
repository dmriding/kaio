# KAIO — Performance Guide

Practical guidance for writing fast GPU kernels with KAIO's
`#[gpu_kernel]` DSL. These patterns apply to NVIDIA GPUs generally,
not just KAIO.

## Memory Coalescing

**Rule:** Adjacent threads should access adjacent memory addresses.

On NVIDIA GPUs, global memory is accessed in 128-byte transactions.
When all 32 threads in a warp read consecutive 4-byte values, that's
one transaction. When they read scattered addresses, it can become
32 separate transactions — a 32× bandwidth penalty.

### Good: coalesced global load

```rust
#[gpu_kernel(block_size = (16, 16))]
fn good_load(a: &[f32], out: &mut [f32], n: u32) {
    let col = block_idx_x() * 16 + thread_idx_x();
    let row = block_idx_y() * 16 + thread_idx_y();
    // thread_idx_x() varies fastest → adjacent threads read
    // adjacent addresses → coalesced
    if row < n {
        if col < n {
            out[row * n + col] = a[row * n + col];
        }
    }
}
```

Adjacent threads (differing in `thread_idx_x()`) access addresses
that differ by 4 bytes. This is one 128-byte transaction per warp.

### Bad: non-coalesced global load

```rust
// DON'T: column-major access with thread_idx_x as row
let row = block_idx_x() * 16 + thread_idx_x();
let col = block_idx_y() * 16 + thread_idx_y();
// Adjacent threads read a[row * n + col] where row varies
// but col is the same → addresses stride by n*4 bytes → uncoalesced
out[row * n + col] = a[row * n + col];
```

Adjacent threads now access addresses `n * 4` bytes apart. Each thread
hits a different 128-byte cache line.

### Key takeaway

In 2D kernels, use `thread_idx_x()` as the fast-varying (column)
index for row-major arrays. This ensures warp-level coalescing.

---

## Shared Memory Bank Conflicts

Shared memory is divided into 32 banks (one per warp lane). When
multiple threads in a warp access the same bank (at different
addresses), the accesses are serialized.

### The problem: column access in tiled matmul

A shared tile `tile_a[64][16]` with stride 16: threads reading
column `ki` access addresses `0*16+ki`, `1*16+ki`, `2*16+ki`, ...

With stride 16, addresses differ by 16 × 4 = 64 bytes = exactly 2
banks apart. This means threads 0 and 16 alias the same bank.

### The fix: padding

Pad the stride to a non-power-of-two:

```rust
// Instead of tile_a[64 * 16] (stride 16):
let tile_a = shared_mem![f32; 1088]; // 64 × 17, stride 17

// Instead of tile_b[16 * 64] (stride 64):
let tile_b = shared_mem![f32; 1040]; // 16 × 65, stride 65
```

With stride 17, column access addresses differ by 17 × 4 = 68 bytes,
which cycles through different banks. No conflicts.

### When to pad

Pad when accessing a shared memory tile along the non-contiguous
dimension (typically columns). The padding adds one element per row,
costing a small amount of shared memory for significant throughput
improvement.

---

## Register Tiling

Register tiling has each thread compute multiple output elements
instead of one. This increases arithmetic intensity — more FMAs per
shared memory load — which is key for compute-bound kernels like
matmul.

### Concept

| Approach | Output per thread | FMAs per K step | Shared loads per K step |
|----------|------------------|-----------------|------------------------|
| Naive | 1×1 = 1 | 1 | 2 (one from A, one from B) |
| 4×4 tiling | 4×4 = 16 | 16 | 8 (4 from A, 4 from B) |

With 4×4 tiling, arithmetic intensity increases from 0.5 FMA/load to
2.0 FMA/load — a 4× improvement in compute-to-memory ratio.

### Tradeoffs

- **More accumulators:** 4×4 uses 16 `f32` registers for accumulators.
  NVIDIA GPUs have 65536 registers per SM shared among all active
  threads. More registers per thread → fewer concurrent threads
  (lower occupancy).
- **Larger tiles:** 4×4 per thread with 256 threads = 64×64 output
  tile. This requires more shared memory (64×17 + 16×65 = 8.5 KB).
- **Diminishing returns at small sizes:** When the matrix fits in a
  few tiles, launch overhead and partial tiles dominate.

### KAIO matmul implementation

See [kaio-ops/src/matmul_kernel.rs](../kaio-ops/src/matmul_kernel.rs)
for the register-tiled kernel: BM=BN=64, BK=16, TM=TN=4, 16 named
accumulators (`acc_00` through `acc_33`).

---

## PTX Inspection Tools

KAIO provides three environment variables for inspecting generated PTX.

### `KAIO_DUMP_PTX=1` — Write PTX to files

```sh
KAIO_DUMP_PTX=1 cargo test -p kaio --test shared_mem_macro -- --ignored --nocapture
```

Writes `.ptx` files to `OUT_DIR` (or current directory). Inspect the
full PTX output for any kernel.

### `KAIO_PTX_STATS=1` — Instruction statistics

```sh
KAIO_PTX_STATS=1 cargo test -p kaio --test shared_mem_macro -- --ignored --nocapture
```

Reports structural statistics for each kernel at first launch:

```
KAIO stats: kernel 'my_kernel' (PTX structure, not runtime profile)
  Instructions: 47 total
  Arithmetic:   12 fma, 8 other
  Memory:       4 ld.global, 2 st.global, 6 ld.shared, 3 st.shared
  Control:      2 bar.sync, 4 branches, 3 setp, 5 mov, 1 cvt
  Registers:    15 r32, 4 r64, 8 f32, 0 f64, 3 pred  (PTX-level, not final HW allocation)
  Shared mem:   8512 bytes
```

**Important:** These are structural counts of emitted PTX. They
describe the instruction mix in KAIO's generated code, useful for
comparing kernel variants. They are **not** runtime profiling data —
final hardware register allocation, occupancy, and instruction
scheduling may differ after the CUDA driver compiles PTX to SASS.
Use `ncu` (NVIDIA Nsight Compute) for runtime profiling.

#### Example: naive vs optimized matmul stats

Comparing stats between kernel variants helps understand where
optimization effort went:

| Metric | Naive (16×16) | Optimized (64×64, 4×4 reg) |
|--------|--------------|---------------------------|
| FMA | ~1 per K step | ~16 per K step |
| ld.shared | ~2 per K step | ~8 per K step |
| Shared mem | 2 KB | 8.5 KB |
| f32 registers | few | 16+ (accumulators) |
| Arithmetic intensity | low | 4× higher |

The optimized kernel trades more registers and shared memory for
dramatically higher FMA density.

### `KAIO_PTX_ANNOTATE=1` — Source annotations in PTX

```sh
KAIO_PTX_ANNOTATE=1 KAIO_DUMP_PTX=1 cargo test -p kaio --test shared_mem_macro -- --ignored --nocapture
```

Adds `// comment` annotations in the emitted PTX showing which source
construct generated each block of instructions:

```ptx
    // shared_mem sdata: [f32; 256]
    // let tx
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %r0;
    // let idx
    mov.u32 %r2, %ctaid.x;
    mul.lo.u32 %r3, %r2, 256;
    add.u32 %r4, %r3, %r1;
    // sdata[...] = ...
    mov.u32 %r5, sdata;
    mul.u32 %r6, %r1, 4;
    add.u32 %r7, %r5, %r6;
    ld.global.f32 %f0, [%rd4];
    st.shared.f32 [%r7], %f0;
    // bar_sync()
    bar.sync 0;
```

This maps the Rust DSL to its PTX output, useful for debugging
lowering issues and understanding performance characteristics.

---

## Benchmarking

See [benchmarks.md](benchmarks.md) for methodology and results.

### Quick reference

```sh
# Scalar f32 matmul benchmark (Phase 4 — naive + optimized vs cuBLAS sgemm)
cargo test -p kaio-ops --test matmul_bench -- --ignored --nocapture

# Tensor-core f16 matmul benchmark (Sprint 6.7 — sync + async vs cuBLAS sgemm)
cargo test -p kaio-ops --test matmul_tc_bench -- --ignored --nocapture
```

Measures kernel execution time only (no allocation or transfer).
Reports TFLOPS for the kernels under test against the cuBLAS sgemm
baseline. 5 warmup iterations + 20 timed iterations, median reported.

## Tensor-Core Matmul Performance (Sprint 6.7)

Multi-warp 64×64 block tile, 4 warps per block, each warp owning a
32×32 sub-quadrant computed via 8 × `mma.sync.m16n8k16` per K-iteration.
Two variants exposed: `matmul_tc` (synchronous shared-mem staging) and
`matmul_tc_async` (cp.async double-buffered A staging). The
`matmul_auto_tc` tuner dispatches between them per-shape via cached
benchmark data (cache miss falls back to `TensorCoreAsync`, the
production winner at large shapes — see Apples-to-apples disclaimer
below).

### Measured on RTX 4090 (sm_89), median of 20 timed iterations after 5 warmups

| Size       | TC sync TFLOPS | TC async TFLOPS | cuBLAS sgemm TFLOPS | sync vs cuBLAS | async vs cuBLAS |
|------------|---------------:|----------------:|--------------------:|---------------:|----------------:|
| 256³       | 0.05           | 0.04            | 1.73                | 2.8%           | 2.5%            |
| 512³       | 0.38           | 0.33            | 10.74               | 3.6%           | 3.1%            |
| 1024³      | 3.01           | 2.72            | 36.84               | 8.2%           | 7.4%            |
| 2048³      | 17.60          | 17.15           | 52.91               | 33.3%          | 32.4%           |
| **4096³**  | **46.53**      | **49.56**       | **58.24**           | **79.9%**      | **85.1%**       |

### Apples-to-apples disclaimer

KAIO TC matmul uses **fp16 inputs with fp32 accumulation**. Comparison
is against **cuBLAS sgemm** (f32 inputs, f32 output) because that is
the existing supported benchmark path in this repo (cudarc 0.19's
`Gemm::gemm` exposes sgemm cleanly). Results should be read as a
**project-local performance baseline, not a claim of apples-to-apples
precision identity.** The fp16-input / fp32-input asymmetry halves
global memory bandwidth on the TC side and unlocks tensor-core
throughput; that gap is part of the value proposition, not a flaw in
the comparison. A true f16-vs-f16 comparison against cuBLAS HGEMM /
GemmEx is tracked tech debt for a future sprint.

### Why small sizes underperform cuBLAS

At 256–1024, TC matmul lands at 3–8% of cuBLAS. This is expected:
the multi-warp kernel launches `(N/64) × (M/64)` blocks and below
1024² there are too few blocks to fill the SM array (1024² = 16
blocks; one block per SM occupies only a sliver of the 4090's 128
SMs). cuBLAS at small sizes uses dispatch heuristics that pick
launch shapes appropriate to the workload — a single library tuned
for the full size range. KAIO's TC matmul is a single kernel
optimized for the large-shape regime where TC throughput matters.
For small shapes, prefer scalar `matmul` (Phase 4) or stay on cuBLAS.

### Path to >85%

The remaining ~15 percentage points of headroom at 4096² is bounded by
two structural choices the Sprint 6.7 kernel hasn't yet made:

- **Vectorized global loads (LDG.128).** The cooperative tile loaders
  use scalar `ld.global.b32` (4-byte loads). cuBLAS uses 16-byte
  vectorized loads for ~4× higher load bandwidth. Sprint 6.7b adds a
  `MemoryOp::LdGlobalB128` family + emits + ptxas verification, then
  rewrites the cooperative loaders to use it.
- **Bank-conflict-aware shared layouts.** Current tile_a / tile_b
  layouts have natural bank-conflict patterns at multi-warp scale
  that the kernel does not yet pad around. Standard CUTLASS-style
  swizzle or `+1` padding eliminates them.

Sprint 6.7b targets pushing into the 90%+ range with these two
optimizations. Phase 7 adds bf16 inputs and larger mma shapes (where
applicable on Hopper+).
