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
fn good_load(a: *const [f32], out: *mut [f32], n: u32) {
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

See [benchmarks.md](benchmarks.md) for the measurement methodology
(warmup / iteration counts, timing technique, input-data generation,
cuBLAS reference setup). Current numbers live below in this file.

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

## Tensor-Core Matmul Performance (Sprints 6.7 + 6.7b)

Multi-warp 64×64 block tile, 4 warps per block, each warp owning a
32×32 sub-quadrant computed via 8 × `mma.sync.m16n8k16` per K-iteration.
Two variants exposed: `matmul_tc` (synchronous shared-mem staging) and
`matmul_tc_async` (cp.async double-buffered A staging). The
`matmul_auto_tc` tuner dispatches between them per-shape via cached
benchmark data (cache miss falls back to a size heuristic — async for
`max(M,N,K) >= 3072`, sync below — matching the measured curve).

**Sprint 6.7b folded in** col-stride padding on shared Tile B (32 → 36
bytes per col) plus a D10 fragment-loader hoist
(`(group_id, thread_id_in_group)` computed once per block). These two
changes alone delivered a 7.4pp uplift on the async path, pushing it
past the 90% stretch target from Sprint 6.7.

### Measured on RTX 4090 (sm_89), 10 consecutive `cargo xtask bench` runs (warm GPU)

Each run is median of 20 timed iterations after 5 warmups; the table
below reports **worst / median / best observed across 10 runs**. Lead
with worst-case ("KAIO does at least X even in the worst run we
measured") — under-promise framing.

**Tensor-core matmul sync (`matmul_tc`, fp16 × fp16 → f32):**

| Size       | KAIO worst | KAIO median | KAIO best | cuBLAS worst | cuBLAS median | worst / worst | median / median |
|------------|-----------:|------------:|----------:|-------------:|--------------:|--------------:|----------------:|
| 256³       | 0.09 TF    | 0.10 TF     | 0.10 TF   | 1.71 TF      | 1.89 TF       | 5.3%          | 5.3%            |
| 512³       | 0.72 TF    | 0.74 TF     | 0.76 TF   | 10.74 TF     | 12.01 TF      | 6.7%          | 6.2%            |
| 1024³      | 5.14 TF    | 5.50 TF     | 5.56 TF   | 31.86 TF     | 32.34 TF      | 16.1%         | 17.0%           |
| 2048³      | 27.11 TF   | 28.81 TF    | 29.55 TF  | 43.12 TF     | 43.48 TF      | 62.9%         | 66.3%           |
| **4096³**  | **54.63 TF** | **60.27 TF** | **61.04 TF** | **51.05 TF** | **58.38 TF** | **107.0%**    | **103.2%**      |

**Tensor-core matmul async (`matmul_tc_async`, fp16 × fp16 → f32):**

| Size       | KAIO worst | KAIO median | KAIO best | cuBLAS worst | cuBLAS median | worst / worst | median / median |
|------------|-----------:|------------:|----------:|-------------:|--------------:|--------------:|----------------:|
| 256³       | 0.09 TF    | 0.09 TF     | 0.09 TF   | 1.71 TF      | 1.89 TF       | 5.3%          | 4.8%            |
| 512³       | 0.53 TF    | 0.69 TF     | 0.71 TF   | 10.74 TF     | 12.01 TF      | 4.9%          | 5.7%            |
| 1024³      | 4.98 TF    | 5.17 TF     | 5.31 TF   | 31.86 TF     | 32.34 TF      | 15.6%         | 16.0%           |
| 2048³      | 27.90 TF   | 29.35 TF    | 30.21 TF  | 43.12 TF     | 43.48 TF      | 64.7%         | 67.5%           |
| **4096³**  | **58.74 TF** | **65.12 TF** | **65.56 TF** | **51.05 TF** | **58.38 TF** | **115.1%**    | **111.5%**      |

At 4096³, KAIO TC async meets-or-beats cuBLAS sgemm in every single
run of the 10 measured. Floor: 115% of cuBLAS worst. Typical: 112% of
cuBLAS median. The 2026-04-14 v0.2.1 headline was "92.5%" (median of
one run); subsequent warm/thermal steady-state across multiple runs
shows KAIO is consistently at or above the reference.

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

### Why async lifts more than sync under 6.7b

The pre-6.7b bank conflicts were on the Tile B fragment-B **read** hot
path (32 lanes per warp × 16 fragment-B loads per block per K-tile).
Col stride 32 B put all 8 thread-groups' bases at just 16 distinct
banks (`(group_id·8 + tig) mod 32`), serialising every single
fragment-B read 2-way across a warp. Post-pad col stride 36 B gives
`(group_id·9 + tig) mod 32` — most banks accessed by a single lane,
only 3 banks remain 2-way, effectively a ~5–8× reduction in
shared-memory serialisation on the hot path.

Async benefits more than sync from this fix because async's cp.async
pipeline already saturates load bandwidth; its remaining bottleneck
was shared-memory contention at fragment-read time, which is exactly
what the padding fixes. Sync is still global-memory-latency-bound.

### Path to higher throughput (future work)

Above the current worst-of-10 ceiling (115% async / 107% sync of
cuBLAS sgemm at 4096² on RTX 4090), the remaining headroom — and the
wider sync-vs-async gap at all sizes — is bounded by structural
choices this kernel hasn't yet made:

- **LDG.128 vectorized global loads (sync path).** The cooperative
  Tile B loader uses 8 × scalar `ld.global.f16` per thread. Switching
  to one `ld.global.v4.b32` per thread is an 8× instruction
  reduction on the global load side — the sync path's primary
  remaining lever. Sprint 6.7b landed the `MemoryOp::LdGlobalB128`
  IR primitive in kaio-core but deliberately did not wire it into
  any kernel: the companion "b32 → 2× b16 unpack" IR primitive needed
  for scattering into col-major shared was not worth the scope
  creep against 6.7b's D10 orthogonality requirement. A future
  sprint can design that primitive properly and then use the
  LDG.128 variant cleanly.
- **bf16 TC matmul / larger mma shapes** — deferred from Phase 7;
  tracked under Phase 9 kernel deepening.
- **ldmatrix.sync.aligned** — deferred from Phase 7; tracked under
  Phase 9 kernel deepening. The real path to closing the remaining
  sync-path gap.

## Quantized Matmul Performance (Sprints 7.1 + 7.2)

Same bench harness as TC matmul (5 warmups + 20 timed iterations per
run, `cargo xtask bench`), 10 consecutive runs on RTX 4090 sm_89,
release build. Worst / median / best distribution across runs.

**`matmul_int8` — W8A8 symmetric (i8 × i8 → s32 → scale → f32):**

| Size       | KAIO worst | KAIO median | KAIO best | cuBLAS sgemm worst | cuBLAS sgemm median |
|------------|-----------:|------------:|----------:|-------------------:|--------------------:|
| 256³       | 0.09 TOPS  | 0.09 TOPS   | 0.09 TOPS | 1.56 TF            | 1.88 TF             |
| 512³       | 0.66 TOPS  | 0.70 TOPS   | 0.72 TOPS | 11.33 TF           | 12.04 TF            |
| 1024³      | 5.11 TOPS  | 5.38 TOPS   | 5.49 TOPS | 31.86 TF           | 32.49 TF            |
| 2048³      | 32.23 TOPS | 32.77 TOPS  | 33.95 TOPS| 43.41 TF           | 43.47 TF            |
| **4096³**  | **84.07 TOPS** | **92.58 TOPS** | **93.38 TOPS** | **49.90 TF** | **56.00 TF** |

**`matmul_int4` — W4A16 GPTQ-style (packed s4 × f16 → f32 via dequant):**

| Size       | KAIO worst | KAIO median | KAIO best | cuBLAS sgemm worst | cuBLAS sgemm median |
|------------|-----------:|------------:|----------:|-------------------:|--------------------:|
| 512³       | 0.46 TOPS  | 0.47 TOPS   | 0.48 TOPS | 11.83 TF           | 11.96 TF            |
| 1024³      | 3.52 TOPS  | 3.66 TOPS   | 3.74 TOPS | 32.34 TF           | 32.54 TF            |
| 2048³      | 21.71 TOPS | 22.27 TOPS  | 22.67 TOPS| 43.38 TF           | 43.50 TF            |
| **4096³**  | **52.02 TOPS** | **57.52 TOPS** | **58.04 TOPS** | **45.55 TF** | **49.00 TF** |

### Apples-to-apples disclaimer for INT columns

The "cuBLAS sgemm" column is a **project-local compute-density
reference**, not a precision-matched comparison:

- `matmul_int8` is W8A8 (i8 in, s32 accumulate, scale to f32 on store).
  True apples-to-apples would be `cublasGemmEx` with `CUDA_R_8I` — that
  interface is not cleanly exposed by `cudarc` 0.19 at the time of
  writing (tracked as tech debt).
- `matmul_int4` is W4A16 with dequant-to-f16 + `mma.m16n8k16.f16.f16.f32`.
  Weight bandwidth alone is 0.5 B/weight vs 4 B for sgemm — an 8×
  memory-bandwidth advantage that dominates the ratio.

Use these columns for **sprint-over-sprint regression detection** on
the KAIO column; treat the "vs sgemm" percentage as indicative, not
definitive.

## Bench coverage today + roadmap

`cargo xtask bench` today runs three benchmark harnesses:
`matmul_tc_bench` (f16 TC sync + async), `matmul_int8_bench`, and
`matmul_int4_bench`. These are the kernels with direct-measurable
cuBLAS references for apples-to-oranges regression detection.

**Not yet in `xtask bench`** (Sprint 8.0.5 will extend coverage):

- `qkv_project_int8` / `qkv_project_int4` — fused tri-output QKV
  projections. Needs a "3× standalone `matmul_int{8,4}`" baseline to
  show the fusion advantage (~3× at decode shapes per internal
  benches).
- `attention_tc` / `attention_tc_causal` — FlashAttention-style
  tensor-core attention. Natural reference is cuDNN MHA; needs a plan
  doc to lock methodology.
- `attention_flash` — IR-API FlashAttention (online softmax).
- Showcase-example kernels (`rms_norm`, `layer_norm`, `softmax`,
  `fused_silu_gate`, `gelu_exact`/`gelu_fast`) — each self-times
  inside its own example binary today; Sprint 8.0.5 will promote
  those to the unified bench harness with consistent warmup +
  worst-of-N framing.

Until 8.0.5 lands, the "KAIO benches every kernel it ships" claim is
incomplete. Current benched coverage: matmul family (4 variants).
Unbenched kernels are correctness-tested but not performance-gated in
CI.
