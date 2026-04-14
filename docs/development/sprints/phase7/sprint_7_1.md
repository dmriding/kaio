# Sprint 7.1 — INT8 dequantize-matmul (Phase 7's quant headline)

**Status:** ✅ Complete
**Branch:** `sprint-7-1` off `main` (post v0.2.2 publish)
**Release target:** v0.3.0 — first minor bump since Phase 6; new public op

## Context

Phase 7.1 is the quant-kernels headline that attracts new users. The DSL
completeness work shipped in Sprint 7.0 (bitwise ops, signed/unsigned shift
preservation, compound bitwise assign) was explicitly scoped to unlock dequant
work. Sprint 7.0.5 shipped the ergonomics surface so cold-start adopters don't
bounce on friction before reaching quant.

The rust-lang.org forum engagement is also relevant: the first external
feature request was explicitly about 4/5/6-bit quantization. The v0.2.2 reply
committed publicly to shipping **INT8 dequantize-matmul as the reference
template**, with the DSL supporting custom bit-width variants beyond that.
Sprint 7.1 delivers the INT8 half of that promise.

Dequant-only (DSL example) is already shipped at `examples/int8_dequant/`.
This sprint delivers the **matmul-fusion** half — the IR-authored
`kaio_ops::matmul_int8` — because reading packed INT8 into registers,
unpacking + dequantizing in a separate pass, then feeding f32 into matmul
destroys the point of quantization. Dequant must fuse with the tensor-core
inner loop: unpack → dequant → feed mma.sync in registers, no round-trip
through shared or global.

## Public contract for v0.3.0 `matmul_int8`

This paragraph is the reference for what 7.1 actually ships. It appears
verbatim in the `matmul_int8` rustdoc, the example README, and this log so
users and future sprints share the same expectations:

**`matmul_int8` in v0.3.0 is:**
- **symmetric** (no zero point)
- **int8 × int8 → f32** (both operands quantized; W8A8 only)
- **single global scalar scale** (one `f32` applied to the full output)
- **sync-only** (async INT8 matmul deferred to 7.1.5+)
- **`K % 32 == 0` required** (plus M/N constraints per fragment shape)
- **the first reference quant op, not the final general quant architecture**
  — GPTQ, AWQ, per-channel, per-group, asymmetric, INT4, W8A16 all come
  later as additive refinements, not as unmet expectations

## The primary unknown — **resolved**

**Does `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` behave as expected
on RTX 4090 (sm_89)?** D1 answered: **yes**. Path FAST viable; the fallback
DEQUANT-F16 path was specced but never needed. The sprint ships on the direct
s8 → tensor-core → s32 → scale path with no fallback active.

## Deliverables — all shipped

**D1a — encoding viability (`01dd1f7`, `48c50a5`):**
`PtxType::S8` added as a memory/mma marker type with `.b32` register decl
(packed four-per-register). `MmaShape::M16N8K32` variant plus
`TensorCoreOp::MmaSyncInt8` emitter landing the full mnemonic
`mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`. Sibling fragment types
`FragmentA_M16N8K32` (4 × .b32), `FragmentB_M16N8K32` (2 × .b32),
`FragmentC_M16N8K32` (4 × .s32) with dedicated allocators —
no overloading of the f16 infrastructure (distinct types keep layout
differences visible to readers of the emitted PTX). `ptxas_verify_mma_int8`
offline-assembler PASS at sm_80+.

**D1b — operand layout correctness (`e9c4a5e`):**
Load/store helpers for global-source `m16n8k32` fragments. 7 adversarial GPU
round-trip tests (all-ones, ascending-byte, single-hot row, single-hot col,
alternating sign, boundary values incl. i8::MIN/MAX, comprehensive random) —
**all 7 bit-exact PASS** on sm_89. Raw s32 accumulator validated against CPU
`i8 × i8 → i32` reference with zero scale, so correctness is isolated from
the scale path.

**D2 — shared-source helpers (`a740d18`):**
`load_fragment_a_m16n8k32_shared_row` and `load_fragment_b_m16n8k32_shared_col`
with stride-parameterized shared-memory inputs, matching the f16 pattern.
3 new `PtxModule::validate` tests confirm the SM-80 gate on the INT8 path.
`ptxas_verify_mma_int8_shared` PASS.

**D3 — fused kernel in `kaio-ops` (`a4d24f2`):**
`matmul_int8_kernel.rs` — 1,000+ LOC IR-authored, mirroring
`matmul_tc_kernel.rs` structure. Block tile 64×64, 4 warps × 32×32 quadrants,
8 `mma.sync.m16n8k32.s8.s8.s32` per K-iter per warp. Cooperative tile loads:
b32 for A (row-major source → row-major shared, byte packing preserved),
byte-level for B (row-major source → col-major shared, one top-level
col-bounds bra-skip). Pre-zero shared tiles + bar.sync. Scale-and-cast
pipeline after the K-loop: `cvt.rn.f32.s32 + mul.f32 + @p st.global.f32`.

**D4 — public API (folded into D3):**
`matmul_int8(device, a: &GpuBuffer<i8>, b: &GpuBuffer<i8>, c: &mut GpuBuffer<f32>, scale: f32, m, n, k)`.
`validate_dims_int8` enforces `K % 32 == 0` with a readable
`KaioError::InvalidConfig` pointing at the mma K-tile structural requirement
(so the failure mode is never a cryptic ptxas error or silently-wrong
partial-tile output). `impl_gpu_type!(i8, PtxType::S8)` makes `GpuBuffer<i8>`
work with the runtime's alloc / h2d / d2h paths. W8A8 rustdoc warning on the
public surface — the type system enforces W8A8 at compile time, but many
local-LLM users expect W8A16 by default, so the explicit note prevents
mental-model confusion.

**D5 — showcase + xtask + bench (`669e0b3`):**
`examples/int8_matmul/` — full W8A8 pipeline demo: quantize f32 to i8 per
tensor, run `matmul_int8`, compare against naive f32 CPU matmul. Tolerance
reflects quantization error, not kernel error. `examples/int8_dequant/`
stays side-by-side as the DSL shift-and-mask primitive demo — the two
examples target different adopter personas (DSL authors vs ops-crate
consumers). `cargo xtask showcase int8matmul` wired as the short
name (compact form matching the `layernorm` precedent).
`kaio-ops/tests/matmul_int8_bench.rs` — internal-regression bench with
cuBLAS sgemm as a rough compute-density reference (apples-to-oranges loud
disclaimer in the bench output; true INT8-vs-INT8 baseline requires raw-FFI
`cublasGemmEx` which is out of scope for v0.3.0).

**D6 — docs, release prep (this commit):**
CHANGELOG 0.3.0 promotion, README feature-table row, phases.md Phase 7.1
completion marker, master plan row updated, version bumps 0.2.2 → 0.3.0
across all 5 crates.

## Results

### Correctness — all green

- **`ptxas_verify_mma_int8` + `ptxas_verify_mma_int8_shared`**: PASS (offline
  assembler accepts the mnemonic at sm_80+).
- **`kaio` adversarial fragment-layout tests** (7/7): PASS bit-exact vs CPU
  `i8 × i8 → i32` reference — boundary values, single-hot rows/cols,
  ascending byte pattern, alternating sign, comprehensive random.
- **`kaio-ops/tests/matmul_int8_e2e.rs`** (9/9): PASS — small-shape bit-exact
  (16×8×32, 64×64×32, 64×64×128, 128×128×128 to 1e-3 abs, 256×256×256 to
  1e-4 rel), edge-M (M=17), edge-N (N=13), boundary i8 values (i8::MIN,
  i8::MAX, -1, 0, 1), and K=31 validation-error test.

### Performance

Measured on RTX 4090 sm_89, `cargo xtask bench matmul_int8_bench` across
6 independent runs. Small sizes (≤1024³) are stable within ±5%; larger
sizes (2048³+) show meaningful run-to-run variance driven by thermal and
scheduler effects, so a single-run number would mislead — the numbers
below are medians across 6 runs, with the observed range in parentheses
where variance is material:

| Size      | KAIO int8 ms  | KAIO TOPS         | cuBLAS sgemm ms | cuBLAS TF         |
|-----------|---------------|-------------------|-----------------|-------------------|
| 256³      | 0.37          | 0.09              | 0.02            | ~1.7              |
| 512³      | 0.38          | 0.71              | 0.03            | ~11               |
| 1024³     | 0.40          | ~5.3              | 0.06–0.07       | 32–37             |
| 2048³     | ~0.54         | **30–35** (med 32)| 0.33–0.40       | 43–53             |
| **4096³** | **1.46–1.71** | **80–94** (med 89)| 2.35–2.66       | 52–58             |

The "vs cuBLAS" column is apples-to-oranges (int8 ops vs f32 ops), so treat
it as a rough compute-density indicator, not a headline claim. The useful
number for regression tracking is the **KAIO TOPS column** — at 4096³,
the **~89 TOPS median** (80–94 TOPS range) is the v0.3.0 first-ship
baseline. Small-size weakness (5–18% for ≤1024³) is kernel-launch-dominated
and not the optimization target for this sprint; the baseline at 2048³+
is where the tensor-core compute balance starts to matter.

Future regression tracking should compare against the **lower bound of
this range** (≈80 TOPS at 4096³) so normal variance doesn't trip false
alarms. Sustained runs below 75 TOPS would indicate a real regression.

**No regressions on the f16 path.** `matmul_tc_bench` post-D3 showed 4096²
TC async at 114.3% of cuBLAS sgemm, 2.28ms — within run-to-run noise of the
pre-sprint baseline (100-102%).

### Forked-decision status

Path FAST is the shipped implementation. Path DEQUANT-F16 was specced
as a fallback if the direct-s8 mma didn't work on sm_89; it was never
activated. The design note remains as a future mixed-precision (W8A16)
jumping-off point but is not on the 7.1 merge-to-main surface. The
48-hour D1b layout-debugging timebox was not consumed — layout passed
first try against the full adversarial matrix.

## Lessons / follow-ups

- **Byte-level B loader is slow but correct** — first-ship chose byte-level
  `ld.global.s8 + st.shared.s8` over a b32 + `prmt.b32` transpose for
  simplicity. At 4096³ the kernel is already competitive; the transpose
  optimization lands in a follow-up if profiling shows it's load-bound.
- **Shared-tile bank math matched prediction** — the `+4` padding from the
  f16 path carried over cleanly to INT8 tile_b (col-stride 36 B). Larger
  paddings (68 / 72 B) were held in reserve as a fallback and not needed;
  the kernel hits the expected throughput band without them.
- **Register count** — `ptxas --verbose` wasn't instrumented this sprint;
  tracked as tech-debt for 7.1.5+.
- **Scale path isolation paid off** — testing raw s32 against CPU `i32` first,
  then scaled f32 against CPU `(i32 as f64) * scale`, meant no scale bugs
  reached the size-matrix tests.
- **Sprint fit vs original plan** — the sprint shipped faster than the
  plan's D1-through-D6 branch-out suggested: D1b's adversarial matrix gave
  enough confidence to collapse D3 into a single pass rather than a gated
  spike. Direct-call public API, sibling fragment types for K=32, and the
  `PtxType::S8` marker type (no new scalar-arith surface) all removed
  scope that would otherwise have been open design questions.
