# Changelog

All notable changes to KAIO will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

Updated at phase completion. Per-sprint detail lives in
[docs/development/sprints/](docs/development/sprints/).

## [Unreleased]

### Sprint 7.2 — INT4 dequantize-matmul (`matmul_int4`)

GPTQ-style W4A16 dequantize-matmul: packed signed-INT4 weights × f16
activations → f32 accumulator via tensor cores. No independent
crates.io release — Phase 7 closes with an aggregate release when
7.1.5 → 7.4 are all complete.

#### Added — Sprint 7.2

- **`kaio_ops::matmul_int4`** — symmetric GPTQ-style W4A16
  dequantize-matmul. Signature: `matmul_int4(device, a: &GpuBuffer<f16>,
  b_packed: &GpuBuffer<u32>, scales: &GpuBuffer<f16>, c: &mut GpuBuffer<f32>,
  m, n, k, group_size)`. Group size fixed at **128** for v1;
  `K % 128 == 0` enforced. Requires SM 8.0+ (Ampere). DEQUANT-F16 path
  fed into `mma.sync.m16n8k16.f16.f16.f32`; no native INT4 tensor core
  on sm_80+, so per-lane unpack + sign-extend + `s32 → f32 → f16` cvt
  chain + scale fold is mandatory before the mma.
- **KAIO-defined packed weight + scale layout**. `b_packed: [K/8, N]`
  col-major u32 with 8 signed nibbles per u32 K-contiguous; `scales:
  [K/group_size, N]` row-major f16, one per `(group, output_col)`
  cell. NOT a drop-in replacement for external AutoGPTQ / exllama /
  GGUF packed model formats — users must repack to the KAIO
  convention. Full rustdoc carries the exact indexing formula.
- **Triple-layer sign-extend canary**. Emit-level token-stream test
  asserts `shr.s32` (not `shr.u32`) at all 8 nibble-extract sites;
  offline `ptxas_verify` gate; GPU e2e `matmul_int4_sign_extend_*`
  round-trip at `0x88888888` / `0x77777777` and mixed-position
  patterns like `0x87654321` — all three layers green on sm_89.
- **`PtxInstruction::MovPack`** (`kaio-core`) — vector-pack move
  primitive emitting `mov.b{N} %dst, {%s0, %s1, ...};`. Needed to
  pack two `.f16` registers produced by the INT4 dequant chain into
  one `.b32` for the tensor-core fragment B feed. Typeless `.b{N}`
  suffix (PTX ISA §9.7.9.10 requires it for the vector-pack form;
  `mov.u32` is rejected). Promoted to framework-level because 7.3
  quant-attention will also need it.
- **`ArithOp::Mul { ty: F16 | BF16, .. }` emitter fix** — was falling
  through to `mul.lo.f16` (integer-only suffix); now correctly emits
  `mul.f16` (PTX defaults `.rn` rounding on half-precision mul).
  Latent bug, uncovered while building the INT4 dequant chain. One
  emit test (`emit_mul_f16`) covers the regression.
- **`examples/int4_matmul/`** — full end-to-end showcase. Demonstrates
  GPTQ-lite symmetric per-column group quantization (`scale =
  max(|w_group|) / 7`, `q = clamp(round(w/scale), -8, 7)`), packing
  into the KAIO `[K/8, N]` col-major u32 layout (reference CPU packer
  `pack_s4_weights`), `matmul_int4` launch, and max-abs / max-rel
  error reporting. Tolerance set to 80% max-rel reflecting INT4's
  inherent noise floor (16 representable values vs 256 for INT8).
  Registered as `cargo xtask showcase int4matmul`.
- **`kaio-ops/tests/matmul_int4_bench.rs`** — median-latency bench vs
  cuBLAS sgemm across 512³/1024³/2048³/4096³. RTX 4090 sm_89
  release-mode: **4096³ median ~57 TOPS, range 49–58 across 6 clean xtask-bench runs**
  (80–101% of cuBLAS sgemm's 52–58 TFLOPS at the same shape). Variance
  reported honestly per the Sprint 7.1 discipline. Apples-to-apples
  disclaimer: KAIO moves 0.5 B per weight vs 4 B for sgemm — the
  comparison is a regression reference, not a definitive TOPS ratio.
- **12 `#[ignore]`d GPU round-trip tests** at
  `kaio-ops/tests/matmul_int4_e2e.rs`, all passing bit-exact on RTX
  4090 sm_89: smallest (16×8×128), sign-extend canaries, multi-group
  (K=256 with differing per-group scales), larger shapes
  (64/128/256³), M/N edges (17×8, 16×13), and 3 validation-error
  paths. Max rel error across all shapes: **4.1e-6** (target was
  1e-3).
- **`kaio_ops::validate_dims_int4`** — host-side shape + alignment
  gate invoked from `matmul_int4`. Rejects `group_size != 128`,
  `K % 128 != 0`, zero-sized dims, and undersized buffers with clean
  `KaioError::InvalidConfig` messages.

#### Scope — exactly as planned

- Symmetric INT4 only (no zero-points). Asymmetric GPTQ is a follow-up.
- Group size fixed at 128. Other group sizes (32 / 64 / 256) deferred.
- W4A16 only. W4A8 / W4A4 / W4A32 deferred.
- GPTQ-style packing only. GGUF Q4_0 / Q4_K / Q4_K_M require separate
  translation layers.
- Sync-only (no `cp.async` for INT4 — matches INT8 posture).
- No fused bias / activation epilogue.
- No auto-tuning across tile sizes or warp counts.
- `kaio-candle::matmul_int4` binding ships in Sprint 7.4.

### Sprint 7.1.5 — Warp + block reductions in the DSL

Expressiveness bridge sprint between quant milestones. Ships the warp-level
reduction primitives that unlock softmax / layer-norm / RMS-norm / loss
functions in pure DSL, plus the missing `min` variants at both warp and
block level. No independent crates.io release — Phase 7 closes with an
aggregate release when 7.1.5 → 7.4 are all complete.

#### Added — Sprint 7.1.5
- **`warp_reduce_sum(f32) -> f32`**, **`warp_reduce_max(f32) -> f32`**,
  **`warp_reduce_min(f32) -> f32`** — per-warp reductions via 5-round
  butterfly shuffle tree. Every lane in the warp gets the full reduction
  result; no shared memory or bar.sync required. In blocks larger than
  32 threads, each warp computes its own independent result (NOT
  block-wide — use `block_reduce_*` for block-wide).
- **`block_reduce_min(f32) -> f32`** — completes the `sum/max/min` trio
  at block level; matches the existing `block_reduce_sum/max` pattern.
- **Whole-warp-multiple compile-time guard** for `warp_reduce_*`: a
  kernel declared with a block size whose total thread count (product
  of all dimensions) is not a multiple of 32 fails to compile with a
  diagnostic pointing at the call. Catches the partial-warp `shfl.sync`
  UB statically — no runtime overhead, no silent misbehavior. Guards
  1D (`block_size = 16`) and 2D (`block_size = (4, 4)`, `(8, 6)` = 48)
  cases together.
- **`emit_warp_tree_reduce` helper** in `kaio-macros/src/lower/builtins.rs`
  — factored out of the existing `lower_block_reduce` path and
  parameterized by shuffle variant (`Down` for block_reduce's existing
  lane-0-only semantics, `Bfly` for warp_reduce's all-lanes semantics),
  combine op, and PTX type. The type parameterization leaves the door
  open for a clean follow-up sprint adding `warp_reduce_sum(i32)` /
  `(f16)` / `(bf16)` without touching the helper.
- **Pre-refactor TokenStream snapshot canary** at
  `kaio-macros/tests/snapshots/block_reduce_sum_f32.tokens.txt` — locks
  the pre-refactor block_reduce expansion structure. D2's helper
  extraction was PTX-byte-identical; the canary proved it. Scoped as a
  refactor canary (regenerate via `KAIO_UPDATE_SNAPSHOT=1` for
  intentional future changes); not a forever byte-identity prison.
- **GPU test coverage (12 new `#[ignore]`d tests in `kaio/tests/reduce_macro.rs`):**
  warp_reduce sum / max / min at 32 threads across all-ones, ascending
  lanes, single-hot, alternating-sign patterns; `block_reduce_min` at
  block sizes 32 / 64 / 128 / 256 / 512; a 2D-block `(8, 4)` warp_reduce
  test; and a **64-thread two-warp independence canary** (warp 0 and
  warp 1 fed different patterns; asserts each warp gets its own
  reduction and the lowering doesn't accidentally blend across warps).
  All 18 tests in the reduce_macro harness pass bit-exact on RTX 4090 sm_89.
- **4 new `trybuild` compile-fail fixtures:** block_size = 16,
  `(4, 4)` (2D product 16), `(8, 6)` (2D product 48 — whole-warp-
  multiple violation), and `i32` passed where `f32` is required
  (type-boundary lock before any future type-generalization sprint).

#### Changed — Sprint 7.1.5
- `lower_block_reduce` extended to accept `mode = "min"` alongside
  existing `"sum"` and `"max"`. Shared-memory layout and regression
  canaries (both symbol-presence and the new TokenStream snapshot)
  unchanged.
- Dispatch match at `kaio-macros/src/lower/builtins.rs` lists four
  new builtin names; `cf04_unknown_call.stderr` trybuild snapshot
  updated to match.

#### Rustdoc contract for warp_reduce_* (public stubs at `kaio/src/gpu_builtins.rs`)
- Per-warp semantics spelled out explicitly ("reduces across the
  calling thread's warp only; in blocks larger than 32 threads each
  warp computes its own independent result").
- Whole-warp-multiple block-size requirement with compile-time-check
  note.
- Convergent-control-flow requirement (data-dependent branches are UB
  that the macro cannot detect at expansion time; identity-value
  padding pattern documented for ragged boundaries — `0.0` for sum,
  `f32::NEG_INFINITY` for max, `f32::INFINITY` for min).
- NaN handling documented as implementation-defined per PTX ISA.

## [0.3.0] — 2026-04-15 — Sprint 7.1: INT8 dequantize-matmul (Phase 7 quant headline)

Sprint 7.1 ships the **reference INT8 symmetric dequantize-matmul**, the
first substantial new public op since Phase 6's v0.2.0 tensor-core matmul.
`matmul_int8` is W8A8 (both operands quantized), symmetric (no zero point),
uses a single global scalar scale applied post-accumulation, and is
sync-only for v0.3.0 (async INT8 follow-up tracked for 7.1.5+). This is
positioned as the **reference quant op, not the final general-quant API** —
GPTQ / AWQ / per-channel / per-group / asymmetric / INT4 / W8A16 all land
in follow-up sprints as additive refinements.

### Added — Sprint 7.1
- **`kaio_ops::matmul_int8`** — `matmul_int8(&device, a: &GpuBuffer<i8>,
  b: &GpuBuffer<i8>, c: &mut GpuBuffer<f32>, scale: f32, m, n, k)`.
  IR-authored fused kernel at
  [`kaio-ops/src/matmul_int8_kernel.rs`](kaio-ops/src/matmul_int8_kernel.rs)
  using the Ampere+ `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`
  instance shape directly (Path FAST per the Sprint 7.1 D1 fork decision).
  Block tile 64×64, 4 warps in 2×2 quadrants, 8 mma.sync per K-iter per
  warp. Scale is applied post-accumulation via `cvt.rn.f32.s32 + mul.f32`.
  `K % 32 == 0` is required (validated before launch with a readable
  `KaioError::InvalidConfig`). SM 8.0+ enforced by `PtxModule::validate`.
- **`PtxType::S8`** in `kaio-core` — marker / packed type for memory ops
  and mma operand suffixes only, not a scalar arithmetic type. Register
  declaration is `.b32` (packed four-per-register); instruction suffix
  is `.s8`.
- **`MmaShape::M16N8K32`** variant — `min_sm = 80`, token `"m16n8k32"`.
  Routes through the existing `PtxModule::validate` SM-gating path.
- **`TensorCoreOp::MmaSyncInt8`** — emits the full INT8 mma mnemonic.
  Sibling fragment types `FragmentA_M16N8K32` (4 × .b32),
  `FragmentB_M16N8K32` (2 × .b32), `FragmentC_M16N8K32` (4 × .s32) —
  distinct from the f16 fragment infrastructure so layout differences
  stay visible.
- **Fragment load/store helpers** for the K=32 INT8 layout (global and
  shared-source variants following the f16 stride-parameterized pattern).
- **`impl GpuType for i8`** — public API takes `GpuBuffer<i8>` directly.
- **`examples/int8_matmul/`** — new showcase demonstrating the full W8A8
  pipeline (quantize f32 to i8, run `matmul_int8`, compare vs naive f32
  CPU matmul). `cargo xtask showcase int8matmul` wired as the short name.
  `examples/int8_dequant/` stays side-by-side as the DSL primitive demo.
- **`kaio-ops/tests/matmul_int8_bench.rs`** — internal-regression bench.
- **`kaio-ops/tests/matmul_int8_e2e.rs`** (9 ignored GPU tests) — bit-exact
  round-trip at small sizes, within-tolerance at 256³, edge-M (M=17),
  edge-N (N=13), boundary i8 values, K=31 validation-error.
- **Adversarial fragment-layout test matrix** in `kaio/tests/mma_sync_int8_fragment.rs`
  (7 ignored GPU tests, all bit-exact PASS on sm_89): identity, all-ones,
  ascending-byte canary, single-hot row/col, alternating sign, boundary
  values (`i8::MIN`, `i8::MAX`, ±1, 0), comprehensive random.
- **ptxas-verify tests** — `ptxas_verify_mma_int8`,
  `ptxas_verify_mma_int8_shared` (9/9 total ptxas_verify tests green).

### Test coverage — Sprint 7.1

Workspace totals at v0.3.0: **329 host tests + 187 `#[ignore]`d GPU
tests**. Sprint 7.1 contributed 16 new GPU tests (7 adversarial
fragment-layout tests in `kaio/tests/mma_sync_int8_fragment.rs` + 9
end-to-end round-trip tests in `kaio-ops/tests/matmul_int8_e2e.rs`)
plus 2 new `ptxas_verify_mma_int8*` host tests and 5 new
`matmul_int8_kernel` structure tests. GPU tests are gated behind
`--ignored` and require an Ampere+ NVIDIA GPU — host CI remains green
without a GPU.

### Added — pre-promotion (landed during Sprint 7.0.5 post-v0.2.2 → rolled up)
- Three standalone showcase examples under `examples/`:
  [`layer_norm/`](examples/layer_norm/) (classic transformer LayerNorm,
  two block-wide reductions), [`softmax/`](examples/softmax/)
  (single-block softmax with subtract-max numerical-stability pattern),
  [`int8_dequant/`](examples/int8_dequant/) (symmetric INT8
  dequantization demonstrating the signed/unsigned shift distinction
  from v0.2.1). `cargo xtask showcase` now runs seven examples total;
  new short names `layernorm`, `softmax`, `int8`, `int8matmul`.

### Changed — Sprint 7.1
- `docs/phases.md` — Phase 7.1 row marked complete with the observed
  80–94 TOPS band at 4096³ (median ~89 across 6 runs).
- `phase7_master_plan.md` — corrected the stale `m16n8k16.s8.s8.s32`
  reference to the correct `m16n8k32.row.col.s32.s8.s8.s32` mnemonic.
- Version bumps 0.2.2 → **0.3.0** across all 5 crates (`kaio`, `kaio-core`,
  `kaio-macros`, `kaio-runtime`, `kaio-ops`). First minor bump since
  Phase 6's v0.2.0 — `matmul_int8` is a real new public op, not a polish
  release.

### Changed — pre-promotion
- Cleaned stale "after v0.2.0 publishes" Cargo.toml comments across
  all showcase examples.
- `examples/softmax` reuses `local_max` in the `exp(...)` step
  instead of re-reading `input[tid]`, avoiding a redundant global
  load per thread.

### Performance — v0.3.0 `matmul_int8` baseline (RTX 4090 sm_89)

Medians across 6 independent bench runs. Small sizes stable within ±5%;
2048³+ shows meaningful run-to-run variance driven by thermal/scheduler
effects, so the table reports observed ranges where variance is material:

| Size  | matmul_int8 ms | matmul_int8 TOPS    | cuBLAS sgemm ms | cuBLAS TF |
|-------|----------------|---------------------|-----------------|-----------|
| 256³  | 0.37           | 0.09                | 0.02            | ~1.7      |
| 512³  | 0.38           | 0.71                | 0.03            | ~11       |
| 1024³ | 0.40           | ~5.3                | 0.06–0.07       | 32–37     |
| 2048³ | ~0.54          | **30–35** (med 32)  | 0.33–0.40       | 43–53     |
| 4096³ | **1.46–1.71**  | **80–94** (med ~89) | 2.35–2.66       | 52–58     |

Apples-to-oranges disclaimer: int8 vs sgemm indicates compute density, not
equivalent-work throughput. The **KAIO TOPS column** is the regression
baseline. At 4096³ the sprint-over-sprint regression floor is ~80 TOPS
(lower bound of observed range); sustained runs below 75 TOPS would
indicate a real regression. Small-size weakness (≤1024³) is kernel-
launch-dominated and not the 7.1 optimization target. The f16 TC path
shows no regression: post-7.1 `matmul_tc_async` at 4096² runs at ~108% of
cuBLAS sgemm, within run-to-run noise of the pre-sprint baseline.

## [0.2.2] — 2026-04-14 — Sprint 7.0.5: Ergonomics fast-track before Phase 7.1

Sprint 7.0.5 is a narrow pre-7.1 ergonomics sprint. It ships a targeted set of adoption-friction fixes that set the table for first-time users arriving via the quant-kernels story. Bigger ergonomics items (module cache, buffer API, reductions, atomics) stay deferred — see [`docs/development/sprints/phase7/phase7_master_plan.md`](docs/development/sprints/phase7/phase7_master_plan.md) for sequencing.

### Added — Sprint 7.0.5
- **D1 (A2)** — One-time debug-build performance note. `KaioDevice::new` emits `[kaio] Note: debug build — GPU kernel performance is ~10-20x slower than --release. ... Correctness is unaffected. Set KAIO_SUPPRESS_DEBUG_WARNING=1 to silence.` on first call in a debug binary. Zero cost in release (`cfg!(debug_assertions)` folds to `false`). Prevents the common "benchmarked in debug, bounced" adoption failure. Message is performance-framed only — correctness testing in debug remains trustworthy.
- **D3 (B3)** — New [`docs/debugging.md`](docs/debugging.md) as the single entry point for diagnosis: troubleshooting flowchart, consolidated env-var reference, `compute-sanitizer` usage, tolerance-choice guidance for floating-point verification, PTX-stats interpretation, common-errors quick reference. README + `docs/index.md` link to it.
- **D5** — New [`cargo xtask`](xtask/) repo-tooling binary. `cargo xtask showcase` runs all three standalone examples from the repo root with no `cd`; `cargo xtask bench` runs the matmul tensor-core benchmark; `cargo xtask all` runs both. Continue-on-error semantics with a pass/fail summary. `.cargo/config.toml` alias makes `cargo xtask` a single verb. README "Try KAIO in 30 seconds" section pivots to use this.
- **D6** — `[package.metadata.docs.rs] targets = ["x86_64-unknown-linux-gnu", "x86_64-pc-windows-msvc"]` in all 5 published crates. docs.rs changed its default on 2026-05-01 to build only one target; this preserves KAIO's Windows-support visibility on docs.rs.

### Changed — Sprint 7.0.5
- **D2 (A4)** — Proc-macro error spans: fixed three identified sites (`kaio-macros/src/parse/attrs.rs` missing-block_size, `kaio-macros/src/lower/mod.rs` LitInt and LitFloat type mismatches) to use specific spans instead of `Span::call_site()`. **Honest finding**: audit of 63 error sites identified only these 3 as potentially improvable, and investigation showed the fixes are defensive (LitInt/LitFloat paths are unreachable per parser type-gating; `attr.span()` vs `call_site()` resolves to the same location for empty attributes). The macro's span handling is already quite good. The fixes stand on principle with zero cost.
- **D4** — [`docs/phases.md`](docs/phases.md) Phase 7 intro gains adoption-friction-rent framing. [`docs/development/sprints/phase7/phase7_master_plan.md`](docs/development/sprints/phase7/phase7_master_plan.md) adds an adoption-ergonomics sequencing section explicitly mapping each identified feedback item to its landing sprint (or its deferral / rejection with rationale). Sprint breakdown gains 7.0.5 and 7.1.5 (reductions) rows.
- README limitations section updated. "Try KAIO in 30 seconds" section added as the first action-oriented surface.

## [0.2.1] — 2026-04-14 — Sprint 6.10 + 7.0: Close open threads + DSL completeness

Sprint 6.10 landed on `main` without a version bump; Sprint 7.0 picked
up the DSL completeness work in the same patch-release window.
Publishing them together as v0.2.1.

### Added — Sprint 7.0 (DSL completeness + Phase 6 closeout + Phase 7 scaffold)
- **D1** — Six new `ArithOp` variants in `kaio-core`: `And`, `Or`, `Xor`, `Shl`, `Shr`, `Not`. `Shr` preserves signed/unsigned distinction per operand type (`shr.s32` for `i32`, `shr.u32` for `u32`) so Phase 7.1+ quant dequantization produces correct arithmetic vs logical shifts on signed packed values. All others use typeless `.b{size}` / `.pred` suffix. 11 emit unit tests + `ptxas_verify_bitops` integration test (now 7/7 ptxas_verify tests).
- **D2** — `#[gpu_kernel]` macro now lowers bitwise binary operators (`&`, `|`, `^`, `<<`, `>>`) and unary NOT (`!`). Unary `!` context-dispatches on source type: integer → bitwise NOT (`not.b32` / `not.b64`), `bool` → logical NOT on predicate (`not.pred`). Non-integer non-bool operands to bitwise / `!` produce clear compile errors. Six macro-level tests including signed/unsigned `Shr` round-trip canaries.
- **D3** — Compound bitwise assignment in `#[gpu_kernel]`: `&=`, `|=`, `^=`, `<<=`, `>>=`. Works on scalar and indexed lvalues (`arr[i] |= mask`). Rides the existing `IndexAssign` desugaring path — no new IR statement type.
- **D4** — Short-circuit `&&` / `||` in kernel bodies, Rust-faithful semantics. `if i < n && arr[i] > 0 { ... }` correctly skips the `arr[i]` read when `i >= n` (bounds-guarded access pattern). Two lowering paths: branch-direct inside `if` conditions (no intermediate predicate register), materialized `.pred` result in expression position (`let mask = a && b;`). Label allocation handles nested / interleaved cases (`(a && b) || (c && d)`). 17 new GPU round-trip tests across bitops, compound bitops, and short-circuit paths; 6 codegen regression tests with named mutation canaries.
- **Phase 6 closeout** — `docs/phases.md` Phase 6 promoted from "Post-v0.1 Roadmap / not a commitment" to fully documented ✅ section matching Phase 4/5 format (Status / Deliverables / Sprint Breakdown / Key Decisions / Performance). `docs/success-criteria.md` gained a Phase 6 section covering the 10 sprints (6.1–6.10 plus 6.7b) with measured outcomes.
- **Phase 7 scaffold** — `docs/development/sprints/phase7/phase7_master_plan.md` + `sprint_7_0.md` land the structure for Phase 7 quantized-kernels work. Master plan mirrors the Phase 5 / Phase 6 template (Architectural Constraints / Decisions / Sprint Breakdown / Success Criteria / Risks).

### Added — Sprint 6.10 (Close open threads)
- **D2** — Four host-level codegen regression tests in `kaio-macros` (no GPU required): `launch_wrapper_emits_correct_block_dim_1d` / `_2d`, `shared_memory_lowering_emits_shared_addr_pattern`, `reduction_lowering_uses_named_symbol`, `launch_wrapper_threads_compute_capability_into_module_build`. CI can now catch macro codegen regressions without a GPU runner. Each test has a regression canary comment; one mutation verified end-to-end.

### Changed — Sprint 7.0
- `CHANGELOG.md` duplicate `[0.2.0]` header artifact removed.
- `docs/development/tech_debt.md` — three DSL items marked **RESOLVED Sprint 7.0**: `&&` / `||` logical operators, compound assignment for shared memory (bitwise variants; arithmetic already worked in Phase 3), `ArithOp::Shr / Shl / And / Or` bitops.

### Changed — Sprint 6.10
- **D1a** — `#[gpu_kernel]` macro-generated `launch()` now uses `device.load_module(&PtxModule)` instead of `device.load_ptx(&str)`. The macro threads `device.info().compute_capability` through to a `build_module(sm: &str) -> PtxModule` helper. User-authored kernels now flow through `PtxModule::validate()` before ptxas — SM mismatches surface as structured `KaioError::Validation` instead of cryptic ptxas errors. `#[gpu_kernel]` user API is unchanged.
- **D1a** — `PTX_CACHE: OnceLock<String>` removed from macro codegen; modules rebuilt per launch. Measured no regression at 4096² async matmul (2.31ms vs 2.32ms baseline, within run-to-run noise). Small-matrix (256²) host-side overhead +0.01ms — expected cost-model change from cache removal.
- **D1a** — Migrated 4 non-macro test call sites (`vector_add_e2e`, `cp_async_roundtrip`, `mma_sync_fragment`) from `load_ptx(&str)` to `load_module(&PtxModule)`.
- **D3** — `kaio-core/tests/common/mod.rs` helpers `build_mma_sync_ptx`, `build_mma_sync_shared_ptx`, `build_cp_async_ptx` now take an explicit `sm: &str` argument. Three `unsafe { std::env::set_var("KAIO_SM_TARGET", ...) }` calls removed from `kaio-core/tests/ptxas_verify.rs`. Test hygiene landmine under parallel runners eliminated.

### Deprecated — Sprint 6.10 D1b
- `KaioDevice::load_ptx(&str)` is `#[deprecated(since = "0.2.1", note = "use load_module(&PtxModule) — runs PtxModule::validate() for readable SM-mismatch errors")]`. Public API preserved (not removed); raw-PTX use cases (external PTX files, hand-written PTX research) remain supported. Migration-guide rustdoc added with before/after example.

## [0.2.0] — 2026-04-13 — Phase 6: Tensor Cores & Async Copies

**Headline:** `kaio_ops::matmul_auto_tc` reaches **92.5% of cuBLAS sgemm
at 4096²** on RTX 4090 (async path) — pure Rust, no CUDA C++, no
Python, no toolchain. fp16 × fp16 → fp32 accumulation.

**Highlights:**
- Full tensor-core matmul stack (Ampere+): public `matmul_tc`,
  `matmul_tc_async`, `matmul_auto_tc` with an auto-tuner cache and a
  size-heuristic cache-miss default.
- fp16 / bf16 type support and packed-half2 register allocator in
  `kaio-core` (`mma.sync`, `cp.async`, fragment containers).
- `PtxModule::validate()` + `KaioDevice::load_module(&PtxModule)` for
  SM-gated validation at kernel-load time.
- Three standalone showcase examples (`examples/fused_silu_gate`,
  `gelu_comparison`, `rms_norm`) that build from a fresh clone.
- Measured TC matmul results across 256-4096, with apples-to-apples
  disclaimer in `docs/performance.md` + `docs/benchmarks.md`.

For the full per-sprint detail, see the sections below. Sprint
documentation lives under
[docs/development/sprints/phase6/](docs/development/sprints/phase6/).

### Added — Sprint 6.1 (fp16/bf16 types)
- `PtxType::F16` / `PtxType::BF16` with `.f16` / `.bf16` PTX suffixes.
- `RegKind::H` (`%h`) / `RegKind::Hb` (`%hb`) register classes with
  independent counters from `%r`/`%f`.
- `cvt` rounding modes generalized for float-to-float (incl. half)
  conversions — emits `.rn` consistently.
- `half` crate as kaio-core's first dependency; cudarc `f16` feature
  enabled. `GpuBuffer<f16>` / `GpuBuffer<bf16>` roundtrip on the device.
- `half::f16` and `half::bf16` implement `GpuType`.

### Added — Sprint 6.2 (mma.sync + cp.async + fragments)
- **`TensorCoreOp::MmaSync`** with `MmaShape::M16N8K16` — emits
  `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` (and the `bf16`
  variant). Ampere+ (SM 8.0+) only.
- **Typed fragment containers** — `FragmentA` (4 × `.b32` packed half2),
  `FragmentB` (2 × `.b32` packed half2), `FragmentC` (4 × `.f32`). Pure
  register bags with `pub` fields; helpers (`alloc_a/b/c`,
  `load_fragment_*_global_*`, `store_fragment_c_global_row`) live as
  free functions in `kaio-core::fragment`.
- **`RegisterAllocator::alloc_packed_half2`** — explicit allocator for
  the `.b32` registers `mma.sync` expects for A/B fragments.
- **`MemoryOp::CpAsync*`** variants: `CpAsyncCaSharedGlobal`
  (`cp.async.ca.shared.global [shared], [global], N` for N ∈ {4,8,16}),
  `CpAsyncCommitGroup`, `CpAsyncWaitGroup { n }`. Size validated at
  construction via `MemoryOp::new_cp_async_ca`.
- **`PtxModule::validate()`** + `ValidationError::SmTooLow` — rejects
  kernels at emit time when they use features (`mma.sync`, `cp.async`)
  requiring a higher target SM than the module declares. Surfaced via
  `kaio-runtime::KaioDevice::load_module`.
- **`KaioDevice::load_module(&PtxModule)`** — preferred entrypoint for
  in-memory modules; validates before handing text to the driver.
- **Gate test** `mma_sync_m16n8k16_fragment_gate` — single mma.sync on
  known-value 16×16 × 16×8 inputs, bit-exact assertion (passes on RTX 4090).
- **Smoke test** `cp_async_ca_roundtrip_4_floats` — primitive global →
  shared → global roundtrip via `cp.async.ca` + `commit_group` +
  `wait_group 0` (passes on RTX 4090).
- ptxas verification for `mma.sync` and `cp.async` at `sm_80+`.
- `KernelStats`: `mma`, `cp_async`, `cp_async_commit`, `cp_async_wait`.

### Changed — Phase 6
- Phase 6 master plan corrected: `mma.sync.m16n8k16` requires SM 8.0+
  (Ampere), not SM 7.0+ (Volta). Earlier Volta/Turing shapes (`m8n8k4`,
  `m16n8k8`) are out of scope.

### Added — Sprint 6.3 (IR-level tensor-core matmul)
- **`kaio_ops::matmul_tc`** — first IR-authored tensor-core matmul in
  KAIO. Computes `C = A × B` with f16 × f16 → f32, `m16n8k16.f16.f32`.
  **Internal (`#[doc(hidden)]`)** for now — requires `M % 16 == 0`,
  `N % 8 == 0`, `K % 16 == 0`. Promotes to a public API in Sprint 6.7
  once edge-tile handling lifts the divisibility constraint.
- **Shared-memory fragment loaders** in `kaio-core::fragment`:
  `load_fragment_a_m16n8k16_shared_row` and
  `load_fragment_b_m16n8k16_shared_col` — free-function siblings to
  the Sprint 6.2 global-source loaders, with a caller-supplied
  `row_stride_bytes` / `col_stride_bytes` parameter for non-native
  tile strides.
- 4 GPU correctness tests (tiny, small, rect, medium) — bit-close to
  CPU reference (max error ~1e-7 against a tolerance of 1e-2).
- 6 host-only `validate_dims_tc` tests + 1 PTX-structure test + 1
  ptxas-verify test for shared-source loaders.

### Fixed — Sprint 6.3
- **`ld.global.f16` / `st.shared.f16` are not valid PTX.** Sprint 6.1's
  `MemoryOp::LdGlobal`/`StGlobal`/`LdShared`/`StShared`/`LdParam`
  emitted the wrong type modifier for half-precision loads/stores —
  PTX ISA §8.7.9 lists `f16` and `bf16` as invalid types for `ld`/`st`.
  The correct form loads a 16-bit half into an `.f16` register via
  `ld.global.b16`. Added `PtxType::ptx_memory_suffix()` that collapses
  `F16`/`BF16` to `.b16`. Register declarations and `cvt` still use
  `.f16`. Found by running the first kernel that actually executed an
  f16 load/store on the GPU.
- **`store_fragment_c_m16n8k16_global_row` row stride was hardcoded.**
  The 6.2 helper assumed a standalone 16×8 D matrix (32-byte row
  stride). For a 16×8 tile inside a larger M×N output (the matmul
  case), the stride needs to be `N * 4` bytes. Added a
  `row_stride_bytes: u32` parameter. Callers: `matmul_tc` emits the
  D-store inline (runtime-valued stride), Sprint 6.2's gate test
  passes `32`.

### Changed — Sprint 6.3
- **API change (Sprint 6.2 caller):** `store_fragment_c_m16n8k16_global_row`
  now takes a `row_stride_bytes: u32` parameter. Pre-6.3 callers
  pass `32` to match old behavior.

### Added — Sprint 6.4 (cp.async double-buffered matmul)
- **`kaio_ops::matmul_tc_async`** — double-buffered variant of
  `matmul_tc` that stages A tiles via `cp.async.ca.shared.global`
  (16 B per thread) while B stays synchronous. Pipeline preamble
  issues `A[0]` + commit; K loop waits on current A, issues next A
  before `mma.sync`, recomputes buffer offsets. Same dimension
  constraints as `matmul_tc` (M%16 = N%8 = K%16 = 0) and same
  SM 8.0+ requirement. **Internal (`#[doc(hidden)]`)** for now;
  promotes to public API in Sprint 6.7 alongside `matmul_tc`.
- Shared memory layout uses two buffers per tile: `tile_a[1024]`
  (2 × 512 B, `.align 16` for cp.async destination alignment) and
  `tile_b[512]` (2 × 256 B). Per-block shared = 1.5 KB.
- `kaio-ops::matmul_tc_kernel::emit_load_b_tile` and
  `validate_dims_tc` promoted from private to `pub(crate)` so the
  async sibling module can reuse them (visibility bump only,
  no behavioral change).
- 4 GPU correctness tests (tiny, small, rect, medium) — bit-close
  to CPU reference with identical error floor as 6.3 (max error
  ~1e-7 against a 1e-2 tolerance).
- 2 host tests — `buffer_offsets_toggle` (pure toggle math) and
  `build_matmul_tc_async_ptx_produces_valid_structure`
  (instruction-centric PTX check: cp.async mnemonics, shared-decl
  sizing, exactly-one mma, exactly-two commit_groups).
- Optional env-gated (`KAIO_SPRINT_6_4_TIMING=1`) timing log on the
  medium test — records async vs sync `ms/iter` for Sprint 6.7's
  multi-warp restructure to improve against. Not a benchmark, not
  in CI output. Current baseline: async is ~7% slower than sync at
  1 warp / block — as predicted, overlap gains require 6.7.

### Added — Sprint 6.5 (TC auto-tuner + `load_module` migration)
- **`kaio_ops::matmul_auto_tc`** — **first Phase 6 user-facing API**.
  Tensor-core auto-tuner for f16 × f16 → f32 matmul. Dispatches
  between `matmul_tc` (sync) and `matmul_tc_async` (`cp.async`
  double-buffered) based on cached benchmark data. Conservative
  default on cache miss: sync variant (matches 6.4's observation
  that async is ~7% slower at 1 warp/block; Sprint 6.7 will likely
  invert this default).
  - **Narrow contract, deliberately temporary.** This is a preview
    surface landing in 6.5 to unblock users who want TC dispatch
    today; production performance (60%+ cuBLAS) ships in Sprint
    6.7's multi-warp restructure — *not* here.
  - **Hardware:** NVIDIA Ampere or newer (SM 8.0+). Pre-Ampere
    callers get a clean `KaioError::InvalidConfig` naming both real
    fallback options (pad/convert inputs, or use the f32
    `matmul_auto` path if f16 precision is not required).
  - **Shape:** `M % 16 == 0 && N % 8 == 0 && K % 16 == 0` —
    **temporary**. Sprint 6.7 will relax via edge-tile handling.
  - **Performance:** single-warp-per-block under the hood.
    Correctness-validated, not yet at the Phase 6 target.
- **`kaio_ops::tune_matmul_tc`** — benchmarks both TC variants at
  the given dimensions, caches the faster one to the shared tuner
  cache file (`~/.cache/kaio/tune_cache.json` or the `KAIO_TUNE_CACHE`
  override). Entries are keyed by `(kernel="matmul_tc", sm_target,
  dims)`, coexisting with scalar `matmul` cache entries.
- **`matmul_tc` and `matmul_tc_async` migrated to `load_module`.**
  Both kernels now emit a `PtxModule` via
  `build_matmul_tc_module(sm)` / `build_matmul_tc_async_module(sm)`
  (replacing the older `build_*_ptx() -> String`) and call
  `device.load_module(&module)`. `PtxModule::validate()` inside
  `load_module` catches sub-Ampere targets cleanly with
  `ValidationError::SmTooLow`, surfaced as `KaioError::Validation`.
  The ad-hoc `device.info().compute_capability` checks in both host
  APIs are deleted.
- New shared test helpers at `kaio-ops/tests/common/mod.rs`
  (`patterned_f16_data`, `cpu_matmul_f16xf16_f32`,
  `assert_close_with_k_scaled_tol`) consolidate what 6.3 / 6.4 /
  6.5 test files were duplicating.
- Host regression tests for `load_module` behavior: four new unit
  tests per kernel asserting sub-Ampere targets produce
  `ValidationError::SmTooLow` with `required=80` and the correct
  feature name; plus a cache-coexistence unit test proving
  `matmul` and `matmul_tc` entries don't collide on shared cache
  keys.

### Breaking — Sprint 6.5
- **`KAIO_SM_TARGET` no longer affects tensor-core kernels.** Both
  `matmul_tc` and `matmul_tc_async` (and therefore `matmul_auto_tc`)
  now derive the emitted module's target SM from
  `device.info().compute_capability` at call time rather than from
  the `KAIO_SM_TARGET` env var. The env var continues to be honored
  by scalar `#[gpu_kernel]` kernels where cross-SM testing and
  pre-Ampere support genuinely matter; for tensor-core kernels it
  was always "lie to the kernel about the GPU," which is the
  problem `load_module` validation was built to solve.
  - **User impact:** near-zero — both kernels are still
    `#[doc(hidden)]` until Sprint 6.7, and public-API users only
    reach them through `matmul_auto_tc` which derives SM from the
    device regardless.
  - **Migration:** if you were setting `KAIO_SM_TARGET` specifically
    to target a TC kernel, stop — the kernel will pick the right
    SM based on the actual device. For scalar kernels, the env var
    still works as before.

### Added — Sprint 6.6 (fused TC attention, internal preview)
- **`kaio_ops::attention_tc`** — first fused tensor-core scaled
  dot-product attention in KAIO. `f16 Q × f16 K × f16 V → f32 out`
  via two back-to-back `mma.sync.m16n8k16` instructions with an
  intra-kernel `cvt.rn.f16.f32` bridge between the f32 softmax
  output and the f16 input to the second matmul — the architectural
  contract every production TC-attention implementation (FlashAttention
  v2, xFormers, FasterTransformer) depends on, validated here on
  RTX 4090 bit-close to the CPU reference. `#[doc(hidden)] pub use`
  — **internal preview only** until Phase 7's FlashAttention-TC
  lands and `attention_auto_tc` becomes the real user-facing
  dispatcher (matches the `matmul_auto_tc` pattern from Sprint 6.5).
  - Narrow contract: SM 8.0+ (Ampere or newer), `seq_q % 16 == 0`,
    `seq_k % 16 == 0`, `d_k % 16 == 0`, `d_v % 8 == 0`,
    `seq_k ≤ 384`, `d_k ≤ 128`, `d_v ≤ 128`. Divisibility and
    `seq_k` cap lifted at Phase 7; the seq_k ceiling exists because
    the full softmax scores matrix lives in shared memory (Phase 7's
    online softmax eliminates it).
  - Correctness-first, **not fast**. Single-warp-per-block kernel
    targeting the 6.3 bring-up philosophy — deliberately slow at
    realistic sizes, restructured for throughput at Sprint 6.7.
- **`kaio_ops::attention_tc_causal`** — standard decoder causal-
  masked variant, same signature. Build-time `causal: bool` flag
  drives PTX emission (zero-runtime-cost branching; two distinct
  modules from one Rust builder). Applies `-3.4e38` mask between
  matmul1 and softmax via `setp.gt.u32` + `selp.f32` per-lane
  branchless select. Global-coordinate math regression-gated by
  a dedicated `row0_self_only` canary test.
- **`kaio_core::instr::ArithOp::Selp`** — new IR variant emitting
  `selp{ty} dst, a, b, p` (PTX ISA §9.7.8.1). Branchless conditional
  assignment; required by the 6.6b causal mask and generally useful
  for any lane-predicated data-flow that should avoid warp divergence.
- Per-binary `#[allow(dead_code)]` on `kaio-ops/tests/common/mod.rs` —
  shared helpers file is compiled per test binary and any one binary
  sees different subsets as "used"; the allow silences the false
  positives without hiding genuinely-unused code.
- Eleven new GPU correctness tests on RTX 4090: five shapes for
  `attention_tc`, five shapes for `attention_tc_causal`, plus the
  causal row-0 canary. Five new host unit tests locking down module
  shape (mma count, cvt presence, mask-op presence for causal),
  `sm_70` rejection via `PtxModule::validate()` for both variants,
  and a shared-memory budget regression test (worst-case
  `SharedDecl` sum + alignment ≤ 46 KB). Test counts: 275 host / 133 GPU.

### Added — Sprint 6.7 (multi-warp TC matmul + edge tiles + benchmark + promotion)
- **`kaio_ops::matmul_tc` and `kaio_ops::matmul_tc_async` promoted to
  stable `pub`** (no longer `#[doc(hidden)]`). The `matmul_auto_tc`
  tuner-dispatched entry point graduates from "Sprint 6.5 preview"
  framing — measurable throughput-class TC matmul is now part of the
  KAIO public surface. **Stable public API, with measurable performance
  uplift over the Sprint 6.5 preview** — still room for additional
  headroom in Sprint 6.7b (vectorized loads + bank-conflict padding)
  and Phase 7 (bf16, larger mma shapes).
- **Multi-warp 64×64 block tile** — block dim becomes `(32, 4, 1)` with
  4 warps per block, each warp owning a 32×32 sub-quadrant computed via
  8 × `mma.sync.m16n8k16` per K-iteration in a 2 m_stripes × 4 n_stripes
  grid. Replaces the Sprint 6.3/6.4 single-warp 16×8-tile-per-block
  layout that spawned ~131k blocks at 4096² with only 32 threads each.
  The new layout lands ~16 resident warps per SM on Ampere+/Ada — full
  occupancy class.
- **Edge-tile predication** — `M` and `N` are no longer required to be
  multiples of 16 / 8. Per-thread bounds checks (cooperative tile loads
  pre-zero shared then bra-skip OOB rows/cols; output stores predicate
  via `setp.lt.and.u32` combining row+col bounds in one instruction)
  handle ragged dimensions. `K % 16 == 0` remains required — the
  mma.sync.m16n8k16 K-tile is structural and the kernel does not pad K
  inside a K-iteration.
- **`matmul_tc_bench.rs` (NEW)** — kaio-ops integration bench for TC
  sync + TC async + cuBLAS sgemm at 256/512/1024/2048/4096. 5 warmup
  + 20 timed-iter median. Run with `cargo test -p kaio-ops --test
  matmul_tc_bench -- --ignored --nocapture`.
- **Measured performance on RTX 4090 (sm_89), 4096²:**
  - TC sync (`matmul_tc`): **46.5 TFLOPS**, **79.9% of cuBLAS sgemm**.
  - TC async (`matmul_tc_async`): **49.6 TFLOPS**, **85.1% of cuBLAS sgemm**.
  - cuBLAS sgemm reference: 58.2 TFLOPS.
  - **Apples-to-apples disclaimer:** KAIO TC matmul uses fp16 inputs
    with fp32 accumulation; cuBLAS sgemm is f32. Comparison is against
    sgemm because it's the existing supported benchmark path in this
    repo (cudarc 0.19's `Gemm::gemm`). Results should be read as a
    project-local performance baseline, not a claim of apples-to-apples
    precision identity.
- **Tuner conservative-default flip (D6):** `matmul_auto_tc` now
  defaults to `TensorCoreAsync` on cache miss (previously
  `TensorCore`). Multi-warp restructure inverts the Sprint 6.4 single-
  warp observation — at 4096² async wins by ~6.5%.
- **`kaio_core::instr::MemoryOp::LdGlobalPred` and `StGlobalPred`** —
  predicated global memory ops (`@[!]p ld.global` / `st.global`).
  Standard pattern for edge-tile bounds checking; first user is the
  multi-warp matmul output store.
- **`kaio_core::instr::ControlOp::SetPAnd`** — `setp.{cmp}.and.{ty}
  dst, lhs, rhs, src_pred` for predicate composition in one PTX
  instruction. Eliminates the need for a separate `and.pred` step
  when combining row + col bounds for OOB stores.
- 12 new pathological-shape GPU tests (6 per kernel): sub-tile
  (7×5×16, 15×7×16), off-by-one against mma boundary (17×9×16,
  33×17×16), mid-range mixed (100×50×64), large off-by-one
  (1023×1023×1024). Plus a per-warp quadrant canary at 64×64×64
  (catches per-warp routing bugs that uniform inputs would mask).
  Test counts: **279 host / 148 GPU** workspace-wide (sync 11 + async 10
  + tuner 6 for TC alone, plus 6.6 attention 11 + 6.5 tuner_test +
  remaining scalar/attention coverage).
- 4 new kaio-core unit tests for the new IR variants
  (`emit_ld_global_pred_b32`, `emit_ld_global_pred_negated_b32`,
  `emit_st_global_pred_f32`, `emit_setp_and_lt_u32`).

### Breaking — Sprint 6.7
- **`matmul_tc` / `matmul_tc_async` divisibility relaxed.** Inputs
  with `M % 16 != 0` or `N % 8 != 0` previously returned
  `KaioError::InvalidConfig`; they now produce correct output via
  edge-tile predication. Code that relied on the error being raised
  (e.g., for input padding logic) needs to either drop the padding
  or check the dimensions client-side.
- **`matmul_auto_tc` cache-miss default is now a size heuristic**
  (`max(M, N, K) >= 3072` → `TensorCoreAsync`, else `TensorCore`).
  Previously always `TensorCore`. Matches the full 6.7 bench curve
  (sync wins 256-2048, async wins 4096). Per-shape cache hits
  override the heuristic; this only affects the first-call path
  before tuning is run. See D6 in
  `docs/development/sprints/phase6/sprint_6_7.md` for the measurement
  and review notes (2026-04-12).
- **`kaio_ops::matmul_tc` and `kaio_ops::matmul_tc_async` are now
  stable `pub` exports.** They were previously `#[doc(hidden)] pub
  use` (test-reachable but not part of the documented surface). Code
  that imported them via `use kaio_ops::matmul_tc;` continues to
  work; code that relied on them being hidden from `cargo doc` output
  will now see them in rustdoc.

### Added — Sprint 6.8 (showcase examples for v0.2.0)
- **`examples/fused_silu_gate/`** — gated SiLU activation
  (`x * sigmoid(x) * gate`), the feedforward primitive in every
  LLaMA / Mistral / Qwen block. Demonstrates `exp` builtin + elementwise
  fusion pattern in ~7 lines of kernel code.
- **`examples/gelu_comparison/`** — exact (tanh) vs fast (sigmoid)
  GELU, side-by-side with per-variant correctness + median timing.
  Includes a "bandwidth-bound teaching moment" section in the README
  explaining why the two variants run at identical wall-clock speed
  despite the compute-op asymmetry — and why kernel fusion matters
  more than arithmetic optimization for ML workloads.
- **`examples/rms_norm/`** — single-block RMSNorm
  (`out[i] = (x[i] / rms) * weight[i]`) covering the
  `block_reduce_sum + sqrt + divide` integration pattern. The README
  is explicit about the single-block limitation (`hidden_dim <=
  block_size`) and names the post-v0.2.0 path to multi-block RMSNorm
  (cross-block reduction primitive or `kaio_ops::rms_norm`).
- Each example is a **standalone Cargo project** with its own
  `[workspace]` table so `cargo run --release` works from a fresh
  clone without touching the parent workspace. `kaio` dependency is
  a path dep during pre-publish; Sprint 6.9 will flip it to
  `kaio = "0.2.0"` at release time.

### Changed — Sprint 6.8
- Workspace `Cargo.toml` gained `exclude = ["examples/*"]` — cosmetic
  belt-and-braces; the per-example empty `[workspace]` tables are what
  actually detaches them, but the workspace exclude makes the intent
  explicit.

### Added — Sprint 6.7b (bank-conflict padding + D10 fragment-loader hoist)
- **Tensor-core matmul async path hits 92.5% of cuBLAS sgemm at 4096²**
  on RTX 4090 sm_89 (up from Sprint 6.7's 85.1%, +7.4pp). Sync path at
  82.3% (up from 79.9%, +2.4pp). This is the v0.2.0 launch headline:
  pure-Rust-authored GPU kernel within 7.5% of a hand-tuned NVIDIA
  library, no CUDA C++ required. Project-local-baseline disclaimer
  unchanged — KAIO is fp16 in / fp32 accumulation vs cuBLAS sgemm
  (f32 in / f32 out); bandwidth asymmetry is part of the value
  proposition.
- **`load_fragment_a_m16n8k16_shared_row` and
  `load_fragment_b_m16n8k16_shared_col`** gained a
  `group_tig_override: Option<(Register, Register)>` parameter — kaio-
  core API. Callers that invoke the loaders multiple times per K-iter
  can compute `(group_id, thread_id_in_group)` once per block and pass
  them here, saving 6 × `div.u32`/`rem.u32` pairs per K-iter. `None`
  preserves the pre-6.7b behaviour (loader computes locally). Resolves
  the long-deferred D10 tech-debt item from Sprint 6.2.
- **Shared Tile B col-stride padding** for bank-conflict relief on the
  fragment-B hot path — stride bumped from 32 B to 36 B per column
  (one 4-byte pad per col; total tile 2304 B data + round-up tail =
  2560 B to satisfy the cooperative pre-zero pass's `THREADS_PER_BLOCK
  × 4` divisibility). Fragment B loader already accepted the stride as
  a parameter; no loader code touched. Bank math: `(group_id·9 + tig)
  mod 32` — most banks 1-way accessed, only 3 banks remain 2-way (vs
  all 16 distinct banks 2-way at stride 32). Measured 7.4pp lift on
  the async path alone.
- **`MemoryOp::LdGlobalB128`** — new kaio-core IR primitive for
  single-instruction 128-bit vectorized global loads (`ld.global.v4.b32
  {%r_i, %r_j, %r_k, %r_l}, [%rd_addr];`). Constructor validates that
  all 4 destinations are b32-class registers. Includes `ptxas_verify`
  coverage at sm_70. **Not wired into any kernel in 6.7b** — the
  primitive ships as well-formed unused IR for a future sprint that
  designs the companion b32-to-b16 split primitive. Kept orthogonal
  per Sprint 6.7b's D10 fallback protocol.

### Changed — Sprint 6.7b
- Sprint 6.7b's tile-B shared constants (`TILE_B_COL_STRIDE_BYTES`,
  `TILE_B_BYTES`) are now `pub(crate)` in `matmul_tc_kernel.rs` and
  imported by `matmul_tc_async_kernel.rs` — single source of truth.
  The async kernel's previous local copies are gone, guaranteeing the
  cooperative store and fragment-B read layouts can never drift.

## [0.1.0] — Phase 5: Fused Attention & Community Release

### Added — Phase 5
- **2D block reductions**: `block_reduce_sum/max` now work in 2D kernels
  via linear thread identity (`tidx + tidy * block_dim_x`).
- **Standard attention**: `attention()`, `attention_causal()` — single-head
  scaled dot-product attention with optional causal masking. Three-kernel
  decomposition (Q*K^T, softmax, P*V).
- **FlashAttention**: `attention_flash()`, `attention_flash_causal()` —
  O(d_k) memory per query, no materialized attention matrix. BLOCK_M=1
  design with online softmax and running output rescaling.
- **Auto-tuner**: `tune_matmul()`, `tune_attention()` benchmark kernel
  variants and cache results as JSON. `matmul_auto()`, `attention_auto()`
  dispatch to the best cached variant with deterministic fallback.
- **Windows CI**: GitHub Actions matrix (Ubuntu + Windows). Doc build job.
- **DSL friction report**: 5 documented friction points from attention
  implementation (no `&&`/`||`, 1D grid inference, no `sqrt()`, no compound
  shared assign, no `-inf` literal).
- 24 attention GPU tests + 7 tuner tests.

### Fixed — Phase 5
- 2D block reductions: previously rejected at compile time, now work via
  linear thread identity.

### Changed — Phase 5
- `kaio-ops` now depends on `serde` + `serde_json` (for tuner cache).

## [0.0.4] — Phase 4: Tiled MatMul & Block-Level API

### Added — Phase 4
- **`kaio-ops` crate** with `matmul()` — tiled matrix multiplication on GPU.
  Host-side API: `matmul(&device, &a, &b, &mut c, m, n, k)`.
- **FMA instruction**: `fma(a, b, c)` builtin for f32 fused multiply-add.
- **2D thread blocks**: `#[gpu_kernel(block_size = (16, 16))]` tuple syntax
  with 2D grid launch model (`grid: (u32, u32, u32)`).
- **2D thread indices**: `thread_idx_y()`, `block_idx_y()`, `block_dim_y()`,
  `grid_dim_y()`.
- **Multi-allocation shared memory**: multiple `shared_mem!` calls per kernel,
  each tracked independently via named PTX symbols.
- **Register-tiled matmul kernel**: 64x64 block tiles, 4x4 per thread,
  bank conflict padding (stride 17/65). 31% of cuBLAS sgemm on RTX 4090.
- **Benchmark harness** vs cuBLAS sgemm: deterministic inputs, 5 warm-up,
  20 measured, median TFLOPS. Results in `docs/benchmarks.md`.
- **`KAIO_PTX_STATS=1`**: instruction/register/shared-mem statistics at
  kernel compile time.
- **`KAIO_PTX_ANNOTATE=1`**: source-construct annotations in emitted PTX.
- **`docs/performance.md`**: coalescing patterns, bank conflict avoidance,
  register tiling guide, PTX inspection tool documentation.
- 207 host tests + 41 GPU tests + 1 benchmark.

### Fixed — Phase 4
- **Launch config block_dim mismatch**: cudarc's `for_num_elems()` hardcoded
  block_dim=1024 regardless of declared `block_size`. 1D launch wrapper now
  computes its own LaunchConfig.
- **Shared memory multi-allocation overlap**: `compute_shared_address()` assumed
  offset 0 for all allocations. Fixed with named-symbol base addressing
  (`Operand::SharedAddr`).
- **FMA builtin argument validation**: previously only validated first argument.
  Now all 3 args are checked for f32 type.
- **`block_reduce_*` in 2D kernels**: reductions derive thread identity from
  TidX only, which is wrong for 2D blocks. Now rejected at compile time with
  clear error message (fix planned for Phase 5).

## [0.0.3] — Phase 3 Complete

### Added — Phase 3: Loops, Reductions & Softmax
- **`for`/`while` loop support** in `#[gpu_kernel]`: `for i in start..end`
  with unsuffixed literal coercion (`0..n` where `n: u32`), `while condition`,
  compound assignment (`+=`, `-=`, `*=`, `/=`), loop variable scoping.
- **Shared memory** (`shared_mem![T; N]`): declare block-scoped SRAM buffers
  in kernel code, accessed via `sdata[idx]` with automatic `ld.shared`/
  `st.shared` dispatch. 32-bit addressing distinct from 64-bit global memory.
- **Barrier synchronization** (`bar_sync()`): `bar.sync 0` for block-level
  thread synchronization.
- **Warp shuffle** (`shfl_sync_down/up/bfly(val, delta, width)`): warp-level
  data exchange via `shfl.sync.{down,up,bfly}.b32`. Full-warp (width=32) only.
- **Block-level reductions** (`block_reduce_sum(val)`, `block_reduce_max(val)`):
  ~35-instruction expansion using warp shuffle tree + shared memory cross-warp
  reduction + broadcast to all threads. Auto-allocated `_kaio_reduce_smem`.
- **Softmax kernel** validated on RTX 4090: row-wise numerically-stable softmax
  using while loops, `block_reduce_max`, `block_reduce_sum`, `exp()`. 10 GPU
  tests including accuracy suite (< 1e-5 error vs CPU reference), edge cases
  (all-zeros, uniform, large values, negative, mixed-sign).
- **`Operand::SharedAddr`** in kaio-core for loading named shared allocation
  base addresses.
- **`SharedDecl`** struct + `PtxKernel::shared_decls` for shared memory
  declarations in PTX kernel preamble.
- **6 new kaio-core instruction variants**: `LdShared`, `StShared` (MemoryOp),
  `BarSync`, `ShflSyncDown`, `ShflSyncUp`, `ShflSyncBfly` (ControlOp).
- `KAIO_SM_TARGET` environment variable: set PTX target (default `sm_70`
  for maximum GPU compatibility, override with e.g. `KAIO_SM_TARGET=sm_89`).
- 200 host tests + 24 GPU tests across workspace.

### Fixed — Phase 3
- **Cvt rounding modifier**: `cvt.f32.u32` now correctly emits `cvt.rn.f32.u32`
  (round to nearest for int→float), `cvt.rzi.u32.f32` (round toward zero for
  float→int, matching Rust `as` semantics). Previously rejected by ptxas.
- **Let-binding register aliasing**: `let mut i = tid` now copies to a fresh
  register instead of aliasing. Prevents mutation of source variable when the
  new binding is reassigned (e.g., `i += stride` no longer corrupts `tid`).
- **Variable shadowing**: `let mut i = tid;` can now be used multiple times in
  the same kernel (e.g., once per while loop). Previously errored with
  "variable already defined".

### Changed — Phase 3
- SM target default changed from hardcoded `sm_89` to configurable via
  `KAIO_SM_TARGET` env var (default `sm_70`).

### Added — Phase 2: Proc Macro DSL
- **`#[gpu_kernel]` proc macro** (`kaio-macros`): full pipeline from Rust function
  syntax to executable PTX. Parse (syn → KernelIR) → lower (IR → PTX instructions)
  → codegen (build_ptx with OnceLock + typed launch wrapper).
- **Supported Rust subset**: arithmetic (`+`, `-`, `*`, `/`, `%`, `+=`, `-=`,
  `*=`, `/=`), comparisons (`<`, `<=`, `>`, `>=`, `==`, `!=`), `if`/`else` with
  `@!pred` predicated branches, array indexing with cvta.to.global caching, `let`
  bindings with type inference, type casts via `as`.
- **19 built-in functions**: thread/block/grid intrinsics (`thread_idx_x`,
  `block_idx_x`, `block_dim_x`, `grid_dim_x`), math functions (`sqrt`, `rsqrt`,
  `abs`, `min`, `max`, `sin`, `cos`), synthesized transcendentals (`exp`, `log`,
  `tanh` via PTX `ex2`/`lg2`/`rcp` primitives).
- **Launch wrapper generation**: typed function signatures matching kernel params,
  automatic grid/block calculation, argument marshaling via `PushKernelArg`.
- **`KAIO_DUMP_PTX`** environment variable: set to `1` to write `.ptx` files to
  disk during compilation for inspection.
- **4 E2E kernels validated on RTX 4090**: `vector_add`, `saxpy`, `fused_relu`,
  `fused_gelu` — all produce correct results against CPU reference implementations.
- **10 trybuild compile-fail tests** (CF1–CF10): unsupported types, non-unit
  return, macro invocations, heap allocation, non-built-in function calls, missing
  `block_size`, non-power-of-2 `block_size`, `block_size` > 1024, lifetime
  parameters, `loop` keyword.
- **`kaio` umbrella crate**: prelude module re-exporting macro + runtime, 
  `gpu_builtins` IDE stub module for autocomplete.
- 168 host tests + 5 GPU tests across workspace

### Fixed — Phase 2
- **cvta.to.global register scope across if/else branches**: conversion registers
  were scoped inside branches, causing undefined-register errors in subsequent
  code. Fix: eager emission during parameter loading, before any control flow.
- **PTX float division requires `.approx`/`.rn` modifier**: bare `div.f32` is
  invalid PTX. Fix: type-aware modifier selection in `ArithOp::Div` emission
  (`.approx` for f32, `.rn` for f64).

### Changed — Phase 2
- Renamed project from PYROS to KAIO across entire codebase (commit `50a3ab0`)

### Added — Phase 1: PTX Foundation
- **PTX code generation** (`kaio-core`): IR types modelling complete PTX programs,
  instruction emitters for arithmetic (add, mad, mul.wide), memory (ld.param,
  ld.global, st.global, cvta.to.global), and control flow (setp, bra, ret).
  Emit trait + PtxWriter produce valid PTX text from an IR tree.
- **CUDA runtime wrapper** (`kaio-runtime`): KaioDevice for GPU context management,
  GpuBuffer<T> for typed device memory, KaioModule/KaioFunction for PTX loading
  and kernel launch via cudarc 0.19.
- **End-to-end `vector_add`**: kernel constructed via Rust IR, emitted to PTX,
  loaded into the CUDA driver, launched on RTX 4090 — produces correct results
  for both single-block (3 elements) and multi-block (10,000 elements).
- **Validation**: all PTX instruction emitters verified byte-for-byte against
  nvcc 12.8 output. ptxas offline verification passes. cudarc smoke test confirms
  host↔device data transfer.
- Virtual workspace with umbrella `kaio` crate re-exporting `kaio-core` + `kaio-runtime`
- 53 host-side tests + 9 GPU-gated tests, 82.8% line coverage
- Per-sprint architectural decision records in `docs/development/sprints/`

### Changed — Phase 1
- PTX ISA version corrected from 7.8 to 8.7 (CUDA 12.8)
- Register declarations use `.b32`/`.b64` (untyped) matching nvcc convention

## [0.0.1] — 2026-04-10

### Added
- Name reservation crate with metadata, README, dual MIT/Apache-2.0 license
- Project design docs: index.md, implementation.md, phases.md, success-criteria.md
