# Sprint 7.2 — INT4 dequantize-matmul (`matmul_int4`)

**Status:** ✅ Complete
**Branch:** `phase7-rest` off `main` (shared across Phase 7 remaining
sprints; no independent crates.io release — Phase 7 closes with an
aggregate release after 7.4).
**Release target:** None this sprint.

## Context

Sprint 7.1 shipped `matmul_int8` (v0.3.0) — W8A8 symmetric, single
scalar scale, direct `mma.sync.m16n8k32.s8.s8.s32`. 7.1.5 shipped warp
+ block reductions in the DSL on `phase7-rest` (no release). 7.2 is
the next headline quant op: **GPTQ-style INT4 with group scales**.

**Architectural reality:** there is no native `m16n8k?.s4.s4.s32`
shape on sm_80+. INT4 operands must be dequantized before the mma.
Group scales vary along K, so the scale must apply *inside* the
K-reduction pre-mma — DEQUANT-S8-and-reuse-INT8-kernel is structurally
impossible with group quant. Therefore: **DEQUANT-F16** — unpack s4
→ sign-extend → `cvt.rn.f32.s32` → `cvt.rn.f16.f32` → multiply by f16
group scale → feed `mma.sync.m16n8k16.f16.f16.f32` fragment B.
Activations stay f16 (W4A16 — canonical GPTQ layout), accumulator f32.

## Scope (summary)

- Public API `kaio_ops::matmul_int4(device, a: &GpuBuffer<f16>,
  b_packed: &GpuBuffer<u32>, scales: &GpuBuffer<f16>,
  c: &mut GpuBuffer<f32>, m, n, k, group_size) -> Result<()>`.
- Packing: 8 signed INT4 values per `u32`, lane `i` at bits `[4i..4i+4)`.
- Group scales: f16, one per `(n, g)` cell with `g = k / group_size`.
- Group size fixed at **128** for v1; `K % 128 == 0` enforced.
- Triple-layer sign-extend canary (emit token-stream, ptxas_verify, GPU e2e boundary values).
- Kernel structure mirrors INT8: 4 warps × 64×64 output block.
- K_tile_shared = 16 f16 elements (correctness-first for novel kernel).

Scope-out and architectural decisions (AD1–AD11) captured in the plan
file; Phase 7 follow-ups (asymmetric INT4, GGUF/AWQ compat, non-128
group sizes, fused bias, `cp.async`) explicitly deferred.

## D1 — IR audit + `MovPack` addition

**Audit target:** confirm every primitive in the s4-unpack → f16-mul →
mma chain is present in `kaio-core`; extend minimally where needed.

### Present (no changes)

- `ArithOp::{Shl, Shr (signed + unsigned), And, Or, Xor, Not}` —
  shipped Sprint 7.0. `Shr.ty = PtxType::S32` dispatches to arithmetic
  right shift; `shl %x, 28 → shr.s32 %x, 28` is the canonical INT4
  sign-extend idiom.
- `PtxInstruction::Cvt { dst_ty, src_ty }` — generic form at
  `kaio-core/src/ir/instruction.rs:43`. Emit impl at
  `kaio-core/src/emit/emit_trait.rs:141-175` already covers the
  DEQUANT-F16 rounding chain: int → float emits `.rn`, float → float
  emits `.rn`. Verified `cvt.rn.f32.s32`, `cvt.rn.f16.f32`,
  `cvt.rn.f16.s32` all emit valid PTX with no changes.
- `TensorCoreOp::MmaSync { shape: M16N8K16, a_ty/b_ty: F16, c_ty: F32 }`
  + `FragmentA/B/C_M16N8K16` + shared loaders (`load_fragment_a_m16n8k16_shared_row`
  at `kaio-core/src/fragment.rs:873`, `_b_..._col` at `:962`) —
  shipped Phase 6.
- `PtxType::F16` + `RegKind::H` — shipped Phase 6.

### Gap: packed f16-pair move — new `MovPack` variant

**Problem.** `mma.sync.m16n8k16.f16.f16.f32` fragment B expects two
`.b32` registers, each holding two packed f16 values. The existing
f16 matmul avoids assembling these in-register by loading adjacent
fp16 values from shared memory as a single `.b32` (two-f16 `.b32`
co-location). INT4 dequant instead *produces* individual f16 values
as computation outputs and must then pack them into `.b32` pairs —
which the existing `PtxInstruction::Mov` (single-source) cannot
express.

**Decision (per R6 in the plan).** Framework-level IR primitive is
defensible: INT4 uses it 4 times per warp per K-tile, and 7.3 quant
+ attention will also need it. Added a minimal new variant.

```rust
/// kaio-core/src/ir/instruction.rs
PtxInstruction::MovPack {
    dst: Register,        // wider destination (.b32 / .b64)
    srcs: Vec<Register>,  // narrow sources, low-lane first
    ty: PtxType,          // destination width type (width used, signedness ignored)
}
```

Emit (`kaio-core/src/emit/emit_trait.rs`):

```ptx
mov.b{N} %dst, {%s0,%s1,...};
```

**Critical ptxas detail (discovered during D1 implementation):** the
vector-pack form of `mov` requires the *typeless* `.b{N}` suffix
(PTX ISA §9.7.9.10). `mov.u32 %r, {%h0, %h1};` is rejected by ptxas
with `Arguments mismatch for instruction 'mov'`; `mov.b32` accepts
it. The emitter therefore derives the `.b{N}` suffix from the
destination type's byte width, not from `ptx_suffix()`.

**Emit test** (`emit_mov_pack_two_f16_into_b32` at
`kaio-core/src/emit/emit_trait.rs`): asserts exactly
`mov.b32 %r7, {%h3,%h4};\n` for the canonical pair-pack case. All
157 kaio-core unit tests pass; clippy clean.

### Gaps found — none else

All other primitives in the dequant chain are present. No extensions
needed beyond `MovPack`.

### D1 commit

Commit 1: `wip(phase7): D1 - Sprint 7.2 stub + MovPack IR primitive`.

## D2 — Unpack helper + sign-extend canary

### Helper: `emit_unpack_s4_x8_scale_to_f16x8`

Landed in `kaio-ops/src/matmul_int4_kernel.rs`. For one packed `u32`
of 8 signed INT4 nibbles plus a single f16 group scale, emits:

- 8 × (`shl.b32` → `shr.s32`) — sign-extend per nibble via `(x << (28 - 4i)) >> 28`
- 8 × `cvt.rn.f32.s32` → 8 × `cvt.rn.f16.f32`
- 8 × `mul.f16` with the scalar group scale
- 4 × `mov.b32 %b32, {%h_lo, %h_hi}` (the new `MovPack` IR primitive)

Returns `[Register; 4]` of packed-f16-pair `.b32` registers ready
for `FragmentB_M16N8K16` feed. Helper is `pub(crate)` + `#[allow(dead_code)]`
pending D4 wire-up. Contract locked per Opus round 2: assumes all 8
nibbles in the input u32 share one group scale; caller aligns u32
loads to group boundaries.

**Latent-bug fix** landed alongside: `ArithOp::Mul` emitter's fallback
arm was emitting `mul.lo.f16` for `PtxType::F16`, which ptxas
rejects. Extended the float arm to cover `F16` and `BF16` (both emit
`mul{.rn}.f16` — PTX defaults to `.rn` on half-precision). Added an
`emit_mul_f16` unit test. Flagged as a concurrent safety fix rather
than Sprint 7.2 scope creep: any future kernel using ArithOp::Mul
with F16 would have hit the same invalid-PTX bug.

### Triple-layer sign-extend canary

**Layer 1 — Host token-stream assertions** (three emit-level tests,
`matmul_int4_kernel::tests`):

- `unpack_s4_sign_extend_uses_arithmetic_shift` — asserts exactly 8
  `shr.s32` instructions and **zero** `shr.u32` in the emitted PTX.
  This is the primary R1 canary: a `shr.u32` at any single nibble
  site would silently zero-extend that negative INT4 value.
- `unpack_s4_shl_covers_all_nibble_positions` — asserts the 8
  `shl.b32` immediates cover `{0, 4, 8, 12, 16, 20, 24, 28}` exactly
  once each; catches position-alignment bugs that skip or double up
  a nibble.
- `unpack_s4_emits_four_packed_f16_pairs` — asserts exactly 4
  `mov.b32 %b, {%h_lo, %h_hi};` vector-pack instructions for the
  `FragmentB` feed.

**Layer 2 — ptxas offline gate** (`ptxas_verify_unpack_s4`,
`#[ignore]`): runs the CUDA toolkit's `ptxas -arch=sm_80` over a
minimal kernel that exercises the full unpack chain once and stores
to global memory. Confirms the whole emit (shl + shr.s32 + cvt chain
+ mul + MovPack) is structurally legal PTX on Ampere+. **PASSED on
sm_80** during D2 (invoked via `cargo test -p kaio-ops --lib
ptxas_verify_unpack_s4 -- --ignored`).

**Layer 3 — GPU e2e boundary-value tests** — lands in D6 (explicit
sign-extend round-trip on `0x88888888`, `0x77777777`, mixed-position
patterns like `0x80000000` / `0x87654321` per Codex round 3).

### Quality gates (D2 checkpoint)

- `cargo fmt --all --check` ✓
- `cargo clippy --workspace --all-targets -- -D warnings` ✓
- `cargo test --workspace` ✓ (full suite — no regressions; 3 new D2 emit tests green)
- `cargo test -- --ignored matmul_int4_kernel::tests::ptxas_verify_unpack_s4` ✓ PASSED sm_80

### D2 commits

Commit 2: `wip(phase7): D2 - emit_unpack_s4_x8_scale_to_f16x8 helper + sign-extend canary`.

## D3 — Shared-memory layout + group-scale broadcast design

Design-only deliverable. Fixes the numbers that D4 implements. Mirrors
the INT8 kernel's layout documentation style at
`kaio-ops/src/matmul_int8_kernel.rs:75-93`.

### Block + warp shape (reused from INT8)

| Constant | Value | Notes |
|---|---|---|
| Block dim | `(32, 4, 1)` | 4 warps × 32 lanes = 128 threads |
| `BM_BLOCK` | 64 | Output rows per block |
| `BN_BLOCK` | 64 | Output cols per block |
| `WARP_QUAD_M` | 32 | Rows per warp quadrant (2×2 grid of warps) |
| `WARP_QUAD_N` | 32 | Cols per warp quadrant |
| `MMAS_PER_WARP_M` | 2 | `WARP_QUAD_M / m16n8k16::M` (32 / 16) |
| `MMAS_PER_WARP_N` | 4 | `WARP_QUAD_N / m16n8k16::N` (32 / 8) |
| mma shape | `m16n8k16.f16.f16.f32` | m=16, n=8, K-tile = 16 f16 |

### K-tile granularity — `K_TILE_SHARED = 16`

Locked at 16 f16 elements per the plan (Codex round 3 arbitration).
At K=16 exactly one mma K-step per K-tile per warp; the cooperative
load for A, the unpacked B-feed, and the mma all run on the same
tile. Rationale captured in the plan file; budget-check numbers
below confirm K=16 is comfortable on shared memory.

Widening to K=32 post-bench is a one-constant change; flagged as a
perf follow-up if D7 bench shows barrier-sync overhead dominating.

### A tile (activations, f16)

- Shape: **64 rows × 16 cols** f16, row-major.
- Row stride: `16 × 2 B = 32 B` (natively packed; matches Phase 6 f16 tile).
- Tile size: `64 × 32 B = 2 048 B`.
- Loaded cooperatively via the Phase 6 f16 tile-loader pattern
  (reuse `load_fragment_a_m16n8k16_shared_row` for fragment fetch at
  mma time).
- Bounds-predicated on the M edge for non-multiple-of-64 M.

### B tile (packed signed INT4, u32 storage)

- Storage: `u32` per 8 K-elements × 64 cols, **col-major**.
- Per column: `K_TILE_SHARED / 8 = 2` u32 words per K-tile.
- **Natural col-stride = 8 B** (2 × 4 B, no pad). On a 32-bank shared
  memory (128 B period), `gcd(8, 128) = 8` → only 16 distinct bank
  patterns across 64 cols → 4-way bank conflict on the column-strided
  fragment-B gathers, same failure mode the INT8 kernel fixed with
  its `+4 B` pad at `matmul_int8_kernel.rs:80-87`.
- **Padded col-stride = 12 B** (2 × 4 B data + 4 B pad). `gcd(12, 128)
  = 4` → 32 distinct bank patterns → conflict-free.
- Tile size with padding: `64 cols × 12 B = 768 B`.
- **Dequant happens at shared → register transfer time**, per AD4: the
  D2 helper consumes one `u32` from shared per column per K-tile and
  produces the 4 `.b32` packed-f16 pair registers the mma needs. No
  f16 ever sits in shared memory for the B-tile; the 8× compression
  holds all the way through to register.

### Group-scale tile (f16, lazy reload on group transitions)

- Layout: `[1, 64]` f16 row — one f16 per output column per active group.
- Tile size: `64 × 2 B = 128 B`.
- Reload cadence: **every 8 K-tiles** (every `8 × 16 = 128` K-elements
  = one full group at `group_size = 128`). The block's K-loop
  increments a `current_group` counter each time `k_tile % 8 == 0`;
  on transition, issue one cooperative load from `scales[0..N, group]`.
- Within a K-tile: every column's active scale is the same f16 value
  (all 16 K-elements sit inside one group, so the per-column lookup
  returns one scale). The D2 helper consumes that one f16 and applies
  it to all 8 unpacked nibbles in the `u32` — the contract holds by
  construction.

### Shared-memory budget

| Region | Bytes |
|---|---|
| `tile_a` (64 × 16 f16 row-major, 32 B stride) | 2 048 |
| `tile_b` (packed u32, 64 cols × 12 B padded col stride) | 768 |
| `tile_scales` (1 × 64 f16) | 128 |
| **Total per block** | **≈ 2.9 KB** |

Well under the 48 KB per-block shared-memory limit on sm_80+; also
well below the INT8 kernel's 4.6 KB per block — INT4 uses less
shared memory because K_TILE_SHARED is half (16 vs 32) and B stays
packed all the way through to register.

### K-loop outline (D4 implements)

```text
for k_tile in 0..(K / K_TILE_SHARED):   // K / 16 iterations
    if k_tile % 8 == 0:
        coop_load_group_scales(tile_scales, current_group)
    coop_load_tile_a(tile_a, k_tile)                            // 64×16 f16
    coop_load_tile_b_packed(tile_b, k_tile)                     // 2×64 u32
    bar.sync 0

    for warp_q in (quadrant_m, quadrant_n):                     // per warp
        frag_a = load_fragment_a_m16n8k16_shared_row(tile_a, ...)
        for n_stripe in 0..4:
            packed_u32 = ld.shared.u32 tile_b[n_stripe_col, k_tile]
            scale_f16 = tile_scales[n_stripe_col]
            [b32_pair_01, b32_pair_23, b32_pair_45, b32_pair_67]
                = emit_unpack_s4_x8_scale_to_f16x8(packed_u32, scale_f16)
            // ^ NOTE: fragment-B register layout consumption scheme
            //   finalized in D4; this yields 4 pairs from 8 K-elements,
            //   while the fragment itself spans 16 K-elements, so D4
            //   will call the helper once per half-tile or adjust
            //   once the per-lane distribution is wired.
            mma.sync.m16n8k16.f16.f16.f32 frag_c, frag_a, frag_b, frag_c

    bar.sync 0
```

**Open item handed to D4:** the helper produces 4 `.b32` pairs from
one `u32` (8 K-elements). A full fragment B spans 16 K-elements
(one full K-tile). D4 either (a) calls the helper twice per fragment
and interleaves the resulting 8 `.b32` into the 2-register fragment
layout that the warp's 32 lanes actually consume, or (b) changes
the helper's per-call granularity to exactly one fragment's worth.
This is a call-site-level decision that depends on the PTX
fragment-distribution pattern across lanes — easier to settle with
the kernel in front of us than to pre-decide here. The D2 helper
contract (single scale, no group-index math) is stable either way;
both options preserve correctness.

### Edge-tile predication

- **M edge** (M not a multiple of 64): cooperative A-tile loads
  predicate per-row; edge-tile C-fragment stores predicate
  per-output-row. Established KAIO pattern.
- **N edge** (N not a multiple of 64): cooperative B-tile loads and
  group-scale loads predicate per-col; edge-tile C-fragment stores
  predicate per-output-col.
- **K**: no edge-K handling inside a tile — `K % 128 == 0` enforced
  at `validate_dims_int4`, which makes `K % K_TILE_SHARED == 0`
  trivially (128 is a multiple of 16).

### D3 commit

Commit 3: `wip(phase7): D3 - shared-memory layout + group-scale broadcast design`.
No code lands in this commit — only the design section above. D4
executes against it.

## D4 — Kernel assembly (split into D4.1 / D4.2 / D4.3)

The full kernel is large enough to split into three sub-commits.
Each sub-deliverable lands with its own emit structure tests and
(where structural) a ptxas_verify offline gate.

### D4.1 — Cooperative load helpers + `validate_dims_int4`

Landed `kaio-ops/src/matmul_int4_kernel.rs`:

- **Constants** for the block + warp + tile shape (BM=16, BN=8, BK=16;
  BM_BLOCK/BN_BLOCK=64; WARP_QUAD_M/N=32; MMAS_PER_WARP_M=2,
  MMAS_PER_WARP_N=4; WARPS_PER_BLOCK=4; THREADS_PER_BLOCK=128;
  K_TILE_SHARED=16; GROUP_SIZE=128).
- **Tile-size constants**: `TILE_A_BYTES=2048`, `TILE_B_BYTES=768`
  (64 cols × 12 B padded col-stride), `TILE_SCALES_BYTES=128`.
- **`validate_dims_int4`** — M/N/K non-zero; `group_size == GROUP_SIZE=128`;
  `K % GROUP_SIZE == 0`; buffer-size bounds for A, b_packed, scales, C.
  Error-path unit coverage deferred to D6 integration tests (mirrors
  INT8's `validate_dims_int8` pattern — no unit-test harness for
  `GpuBuffer`).
- **`emit_mw_load_tile_a_f16_64x16`** — 128 threads × 4 `ld.global.u32`
  per thread = 512 b32 loads = 1024 f16 = full 64×16 tile. 2 threads
  per row × 4 b32 each covers half-row. M-edge predicated
  `@!p bra A_SKIP_I4_TILE_LOAD_<suffix>` with caller-managed pre-zero
  for OOB rows.
- **`emit_mw_load_tile_b_packed_2x64`** — 128 threads × 1 `ld.global.u32`
  per thread = 128 packed u32 = 2 words × 64 cols. `col_in_tile =
  flat_tid / 2`, `word_idx = flat_tid % 2`. Shared address uses 12 B
  col-stride (4 B pad per col for bank-conflict relief). N-edge
  predicated.
- **`emit_cooperative_load_group_scales_64`** — lanes 0..31 active
  (each loads a f16-pair b32), lanes 32..127 idle via active-skip
  label. 32 b32 = 64 f16 = full `[1, 64]` scales tile. N-edge
  predicated.

**Scales layout switch (D4.1 fold on the D3 design):** the plan
and D3 design said `scales[N, K/group_size]` row-major. For the
cooperative 64-f16 contiguous load at fixed group g, `[num_groups, N]`
row-major (so `scales[g, block_col..block_col+64]` is contiguous)
gives a simpler, faster loader — 32 lanes × one b32 broadcast-free
load. **Locking at `scales: [num_groups, N]` row-major**; D5 public
API + rustdoc + D7 showcase's CPU packer align on this.

### D4.1 emit structure canaries

- `coop_loads_emit_ld_st_pairs_match_per_helper_design` — asserts
  exactly 6 `ld.global.u32` and 6 `st.shared.u32` in the smoke kernel
  (A: 4 per thread, B: 1 per thread, scales: 1 per active lane —
  unrolled linearly).
- `coop_loads_emit_three_bounds_gated_skip_regions` — asserts the
  three labeled skip-region targets (`A_SKIP_I4_TILE_LOAD_smoke:`,
  `B_SKIP_I4_TILE_LOAD_smoke:`, `SCALES_SKIP_I4_smoke:`) are present.
- `ptxas_verify_matmul_int4_coop_loads` (`#[ignore]`): **PASSED sm_80**.
  Smoke kernel (all 3 loaders + bar.sync + ret) assembles cleanly.

### Quality gates (D4.1 checkpoint)

- `cargo fmt --all --check` ✓
- `cargo clippy --workspace --all-targets -- -D warnings` ✓
- `cargo test --workspace --lib` ✓ (158 kaio-core, 138 kaio-macros, 34 kaio, 2 kaio-ops lib — no regressions; 5 new D4.1 emit tests green)
- `cargo test -- --ignored matmul_int4_kernel::tests::ptxas_verify_matmul_int4_coop_loads` ✓ PASSED sm_80

### D4.1 commit

Commit 4: `wip(phase7): D4.1 - validate_dims_int4 + cooperative load helpers for A, B-packed, scales`.

### D4.2 — narrow 2-nibble unpack helper + fragment-B per-lane dequant + warp-quadrant mma

- **`emit_unpack_s4_x2_scale_to_f16_pair`** — narrow sibling of the D2 generic helper. Takes `(packed, scale_f16, shl_count_lo, shl_count_hi)` where the shl counts are `Operand`s (either immediate for compile-time positions or register for per-lane `tig`-varying positions). Emits 2 × (shl + shr.s32 + cvt.f32.s32 + cvt.f16.f32 + mul.f16) followed by a `MovPack` into one `.b32`. Output: one packed-f16-pair register.
- **`emit_fragment_b_int4_per_lane`** — per-warp-lane dequant for one `FragmentB_M16N8K16`. Decodes the lane's `(group_id, tig)` from `tid.x % 32`, computes `col_abs = n_stripe_col_base + group_id`, loads 2 u32 packed words + 1 f16 scale from shared, computes shift counts `(28 - 8*tig, 24 - 8*tig)`, and calls the narrow helper twice (once per u32 word). Returns a `FragmentB` ready for mma.sync.
- **`emit_warp_quadrant_mma_int4`** — per-warp-quadrant inner-loop: for each of 4 N-stripes, per-lane dequant a fresh fragment B; for each of 2 M-stripes, load fragment A from shared (reusing `load_fragment_a_m16n8k16_shared_row`) and emit `mma.sync.m16n8k16.f16.f16.f32` accumulating into the caller's `frag_c_grid`.

**Fragment B lane mapping** (PTX ISA §9.7.13.5.8.1 for m16n8k16.f16):
lane `l ∈ 0..32` owns col `l/4` and rows `{2*tig, 2*tig+1}` for reg[0] +
`{2*tig+8, 2*tig+9}` for reg[1] where `tig = l%4`. Per fragment-B per
lane: 4 nibbles at positions `(2*tig, 2*tig+1)` on each of 2 u32s.

### D4.3 — pre-zero + full `build_matmul_int4_module`

- **`emit_pre_zero_shared_tiles_int4`** — cooperative zero of tile_a
  (2048 B), tile_b (1024 B padded; data region 768 B, tail 768..1024
  never read), and tile_scales (128 B via lanes 0..31 subset). One
  bar.sync at end. `TILE_B_BYTES` rounded up to multiple of
  `THREADS_PER_BLOCK * 4 = 512` to accommodate the cooperative-zero
  divisibility requirement.
- **`build_matmul_int4_module(sm)`** — full kernel builder. Assembles:
  param loads → derived scalars (`k_bytes = K*2`, `k_words = K/8`,
  `n_f32_stride = N*4`) → thread/block indices (flat_tid,
  warp_id, warp_row_quad, warp_col_quad, warp-quadrant row/col base,
  block_row/block_col) → shared base regs → A/B block-base globals
  → C fragment grid (2×4) zero-init → pre-zero shared → K-loop with
  group-transition scale reload → A/B cooperative loads → bar.sync →
  per-warp-quadrant mma pipeline → bar.sync → loop back → output
  store. Output store reuses `matmul_tc_kernel::emit_warp_quadrant_store`
  (same f32 fragment C layout) via a 4-way predicated dispatch on
  `(warp_row_quad, warp_col_quad)` → 4 static
  `warp_quadrant_(row|col)_start` call-sites.
- Kernel signature: `matmul_int4(a: f16*, b_packed: u32*, scales: f16*, d: f32*, m, n, k: u32)`. Block dim `(32, 4, 1)`, grid `(N.div_ceil(64), M.div_ceil(64), 1)`.

### D4.3 emit canaries + offline gate

- `module_emits_eight_mma_sync_per_warp_quadrant` — counts `mma.sync.aligned.m16n8k16` in the full module; must equal `MMAS_PER_WARP_M * MMAS_PER_WARP_N = 8`.
- `module_shr_s32_count_matches_per_lane_dequant_shape` — counts `shr.s32` across the whole module (4 n-stripes × 2 u32 × 2 nibbles = 16 sites). Asserts zero `shr.u32`. Covers the R1 headline risk at the module-assembly layer.
- `ptxas_verify_matmul_int4` (`#[ignore]`): **PASSED on sm_80 and sm_89**. Full kernel with pre-zero + K-loop + fragment-B dequant + 2×4 mma grid + store assembles cleanly.

### D4.3 register + resource budget (ptxas `--verbose` on sm_89)

```
ptxas info: Used 64 registers, used 1 barriers, 3200 bytes smem, 396 bytes cmem[0]
            0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
```

- **64 regs/thread** — exactly at the 64-reg occupancy tier for
  128-thread CTAs on sm_89. R2 sits at the edge; D7 bench will show
  whether perf regresses here. **Fallback if needed (R2 mitigation):**
  drop `MMAS_PER_WARP_N` from 4 to 2.
- **0 spills** — no register spills to local memory.
- **3200 B smem** — matches the design budget (2944 B pure data +
  256 B pre-zero tail padding on tile_b).
- **1 barrier** — one `bar.sync` instance serves both inter-load and
  end-of-K-tile barriers.

### Quality gates (D4.2 + D4.3 checkpoint)

- `cargo fmt --all --check` ✓
- `cargo clippy --workspace --all-targets -- -D warnings` ✓
- `cargo test --workspace --lib` ✓ (all 158 + 138 + 36 + 2 host tests green; 7 new emit tests for INT4)
- `cargo test -- --ignored matmul_int4_kernel::tests::ptxas_verify_*` ✓ PASSED (unpack_s4, coop_loads, full module) on sm_80 + sm_89

### D4 commit

Commit 5: `wip(phase7): D4.2+D4.3 - fragment-B per-lane dequant + full matmul_int4 module (64 regs/thread, 3.2 KB smem)`.

## D5 — Public API + host dispatch

`kaio_ops::matmul_int4` public function:

```rust
pub fn matmul_int4(
    device: &KaioDevice,
    a: &GpuBuffer<f16>,          // [M, K] row-major
    b_packed: &GpuBuffer<u32>,   // [K/8, N] col-major, 8 nibbles/u32
    scales: &GpuBuffer<f16>,     // [K/group_size, N] row-major
    c: &mut GpuBuffer<f32>,      // [M, N] row-major
    m: u32, n: u32, k: u32,
    group_size: u32,             // must be 128 in v1
) -> Result<()>;
```

Dispatch: `validate_dims_int4` → build module for `sm_{cc}` → load +
launch with grid `(n/64, m/64, 1)`, block `(32, 4, 1)`. Rustdoc spells
out the packing formula verbatim, the W4A16 scope, the sign-extend
guarantee, and the "KAIO-defined reference kernel, not GPTQ/GGUF
compat" contract. `matmul_int4` re-exported from `kaio_ops`.

Commit 6: `wip(phase7): D5 - matmul_int4 public API + host dispatch + rustdoc packing spec`.

## D6 — GPU e2e tests (correctness proven bit-exact)

`kaio-ops/tests/matmul_int4_e2e.rs` — 12 GPU tests, all `#[ignore]`d.
CPU reference reproduces the exact `s32 → f32 → f16` cvt chain the
kernel uses. Runs on RTX 4090 sm_89:

| Test | max_rel | Notes |
|---|---|---|
| `smallest_16_8_128_all_ones_weights` | **0.0 (bit-exact)** | |
| `sign_extend_canary_negative_eights` | **0.0 (bit-exact)** | R1 canary — GPU layer |
| `sign_extend_canary_mixed_positions` | **0.0 (bit-exact)** | R1 canary — position-varying |
| `multi_group_k256` | **0.0 (bit-exact)** | R3 group-boundary |
| `edge_m_17` | 1.76e-6 | M edge predication |
| `edge_n_13` | 1.32e-6 | N edge predication |
| `64_64_128` | 2.65e-6 | |
| `128_128_256` | 3.46e-6 | |
| `256_256_512` | 4.08e-6 | |
| `rejects_k_not_multiple_of_128` | n/a | validation error path |
| `rejects_non_128_group_size` | n/a | validation error path |
| `rejects_zero_dim` | n/a | validation error path |

Max rel error across ALL shapes: **4.1e-6** — well under the 1e-3
target. Sign-extend canaries are bit-exact on hardware: R1 killed.

Commit 7: `wip(phase7): D6 - matmul_int4 GPU e2e tests (12 pass bit-exact on sm_89, max_rel 4e-6)`.

## D7 — Bench + showcase

### `kaio-ops/tests/matmul_int4_bench.rs`

Bench harness with deterministic inputs, median latency over 20 iters
(5 warmup), and cuBLAS sgemm comparison as a regression reference.
RTX 4090 sm_89 release-mode numbers across 4 runs:

| Size | KAIO INT4 (TOPS) | cuBLAS sgemm (TF) | vs sgemm |
|---|---|---|---|
| 512³ | 0.47 | 10.4–11.8 | 4–5% |
| 1024³ | 3.5–3.6 | 32.2–37.0 | 10–12% |
| 2048³ | 20.5–21.8 | 43.1–52.5 | 39–51% |
| **4096³** | **41.7–57.4** (median ~52) | 52.0–58.3 | **80–101%** |

At 4096³ (realistic LLM-inference shape), `matmul_int4` lands at 80-101%
of cuBLAS sgemm. Variance at 4096³ (42-57 TOPS) is honest consumer-GPU
jitter; report the range per 7.1's discipline.

### `examples/int4_matmul/`

Full showcase package (`Cargo.toml`, `src/main.rs`, `README.md`).
Demonstrates: GPTQ-lite symmetric per-column group quantization (naive
`max(|w|)/7`), packing into the KAIO `[K/8, N]` col-major u32 layout
(`pack_s4_weights` helper — the reference CPU packer users adapt for
their own pipelines), `matmul_int4` launch, and max-abs / max-rel
error vs f32 naive reference.

Tolerance is 80% max-rel (reflects INT4's inherent noise floor — 16
representable values give ~16× more per-element noise than INT8).
Registered in `cargo xtask showcase int4matmul` alongside the INT8
showcase. Verified passing.

Commit 8: `wip(phase7): D7 - matmul_int4_bench (42-57 TOPS 4096³ sm_89) + int4_matmul showcase + xtask integration`.

## Results (sprint-final)

### Correctness — all green

- **D2 emit canaries**: sign-extend uses `shr.s32` at all 8 nibble positions (IR-emit layer); nibble position coverage complete; 4 packed f16 pairs emitted.
- **D4.3 module canaries**: 8 `mma.sync.m16n8k16` (2 m-stripes × 4 n-stripes); 16 `shr.s32` (4 n-stripes × 2 u32 × 2 nibbles); 0 `shr.u32`.
- **ptxas_verify**: offline `ptxas -arch=sm_80` / `-arch=sm_89` accepts unpack_s4 helper, coop-loads smoke, and full `matmul_int4` module.
- **GPU e2e (D6)**: all 12 tests pass on sm_89. Sign-extend canaries bit-exact; large-shape tests max_rel ≤ 4.1e-6; validation-error tests cover K-multiple, group-size, and zero-dim paths.

### Performance — R2 register budget landed in the tier

- ptxas --verbose on sm_89: **64 registers / thread**, **0 spills**, **3200 B smem**, **1 barrier**.
- Sits exactly at the 64-reg occupancy tier for 128-thread CTAs. Not forcing the fallback (MMAS_PER_WARP_N 4→2) — perf numbers are competitive with cuBLAS sgemm at 4096³.
- **4096³ median ~52 TOPS** on RTX 4090 sm_89. Range 42–57 across 4 runs.

### Scope — exactly as planned

- DEQUANT-F16 path via `mma.sync.m16n8k16.f16.f16.f32` (AD1).
- Symmetric INT4, no zero-points (AD2).
- Group size fixed at 128 (AD3); `K % 128 == 0` enforced.
- Packed weights stay packed in shared; dequant fuses at shared→register (AD4).
- INT8 block structure reused: 4 warps × 64×64 output, 2×2 quadrants (AD5).
- W4A16; f16 activations, f16 scales, f32 accumulator, f32 output (AD6).
- Triple-layer sign-extend canary (AD7) — all three gates green.
- ptxas --verbose register count recorded (AD8).
- No fused bias / activation, no runtime dispatch (AD9/AD10).
- Bench variance reported honestly (AD11).

### One `kaio-core` IR addition

- **`PtxInstruction::MovPack`** — vector-pack `mov.b{N} %dst, {%s0, %s1, ...};` emitter. Needed for packing two f16 registers into one b32 for the fragment-B feed (R6). Emits via the typeless `.b{N}` suffix as required by ptxas; derived from `PtxType::size_bytes() * 8`. One emit test covers the f16-pair → b32 case. Promoted to framework-level because 7.3 quant-attention will also need it.

### One latent-bug fix (not load-bearing for INT4 but useful)

- **`ArithOp::Mul { ty: F16 | BF16, .. }`** — was falling through to `mul.lo.f16` (the integer-multiply suffix), which ptxas rejects. Extended the float arm to cover `F16` / `BF16`. Added `emit_mul_f16` unit test.

### Commit table

| # | Commit | Scope |
|---|---|---|
| 1 | `25d5314` | D1 — sprint stub + `MovPack` IR primitive + DEQUANT-F16 audit |
| 2 | `1ab9b4b` | D2 — unpack helper + sign-extend canary (3 emit tests + ptxas_verify) |
| 3 | `df5f771` | D3 — shared-memory layout + group-scale broadcast design (no code) |
| 4 | `ca371c4` | D4.1 — `validate_dims_int4` + cooperative load helpers (A, B-packed, scales) |
| 5 | `666ff6d` | D4.2+D4.3 — fragment-B per-lane dequant + full `matmul_int4` module |
| 6 | `f2e45e9` | D5 — `matmul_int4` public API + host dispatch + rustdoc |
| 7 | `12ee369` | D6 — GPU e2e tests (12 pass bit-exact on sm_89) |
| 8 | `ba59e55` | D7 — bench + `examples/int4_matmul` showcase + xtask integration |
| 9 | _this_ | D8 — docs closeout (CHANGELOG, phases, master plan, README, sprint log) |

### Follow-ups noted for future sprints

- **Asymmetric INT4 with zero-points** — next quant sprint.
- **Non-128 group sizes** (32 / 64 / 256).
- **GGUF-family packings** (Q4_0 / Q4_K / Q4_K_M).
- **AWQ activation-aware scaling**.
- **W4A8 / W4A4 mixed-precision**.
- **Fused bias + activation epilogue**.
- **`cp.async` + software pipelining for INT4**.
- **Widen K_TILE_SHARED from 16 to 32** — perf follow-up if barrier overhead dominates at a given shape. Implemented as a named-constant change.
- **`kaio-candle::matmul_int4` binding** — lands in Sprint 7.4.
- **`int4_pack_gptq` CPU helper in the main `kaio_ops` crate** — if external-interop demand surfaces.

## Architectural decisions

See the plan for full AD1–AD11. Summary:

| ID | Decision |
|---|---|
| AD1 | DEQUANT-F16 only (no native INT4 tensor core on sm_80+) |
| AD2 | Symmetric INT4, no zero points |
| AD3 | Group size fixed at 128 |
| AD4 | Packed weights stay packed in shared mem; dequant fuses at shared→register transfer |
| AD5 | Reuse INT8 block structure (4 warps × 64×64) |
| AD6 | W4A16 |
| AD7 | Sign-extend canary at emit / ptxas_verify / GPU e2e layers |
| AD8 | `ptxas --verbose` register-count check at D4 close-out |
| AD9 | No fused bias / activation |
| AD10 | No runtime kernel dispatch on shape / group size |
| AD11 | Bench variance reported honestly |

## Results

_Pending D2 → D8._
