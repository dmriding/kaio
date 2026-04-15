# Sprint 7.2 — INT4 dequantize-matmul (`matmul_int4`)

**Status:** 🚧 In progress (D1 + D2 + D3 + D4.1 landed; D4.2 next)
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
