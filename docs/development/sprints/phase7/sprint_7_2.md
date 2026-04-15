# Sprint 7.2 — INT4 dequantize-matmul (`matmul_int4`)

**Status:** 🚧 In progress (D1 landed)
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
