# Sprint 9.3 — `ldmatrix.sync.aligned` IR primitive + matmul_tc loader measurement

**Status:** ✅ Complete (2026-06-12)
**Branch:** `phase9` (PR to `main` pending Phase 9 close)
**Gate:** Optional for v0.5.0 (the 9.2 hard gate had already shipped);
on the v0.5.0 critical path for timing only (the release review needs a
frozen surface).

---

## Context

The sync `matmul_tc` loads its A fragments with four per-thread
`ld.shared.b32` plus hand-computed offset arithmetic. PTX has a
dedicated warp-collective instruction for exactly this job —
`ldmatrix.sync.aligned` — which loads 8×8 matrix tiles and distributes
them across the warp in the register layout `mma.sync` expects. 9.3
shipped in two halves: **9.3a**, the IR primitive with full validation
and toolchain verification (in the release unconditionally — the
primitive is useful regardless of downstream uplift); and **9.3b**, a
rewire of the matmul_tc fragment-A loader whose v0.5.0 inclusion was
explicitly *measurement-decided* — ship gate non-regression +
correctness, with any uplift recorded as an outcome, not a gate.

## What shipped

### ISA audit (first-commit gate)

Before any IR code: per-variant `ptxas --verify` probes plus the PTX
ISA docs settled the open questions. All six `m8n8.{x1,x2,x4}` ×
`{plain, .trans}` `.b16` forms compile clean at sm_75/sm_80/sm_89, and
the negative control at sm_70 has ptxas itself reporting "Feature
'ldmatrix' requires .target sm_75 or higher" — so `min_sm = 75` flat,
no variant dependence (`.b16` is element *bits*, not a dtype; f16 and
bf16 tiles load identically, dissolving the master plan's
variant-dependent-minimum worry). The 16-byte row-address alignment is
a **runtime** rule ptxas does not statically check — handled by
declaring the A tile `align: 16`.

### `TensorCoreOp::LdMatrix` (C1–C2)

New kaio-core IR variant: `{ dst: LdMatrixDst, addr, trans }`, where
`LdMatrixDst::X2/X4` binds the `.num` token to the register count by
construction (a width/arity mismatch is unrepresentable). `.b16`
hardcoded in the emit arm per the sibling-variant house style. Module
validation grew two things: the first sub-Ampere tensor-core SM tier
(75, pinned by a unit test so it can't be silently "normalized" to 80)
and semantic register-type checks
(`ValidationError::LdMatrixBadRegType`) — unlike the mma variants'
typed fragment wrappers, `LdMatrix` carries raw registers, so
non-`.b32`-class dst/addr registers are rejected at module load
instead of failing as cryptic ptxas errors at JIT time. Coverage:
emission string tests for both locked forms, module-validate tests
(sm_75 accepts, sm_70 rejects, mma still gates at 80 even alongside
ldmatrix, bad-register rejections) and `ptxas_verify_ldmatrix` through
the real IR→emit path at both sm_75 and sm_80.

### Fragment-A loader + hardware contract gate (C3)

`load_fragment_a_m16n8k16_ldmatrix` in kaio-core: per lane,
`row = lane & 15`, `col_byte = lane & 16`, one address add, one
`ldmatrix.x4` — 4 ALU + 1 load per stripe vs the existing loader's
~9 ALU + 4 loads. The load-bearing claim (the lane→matrix→register
distribution reproduces the existing loader's quadrant order
[top-left, bottom-left, top-right, bottom-right]) is hardware
behavior, unverifiable by reading code — so a dedicated GPU gate
(`kaio/tests/ldmatrix_fragment_contract.rs`) proves the new loader's
fragments **bit-equal to the `ld.shared` loader it would replace** on
identical shared-tile data, at both A stripe bases, with
every-element-distinct tile data and a staging self-check so a
silent-zero run can't masquerade as equivalence. Passed first run.

### matmul_tc rewire machinery (C4)

The A-loader call sits in a helper shared with the async kernel, so
the loader became a per-call-site parameter
(`FragALoaderKind { LdShared, LdMatrix }`) — async pins `LdShared`;
the choice of sync default is one line. `tile_a` declaration bumped to
`align: 16` (the ldmatrix runtime contract; harmless for `ld.shared`).
A `#[doc(hidden)] matmul_tc_ldmatrix` sibling (precedent:
`matmul_naive`) exposes the alternate path so both variants can run in
one process. Structure tests pin the instruction-mix delta: the two
builds differ by exactly 8 `ld.shared` → 2 `ldmatrix.x4`, nothing
else; kernel stats grew a dedicated `ldmatrix` counter so the mix
shift stays visible.

### Measurement and the parking decision (C5)

New `matmul_tc_ldmatrix_bench`: a **bit-exact correctness pre-gate**
(both loader builds on identical inputs, aligned + ragged shapes —
runs *before* any timing, so a mapping bug can never produce fast
numbers for wrong output), then interleaved per-iteration A/B ratios
with alternating launch order (the locked SC-2 methodology) at
256³–4096³, with one-sided non-regression gates at every shape
(median ≥ 97%, worst ≥ 85%). Registered in `cargo xtask bench`.

Result on RTX 4090 sm_89, three release runs: 4096³ interleaved
median 102.75% / 104.49% / 102.82%, flat at smaller shapes,
non-regression everywhere. That is the signature of an effect at the
±3% structural noise floor, and it matches the risk analysis made at
planning time: at the A tile's 32-byte row stride the bank-conflict
pattern is unchanged by ldmatrix, and the sync path is global-load
bound, so an instruction-issue reduction alone does not move
wall-clock. **Maintainer call: the production default stays on the
proven `ld.shared` path; the ldmatrix rewire ships built-and-parked**
— a deferred win, not a null result. The flip is one line when an
XOR-swizzle tile layout removes the bank conflicts and lets the
collective load pay. The A/B bench stays permanently as the parked
path's rot detector.

### Docs (C6)

`docs/performance.md` gained a "measured, parked" section with the
three-run numbers, the mechanism, and an explicit note that the A/B
bench's absolute TF columns (short-burst runs, no sustained clock
ramp) are not comparable to the published worst-of-10 tables — the
interleaved ratios are thermal-invariant, so the verdict holds.
The "Path to higher throughput" list now names XOR-swizzle + default
flip as the lever, replacing the pre-measurement "ldmatrix is the
real path" claim. CHANGELOG updated.

## Tests

- kaio-core: 4 emission/accessor unit tests + 5 module-validate tests
  (new SM tier + register typing) + `ptxas_verify_ldmatrix` (sm_75 +
  sm_80).
- `kaio/tests/ldmatrix_fragment_contract.rs`: the GPU bit-equality
  contract gate (both stripes, 256 registers compared, staging
  self-checks).
- kaio-ops: 2 new structure tests (LdMatrix build shape; exact
  instruction-mix delta between loader builds) + updated SM-gate tests
  (including: mma still rejects sm_75 even with ldmatrix present).
- `matmul_tc_ldmatrix_bench`: bit-exact pre-gate + per-shape one-sided
  non-regression gates — green on all three release runs.
- Full regression net post-rewire: matmul_tc (11), matmul_tc_async
  (10 — the shared-helper blast-radius check), bf16 sync + async
  (50), int4 (12), qkv int8/int4 (19), f16 TC tuner (6) all green;
  439 host tests; `cargo xtask bench` all 9 green at the rewire
  checkpoint (both SC-2 hard gates held).

## What didn't change

- `matmul_tc` public API and its production instruction stream (the
  default loader is the same `ld.shared` path as before this sprint;
  the only emitted difference is the `tile_a` `align: 16` declaration).
- The async, bf16, int8, int4, and qkv kernels: byte-identical PTX
  (the shared-helper parameterization is compile-time).
- The `ld.shared` fragment loaders in kaio-core (live consumers across
  the int4/qkv/bf16 paths — and now also the contract-gate reference).
- kaio-candle: no bridge changes (kernel-internal work only).
- Versions: workspace 0.4.1 / kaio-candle 0.1.1 (v0.5.0 is the
  aggregate phase release).

## Follow-ups

- **XOR-swizzle A-tile layout (post-v0.5.0):** de-conflict the 32-B
  row stride, re-run the A/B bench, and flip the sync default to
  `LdMatrix` if the measurement clears the noise floor — the
  machinery, contract gate, and bench are all in place.
- **`.x2.trans` is emit-tested and ptxas-verified only.** No loader
  consumes it yet; its transposed lane layout is validated by nothing
  until a fragment-B sprint wires it. Do not build on it on faith.
- **Possible later consolidation:** when (if) the async kernel and
  fragment B migrate to ldmatrix, the `FragALoaderKind::LdShared` arm,
  the hidden sibling, and the A/B bench retire together.

9.3 closed Phase 9's planned sprint list. The release review over the
frozen surface and the version-bump pass followed, closing the phase
as v0.5.0 (2026-06-12).
