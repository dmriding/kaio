# Sprint 6.3 — IR-level Tensor-Core Matmul (m16n8k16, SM 8.0+)

**Status:** Done
**Branch:** phase6
**Parent:** `fcf7a77` (6.2 complete + log-fix)

## Goal

Turn the Sprint 6.2 `mma.sync` primitive into a real matmul kernel.
First IR-authored tensor-core kernel in `kaio-ops`. Correctness first:
one warp per block, one 16×8 output tile per block, K dimension looped,
shared-memory staging without `cp.async` or `ldmatrix`. The goal is
"does mma.sync + our fragment mapping actually compute matmul?" —
not performance. The 60%+ cuBLAS target is Sprint 6.7's problem.

Secondary outcome: shared-memory-source fragment load helpers
(`load_fragment_a_m16n8k16_shared_row`, `load_fragment_b_m16n8k16_shared_col`)
ship in `kaio-core::fragment` as sibling free functions to the
global-source helpers from 6.2 — validating the "free functions, not
methods" design choice when a second memory space shows up.

## What shipped

### `kaio-core::fragment`

- `load_fragment_a_m16n8k16_shared_row(alloc, kernel, tile_base_shared, tid_x, row_stride_bytes)`
  — emits `ld.shared.b32` × 4 at the NVIDIA canonical thread-data
  offsets, re-derived from the PTX ISA §9.7.13.5.8.1 figure (not
  copy-pasted from the global variant — D3 discipline).
- `load_fragment_b_m16n8k16_shared_col(alloc, kernel, tile_base_shared, tid_x, col_stride_bytes)`
  — same for B, `ld.shared.b32` × 2.
- Internal `u32_shared_addr_from_offset` helper analogous to
  `u64_addr_from_u32_offset` from the global path.
- **API break (Sprint 6.2 caller):** `store_fragment_c_m16n8k16_global_row`
  gained a `row_stride_bytes: u32` parameter (see "Bugs caught" below).

### `kaio-ops::matmul_tc_kernel`

- `build_matmul_tc_ptx()` — IR kernel builder. One warp per block,
  grid `(ceil(N/8), ceil(M/16), 1)`, block dim `(32, 1, 1)`.
- Internal `emit_load_a_tile` (row-major → row-major `.b32` copy) and
  `emit_load_b_tile` (row-major → column-major transpose via single-fp16
  loads/stores).
- Inline per-thread D-fragment store in the kernel (bypasses
  `store_fragment_c_m16n8k16_global_row` because the matmul needs
  a runtime-valued row stride; the helper takes a compile-time u32).
  Documented as carry-forward: add a register-stride variant of the
  helper when Sprint 6.4+ needs it.
- `matmul_tc(device, a, b, c, m, n, k) -> Result<()>` — validates
  dimensions, checks SM ≥ 8.0 at runtime via `KaioDevice::info()`,
  builds PTX, loads via `load_ptx`, launches.

### Visibility

- `matmul_tc` is `pub fn` in the module, re-exported via
  `#[doc(hidden)] pub use matmul_tc_kernel::matmul_tc;` in
  `kaio-ops/src/lib.rs` with a `// TEMP:` comment flagging the
  promotion trigger (Sprint 6.7 lifting the divisibility constraint).
- No README feature announcement. The `matmul_tc` API is internal
  until 6.7 relaxes the divisibility constraint.

### Tests

- 4 kaio-core fragment unit tests (shared-source A/B, each with
  instruction-count + stride-parameter checks).
- `ptxas_verify_mma_sync_shared` in `kaio-core/tests/ptxas_verify.rs`
  — passes at sm_80.
- 6 kaio-ops host-only tests for `validate_dims_tc` covering each
  invalid-shape case + buffer-too-small + valid-shape positives.
- 1 kaio-ops host structural test: `build_matmul_tc_ptx_produces_valid_structure`
  checks the emitted PTX has the expected entry, shared decls,
  `mma.sync` string, `bar.sync`, `K_LOOP:` label and `bra`.
- 4 GPU correctness tests in `kaio-ops/tests/matmul_tc_api.rs`:
  `tiny_16_8_16`, `small_32_16_32`, `rect_128_8_16`, `medium_64_64_64`.
  All pass on RTX 4090 (sm_89) with bit-close-to-zero error.

## Gate results

All 4 correctness tests run against a fp32 CPU reference (f16 inputs
promoted, f32 accumulate). Tolerance formula: `K * 2^-10 * max_abs_input_product`.

| Test | Dimensions | abs_tol | max_abs_err | % of tol |
|---|---|---|---|---|
| tiny | 16×16 × 16×8 | 3.91e-3 | 1.19e-7 | 0.0% |
| rect | 128×16 × 16×8 | 3.91e-3 | 1.19e-7 | 0.0% |
| small | 32×32 × 32×16 | 7.81e-3 | 9.54e-7 | 0.0% |
| medium | 64×64 × 64×64 | 1.56e-2 | 4.77e-7 | 0.0% |

Observed error is ~1e-7 — four orders of magnitude below the bound.
This is floating-point rounding at the fp32 precision floor, not
fp16-multiply error. The kernel is as close to bit-exact as the GPU
accumulator reduction order allows.

## Bugs caught during execution

Two pre-existing bugs surfaced and were fixed. Both were invisible
until a real GPU matmul executed.

### Bug 1 — `ld.global.f16` / `st.shared.f16` are invalid PTX

**Root cause:** Sprint 6.1 added `PtxType::F16`/`BF16` and emitted
`ld`/`st` with `.f16`/`.bf16` type modifiers. Per PTX ISA §8.7.9,
`ld` and `st`'s valid type set is
`{b8, b16, b32, b64, b128, u8, u16, u32, u64, s8, s16, s32, s64, f32, f64}`
— `.f16` and `.bf16` are **not** valid type modifiers for scalar
memory ops. The correct form loads a 16-bit half into an `.f16`
register with `ld.global.b16 %h, [addr];`.

**Why it wasn't caught in 6.1:** the `emit_ld_global_f16` unit test
asserted the string output without ever running it through ptxas.
The `f16_buffer_roundtrip` GPU test uses `GpuBuffer` transfers, never
an emitted kernel. This sprint was the first code path to actually
execute a kernel with f16 load/store.

**Fix:** added `PtxType::ptx_memory_suffix()` that returns `.b16` for
`F16`/`BF16` and falls through to `ptx_suffix()` otherwise. All
memory-op emit sites (`MemoryOp::LdParam`, `LdGlobal`, `StGlobal`,
`LdShared`, `StShared`) now use the memory-specific suffix. Unit
tests updated to assert the correct PTX strings. `cvt` and arithmetic
sites continue to use `ptx_suffix()` because they accept `.f16`/`.bf16`
natively.

### Bug 2 — `store_fragment_c_m16n8k16_global_row` hardcoded row stride

**Root cause:** Sprint 6.2 shipped `store_fragment_c_m16n8k16_global_row`
with a hardcoded 32-byte row stride, correct for the standalone 16×8
D matrix the gate test used. For a 16×8 tile inside a larger M×N
output (the matmul case), the row stride needs to be `N * 4` bytes.

**Why it wasn't caught in 6.2:** the gate test's D was exactly 16×8,
so the hardcoded and "true" strides happened to match. The `rect_128_8_16`
test would have caught it (stride-mismatch failure only appears when
M×N > 16×8), but that test didn't exist until this sprint.

**Fix:** added `row_stride_bytes: u32` parameter to
`store_fragment_c_m16n8k16_global_row`. Gate test and fragment unit
test pass `32` (the native stride). `matmul_tc` does not use the
helper because its stride (`N * 4`) is a **runtime value** — the
u32 parameter takes a compile-time constant. Matmul emits the store
inline with a register-based stride. This is carry-forward tech
debt: future sprints that want compile-time-unknown strides need a
register-stride variant of the helper.

Both bugs exemplify a pattern Dave flagged in the 6.2 retrospective:
silent correctness bugs that unit tests don't catch but real kernels
do. 6.3 was the right place for these to surface.

## Tech-debt rollup

The four entries deferred from the 6.2 retrospective are now in
`docs/development/tech_debt.md`:

- ArithOp bitops (shr/shl/and/or) in the IR
- `group_id`/`thread_id_in_group` hoisting in fragment helpers
- `PtxModule::validate()` bypass via `load_ptx(&str)`
- No GPU runtime test for scalar f16 kernel execution

Added in this sprint:

- **`store_fragment_c_m16n8k16_global_row` register-stride variant**
  — needed once a kernel wants a runtime-valued row stride without
  emitting the store inline (Sprint 6.4+ may want this for cp.async
  staging patterns).

## Internal contract flag

Sprint 6.3 locked in a shared-memory layout that 6.4 and 6.7 must
treat as policy, not incidental:

- `tile_a` — row-major, 32-byte row stride (matches global A)
- `tile_b` — column-major, 32-byte column stride (transposed from
  global B)

Rationale in the `matmul_tc_kernel.rs` module docstring. The
B-fragment loader's two-adjacent-fp16-per-half2 contract **requires**
column-major shared B; row-major would force the fragment loader to
do unpacked `.b16` loads + bitwise pack.

## Verification

All gates green before commit:

- `cargo fmt --all --check` ✓
- `cargo clippy --workspace --all-targets -- -D warnings` ✓
- `cargo test --workspace` — 262 host tests pass (up from 247)
- `cargo test --workspace -- --ignored` (sm_89) — 108 GPU tests
  pass (up from 104), including the 4 new matmul_tc tests
- `cargo doc --workspace --no-deps` — clean

## Files

| File | Change |
|------|--------|
| `kaio-core/src/types.rs` | +`ptx_memory_suffix()` on `PtxType` + test |
| `kaio-core/src/instr/memory.rs` | all ld/st use `ptx_memory_suffix` (5 sites); f16 unit tests updated for `.b16` |
| `kaio-core/src/fragment.rs` | +shared-source A/B loaders + 4 unit tests; `store_fragment_c_m16n8k16_global_row` takes `row_stride_bytes` |
| `kaio-core/src/emit/emit_trait.rs` | emit_kernel_f16_flow test asserts `.b16` |
| `kaio-core/tests/common/mod.rs` | +`build_mma_sync_shared_ptx` |
| `kaio-core/tests/ptxas_verify.rs` | +`ptxas_verify_mma_sync_shared` |
| `kaio-ops/Cargo.toml` | +kaio-core, +cudarc, +half as regular deps |
| `kaio-ops/src/lib.rs` | +`mod matmul_tc_kernel;` + `#[doc(hidden)] pub use matmul_tc` |
| `kaio-ops/src/matmul_tc_kernel.rs` | **NEW** — full IR kernel, `matmul_tc` API, host-only validate tests |
| `kaio-ops/tests/matmul_tc_api.rs` | **NEW** — 4 GPU correctness tests |
| `kaio/tests/mma_sync_fragment.rs` | `store_fragment_c` call updated for row_stride_bytes |
| `docs/development/tech_debt.md` | 4 deferred 6.2 entries folded in + new entry |
| `docs/development/sprints/phase6/sprint_6_3.md` | **NEW** — this doc |
| `docs/development/sprints/phase6/PHASE_6_LOG.md` | 6.3 row complete |
| `CHANGELOG.md` | 6.3 bullets |
| `README.md` | 6.3 checklist ✅ (internal only, no feature announce) |

## Reviewer input

- **Opus 4.6:** flagged `pub(crate)` vs `pub` (fixed: `#[doc(hidden)] pub use`),
  missing non-square test (added: `rect_128_8_16`), B shared layout
  not explicitly stated (fixed: column-major in docstring), units
  mixing in staging math (fixed: bytes-pinned throughout).
- **Dave:** caught that `pub` matmul_tc with divisibility restrictions
  would generate support issues (fixed: internal-only); caught that my
  plan had multiplications flipped (re-verified math). Approved the
  `.f16` single-byte B staging (no bitwise pack intrinsic).
- **Codex 5.4:** bar.sync semantics (fixed: now says "block-wide fence
  for tile reuse, not mma protection"), failure diagnostics (fixed:
  full tolerance/worst-case block), validate_dims_tc host tests
  (added: 6 tests), TEMP-safeguard on hidden export (added).

## Carry-forward to 6.4

Sprint 6.4 adds `cp.async` double buffering. Pre-conditions 6.3
leaves in place:

- Shared A row-major / shared B column-major layout is policy
- `store_fragment_c` needs a runtime-stride variant if any 6.4
  pattern wants to write D through it (matmul_tc currently emits
  inline; fine for now)
- `group_id`/`thread_id_in_group` hoisting becomes more valuable with
  multi-buffer pipelines
- Any new tensor-core shape will hit the `.b16` memory-op convention
  established here — document once, in the module docstring
