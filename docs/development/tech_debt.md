# KAIO — Technical Debt Tracker

Items identified during development that are deferred but should be
addressed before the next release milestone. v0.1.0 shipped (Phase 5);
v0.2.0 is tracked under Phase 6. Organized by priority.

## High Priority (before Phase 5)

### ~~block*reduce*\* 2D support~~ RESOLVED (Sprint 5.1)

Reductions now compute linear tid = `tidx + tidy * block_dim_x` for
2D kernels. 1D kernels unchanged (zero extra instructions). Tested
with square (16x16), asymmetric (32x8), and identity-based 2D blocks.
**Resolved:** Sprint 5.1

### ~~Host-level codegen regression tests~~ RESOLVED (Sprint 6.10 D2)

Four host-side codegen regression tests landed in `kaio-macros`, all passing in CI without a GPU:

- `launch_wrapper_emits_correct_block_dim_1d` / `_2d` — verify `LaunchConfig.block_dim` matches declared `block_size` for both 1D and 2D attribute forms
- `shared_memory_lowering_emits_shared_addr_pattern` — verify `Operand::SharedAddr(...)` + `Add` pattern for `shared_mem![]` lowering
- `reduction_lowering_uses_named_symbol` — verify `"_kaio_reduce_smem"` literal used by `block_reduce_sum` / `_max` lowering
- `launch_wrapper_threads_compute_capability_into_module_build` — added in D2 as `#[ignore]`'d spec-stub for D1a, activated after D1a landed; verifies macro reads `device.info().compute_capability`, calls `load_module`, does NOT emit `load_ptx`

Each test has a regression canary comment naming the mutation it guards against. One mutation verified end-to-end (`Operand::SharedAddr` → `Operand::MUTATION_TEST` in `lower/memory.rs` — test failed with the expected diagnostic, then reverted). Other three tests use identical substring-match mechanics, validated by parity.

**Resolved:** Sprint 6.10 D2

### fma() f64 support

`fma()` builtin validates all args as f32 only. The `ArithOp::Fma` variant
supports f64 in the IR but the builtin and IDE stub are f32-only.
**Added:** Sprint 4.1 | **Sprint:** when f64 kernels are needed

## Medium Priority

### Shared memory base register hoisting

`mov.u32 base, <symbol>` is emitted for every shared memory access, even
inside tight loops. Could be hoisted to emit once before the loop body.
Adds 2 instructions per access — acceptable for correctness but becomes
a performance concern in matmul inner loops.
**Added:** Sprint 4.2 | **Sprint:** Phase 4.6 optimization

### Shared memory helper abstraction

Two code paths generate the same `mov+mul+add` addressing pattern:
generic `compute_shared_address()` and reduction codegen in builtins.rs.
Extract a shared helper to prevent drift as the pattern evolves.
**Added:** Sprint 4.2 review | **Sprint:** when either path changes

### ~~Windows CI~~ RESOLVED (Sprint 5.6)

GitHub Actions matrix now includes Ubuntu + Windows. Doc build job
added with `RUSTDOCFLAGS="-D warnings"`.
**Resolved:** Sprint 5.6

## Low Priority

### `&&` / `||` logical operators in kernel DSL

Currently not supported — users must use nested `if` statements.
Discovered in Sprint 4.1 when writing 2D kernel test.
**Added:** Sprint 4.1 | **Sprint:** TBD

### Compound assignment for shared memory

`sdata[idx] += val` is not supported — requires `sdata[idx] = sdata[idx] + val`.
**Added:** Phase 3 | **Sprint:** TBD

### ArithOp::Shr / Shl / And / Or (bitops) in the IR

Two call sites currently use `div.u32` / `rem.u32` by power-of-two
constants where bit ops would be cleaner PTX:

- Reduction warp_id: `div.u32 warp_id, tid, 32` could be `shr.u32 warp_id, tid, 5`
- Fragment helpers (Sprint 6.2, `kaio-core/src/fragment.rs::compute_group_thread_ids`):
  `div.u32 group_id, tid, 4` + `rem.u32 tig, tid, 4` could be
  `shr.u32 group_id, tid, 2` + `and.b32 tig, tid, 3`

The driver lowers constant-power-of-two div/rem correctly (no
correctness issue), but adding proper bitops to `ArithOp` would
produce cleaner PTX and be useful beyond tensor-core fragment math.
**Added:** Phase 3 / extended Sprint 6.2 | **Sprint:** Phase 4.6 optimization

### `MemoryOp::LdGlobalB128` landed but not wired into any kernel

Sprint 6.7b landed the vectorized 128-bit global load IR primitive
(`ld.global.v4.b32 {...}, [...];`) in `kaio-core/src/instr/memory.rs`
with constructor validation (4 b32 destinations + 1 u64 address),
emit path, 6 unit tests, and `ptxas_verify_ld_global_b128` coverage.
The primitive is well-formed and cleanly isolatable.

**Why it's not used yet:** wiring LDG.128 into the cooperative Tile B
global load requires a companion "unpack b32 into two b16 half
values" IR primitive so that the 8 fp16 values in the 4 loaded b32
registers can be scattered into 8 different col-major shared
positions (the current layout keeps fragment B's `.row.col`
requirement intact). The two options are `mov.b32 {h_lo, h_hi},
src` (vector-splitting move, cleanest PTX) or an `shr` + implicit
b32-to-b16 store truncation pattern. Neither exists in kaio-core
today. Adding that primitive under 6.7b's implementation-deadline
pressure cut against the D10 orthogonality requirement, so 6.7b
shipped without it and the LDG.128 variant stays as a future-sprint
anchor.

A dedicated sync-path-optimisation sprint (or Phase 7 work on
`ldmatrix.sync.aligned`) can pick this up cleanly by designing the
b32-to-b16 split primitive properly first, then wiring LDG.128 into
`emit_mw_load_tile_b_16x64` as a per-block-setp-gated fast path
(interior blocks with `N % 8 == 0`).

**Added:** Sprint 6.7b | **Sprint:** TBD (post-0.2.0)

### ~~`group_id` / `thread_id_in_group` hoisting in fragment helpers~~ RESOLVED (Sprint 6.7b)

`load_fragment_a_m16n8k16_shared_row` and
`load_fragment_b_m16n8k16_shared_col` in
`kaio-core/src/fragment.rs` now accept a
`group_tig_override: Option<(Register, Register)>` parameter. When
`None`, behaviour matches pre-6.7b (internal `div.u32`/`rem.u32`
emit). When `Some((g, t))`, the loaders skip the internal compute
and use the caller-supplied registers. The multi-warp matmul_tc
kernels compute `(group_id, tig)` once at kernel start and thread
them through `emit_warp_quadrant_mma` to each of the 6 fragment
loader calls per K-iter, saving 6 × `div.u32`/`rem.u32` pairs per
K-iter. Combined with Tile B col-stride padding (32 → 36 B), the
measured 6.7b uplift over 6.7 was +2.4pp sync / +7.4pp async
(79.9→82.3 / 85.1→92.5).
**Resolved:** Sprint 6.7b

### ~~`PtxModule::validate()` bypass via `load_ptx(&str)`~~ RESOLVED (Sprint 6.10 D1)

Full resolution across D1a (migration) + D1b (deprecation):

- **D1a** — `#[gpu_kernel]` proc-macro codegen in `kaio-macros/src/codegen/` now emits `device.load_module(&PtxModule)` instead of `device.load_ptx(&str)`. The macro reads `device.info().compute_capability` at launch time, formats as `sm_XX`, passes it to a generated `build_module(sm: &str) -> PtxModule` function, and calls `load_module`. The `PTX_CACHE: OnceLock<String>` cache was removed; modules are rebuilt per launch (microseconds of host overhead; measured no 4096² kernel-time regression, +0.01ms at 256² as the expected cost-model change).
- **D1a** also migrated 4 non-macro test call sites (`vector_add_e2e`, `cp_async_roundtrip`, `mma_sync_fragment`) to the `load_module` path. The one remaining `load_ptx(&str)` caller is `load_module()` itself at `device.rs:117` (private implementation detail after `validate()`).
- **D1b** — Added `#[deprecated(since = "0.2.1", note = "use load_module(&PtxModule) — runs PtxModule::validate() for readable SM-mismatch errors")]` on `KaioDevice::load_ptx(&str)` with a migration-guide rustdoc. Public API preserved (not removed — raw-PTX use cases still supported for external PTX files / hand-written PTX research). Internal self-call in `load_module` gated with `#[allow(deprecated)]` to prevent warning surface.

Trust-boundary fix complete: user-authored kernels with Ampere-gated features (`mma.sync`, `cp.async`) on sub-Ampere targets now surface a structured `KaioError::Validation` via `PtxModule::validate()` instead of a cryptic ptxas error.

**Added:** Sprint 6.2 | **Elevated:** Sprint 6.4 | **Partial resolution:** Sprint 6.5 (TC kernels migrated) | **Resolved:** Sprint 6.10 D1a + D1b (macro-codegen migration + deprecation)

### No GPU runtime test for scalar f16 kernel execution

Sprint 6.1 added `PtxType::F16`/`BF16`, `RegKind::H`/`Hb`, and cvt
rounding. The host-side emit test `emit_kernel_f16_flow` proves the
PTX string is correct for a load/cvt/add/cvt/store flow, and
`f16_buffer_roundtrip` proves `GpuBuffer<f16>` moves data. But no
GPU test actually launches a kernel that does scalar f16 arithmetic
and verifies the result — so if the CVT rounding modes were wrong,
only ptxas would catch it (and only if ptxas is strict about it).

Low priority — the fragment / mma.sync GPU tests (Sprint 6.2) exercise
f16 load + packed storage + f32 conversion in the tensor-core path,
which is indirect coverage. A dedicated scalar-f16 GPU roundtrip
test would close the gap explicitly.
**Added:** Sprint 6.2 (retroactive 6.1 gap) | **Sprint:** TBD

### ~~`ptxas_verify` tests mutate process-global `KAIO_SM_TARGET`~~ RESOLVED (Sprint 6.10 D3)

Fixed by option 1: three builders in `kaio-core/tests/common/mod.rs` (`build_mma_sync_ptx`, `build_mma_sync_shared_ptx`, `build_cp_async_ptx`) now take `sm: &str` as an explicit argument. All three `unsafe { std::env::set_var("KAIO_SM_TARGET", ...) }` calls in `kaio-core/tests/ptxas_verify.rs` are removed. Tests pass the SM target directly to the builder.

**Audit surfaced (not in scope for D3):** three other helpers (`build_vector_add_ptx`, `build_shared_mem_ptx`, `build_ld_global_b128_ptx`) still read `KAIO_SM_TARGET` internally. Their callers do NOT mutate the env var — no hygiene issue today. Parameterizing them for consistency is a minor follow-up, not a correctness concern.

**Added:** Sprint 6.4 review, 2026-04-12 | **Resolved:** Sprint 6.10 D3

### Pathological-shape benchmark CPU reference dominates wall time

The 1023×1023×1024 GPU correctness test in `matmul_tc_api.rs` /
`matmul_tc_async_api.rs` (Sprint 6.7 Gate C) takes ~67 seconds per
test. The kernel itself completes in milliseconds; the runtime is
dominated by the host-side O(M·N·K) ≈ 1e9-op CPU reference loop in
`cpu_matmul_f16xf16_f32`. Acceptable for a correctness gate — the
test does what it says — but slows down `cargo test --workspace
-- --ignored` runs noticeably.

Candidates: (1) sample the output at fixed indices and compare only
those against a partial reference (cheap analytical formula or a
Rayon-parallelized partial cpu_matmul); (2) replace bit-close with
a checksum (XOR of the lower 32 bits of every output as f32 bit
pattern, compared against a CPU-computed checksum); (3) lower the
"large off-by-one" to 511×511×512 — keeps the off-by-one stress vs
512 boundary without the 4× runtime hit.

**Added:** Sprint 6.7 Gate C | **Sprint:** TBD (low priority)
