# KAIO — Technical Debt Tracker

Items identified during development that are deferred but should be
addressed before v0.1.0 release. Organized by priority.

## High Priority (before Phase 5)

### ~~block*reduce*\* 2D support~~ RESOLVED (Sprint 5.1)

Reductions now compute linear tid = `tidx + tidy * block_dim_x` for
2D kernels. 1D kernels unchanged (zero extra instructions). Tested
with square (16x16), asymmetric (32x8), and identity-based 2D blocks.
**Resolved:** Sprint 5.1

### Host-level codegen regression tests

Critical fixes (launch config block_dim, shared addressing) are only
verified by ignored GPU tests — not part of CI. Add host-level tests:

- launch wrapper emits correct `block_dim` matching declared `block_size`
- shared memory lowering emits `Operand::SharedAddr` + `Add` pattern
- reduction lowering uses `_kaio_reduce_smem` named-symbol addressing
  **Added:** Sprint 4.2 review | **Sprint:** Phase 4.8 or Phase 5

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

### `group_id` / `thread_id_in_group` hoisting in fragment helpers

Every `load_fragment_*` / `store_fragment_*` call in
`kaio-core/src/fragment.rs` independently emits its own `div.u32` +
`rem.u32` to derive `groupID` and `threadID_in_group` from `%tid.x`.
For the Sprint 6.2 gate test (3 helper calls) this is 6 extra
instructions per warp — negligible. For Sprint 6.3's tiled matmul
(multiple fragment loads per K-tile, many K-tiles) the same
computation gets repeated dozens of times and should be hoisted.

Cheapest fix: have the helpers accept `(group_id, thread_id_in_group)`
registers as an optional parameter, with a `None` → "compute locally"
fallback. Or: introduce a `FragmentWarpContext` struct holding the
two registers + `%tid.x`, computed once at kernel start and threaded
through the helpers. Decide when 6.3 shows the pattern concretely.
**Added:** Sprint 6.2 | **Sprint:** Phase 6.3 when the shape of the
tiled kernel makes the right trade-off obvious

### `PtxModule::validate()` bypass via `load_ptx(&str)`

`kaio-runtime::KaioDevice::load_module(&PtxModule)` calls
`PtxModule::validate()` before handing PTX text to the driver. But
`KaioDevice::load_ptx(&str)` (the older raw-text entrypoint) does
not — a user who builds a module, emits to text manually, and
loads via `load_ptx(ptx_text)` skips SM validation.

Not a correctness bug (ptxas will still catch it downstream, just
with a cryptic error). Candidates if this bites:

1. Re-parse the `.target sm_NN` line from the PTX text and do a
   minimal regex check for known-SM-gated features (`mma.sync`,
   `cp.async`). Hacky.
2. Deprecate `load_ptx(&str)` in favor of `load_module(&PtxModule)`
   and migrate all internal callers. Cleanest.
3. Just document the divergence and leave it.

Leaning toward option 2 in a future sprint, after the macro codegen
path has fully moved to the `PtxModule` entrypoint.

**Priority elevated (Sprint 6.4 retrospective, Codex 5.4 review):**
With 6.4 adding a second internal TC kernel (`matmul_tc_async`) on
top of `matmul_tc`, the surface area that bypasses SM validation
has grown — `mma.sync`, `cp.async`, and two internal kernel variants
all take the `load_ptx(&str)` path. This is now a trust-boundary
issue, not just an optimization. Sprint 6.5's auto-tuner is a
natural centralization point for kernel loading; migrating all
internal callers to `load_module(&PtxModule)` there, then deprecating
`load_ptx(&str)`, should be one of the **first** cleanup targets
after 6.5 or during post-Phase-6 stabilization.

**Added:** Sprint 6.2 | **Elevated:** Sprint 6.4 | **Sprint:** target 6.5 / post-Phase-6

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

### `ptxas_verify` tests mutate process-global `KAIO_SM_TARGET`

Three of the `kaio-core/tests/ptxas_verify.rs` tests
(`ptxas_verify_mma_sync`, `ptxas_verify_mma_sync_shared`,
`ptxas_verify_cp_async`) call `unsafe { std::env::set_var("KAIO_SM_TARGET",
...) }` to force an Ampere-or-better target for the PTX they're about
to verify. The test binary's single-threaded default makes this safe
in practice, but there's no harness-level enforcement (no
`#[serial]`, no cargo-nextest isolation), and any future parallel
test that reads `KAIO_SM_TARGET` could race against these writes.

Not a correctness issue in the PTX itself — the test harness's
default `--test-threads=1` inside a single binary keeps this
serialized — but it's a hygiene gap that would bite if someone runs
with `--test-threads=N` or a future test starts mutating env vars
from another thread. Candidates:

1. Replace `set_var` with a per-test PTX re-build path that accepts
   the SM target as an argument, sidestepping the env var entirely.
2. Add a `serial_test` dependency and gate the affected tests.
3. Document the constraint and leave it.

Option 1 is cleanest. The `build_mma_sync_ptx` / `build_cp_async_ptx`
helpers in `tests/common/mod.rs` currently read `KAIO_SM_TARGET`
internally — parameterize them to take an SM target instead, and
the env-var mutation disappears.
**Added:** Sprint 6.4 (Codex adversarial review 2026-04-12) | **Sprint:** TBD
