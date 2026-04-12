# Sprint 6.4 — cp.async Double-Buffered Tensor-Core Matmul (SM 8.0+)

**Status:** Done
**Branch:** phase6
**Parent:** `fefb72c` (6.3 complete + log-fix)

## Goal

Layer `cp.async` double-buffering on top of Sprint 6.3's `matmul_tc`
to overlap the next K-tile's A-tile load with the current K-tile's
`mma.sync`. Sprint 6.4 is a **pipeline-pattern sprint**, not a
performance sprint — at one warp per block with one small mma per
K-tile there's only ~16 cycles of compute to hide ~100+ cycles of
memory, so overlap wins require Sprint 6.7's multi-warp restructure.
Goal: correctness + a reusable pipeline skeleton.

## What shipped

### `kaio-ops::matmul_tc_async_kernel` (new module)

- `emit_load_a_tile_async` — 32 threads × one `cp.async.ca.shared.global`
  at `size = 16` per thread = 512 B per A tile. Per-thread address
  math: `row = lane / 2`, `col_pair_byte = (lane % 2) * 16`. SAFETY
  breadcrumb comment at the emission site documents the 16-byte
  alignment invariant (requires `BK = 16` and `K % 16 == 0`).
- `build_matmul_tc_async_ptx` — full IR kernel. Preamble issues
  `A[0]` async, sync-stores `B[0]`, commits one group. K loop:
  `wait_group 0` → `bar.sync` → optional issue-`A[k+1]` + sync-store
  `B[k+1]` (predicated on `k+1 < num_k_tiles`) → fragment loads from
  current buffer → `mma.sync` → buffer toggle. No trailing `bar.sync`
  after mma — double-buffered disjoint reads/writes make it
  unnecessary, in contrast to 6.3's single-buffer design.
- `matmul_tc_async(device, a, b, c, m, n, k)` — host API with the
  same dim constraints as `matmul_tc` (M%16 = N%8 = K%16 = 0) and
  the same SM 8.0+ runtime check.
- `buffer_offsets(k_tile)` — pure helper returning
  `(a_cur, a_nxt, b_cur, b_nxt)` byte offsets. The kernel builder
  inlines equivalent register arithmetic; this helper exists to
  make the toggle math host-testable (see
  `buffer_offsets_toggle` test).

### `kaio-ops::matmul_tc_kernel` (visibility bump only)

- `validate_dims_tc` and `emit_load_b_tile` promoted from private to
  `pub(crate)`. No behavioral change — Sprint 6.4's async module
  reuses both unchanged.

### Visibility

- `matmul_tc_async` is `pub fn` in the module, re-exported via
  `#[doc(hidden)] pub use` in `kaio-ops/src/lib.rs` with a `TEMP:`
  comment flagging the promotion trigger (Sprint 6.7 lifting the
  divisibility constraint). Same pattern as `matmul_tc`.
- No README feature announcement. Internal-only API until 6.7.

### Tests

- 2 new host tests in `matmul_tc_async_kernel`:
  `buffer_offsets_toggle` (verifies `(0→0/512/0/256)`,
  `(1→512/0/256/0)`, etc.) and
  `build_matmul_tc_async_ptx_produces_valid_structure` (instruction-
  centric: `cp.async.ca.shared.global` + `commit_group` + `wait_group`
  + `mma.sync.aligned.m16n8k16` + exact `.shared` decl strings +
  `mma_count == 1` + `commit_group_count == 2`).
- 4 new GPU correctness tests in
  `kaio-ops/tests/matmul_tc_async_api.rs`: `tiny_16_8_16`,
  `small_32_16_32`, `rect_128_8_16` (non-square), `medium_64_64_64`.
  Each compares element-wise against a fp32 CPU reference with the
  6.3 K-scaled tolerance (`K * 2^-10 * max_abs_input_product`) and
  prints a full diagnostic block on tolerance miss. Test-data
  generator + reference are inlined (same pattern as the 6.3 suite).

## Gate results

All 4 correctness tests pass with bit-close-to-zero error,
identical to 6.3's numbers:

| Test | Dimensions | abs_tol | max_abs_err | % of tol |
|---|---|---|---|---|
| tiny | 16×16 × 16×8 | 3.91e-3 | 1.19e-7 | 0.0% |
| small | 32×32 × 32×16 | 7.81e-3 | 9.54e-7 | 0.0% |
| rect | 128×16 × 16×8 | 3.91e-3 | 1.19e-7 | 0.0% |
| medium | 64×64 × 64×64 | 1.56e-2 | 4.77e-7 | 0.0% |

Error floor is the same fp32 accumulator rounding observed in 6.3,
four orders of magnitude below the bound. First-try green — no
correctness issues surfaced during execution.

## Sanity timing observation (env-gated)

With `KAIO_SPRINT_6_4_TIMING=1` set on the `medium_64_64_64` test,
release build, 50 iterations after a warm-up launch:

| Kernel | ms/iter | ratio |
|---|---|---|
| `matmul_tc` (sync, 6.3) | 0.250 | 1.00× |
| `matmul_tc_async` (6.4) | 0.269 | 1.07× |

**Async is ~7% slower than sync at this workload.** This is the
expected outcome and was predicted in the plan: at 1 warp per
block × 1 small `mma.sync` per K-tile, there is not enough compute
to hide cp.async's latency, and the `wait_group 0` + `bar.sync`
overhead is net-negative. Sprint 6.7's multi-warp restructure is
what makes async beat sync — not 6.4. This number belongs in the
sprint log as a baseline for 6.7 to improve against, not as a
claim of current performance.

Not checked into any benchmark doc and not present in default CI
output (env-gated).

## Architectural decisions made

From the plan (all folded in as-shipped):

- **D1 — Sibling kernel, not flag on matmul_tc.** Two complete
  kernels the auto-tuner can pick between; `matmul_tc` stays
  untouched as a trusted baseline.
- **D2 — Double-buffer layout: `tile_a[1024]` align 16, `tile_b[512]`
  align 4.** align 16 on `tile_a` is load-bearing for
  `cp.async.ca size = 16` destination alignment. Buffer selection
  via runtime `(k_tile & 1) * buf_bytes` add — not XOR-toggle,
  which would require size-aligned base addresses that `SharedDecl`
  doesn't guarantee.
- **D3 — A async, B sync.** `cp.async` requires contiguous byte
  ranges; B's row-major-global → column-major-shared layout is a
  strided gather that cp.async cannot express. Rather than break
  6.3's shared-layout contract (which the fragment loader depends
  on) or add a transpose staging buffer, B stays synchronous.
  Revisiting is 6.7 work that will likely coincide with the
  multi-warp restructure.
- **D4 — cp.async granularity = 16 B per thread.** One `cp.async.ca`
  per thread per A tile. Alignment proof spelled out in the
  emission-site SAFETY comment.
- **D5 — wait-at-top pipeline ordering.** `wait_group 0` + `bar.sync`
  at the top of each iter, then `@p_has_next` issue-next, then
  compute. Loop-entry invariant documented in the module docstring.
- **D6 — No trailing bar.sync after mma.** Disjoint shared regions
  (cur vs nxt buffers) mean no race between iterations; the
  top-of-next-iter `bar.sync` handles everything. This differs from
  6.3 intentionally and is called out in the docstring.
- **D8 — `validate_dims_tc` shared via `pub(crate)`.** Single
  implementation, two callers; host-only dim-validation tests live
  with the 6.3 module and cover both.
- **D9 — SM check reuses 6.3's pattern** via `KaioDevice::info()`.
  Error message names the feature (`"cp.async + mma.sync.m16n8k16"`).
- **D10 — Host structural test is instruction-centric, not
  label-centric.** Labels (`K_LOOP:`, `SKIP_NEXT_ISSUE:`) are
  internal spellings; tests instead assert mnemonics, `.shared`
  decl strings, `mma_count == 1`, and `commit_group_count == 2`.
- **D11 — Timing log gated on `KAIO_SPRINT_6_4_TIMING=1`.** Default
  off so CI logs stay clean.

## Bugs caught during execution

None. First-try green on all four correctness tests — attributable
to 6.3 having shaken out two silent PTX emission bugs (the `.b16`
memory suffix issue and the `row_stride_bytes` parameter) that would
have surfaced here otherwise.

## Tech-debt rollup

No new entries. Existing queue items relevant to 6.4:

- **`PtxModule::validate()` bypass via `load_ptx(&str)`** — escalated
  urgency per Codex review. With 6.4 adding a second internal TC
  kernel, the surface area that bypasses SM validation has grown:
  mma.sync, cp.async, two internal kernel variants. Sprint 6.5's
  auto-tuner is the natural centralization point for load paths;
  this should be one of the first cleanup targets after 6.5.
- `store_fragment_c_m16n8k16_global_row` register-stride variant —
  still deferred. 6.4 reuses 6.3's inline D-store (stride is
  runtime-valued).
- `group_id` / `thread_id_in_group` hoisting — still deferred. 6.4
  is single-mma-per-K-tile so the re-compute cost is modest (~8
  extra instructions on medium test).
- ArithOp bitops (shr/shl/and/or) in the IR — still deferred. Would
  simplify cp.async's per-thread `(lane / 2, (lane % 2) * 16)` math
  to shifts and masks; orthogonal to the pipeline work.

## Internal contract flag (preserved from 6.3)

Sprint 6.3 established, and 6.4 confirms:

- `tile_a` — row-major, 32-byte row stride (matches global A)
- `tile_b` — column-major, 32-byte column stride (transposed from
  global B)

6.4 adds: **double-buffered** layout. `tile_a` is 1024 B (2 × 512),
`tile_b` is 512 B (2 × 256). Total per-block shared = 1536 B.
`tile_a` requires `align = 16` for `cp.async.ca size = 16`
destination alignment.

Sprint 6.7's multi-warp restructure must either preserve these
layouts or update the fragment loaders in lockstep. Any new shape
(m16n8k8, m16n8k32) with different fragment adjacency requirements
will need its own layout + loader pair.

## Verification

All gates green before commit:

- `cargo fmt --all --check` ✓
- `cargo clippy --workspace --all-targets -- -D warnings` ✓
- `cargo test --workspace` — 264 host tests pass (up from 262)
- `cargo test --workspace -- --ignored` (sm_89) — 112 GPU tests
  pass (up from 108), including the 4 new matmul_tc_async tests
- `cargo doc --workspace --no-deps` — clean

## Files

| File | Change |
|------|--------|
| `kaio-ops/src/matmul_tc_async_kernel.rs` | **NEW** — full IR kernel, `matmul_tc_async` API, `buffer_offsets` helper, 2 host tests |
| `kaio-ops/src/matmul_tc_kernel.rs` | `validate_dims_tc` and `emit_load_b_tile` promoted to `pub(crate)` (visibility only) |
| `kaio-ops/src/lib.rs` | `mod matmul_tc_async_kernel;` + `#[doc(hidden)] pub use` with TEMP comment |
| `kaio-ops/tests/matmul_tc_async_api.rs` | **NEW** — 4 GPU correctness tests + env-gated timing block |
| `docs/development/sprints/phase6/sprint_6_4.md` | **NEW** — this doc |
| `docs/development/sprints/phase6/PHASE_6_LOG.md` | 6.4 row complete |
| `CHANGELOG.md` | 6.4 bullets (internal / doc-hidden) |
| `README.md` | 6.4 checklist ✅ (internal only, no feature announce) |

## Reviewer input

- **Opus 4.6:** green light with two documentation additions —
  explicit B sync-store race-freedom note in the module docstring
  (folded into D6), and a SAFETY breadcrumb at the `cp.async.ca`
  emission site (folded into D4).
- **Codex 5.4:** green light with four adjustments — explicit
  loop-entry invariant block in the module docstring (folded into
  D5), instruction-centric structural test instead of label-string
  checks (folded into D10), host-only buffer-offset toggle unit
  test via extracted `buffer_offsets` helper (folded into D10),
  env-var gate on the timing log (folded into D11). Also
  escalated `load_ptx(&str)` validation-bypass tech-debt urgency.

## Carry-forward to 6.5

Sprint 6.5 adds the 3-way auto-tuner (scalar / TC / TC+async).
Pre-conditions 6.4 leaves in place:

- `matmul_tc_async` callable as `#[doc(hidden)] pub` — tuner can
  dispatch internally.
- Same dim constraints as `matmul_tc` (M%16 = N%8 = K%16 = 0); no
  new capability gate beyond SM 8.0+ which mma.sync already requires.
- At 1 warp / block × 1 small mma per K-tile, async is ~7% slower
  than sync on 64×64×64. The tuner should prefer sync over async
  on this configuration until 6.7's multi-warp restructure raises
  async's ceiling. The sprint_6_4 timing observation is the first
  data point; 6.5's tuner can use it to set defaults.
- `PtxModule::validate()` bypass tech-debt is now load-bearing
  enough that 6.5's tuner dispatch is a natural place to
  centralize kernel loading through `load_module` and drop the
  `load_ptx(&str)` path.
