# Sprint 7.4b-part1 — `kaio-candle::matmul_int8` binding

**Status:** ✅ Complete
**Branch:** `phase7-wrap` (continues from 7.4a's `7952009`).
**Release target:** bundled with 7.4b-part2 at Phase 7 aggregate release; kaio-candle stays on `0.1.0` for now.

---

## Context

7.4a shipped `kaio-candle` with five forward `CustomOp` bindings and a complete set of reusable bridge primitives. `kaio_ops::matmul_int8` (the W8A8 symmetric-quant matmul at 80–94 TOPS on RTX 4090 sm_89) was the next kernel in the forward trinity. Part1 adds it as a sixth binding — a clean continuation of the 7.4a pattern.

The sprint was split into part1 (this doc — clean CustomOp2 extension, ~1 day) and part2 (follow-up sprint — introduces the direct-call pattern for `qkv_project_int{4,8}` which doesn't fit CustomOpN).

## Scope

- `MatmulInt8Op` (`CustomOp2`) + `matmul_int8(device, a, b, scale)` wrapper.
- Scalar `f32` scale threaded through the op struct (same pattern as `AttentionTcOp::causal`).
- Bit-exact GPU integration tests at 256³, 1024³, and 4096³ with spread scale values.
- Rejection tests for `.t()` and `.narrow(...)`.
- README op-surface row + dtype-convention paragraph.
- Runnable example `matmul_int8_candle.rs`.

## Out of scope

- Backward / autograd (7.4c).
- `qkv_project_int{4,8}` (7.4b-part2).
- Changes to `bridge.rs`.
- CUDA Graph compat (carries over the same `cuCtxSynchronize` fence limitation as 7.4a).

---

## What shipped

### `matmul_int8` binding

| Op | Trait | Kernel | Shape contract |
|---|---|---|---|
| `matmul_int8` | `CustomOp2` (with `scale: f32` field) | `kaio_ops::matmul_int8` | `u8[M, K] × u8[K, N] → f32[M, N]` with scalar `f32` scale |

Both input tensors are `DType::U8` on the candle side (see AD2 below), reinterpreted as signed INT8 inside the bridge. Output is `DType::F32` matching the kaio-ops accumulator.

### Tests + example

- **5 new GPU tests** (`#[ignore]`-gated): 3 bit-exact cross-checks + 2 rejection tests. All green on RTX 4090 sm_89.
- Scale values: `0.00125` (realistic INT8 quant ≈ `max_abs / 127`), `1.0` (identity), `47.3` (large). Spread chosen so a dropped-scale bug surfaces obviously wrong in at least two of the three regimes.
- **Total `kaio-candle` GPU test count: 20** (15 from 7.4a + 5 from part1).
- New example `matmul_int8_candle.rs` produces non-NaN output on the dev machine.

---

## Architectural decisions

### AD1 — `CustomOp2` with `scale: f32` on the op struct

Mirrors `AttentionTcOp::causal` from 7.4a D5. Scalar args ride on the op struct because candle's `CustomOp2` trait only hands through the two tensor inputs and a device reference. Single `name()` = `"kaio::matmul_int8"`; no variant needed.

### AD2 — `DType::U8` on the candle side, reinterpreted as `i8` in the bridge

candle's `DType` enum has no `I8` variant (variants at byte-width are `U8` only). The established candle convention for INT8 quant tensors is `DType::U8` with the bytes interpreted as signed INT8. The bridge reinterprets `&CudaSlice<u8>` as `&CudaSlice<i8>` via a same-layout transmute: `u8` and `i8` have identical size (1 byte) and alignment, and `cudarc::CudaSlice<T>` carries `T` only as a `PhantomData` marker, so the transmute is metadata-only with no device I/O. Read-only input invariant (matmul inputs never written by the kernel) makes aliasing safe.

### AD3 — Zero new bridge primitives

Every helper called inside `cuda_fwd` already exists from 7.4a: `ensure_rank2_contiguous_zero_offset`, `ensure_ordinal_match`, `slice_ref_from_storage::<u8>`, `buffer_ref_from_slice_readonly`, `sync_before_launch`, `kaio_err`, `sync_after_launch`, `storage_from_slice`. The only new code outside `matmul_int8.rs` itself is a small `reinterpret_u8_slice_as_i8` function local to the module — a dtype-specific transmute, not a new bridge primitive.

### AD4 — Scale-value spread catches dropped-scale bugs

The three test shapes use three distinct scale values across orders of magnitude (`0.00125 / 1.0 / 47.3`). A bridge bug that drops the scale before it reaches the kernel would produce correct output at `1.0` and subtly-wrong output at `0.01`, which a single test might miss; spreading across regimes makes the bug produce obviously-wrong output in at least two of the three tests, and the failing test's scale localises which regime broke.

---

## Results

### Correctness — all green

- 4 host tests (bridge shape gate) unchanged from 7.4a.
- 20 GPU integration tests pass bit-exact on RTX 4090 sm_89 (was 15, +5).
- `fmt` + `clippy --features cuda --all-targets -- -D warnings` + `doc --features cuda` + `doc --no-default-features` + `check --no-default-features` + `build --features cuda --examples` all clean.
- Ship-gate #14 manual run: `matmul_int8_candle` example produces non-NaN output of shape `[128, 128]`.

### Scope — all D sections landed

- D1 `matmul_int8.rs` module + `lib.rs` wire-up — ✅
- D2 5 GPU integration tests (bit-exact at 256³ / 1024³ / 4096³ + 2 rejection) — ✅
- D3 README row + dtype-convention paragraph + `matmul_int8_candle.rs` example + this sprint doc — ✅
- D4 ship gates + single commit — ✅

---

## Follow-ups

- **7.4b-part2.** Direct-call bridge pattern + `qkv_project_int8` + `qkv_project_int4`. The fused tri-output ops don't fit `CustomOpN` (max 3 inputs, single output); part2 introduces a new free-function pattern with multi-output return types.
- **7.4c.** Backward kernels + event-based stream plumbing (unblocks CUDA Graph capture) across all existing bridge ops.
- **kaio 0.3.1 patch release.** `dynamic-linking` feature needs to publish before `kaio-candle 0.1.x` can resolve against crates.io.
