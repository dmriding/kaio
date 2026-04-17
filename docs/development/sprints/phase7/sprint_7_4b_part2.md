# Sprint 7.4b-part2 — Direct-call bridge pattern + `qkv_project_int{4,8}`

**Status:** ✅ Complete
**Branch:** `phase7-wrap`
**Release target:** bundled into Phase 7 aggregate release.

---

## Context

7.4a + 7.4b-part1 shipped 6 forward `CustomOp` bindings in `kaio-candle`. The fused tri-output QKV projection kernels (`qkv_project_int8`, `qkv_project_int4`) don't fit candle's `CustomOpN` traits — max 3 inputs, single output. Part2 introduces a **direct-call bridge pattern**: free functions that extract `CudaStorage` from `&Tensor` inputs, validate, call the fused kernel, and return `Result<(Tensor, Tensor, Tensor)>`.

## Scope

- Two direct-call free functions: `qkv_project_int8` (4 tensor + 3 scalar inputs → 3 f16 outputs) and `qkv_project_int4` (7 tensor inputs → 3 f16 outputs).
- Bridge extensions: `reinterpret_u8_slice_as_i8` promoted to `bridge.rs`, `ensure_rank2_contiguous_zero_offset_named` added for named-parameter error messages.
- Bit-exact GPU integration tests with independent per-projection assertions + Q/K/V differentiation canaries for both ops.

## Out of scope

- Backward / autograd (7.4c).
- Refactoring existing `CustomOpN` ops to the new pattern.
- CUDA Graph capture (same `cuCtxSynchronize` fence limitation).

---

## What shipped

### Direct-call bridge pattern

Free functions that bypass `CustomOpN` entirely. Storage is extracted via `Tensor::storage_and_layout()` → `Storage::Cuda` match. Outputs are constructed via `Tensor::from_storage(Storage::Cuda(...), shape, BackpropOp::none(), false)` which internally applies `Layout::contiguous(shape)`.

Gradient-tracked inputs are explicitly rejected before storage extraction — direct-call ops bypass candle's `BackpropOp` graph tracking, so passing a tracked tensor would silently sever the computation graph. Error messages require `.detach()`.

### `qkv_project_int8`

| Op | Pattern | Inputs | Outputs |
|---|---|---|---|
| `qkv_project_int8` | Direct-call | `x: f16[M,K]`, `w_q/w_k/w_v: u8-as-i8[K,N]`, 3× `f32` scales | `(f16[M,N], f16[M,N], f16[M,N])` |

### `qkv_project_int4`

| Op | Pattern | Inputs | Outputs |
|---|---|---|---|
| `qkv_project_int4` | Direct-call | `x: f16[M,K]`, `w_q/w_k/w_v: u32[K/8,N]`, `sq/sk/sv: f16[K/128,N]` | `(f16[M,N], f16[M,N], f16[M,N])` |

`group_size=128` locked. `K` must be a multiple of 128.

### Tests

- **12 new GPU tests**: 2 bit-exact shapes per int8, 3 per int4 (including K=512 for 4-group coverage), 2 Q/K/V differentiation canaries, 2+3 rejection tests.
- **Total kaio-candle GPU tests: 32** (was 20).
- All bit-exact on RTX 4090 sm_89.

---

## Architectural decisions

### AD1 — Direct-call free functions

`CustomOpN`'s 3-input cap and single-output return type are hard trait constraints. Direct-call preserves the fused single-kernel-launch semantics and returns `(Tensor, Tensor, Tensor)` naturally.

### AD2 — Storage extraction via `Tensor::storage_and_layout()`

Guards live in the function scope, `Storage::Cuda` match is inline (no helper that fights the borrow checker). Guards drop naturally after `sync_after_launch`.

### AD3 — Named-input error messages

Error messages cite parameter names (`"w_k: expected contiguous..."`) via the new `ensure_rank2_contiguous_zero_offset_named` bridge helper.

### AD4 — `DType::F16` output

`CustomOp`-based ops return `f32` matching the kaio-ops accumulator. Direct-call ops return `f16` because the fused kernel performs the `f32→f16` conversion internally as part of the projection fusion.

---

## Follow-ups

- **7.4c** — backward kernels + event-based stream plumbing (unblocks CUDA Graph capture) across all bridge ops.
- **kaio 0.3.1 patch release** — `dynamic-linking` feature needed on crates.io before `kaio-candle` can publish.
