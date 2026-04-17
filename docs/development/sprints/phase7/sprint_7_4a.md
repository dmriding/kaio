# Sprint 7.4a — `kaio-candle` bridge crate (forward-only)

**Status:** ✅ Complete (D1–D7 shipped)
**Branch:** `phase7-wrap` off `phase7-ship`; single merge to `main` once 7.4b closes.
**Release target:** `kaio-candle 0.1.0` on crates.io after `kaio 0.3.1` (dynamic-linking feature) publishes.

---

## Context

Phase 7 shipped five forward-path GPU kernels usable from pure `kaio-ops`
calls: `matmul_tc`, `matmul_tc_async`, `matmul_int4`, `attention_tc`,
`attention_tc_causal`. To reach the population already writing Rust ML
code against candle, those kernels need a `candle_core::CustomOp` surface
that takes `Tensor` in and hands `Tensor` out. Sprint 7.4a builds that
bridge.

## Scope

- Forward-only `CustomOp2` / `CustomOp3` implementations for all five
  kernels — backward (`bwd()`) defaults to candle's
  `BackwardNotSupported`. Training integrations land in 7.4c.
- Standalone crate `kaio-candle/`, excluded from the main workspace
  (cudarc `dynamic-loading` vs `dynamic-linking` feature mutual
  exclusion — see AD1 below).
- Bit-exact GPU integration tests cross-checking candle-routed calls
  against direct `kaio-ops` calls on the same input bits.
- Runnable examples, weekly candle-HEAD compat CI, publish prep.

## Out of scope

- Backward kernels / autograd. Forward-only for 7.4a; training ops wait
  for 7.4c.
- The remaining quant ops (`matmul_int8`, `qkv_project_int8`,
  `qkv_project_int4`). These land in 7.4b on the same branch.
- Explicit candle `CudaStream` plumbing — the bridge uses the KAIO
  default stream with `cuCtxSynchronize` fences on each side of the
  launch. CUDA Graph capture is incompatible for the same reason; see
  AD5 and the follow-ups.
- CPU / Metal fallback.

---

## What shipped

### Five forward CustomOp bindings

| Op | Trait | Kernel | Shape contract |
|---|---|---|---|
| `matmul_tc` | `CustomOp2` | `kaio_ops::matmul_tc` | `f16[M, K] × f16[K, N] → f32[M, N]` |
| `matmul_tc_async` | `CustomOp2` | `kaio_ops::matmul_tc_async` | same |
| `matmul_int4` | `CustomOp3` | `kaio_ops::matmul_int4` | `f16[M, K] × u32[K/8, N] × f16[K/128, N] → f32[M, N]` |
| `attention_tc` | `CustomOp3` | `kaio_ops::attention_tc` | `f16[seq_q, d_k] × f16[seq_k, d_k] × f16[seq_k, d_v] → f32[seq_q, d_v]` |
| `attention_tc_causal` | `CustomOp3` (`causal: bool` field) | `kaio_ops::attention_tc_causal` | same |

All ops take an explicit `Arc<KaioDevice>` as the first argument and the
candle `Tensor` inputs as the remainder. Every call checks that the
KAIO device and the candle device share the same CUDA ordinal.

### Bridge primitives (`kaio-candle/src/bridge.rs`, crate-private)

- `slice_ref_from_storage<T>(&CudaStorage) → Result<&CudaSlice<T>>`
- `storage_from_slice<T>(CudaSlice<T>, CudaDevice) → CudaStorage`
- `buffer_ref_from_slice_readonly<T>(&CudaSlice<T>) → &GpuBuffer<T>`
  — `repr(transparent)` newtype cast under an aliasing contract
  (read-only inputs only) verified by compile-time `static_assertions`.
- `ensure_ordinal_match(&CudaDevice, &KaioDevice) → Result<()>`
- `sync_before_launch` + `sync_after_launch` — `cuCtxSynchronize`
  fences on either side of every kernel launch (AD5).
- `ensure_rank2_contiguous_zero_offset` — rejects rank ≠ 2,
  non-contiguous, and non-zero storage offset with concrete error
  messages and reshape hints.

### Tests + docs + CI

- **15 GPU integration tests** (`#[ignore]`-gated, require a CUDA
  device): 12 bit-exact cross-checks across the five ops at 2–3 shapes
  each, plus 3 rejection tests (`matmul_tc` with `.t()` + `.narrow()`,
  `matmul_int4` with K not a multiple of 128). All green on RTX 4090
  sm_89.
- **4 host tests** exercising the rank + contiguity + offset gate.
- `cargo fmt`, `cargo clippy --features cuda --all-targets -- -D warnings`,
  `cargo doc` clean with and without the `cuda` feature.
- `cargo check --no-default-features` succeeds on a no-CUDA-toolkit
  host (AD8 / ship gate #10).
- Runnable examples: `matmul_tc_candle.rs`, `attention_tc_candle.rs`.
- Weekly candle-HEAD compatibility workflow
  (`.github/workflows/candle-head.yml`) with step-level
  `continue-on-error` + auto-issue on failure.

---

## Architectural decisions

### AD1 — Standalone crate, not a workspace member

cudarc rejects `dynamic-loading` + `dynamic-linking` as simultaneously
active features. Main KAIO defaults to `dynamic-loading` so host tests
and CI run without the CUDA toolkit; candle-core's `cuda` feature
activates `dynamic-linking`. Cargo unions features across a workspace
build, so including `kaio-candle` in the main workspace would force
every main-workspace build to also carry `dynamic-linking`, breaking
no-CUDA CI.

`kaio-candle/` lives at the repo root with its own `Cargo.toml` and
excluded from the workspace. It consumes `kaio` and `kaio-ops` with
`default-features = false, features = ["dynamic-linking"]`, which in
turn required a prep refactor: `dynamic-loading` / `dynamic-linking`
became per-crate opt-in feature flags across `kaio-runtime`, `kaio`,
and `kaio-ops` (unreleased, will land in kaio 0.3.1).

### AD2 — Explicit `Arc<KaioDevice>` argument

No automatic construction of a KAIO device from a candle `CudaDevice`.
The user creates both. Per P4 pre-flight: both wrappers retain the same
CUDA primary context via `cuDevicePrimaryCtxRetain` at the driver level,
so an ordinal-equality check is the right enforcement — there is no
cache state to maintain, no dependency on cudarc's interning across
versions, and no hidden lifetime surprises.

### AD3 — `repr(transparent)` read-only newtype bridge

`GpuBuffer<T>` is `#[repr(transparent)]` over `CudaSlice<T>`.
`buffer_ref_from_slice_readonly` casts `&CudaSlice<T>` to `&GpuBuffer<T>`
for kernel input. Soundness rests on three pillars:

1. Compile-time size + alignment asserts via `static_assertions` in
   `kaio-runtime::buffer::repr_soundness`.
2. Aliasing contract in the function's docstring: only read-only
   inputs may be bridged via this path; outputs are allocated fresh
   inside the bridge and handed back by value.
3. Lifetime invariant: the cast `&GpuBuffer<T>` never escapes the
   `cuda_fwd` scope, and the post-launch `cuCtxSynchronize` (AD5)
   guarantees the kernel has finished before the reference drops.

### AD4 — Input gate: rank-2, contiguous, zero-offset

Every bridge call runs `ensure_rank2_contiguous_zero_offset` on each
input. Rejects higher-rank tensors (with a concrete `.reshape(...)`
hint), non-contiguous layouts from `.t()`, and non-zero storage offsets
from `.narrow(...)` / `.slice(...)`. Call `.contiguous()?` upstream to
compact. Multi-head attention callers flatten `[heads, seq, d]` to
`[heads * seq, d]` or call per-head.

### AD5 — `cuCtxSynchronize` fences per launch

Every bridge call does `candle_dev.synchronize()` immediately before
and immediately after the `kaio-ops` kernel launch. Cost: two context
syncs per bridge call (dominates tiny-shape decode latency; negligible
on prefill shapes). Benefit: the bridge is stream-safe against candle's
internal stream scheduling without reaching into candle's cudarc
internals. Trade-off: CUDA Graph capture regions forbid
`cuCtxSynchronize` and return `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`.
Event-based stream plumbing in 7.4c unblocks Graph usage.

### AD6 — f32 output contract, no CPU fallback

All ops return `DType::F32` matching the `kaio-ops` accumulator. Users
who need f16 for downstream graph continuation cast via
`.to_dtype(DType::F16)?`. `cpu_fwd` returns a loud error rather than
silently routing to `candle.matmul()` — KAIO's value is GPU-specific
PTX and a silent CPU fallback would mask every perf claim the crate
makes.

### AD7 — No-CUDA-feature leg compiles as an empty shell

Without the `cuda` feature, `kaio-candle` compiles with all modules
`#[cfg(feature = "cuda")]`-gated. Consumers who forget the feature see
a "function not found" compile error when they try to call the bridge
(matches candle-core's own opt-in `cuda` pattern) rather than a
`compile_error!` that would break `cargo check --no-default-features`
on a no-CUDA runner. Ship gate #10 exercises this path.

---

## Results

### Correctness — all green

- 4 host tests (bridge shape gate) pass on every host.
- 15 GPU integration tests pass bit-exact on RTX 4090 sm_89.
- Main workspace `fmt` + `clippy -D warnings` + `doc` clean.
- Standalone `kaio-candle`: `fmt` + `clippy --features cuda --all-targets -- -D warnings` +
  `test --lib` + `doc --features cuda` + `doc --no-default-features` +
  `check --no-default-features` + `build --features cuda --examples`
  all clean.
- Ship-gate #14 manual example runs produce non-NaN outputs of expected
  shape for both examples.

### Scope — all D sections landed

- D1 scaffold + kaio-runtime bridge API (`GpuBuffer::from_cuda_slice`
  promoted to `pub`, `into_cuda_slice` added, `repr(transparent)` +
  `repr_soundness` asserts, `KaioDevice::ordinal()`). ✅
- D2 bridge primitives (slice / storage / buffer / ordinal / sync /
  error / rank-2 gate). ✅
- D3 `matmul_tc` + `matmul_tc_async`. ✅
- D4 `matmul_int4`. ✅
- D5 `attention_tc` + `attention_tc_causal`. ✅
- D6a README + runnable examples + Cargo.toml polish +
  `cargo doc` clean on both feature paths. ✅
- D6b weekly candle-HEAD CI workflow + README badge +
  `PUBLISH_ORDER.md` update for the kaio-candle publish lane. ✅
- D7 this doc + `PHASE_7_LOG.md` row + `CHANGELOG.md` entry + root
  `README.md` update. ✅

### Performance note

Bench numbers KAIO publishes for `matmul_tc_async` (92.5% cuBLAS sgemm
at 4096² on sm_89) and the other kernels are measured via direct
`kaio-ops` calls, not through the bridge. Each bridge call issues two
`cuCtxSynchronize` fences around the kernel launch (AD5). Tiny-shape
decode is dominated by this fence overhead; prefill shapes less so.
Event-based stream plumbing in 7.4c will lift the dominant fences and
close the bridge-vs-direct gap on decode shapes.

---

## Follow-ups for future sprints

- **7.4b (same branch)** — `matmul_int8`, `qkv_project_int8`,
  `qkv_project_int4` bridge bindings. Same pattern, incremental.
- **7.4c — backward + stream plumbing.** Backward `CustomOp` support
  for training integrations; event-based stream handoff so the bridge
  stops issuing `cuCtxSynchronize` and becomes CUDA Graph-capture
  compatible.
- **kaio 0.3.1 patch release.** `dynamic-linking` feature added
  post-0.3.0 across `kaio-runtime` / `kaio` / `kaio-ops`; needs a patch
  release before `kaio-candle 0.1.0` can resolve against crates.io.
- **FlashAttention-TC.** Lifts the `seq_k ≤ 384` shared-memory cap on
  `attention_tc` and `attention_tc_causal`.
