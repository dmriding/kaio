# Sprint 3.4 — Barrier + Shuffle Built-in Functions

**Status:** Complete
**Commit:** `6fde908`
**Date:** 2026-04-11
**Depends on:** Sprint 3.2 (kaio-core instructions), Sprint 3.3 (shared mem DSL)

## Context

Adds `bar_sync()` and `shfl_sync_down/up/bfly()` as callable built-in
functions in the `#[gpu_kernel]` macro. Also delivers the GPU E2E test
for shared memory that was deferred from Sprint 3.3.

## Decisions

### bar_sync() returns dummy register

**Context:** `lower_builtin` always returns `(Ident, KernelType, TokenStream)`.
`bar_sync()` is a side-effect-only statement.

**Decision:** Allocate an unused U32 register. Callers of bare expression
statements discard the result. Unused PTX registers are harmless.

### shfl_sync width hardcoded to 32

**Context:** The `c` operand is a compile-time literal. Extracting width
from a lowered register is not possible without parser changes.

**Decision:** Accept 3 args, hardcode `c` for full-warp (width=32):
down=31, up=0, bfly=31. All validated via ptxas in Sprint 3.2. Sub-warp
shuffles deferred.

### Shared memory addressing simplified

**Context:** Initial Sprint 3.3 implementation used `mov.u32 %r, sdata;`
to load the named shared allocation's base address. This caused
CUDA_ERROR_ILLEGAL_ADDRESS at runtime.

**Decision:** Simplified to direct byte-offset addressing: the address for
`sdata[i]` is just `i * sizeof(T)`. Works correctly for a single shared
allocation (base at offset 0). Multi-allocation addressing (Sprint 3.5)
will need revisiting — likely using cumulative byte offsets tracked at
codegen time rather than PTX named references.

### GPU test needs bounds check

**Context:** The shared memory GPU test initially crashed with
CUDA_ERROR_ILLEGAL_ADDRESS. Root cause: no `if tid < n` guard, so threads
in the block with index >= n could access out-of-bounds global memory.

**Decision:** Added bounds check to the test kernel. This is the standard
GPU pattern — always guard memory accesses with a bounds check.

## Scope

**In:**
- `bar_sync()` built-in function
- `shfl_sync_down/up/bfly(val, delta, width)` built-in functions
- `lower_shfl_sync()` helper reducing duplication across 3 variants
- Updated error message listing all available builtins
- IDE stubs for all 4 new functions
- GPU E2E: shared memory write → barrier → read → verify (2 tests)
- 7 new host tests in builtins.rs

**Out:**
- Sub-warp shuffles (width < 32)
- Cvt rounding modifier fix (Sprint 3.8)

## Results

Completed.

**Bug found and fixed:** Shared memory GPU test crashed without bounds
check (`if tid < n`). Threads beyond `n` accessed out-of-bounds global
memory. Standard GPU pattern — always guard with bounds check.

**Shared memory addressing simplified:** Removed `mov.u32 %r, sdata` base
address loading (caused runtime errors). Byte offset alone works for single
allocations. `Operand::SharedAddr` remains in kaio-core for future use.

**Files created:** 1
- `kaio/tests/shared_mem_macro.rs` (2 GPU E2E tests)

**Files modified:** 3
- `kaio-macros/src/lower/builtins.rs` (bar_sync, lower_shfl_sync helper,
  3 shfl dispatch arms, updated error message, 7 new tests)
- `kaio-macros/src/lower/memory.rs` (simplified compute_shared_address)
- `kaio/src/gpu_builtins.rs` (bar_sync, shfl_sync_* IDE stubs)
- `kaio/tests/compile_fail/cf04_unknown_call.stderr` (updated error msg)

**Tests:** 198 total (78 kaio-core + 113 kaio-macros + 7 integration),
all passing. 12 GPU tests (10 prior + 2 new), all passing on RTX 4090.

**Quality gates:** `cargo fmt --check` clean, `cargo clippy -- -D warnings`
zero warnings, all tests pass.
