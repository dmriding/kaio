# Sprint 1.6 — Runtime Device + Buffers

**Commit:** pending
**Status:** Complete

## Context

Sprint 1.6 crosses from Layer 1 (pyros-core / PTX codegen) into Layer 2
(pyros-runtime / CUDA runtime). It wraps cudarc 0.19's driver API into
PYROS-specific types: `PyrosDevice`, `GpuBuffer<T>`, `DeviceInfo`, and
`PyrosError`.

## Decisions

### PyrosDevice field types — Arc vs owned

**Context:** cudarc's `CudaContext::new()` returns `Arc<CudaContext>`.
`default_stream()` takes `&Arc<Self>` and returns `Arc<CudaStream>`.
Should PyrosDevice store the Arcs directly or unwrap them?

**Decision:** Store `Arc<CudaContext>` and `Arc<CudaStream>` directly.
cudarc's API is designed around Arc — `CudaSlice` internally clones the
Arc to ensure the context outlives the allocation. Fighting this design
by unwrapping would create lifetime issues. The Arcs are cheap (pointer-sized).

### GpuBuffer::to_host signature — buffer method vs device method

**Context:** Transferring data from GPU to host requires a CUDA stream.
Two options:
1. `buf.to_host(&device)` — buffer knows its data, borrows device for stream
2. `device.read_back(&buf)` — device owns the operation

**Decision:** `buf.to_host(&device)`. More natural to call on the buffer
("give me your data") than on the device ("read this buffer"). The
borrow of device is lightweight. Alternative would store Arc<CudaStream>
in every buffer, adding overhead to every allocation for a method called
infrequently.

### T bounds — GpuType or cudarc's traits?

**Context:** Should `alloc_from<T>` require `T: GpuType` (PYROS trait)
or `T: DeviceRepr` (cudarc trait)?

**Discovery:** cudarc's methods have specific trait requirements:
- `clone_htod`: `T: DeviceRepr`
- `alloc_zeros`: `T: DeviceRepr + ValidAsZeroBits`
- `clone_dtoh`: `T: DeviceRepr`

**Decision:** Use cudarc's trait bounds directly. `alloc_from` requires
`DeviceRepr`, `alloc_zeros` requires `DeviceRepr + ValidAsZeroBits`.
The GpuType bound is not needed for allocation — it becomes relevant in
Sprint 1.7 for launch type safety. All GpuType types satisfy cudarc's
bounds in practice, but the Rust compiler can't prove this automatically.

### DeviceInfo — minimal vs full

**Context:** docs/implementation.md specs DeviceInfo with name, compute
capability, total memory, SM count, max threads, max shared memory, warp
size. Sprint 1.6 only needs enough to validate the GPU.

**Decision:** Three fields only: name, compute_capability, total_memory.
Additional fields (SM count, max threads, shared memory, warp size) deferred
to Phase 3/4 when occupancy calculations matter. Added a doc comment noting
planned fields.

### DeviceInfo — unsafe attribute queries

**Context:** cudarc's `device::get_attribute()` and `device::total_mem()`
are unsafe functions (raw FFI into the CUDA driver). We need them for
compute capability.

**Decision:** Wrap in unsafe blocks with SAFETY comments. The device handle
is valid (obtained from `device::get(ordinal)`), and the attribute queries
are read-only with no aliasing concerns. This is acceptable `unsafe` usage
in a library — the invariant is simple and well-documented.

### Debug impl — derive vs manual

**Context:** `PyrosDevice` contains `Arc<CudaContext>` and `Arc<CudaStream>`
which may not implement `Debug`. `#[derive(Debug)]` would fail.

**Decision:** Manual `Debug` impl printing just the ordinal number. The
internal cudarc handles are opaque — showing memory addresses isn't useful.
The ordinal is the meaningful identifier.

### context() accessor — allow dead_code

**Context:** `PyrosDevice::context()` is pre-wired for Sprint 1.7's
`load_ptx` method but isn't called yet. clippy with `-D warnings` would
error on this.

**Decision:** `#[allow(dead_code)]` with a comment noting it's used in
Sprint 1.7. Removing and re-adding methods between sprints creates
unnecessary churn. Pre-wiring with an allow attribute is cleaner.

## Scope

**In:** PyrosError (5 variants), PyrosDevice (new, info, alloc_from,
alloc_zeros), GpuBuffer (len, is_empty, inner, inner_mut, to_host),
DeviceInfo (name, compute_capability, total_memory), Result type alias.
7 GPU-gated tests.

**Out:** PTX loading, kernel launch, multi-device, async streams, GpuType
bounds on allocation.

## Results

Completed with three compile-time fixes:
1. `device::get_attribute()` is unsafe — added unsafe blocks with SAFETY comments
2. `PyrosDevice` needed Debug for test assertions — added manual impl
3. `context()` unused warning — added `#[allow(dead_code)]`

cudarc `to_host` trait bounds: `T: DeviceRepr + Default + Clone + Unpin`
was the initial guess. Actual bound needed was just `DeviceRepr` (cudarc
0.19's `clone_dtoh` is generic over `Src: DevicePtr<T>`).

**Quality gates:**
- `cargo build --workspace`: clean (no GPU needed)
- `cargo test -p pyros-runtime -- --ignored`: **7 passed** on RTX 4090
- `cargo fmt --all --check`: clean
- `cargo clippy --workspace --all-targets -- -D warnings`: clean

**GPU test note:** Standard `cargo test --workspace` skips GPU tests
(they're `#[ignore]`). Full quality gate on GPU machines:
`cargo test --workspace && cargo test -p pyros-runtime -- --ignored`

**Files created:** 3 (error.rs, device.rs, buffer.rs)
**Files modified:** 1 (lib.rs — complete rewrite from stub)
