# Sprint 7.4c — Event-based stream synchronization

**Status:** ✅ Complete
**Branch:** `phase7-wrap`
**Release target:** bundled into Phase 7 aggregate release.

---

## Context

7.4a + 7.4b shipped 8 forward ops in `kaio-candle`. Every bridge call issued two `cuCtxSynchronize` fences for stream safety — blocking the entire CUDA context per call. This dominated decode-shape latency and was banned during CUDA Graph capture.

## What shipped

Replaced `cuCtxSynchronize` with event-based cross-stream synchronization via cudarc's `CudaStream::join()` (`cuEventRecord` + `cuStreamWaitEvent` internally). GPU-side only, no CPU blocking, CUDA Graph capture-compatible at the primitive level.

### Changes

- `bridge::sync_before_launch` + `sync_after_launch` — internals replaced, signature gains `&KaioDevice` parameter (both streams needed for cross-stream event sync).
- 7 call sites updated across all op modules (5 CustomOp + 2 direct-call).
- `bridge::driver_err` helper added for cudarc `DriverError` → candle `Error` conversion.
- API-path smoke test validates event/join calls on real hardware.
- Known-limitations docs downgraded from "CUDA Graph incompatible" to "partially unblocked" — default-stream capture is still banned by CUDA itself.
- Bench-caveat docs updated to note event-allocation micro-overhead replacing the heavier context-sync.

### What didn't change

- Zero API changes to the public surface. All 8 ops have the same signatures.
- Zero new ops.
- All 32 existing bit-exact GPU tests pass unchanged — the sync mechanism is invisible to kernel correctness.

---

## Follow-ups

- **7.4d** — matmul_tc + matmul_tc_async backward (analytical, forward kernel reuse). ✅ Complete.
- **Event caching** — `record_event(None)` allocates a transient `CudaEvent` per call. Caching handles inside CustomOp structs eliminates this overhead. Future optimization.
- **Non-default stream verification** — full CUDA Graph capture requires both candle and KAIO on non-default streams. Testing this needs a graph-capture harness.
- **kaio 0.3.1 patch + kaio-candle 0.1.0 publish** — `dynamic-linking` feature needed on crates.io.
- **Phase 7 close** — `phase7-wrap → phase7-ship → main` merge, v0.4.0 aggregate release.
