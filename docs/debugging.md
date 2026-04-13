# Debugging KAIO kernels

This is the single entry point for diagnosing problems with KAIO code. If you hit a launch error, a wrong answer, a NaN, or a performance surprise, start here.

> **Note on maintenance:** this document aggregates authoritative references elsewhere in the docs. If a detail here conflicts with [`docs/performance.md`](performance.md) (PTX inspection) or [`docs/testing-strategy.md`](testing-strategy.md) (test patterns), fix the authoritative ref first, then update the link or summary here.

## Why GPU debugging differs from CPU

Three facts shape everything below:

1. **GPU launches are asynchronous.** `kernel::launch(...)?` returns before the kernel finishes (often before it even starts). An error reported by a later call (like `buf.to_host(&device)?`) may actually have come from the preceding launch.
2. **There's no line-by-line debugger.** No `gdb`, no breakpoints inside a kernel. You inspect kernels through PTX dumps, CPU reference comparison, and `compute-sanitizer`.
3. **Silent corruption is the common failure mode.** A kernel with an out-of-bounds read produces a wrong number, not a crash. Catching wrong output is entirely on you — which is why KAIO's own test suite is built around CPU reference comparison with bit-exact or tolerance-bounded assertions.

KAIO's position: make the *compilable* kernel correct-by-construction whenever possible (type system, `PtxModule::validate()`), and give you the tools to verify the rest.

## Troubleshooting flowchart

When something goes wrong, walk this path:

```
1. Does it compile?
   ├─ No  → see "Compile errors" below
   └─ Yes ↓

2. Does kernel::launch(...) succeed?
   ├─ No, returns KaioError::Validation → see "Validation errors"
   ├─ No, returns KaioError::Driver → see "Driver / launch errors"
   └─ Yes ↓

3. Does buf.to_host(...) succeed?
   ├─ No (async launch error surfaces here) → see "Async launch errors"
   └─ Yes ↓

4. Is the output correct?
   ├─ No, NaN/Inf          → see "Silent NaN" below
   ├─ No, wrong-but-finite → see "Correctness verification"
   └─ Yes ↓

5. Is performance what you expect?
   ├─ No, in debug build   → rebuild with --release (see "Debug-build note")
   ├─ No, in release build → see "Performance diagnosis"
   └─ Yes — you're done.
```

## Environment variables

KAIO exposes a small set of env vars for inspection and diagnosis. All are opt-in — unset defaults to normal operation.

| Variable | Effect | Reference |
|---|---|---|
| `KAIO_DUMP_PTX=1` | Write generated PTX for each kernel to `OUT_DIR` (or current dir) | [performance.md § PTX Inspection Tools](performance.md) |
| `KAIO_PTX_STATS=1` | Print structural PTX statistics (instruction counts, register use, shared-mem bytes) on first launch | [performance.md § PTX Inspection Tools](performance.md) |
| `KAIO_PTX_ANNOTATE=1` | Add `// <rust-source-construct>` comments to emitted PTX, mapping DSL back to instructions | [performance.md § PTX Inspection Tools](performance.md) |
| `KAIO_SM_TARGET=sm_XX` | Override the compute capability used for PTX emission (default: derived from the device) | [kaio-core/src/ir/module.rs](../kaio-core/src/ir/module.rs) |
| `KAIO_TUNE_CACHE=<path>` | Auto-tuner cache JSON file location (for `matmul_auto_tc` and similar) | [kaio-ops/src/tuner.rs](../kaio-ops/src/tuner.rs) |
| `KAIO_SUPPRESS_DEBUG_WARNING=1` | Suppress the one-time "debug build" note that `KaioDevice::new` emits on debug binaries | This doc — see "Debug-build note" |

Stack them freely:

```sh
KAIO_DUMP_PTX=1 KAIO_PTX_ANNOTATE=1 cargo test -p kaio --test shared_mem_macro -- --ignored --nocapture
```

## Compile errors

`#[gpu_kernel]` rejects syntax it doesn't support at macro-expansion time. The error message points at the offending source location (in most cases — see [Sprint 7.0.5 finding](development/sprints/phase7/sprint_7_0_5.md) on the span audit for caveats).

Common rejected constructs:
- Closures, traits, generics, method calls, string operations
- `return` statements (structure the kernel so control flows to the end)
- `match` on most patterns (use `if`/`else if` instead; full `match` is post-Phase 7)
- `loop {}` (use bounded `for` or `while`)
- Unsupported numeric suffixes in literals (e.g. `42u8` pre-Phase 7.1)

See [success-criteria.md § Phase 2 CF tests](success-criteria.md) for the full compile-fail matrix and [`kaio/tests/compile_fail/`](../kaio/tests/compile_fail) for the trybuild fixtures.

## Validation errors

`KaioError::Validation` fires when `PtxModule::validate()` rejects a kernel before handing PTX to the driver. Since Sprint 6.10, user kernels go through this gate — so SM-version mismatches (e.g. using `mma.sync` on a pre-Ampere GPU) surface here with a readable message instead of as an opaque ptxas error.

Most common cause: the kernel uses `mma.sync` or `cp.async` but `device.info()?.compute_capability` is below `sm_80`. Fix: run on Ampere+, or use the scalar fallback path (`matmul` instead of `matmul_tc`).

## Driver / launch errors

If `kernel::launch(...)?` returns `KaioError::Driver` (`CUDA_ERROR_*`), the failure happened in the driver before the kernel started. Common causes:

- **`invalid argument`** — launch config mismatch (grid/block dims exceed device limits, shared memory over `48 KB` default). Use `device.info()?` to check device limits.
- **`invalid device pointer`** — a buffer was freed or came from a different context. Confirm buffers are allocated via `device.alloc_*` on the same device you launch on.
- **`not enough resources`** — per-thread register pressure dropped CTA occupancy to zero, or shared-memory request exceeds the per-SM pool. Run with `KAIO_PTX_STATS=1` to see PTX-level register usage. See [performance.md § interpreting PTX stats](performance.md) for the register-pressure → occupancy link.

## Async launch errors

A launch can succeed synchronously and still fail asynchronously. The failure surfaces on the next synchronizing call — typically `buf.to_host(&device)?` or the next `launch(...)?`. If `to_host` fails but the preceding `launch` didn't, the kernel itself faulted. See "Silent NaN" and the `compute-sanitizer` section below.

## Debug-build note

On first `KaioDevice::new(...)` in a debug build, KAIO emits:

```
[kaio] Note: debug build — GPU kernel performance is ~10-20x slower than --release. Use `cargo run --release` / `cargo test --release` for representative performance numbers. Correctness is unaffected. Set KAIO_SUPPRESS_DEBUG_WARNING=1 to silence.
```

This is strictly a performance note — **correctness is unaffected** by debug mode. If you're running `cargo test` in debug to verify kernel output, the output is trustworthy. The note exists because the most common adoption failure for GPU frameworks is users benchmarking in debug, seeing terrible numbers, and leaving. KAIO tries to intercept that mistake before it happens.

Release builds emit nothing. CI / test harnesses that intentionally run in debug can set `KAIO_SUPPRESS_DEBUG_WARNING=1` to silence.

## Silent NaN / wrong output

This is the hardest failure mode in GPU programming. The kernel ran, `to_host` succeeded, and the output is wrong.

Walk this list in order:

**1. Compare against a CPU reference.** This is what every test in `kaio/tests/` and `kaio-ops/tests/` does. Write the same math in `for` loops on the host, compare element-by-element. For simple math you can assert bit-exact equality via `.to_bits() == .to_bits()`. For floating-point with any accumulation, use a tolerance.

**2. Use small inputs first.** If your matmul is wrong at 4096×4096, test 16×16 with known values. Silent corruption usually reproduces at tiny shapes. Bisect: `16×16 → 64×64 → 256×256 → ...`

**3. Run under `compute-sanitizer`.** See the dedicated section below. OOB reads, race conditions, and uninitialized reads are the usual culprits for silent wrong answers. `compute-sanitizer` catches all three.

**4. Dump PTX.** `KAIO_DUMP_PTX=1` writes the emitted PTX next to your binary. Read it — not all of it, just the inner loop — and verify it matches your mental model. Is the address calculation right? Is the `ld.shared.f32` offset what you expect? Unexpected PTX is a codegen bug; expected PTX means the bug is in your kernel logic or launch config.

**5. Check indices for OOB.** A thread computing `idx = tid + blockIdx.x * blockDim.x` will have `idx >= n` for the last partial block. Every array access needs `if idx < n { ... }` guard (or rely on `&&` short-circuit since Sprint 7.0: `if idx < n && arr[idx] > 0 { ... }`).

## Correctness verification

### Choose a tolerance

Bit-exact comparison (`a.to_bits() == b.to_bits()`) works for operations that don't accumulate float error: integer math, single-step fp operations with exactly-representable inputs, bit manipulation. KAIO's `mma_sync_fragment_gate` test is bit-exact because the expected values are integers representable in f32.

Tolerance-based comparison is needed for:
- Accumulated floating-point (matmul's inner-product reduction, softmax, reductions)
- Operations involving transcendentals (`exp`, `log`, `sin`)
- Mixed-precision paths (fp16 × fp16 → fp32 accumulation)

A practical tolerance for matmul at size `N`:

```rust
let rel_tol = 1e-4;
let abs_tol = 1e-4 * (N as f32).sqrt();
let error = (got - expected).abs();
let allowed = abs_tol + rel_tol * expected.abs();
assert!(error < allowed, "mismatch at ...");
```

The `sqrt(N)` in `abs_tol` reflects that error grows with the number of accumulation steps. For attention with long sequences, scale similarly.

See [testing-strategy.md § numerical testing](testing-strategy.md) for KAIO's own tolerance patterns.

### Bit-exact vs tolerance — when each applies

| Use bit-exact when | Use tolerance when |
|---|---|
| Integer arithmetic (any size) | Any fp accumulation |
| Bit manipulation (`&`, `\|`, shifts) | `exp`, `log`, trig, `sqrt` |
| Integer-valued fp (e.g. `2.0 * 3.0 = 6.0`) | fp16 intermediates |
| Single-step fp with exact representation | Any kernel where you can't compute the expected rounding |

If you're not sure: bit-exact first. Only widen the bound when you can explain the magnitude.

## Running under compute-sanitizer

`compute-sanitizer` (formerly `cuda-memcheck`) ships with the CUDA toolkit. It catches:
- Out-of-bounds device-memory reads and writes
- Race conditions on shared memory
- Uninitialized device-memory reads
- Misaligned accesses

To run KAIO tests under it:

```sh
# Windows (adjust path to your CUDA toolkit install)
compute-sanitizer --tool memcheck cargo test -p kaio --test bitops_macro -- --ignored bitops_all_smoke

# Linux
compute-sanitizer --tool memcheck cargo test -p kaio --test bitops_macro -- --ignored bitops_all_smoke
```

Key flags:
- `--tool memcheck` (default) — OOB + uninit reads
- `--tool racecheck` — shared-memory races
- `--tool initcheck` — uninitialized device-memory reads only
- `--tool synccheck` — `__syncthreads` / `bar.sync` correctness

Example output from an OOB read:

```
========= Invalid __global__ read of size 4 bytes
=========     at 0x90 in my_kernel
=========     by thread (31,0,0) in block (0,0,0)
=========     Address 0x7fc0000400 is out of bounds
```

The message includes thread index — invaluable for diagnosing a bug that only fires on the edge thread of the last partial block.

**KAIO-specific notes:**
- Ignore-gated GPU tests are the canonical target. Run with `-- --ignored <test-name>` as above.
- `compute-sanitizer` significantly slows down execution (expect 10-100× slower). Use on small shapes first, scale up only if needed to reproduce.

## Interpreting PTX stats

`KAIO_PTX_STATS=1` reports the structural shape of emitted PTX. The critical numbers for debugging:

- **Total instructions** — rough complexity indicator. Sudden jumps between kernel variants indicate unexpected codegen.
- **Registers (r32/r64/f32/f64/pred)** — PTX-level register usage. Not final SASS usage, but directly indicative of register pressure. If you see > 64 per thread on sm_89 with 128-thread CTAs, you may be losing occupancy (< 2 CTAs/SM).
- **Shared mem bytes** — per-CTA shared-memory request. Exceeding 48 KB requires opting in via launch config or triggers "not enough resources" launch failures.
- **bar.sync count** — sanity check: every shared-memory write that other threads read must be followed by `bar.sync` (in the DSL: `bar_sync()`).

For the authoritative reference including the raw output format, see [performance.md § PTX inspection tools](performance.md).

**What PTX stats are NOT:** runtime profiling data. Final SASS register usage, occupancy, and instruction scheduling are determined by the driver's PTX → SASS compiler. For runtime profiling use `nvprof` / `ncu` (NVIDIA Nsight Compute), not KAIO's stats.

## Performance diagnosis

If the kernel is correct but slow in release builds:

1. **Rebuild in release.** Confirm you're actually benchmarking release. The debug-build note exists for a reason.
2. **Check the benchmark methodology.** See [performance.md § methodology](performance.md) — warmup iterations, median of timed iterations, exclude first launch (module load).
3. **Run `KAIO_PTX_STATS=1`.** Compare register counts, instruction mix, shared-memory usage vs a reference kernel (e.g. KAIO's own `matmul_tc` for matmul-shaped workloads).
4. **Use Nsight Compute.** `ncu` is the authoritative profiler. KAIO's PTX stats are a structural approximation, not a replacement.

## Common errors quick reference

| Symptom | Likely cause | First step |
|---|---|---|
| "not enough resources" on launch | Register pressure → occupancy cliff | `KAIO_PTX_STATS=1`, check register counts |
| `KaioError::Validation` "requires sm_XX+" | Kernel uses Ampere feature on older GPU | Use scalar fallback; check `device.info()?.compute_capability` |
| Output is NaN on first element only | Uninitialized shared memory; missing `bar_sync()` before read | Dump PTX, check `bar.sync` placement; run racecheck |
| Output wrong on last partial block only | OOB access on edge threads | Add `if idx < n` guard; run memcheck |
| Output drifts as N grows | fp accumulation error, tolerance too tight | Widen tolerance ∝ √N for matmul-shaped |
| Terrible perf in `cargo run` | Debug build | `cargo run --release` |
| `matmul_auto_tc` picks wrong variant | Stale auto-tune cache | Delete `KAIO_TUNE_CACHE` file or set to fresh path |

---

## Further reading

- [`docs/performance.md`](performance.md) — PTX inspection deep dive, benchmark methodology
- [`docs/testing-strategy.md`](testing-strategy.md) — KAIO's own test patterns (CPU reference, tolerance, trybuild compile-fail)
- [`docs/implementation.md`](implementation.md) — error types, platform specifics
- [`docs/limitations.md`](limitations.md) — what KAIO does not do and why
- [`docs/development/sprints/phase7/sprint_7_0_5.md`](development/sprints/phase7/sprint_7_0_5.md) — Sprint 7.0.5 span-audit finding (notes on where error messages currently point)
