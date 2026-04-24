# Phase 8 Master Plan — Python Bindings + Pointer Syntax

**Status:** Scaffold complete; further Python work user-demand-gated.
8.0 pointer syntax, 8.0.5 bench coverage, and 8.1 PyO3 scaffold all
landed on `main`. 8.2 (broader op exposure), 8.3 (PyTorch
cross-validation), and 8.4 (PyPI + CI matrix) are deliberately
**unscheduled** — the scaffold proves the path works; the ongoing
maintenance cost of keeping a Python surface in lockstep with a
fast-moving Rust API, with zero concrete Python users at time of
writing, would slow the Rust core without a return. See "Scope
decision" below.
**Depends on:** Phase 7 complete (v0.4.0, 2026-04-18)
**Per-sprint logs:** see the `sprint_8_*.md` files in this directory
and the per-sprint status row in [`PHASE_8_LOG.md`](PHASE_8_LOG.md)
for commit hashes and shipped outcomes.
**Sprint order (as planned + actually shipped):** 8.0 → 8.0.5 → 8.1 →
8.2 → 8.3 → 8.4. 8.0 was the absorbed Phase 8.5 from the original
roadmap — pulled forward so the syntax change landed before PyO3
widened the public-facing surface; see
[`sprint_8_0.md`](sprint_8_0.md) for the absorption rationale.

> **Note:** This document is the long-form Phase 8 planning reference.
> It captures architecture, risks, and sprint sequencing as they were
> understood at planning time. Authoritative state for shipped work
> lives in the per-sprint log files above and in
> [`docs/phases.md`](../../../phases.md). If this plan disagrees with
> a sprint log, the sprint log wins.

## Goal

Put KAIO kernels in front of Python users without making them learn
Rust, install a Rust toolchain, or leave their existing workflow.
`pip install kaio` should work on Windows + Linux, support Python ≥
3.10, and expose `kaio.matmul_tc(a, b)` / `kaio.attention(q, k, v)` /
the quantized matmul family as ops that accept NumPy arrays and
return NumPy arrays (or Torch tensors when Torch is imported).

Python is where most ML practitioners live, and Triton — the closest
Rust-ecosystem alternative for custom-kernel authoring — is
Linux-only. KAIO's Windows-capable story is a differentiator that
only pays out if Python users can reach it. Phase 8 closes that gap.

## Critical Architectural Constraints

**1. The Python crate is standalone, not a workspace member.** Same
reason `kaio-candle` is standalone: `cudarc` exposes
`dynamic-loading` and `dynamic-linking` as additive-but-mutually-
incompatible features. A workspace-member Python crate would union-
merge those features across every dependent, re-opening the exact
feature-collision class of bug Phase 7 solved for candle. `kaio-py`
lives next to the workspace and pulls `kaio` / `kaio-ops` as path
dependencies with `default-features = false` + explicit feature
selection.

**2. The Python wrapper is thin — it does not re-design the API.**
The Rust `kaio-ops` surface (host `fn matmul_tc(device, a, b, c, m,
n, k)`) maps one-to-one to the Python surface
(`kaio.matmul_tc(a, b)`; shape inferred from NumPy, device implicit
or explicit). Python-first ergonomics are for Phase 8.2+ as they
surface from actual user feedback. 8.1 lands the smallest viable
shim.

**3. ABI compatibility is abi3, pinned.** Wheels build once per
architecture + platform and run across Python 3.10 → 3.12+ without
per-version rebuilds. `maturin --features pyo3/abi3-py310` is the
reference build. This is a non-negotiable for the "pip install works"
story — per-Python-version wheels multiply CI cost and packaging
surface unnecessarily.

**4. The GIL must be released during GPU kernel execution.** Python
blocks threading by default; a kernel launch that takes 60 µs and
holds the GIL serializes every other Python thread. PyO3's
`Python::allow_threads` wraps the `kaio_ops` call so concurrent
Python threads (or Python's own interpreter work) can proceed while
the kernel runs. Applies to every op, not just the slow ones — users
will want to overlap CPU work with GPU work in 8.3+.

**5. Device memory ownership crosses the FFI boundary via
reference-counted handles, not raw pointers.** Python code can hold
a `kaio.Tensor` (Python-side Py class wrapping `Arc<GpuBuffer<T>>`)
across multiple kernel calls, drops it when reference count hits
zero, and never sees a raw CUDA pointer. Manual `.free()` is not
required; Python's garbage collector drives it.

The `kaio.Device` wrapper must also be `Arc`-held, and every
`kaio.Tensor` must hold a clone of that `Arc` — **not a borrowed
reference.** Without this, Python GC can drop the Device before the
Tensors that reference its CUDA context, producing a use-after-free
at the driver level. The `kaio-candle` bridge already follows this
pattern (`Arc<KaioDevice>` held by every `CustomOp` input); Sprint
8.1 inherits it. This is the one ownership invariant that cannot
be compromised — document it at the Python API level, enforce it
through the type system, and never let a Tensor outlive its Device
even transitively.

## Key Architectural Decisions

### 1. Pointer syntax (Sprint 8.0 — complete)

**Absorbed from the original Phase 8.5.** `#[gpu_kernel]` accepts
`*const [T]` / `*mut [T]` as primary kernel-parameter syntax;
`&[T]` / `&mut [T]` retained as permanent sugar (identical PTX
lowering). Resolves the DSL-vs-compiled-Rust aliasing-contract
mismatch raised in [#13](https://github.com/dmriding/kaio/issues/13).
Shipped in v0.4.1 with parser-only extension — zero IR / codegen /
runtime change. See [`sprint_8_0.md`](sprint_8_0.md) for the full
outline and [`../../rfcs/rfc-0001-pointer-syntax.md`](../../rfcs/rfc-0001-pointer-syntax.md)
for the design.

Pulled forward from the original schedule so the syntax migration
landed before PyO3 widened KAIO's documented public surface — every
Python example that shows `#[gpu_kernel]` cross-references uses the
new form.

### 2. Bench coverage extension (Sprint 8.0.5 — complete)

`cargo xtask bench` now drives seven benchmark harnesses covering
the shipped high-level kernel families plus the showcase kernels:
three matmul variants, `qkv_project` (fused INT4 vs 3× standalone +
INT8 absolute TOPS), `attention_tc` (short-seq self-attention, TC
path), `attention_flash` (long-seq self-attention, online softmax),
and a unified `norm_activation_bench` covering RMSNorm / LayerNorm /
softmax (reductions at a launch-overhead-reference `n=256`) plus
SiLU-gate / exact GELU / fast GELU (elementwise sweep to 4M
elements). See [`sprint_8_0_5.md`](sprint_8_0_5.md) for the full
outline; published worst/median/best tables live in
[`docs/performance.md`](../../../performance.md).

Additive measurement coverage, no version bump. Closes the "KAIO
benches every kernel it ships" credibility gap flagged in
`performance.md` roadmap section.

### 3. PyO3 scaffold (Sprint 8.1 — complete)

Standalone `kaio-py/` crate at repo root (not a workspace member,
same rationale as `kaio-candle`). PyO3 0.28 + rust-numpy 0.28 +
maturin ≥ 1.12 + abi3-py310. Exposes `kaio.Device`,
`kaio.Tensor` (NumPy roundtrip for f16 / f32 with C-contiguity
enforcement), `kaio.KaioError` (single exception class; subclasses
deferred), and `kaio.matmul_tc` as the end-to-end smoke kernel.
`examples/hello.py` runs `Device → Tensor → kernel → NumPy` on a
fresh Python install, validating correctness against a NumPy f32
reference.

See [`sprint_8_1.md`](sprint_8_1.md) for the full post-delivery
outline.

### 4. Scope decision (post-8.1): further Python work is user-demand-gated

After shipping the 8.1 scaffold, the decision was made not to
proceed with 8.2 → 8.4 on a scheduled cadence. Rationale:

- **No concrete Python user at time of writing.** The long-term
  roadmap's priority hierarchy puts "user-reported pain" at the
  top; no pain has been reported from Python yet. 8.2+ would be
  speculative build-out rather than demand-response.
- **Maintenance multiplier on a solo-maintainer project.** Every
  Rust op added after Phase 9+ would need wrapping, dtype
  handling, NumPy interop verification, error-path tests, README
  updates, and a wheel republish. For a fast-moving Rust core
  with no Python audience yet, that tax compounds against the
  more important Rust work.
- **The scaffold is the signal.** The 8.1 scaffold is proof that
  KAIO's bindings path works end-to-end — Python users evaluating
  KAIO now see a working `kaio.matmul_tc` demo. That's sufficient
  to communicate intent and possibility without committing to an
  open-ended ongoing deliverable.

The unscheduled sections that remain documented below describe
**what 8.2 / 8.3 / 8.4 would look like** if a user request
triggers them. Treat these as shovel-ready specs, not commitments.

### 5. Unscheduled — core ops exposure (nominal "Sprint 8.2")

If a user request triggers broader op exposure, the intended
surface is:

- `kaio.matmul_tc` + `kaio.matmul_tc_async` (f16 Q × f16 K → f32 out)
- `kaio.matmul_int8` (W8A8)
- `kaio.matmul_int4` (W4A16 GPTQ)
- `kaio.qkv_project_int4` + `kaio.qkv_project_int8` (fused tri-output)
- `kaio.attention_tc` + `kaio.attention_tc_causal` (short-seq TC; `seq_k ≤ 384`)
- `kaio.attention_flash` + `kaio.attention_flash_causal` (online softmax, `d_k ≤ 256`)

Intentionally not exposed: `kaio_ops::softmax` (single-block
`n ≤ 256` trap for Python users), and `rms_norm` / `layer_norm`
which aren't in `kaio_ops` as publics yet — those wait for
multi-block Ops Track versions.

Tensor dtypes that would extend: i8 (INT8 matmul), packed u32
(INT4 matmul). Error taxonomy would split: `KaioError` as base,
`KaioValidationError` / `KaioDeviceError` / `KaioPtxError`
subclasses.

Op chaining without CPU round-trip already works from the 8.1
scaffold — `Tensor → kernel → Tensor` stays on device. Further
ops inherit that pattern.

### 6. Unscheduled — cross-validation + Python benchmarks (nominal "Sprint 8.3")

Would add `tests/` with PyTorch-reference comparison per op,
gated behind `@pytest.mark.gpu_torch` (CI-skipped, maintainer-run
locally — same pattern as the coverage badge and `cargo xtask
bench` numbers). Plus a `bench/` directory measuring FFI dispatch
overhead separately from kernel time. Committed output artifact
(markdown table or JSON) as the regression signal, not a CI gate.

### 7. Unscheduled — PyPI + CI (nominal "Sprint 8.4")

Would add `pip install kaio` via PyPI publish (`maturin upload`),
a GitHub Actions `{windows, ubuntu} × {3.10, 3.11, 3.12}`
CPU-only smoke matrix (wheel build + `import kaio; version` +
type-surface assertion), and a publish runbook.

**Publication stance as of 8.1 landing:** the scaffold lives in
`kaio-py/`, builds cleanly, and is installable today via:

```sh
pip install git+https://github.com/dmriding/kaio.git#subdirectory=kaio-py
```

No PyPI wheel. The first user request that justifies the ops
expansion also justifies the PyPI publish.

## Sprint Breakdown

| Sprint | Scope | Status | Key Deliverable |
|--------|-------|--------|-----------------|
| [8.0](sprint_8_0.md) | Pointer syntax (RFC-0001) — `*mut [T]` / `*const [T]` in `#[gpu_kernel]` | ✅ Complete | Parser-only extension, zero IR / runtime change; `&[T]` / `&mut [T]` retained as permanent sugar; v0.4.1 |
| [8.0.5](sprint_8_0_5.md) | Bench coverage extension | ✅ Complete | Seven bench harnesses under `cargo xtask bench`; new `performance.md` sections for QKV / attention / norm-activation; no API / runtime change |
| [8.1](sprint_8_1.md) | PyO3 scaffold | ✅ Complete | New standalone `kaio-py` crate; `Device` + `Tensor` + `KaioError` + `matmul_tc` smoke kernel; abi3-py310 wheel target; NumPy roundtrip end-to-end |
| 8.2 | Core ops exposure | ⏸️ Unscheduled | Triggered by user request — see §5 for spec |
| 8.3 | Cross-validation + Python bench | ⏸️ Unscheduled | Triggered by user request — see §6 for spec |
| 8.4 | PyPI + CI | ⏸️ Unscheduled | Triggered by user request — see §7 for spec |

## Dependency Graph

```
8.0 (pointer syntax) -> 8.0.5 (bench coverage) -> 8.1 (PyO3 scaffold)
                                                        |
                                                        v
                                                  8.2 (core ops exposure)
                                                        |
                                                        v
                                                  8.3 (cross-validation + bench)
                                                        |
                                                        v
                                                  8.4 (packaging + CI)
```

8.0, 8.0.5, and 8.1 are complete. 8.2 → 8.4 are unscheduled —
triggered by a user request (tracked via the
`python-binding-request` issue template), not by sprint cadence.
If triggered, they'd still run sequentially as originally specified.

Phase 9 (FlashAttention backward + kernel deepening) is a separate
phase tracked in [`phases.md`](../../../phases.md) and is the next
sprint boundary the Rust core picks up. Phase 9 adds PTX kernels
over whatever Python surface exists; since the Python surface is
scaffold-only, Phase 9 work has no cross-phase ripple.

## Out of Scope for Phase 8

Deliberately deferred to later phases or unscoped entirely:

- **Torch / JAX-native tensor interop beyond NumPy.** Phase 8 ships
  NumPy as the canonical interchange dtype; a Python user with
  PyTorch tensors calls `.numpy()` first. Zero-copy torch interop
  via `__cuda_array_interface__` is an 8.5+ candidate, not on the
  critical path.
- **Python-native kernel authoring.** KAIO's authoring surface is
  Rust via `#[gpu_kernel]`. Python is a call-site, not a compile-
  time target. "Write kernels in Python" is a fundamentally
  different project (closer to Triton) and explicitly out of scope.
- **Autograd.** KAIO's autograd story is the Rust `kaio-candle`
  bridge with its `CustomOp::bwd()` entrypoints (Phase 7.4d
  landed `matmul_tc` + `matmul_tc_async` backward). Python
  autograd would re-wire into PyTorch's autograd system via
  `torch.autograd.Function`; that's an 8.5+ item once a user
  actually asks for it.
- **`kaio.Tensor` as a general-purpose tensor abstraction.**
  Staying explicitly thin. The Python `kaio.Tensor` holds a GPU
  buffer with shape + dtype; it does not grow numpy-style slicing,
  broadcasting, or arithmetic operators. Users doing tensor math
  in Python stay in NumPy / PyTorch and call KAIO ops at the
  compute-hot spots.
- **Async / streaming Python API.** Two `kaio` ops back-to-back
  already stay on device (the `Tensor` handle carries the GPU
  buffer; no CPU round-trip between ops — see §4). What's out of
  scope for Phase 8 is explicit **stream ordering** — exposing
  `kaio.Stream` / `kaio.Event` so Python users can launch
  multiple ops on independent streams without a
  `.synchronize()` between them, overlap host work with GPU
  work under explicit control, or capture CUDA graphs from
  Python. That's a Phase 8.5+ ergonomics item driven by user
  patterns, not day-one scaffold work.

## Success Criteria

1. `pip install kaio` on a clean Windows or Linux Python 3.10+
   environment installs and imports without errors. (8.4)
2. `kaio.matmul_tc(a, b)` with NumPy f16 inputs produces correct
   output within documented f16 tolerance vs a PyTorch / NumPy
   reference, running on the system CUDA driver without a CUDA
   toolkit install. (8.1 smoke; 8.3 full cross-validation.)
3. `kaio.attention_flash(q, k, v)` runs on a fresh Python install
   and produces output within f32 tolerance vs a PyTorch reference
   attention. (8.2 correctness; 8.3 documented tolerance.)
4. FFI overhead per op is bounded by a documented number (goal:
   under ~10 µs) separable from kernel execution time in the
   bench output. (8.3)
5. GitHub Actions `{windows-latest, ubuntu-latest} × {3.10, 3.11,
   3.12}` wheel-build + smoke-install + one-matmul passes across
   the full matrix on main. (8.4)

## Key Risks

1. **cudarc feature-union collision.** Same class of bug Phase 7
   hit with `kaio-candle`. Mitigation: standalone `kaio-py` crate
   (not a workspace member), explicit `default-features = false`
   on the `kaio` / `kaio-ops` path deps, feature converge explicit
   and documented in `kaio-py/Cargo.toml`. Confirmed working pattern
   from `kaio-candle`.
2. **abi3 + dtype compat matrix.** NumPy's `np.float16` + packed
   u32 storage for INT4 exposes edge cases; some numpy / abi3
   combinations historically had ugly interactions. Mitigation:
   pin tested numpy versions in the wheel build matrix; documented
   "known working" versions in the README; fail loudly with a
   clear error on unsupported numpy versions rather than silently
   producing wrong output.
3. **GIL contention during kernel execution.** If `allow_threads`
   is not applied uniformly, Python threads stall during kernel
   launches. Mitigation: macro-level or helper-pattern wrapper so
   every op release the GIL by default; integration test that
   launches two concurrent Python threads and asserts both make
   progress during a kernel call.
4. **Wheel size.** A fat wheel with CUDA driver-dependent paths
   can balloon. Mitigation: dynamic-loading cudarc (no CUDA toolkit
   baked in); target wheel under 20 MB per platform; `maturin
   --strip` on release builds.
5. **Maintenance cost of the CI matrix.** 6 cells across 2 OSes ×
   3 Python versions, each rebuilding a Rust compile + native
   linking. Mitigation: abi3 means the wheel only has to *build*
   per-Python-version for CI confidence but the *installer* is
   the same binary — keep the matrix tight, don't add
   per-point-release coverage. GPU + PyTorch cross-validation is
   explicitly not on CI (see §5 + §6); that drops the matrix cost
   from "rent GPU runners per PR" to "rent CPU runners per PR."
6. **PyPI publish credentials.** Manual publish is the initial
   story (see 8.4); lost / rotated credentials could delay a
   point release. Mitigation: documented publish runbook in
   internal docs; two maintainers with publish tokens where
   possible.

## Review Context

Phase 8 is the first phase where KAIO's user surface extends beyond
the Rust ecosystem. Sprint-level plans for 8.1 → 8.4 each receive
the full planning + adversarial review cycle as they kick off. This
master plan establishes the constraints (standalone crate, abi3,
GIL-released ops, reference-counted `Tensor`, thin wrapper
discipline) so sprint-level plans can reference them without re-
litigating.
