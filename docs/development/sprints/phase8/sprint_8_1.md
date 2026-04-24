# Sprint 8.1 — PyO3 scaffold (Phase 8 kickoff)

**Status:** ✅ Complete (2026-04-24)
**Branch:** `8.1` (PR to `main` pending)

---

## Context

Phase 8's stated goal is `pip install kaio` on Windows + Linux — Python users gain access to KAIO's tensor-core matmul, attention, and quantized kernels without learning Rust or leaving their workflow. Sprint 8.1 lands the minimum viable scaffold: a new standalone `kaio-py` crate (PyO3 + rust-numpy + maturin), the foundational Python classes, and one end-to-end smoke kernel proving the full plumbing path (NumPy → GPU buffer → kernel launch → NumPy). Broader op coverage, richer error taxonomy, cross-validation, and PyPI packaging all split into Sprints 8.2 → 8.4.

The master plan locked in five architectural invariants before execution: standalone-crate status to avoid the `cudarc` feature-union collision that Phase 7 hit with `kaio-candle`, abi3-py310 for single-wheel-per-platform distribution, GIL released during kernel execution via `Python::detach`, reference-counted `Arc<KaioDevice>` propagated to every `Tensor`, and thin-wrapper discipline (no API redesign in the Python layer). 8.1 inherits all five.

## What shipped

### New crate — `kaio-py/`

- Standalone Cargo crate at repo root, listed in `workspace.exclude` alongside `kaio-candle` for the same cudarc-feature-union reason.
- `[lib] name = "kaio"` with `crate-type = ["cdylib"]` for Python import.
- The Rust crate `kaio` is aliased via `kaio-rs = { package = "kaio", path = "../kaio", ... }` to avoid a name collision with this crate's `[lib] name`; internal imports use `kaio_rs::...`.
- Dependencies: `pyo3 = "0.28"` (with `abi3-py310`, **no** `extension-module` feature — deprecated by PyO3 for maturin ≥ 1.9.4), `numpy = "0.28"` (with the `half` feature for `f16 ↔ NumPy` interop via `numpy::Element`), `kaio` + `kaio-ops` pulled with `default-features = false, features = ["dynamic-loading"]`, plus `half` and `thiserror`.
- `pyproject.toml` pins `maturin >= 1.12, < 2`, `requires-python = ">=3.10"`, project metadata for PyPI display (classifiers, URLs), and no `[tool.maturin]` table (no `extension-module` feature, no `python-source` — pure extension-only module).

### Python surface

- `kaio.Device` — wraps `Arc<KaioDevice>`; exposes `.name`, `.compute_capability`, and a `__repr__`. `Device(index)` opens the CUDA context; invalid ordinals raise `kaio.KaioError`. Confirmed Send-safe (no `#[pyclass(unsendable)]` needed).
- `kaio.Tensor` — owns an enum `TensorStorage { F16(GpuBuffer<f16>), F32(GpuBuffer<f32>) }` plus shape + the parent Device's `Arc`. `Tensor.from_numpy(device, array)` dispatches on NumPy dtype at runtime (single entry point, no PyO3-incompatible method overloading) and rejects non-C-contiguous inputs with a clear error pointing to `np.ascontiguousarray`. `.to_numpy()` copies back. `__len__` returns the first dim (NumPy / Torch convention); `.numel` returns the product. Enum-based storage is exhaustive-match-checked — the compiler will surface every site that needs a new arm when 8.2 adds `i8` + packed-`u32`.
- `kaio.KaioError` — single exception class (subclassed from Python's `Exception`) routed through two helpers in `errors.rs`: `map_kaio_err(e: kaio_rs::KaioError) -> PyErr` for Rust-Result lifts, and `kaio_err(msg)` for Python-layer validation messages. Both land on the same exception class. A `From<kaio_rs::KaioError> for PyErr` impl was considered and rejected — it would be an orphan impl (both trait and type foreign to `kaio-py`; Rust rejects this at compile time). Subclasses (`KaioValidationError`, `KaioDeviceError`, etc.) are deferred to 8.2 when broader op coverage earns them.
- `kaio.matmul_tc(a, b)` — the smoke kernel. Validates both inputs are `float16` with dtype-specific error on mismatch, cross-checks Device identity via `Arc::ptr_eq` (so two separate `kaio.Device(0)` calls are treated as distinct — tensors must come from the *same Device object*), validates shapes (2-D only, K consistency), converts `usize` dims to `u32` with bounds checks that fail fast rather than truncate, allocates an `f32` output buffer, releases the GIL via `py.detach(|| kaio_ops::matmul_tc(...))`, and wraps the result as a new `Tensor`.

### Example + documentation

- `kaio-py/examples/hello.py` — Device → two f16 NumPy arrays → `kaio.matmul_tc` → `to_numpy()` → max-abs-err check against a NumPy f32 reference. Runs end-to-end on the reference RTX 4090 (sm_89) in under a second and returns exit 0 on pass.
- `kaio-py/README.md` — quick-start, usage sketch, error-handling pattern, explicit "Known limitations" section covering C-contiguity, Device-identity-by-object, single-kernel-exposed, no-PyPI-yet, no-autograd. Forward pointer to RFC-0001 pointer syntax for users writing custom kernels in Rust.

### Workspace

- Root `Cargo.toml` adds `kaio-py` to `workspace.exclude`; the comment is extended from the kaio-candle-only version to cover both bridge crates with the same feature-union rationale.
- `.gitignore` adds `kaio-py/target/`, `kaio-py/Cargo.lock` (standalone-crate convention), and the common Python artifact patterns (`__pycache__/`, `*.pyc`, `*.pyo`, `*.egg-info/`, `.pytest_cache/`, `.venv/`, `venv/`, `env/`).

## Tests

No new automated tests — the 8.1 deliverable is end-to-end plumbing validated by `examples/hello.py` (manual GPU smoke) and the abi3 wheel-build path (reproducible via `maturin build --release`). Unit tests on the Python surface land in Sprint 8.3 alongside the PyTorch cross-validation work. All existing Rust tests (workspace host + `--ignored` GPU) run unchanged; the kaio-py additions are purely additive.

Verified manually against the plan's acceptance list:

1. `cargo build --manifest-path kaio-py/Cargo.toml --release` — clean.
2. `maturin build --release` — produces an abi3-py310 wheel (`kaio-0.1.0-cp310-abi3-*.whl`).
3. `pip install <wheel>` + `python -c "import kaio"` — module imports on a clean Python 3.11 interpreter, exposes `Device`, `Tensor`, `KaioError`, `matmul_tc`.
4. `python examples/hello.py` — GPU name printed, matmul correctness within f16 tolerance, "hello, KAIO from Python." printed.
5. `kaio.Device(999)` raises `kaio.KaioError` (not a Rust panic).
6. `kaio.matmul_tc(f32_tensor, f16_tensor)` raises `kaio.KaioError` with a dtype-specific message.
7. `kaio.matmul_tc(a_MK, b_JN)` where K ≠ J raises `kaio.KaioError` with a shape-specific message.
8. Non-C-contiguous NumPy input to `from_numpy` raises `kaio.KaioError` pointing the user to `np.ascontiguousarray`.
9. Tensors from two distinct `kaio.Device(0)` calls rejected by `matmul_tc` with a device-identity error.

## What didn't change

- Zero changes to `kaio`, `kaio-core`, `kaio-macros`, `kaio-ops`, or `kaio-runtime`. The scaffold is a pure consumer of the existing Rust public API.
- No new Rust kernels; no macro changes; no runtime changes; no bench additions.
- No CI additions — Rust workflows run against the main workspace as before; `kaio-py` CI (wheel build + install smoke on Windows + Linux across Python 3.10–3.12) lands in Sprint 8.4.
- No PyPI publication. `pip install kaio` does not yet work; the path is `maturin develop` from `kaio-py/` for now.
- No version bump — the scaffold is additive and the public Rust API is unchanged.

## Known limitations (intentional for 8.1)

- **One kernel.** Only `matmul_tc` is exposed from Python. Broader coverage is 8.2.
- **Two dtypes.** `Tensor` accepts `float16` + `float32` only. INT8 + packed-INT4 storage variants land in 8.2 alongside the quantized-matmul exposure.
- **Single error class.** All errors surface as `kaio.KaioError`. Subclass hierarchy is 8.2.
- **No pytest / PyTorch cross-validation in the repo yet.** Numerical validation is the manual check in `hello.py` until 8.3.
- **Dev-only install.** `maturin develop` for iteration; no wheel on PyPI until 8.4.

## Follow-ups

- **Sprint 8.2 — Core ops exposure.** `matmul_tc_async`, `matmul_int8`, `matmul_int4`, `qkv_project_int{4,8}`, `attention_tc[_causal]`, `attention_flash[_causal]`. New `Tensor` dtypes (i8, packed u32). Error subclass split. Deferred-to-8.2 opportunities (notably `kaio.softmax` — only ships once a multi-block version lands on the Ops Track so Python users don't hit a misleading `n ≤ 256` cap).
- **Sprint 8.3 — Cross-validation + Python bench.** PyTorch-equivalent tests in `tests/` gated behind a pytest marker CI skips (local-run, manually posted). FFI-overhead bench separately quantifying Python-dispatch cost.
- **Sprint 8.4 — Packaging + CI.** Windows + Linux wheels on PyPI via `maturin upload`. GitHub Actions `{windows, ubuntu} × {3.10, 3.11, 3.12}` CPU-only smoke matrix.
