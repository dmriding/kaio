# kaio (Python)

Python bindings for [KAIO](https://github.com/dmriding/kaio) — a
Rust-native GPU kernel authoring framework. Tensor-core matmul,
attention, and quantized kernels callable from Python on Windows +
Linux, no CUDA toolkit install required.

## Status

**Sprint 8.1 scaffold.** The Python package is under active
development. `import kaio` works after `maturin develop`; the public
API is minimal while the plumbing stabilizes.

- Shipped in 8.1: `kaio.Device`, `kaio.Tensor` (NumPy roundtrip for
  `float16` / `float32`), `kaio.KaioError`, `kaio.matmul_tc`.
- Next sprint (8.2): matmul_tc_async, matmul_int8, matmul_int4,
  fused QKV projections, attention_tc + flash variants. Error-class
  subclasses (`KaioValidationError`, `KaioDeviceError`, etc.).
- Later sprints: cross-validation against PyTorch (8.3), `pip
  install kaio` from PyPI with wheel packaging and CI (8.4).

## Requirements

- Python ≥ 3.10
- NVIDIA GPU + driver (no CUDA toolkit install needed; KAIO uses
  `cudarc`'s `dynamic-loading`)
- SM 8.0+ (Ampere or newer) for tensor-core ops
- Rust toolchain + [maturin](https://www.maturin.rs/) **for
  development only** — wheels built in Sprint 8.4+ will not require
  these on end-user machines.

## Quick start (development)

```sh
pip install maturin numpy
cd kaio-py
maturin develop --release   # builds the extension, installs into active venv
python examples/hello.py
```

Expected output:

```
GPU: NVIDIA GeForce RTX 4090 (sm_89)

inputs:  A=Tensor(shape=[64, 64], dtype=float16), B=Tensor(shape=[64, 64], dtype=float16)
output:  C=Tensor(shape=[64, 64], dtype=float32)

output shape:   (64, 64), dtype: float32
max abs error:  0.0000  (vs NumPy f32 reference)

hello, KAIO from Python.
```

If `pyo3-build-config` fails with "failed to run the Python
interpreter," set `PYO3_PYTHON` to an explicit path:

```sh
PYO3_PYTHON=/path/to/python3 maturin develop --release
```

## Usage

```python
import numpy as np
import kaio

device = kaio.Device(0)

# Build two f16 matrices via NumPy; KAIO accepts NumPy arrays directly.
a = kaio.Tensor.from_numpy(device, np.random.randn(1024, 1024).astype(np.float16))
b = kaio.Tensor.from_numpy(device, np.random.randn(1024, 1024).astype(np.float16))

# Tensor-core matmul, f16 in / f32 out. GIL released during kernel.
c = kaio.matmul_tc(a, b)

# Tensor → Tensor passes through GPU without host round-trip; only
# copy back to NumPy when you need the result on the CPU.
result = c.to_numpy()
print(result.shape, result.dtype)  # (1024, 1024) float32
```

## Error handling

All errors — validation, device, kernel launch — surface as
`kaio.KaioError`, subclassed from Python's `Exception`:

```python
try:
    c = kaio.matmul_tc(a, b)
except kaio.KaioError as e:
    print(f"KAIO error: {e}")
```

Subclasses (`KaioValidationError`, `KaioDeviceError`,
`KaioPtxError`) land in Sprint 8.2 when the op surface is wide
enough to justify them.

## Known limitations

- **C-contiguity required on inputs.** `kaio.Tensor.from_numpy`
  rejects non-C-contiguous NumPy arrays with a clear `KaioError`. If
  you're passing a sliced or transposed view, wrap it with
  `np.ascontiguousarray(x)` before the call.
- **Device identity is by object, not ordinal.** Two separate
  `kaio.Device(0)` calls produce distinct Python objects, and the
  `matmul_tc` cross-input check compares objects via
  `Arc::ptr_eq`. In practice: construct one `kaio.Device` per GPU at
  startup and reuse it. Passing Tensors from two different
  `Device(0)` objects into the same op raises `KaioError`.
- **Single kernel exposed.** Only `matmul_tc` in Sprint 8.1. Wider
  op coverage is 8.2.
- **No wheel on PyPI yet.** `pip install kaio` does not work;
  development install via `maturin develop` is the only path until
  Sprint 8.4 ships the wheel-packaging + CI story.
- **No autograd.** Python wraps KAIO's forward ops. Autograd
  integration via PyTorch `torch.autograd.Function` is post-Phase-8.

## Project layout

```
kaio-py/
  Cargo.toml        # PyO3 0.28 + rust-numpy + kaio (aliased as kaio-rs)
  pyproject.toml    # maturin build backend, abi3-py310
  README.md         # this file
  src/
    lib.rs          # #[pymodule] entry point
    device.rs       # Device class (Arc<KaioDevice>)
    tensor.rs       # Tensor class with TensorStorage enum (F16/F32)
    errors.rs       # KaioError exception + map_kaio_err / kaio_err helpers
    ops.rs          # matmul_tc (the 8.1 smoke kernel)
  examples/
    hello.py        # end-to-end NumPy → Device → kernel → NumPy demo
```

## Writing custom kernels in Rust

KAIO's authoring surface remains Rust, via the `#[gpu_kernel]` macro
in the main `kaio` crate. If you're dropping into Rust from Python to
write a custom kernel, use the pointer form (`*mut [T]` /
`*const [T]`) as the primary signature per
[RFC-0001](../docs/development/rfcs/rfc-0001-pointer-syntax.md).
Reference-form (`&mut [T]` / `&[T]`) remains supported as permanent
sugar.

## License

Dual-licensed under MIT OR Apache-2.0. Same as the parent KAIO
project.
