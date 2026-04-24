# kaio (Python)

Python bindings for [KAIO](https://github.com/dmriding/kaio) — a
Rust-native GPU kernel authoring framework. Tensor-core matmul,
attention, and quantized kernels callable from Python on Windows +
Linux, no CUDA toolkit install required.

## Status

**Sprint 8.1 scaffold.** The Python package is under active
development. `import kaio` works after `maturin develop`; the
public API is minimal while the plumbing stabilizes.

- Shipped: none yet (this is the scaffold-building sprint).
- Next sprint (8.2): core ops — matmul (f32 / INT8 / INT4),
  attention, fused QKV projection.
- Later sprints: cross-validation against PyTorch (8.3), `pip
  install kaio` from PyPI with wheel packaging and CI (8.4).

## Requirements

- Python ≥ 3.10
- NVIDIA GPU + driver (no CUDA toolkit install required; KAIO uses
  `cudarc`'s `dynamic-loading`)
- Rust toolchain + [maturin](https://www.maturin.rs/) — **only for
  development**; wheels built in Sprint 8.4+ will not require these
  on end-user machines.

## Quick start (development)

```sh
pip install maturin
cd kaio-py
maturin develop
python -c "import kaio; print(kaio.__doc__)"
```

## Project layout

```
kaio-py/
  Cargo.toml        # PyO3 + rust-numpy + kaio (aliased as kaio-rs)
  pyproject.toml    # maturin build backend, abi3-py310
  README.md         # this file
  src/
    lib.rs          # #[pymodule] entry point
```

## License

Dual-licensed under MIT OR Apache-2.0. Same as the parent KAIO
project.
