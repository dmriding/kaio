# PYROS

**Rust-native GPU kernel authoring framework.**

PYROS (πῦρ — fire) lets developers write GPU compute kernels in Rust and
compile them to PTX for execution on NVIDIA GPUs. It is a Rust alternative
to OpenAI's Triton, targeting Windows and Linux from day one, with
compile-time PTX emission and Rust's type-safety guarantees.

## Why PYROS?

- **Cross-platform from day one.** Windows and Linux. `cargo build` just works.
- **Compile-time PTX emission.** Kernels compile during `cargo build` via proc macros. Zero cold-start.
- **Rust type safety.** Catch out-of-bounds indexing, dtype mismatches, and synchronization errors at compile time.
- **Embeddable anywhere.** Use from Rust natively, from C/C++ via FFI, from Python via PyO3.

## Architecture

PYROS is structured in four layers:

```
Layer 4: Block-Level Operations        (tiled matmul, fused attention)
Layer 3: Proc Macro DSL                (#[gpu_kernel], user-facing API)
Layer 2: Runtime                       (kernel launch, memory mgmt via cudarc)
Layer 1: PTX Codegen                   (instruction emission, IR)
```

## Crate Structure

| Crate | Description |
|-------|-------------|
| `pyros` | Umbrella crate — re-exports everything |
| `pyros-core` | PTX IR types, instruction emitters, PtxWriter |
| `pyros-runtime` | CUDA driver API wrapper, kernel launch, device memory |

## Current Status

**Phase 1 — PTX Foundation** is in progress. Building the IR and runtime
layers with one working end-to-end kernel (`vector_add`) as the milestone.

See [docs/development/PHASE_1_LOG.md](docs/development/PHASE_1_LOG.md) for
sprint-by-sprint progress, and [CHANGELOG.md](CHANGELOG.md) for release
history.

## Target Hardware

- **Primary:** NVIDIA GPUs, SM 7.0+ (Volta and newer)
- **Development GPU:** RTX 4090 (SM 8.9, Ada Lovelace)
- **Platforms:** Windows 10/11, Linux (Ubuntu 22.04+)

## Building

```sh
# Requires Rust 1.94+ (pinned via rust-toolchain.toml)
cargo build --workspace
cargo test --workspace
```

GPU-dependent tests are gated behind `#[ignore]` and require an NVIDIA GPU:

```sh
cargo test --workspace -- --ignored
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or
  <http://opensource.org/licenses/MIT>)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
