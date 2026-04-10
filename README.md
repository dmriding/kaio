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
| `pyros` | Umbrella crate — re-exports `pyros-core` and `pyros-runtime` |
| `pyros-core` | PTX IR types, instruction emitters, PtxWriter |
| `pyros-runtime` | CUDA driver API wrapper, kernel launch, device memory |

## Current Status

**Phase 1 — PTX Foundation — complete.** The IR and runtime layers can
construct, emit, load, and execute GPU kernels. The `vector_add` kernel
runs on real hardware (RTX 4090, verified on both single-block and
multi-block launches).

**Phase 2 — Proc Macro DSL** is next: `#[gpu_kernel]` attribute macro
that transforms Rust function syntax into PTX. See
[docs/phases.md](docs/phases.md) for the full roadmap.

### Phase 1 Example (IR API)

```rust
use pyros_core::emit::{Emit, PtxWriter};
use pyros_core::instr::{ArithOp, MadMode, special};
use pyros_core::instr::control::{CmpOp, ControlOp};
use pyros_core::instr::memory::MemoryOp;
use pyros_core::ir::*;
use pyros_core::types::PtxType;

// Build a vector_add kernel via the IR API
let mut alloc = RegisterAllocator::new();
let mut kernel = PtxKernel::new("vector_add");
kernel.add_param(PtxParam::pointer("a_ptr", PtxType::F32));
kernel.add_param(PtxParam::pointer("b_ptr", PtxType::F32));
kernel.add_param(PtxParam::pointer("c_ptr", PtxType::F32));
kernel.add_param(PtxParam::scalar("n", PtxType::U32));

// ... (build instructions using alloc + kernel.push()) ...

// Emit to PTX text
let mut module = PtxModule::new("sm_89");
module.add_kernel(kernel);
let mut w = PtxWriter::new();
module.emit(&mut w).unwrap();
let ptx_text = w.finish();

// Load and run on GPU
use pyros_runtime::{PyrosDevice, LaunchConfig};
let device = PyrosDevice::new(0)?;
let module = device.load_ptx(&ptx_text)?;
let func = module.function("vector_add")?;
// ... allocate buffers, launch kernel, read results ...
```

See [pyros-runtime/tests/vector_add_e2e.rs](pyros-runtime/tests/vector_add_e2e.rs)
for the complete working example.

## Target Hardware

- **Primary:** NVIDIA GPUs, SM 7.0+ (Volta and newer)
- **Development GPU:** RTX 4090 (SM 8.9, Ada Lovelace)
- **Platforms:** Windows 10/11, Linux (Ubuntu 22.04+)

## Building

```sh
# Requires Rust 1.94+ (pinned via rust-toolchain.toml)
cargo build --workspace
cargo test --workspace           # host-only tests (no GPU required)
cargo test -p pyros-runtime -- --ignored   # GPU tests (requires NVIDIA GPU)
```

## Development

Sprint-by-sprint progress with full architectural decision records:
- [Phase 1 Sprint Log](docs/development/PHASE_1_LOG.md)
- [Sprint docs with reasoning traces](docs/development/sprints/)
- [CHANGELOG](CHANGELOG.md)

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
