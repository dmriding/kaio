# KAIO Showcase Examples

Standalone Cargo projects demonstrating real-world ML kernels written
with `#[gpu_kernel]`. Each example is self-contained: clone the repo,
`cd` into the example directory, and `cargo run --release`. Unlike
[`kaio/examples/`](../kaio/examples/) (which are single-file examples
compiled as part of the workspace), these are independent projects
with their own `Cargo.toml` so users see exactly what a dependency on
KAIO looks like.

## Examples

| Short name  | Directory            | Kernel                                           | Why it matters                                       |
|-------------|----------------------|--------------------------------------------------|------------------------------------------------------|
| `silu`      | `fused_silu_gate/`   | `out = x * sigmoid(x) * gate`                    | Every LLaMA / Mistral feedforward layer              |
| `gelu`      | `gelu_comparison/`   | Exact (tanh) vs fast (sigmoid) GELU side by side | BERT / GPT activations + kernel-variant workflow     |
| `rms`       | `rms_norm/`          | Single-block RMSNorm                             | LLaMA-family normalization (replaces LayerNorm)      |
| `layernorm` | `layer_norm/`        | Single-block LayerNorm                           | Classic transformer normalization (BERT, GPT-2, T5)  |
| `softmax`   | `softmax/`           | Single-block softmax with max-sub stability      | Attention normalization; reduction-heavy primitive   |
| `int8`      | `int8_dequant/`      | Symmetric INT8 dequantization                    | Quantized-weight unpack; signed-shift DSL showcase   |

Each example ships with a `Cargo.toml`, an `src/main.rs` (kernel + CPU
reference + PASS/FAIL + median timing), and a `README.md` that leads
with the kernel source block.

## Using these examples from a fresh clone

The fastest way to see them all run, with no `cd` required:

```sh
git clone https://github.com/dmriding/kaio.git
cd kaio
cargo xtask showcase              # all six in sequence
cargo xtask showcase silu         # just fused_silu_gate
cargo xtask showcase --list       # list available names
```

Each example is also a standalone Cargo project you can run
individually. The "standalone" part proves that the KAIO dependency
works from a fresh `Cargo.toml`, not just from inside the workspace.
To use an example that way:

```sh
cd kaio/examples/fused_silu_gate
cargo run --release
```

Requires an NVIDIA GPU with driver installed. No CUDA toolkit needed.
