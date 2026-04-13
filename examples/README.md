# KAIO Showcase Examples

Standalone Cargo projects demonstrating real-world ML kernels written with
`#[gpu_kernel]`. Each example is self-contained — clone the repo, `cd` into
the example directory, and `cargo run --release`. Unlike [`kaio/examples/`](../kaio/examples/)
(which are single-file examples compiled as part of the workspace), these
are independent projects with their own `Cargo.toml` so users see exactly
what a dependency on KAIO looks like.

## Status

These directories are intentionally empty until **Sprint 6.8** lands the
first three showcases. See
[`docs/development/sprints/phase6/sprint_6_8.md`](../docs/development/sprints/phase6/sprint_6_8.md)
for the plan.

## Planned for Sprint 6.8 (pre-v0.2.0)

| Example                  | Kernel                                           | Why it matters                                      |
|--------------------------|--------------------------------------------------|------------------------------------------------------|
| `fused_silu_gate/`       | `out = x * sigmoid(x) * gate`                   | Every LLaMA / Mistral feedforward layer             |
| `gelu_comparison/`       | Exact (tanh) vs fast (sigmoid) GELU, side by side | BERT / GPT activations + kernel-variant workflow   |
| `rms_norm/`              | Single-block RMSNorm                             | LLaMA-family normalization (replaces LayerNorm)    |

Each example ships with a `Cargo.toml`, an `src/main.rs` (kernel + CPU
reference + PASS/FAIL + median timing), and a `README.md` that leads with
the kernel source block.

## Deferred to post-0.2.0

Candidates that don't fit Sprint 6.8's scope but are good future showcases:

- **`fused_layer_norm/`** — classic LayerNorm (mean + variance in one pass).
  Deferred because RMSNorm already covers the normalization-is-easy pitch.
- **`rotary_embeddings/`** — RoPE positional encoding, the other load-bearing
  LLaMA primitive. Needs element-wise `sin` / `cos` which aren't in the
  `#[gpu_kernel]` macro's builtin set today — adding them is its own task.
- **`vector_cosine_similarity/`** — batched dot-product + `rsqrt` normalization,
  a useful retrieval-kernel demo but less iconic for the v0.2.0 launch story.

File a GitHub issue if you'd like to see any of these prioritized.

## Using these examples from a fresh clone

```sh
git clone https://github.com/dmriding/kaio.git
cd kaio/examples/fused_silu_gate   # (once Sprint 6.8 lands)
cargo run --release
```

Requires an NVIDIA GPU with driver installed. No CUDA toolkit needed.
