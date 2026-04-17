# `int4_matmul` — W4A16 GPTQ-style dequantize-matmul

End-to-end demo of `kaio_ops::matmul_int4` (Sprint 7.2): symmetric
INT4 weights × f16 activations → f32 output via tensor cores.

## What it does

1. Generate deterministic f32 weight + activation matrices.
2. Quantize weights with a GPTQ-lite symmetric per-column group
   scheme (`scale = max(|w_group|) / 7`, `q = clamp(round(w/scale), -8, 7)`).
3. Pack the 8-per-u32 signed INT4 layout KAIO expects.
4. Run `matmul_int4` on GPU.
5. Compare against naive f32 CPU matmul, report max absolute + relative error.

## Run it

```sh
cargo run --release
```

Requires a compute-capability 8.0+ NVIDIA GPU (Ampere or newer).

## Kernel correctness vs quantization quality

The reported max-rel error reflects **INT4 quantization round-trip
error**, NOT kernel error. Signed INT4 has only 16 representable
values, so per-element quant noise is large — commonly 50-80% max-rel
on random uniform inputs with a naive max-abs group scale. Real GPTQ
uses activation-aware scaling and error compensation to tighten this;
the showcase uses the simplest scheme to keep focus on the kernel.

For **bit-exact kernel correctness** — GPU output matches an exact
f16 dequant-chain CPU reference — see:

```sh
cargo test -p kaio-ops --test matmul_int4_e2e -- --ignored
```

## KAIO packed INT4 convention (NOT external GPTQ/GGUF compatible)

For a logical weight matrix `B[k, n]` with shape `[K, N]`:

```text
word_index   = (k / 8) + n * (K / 8)   // index into b_packed (u32, col-major)
nibble_index = k % 8                    // lane within the word
nibble_bits  = (b_packed[word_index] >> (4 * nibble_index)) & 0xF
signed_value = sign_extend_from_4_bits(nibble_bits)  // in [-8, +7]
```

Scales: `[K/group_size, N]` row-major f16. One scale per `(group, output_col)`.

Users bringing pre-quantized AutoGPTQ / exllama / GGUF models must
repack to this layout. The `pack_s4_weights` / `quantize_gptq_lite`
helpers in `src/main.rs` are a working CPU reference packer — copy or
adapt them for your pipeline.

## Related docs

- `kaio-ops/src/matmul_int4_kernel.rs` — kernel module + full rustdoc.
- `kaio-ops/tests/matmul_int4_e2e.rs` — bit-exact GPU correctness tests.
- `kaio-ops/tests/matmul_int4_bench.rs` — perf bench vs cuBLAS sgemm.
- `docs/development/sprints/phase7/sprint_7_2.md` — sprint log.
