# int8_matmul — symmetric INT8 dequantize-matmul

Full-pipeline showcase for the `kaio_ops::matmul_int8` op shipped in
KAIO v0.3.0:

1. quantize two f32 matrices to i8 using a simple per-tensor max-abs
   calibration,
2. run `matmul_int8` on-device (tensor-core `mma.sync.m16n8k32.s8.s8.s32`,
   s32 accumulator, single scalar post-accumulation scale),
3. compare the scaled f32 output against a naive f32 CPU matmul of
   the original f32 inputs.

The error you see is **quantization error**, not kernel error —
the i8 round-trip loses precision on values that don't land on exact
i8 quantization levels. A broken kernel would show radically worse
error (or structural NaN/zero patterns).

## What `matmul_int8` is (v0.3.0)

- **W8A8**: both operands must be quantized to i8. Mixed-precision
  W8A16 (i8 weights × f16 activations) is NOT supported; that would
  be a distinct future op.
- **Symmetric** (zero-point = 0). Asymmetric quant is a future
  additive refinement.
- **Single global scalar scale** applied post-accumulation —
  one `f32` for the full output.
- **`K % 32 == 0` required** (the mma K-tile is structural;
  non-multiples return `KaioError::InvalidConfig`).
- **Sync-only** in 7.1; async INT8 is a known follow-up.

This is positioned as the **reference quant op**, not the final
general-quant architecture. GPTQ / AWQ / per-channel / per-group /
INT4 all land as future additive refinements in later sprints.

## Running

```bash
cd examples/int8_matmul
cargo run --release
```

Requires an Ampere+ NVIDIA GPU (SM 8.0+).

## Scope / limitations

- Single block tile geometry (`matmul_int8` internally uses a 64×64
  output block with 4 warps × 2×2 sub-quadrants). Any M, N positive
  value is accepted; K must be a multiple of 32.
- Sync kernel, not cp.async pipelined. Async INT8 is a known
  follow-up — measured sync performance is the v0.3.0 baseline.
- Per-tensor scale only. For per-channel / per-group quant, the
  kernel would need to accept `scales: &[f32]` — that is a future
  additive API, not in v0.3.0.

## Bit-exact correctness validation

For bit-exact i8 × i8 → i32 round-trip validation (independent of
quantization error), see
`kaio-ops/tests/matmul_int8_e2e.rs` — run with
`cargo test -p kaio-ops --test matmul_int8_e2e -- --ignored`.
