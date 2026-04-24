# int8_dequant — symmetric INT8 dequantization

INT8 is the recognizable, workhorse quantized-weight format. Values are
stored as signed 8-bit integers in `[-128, 127]`, four packed per
`u32`, and a single scalar `scale` multiplier reconstructs the
floating-point weight as:

```
dequant_f32 = (i8_value) * scale
```

This example demonstrates the dequantize primitive alone — it does NOT
fuse with matmul. Full INT8 dequantize-matmul using tensor cores is the
planned Phase 7.1 milestone.

## The kernel

```rust
#[gpu_kernel(block_size = 256)]
fn dequant_i8(packed: *const [u32], out: *mut [f32], scale: f32, n_words: u32) {
    let tid = thread_idx_x();

    if tid < n_words {
        let word = packed[tid];

        // Extract each byte, sign-extend to i32, cast to f32, scale.
        let b0 = (((word & 0xFF) as i32) << 24) >> 24;
        let b1 = ((((word >> 8) & 0xFF) as i32) << 24) >> 24;
        let b2 = ((((word >> 16) & 0xFF) as i32) << 24) >> 24;
        let b3 = ((((word >> 24) & 0xFF) as i32) << 24) >> 24;

        let base = tid * 4;
        out[base] = (b0 as f32) * scale;
        out[base + 1] = (b1 as f32) * scale;
        out[base + 2] = (b2 as f32) * scale;
        out[base + 3] = (b3 as f32) * scale;
    }
}
```

## How the sign-extension works

The non-obvious line is:

```rust
let b0 = (((word & 0xFF) as i32) << 24) >> 24;
```

Reading it step by step:

1. `word & 0xFF` — mask off the low byte. Result is `u32` in `[0, 255]`.
2. `as i32` — reinterpret as signed. The value still fits in `i32`
   positively, so no change yet.
3. `<< 24` — shift left so bit 7 of the byte becomes bit 31 of the
   `i32` (the sign bit).
4. `>> 24` on an `i32` — **arithmetic right shift**. The PTX instruction
   is `shr.s32` and it sign-extends: a negative byte ends up as a
   correctly-signed `i32`.

If the `>>` on step 4 were a *logical* shift (`shr.u32` or the
typeless `shr.b32`), the dequantized value would silently be wrong for
any negative byte — `-1` would come back as `255`, `-128` as `128`, and
so on. No ptxas error, no crash, just bit-exact wrong results.

The signed/unsigned-shift distinction shipped in KAIO v0.2.1 is exactly
what protects this kernel. In `kaio/tests/bitops_macro.rs` there is an
end-to-end test (`shr_arithmetic_i32_sign_extends`) whose comment
explicitly flags this case as the motivating scenario.

## Scope / limitations

- **Symmetric only**: `zero_point = 0`, implicit. For asymmetric INT8
  you would subtract a per-tensor or per-channel `zero_point` before
  scaling.
- **One scalar scale**: real quantized formats (Q4_K, Q6_K, AWQ, GPTQ,
  etc.) use per-channel, per-group, or per-block scales. The DSL can
  express those — you just change `scale: f32` to `scales: &[f32]` and
  index into it. Outside this example's scope.
- **INT8 only**: 4-bit, 5-bit, and 6-bit variants use the same DSL
  primitives (bitwise AND with a narrower mask, correspondingly
  smaller shift amounts for sign-extension). 4-bit in particular packs
  8 values per `u32` with a 4-bit mask. KAIO v0.2.1 / v0.2.2 supports
  this pattern — just not in this example.
- **No matmul**: this is the dequantize primitive only. The matmul
  fusion (reading dequantized weights directly into a tensor-core
  accumulator) is Phase 7.1.
- **Single-block**: `n_words <= 256` (1024 output elements). Multi-block
  is a one-line change to the thread-index formula, but the single-block
  case keeps the pattern visible.

## Running

```sh
cargo run --release
```

Requires an NVIDIA GPU with an installed driver. No CUDA toolkit
needed.

## Output

```
=== int8_dequant ===
Packed input:      256 u32 words  (1024 i8 values, single-block)
Scale:             0.0125
Correctness:       PASS  (max_abs_err = 0.00e0)
Median latency:    XXX.X μs  (of 100 timed runs, 5 warm-ups skipped)
```

Correctness is checked bit-exactly against a CPU reference that does
the same unpack-sign-extend-scale in safe Rust. Because the i8 range
fits exactly in f32 and there is a single scale multiplication, the
result is `0.0` absolute error, not "within tolerance" — the GPU and
CPU produce identical bits.
