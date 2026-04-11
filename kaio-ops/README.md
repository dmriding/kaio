# kaio-ops

Pre-built GPU operations for [KAIO](https://github.com/dmriding/kaio).

## Matrix Multiplication

```rust
use kaio::prelude::*;
use kaio_ops::matmul;

let device = KaioDevice::new(0)?;
let a = device.alloc_from(&a_data)?;  // M x K, row-major f32
let b = device.alloc_from(&b_data)?;  // K x N, row-major f32
let mut c = device.alloc_zeros::<f32>(m * n)?;
matmul(&device, &a, &b, &mut c, m, n, k)?;
```

Tiled implementation using shared memory and register tiling (64x64
blocks, 4x4 per thread). Handles arbitrary dimensions with bounds
checking.

## Attention

```rust
use kaio::prelude::*;
use kaio_ops::attention;

let device = KaioDevice::new(0)?;
let q = device.alloc_from(&q_data)?;  // (seq_len, d_k), row-major f32
let k = device.alloc_from(&k_data)?;
let v = device.alloc_from(&v_data)?;
let mut out = device.alloc_zeros::<f32>(seq_len * d_k)?;
attention(&device, &q, &k, &v, &mut out, seq_len, d_k)?;
```

Single-head scaled dot-product attention. Variants:
- `attention()` / `attention_causal()` — standard (materialized)
- `attention_flash()` / `attention_flash_causal()` — FlashAttention
  (O(d_k) memory, no attention matrix materialization, d_k <= 256)

## Auto-Tuner

```rust
use kaio_ops::{tune_matmul, matmul_auto};

tune_matmul(&device, 1024, 1024, 1024)?;  // benchmark once
matmul_auto(&device, &a, &b, &mut c, m, n, k)?;  // uses best variant
```

## Performance

Benchmarked on RTX 4090 vs cuBLAS sgemm:

| Size | KAIO (TFLOPS) | cuBLAS (TFLOPS) | Ratio |
|------|--------------|----------------|-------|
| 1024x1024 | 5.96 | 37.61 | 15.8% |
| 2048x2048 | 13.32 | 43.14 | 30.9% |
| 4096x4096 | 17.44 | 56.00 | 31.2% |

Current: 31% of cuBLAS sgemm (scalar loads, 4x4 register tiling).
Planned: vectorized loads (LDG.128), double buffering, size-based dispatch.

## Requirements

- NVIDIA GPU (SM 7.0+)
- CUDA driver installed
- Rust 1.94+

## License

MIT OR Apache-2.0
