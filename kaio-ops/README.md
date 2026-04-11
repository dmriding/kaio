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
