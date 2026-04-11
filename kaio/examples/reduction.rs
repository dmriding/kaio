//! Sum reduction — load from global memory, reduce with shared memory.
//!
//! Run: cargo run --example reduction -p kaio

use kaio::prelude::*;

#[gpu_kernel(block_size = 256)]
fn sum_reduce(input: &[f32], out: &mut [f32], n: u32) {
    let tid = thread_idx_x();
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();

    // Load from global memory (0.0 for out-of-bounds threads)
    let mut val = 0.0f32;
    if idx < n {
        val = input[idx];
    }

    // Block-wide sum reduction using shared memory + warp shuffles
    let total = block_reduce_sum(val);

    // Thread 0 writes the block result
    if tid == 0u32 {
        out[block_idx_x()] = total;
    }
}

fn main() -> Result<()> {
    let device = KaioDevice::new(0)?;

    // 256 elements: 1 + 2 + 3 + ... + 256 = 32896
    let n = 256u32;
    let input: Vec<f32> = (1..=n).map(|i| i as f32).collect();
    let cpu_sum: f32 = input.iter().sum();

    let d_input = device.alloc_from(&input)?;
    let mut d_out = device.alloc_zeros::<f32>(1)?;

    sum_reduce::launch(&device, &d_input, &mut d_out, n)?;

    let result = d_out.to_host(&device)?;
    println!("input:    [1.0, 2.0, 3.0, ..., 256.0]");
    println!("GPU sum:  {}", result[0]);
    println!("CPU sum:  {cpu_sum}");
    println!("match:    {}", (result[0] - cpu_sum).abs() < 1e-1);

    Ok(())
}
