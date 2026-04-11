//! Simplest possible GPU kernel — add two vectors element-wise.
//!
//! Run: cargo run --example vector_add -p kaio

use kaio::prelude::*;

#[gpu_kernel(block_size = 256)]
fn vector_add(a: &[f32], b: &[f32], out: &mut [f32], n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        out[idx] = a[idx] + b[idx];
    }
}

fn main() -> Result<()> {
    let device = KaioDevice::new(0)?;

    let a_host = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b_host = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
    let n = a_host.len() as u32;

    let a = device.alloc_from(&a_host)?;
    let b = device.alloc_from(&b_host)?;
    let mut out = device.alloc_zeros::<f32>(n as usize)?;

    vector_add::launch(&device, &a, &b, &mut out, n)?;

    let result = out.to_host(&device)?;
    println!("a:     {:?}", a_host);
    println!("b:     {:?}", b_host);
    println!("a + b: {:?}", result);

    Ok(())
}
