//! SAXPY — y = alpha * x + y — shows scalar parameter passing.
//!
//! Run: cargo run --example saxpy -p kaio

use kaio::prelude::*;

#[gpu_kernel(block_size = 256)]
fn saxpy(x: &[f32], y: &mut [f32], alpha: f32, n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        y[idx] = alpha * x[idx] + y[idx];
    }
}

fn main() -> Result<()> {
    let device = KaioDevice::new(0)?;

    let n = 8u32;
    let alpha = 2.5f32;
    let x_host = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y_host = [10.0f32, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0];

    let x = device.alloc_from(&x_host)?;
    let mut y = device.alloc_from(&y_host)?;

    println!("x:     {:?}", x_host);
    println!("y:     {:?}", y_host);
    println!("alpha: {alpha}");

    saxpy::launch(&device, &x, &mut y, alpha, n)?;

    let result = y.to_host(&device)?;
    println!("y = alpha * x + y: {:?}", result);

    Ok(())
}
