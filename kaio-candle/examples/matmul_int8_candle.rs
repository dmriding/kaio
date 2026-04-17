//! Minimal `kaio_candle::matmul_int8` example (W8A8 symmetric quant).
//!
//! Allocates two `u8`-dtype candle tensors (candle convention for INT8 —
//! see `kaio_candle::matmul_int8` module docs), calls the bridge with a
//! fixed scalar scale, and prints a small corner of the f32 output.
//!
//! Build + run:
//! ```sh
//! cd kaio-candle
//! cargo run --release --features cuda --example matmul_int8_candle
//! ```

use std::sync::Arc;

use candle_core::{Device, Tensor};
use kaio::prelude::KaioDevice;

fn main() -> anyhow::Result<()> {
    let candle_dev = Device::new_cuda(0)?;
    let kaio_dev = Arc::new(KaioDevice::new(0)?);

    let m = 128usize;
    let k = 128usize;
    let n = 128usize;
    let scale = 0.01f32;

    // Deterministic signed-INT8 patterned data in the [-63..=63] range.
    // `as u8` preserves bits for i8 → u8 (so -1_i8 becomes 255_u8 and the
    // bridge reinterprets it back to -1 on the GPU side).
    let a_host: Vec<u8> = (0..m * k)
        .map(|i| (((i % 127) as i32 - 63) as i8) as u8)
        .collect();
    let b_host: Vec<u8> = (0..k * n)
        .map(|i| (((i % 97) as i32 - 48) as i8) as u8)
        .collect();

    let a = Tensor::from_vec(a_host, (m, k), &candle_dev)?;
    let b = Tensor::from_vec(b_host, (k, n), &candle_dev)?;

    let c = kaio_candle::matmul_int8(&kaio_dev, &a, &b, scale)?;
    let c_host: Vec<f32> = c.flatten_all()?.to_vec1::<f32>()?;

    println!("matmul_int8 output shape: {:?}", c.shape().dims());
    println!("scale: {scale}");
    println!("first 8 values: {:?}", &c_host[..8]);
    println!(
        "c[0,0] = {:.6}, c[{m_m1},{n_m1}] = {:.6}",
        c_host[0],
        c_host[m * n - 1],
        m_m1 = m - 1,
        n_m1 = n - 1,
    );

    Ok(())
}
