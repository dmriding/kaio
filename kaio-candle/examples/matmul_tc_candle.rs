//! Minimal `kaio_candle::matmul_tc` example.
//!
//! Allocates two f16 matrices via candle on CUDA, calls the KAIO
//! tensor-core matmul through the bridge, and prints a small corner of
//! the f32 output. Shapes are small so the example finishes fast.
//!
//! Build + run:
//! ```sh
//! cd kaio-candle
//! cargo run --release --features cuda --example matmul_tc_candle
//! ```

use std::sync::Arc;

use candle_core::{Device, Tensor};
use half::f16;
use kaio::prelude::KaioDevice;

fn main() -> anyhow::Result<()> {
    let candle_dev = Device::new_cuda(0)?;
    let kaio_dev = Arc::new(KaioDevice::new(0)?);

    let m = 128usize;
    let k = 128usize;
    let n = 128usize;

    // Deterministic patterned data so output is reproducible.
    let a_host: Vec<f16> = (0..m * k)
        .map(|i| f16::from_f32((i % 17) as f32 * 0.01 - 0.08))
        .collect();
    let b_host: Vec<f16> = (0..k * n)
        .map(|i| f16::from_f32((i % 13) as f32 * 0.02 - 0.13))
        .collect();

    let a = Tensor::from_vec(a_host, (m, k), &candle_dev)?;
    let b = Tensor::from_vec(b_host, (k, n), &candle_dev)?;

    let c = kaio_candle::matmul_tc(&kaio_dev, &a, &b)?;
    let c_host: Vec<f32> = c.flatten_all()?.to_vec1::<f32>()?;

    println!("matmul_tc output shape: {:?}", c.shape().dims());
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
