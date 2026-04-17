//! Minimal `kaio_candle::attention_tc` + `attention_tc_causal` example.
//!
//! Builds Q/K/V f16 rank-2 tensors on candle-CUDA and invokes both the
//! non-causal and causal bridge entry points, printing shape + corner
//! values. Shapes are small (well under the kaio-ops `seq_k ≤ 384`
//! shared-memory cap) so the example finishes fast.
//!
//! Build + run:
//! ```sh
//! cd kaio-candle
//! cargo run --release --features cuda --example attention_tc_candle
//! ```

use std::sync::Arc;

use candle_core::{Device, Tensor};
use half::f16;
use kaio::prelude::KaioDevice;

fn main() -> anyhow::Result<()> {
    let candle_dev = Device::new_cuda(0)?;
    let kaio_dev = Arc::new(KaioDevice::new(0)?);

    let seq_q = 64usize;
    let seq_k = 64usize;
    let d_k = 64usize;
    let d_v = 64usize;

    let q_host: Vec<f16> = (0..seq_q * d_k)
        .map(|i| f16::from_f32((i % 19) as f32 * 0.02 - 0.18))
        .collect();
    let k_host: Vec<f16> = (0..seq_k * d_k)
        .map(|i| f16::from_f32((i % 23) as f32 * 0.015 - 0.17))
        .collect();
    let v_host: Vec<f16> = (0..seq_k * d_v)
        .map(|i| f16::from_f32((i % 29) as f32 * 0.01 - 0.14))
        .collect();

    let q = Tensor::from_vec(q_host, (seq_q, d_k), &candle_dev)?;
    let k = Tensor::from_vec(k_host, (seq_k, d_k), &candle_dev)?;
    let v = Tensor::from_vec(v_host, (seq_k, d_v), &candle_dev)?;

    let out_full = kaio_candle::attention_tc(&kaio_dev, &q, &k, &v)?;
    let out_full_host: Vec<f32> = out_full.flatten_all()?.to_vec1::<f32>()?;
    println!("attention_tc output shape: {:?}", out_full.shape().dims());
    println!("  first 8 values: {:?}", &out_full_host[..8]);

    let out_causal = kaio_candle::attention_tc_causal(&kaio_dev, &q, &k, &v)?;
    let out_causal_host: Vec<f32> = out_causal.flatten_all()?.to_vec1::<f32>()?;
    println!(
        "attention_tc_causal output shape: {:?}",
        out_causal.shape().dims()
    );
    println!("  first 8 values: {:?}", &out_causal_host[..8]);

    // Causal-row-0 only sees key 0, so it must differ from the non-causal
    // row-0 (which mixes the full key sequence). A quick sanity check.
    let row0_diff = out_full_host[0] - out_causal_host[0];
    println!(
        "row-0 col-0 non-causal vs causal delta: {:.6} (expected non-zero)",
        row0_diff
    );

    Ok(())
}
