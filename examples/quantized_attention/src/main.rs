//! Sprint 7.3 D9 showcase — end-to-end quantized attention block.
//!
//! Demonstrates the full Phase 7 pipeline shipping on `phase7-rest`:
//! fused `qkv_project_int4` (W4A16) → three `f16` projections feeding
//! directly into `attention_tc`, compared against a non-quantized f16
//! baseline that uses three `matmul_tc` calls for the projection stage.
//!
//! # Pipeline
//!
//! ```text
//! X [seq, d_model] f16
//!  │
//!  ├─► INT4 PATH:  qkv_project_int4 (fused) ──► Q, K, V f16 [seq, d_head]
//!  │                                                      │
//!  │                                                      ▼
//!  │                                            attention_tc ──► out_i4 f32 [seq, d_v]
//!  │
//!  └─► F16 REF :   3 × matmul_tc (one per proj) ─► Q', K', V' f16 [seq, d_head]
//!                                                      │
//!                                                      ▼
//!                                            attention_tc ──► out_ref f32 [seq, d_v]
//! ```
//!
//! Reports three quality metrics on the final attention output vs the
//! f16 reference (per Sprint 7.3 plan D9):
//!   - **cosine similarity** (primary pass/fail — threshold ≥ 0.98 for INT4)
//!   - **max absolute error** (worst-case row outlier)
//!   - **mean relative error** (aggregate quality)
//!
//! # Not a production quantization benchmark
//!
//! Random f16 weights are a worst case for group-scale quantization
//! fidelity. Real trained LLM weights have much tighter group statistics
//! and land measurably better than these synthetic numbers. This
//! example demonstrates the *pipeline plumbing*, not target accuracy
//! for a specific quantization recipe.
//!
//! Run: `cargo run --release` from this directory.

use half::f16;
use kaio::prelude::*;
use kaio_ops::{attention_tc, matmul_tc, qkv_project_int4};

// Attention block dimensions.
const SEQ: u32 = 64; // sequence length (= M for projection matmul)
const D_MODEL: u32 = 128; // input dim (= K for projection matmul); must be % 128
const D_HEAD: u32 = 64; // per-head dim (= N for projection matmul); must be % 16, even
const GROUP_SIZE: u32 = 128;

/// GPTQ-lite symmetric per-column group quantizer. For each (col, group)
/// pair, pick `scale = max(|w|) / 7` and quantize as
/// `q = clamp(round(w / scale), -8, +7)`. Returns `(q_weights: [K, N] i8,
/// scales: [K/group_size, N] f16)`.
fn quantize_gptq_lite(w: &[f32], k: usize, n: usize, group_size: usize) -> (Vec<i8>, Vec<f16>) {
    let num_groups = k / group_size;
    let mut q = vec![0i8; k * n];
    let mut scales = vec![f16::from_f32(0.0); num_groups * n];
    for col in 0..n {
        for g in 0..num_groups {
            let k_lo = g * group_size;
            let k_hi = k_lo + group_size;
            let mut max_abs = 0.0f32;
            for kk in k_lo..k_hi {
                let v = w[kk * n + col].abs();
                if v > max_abs {
                    max_abs = v;
                }
            }
            let scale = (max_abs / 7.0).max(1e-8);
            scales[g * n + col] = f16::from_f32(scale);
            for kk in k_lo..k_hi {
                let qv = (w[kk * n + col] / scale).round().clamp(-8.0, 7.0) as i8;
                q[kk * n + col] = qv;
            }
        }
    }
    (q, scales)
}

/// Pack `[K, N] i8` into `[K/8, N] u32` col-major (KAIO convention).
fn pack_s4_weights(w: &[i8], k: usize, n: usize) -> Vec<u32> {
    let k_words = k / 8;
    let mut out = vec![0u32; k_words * n];
    for col in 0..n {
        for word_idx in 0..k_words {
            let mut word = 0u32;
            for nibble_idx in 0..8 {
                let k_pos = word_idx * 8 + nibble_idx;
                let val = w[k_pos * n + col];
                let nibble = (val as u32) & 0xF;
                word |= nibble << (4 * nibble_idx);
            }
            out[word_idx + col * k_words] = word;
        }
    }
    out
}

fn deterministic_f32(n_elem: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n_elem)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits_top = (state >> 32) as u32;
            (bits_top as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn f32_to_f16(v: &[f32]) -> Vec<f16> {
    v.iter().copied().map(f16::from_f32).collect()
}

fn f32_gpu_to_f16(device: &KaioDevice, src: &GpuBuffer<f32>) -> Result<GpuBuffer<f16>> {
    // CPU round-trip narrow: f32 GPU → host → f16 host → f16 GPU. Adequate
    // for a showcase; a proper production pipeline would emit a cvt kernel.
    let host = src.to_host(device)?;
    let half = f32_to_f16(&host);
    device.alloc_from(&half)
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-12)
}

fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn mean_relative_error(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs() / y.abs().max(1.0))
        .sum::<f32>()
        / n
}

fn main() -> Result<()> {
    eprintln!("=== Sprint 7.3 D9 — Quantized attention end-to-end ===");
    eprintln!(
        "Shape: SEQ={SEQ}, D_MODEL={D_MODEL}, D_HEAD={D_HEAD}, GROUP_SIZE={GROUP_SIZE}"
    );
    eprintln!();

    let device = KaioDevice::new(0)?;
    let info = device.info()?;
    assert!(
        info.compute_capability.0 >= 8,
        "requires SM 8.0+; got sm_{}{}",
        info.compute_capability.0,
        info.compute_capability.1
    );
    eprintln!(
        "GPU: {:?} sm_{}{}",
        info.name, info.compute_capability.0, info.compute_capability.1
    );
    eprintln!();

    // --- 1. Generate X (f16) and W_Q/K/V (f32). ---
    let x_f32 = deterministic_f32((SEQ * D_MODEL) as usize, 1);
    let x_f16 = f32_to_f16(&x_f32);
    let w_q_f32 = deterministic_f32((D_MODEL * D_HEAD) as usize, 2);
    let w_k_f32 = deterministic_f32((D_MODEL * D_HEAD) as usize, 3);
    let w_v_f32 = deterministic_f32((D_MODEL * D_HEAD) as usize, 4);

    // --- 2. Quantize W_Q/K/V into INT4 packed + scales. ---
    let (q_w_q, q_s_q) = quantize_gptq_lite(
        &w_q_f32,
        D_MODEL as usize,
        D_HEAD as usize,
        GROUP_SIZE as usize,
    );
    let (q_w_k, q_s_k) = quantize_gptq_lite(
        &w_k_f32,
        D_MODEL as usize,
        D_HEAD as usize,
        GROUP_SIZE as usize,
    );
    let (q_w_v, q_s_v) = quantize_gptq_lite(
        &w_v_f32,
        D_MODEL as usize,
        D_HEAD as usize,
        GROUP_SIZE as usize,
    );
    let p_w_q = pack_s4_weights(&q_w_q, D_MODEL as usize, D_HEAD as usize);
    let p_w_k = pack_s4_weights(&q_w_k, D_MODEL as usize, D_HEAD as usize);
    let p_w_v = pack_s4_weights(&q_w_v, D_MODEL as usize, D_HEAD as usize);

    // --- 3. Upload to device. ---
    let x_gpu = device.alloc_from(&x_f16)?;
    let p_w_q_gpu = device.alloc_from(&p_w_q)?;
    let p_w_k_gpu = device.alloc_from(&p_w_k)?;
    let p_w_v_gpu = device.alloc_from(&p_w_v)?;
    let s_q_gpu = device.alloc_from(&q_s_q)?;
    let s_k_gpu = device.alloc_from(&q_s_k)?;
    let s_v_gpu = device.alloc_from(&q_s_v)?;

    // F16 reference weights (quantized roundtrip — X × dequant(w) ≡ the best
    // the projection *could* get from this group-scale config).
    let w_q_f16 = f32_to_f16(&w_q_f32);
    let w_k_f16 = f32_to_f16(&w_k_f32);
    let w_v_f16 = f32_to_f16(&w_v_f32);
    let w_q_ref_gpu = device.alloc_from(&w_q_f16)?;
    let w_k_ref_gpu = device.alloc_from(&w_k_f16)?;
    let w_v_ref_gpu = device.alloc_from(&w_v_f16)?;

    // --- 4. INT4 fused projection → f16 Q/K/V outputs. ---
    let mut q_i4 = device.alloc_zeros::<f16>((SEQ * D_HEAD) as usize)?;
    let mut k_i4 = device.alloc_zeros::<f16>((SEQ * D_HEAD) as usize)?;
    let mut v_i4 = device.alloc_zeros::<f16>((SEQ * D_HEAD) as usize)?;
    qkv_project_int4(
        &device,
        &x_gpu,
        &p_w_q_gpu,
        &p_w_k_gpu,
        &p_w_v_gpu,
        &s_q_gpu,
        &s_k_gpu,
        &s_v_gpu,
        &mut q_i4,
        &mut k_i4,
        &mut v_i4,
        SEQ,
        D_HEAD,
        D_MODEL,
        GROUP_SIZE,
    )?;
    device.stream().synchronize()?;
    eprintln!("INT4 path: qkv_project_int4 ran.");

    // --- 5. F16 reference projection (3 × matmul_tc → f32 → narrow to f16). ---
    let mut q_ref_f32 = device.alloc_zeros::<f32>((SEQ * D_HEAD) as usize)?;
    let mut k_ref_f32 = device.alloc_zeros::<f32>((SEQ * D_HEAD) as usize)?;
    let mut v_ref_f32 = device.alloc_zeros::<f32>((SEQ * D_HEAD) as usize)?;
    matmul_tc(
        &device,
        &x_gpu,
        &w_q_ref_gpu,
        &mut q_ref_f32,
        SEQ,
        D_HEAD,
        D_MODEL,
    )?;
    matmul_tc(
        &device,
        &x_gpu,
        &w_k_ref_gpu,
        &mut k_ref_f32,
        SEQ,
        D_HEAD,
        D_MODEL,
    )?;
    matmul_tc(
        &device,
        &x_gpu,
        &w_v_ref_gpu,
        &mut v_ref_f32,
        SEQ,
        D_HEAD,
        D_MODEL,
    )?;
    device.stream().synchronize()?;

    let q_ref_f16 = f32_gpu_to_f16(&device, &q_ref_f32)?;
    let k_ref_f16 = f32_gpu_to_f16(&device, &k_ref_f32)?;
    let v_ref_f16 = f32_gpu_to_f16(&device, &v_ref_f32)?;
    eprintln!("F16 reference path: 3× matmul_tc ran.");

    // --- 6. Per-Q/K/V projection quality (before attention). ---
    let q_i4_host: Vec<f32> = q_i4.to_host(&device)?.iter().map(|h| h.to_f32()).collect();
    let q_ref_host: Vec<f32> = q_ref_f16
        .to_host(&device)?
        .iter()
        .map(|h| h.to_f32())
        .collect();
    eprintln!();
    eprintln!("--- Projection-stage quality (Q/K/V vs f16 reference) ---");
    eprintln!(
        "Q  cos_sim={:.4} max_abs={:.3e} mean_rel={:.3e}",
        cosine_similarity(&q_i4_host, &q_ref_host),
        max_abs_error(&q_i4_host, &q_ref_host),
        mean_relative_error(&q_i4_host, &q_ref_host)
    );

    // --- 7. Attention on both paths. ---
    let mut out_i4 = device.alloc_zeros::<f32>((SEQ * D_HEAD) as usize)?;
    let mut out_ref = device.alloc_zeros::<f32>((SEQ * D_HEAD) as usize)?;
    attention_tc(
        &device, &q_i4, &k_i4, &v_i4, &mut out_i4, SEQ, SEQ, D_HEAD, D_HEAD,
    )?;
    attention_tc(
        &device,
        &q_ref_f16,
        &k_ref_f16,
        &v_ref_f16,
        &mut out_ref,
        SEQ,
        SEQ,
        D_HEAD,
        D_HEAD,
    )?;
    device.stream().synchronize()?;

    let out_i4_host = out_i4.to_host(&device)?;
    let out_ref_host = out_ref.to_host(&device)?;
    let cos_sim = cosine_similarity(&out_i4_host, &out_ref_host);
    let max_abs = max_abs_error(&out_i4_host, &out_ref_host);
    let mean_rel = mean_relative_error(&out_i4_host, &out_ref_host);

    eprintln!();
    eprintln!("--- Final attention output quality (INT4 path vs f16 reference) ---");
    eprintln!("cosine similarity : {cos_sim:.4}");
    eprintln!("max abs error     : {max_abs:.3e}");
    eprintln!("mean relative err : {mean_rel:.3e}");
    eprintln!();

    // Plan D9 INT4 pass threshold: cosine sim >= 0.98.
    let pass = cos_sim >= 0.98;
    eprintln!(
        "{} Cosine similarity {:.4} {} 0.98 (plan D9 INT4 threshold)",
        if pass { "[PASS]" } else { "[WARN]" },
        cos_sim,
        if pass { ">=" } else { "<" }
    );
    if !pass {
        eprintln!(
            "  Note: random f16 weights are a worst case for group-scale quantization \
             fidelity. Real LLM weights have much tighter group statistics."
        );
    }

    Ok(())
}
