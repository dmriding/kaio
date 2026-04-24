#![allow(clippy::too_many_arguments)]

//! Benchmark: unified harness for normalization + activation showcase
//! kernels — `rms_norm`, `layer_norm`, `softmax`, `fused_silu_gate`,
//! `gelu_exact`, `gelu_fast`.
//!
//! Run with:
//! ```sh
//! cargo test -p kaio-ops --test norm_activation_bench -- --ignored --nocapture
//! ```
//!
//! # What is measured
//!
//! Latency (median µs) + throughput (elems/s) + effective bandwidth
//! (GB/s, model-level bytes ÷ time) for each kernel. No cuBLAS / cuDNN
//! baseline — these are bandwidth-bound elementwise and single-block
//! reduction kernels without a natural library reference in `cudarc`.
//!
//! # Two shape regimes
//!
//! The kernels split into two groups by how they use the block:
//!
//! - **Reductions** (`rms_norm`, `layer_norm`, `softmax`): single-block
//!   by construction — they use `block_reduce_sum` / `block_reduce_max`,
//!   which is block-scope. `n` is capped at `block_size = 256`. These
//!   kernels are measured once per kernel at `n=256`; the number
//!   captures launch overhead + the reduction's cost at that scale.
//!   Multi-block versions of these are a future Ops Track item.
//! - **Elementwise** (`fused_silu_gate`, `gelu_exact`, `gelu_fast`):
//!   fully multi-block (`block_idx_x() * block_dim_x()` addressing).
//!   Swept across `{256K, 1M, 4M}` elements for a bandwidth-saturation
//!   curve.
//!
//! # "Effective GB/s" framing
//!
//! The reported bandwidth column is **model-level** — how many bytes
//! the kernel logically reads + writes once, divided by wall-clock.
//! It is **not** a measurement of achieved HBM bandwidth; reductions
//! re-read from shared memory during the reduction tree, and the
//! driver's L2 / shared-memory reuse can lower the actual HBM traffic
//! below the model number. Per-kernel byte accounting is documented
//! alongside each row.
//!
//! # Pointer syntax
//!
//! Each copied kernel below uses `*const [T]` / `*mut [T]` pointer
//! syntax, preserved from the example-crate sources migrated in
//! Sprint 8.0 (RFC-0001). That makes this bench a
//! perf-under-measurement proof-of-life for the pointer-syntax change
//! on top of the correctness smoke test at
//! `kaio/tests/macro_pointer_smoke.rs`.

use std::time::Instant;

use kaio::prelude::*;

// Note: `kaio_ops::softmax` is the public Phase-3 op; this bench uses a
// copied-from-example single-block `softmax` below so the row is
// apples-to-apples with the other showcase kernels (same block_size,
// same macro path, same SYNC-from-example provenance).

// SYNC: copied from examples/rms_norm/src/main.rs — keep in sync.
#[gpu_kernel(block_size = 256)]
fn rms_norm(x: *const [f32], weight: *const [f32], out: *mut [f32], n: u32, eps: f32) {
    let tid = thread_idx_x();
    let mut val = 0.0f32;
    if tid < n {
        val = x[tid];
    }
    let sq = val * val;
    let sum_sq = block_reduce_sum(sq);
    let inv_rms = 1.0f32 / sqrt(sum_sq / (n as f32) + eps);
    if tid < n {
        out[tid] = val * inv_rms * weight[tid];
    }
}

// SYNC: copied from examples/layer_norm/src/main.rs — keep in sync.
#[gpu_kernel(block_size = 256)]
fn layer_norm(
    x: *const [f32],
    gamma: *const [f32],
    beta: *const [f32],
    out: *mut [f32],
    n: u32,
    eps: f32,
) {
    let tid = thread_idx_x();
    let mut val = 0.0f32;
    if tid < n {
        val = x[tid];
    }
    let sum = block_reduce_sum(val);
    let mean = sum / (n as f32);
    let mut centered = 0.0f32;
    if tid < n {
        centered = val - mean;
    }
    let var_sum = block_reduce_sum(centered * centered);
    let inv_std = 1.0f32 / sqrt(var_sum / (n as f32) + eps);
    if tid < n {
        out[tid] = centered * inv_std * gamma[tid] + beta[tid];
    }
}

// SYNC: copied from examples/softmax/src/main.rs — keep in sync.
#[gpu_kernel(block_size = 256)]
fn softmax(input: *const [f32], output: *mut [f32], n: u32) {
    let tid = thread_idx_x();
    let mut local_max = -3.402823e38f32;
    if tid < n {
        local_max = input[tid];
    }
    let row_max = block_reduce_max(local_max);
    let mut exp_val = 0.0f32;
    if tid < n {
        exp_val = exp(local_max - row_max);
    }
    let row_sum = block_reduce_sum(exp_val);
    if tid < n {
        output[tid] = exp_val / row_sum;
    }
}

// SYNC: copied from examples/fused_silu_gate/src/main.rs — keep in sync.
#[gpu_kernel(block_size = 256)]
fn fused_silu_gate(x: *const [f32], gate: *const [f32], out: *mut [f32], n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        let xi = x[idx];
        let sig = 1.0f32 / (1.0f32 + exp(-xi));
        out[idx] = xi * sig * gate[idx];
    }
}

// SYNC: copied from examples/gelu_comparison/src/main.rs — keep in sync.
#[gpu_kernel(block_size = 256)]
fn gelu_exact(x: *const [f32], out: *mut [f32], n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        let xi = x[idx];
        let c: f32 = 0.7978845608028654f32; // sqrt(2/π)
        let inner = c * (xi + 0.044715f32 * xi * xi * xi);
        out[idx] = 0.5f32 * xi * (1.0f32 + tanh(inner));
    }
}

// SYNC: copied from examples/gelu_comparison/src/main.rs — keep in sync.
#[gpu_kernel(block_size = 256)]
fn gelu_fast(x: *const [f32], out: *mut [f32], n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        let xi = x[idx];
        out[idx] = xi / (1.0f32 + exp(-1.702f32 * xi));
    }
}

fn deterministic_f32(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0
        })
        .collect()
}

fn median_seconds<F: FnMut()>(mut launch: F, warmup: usize, iters: usize) -> f64 {
    for _ in 0..warmup {
        launch();
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        launch();
        times.push(start.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[iters / 2]
}

fn print_header() {
    eprintln!(
        "{:<18} {:>12} {:>12} {:>14} {:>14}",
        "kernel", "n", "median µs", "Gelems/s", "eff GB/s"
    );
    eprintln!("{}", "-".repeat(76));
}

fn print_row(kernel: &str, n: u32, median_s: f64, bytes_per_invoke: f64) {
    let elems_per_s = (n as f64) / median_s;
    let eff_bw = bytes_per_invoke / median_s / 1e9;
    eprintln!(
        "{:<18} {:>12} {:>12.2} {:>14.3} {:>14.2}",
        kernel,
        n,
        median_s * 1e6,
        elems_per_s / 1e9,
        eff_bw,
    );
}

const REDUCTION_N: u32 = 256;

const ELEMENTWISE_SHAPES: &[u32] = &[
    1 << 18, // 262,144   (small — 1 MiB f32)
    1 << 20, // 1,048,576 (medium — 4 MiB f32)
    1 << 22, // 4,194,304 (large — 16 MiB f32)
];

const EPS_RMS: f32 = 1e-6;
const EPS_LN: f32 = 1e-5;

#[test]
#[ignore]
fn benchmark_norm_activation() {
    let device = KaioDevice::new(0).expect("GPU required");
    let info = device.info().expect("device info");

    eprintln!();
    eprintln!("=== KAIO norm + activation benchmark (single-head, f32) ===");
    eprintln!(
        "GPU: {:?}  sm_{}{}",
        info.name, info.compute_capability.0, info.compute_capability.1
    );
    eprintln!("Warmup 5 | Iters 20 | Median | Effective bandwidth = model bytes / time");
    eprintln!();

    // --- Reductions (single-block, n fixed at 256) ---

    eprintln!("--- Reductions (single-block, n = {REDUCTION_N}) ---");
    print_header();

    let n = REDUCTION_N;
    let nz = n as usize;

    let x_rms = device.alloc_from(&deterministic_f32(nz, 1)).unwrap();
    let w_rms = device.alloc_from(&deterministic_f32(nz, 2)).unwrap();
    let mut out_rms = device.alloc_zeros::<f32>(nz).unwrap();
    let rms_s = median_seconds(
        || {
            rms_norm::launch(&device, &x_rms, &w_rms, &mut out_rms, n, EPS_RMS).unwrap();
            device.stream().synchronize().unwrap();
        },
        5,
        20,
    );
    // rms_norm: reads x[N] + weight[N], writes out[N] → 3N f32 = 12N bytes.
    print_row("rms_norm", n, rms_s, 12.0 * n as f64);

    let x_ln = device.alloc_from(&deterministic_f32(nz, 3)).unwrap();
    let g_ln = device.alloc_from(&deterministic_f32(nz, 4)).unwrap();
    let b_ln = device.alloc_from(&deterministic_f32(nz, 5)).unwrap();
    let mut out_ln = device.alloc_zeros::<f32>(nz).unwrap();
    let ln_s = median_seconds(
        || {
            layer_norm::launch(&device, &x_ln, &g_ln, &b_ln, &mut out_ln, n, EPS_LN).unwrap();
            device.stream().synchronize().unwrap();
        },
        5,
        20,
    );
    // layer_norm: reads x[N] + gamma[N] + beta[N], writes out[N] → 4N f32 = 16N bytes.
    print_row("layer_norm", n, ln_s, 16.0 * n as f64);

    let x_sm = device.alloc_from(&deterministic_f32(nz, 6)).unwrap();
    let mut out_sm = device.alloc_zeros::<f32>(nz).unwrap();
    let sm_s = median_seconds(
        || {
            softmax::launch(&device, &x_sm, &mut out_sm, n).unwrap();
            device.stream().synchronize().unwrap();
        },
        5,
        20,
    );
    // softmax: reads input[N], writes output[N] → 2N f32 = 8N bytes.
    print_row("softmax", n, sm_s, 8.0 * n as f64);

    // --- Elementwise (multi-block, sweep n) ---

    eprintln!();
    eprintln!("--- Elementwise (multi-block, sweep) ---");
    print_header();

    for &n in ELEMENTWISE_SHAPES {
        let nz = n as usize;
        let x = device.alloc_from(&deterministic_f32(nz, 10)).unwrap();
        let gate = device.alloc_from(&deterministic_f32(nz, 11)).unwrap();
        let mut out = device.alloc_zeros::<f32>(nz).unwrap();

        let silu_s = median_seconds(
            || {
                fused_silu_gate::launch(&device, &x, &gate, &mut out, n).unwrap();
                device.stream().synchronize().unwrap();
            },
            5,
            20,
        );
        // fused_silu_gate: reads x[N] + gate[N], writes out[N] → 3N f32 = 12N bytes.
        print_row("fused_silu_gate", n, silu_s, 12.0 * n as f64);

        let exact_s = median_seconds(
            || {
                gelu_exact::launch(&device, &x, &mut out, n).unwrap();
                device.stream().synchronize().unwrap();
            },
            5,
            20,
        );
        // gelu_exact: reads x[N], writes out[N] → 2N f32 = 8N bytes.
        print_row("gelu_exact", n, exact_s, 8.0 * n as f64);

        let fast_s = median_seconds(
            || {
                gelu_fast::launch(&device, &x, &mut out, n).unwrap();
                device.stream().synchronize().unwrap();
            },
            5,
            20,
        );
        // gelu_fast: reads x[N], writes out[N] → 2N f32 = 8N bytes.
        print_row("gelu_fast", n, fast_s, 8.0 * n as f64);
    }

    eprintln!();
    eprintln!("Byte accounting (model-level read + write, one pass each):");
    eprintln!("  rms_norm         12N bytes  (x + weight + out)");
    eprintln!("  layer_norm       16N bytes  (x + gamma + beta + out)");
    eprintln!("  softmax           8N bytes  (input + output)");
    eprintln!("  fused_silu_gate  12N bytes  (x + gate + out)");
    eprintln!("  gelu_exact        8N bytes  (x + out)");
    eprintln!("  gelu_fast         8N bytes  (x + out)");
}
