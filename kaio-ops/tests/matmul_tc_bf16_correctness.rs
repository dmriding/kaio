//! GPU correctness tests for `matmul_tc_bf16` (Sprint 9.1).
//!
//! Full D5 grid: shape × magnitude with shape-scoped reference strategy.
//!
//! | Shape class           | Magnitudes run                            | Reference        |
//! |-----------------------|-------------------------------------------|------------------|
//! | Small (32³, 64³)      | small / medium / large / near-denorm      | dense f64        |
//! | Medium (256³, 512³)   | small / medium / large / near-denorm      | dense f64        |
//! | Large 2048³           | small + one large smoke                   | sampled-cell f64 |
//! | Large 4096³           | small only (also the bench shape)         | sampled-cell f64 |
//! | Non-square / odd-N    | small only (edge-tile predication)        | dense f64        |
//!
//! Total: 8 small + 8 medium + 2 large 2048³ + 1 large 4096³ + 2 non-square = 21 tests.
//!
//! Magnitudes:
//! - **small:** `[-0.5, 0.5]` patterned (within `[-1, 1]`).
//! - **medium:** patterned × 200 → peak `|x| ≈ 100`.
//! - **large:** patterned × 2e14 → peak `|x| ≈ 1e14` — caps at the
//!   level that keeps `|a| × |b| × K` under f32 max in the accumulator
//!   while exercising bf16's full f32-style exponent range.
//! - **near-denorm:** positive-only `[1e-18, ~1.94e-18]` — the
//!   positive bias keeps products from cancelling to zero so the
//!   nonzero-output assertion has signal.
//!
//! Tolerances:
//! - **standard (small/medium/large mag):** `rel_err < 1e-2 || abs_err < 1e-3`.
//! - **near-denorm:** `rel_err < 1e-1` only + nonzero-output assertion
//!   (the latter is the "kernel returns zero on small inputs" canary).

use kaio::prelude::*;
use kaio_ops::matmul_tc_bf16;

mod common;
use common::{
    assert_bf16_close_d5, assert_bf16_close_d5_near_denorm, assert_bf16_close_d5_sampled,
    cpu_matmul_bf16xbf16_f64, patterned_bf16_data, patterned_bf16_magnitude,
    patterned_bf16_near_denorm, sample_cells, sampled_cell_f64_reference,
};

// ----------------------------------------------------------------------------
// Test harnesses
// ----------------------------------------------------------------------------

fn require_gpu_ampere(label: &str) -> KaioDevice {
    let device = KaioDevice::new(0).expect("GPU required");
    let info = device.info().expect("device info");
    assert!(
        info.compute_capability.0 >= 8,
        "{label}: matmul_tc_bf16 requires SM 8.0+ (have sm_{}{})",
        info.compute_capability.0,
        info.compute_capability.1
    );
    device
}

fn launch_bf16(
    device: &KaioDevice,
    a_host: &[half::bf16],
    b_host: &[half::bf16],
    m: usize,
    n: usize,
    k: usize,
    label: &str,
) -> Vec<f32> {
    let a = device.alloc_from(a_host).expect("alloc A");
    let b = device.alloc_from(b_host).expect("alloc B");
    let mut c = device.alloc_zeros::<f32>(m * n).expect("alloc C");
    matmul_tc_bf16(device, &a, &b, &mut c, m as u32, n as u32, k as u32)
        .unwrap_or_else(|e| panic!("{label}: matmul_tc_bf16 failed: {e}"));
    c.to_host(device).expect("C to host")
}

/// Dense f64 reference + D5 standard tolerance. Used at small/medium
/// shapes for all four magnitude classes (modulo near-denorm which
/// uses [`run_dense_near_denorm`]).
fn run_dense(
    m: usize,
    n: usize,
    k: usize,
    a_host: Vec<half::bf16>,
    b_host: Vec<half::bf16>,
    label: &str,
) {
    let device = require_gpu_ampere(label);
    let got = launch_bf16(&device, &a_host, &b_host, m, n, k, label);
    let expected = cpu_matmul_bf16xbf16_f64(&a_host, &b_host, m, n, k);
    assert_bf16_close_d5(&got, &expected, m, n, label);
}

/// Dense f64 reference + D5 near-denorm tolerance + nonzero-output gate.
fn run_dense_near_denorm(m: usize, n: usize, k: usize, label: &str) {
    let a_host = patterned_bf16_near_denorm(m * k);
    let b_host = patterned_bf16_near_denorm(k * n);
    let device = require_gpu_ampere(label);
    let got = launch_bf16(&device, &a_host, &b_host, m, n, k, label);
    let expected = cpu_matmul_bf16xbf16_f64(&a_host, &b_host, m, n, k);
    assert_bf16_close_d5_near_denorm(&got, &expected, m, n, label);
}

/// Sampled-cell f64 reference + D5 standard tolerance. Used at large
/// shapes (2048³, 4096³) where a dense f64 reference would be
/// prohibitive (~68B FMAs at 4096³).
#[allow(clippy::too_many_arguments)]
fn run_sampled(
    m: usize,
    n: usize,
    k: usize,
    a_host: Vec<half::bf16>,
    b_host: Vec<half::bf16>,
    n_samples: usize,
    seed: u64,
    label: &str,
) {
    let device = require_gpu_ampere(label);
    let got = launch_bf16(&device, &a_host, &b_host, m, n, k, label);
    let cells = sample_cells(m, n, n_samples, seed);
    let expected = sampled_cell_f64_reference(&a_host, &b_host, m, n, k, &cells);
    assert_bf16_close_d5_sampled(&got, &cells, &expected, m, n, label);
}

// ----------------------------------------------------------------------------
// Small shapes (32³, 64³) — all four magnitude classes, dense f64 reference
// ----------------------------------------------------------------------------

#[test]
#[ignore]
fn tc_bf16_32_32_32_small() {
    let (m, n, k) = (32, 32, 32);
    run_dense(
        m,
        n,
        k,
        patterned_bf16_data(m * k),
        patterned_bf16_data(k * n),
        "32_32_32_small",
    );
}

#[test]
#[ignore]
fn tc_bf16_32_32_32_medium() {
    let (m, n, k) = (32, 32, 32);
    run_dense(
        m,
        n,
        k,
        patterned_bf16_magnitude(m * k, 200.0),
        patterned_bf16_magnitude(k * n, 200.0),
        "32_32_32_medium",
    );
}

#[test]
#[ignore]
fn tc_bf16_32_32_32_large() {
    let (m, n, k) = (32, 32, 32);
    run_dense(
        m,
        n,
        k,
        patterned_bf16_magnitude(m * k, 2e14),
        patterned_bf16_magnitude(k * n, 2e14),
        "32_32_32_large",
    );
}

#[test]
#[ignore]
fn tc_bf16_32_32_32_near_denorm() {
    run_dense_near_denorm(32, 32, 32, "32_32_32_near_denorm");
}

#[test]
#[ignore]
fn tc_bf16_64_64_64_small() {
    let (m, n, k) = (64, 64, 64);
    run_dense(
        m,
        n,
        k,
        patterned_bf16_data(m * k),
        patterned_bf16_data(k * n),
        "64_64_64_small",
    );
}

#[test]
#[ignore]
fn tc_bf16_64_64_64_medium() {
    let (m, n, k) = (64, 64, 64);
    run_dense(
        m,
        n,
        k,
        patterned_bf16_magnitude(m * k, 200.0),
        patterned_bf16_magnitude(k * n, 200.0),
        "64_64_64_medium",
    );
}

#[test]
#[ignore]
fn tc_bf16_64_64_64_large() {
    let (m, n, k) = (64, 64, 64);
    run_dense(
        m,
        n,
        k,
        patterned_bf16_magnitude(m * k, 2e14),
        patterned_bf16_magnitude(k * n, 2e14),
        "64_64_64_large",
    );
}

#[test]
#[ignore]
fn tc_bf16_64_64_64_near_denorm() {
    run_dense_near_denorm(64, 64, 64, "64_64_64_near_denorm");
}

// ----------------------------------------------------------------------------
// Medium shapes (256³, 512³) — all four magnitude classes, dense f64 reference
// ----------------------------------------------------------------------------

#[test]
#[ignore]
fn tc_bf16_256_256_256_small() {
    let (m, n, k) = (256, 256, 256);
    run_dense(
        m,
        n,
        k,
        patterned_bf16_data(m * k),
        patterned_bf16_data(k * n),
        "256_256_256_small",
    );
}

#[test]
#[ignore]
fn tc_bf16_256_256_256_medium() {
    let (m, n, k) = (256, 256, 256);
    run_dense(
        m,
        n,
        k,
        patterned_bf16_magnitude(m * k, 200.0),
        patterned_bf16_magnitude(k * n, 200.0),
        "256_256_256_medium",
    );
}

#[test]
#[ignore]
fn tc_bf16_256_256_256_large() {
    let (m, n, k) = (256, 256, 256);
    run_dense(
        m,
        n,
        k,
        patterned_bf16_magnitude(m * k, 2e14),
        patterned_bf16_magnitude(k * n, 2e14),
        "256_256_256_large",
    );
}

#[test]
#[ignore]
fn tc_bf16_256_256_256_near_denorm() {
    run_dense_near_denorm(256, 256, 256, "256_256_256_near_denorm");
}

#[test]
#[ignore]
fn tc_bf16_512_512_512_small() {
    let (m, n, k) = (512, 512, 512);
    run_dense(
        m,
        n,
        k,
        patterned_bf16_data(m * k),
        patterned_bf16_data(k * n),
        "512_512_512_small",
    );
}

#[test]
#[ignore]
fn tc_bf16_512_512_512_medium() {
    let (m, n, k) = (512, 512, 512);
    run_dense(
        m,
        n,
        k,
        patterned_bf16_magnitude(m * k, 200.0),
        patterned_bf16_magnitude(k * n, 200.0),
        "512_512_512_medium",
    );
}

#[test]
#[ignore]
fn tc_bf16_512_512_512_large() {
    let (m, n, k) = (512, 512, 512);
    run_dense(
        m,
        n,
        k,
        patterned_bf16_magnitude(m * k, 2e14),
        patterned_bf16_magnitude(k * n, 2e14),
        "512_512_512_large",
    );
}

#[test]
#[ignore]
fn tc_bf16_512_512_512_near_denorm() {
    run_dense_near_denorm(512, 512, 512, "512_512_512_near_denorm");
}

// ----------------------------------------------------------------------------
// Large 2048³ — small magnitude + one large-magnitude smoke, sampled-cell f64
// ----------------------------------------------------------------------------

const LARGE_N_SAMPLES: usize = 100;
const LARGE_SAMPLE_SEED: u64 = 0x9E37_79B9_7F4A_7C15; // Knuth golden ratio constant

#[test]
#[ignore]
fn tc_bf16_2048_2048_2048_small() {
    let (m, n, k) = (2048, 2048, 2048);
    run_sampled(
        m,
        n,
        k,
        patterned_bf16_data(m * k),
        patterned_bf16_data(k * n),
        LARGE_N_SAMPLES,
        LARGE_SAMPLE_SEED,
        "2048_2048_2048_small",
    );
}

#[test]
#[ignore]
fn tc_bf16_2048_2048_2048_large() {
    let (m, n, k) = (2048, 2048, 2048);
    run_sampled(
        m,
        n,
        k,
        patterned_bf16_magnitude(m * k, 2e14),
        patterned_bf16_magnitude(k * n, 2e14),
        LARGE_N_SAMPLES,
        LARGE_SAMPLE_SEED,
        "2048_2048_2048_large",
    );
}

// ----------------------------------------------------------------------------
// Large 4096³ — small magnitude only (also the bench shape), sampled-cell f64
// ----------------------------------------------------------------------------

#[test]
#[ignore]
fn tc_bf16_4096_4096_4096_small() {
    let (m, n, k) = (4096, 4096, 4096);
    run_sampled(
        m,
        n,
        k,
        patterned_bf16_data(m * k),
        patterned_bf16_data(k * n),
        LARGE_N_SAMPLES,
        LARGE_SAMPLE_SEED,
        "4096_4096_4096_small",
    );
}

// ----------------------------------------------------------------------------
// Non-square / odd-N — small magnitude only, dense f64 (shapes are small)
// ----------------------------------------------------------------------------

/// Non-square shape from D5: A is 64×32, B is 32×128 → M=64, N=128, K=32.
/// Exercises edge-tile predication on the N axis (128 % 64 == 0 but the
/// 2×N-block path is structurally different from the square 64×64 case).
#[test]
#[ignore]
fn tc_bf16_64_128_32_small() {
    let (m, n, k) = (64, 128, 32);
    run_dense(
        m,
        n,
        k,
        patterned_bf16_data(m * k),
        patterned_bf16_data(k * n),
        "64_128_32_small",
    );
}

/// Odd-N shape: M=65, N=17, K=32. K=32 satisfies the `K % 16 == 0`
/// kernel constraint; M and N are deliberately small odd values to
/// stress edge-tile predication in both row and column directions.
#[test]
#[ignore]
fn tc_bf16_65_17_32_small() {
    let (m, n, k) = (65, 17, 32);
    run_dense(
        m,
        n,
        k,
        patterned_bf16_data(m * k),
        patterned_bf16_data(k * n),
        "65_17_32_small",
    );
}
