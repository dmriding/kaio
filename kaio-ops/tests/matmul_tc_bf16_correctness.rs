//! GPU correctness tests for `matmul_tc_bf16` (Sprint 9.1).
//!
//! C4 (this commit) ships the smallest cell of the D5 correctness
//! matrix — the 64×64×64 / small-magnitude smoke that confirms the
//! kernel computes the right answer end-to-end on real hardware.
//! The remaining ~20 tests across the D5 shape × magnitude grid
//! (small / medium / large / non-square × small / medium / large /
//! near-denorm magnitudes, with shape-scoped reference strategy)
//! land at C5.
//!
//! Reference: dense CPU f64 matmul per D5 ("Reference strategy by
//! shape" — dense f64 for small/medium shapes). Tolerance: D5
//! "standard" bound `rel_err < 1e-2 || abs_err < 1e-3`.

use kaio::prelude::*;
use kaio_ops::matmul_tc_bf16;

mod common;
use common::{assert_bf16_close_d5, cpu_matmul_bf16xbf16_f64, patterned_bf16_data};

fn run_matmul_tc_bf16_test(m: usize, n: usize, k: usize, label: &str) {
    let device = KaioDevice::new(0).expect("GPU required");
    let info = device.info().expect("device info");
    assert!(
        info.compute_capability.0 >= 8,
        "{label}: matmul_tc_bf16 requires SM 8.0+ (have sm_{}{})",
        info.compute_capability.0,
        info.compute_capability.1
    );

    let a_host = patterned_bf16_data(m * k);
    let b_host = patterned_bf16_data(k * n);

    let a = device.alloc_from(&a_host).expect("alloc A");
    let b = device.alloc_from(&b_host).expect("alloc B");
    let mut c = device.alloc_zeros::<f32>(m * n).expect("alloc C");

    matmul_tc_bf16(&device, &a, &b, &mut c, m as u32, n as u32, k as u32)
        .unwrap_or_else(|e| panic!("{label}: matmul_tc_bf16 failed: {e}"));

    let got = c.to_host(&device).expect("C to host");
    let expected = cpu_matmul_bf16xbf16_f64(&a_host, &b_host, m, n, k);
    assert_bf16_close_d5(&got, &expected, m, n, label);
}

/// Sprint 9.1 C4 — first GPU correctness signal for matmul_tc_bf16.
/// 64×64×64, magnitude `[-0.5, 0.5]` (patterned), small enough that
/// a single 64×64 block covers the whole output (all 4 warps in
/// quadrants, K-loop runs 4 iterations of 16-element tiles).
#[test]
#[ignore] // requires NVIDIA GPU (SM 8.0+)
fn tc_bf16_matmul_smoke_64_64_64() {
    run_matmul_tc_bf16_test(64, 64, 64, "smoke_64_64_64");
}
