//! E2E tests: block_reduce_sum and block_reduce_max through #[gpu_kernel].
//!
//! Tests the multi-instruction reduction expansion on real hardware.
//! All tests use f32 constants to avoid the known cvt rounding bug.

use kaio::prelude::*;

// --- block_reduce_sum: each thread contributes 1.0, result = block_size ---

#[gpu_kernel(block_size = 256)]
fn reduce_sum_ones(out: &mut [f32], n: u32) {
    let tid = thread_idx_x();
    let total = block_reduce_sum(1.0f32);
    if tid < n {
        out[tid] = total;
    }
}

#[test]
#[ignore] // requires NVIDIA GPU
fn block_reduce_sum_all_ones() {
    let device = KaioDevice::new(0).expect("GPU required");
    let n = 256u32;
    let mut out = device.alloc_zeros::<f32>(n as usize).expect("alloc out");

    reduce_sum_ones::launch(&device, &mut out, n).expect("launch failed");

    let result = out.to_host(&device).expect("to_host");
    // Every thread should see the sum = 256.0 (broadcast)
    for (i, &val) in result.iter().enumerate() {
        assert!(
            (val - 256.0f32).abs() < 1e-3,
            "block_reduce_sum failed at thread {i}: expected 256.0, got {val}"
        );
    }
}

// --- block_reduce_max: thread 0 contributes 999.0, others 1.0 ---

#[gpu_kernel(block_size = 256)]
fn reduce_max_one_high(out: &mut [f32], n: u32) {
    let tid = thread_idx_x();
    let mut val = 1.0f32;
    if tid == 0u32 {
        val = 999.0f32;
    }
    let maximum = block_reduce_max(val);
    if tid < n {
        out[tid] = maximum;
    }
}

#[test]
#[ignore] // requires NVIDIA GPU
fn block_reduce_max_finds_highest() {
    let device = KaioDevice::new(0).expect("GPU required");
    let n = 256u32;
    let mut out = device.alloc_zeros::<f32>(n as usize).expect("alloc out");

    reduce_max_one_high::launch(&device, &mut out, n).expect("launch failed");

    let result = out.to_host(&device).expect("to_host");
    // Every thread should see max = 999.0 (broadcast)
    for (i, &val) in result.iter().enumerate() {
        assert!(
            (val - 999.0f32).abs() < 1e-3,
            "block_reduce_max failed at thread {i}: expected 999.0, got {val}"
        );
    }
}

// --- 2D block reduction tests ---
// These verify that linear_tid = tidx + tidy * block_dim_x is computed
// correctly in lower_block_reduce() for 2D kernels.

#[gpu_kernel(block_size = (16, 16))]
fn reduce_sum_2d(input: &[f32], out: &mut [f32], n: u32) {
    let tid = thread_idx_x() + thread_idx_y() * block_dim_x();
    let idx = tid + block_idx_x() * 256u32;
    let mut val = 0.0f32;
    if idx < n {
        val = input[idx];
    }
    let total = block_reduce_sum(val);
    if tid == 0u32 {
        out[block_idx_x()] = total;
    }
}

#[test]
#[ignore] // requires NVIDIA GPU
fn block_reduce_sum_2d_16x16() {
    let device = KaioDevice::new(0).expect("GPU required");
    let n = 256u32;
    let input: Vec<f32> = (1..=n).map(|i| i as f32).collect();
    let expected: f32 = input.iter().sum();

    let d_input = device.alloc_from(&input).expect("alloc input");
    let mut d_out = device.alloc_zeros::<f32>(1).expect("alloc out");

    let grid = (1u32, 1u32, 1u32);
    reduce_sum_2d::launch(&device, &d_input, &mut d_out, n, grid).expect("launch failed");

    let result = d_out.to_host(&device).expect("to_host");
    assert!(
        (result[0] - expected).abs() < 1e-1,
        "2D sum: expected {expected}, got {}",
        result[0]
    );
}

#[gpu_kernel(block_size = (16, 16))]
fn reduce_max_2d(input: &[f32], out: &mut [f32], n: u32) {
    let tid = thread_idx_x() + thread_idx_y() * block_dim_x();
    let idx = tid + block_idx_x() * 256u32;
    let mut val = 0.0f32;
    if idx < n {
        val = input[idx];
    }
    let maximum = block_reduce_max(val);
    if tid == 0u32 {
        out[block_idx_x()] = maximum;
    }
}

#[test]
#[ignore] // requires NVIDIA GPU
fn block_reduce_max_2d_16x16() {
    let device = KaioDevice::new(0).expect("GPU required");
    let n = 256u32;
    let input: Vec<f32> = (1..=n).map(|i| i as f32).collect();

    let d_input = device.alloc_from(&input).expect("alloc input");
    let mut d_out = device.alloc_zeros::<f32>(1).expect("alloc out");

    let grid = (1u32, 1u32, 1u32);
    reduce_max_2d::launch(&device, &d_input, &mut d_out, n, grid).expect("launch failed");

    let result = d_out.to_host(&device).expect("to_host");
    assert!(
        (result[0] - 256.0f32).abs() < 1e-3,
        "2D max: expected 256.0, got {}",
        result[0]
    );
}

// --- Asymmetric 2D block (32x8 = 256 threads) ---

#[gpu_kernel(block_size = (32, 8))]
fn reduce_sum_2d_asym(input: &[f32], out: &mut [f32], n: u32) {
    let tid = thread_idx_x() + thread_idx_y() * block_dim_x();
    let idx = tid + block_idx_x() * 256u32;
    let mut val = 0.0f32;
    if idx < n {
        val = input[idx];
    }
    let total = block_reduce_sum(val);
    if tid == 0u32 {
        out[block_idx_x()] = total;
    }
}

#[test]
#[ignore] // requires NVIDIA GPU
fn block_reduce_sum_2d_asymmetric_32x8() {
    let device = KaioDevice::new(0).expect("GPU required");
    let n = 256u32;
    let input: Vec<f32> = (1..=n).map(|i| i as f32).collect();
    let expected: f32 = input.iter().sum();

    let d_input = device.alloc_from(&input).expect("alloc input");
    let mut d_out = device.alloc_zeros::<f32>(1).expect("alloc out");

    let grid = (1u32, 1u32, 1u32);
    reduce_sum_2d_asym::launch(&device, &d_input, &mut d_out, n, grid).expect("launch failed");

    let result = d_out.to_host(&device).expect("to_host");
    assert!(
        (result[0] - expected).abs() < 1e-1,
        "2D asymmetric sum: expected {expected}, got {}",
        result[0]
    );
}

// --- Identity-based test: value = tidx * 100 + tidy ---
// If reduction incorrectly uses raw TidX, row aliasing produces wrong result.

#[gpu_kernel(block_size = (16, 16))]
fn reduce_sum_2d_identity(out: &mut [f32]) {
    let tid = thread_idx_x() + thread_idx_y() * block_dim_x();
    let tidx = thread_idx_x();
    let tidy = thread_idx_y();
    // Each thread contributes a unique value derived from both indices
    let val = tidx as f32 * 100.0 + tidy as f32;
    let total = block_reduce_sum(val);
    if tid == 0u32 {
        out[0u32] = total;
    }
}

#[test]
#[ignore] // requires NVIDIA GPU
fn block_reduce_sum_2d_identity_based() {
    let device = KaioDevice::new(0).expect("GPU required");
    let mut out = device.alloc_zeros::<f32>(1).expect("alloc out");

    let grid = (1u32, 1u32, 1u32);
    reduce_sum_2d_identity::launch(&device, &mut out, grid).expect("launch failed");

    // CPU reference: sum of (tidx * 100 + tidy) for tidx in 0..16, tidy in 0..16
    let mut expected = 0.0f32;
    for tidy in 0u32..16 {
        for tidx in 0u32..16 {
            expected += tidx as f32 * 100.0 + tidy as f32;
        }
    }
    // expected = 100 * sum(0..16) * 16 + sum(0..16) * 16
    // = 100 * 120 * 16 + 120 * 16 = 192000 + 1920 = 193920

    let result = out.to_host(&device).expect("to_host");
    assert!(
        (result[0] - expected).abs() < 1.0,
        "2D identity sum: expected {expected}, got {}",
        result[0]
    );
}

// =========================================================================
// Sprint 7.1.5 — block_reduce_min tests
// =========================================================================

// block_reduce_min: thread 0 contributes -999.0, others 1.0. Result must be
// -999.0 broadcast to every thread.

#[gpu_kernel(block_size = 256)]
fn reduce_min_one_low(out: &mut [f32], n: u32) {
    let tid = thread_idx_x();
    let mut val = 1.0f32;
    if tid == 0u32 {
        val = -999.0f32;
    }
    let minimum = block_reduce_min(val);
    if tid < n {
        out[tid] = minimum;
    }
}

#[test]
#[ignore] // requires NVIDIA GPU
fn block_reduce_min_finds_lowest_256() {
    let device = KaioDevice::new(0).expect("GPU required");
    let n = 256u32;
    let mut out = device.alloc_zeros::<f32>(n as usize).expect("alloc out");
    reduce_min_one_low::launch(&device, &mut out, n).expect("launch failed");
    let result = out.to_host(&device).expect("to_host");
    for (i, &val) in result.iter().enumerate() {
        assert!(
            (val - (-999.0f32)).abs() < 1e-3,
            "block_reduce_min at thread {i}: expected -999.0, got {val}"
        );
    }
}

// block_reduce_min across the full set of supported block sizes. Each
// thread contributes `(tid as f32 - 50.0)` so the min is `-50.0` (thread 0).

macro_rules! block_reduce_min_size_test {
    ($fn_name:ident, $kernel_name:ident, $block:expr) => {
        #[gpu_kernel(block_size = $block)]
        fn $kernel_name(out: &mut [f32], n: u32) {
            let tid = thread_idx_x();
            let val = (tid as f32) - 50.0f32;
            let minimum = block_reduce_min(val);
            if tid < n {
                out[tid] = minimum;
            }
        }
        #[test]
        #[ignore]
        fn $fn_name() {
            let device = KaioDevice::new(0).expect("GPU required");
            let n = $block as u32;
            let mut out = device.alloc_zeros::<f32>(n as usize).expect("alloc");
            $kernel_name::launch(&device, &mut out, n).expect("launch");
            let result = out.to_host(&device).expect("to_host");
            for &v in result.iter() {
                assert!(
                    (v - (-50.0f32)).abs() < 1e-3,
                    "block_reduce_min at block_size {}: expected -50.0, got {}",
                    $block,
                    v
                );
            }
        }
    };
}

block_reduce_min_size_test!(block_reduce_min_bs32, reduce_min_bs32_k, 32);
block_reduce_min_size_test!(block_reduce_min_bs64, reduce_min_bs64_k, 64);
block_reduce_min_size_test!(block_reduce_min_bs128, reduce_min_bs128_k, 128);
block_reduce_min_size_test!(block_reduce_min_bs512, reduce_min_bs512_k, 512);

// =========================================================================
// Sprint 7.1.5 — warp_reduce_{sum,max,min} tests
// =========================================================================

// warp_reduce_sum: single warp (32 threads), each lane contributes 1.0.
// Every lane must see the full warp sum = 32.0 after the call.

#[gpu_kernel(block_size = 32)]
fn warp_sum_ones_kernel(out: &mut [f32], n: u32) {
    let tid = thread_idx_x();
    let s = warp_reduce_sum(1.0f32);
    if tid < n {
        out[tid] = s;
    }
}

#[test]
#[ignore]
fn warp_reduce_sum_all_ones_broadcasts_to_all_lanes() {
    let device = KaioDevice::new(0).expect("GPU required");
    let n = 32u32;
    let mut out = device.alloc_zeros::<f32>(n as usize).expect("alloc");
    warp_sum_ones_kernel::launch(&device, &mut out, n).expect("launch");
    let result = out.to_host(&device).expect("to_host");
    for (i, &v) in result.iter().enumerate() {
        assert!(
            (v - 32.0f32).abs() < 1e-3,
            "warp_reduce_sum at lane {i}: expected 32.0, got {v}"
        );
    }
}

// warp_reduce_sum: ascending lane values 0..32 → sum = 0+1+...+31 = 496.

#[gpu_kernel(block_size = 32)]
fn warp_sum_ascending_kernel(out: &mut [f32], n: u32) {
    let tid = thread_idx_x();
    let s = warp_reduce_sum(tid as f32);
    if tid < n {
        out[tid] = s;
    }
}

#[test]
#[ignore]
fn warp_reduce_sum_ascending_lanes() {
    let device = KaioDevice::new(0).expect("GPU required");
    let n = 32u32;
    let mut out = device.alloc_zeros::<f32>(n as usize).expect("alloc");
    warp_sum_ascending_kernel::launch(&device, &mut out, n).expect("launch");
    let result = out.to_host(&device).expect("to_host");
    for (i, &v) in result.iter().enumerate() {
        assert!(
            (v - 496.0f32).abs() < 1e-3,
            "warp_reduce_sum ascending at lane {i}: expected 496.0, got {v}"
        );
    }
}

// warp_reduce_max: single-hot pattern. Lane 17 contributes 1000.0, others
// contribute -1.0 * lane_idx. Every lane must see 1000.0 after the call.

#[gpu_kernel(block_size = 32)]
fn warp_max_single_hot_kernel(out: &mut [f32], n: u32) {
    let tid = thread_idx_x();
    let mut v = -(tid as f32);
    if tid == 17u32 {
        v = 1000.0f32;
    }
    let m = warp_reduce_max(v);
    if tid < n {
        out[tid] = m;
    }
}

#[test]
#[ignore]
fn warp_reduce_max_single_hot_at_lane_17() {
    let device = KaioDevice::new(0).expect("GPU required");
    let n = 32u32;
    let mut out = device.alloc_zeros::<f32>(n as usize).expect("alloc");
    warp_max_single_hot_kernel::launch(&device, &mut out, n).expect("launch");
    let result = out.to_host(&device).expect("to_host");
    for (i, &v) in result.iter().enumerate() {
        assert!(
            (v - 1000.0f32).abs() < 1e-3,
            "warp_reduce_max single-hot at lane {i}: expected 1000.0, got {v}"
        );
    }
}

// warp_reduce_min: single-hot-low pattern. Lane 7 contributes -500.0,
// others contribute (lane_idx as f32). Every lane must see -500.0.

#[gpu_kernel(block_size = 32)]
fn warp_min_single_hot_kernel(out: &mut [f32], n: u32) {
    let tid = thread_idx_x();
    let mut v = tid as f32;
    if tid == 7u32 {
        v = -500.0f32;
    }
    let m = warp_reduce_min(v);
    if tid < n {
        out[tid] = m;
    }
}

#[test]
#[ignore]
fn warp_reduce_min_single_hot_at_lane_7() {
    let device = KaioDevice::new(0).expect("GPU required");
    let n = 32u32;
    let mut out = device.alloc_zeros::<f32>(n as usize).expect("alloc");
    warp_min_single_hot_kernel::launch(&device, &mut out, n).expect("launch");
    let result = out.to_host(&device).expect("to_host");
    for (i, &v) in result.iter().enumerate() {
        assert!(
            (v - (-500.0f32)).abs() < 1e-3,
            "warp_reduce_min single-hot at lane {i}: expected -500.0, got {v}"
        );
    }
}

// warp_reduce_sum: alternating-sign pattern. Lane even contributes +1.0,
// lane odd contributes -1.0. Sum = 0.0 across all 32 lanes.

#[gpu_kernel(block_size = 32)]
fn warp_sum_alt_sign_kernel(out: &mut [f32], n: u32) {
    let tid = thread_idx_x();
    let mut v = 1.0f32;
    if (tid & 1u32) == 1u32 {
        v = -1.0f32;
    }
    let s = warp_reduce_sum(v);
    if tid < n {
        out[tid] = s;
    }
}

#[test]
#[ignore]
fn warp_reduce_sum_alternating_sign() {
    let device = KaioDevice::new(0).expect("GPU required");
    let n = 32u32;
    let mut out = device.alloc_zeros::<f32>(n as usize).expect("alloc");
    warp_sum_alt_sign_kernel::launch(&device, &mut out, n).expect("launch");
    let result = out.to_host(&device).expect("to_host");
    for (i, &v) in result.iter().enumerate() {
        assert!(
            v.abs() < 1e-3,
            "warp_reduce_sum alt-sign at lane {i}: expected 0.0, got {v}"
        );
    }
}

// =========================================================================
// 64-thread TWO-WARP INDEPENDENCE canary (round 3 high-value test)
// =========================================================================
//
// Block size = 64 = 2 warps. Warp 0 (lanes 0..32) sees pattern A, warp 1
// (lanes 32..64) sees pattern B with different deterministic values.
// Each warp must compute its OWN independent sum, not a cross-warp blend.
// Catches any lowering bug that accidentally reduces across warps.
//
// Warp 0: lane value = (lane as f32)       → sum = 0+1+...+31 = 496.0
// Warp 1: lane value = (lane as f32 * 2.0) → sum = 2*(32+...+63) = 3040.0

#[gpu_kernel(block_size = 64)]
fn warp_two_warp_independence_kernel(out: &mut [f32], n: u32) {
    let tid = thread_idx_x();
    let mut v = tid as f32;
    if tid >= 32u32 {
        v = (tid as f32) * 2.0f32;
    }
    let s = warp_reduce_sum(v);
    if tid < n {
        out[tid] = s;
    }
}

#[test]
#[ignore]
fn warp_reduce_sum_two_warps_independent() {
    let device = KaioDevice::new(0).expect("GPU required");
    let n = 64u32;
    let mut out = device.alloc_zeros::<f32>(n as usize).expect("alloc");
    warp_two_warp_independence_kernel::launch(&device, &mut out, n).expect("launch");
    let result = out.to_host(&device).expect("to_host");

    // Warp 0 (lanes 0..32) should see 496.0.
    for (i, &v) in result.iter().take(32).enumerate() {
        assert!(
            (v - 496.0f32).abs() < 1e-3,
            "warp 0 lane {i}: expected 496.0 (per-warp sum of 0..32), got {v} \
             — this indicates the reduction BLENDED ACROSS WARPS, which is \
             the specific semantic bug this test guards against"
        );
    }
    // Warp 1 (lanes 32..64) should see 2 * (32+33+...+63) = 3040.0.
    for (i, &v) in result.iter().skip(32).take(32).enumerate() {
        let lane = i + 32;
        assert!(
            (v - 3040.0f32).abs() < 1e-3,
            "warp 1 lane {lane}: expected 3040.0 (per-warp sum of 64..126 step 2), got {v} \
             — this indicates the reduction BLENDED ACROSS WARPS"
        );
    }
}

// =========================================================================
// 2D block with product = 32 (one full warp via 2D layout)
// =========================================================================

#[gpu_kernel(block_size = (8, 4))]
fn warp_sum_2d_8x4_kernel(out: &mut [f32], n: u32) {
    let lane = thread_idx_x() + thread_idx_y() * 8u32;
    let s = warp_reduce_sum(lane as f32);
    if lane < n {
        out[lane] = s;
    }
}

#[test]
#[ignore]
fn warp_reduce_sum_2d_block_8x4() {
    let device = KaioDevice::new(0).expect("GPU required");
    let n = 32u32;
    let mut out = device.alloc_zeros::<f32>(n as usize).expect("alloc");
    let grid = (1u32, 1u32, 1u32);
    warp_sum_2d_8x4_kernel::launch(&device, &mut out, n, grid).expect("launch");
    let result = out.to_host(&device).expect("to_host");
    // Linear tid maps 0..32; sum is 0+1+...+31 = 496.
    for (i, &v) in result.iter().enumerate() {
        assert!(
            (v - 496.0f32).abs() < 1e-3,
            "warp_reduce_sum 2D (8,4) at lane {i}: expected 496.0, got {v}"
        );
    }
}
