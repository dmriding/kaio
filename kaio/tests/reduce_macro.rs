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
