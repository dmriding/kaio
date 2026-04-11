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
