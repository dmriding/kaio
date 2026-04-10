//! E2E test: shared memory + barrier through the #[gpu_kernel] macro.
//!
//! Tests shared_mem![], bar_sync(), and shared memory indexing on real hardware.
//! Deferred from Sprint 3.3 (needed bar_sync builtin from Sprint 3.4).

use kaio::prelude::*;

/// Baseline test — write constant to global memory without shared memory.
/// If this fails, the issue is with the launch wrapper, not shared memory.
#[gpu_kernel(block_size = 256)]
fn global_constant(out: &mut [f32], n: u32) {
    let tid = thread_idx_x();
    if tid < n {
        out[tid] = 42.0f32;
    }
}

/// Write a constant to shared memory, barrier, read back — verify round-trip.
/// Uses 1.0f32 constant to avoid the known cvt rounding modifier bug
/// (cvt.f32.u32 lacks .rn — Sprint 3.8 fix).
#[gpu_kernel(block_size = 256)]
fn shared_roundtrip(out: &mut [f32], n: u32) {
    let tid = thread_idx_x();
    let sdata = shared_mem![f32; 256];
    if tid < n {
        sdata[tid] = 1.0f32;
        bar_sync();
        out[tid] = sdata[tid];
    }
}

#[test]
#[ignore] // requires NVIDIA GPU
fn baseline_global_write() {
    let device = KaioDevice::new(0).expect("GPU required");
    let n = 256u32;
    let mut out = device.alloc_zeros::<f32>(n as usize).expect("alloc out");
    global_constant::launch(&device, &mut out, n).expect("launch failed");
    let result = out.to_host(&device).expect("to_host");
    assert!(
        (result[0] - 42.0f32).abs() < 1e-6,
        "baseline global write failed: got {}",
        result[0]
    );
}

#[test]
#[ignore] // requires NVIDIA GPU
fn shared_write_barrier_read() {
    let device = KaioDevice::new(0).expect("GPU required");

    // Use n = block_size = 256 → exactly 1 block
    let n = 256u32;
    let mut out = device.alloc_zeros::<f32>(n as usize).expect("alloc out");

    // Sync device before launch to clear any prior errors
    shared_roundtrip::launch(&device, &mut out, n).expect("launch failed");

    let result = out.to_host(&device).expect("to_host");
    for (i, &val) in result.iter().enumerate() {
        assert!(
            (val - 1.0f32).abs() < 1e-6,
            "shared memory round-trip failed at index {i}: expected 1.0, got {val}"
        );
    }
}
