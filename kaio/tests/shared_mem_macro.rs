//! E2E test: shared memory + barrier through the #[gpu_kernel] macro.
//!
//! Tests shared_mem![], bar_sync(), and shared memory indexing on real hardware.
//! Sprint 4.2 adds multi-allocation tests (two+ shared arrays in one kernel).

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

// --- Sprint 4.2: Multi-allocation shared memory tests ---

/// Two shared arrays in one kernel. Writes different patterns to each,
/// barrier, reads back. Verifies no cross-contamination between allocations.
#[gpu_kernel(block_size = 256)]
fn two_shared_arrays(out_a: &mut [f32], out_b: &mut [f32], n: u32) {
    let tid = thread_idx_x();
    let tile_a = shared_mem![f32; 256];
    let tile_b = shared_mem![f32; 256];
    if tid < n {
        // Write distinct patterns
        tile_a[tid] = tid as f32;
        tile_b[tid] = tid as f32 + 1000.0;
        bar_sync();
        // Read back from each — must not alias
        out_a[tid] = tile_a[tid];
        out_b[tid] = tile_b[tid];
    }
}

/// Shared array coexists with block_reduce_sum. Both use shared memory
/// internally — must not overlap.
#[gpu_kernel(block_size = 256)]
fn shared_plus_reduce(data: &mut [f32], n: u32) {
    let tid = thread_idx_x();
    let scratch = shared_mem![f32; 256];
    if tid < n {
        scratch[tid] = 1.0f32;
    }
    bar_sync();
    let sum = block_reduce_sum(scratch[tid]);
    if tid < n {
        data[tid] = sum;
    }
}

/// Three shared arrays — verifies cumulative addressing for 3+ allocations.
#[gpu_kernel(block_size = 256)]
fn three_shared_arrays(out: &mut [f32], n: u32) {
    let tid = thread_idx_x();
    let a = shared_mem![f32; 256];
    let b = shared_mem![f32; 256];
    let c = shared_mem![f32; 256];
    if tid < n {
        a[tid] = 1.0f32;
        b[tid] = 2.0f32;
        c[tid] = 3.0f32;
        bar_sync();
        // Sum all three — should be 6.0
        out[tid] = a[tid] + b[tid] + c[tid];
    }
}

#[test]
#[ignore] // requires NVIDIA GPU
fn two_shared_arrays_no_aliasing() {
    let device = KaioDevice::new(0).expect("GPU required");
    let n = 256u32;
    let mut out_a = device.alloc_zeros::<f32>(n as usize).unwrap();
    let mut out_b = device.alloc_zeros::<f32>(n as usize).unwrap();

    two_shared_arrays::launch(&device, &mut out_a, &mut out_b, n).unwrap();

    let result_a = out_a.to_host(&device).unwrap();
    let result_b = out_b.to_host(&device).unwrap();
    for i in 0..n as usize {
        assert_eq!(
            result_a[i], i as f32,
            "tile_a[{i}]: expected {}, got {}",
            i as f32, result_a[i]
        );
        assert_eq!(
            result_b[i],
            i as f32 + 1000.0,
            "tile_b[{i}]: expected {}, got {}",
            i as f32 + 1000.0,
            result_b[i]
        );
    }
}

#[test]
#[ignore] // requires NVIDIA GPU
fn shared_array_plus_reduction() {
    let device = KaioDevice::new(0).expect("GPU required");
    let n = 256u32;
    let mut data = device.alloc_zeros::<f32>(n as usize).unwrap();

    shared_plus_reduce::launch(&device, &mut data, n).unwrap();

    let result = data.to_host(&device).unwrap();
    // All threads wrote 1.0 to shared, reduce_sum should produce 256.0
    let expected = 256.0f32;
    for (i, &val) in result.iter().enumerate() {
        assert!(
            (val - expected).abs() < 1e-3,
            "shared+reduce at [{i}]: expected {expected}, got {val}"
        );
    }
}

#[test]
#[ignore] // requires NVIDIA GPU
fn three_shared_arrays_correctness() {
    let device = KaioDevice::new(0).expect("GPU required");
    let n = 256u32;
    let mut out = device.alloc_zeros::<f32>(n as usize).unwrap();

    three_shared_arrays::launch(&device, &mut out, n).unwrap();

    let result = out.to_host(&device).unwrap();
    for (i, &val) in result.iter().enumerate() {
        assert!(
            (val - 6.0f32).abs() < 1e-6,
            "three arrays at [{i}]: expected 6.0, got {val}"
        );
    }
}
