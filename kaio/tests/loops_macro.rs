//! E2E tests: loop constructs through the #[gpu_kernel] macro.

use kaio::prelude::*;

// --- For loop: sum integers 0..n ---

#[gpu_kernel(block_size = 256)]
fn sum_range(out: &mut [u32], n: u32) {
    let tid = thread_idx_x();
    if tid == 0u32 {
        let mut acc = 0u32;
        for i in 0u32..n {
            acc += i;
        }
        out[0u32] = acc;
    }
}

#[test]
#[ignore] // requires NVIDIA GPU
fn for_loop_sum() {
    let device = KaioDevice::new(0).expect("GPU required");

    let n: u32 = 100;
    let expected = n * (n - 1) / 2; // sum of 0..100

    let mut out = device.alloc_zeros::<u32>(1).expect("alloc out");
    sum_range::launch(&device, &mut out, n).expect("launch failed");

    let result = out.to_host(&device).expect("to_host");
    assert_eq!(
        result[0], expected,
        "for loop sum 0..{n} = {expected}, got {}",
        result[0]
    );
}

// --- While loop: halve until below threshold ---

#[gpu_kernel(block_size = 256)]
fn while_halve(out: &mut [f32], n: u32) {
    let tid = thread_idx_x();
    if tid == 0u32 {
        let mut val = 1024.0f32;
        while val > 1.0f32 {
            val = val * 0.5f32;
        }
        out[0u32] = val;
    }
}

#[test]
#[ignore] // requires NVIDIA GPU
fn while_loop_converge() {
    let device = KaioDevice::new(0).expect("GPU required");

    let mut out = device.alloc_zeros::<f32>(1).expect("alloc out");
    while_halve::launch(&device, &mut out, 1u32).expect("launch failed");

    let result = out.to_host(&device).expect("to_host");
    // 1024 * 0.5^10 = 1.0, then 1.0 > 1.0 is false → loop exits at 1.0
    assert!(
        (result[0] - 1.0f32).abs() < 1e-6,
        "while loop should converge to 1.0, got {}",
        result[0]
    );
}

// --- While loop with stride + compound assignment ---

#[gpu_kernel(block_size = 256)]
fn strided_sum(data: &[f32], out: &mut [f32], n: u32) {
    let tid = thread_idx_x();
    let stride = block_dim_x();
    let mut acc = 0.0f32;
    let mut i = tid;
    while i < n {
        acc += data[i];
        i += stride;
    }
    out[tid] = acc;
}

#[test]
#[ignore] // requires NVIDIA GPU
fn while_strided_accumulate() {
    let device = KaioDevice::new(0).expect("GPU required");

    // Use n = block_size so exactly 1 block launches (avoids multi-block
    // data race on out[] — all blocks use thread_idx_x as output index).
    // Each thread reads exactly data[tid] once (stride = block_size = n).
    let n: u32 = 256;
    let data_host: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
    let expected_sum: f32 = data_host.iter().sum(); // sum of 1..257

    let data = device.alloc_from(&data_host).expect("alloc data");
    let mut out = device.alloc_zeros::<f32>(256).expect("alloc out");

    strided_sum::launch(&device, &data, &mut out, n).expect("launch failed");

    let result = out.to_host(&device).expect("to_host");
    // Each thread reads data[tid] exactly once, so out[tid] = data[tid].
    // Sum of all out[] = sum of data[] = sum of 1..257.
    let gpu_sum: f32 = result.iter().sum();
    assert!(
        (gpu_sum - expected_sum).abs() < 1.0,
        "strided sum should be {expected_sum}, got {gpu_sum}"
    );
}
