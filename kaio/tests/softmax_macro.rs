//! E2E test: softmax kernel through the #[gpu_kernel] macro.
//!
//! This is the Phase 3 capstone — a real GPU algorithm using loops,
//! reductions (block_reduce_max, block_reduce_sum), exp(), and compound
//! index expressions (input[row_offset + i]).
//!
//! Row-wise softmax, single block per row.

use kaio::prelude::*;

// --- The softmax kernel ---

#[gpu_kernel(block_size = 256)]
fn softmax_row(input: &[f32], output: &mut [f32], row_len: u32) {
    let tid = thread_idx_x();
    let bsize = block_dim_x();
    let row_offset = block_idx_x() * row_len;

    // Step 1: Find row max (strided loop + reduce)
    let mut local_max = -3.402823e+38f32;
    let mut i1 = tid;
    while i1 < row_len {
        let val = input[row_offset + i1];
        if val > local_max {
            local_max = val;
        }
        i1 += bsize;
    }
    let row_max = block_reduce_max(local_max);

    // Step 2: Compute exp(x - max) and sum (strided loop + reduce)
    let mut local_sum = 0.0f32;
    let mut i2 = tid;
    while i2 < row_len {
        local_sum += exp(input[row_offset + i2] - row_max);
        i2 += bsize;
    }
    let row_sum = block_reduce_sum(local_sum);

    // Step 3: Normalize (strided loop)
    let mut i3 = tid;
    while i3 < row_len {
        output[row_offset + i3] = exp(input[row_offset + i3] - row_max) / row_sum;
        i3 += bsize;
    }
}

// --- CPU reference ---

fn cpu_softmax(row: &[f32]) -> Vec<f32> {
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = row.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// --- GPU tests ---

#[test]
#[ignore] // requires NVIDIA GPU
fn softmax_small_row() {
    let device = KaioDevice::new(0).expect("GPU required");

    let row_len = 128u32;
    let input_host: Vec<f32> = (0..row_len).map(|i| i as f32).collect();
    let expected = cpu_softmax(&input_host);

    let input = device.alloc_from(&input_host).expect("alloc input");
    let mut output = device
        .alloc_zeros::<f32>(row_len as usize)
        .expect("alloc output");

    softmax_row::launch(&device, &input, &mut output, row_len).expect("launch failed");

    let result = output.to_host(&device).expect("to_host");
    let err = max_abs_error(&result, &expected);
    assert!(
        err < 1e-5,
        "softmax_small_row: max absolute error {err} exceeds tolerance 1e-5"
    );
}

#[test]
#[ignore] // requires NVIDIA GPU
fn softmax_uniform() {
    let device = KaioDevice::new(0).expect("GPU required");

    let row_len = 128u32;
    let input_host = vec![1.0f32; row_len as usize];
    let expected_val = 1.0 / row_len as f32;

    let input = device.alloc_from(&input_host).expect("alloc input");
    let mut output = device
        .alloc_zeros::<f32>(row_len as usize)
        .expect("alloc output");

    softmax_row::launch(&device, &input, &mut output, row_len).expect("launch failed");

    let result = output.to_host(&device).expect("to_host");
    for (i, &val) in result.iter().enumerate() {
        assert!(
            (val - expected_val).abs() < 1e-5,
            "softmax_uniform: element {i} = {val}, expected {expected_val}"
        );
    }
}

#[test]
#[ignore] // requires NVIDIA GPU
fn softmax_all_zeros() {
    let device = KaioDevice::new(0).expect("GPU required");

    let row_len = 128u32;
    let input_host = vec![0.0f32; row_len as usize];
    let expected_val = 1.0 / row_len as f32;

    let input = device.alloc_from(&input_host).expect("alloc input");
    let mut output = device
        .alloc_zeros::<f32>(row_len as usize)
        .expect("alloc output");

    softmax_row::launch(&device, &input, &mut output, row_len).expect("launch failed");

    let result = output.to_host(&device).expect("to_host");
    for (i, &val) in result.iter().enumerate() {
        assert!(
            (val - expected_val).abs() < 1e-5,
            "softmax_all_zeros: element {i} = {val}, expected {expected_val}"
        );
    }
}

#[test]
#[ignore] // requires NVIDIA GPU
fn softmax_large_values() {
    let device = KaioDevice::new(0).expect("GPU required");

    // Large values that would overflow exp() without max subtraction
    let row_len = 128u32;
    let input_host: Vec<f32> = (0..row_len).map(|i| 1000.0 + i as f32).collect();
    let expected = cpu_softmax(&input_host);

    let input = device.alloc_from(&input_host).expect("alloc input");
    let mut output = device
        .alloc_zeros::<f32>(row_len as usize)
        .expect("alloc output");

    softmax_row::launch(&device, &input, &mut output, row_len).expect("launch failed");

    let result = output.to_host(&device).expect("to_host");

    // No inf/nan
    for (i, &val) in result.iter().enumerate() {
        assert!(
            val.is_finite(),
            "softmax_large_values: element {i} is not finite: {val}"
        );
    }

    // Accuracy
    let err = max_abs_error(&result, &expected);
    assert!(
        err < 1e-4,
        "softmax_large_values: max absolute error {err} exceeds tolerance 1e-4"
    );
}

#[test]
#[ignore] // requires NVIDIA GPU
fn softmax_sums_to_one() {
    let device = KaioDevice::new(0).expect("GPU required");

    let row_len = 128u32;
    let input_host: Vec<f32> = (0..row_len).map(|i| (i as f32) * 0.1 - 6.4).collect();

    let input = device.alloc_from(&input_host).expect("alloc input");
    let mut output = device
        .alloc_zeros::<f32>(row_len as usize)
        .expect("alloc output");

    softmax_row::launch(&device, &input, &mut output, row_len).expect("launch failed");

    let result = output.to_host(&device).expect("to_host");
    let sum: f32 = result.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "softmax output should sum to 1.0, got {sum}"
    );
}

// --- Sprint 3.7: Accuracy suite at varying numerical complexity ---

#[test]
#[ignore] // requires NVIDIA GPU
fn softmax_accuracy_small_range() {
    let device = KaioDevice::new(0).expect("GPU required");

    // Small values: [0.0, 0.1, 0.2, ..., 12.7]
    let row_len = 128u32;
    let input_host: Vec<f32> = (0..row_len).map(|i| i as f32 * 0.1).collect();
    let expected = cpu_softmax(&input_host);

    let input = device.alloc_from(&input_host).expect("alloc input");
    let mut output = device
        .alloc_zeros::<f32>(row_len as usize)
        .expect("alloc output");

    softmax_row::launch(&device, &input, &mut output, row_len).expect("launch failed");

    let result = output.to_host(&device).expect("to_host");
    let err = max_abs_error(&result, &expected);
    assert!(
        err < 1e-5,
        "softmax_accuracy_small_range: max abs error {err} exceeds 1e-5"
    );
}

#[test]
#[ignore] // requires NVIDIA GPU
fn softmax_accuracy_medium_range() {
    let device = KaioDevice::new(0).expect("GPU required");

    // Medium range spanning [-50, 50], full block_size utilization
    let row_len = 256u32;
    let input_host: Vec<f32> = (0..row_len)
        .map(|i| (i as f32) * (100.0 / 256.0) - 50.0)
        .collect();
    let expected = cpu_softmax(&input_host);

    let input = device.alloc_from(&input_host).expect("alloc input");
    let mut output = device
        .alloc_zeros::<f32>(row_len as usize)
        .expect("alloc output");

    softmax_row::launch(&device, &input, &mut output, row_len).expect("launch failed");

    let result = output.to_host(&device).expect("to_host");
    let err = max_abs_error(&result, &expected);
    assert!(
        err < 1e-5,
        "softmax_accuracy_medium_range: max abs error {err} exceeds 1e-5"
    );
}

#[test]
#[ignore] // requires NVIDIA GPU
fn softmax_accuracy_large_range() {
    let device = KaioDevice::new(0).expect("GPU required");

    // Large values: [900.0, 900.1, ..., 912.7] — tests exp stability
    let row_len = 128u32;
    let input_host: Vec<f32> = (0..row_len).map(|i| 900.0 + i as f32 * 0.1).collect();
    let expected = cpu_softmax(&input_host);

    let input = device.alloc_from(&input_host).expect("alloc input");
    let mut output = device
        .alloc_zeros::<f32>(row_len as usize)
        .expect("alloc output");

    softmax_row::launch(&device, &input, &mut output, row_len).expect("launch failed");

    let result = output.to_host(&device).expect("to_host");
    let err = max_abs_error(&result, &expected);
    assert!(
        err < 1e-4,
        "softmax_accuracy_large_range: max abs error {err} exceeds 1e-4"
    );
}

#[test]
#[ignore] // requires NVIDIA GPU
fn softmax_negative_values() {
    let device = KaioDevice::new(0).expect("GPU required");

    // All negative: [-12.8, -12.7, ..., -0.1]
    let row_len = 128u32;
    let input_host: Vec<f32> = (0..row_len)
        .map(|i| -((row_len - i) as f32) * 0.1)
        .collect();
    let expected = cpu_softmax(&input_host);

    let input = device.alloc_from(&input_host).expect("alloc input");
    let mut output = device
        .alloc_zeros::<f32>(row_len as usize)
        .expect("alloc output");

    softmax_row::launch(&device, &input, &mut output, row_len).expect("launch failed");

    let result = output.to_host(&device).expect("to_host");
    let err = max_abs_error(&result, &expected);
    assert!(
        err < 1e-5,
        "softmax_negative_values: max abs error {err} exceeds 1e-5"
    );

    // Should still be valid probabilities
    for (i, &val) in result.iter().enumerate() {
        assert!(
            val > 0.0 && val < 1.0,
            "element {i} = {val}, not a valid probability"
        );
    }
}

#[test]
#[ignore] // requires NVIDIA GPU
fn softmax_mixed_sign() {
    let device = KaioDevice::new(0).expect("GPU required");

    // Alternating: [-5.0, 5.0, -5.0, 5.0, ...]
    let row_len = 128u32;
    let input_host: Vec<f32> = (0..row_len)
        .map(|i| if i % 2 == 0 { -5.0f32 } else { 5.0f32 })
        .collect();
    let expected = cpu_softmax(&input_host);

    let input = device.alloc_from(&input_host).expect("alloc input");
    let mut output = device
        .alloc_zeros::<f32>(row_len as usize)
        .expect("alloc output");

    softmax_row::launch(&device, &input, &mut output, row_len).expect("launch failed");

    let result = output.to_host(&device).expect("to_host");
    let err = max_abs_error(&result, &expected);
    assert!(
        err < 1e-5,
        "softmax_mixed_sign: max abs error {err} exceeds 1e-5"
    );
}
