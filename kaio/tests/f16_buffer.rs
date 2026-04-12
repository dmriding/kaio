//! GPU test: GpuBuffer<f16> roundtrip — host → device → host.
//!
//! Proves that cudarc's `DeviceRepr` for `half::f16` works with KAIO's
//! `GpuBuffer` and `KaioDevice`. Sprint 6.1 runtime validation.

use half::f16;
use kaio::prelude::*;

#[test]
#[ignore] // requires NVIDIA GPU
fn f16_buffer_roundtrip() {
    let device = KaioDevice::new(0).expect("GPU required");

    // Create f16 test data: 0.0, 0.5, 1.0, 1.5, 2.0
    let host_data: Vec<f16> = vec![
        f16::from_f32(0.0),
        f16::from_f32(0.5),
        f16::from_f32(1.0),
        f16::from_f32(1.5),
        f16::from_f32(2.0),
    ];

    // Transfer to device
    let gpu_buf = device.alloc_from(&host_data).expect("alloc f16 buffer");
    assert_eq!(gpu_buf.len(), 5);

    // Transfer back to host
    let result = gpu_buf.to_host(&device).expect("f16 DtoH transfer");

    // Verify bit-exact roundtrip
    assert_eq!(result.len(), host_data.len());
    for (i, (got, expected)) in result.iter().zip(host_data.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "f16 roundtrip mismatch at index {i}: got {got}, expected {expected}"
        );
    }
}

#[test]
#[ignore] // requires NVIDIA GPU
fn bf16_buffer_roundtrip() {
    let device = KaioDevice::new(0).expect("GPU required");

    // Create bf16 test data
    let host_data: Vec<half::bf16> = vec![
        half::bf16::from_f32(0.0),
        half::bf16::from_f32(1.0),
        half::bf16::from_f32(-1.0),
        half::bf16::from_f32(42.0),
    ];

    let gpu_buf = device.alloc_from(&host_data).expect("alloc bf16 buffer");
    let result = gpu_buf.to_host(&device).expect("bf16 DtoH transfer");

    assert_eq!(result.len(), host_data.len());
    for (i, (got, expected)) in result.iter().zip(host_data.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "bf16 roundtrip mismatch at index {i}: got {got}, expected {expected}"
        );
    }
}
