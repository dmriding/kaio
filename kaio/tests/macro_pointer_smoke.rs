//! E2E smoke test: mixed reference and pointer kernel parameter forms through
//! the `#[gpu_kernel]` macro (RFC-0001).
//!
//! Parser unit tests prove *acceptance* of the new syntax; this test proves
//! the whole macro → codegen → launch path stays consistent across all four
//! parameter forms (`&[T]`, `*const [T]`, `&mut [T]`, `*mut [T]`) with zero
//! divergence.

use kaio::prelude::*;

#[gpu_kernel(block_size = 256)]
fn mixed_forms(a: &[f32], b: *const [f32], c: &mut [f32], d: *mut [f32], n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        d[idx] = a[idx] + b[idx];
        c[idx] = a[idx] * 2.0;
    }
}

#[test]
#[ignore] // requires NVIDIA GPU
fn mixed_forms_correctness() {
    let device = KaioDevice::new(0).expect("GPU required");

    let n: u32 = 1024;
    let a_host: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..n).map(|i| (i * 3) as f32).collect();
    let expected_c: Vec<f32> = a_host.iter().map(|x| x * 2.0).collect();
    let expected_d: Vec<f32> = a_host.iter().zip(&b_host).map(|(x, y)| x + y).collect();

    let a = device.alloc_from(&a_host).expect("alloc a");
    let b = device.alloc_from(&b_host).expect("alloc b");
    let mut c = device.alloc_zeros::<f32>(n as usize).expect("alloc c");
    let mut d = device.alloc_zeros::<f32>(n as usize).expect("alloc d");

    mixed_forms::launch(&device, &a, &b, &mut c, &mut d, n).expect("launch failed");

    let c_result = c.to_host(&device).expect("to_host c");
    let d_result = d.to_host(&device).expect("to_host d");
    assert_eq!(
        c_result, expected_c,
        "c (ref-mut form) produced wrong results"
    );
    assert_eq!(
        d_result, expected_d,
        "d (ptr-mut form) produced wrong results"
    );
}
