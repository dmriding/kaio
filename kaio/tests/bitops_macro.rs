//! Sprint 7.0 D2: GPU round-trip tests for the bitwise operator DSL.
//!
//! Each kernel here exercises a specific lowering path:
//! - `bitops_all`  — `&`, `|`, `^` end-to-end in a single kernel
//! - `shift_left`  — `<<` on u32 and i32 (typeless, both valid)
//! - `shr_logical_u32`   — `u32 >> n` must zero-extend (AD2 canary)
//! - `shr_arithmetic_i32` — `i32 >> n` must sign-extend (AD2 canary)
//! - `not_bitwise_u32`   — unary `!` on integer dispatches to bitwise NOT
//!
//! The two Shr kernels paired together are the AD2 end-to-end guarantee:
//! if signedness ever collapses anywhere in the parse → kernel_ir → lower
//! → emit chain, one of these tests fails loudly. Phase 7.1+ INT8/INT4
//! dequantization relies on this distinction.

use kaio::prelude::*;

#[gpu_kernel(block_size = 32)]
fn bitops_all(a: &[u32], b: &[u32], out: &mut [u32], n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        let x = a[idx];
        let y = b[idx];
        // Pack three bitop results into one output using shifts.
        // High byte = x & y, mid byte = (x | y) & 0xFF, low byte = (x ^ y) & 0xFF.
        let and_v = x & y;
        let or_v = x | y;
        let xor_v = x ^ y;
        out[idx] = ((and_v & 0xFF) << 16) | ((or_v & 0xFF) << 8) | (xor_v & 0xFF);
    }
}

#[gpu_kernel(block_size = 32)]
fn shift_left(src: &[u32], out: &mut [u32], n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        out[idx] = src[idx] << 4;
    }
}

#[gpu_kernel(block_size = 32)]
fn shr_logical_u32(src: &[u32], out: &mut [u32], n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        // u32 >> 1 — must be logical shift (zero-extend high bit).
        // 0xFFFFFFFF >> 1 == 0x7FFFFFFF if correct, 0xFFFFFFFF if broken.
        out[idx] = src[idx] >> 1;
    }
}

#[gpu_kernel(block_size = 32)]
fn shr_arithmetic_i32(src: &[i32], out: &mut [i32], n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        // i32 >> 1 — must be arithmetic shift (sign-extend high bit).
        // -2 >> 1 == -1 if correct, 0x7FFFFFFF if broken (sign bit lost).
        out[idx] = src[idx] >> 1;
    }
}

#[gpu_kernel(block_size = 32)]
fn not_bitwise_u32(src: &[u32], out: &mut [u32], n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        // !0x00000000u32 == 0xFFFFFFFFu32 — integer dispatch to ArithOp::Not{b32}.
        out[idx] = !src[idx];
    }
}

#[test]
#[ignore] // requires NVIDIA GPU
fn bitops_all_smoke() {
    let device = KaioDevice::new(0).expect("GPU required");

    let a_host: [u32; 4] = [0xFF00FF00, 0x12345678, 0xAAAAAAAA, 0xFFFFFFFF];
    let b_host: [u32; 4] = [0x00FF00FF, 0x87654321, 0x55555555, 0x00000000];
    let n: u32 = 4;

    let a = device.alloc_from(&a_host).expect("alloc a");
    let b = device.alloc_from(&b_host).expect("alloc b");
    let mut out = device.alloc_zeros::<u32>(n as usize).expect("alloc out");

    bitops_all::launch(&device, &a, &b, &mut out, n).expect("launch");

    let result = out.to_host(&device).expect("to_host");
    let expected: Vec<u32> = a_host
        .iter()
        .zip(&b_host)
        .map(|(x, y)| {
            let and_v = x & y;
            let or_v = x | y;
            let xor_v = x ^ y;
            ((and_v & 0xFF) << 16) | ((or_v & 0xFF) << 8) | (xor_v & 0xFF)
        })
        .collect();
    assert_eq!(result, expected, "bitops_all produced wrong results");
}

#[test]
#[ignore] // requires NVIDIA GPU
fn shift_left_u32_smoke() {
    let device = KaioDevice::new(0).expect("GPU required");

    let src_host: [u32; 4] = [1, 0x0000FFFF, 0x10000000, 0xF];
    let n: u32 = 4;

    let src = device.alloc_from(&src_host).expect("alloc src");
    let mut out = device.alloc_zeros::<u32>(n as usize).expect("alloc out");

    shift_left::launch(&device, &src, &mut out, n).expect("launch");

    let result = out.to_host(&device).expect("to_host");
    let expected: Vec<u32> = src_host.iter().map(|x| x << 4).collect();
    assert_eq!(result, expected, "shift_left produced wrong results");
}

#[test]
#[ignore] // requires NVIDIA GPU
fn shr_logical_u32_zero_extends() {
    // AD2 end-to-end canary (logical half): u32 >> 1 must zero-extend.
    // If broken (signedness collapse → shr.s32), this test catches it.
    let device = KaioDevice::new(0).expect("GPU required");

    let src_host: [u32; 4] = [0xFFFFFFFF, 0x80000000, 0x80000001, 2];
    let expected: [u32; 4] = [0x7FFFFFFF, 0x40000000, 0x40000000, 1];
    let n: u32 = 4;

    let src = device.alloc_from(&src_host).expect("alloc src");
    let mut out = device.alloc_zeros::<u32>(n as usize).expect("alloc out");

    shr_logical_u32::launch(&device, &src, &mut out, n).expect("launch");

    let result = out.to_host(&device).expect("to_host");
    assert_eq!(
        result,
        expected.to_vec(),
        "u32 >> 1 must zero-extend (logical shift). \
         If 0xFFFFFFFF>>1 == 0xFFFFFFFF, signedness collapsed somewhere."
    );
}

#[test]
#[ignore] // requires NVIDIA GPU
fn shr_arithmetic_i32_sign_extends() {
    // AD2 end-to-end canary (arithmetic half): i32 >> 1 must sign-extend.
    // -2 >> 1 == -1 if correct. If broken (emits shr.u32 or shr.b32),
    // -2 as u32 is 0xFFFFFFFE, logically-shifted gives 0x7FFFFFFF which
    // is a large positive number, not -1.
    let device = KaioDevice::new(0).expect("GPU required");

    let src_host: [i32; 4] = [-2, -1, i32::MIN, 4];
    let expected: [i32; 4] = [-1, -1, i32::MIN / 2, 2];
    let n: u32 = 4;

    let src = device.alloc_from(&src_host).expect("alloc src");
    let mut out = device.alloc_zeros::<i32>(n as usize).expect("alloc out");

    shr_arithmetic_i32::launch(&device, &src, &mut out, n).expect("launch");

    let result = out.to_host(&device).expect("to_host");
    assert_eq!(
        result,
        expected.to_vec(),
        "i32 >> 1 must sign-extend (arithmetic shift). \
         If -2>>1 != -1, signedness collapsed and INT8 dequant will \
         silently produce wrong weights on negative packed values."
    );
}

#[test]
#[ignore] // requires NVIDIA GPU
fn not_bitwise_u32_flips_bits() {
    let device = KaioDevice::new(0).expect("GPU required");

    let src_host: [u32; 4] = [0x00000000, 0xFFFFFFFF, 0xAAAAAAAA, 0x12345678];
    let expected: [u32; 4] = [0xFFFFFFFF, 0x00000000, 0x55555555, !0x12345678u32];
    let n: u32 = 4;

    let src = device.alloc_from(&src_host).expect("alloc src");
    let mut out = device.alloc_zeros::<u32>(n as usize).expect("alloc out");

    not_bitwise_u32::launch(&device, &src, &mut out, n).expect("launch");

    let result = out.to_host(&device).expect("to_host");
    assert_eq!(
        result,
        expected.to_vec(),
        "!u32 must flip all bits via ArithOp::Not {{ ty: U32 }} → not.b32"
    );
}
