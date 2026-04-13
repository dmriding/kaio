//! Sprint 7.0 D3: compound bitwise assignment end-to-end.
//!
//! Exercises the `arr[i] OP= value` path for every bitwise compound
//! operator (`&=`, `|=`, `^=`, `<<=`, `>>=`). Desugars through the same
//! `IndexAssign { value: BinOp(Index(...), base_op, rhs) }` shape as the
//! existing arithmetic compound assignments (Phase 3).
//!
//! Also covers scalar compound bitwise assign (`let mut x = ...; x |= n;`)
//! via the Assign path in the same desugaring function.

use kaio::prelude::*;

#[gpu_kernel(block_size = 32)]
fn compound_or_indexed(out: &mut [u32], mask: u32, n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        out[idx] |= mask;
    }
}

#[gpu_kernel(block_size = 32)]
fn compound_and_indexed(out: &mut [u32], mask: u32, n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        out[idx] &= mask;
    }
}

#[gpu_kernel(block_size = 32)]
fn compound_xor_indexed(out: &mut [u32], mask: u32, n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        out[idx] ^= mask;
    }
}

#[gpu_kernel(block_size = 32)]
fn compound_shl_indexed(out: &mut [u32], shift: u32, n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        out[idx] <<= shift;
    }
}

#[gpu_kernel(block_size = 32)]
fn compound_shr_indexed(out: &mut [u32], shift: u32, n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        out[idx] >>= shift;
    }
}

#[gpu_kernel(block_size = 32)]
fn compound_scalar_or(src: &[u32], out: &mut [u32], n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        let mut acc = src[idx];
        acc |= 0xFF000000;
        acc &= 0xFF00FFFF;
        out[idx] = acc;
    }
}

#[test]
#[ignore] // requires NVIDIA GPU
fn compound_or_indexed_writes_mask() {
    let device = KaioDevice::new(0).expect("GPU required");
    let n: u32 = 4;
    let initial: [u32; 4] = [0x0F0F0F0F, 0x12345678, 0x00000000, 0xAAAAAAAA];
    let mask: u32 = 0xFF000000;
    let expected: Vec<u32> = initial.iter().map(|x| x | mask).collect();

    let mut out = device.alloc_from(&initial).expect("alloc out");
    compound_or_indexed::launch(&device, &mut out, mask, n).expect("launch");
    let result = out.to_host(&device).expect("to_host");
    assert_eq!(result, expected);
}

#[test]
#[ignore] // requires NVIDIA GPU
fn compound_and_indexed_masks() {
    let device = KaioDevice::new(0).expect("GPU required");
    let n: u32 = 4;
    let initial: [u32; 4] = [0xFFFFFFFF, 0x12345678, 0xAAAAAAAA, 0x55555555];
    let mask: u32 = 0x0000FFFF;
    let expected: Vec<u32> = initial.iter().map(|x| x & mask).collect();

    let mut out = device.alloc_from(&initial).expect("alloc out");
    compound_and_indexed::launch(&device, &mut out, mask, n).expect("launch");
    let result = out.to_host(&device).expect("to_host");
    assert_eq!(result, expected);
}

#[test]
#[ignore] // requires NVIDIA GPU
fn compound_xor_indexed_flips() {
    let device = KaioDevice::new(0).expect("GPU required");
    let n: u32 = 4;
    let initial: [u32; 4] = [0xFFFFFFFF, 0xAAAAAAAA, 0x12345678, 0x00000000];
    let mask: u32 = 0x0F0F0F0F;
    let expected: Vec<u32> = initial.iter().map(|x| x ^ mask).collect();

    let mut out = device.alloc_from(&initial).expect("alloc out");
    compound_xor_indexed::launch(&device, &mut out, mask, n).expect("launch");
    let result = out.to_host(&device).expect("to_host");
    assert_eq!(result, expected);
}

#[test]
#[ignore] // requires NVIDIA GPU
fn compound_shl_indexed_shifts() {
    let device = KaioDevice::new(0).expect("GPU required");
    let n: u32 = 4;
    let initial: [u32; 4] = [1, 0x00FF00FF, 0x000F, 0xAAAAAAAA];
    let shift: u32 = 4;
    let expected: Vec<u32> = initial.iter().map(|x| x << shift).collect();

    let mut out = device.alloc_from(&initial).expect("alloc out");
    compound_shl_indexed::launch(&device, &mut out, shift, n).expect("launch");
    let result = out.to_host(&device).expect("to_host");
    assert_eq!(result, expected);
}

#[test]
#[ignore] // requires NVIDIA GPU
fn compound_shr_indexed_shifts() {
    let device = KaioDevice::new(0).expect("GPU required");
    let n: u32 = 4;
    let initial: [u32; 4] = [0xFFFFFFFF, 0x80000000, 0x000000FF, 16];
    let shift: u32 = 1;
    let expected: Vec<u32> = initial.iter().map(|x| x >> shift).collect();

    let mut out = device.alloc_from(&initial).expect("alloc out");
    compound_shr_indexed::launch(&device, &mut out, shift, n).expect("launch");
    let result = out.to_host(&device).expect("to_host");
    assert_eq!(result, expected);
}

#[test]
#[ignore] // requires NVIDIA GPU
fn compound_scalar_or_and_chain() {
    let device = KaioDevice::new(0).expect("GPU required");
    let n: u32 = 4;
    let src_host: [u32; 4] = [0x00000000, 0x12345678, 0xFFFFFFFF, 0xAAAAAAAA];
    let expected: Vec<u32> = src_host
        .iter()
        .map(|x| {
            let mut acc = *x;
            acc |= 0xFF000000;
            acc &= 0xFF00FFFF;
            acc
        })
        .collect();

    let src = device.alloc_from(&src_host).expect("alloc src");
    let mut out = device.alloc_zeros::<u32>(n as usize).expect("alloc out");
    compound_scalar_or::launch(&device, &src, &mut out, n).expect("launch");
    let result = out.to_host(&device).expect("to_host");
    assert_eq!(result, expected);
}
