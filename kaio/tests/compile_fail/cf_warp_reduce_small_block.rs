// Sprint 7.1.5: warp_reduce_* requires block_size to be a whole-warp
// multiple (>= 32, divisible by 32). A 16-thread 1D block is a valid
// power-of-2 block size but triggers the partial-warp UB guard.
use kaio::gpu_kernel;
use kaio::prelude::*;

#[gpu_kernel(block_size = 16)]
fn kernel(out: &mut [f32], n: u32) {
    let tid = thread_idx_x();
    let s = warp_reduce_sum(1.0f32);
    if tid < n {
        out[tid] = s;
    }
}

fn main() {}
