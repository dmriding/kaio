// Sprint 7.1.5: warp_reduce_* checks the PRODUCT of 2D block dimensions
// against the whole-warp-multiple requirement. (4, 4) = 16 total threads
// — less than a full warp — must be rejected.
use kaio::gpu_kernel;
use kaio::prelude::*;

#[gpu_kernel(block_size = (4, 4))]
fn kernel(out: &mut [f32], n: u32) {
    let tid = thread_idx_x() + thread_idx_y() * 4u32;
    let s = warp_reduce_sum(1.0f32);
    if tid < n {
        out[tid] = s;
    }
}

fn main() {}
