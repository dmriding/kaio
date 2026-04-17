// Sprint 7.1.5: 48 total threads = 1 full warp + 1 partial warp of 16
// live lanes. shfl.sync.bfly with mask 0xFFFFFFFF has undefined behavior
// on that partial warp even though a full warp also exists. The guard
// rejects any block_size whose total thread count is not a multiple of 32.
// (8 * 6 = 48 — 2D form used because 1D pow-of-2 validation already
// rejects 48 before this guard fires.)
use kaio::gpu_kernel;
use kaio::prelude::*;

#[gpu_kernel(block_size = (8, 6))]
fn kernel(out: &mut [f32], n: u32) {
    let tid = thread_idx_x() + thread_idx_y() * 8u32;
    let s = warp_reduce_sum(1.0f32);
    if tid < n {
        out[tid] = s;
    }
}

fn main() {}
