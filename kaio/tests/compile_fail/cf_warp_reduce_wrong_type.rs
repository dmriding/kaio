// Sprint 7.1.5 (round-3 boundary canary): warp_reduce_sum is
// f32-only in v0.3.0. Passing an i32 must be rejected by the type
// checker against the host-stub signature `fn warp_reduce_sum(_val: f32) -> f32`.
// This lock holds the f32-only contract before any future type-generalization
// sprint loosens it.
use kaio::gpu_kernel;
use kaio::prelude::*;

#[gpu_kernel(block_size = 32)]
fn kernel(out: &mut [f32], n: u32) {
    let tid = thread_idx_x();
    let bad: i32 = 7i32;
    let s = warp_reduce_sum(bad);
    if tid < n {
        out[tid] = s as f32;
    }
}

fn main() {}
