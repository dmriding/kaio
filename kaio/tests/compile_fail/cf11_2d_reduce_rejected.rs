// CF11: block_reduce_sum rejected in 2D kernels.
// Reductions use TidX for warp lane identity, which is wrong for 2D blocks
// where linear tid = tidx + bdimx * tidy. Tracked in tech_debt.md for Phase 5.
use kaio::gpu_kernel;

#[gpu_kernel(block_size = (16, 16))]
fn kernel(data: &[f32], out: &mut [f32], n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        let val = data[idx];
        let sum = block_reduce_sum(val);
        out[idx] = sum;
    }
}

fn main() {}
