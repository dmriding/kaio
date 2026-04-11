// CF12: fma() requires all f32 arguments.
use kaio::gpu_kernel;

#[gpu_kernel(block_size = 256)]
fn kernel(out: &mut [f32], n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        let a = 1u32;
        let b = 2.0f32;
        let c = 3.0f32;
        out[idx] = fma(a, b, c);
    }
}

fn main() {}
