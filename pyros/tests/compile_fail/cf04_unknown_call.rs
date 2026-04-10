// CF4/5: Kernel calls a function not in the built-in registry.
use pyros::gpu_kernel;

#[gpu_kernel(block_size = 256)]
fn kernel(data: &[f32], n: u32) {
    let idx = thread_idx_x();
    unknown_function(data, idx);
}

fn main() {}
