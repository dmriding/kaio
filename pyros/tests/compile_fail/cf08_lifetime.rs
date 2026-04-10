// CF9: Kernel parameter uses a lifetime.
use pyros::gpu_kernel;

#[gpu_kernel(block_size = 256)]
fn kernel(data: &'static [f32], n: u32) {}

fn main() {}
