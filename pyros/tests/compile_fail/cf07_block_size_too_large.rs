// CF8: block_size exceeds 1024.
use pyros::gpu_kernel;

#[gpu_kernel(block_size = 2048)]
fn kernel(n: u32) {}

fn main() {}
