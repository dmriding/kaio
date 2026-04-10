// CF7: block_size not a power of 2.
use kaio::gpu_kernel;

#[gpu_kernel(block_size = 100)]
fn kernel(n: u32) {}

fn main() {}
