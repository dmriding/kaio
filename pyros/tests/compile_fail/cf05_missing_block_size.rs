// CF6: Missing block_size attribute.
use pyros::gpu_kernel;

#[gpu_kernel]
fn kernel(n: u32) {}

fn main() {}
