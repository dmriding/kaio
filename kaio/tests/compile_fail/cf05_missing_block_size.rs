// CF6: Missing block_size attribute.
use kaio::gpu_kernel;

#[gpu_kernel]
fn kernel(n: u32) {}

fn main() {}
