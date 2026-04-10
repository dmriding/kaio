// CF1: Kernel parameter is an unsupported type (String).
use pyros::gpu_kernel;

#[gpu_kernel(block_size = 256)]
fn kernel(name: String) {}

fn main() {}
