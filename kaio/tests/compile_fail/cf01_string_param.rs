// CF1: Kernel parameter is an unsupported type (String).
use kaio::gpu_kernel;

#[gpu_kernel(block_size = 256)]
fn kernel(name: String) {}

fn main() {}
