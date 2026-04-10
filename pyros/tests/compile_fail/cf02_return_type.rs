// CF2: Kernel returns a value (must return ()).
use pyros::gpu_kernel;

#[gpu_kernel(block_size = 256)]
fn kernel(n: u32) -> u32 {
    n
}

fn main() {}
