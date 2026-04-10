// CF10: Kernel uses loop (not supported until Phase 3).
use kaio::gpu_kernel;

#[gpu_kernel(block_size = 256)]
fn kernel(n: u32) {
    loop {}
}

fn main() {}
