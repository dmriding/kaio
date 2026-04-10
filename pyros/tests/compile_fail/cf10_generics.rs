// Extra: Kernel uses generic type parameters.
use pyros::gpu_kernel;

#[gpu_kernel(block_size = 256)]
fn kernel<T>(data: &[f32], n: u32) {}

fn main() {}
