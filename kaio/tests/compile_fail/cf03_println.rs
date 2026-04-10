// CF3: Kernel uses a macro invocation (println!).
use kaio::gpu_kernel;

#[gpu_kernel(block_size = 256)]
fn kernel(n: u32) {
    println!("hello");
}

fn main() {}
