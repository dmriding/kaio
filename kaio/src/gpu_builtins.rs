//! Stub functions for GPU built-in operations.
//!
//! These functions exist for IDE support (rust-analyzer autocomplete + docs)
//! and type checking. The `#[gpu_kernel]` macro recognizes calls to these
//! names and replaces them with PTX instructions — the stubs are never
//! actually executed in generated code.
//!
//! Calling these outside a `#[gpu_kernel]` function will panic.

/// Returns the thread index in the X dimension.
pub fn thread_idx_x() -> u32 {
    panic!("thread_idx_x() can only be called inside a #[gpu_kernel] function")
}

/// Returns the thread index in the Y dimension.
pub fn thread_idx_y() -> u32 {
    panic!("thread_idx_y() can only be called inside a #[gpu_kernel] function")
}

/// Returns the thread index in the Z dimension.
pub fn thread_idx_z() -> u32 {
    panic!("thread_idx_z() can only be called inside a #[gpu_kernel] function")
}

/// Returns the block index in the X dimension.
pub fn block_idx_x() -> u32 {
    panic!("block_idx_x() can only be called inside a #[gpu_kernel] function")
}

/// Returns the block index in the Y dimension.
pub fn block_idx_y() -> u32 {
    panic!("block_idx_y() can only be called inside a #[gpu_kernel] function")
}

/// Returns the block index in the Z dimension.
pub fn block_idx_z() -> u32 {
    panic!("block_idx_z() can only be called inside a #[gpu_kernel] function")
}

/// Returns the block dimension (threads per block) in the X dimension.
pub fn block_dim_x() -> u32 {
    panic!("block_dim_x() can only be called inside a #[gpu_kernel] function")
}

/// Returns the block dimension (threads per block) in the Y dimension.
pub fn block_dim_y() -> u32 {
    panic!("block_dim_y() can only be called inside a #[gpu_kernel] function")
}

/// Returns the block dimension (threads per block) in the Z dimension.
pub fn block_dim_z() -> u32 {
    panic!("block_dim_z() can only be called inside a #[gpu_kernel] function")
}

/// Returns the grid dimension (blocks per grid) in the X dimension.
pub fn grid_dim_x() -> u32 {
    panic!("grid_dim_x() can only be called inside a #[gpu_kernel] function")
}

/// Returns the grid dimension (blocks per grid) in the Y dimension.
pub fn grid_dim_y() -> u32 {
    panic!("grid_dim_y() can only be called inside a #[gpu_kernel] function")
}

/// Returns the grid dimension (blocks per grid) in the Z dimension.
pub fn grid_dim_z() -> u32 {
    panic!("grid_dim_z() can only be called inside a #[gpu_kernel] function")
}

/// Approximate square root (f32). GPU fast-math.
pub fn sqrt(_x: f32) -> f32 {
    panic!("sqrt() can only be called inside a #[gpu_kernel] function")
}

/// Absolute value. Works on any numeric type.
pub fn abs<T>(_x: T) -> T {
    panic!("abs() can only be called inside a #[gpu_kernel] function")
}

/// Minimum of two values.
pub fn min<T>(_x: T, _y: T) -> T {
    panic!("min() can only be called inside a #[gpu_kernel] function")
}

/// Maximum of two values.
pub fn max<T>(_x: T, _y: T) -> T {
    panic!("max() can only be called inside a #[gpu_kernel] function")
}

/// Natural exponential (e^x). GPU fast-math (f32).
pub fn exp(_x: f32) -> f32 {
    panic!("exp() can only be called inside a #[gpu_kernel] function")
}

/// Natural logarithm (ln(x)). GPU fast-math (f32).
pub fn log(_x: f32) -> f32 {
    panic!("log() can only be called inside a #[gpu_kernel] function")
}

/// Hyperbolic tangent. GPU fast-math (f32).
pub fn tanh(_x: f32) -> f32 {
    panic!("tanh() can only be called inside a #[gpu_kernel] function")
}

/// Fused multiply-add: `a * b + c` in one operation.
///
/// Uses hardware FMA (`fma.rn.f32`) for better precision and performance
/// than separate multiply + add. Essential for matmul inner loops.
pub fn fma(_a: f32, _b: f32, _c: f32) -> f32 {
    panic!("fma() can only be called inside a #[gpu_kernel] function")
}

/// Synchronize all threads in the block at a barrier.
///
/// All threads must reach this point before any can proceed.
/// Equivalent to `__syncthreads()` in CUDA C++.
pub fn bar_sync() {
    panic!("bar_sync() can only be called inside a #[gpu_kernel] function")
}

/// Warp shuffle down — each thread reads from the lane `delta` positions below.
///
/// `width` must be 32 (full warp). Returns the value from the source lane.
pub fn shfl_sync_down(_val: f32, _delta: u32, _width: u32) -> f32 {
    panic!("shfl_sync_down() can only be called inside a #[gpu_kernel] function")
}

/// Warp shuffle up — each thread reads from the lane `delta` positions above.
///
/// `width` must be 32 (full warp). Returns the value from the source lane.
pub fn shfl_sync_up(_val: f32, _delta: u32, _width: u32) -> f32 {
    panic!("shfl_sync_up() can only be called inside a #[gpu_kernel] function")
}

/// Warp shuffle butterfly (XOR) — each thread reads from the lane at `lane XOR lane_mask`.
///
/// `width` must be 32 (full warp). Returns the value from the source lane.
pub fn shfl_sync_bfly(_val: f32, _lane_mask: u32, _width: u32) -> f32 {
    panic!("shfl_sync_bfly() can only be called inside a #[gpu_kernel] function")
}

/// Compute the sum of `val` across all threads in the block.
///
/// Result is broadcast to ALL threads — every thread gets the total sum.
/// Uses warp shuffle + shared memory internally.
pub fn block_reduce_sum(_val: f32) -> f32 {
    panic!("block_reduce_sum() can only be called inside a #[gpu_kernel] function")
}

/// Compute the max of `val` across all threads in the block.
///
/// Result is broadcast to ALL threads — every thread gets the maximum.
/// Uses warp shuffle + shared memory internally.
pub fn block_reduce_max(_val: f32) -> f32 {
    panic!("block_reduce_max() can only be called inside a #[gpu_kernel] function")
}

/// Declare a shared memory buffer inside a `#[gpu_kernel]` function.
///
/// ```ignore
/// let sdata = shared_mem![f32; 256];  // 256 f32 elements in shared memory
/// sdata[tid] = value;                 // write via st.shared
/// let val = sdata[tid];               // read via ld.shared
/// ```
///
/// Shared memory is block-scoped SRAM — all threads in a block share the same
/// allocation. Use `bar_sync()` to synchronize between writes and reads.
#[macro_export]
macro_rules! shared_mem {
    ($ty:ty; $n:expr) => {
        compile_error!("shared_mem![] can only be used inside a #[gpu_kernel] function")
    };
}
