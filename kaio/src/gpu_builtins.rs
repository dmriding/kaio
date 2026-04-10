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
