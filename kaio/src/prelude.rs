//! The KAIO prelude — everything needed to write and launch GPU kernels.
//!
//! ```ignore
//! use kaio::prelude::*;
//!
//! #[gpu_kernel(block_size = 256)]
//! fn vector_add(a: *const [f32], b: *const [f32], out: *mut [f32], n: u32) {
//!     let idx = thread_idx_x() + block_idx_x() * block_dim_x();
//!     if idx < n {
//!         out[idx] = a[idx] + b[idx];
//!     }
//! }
//! ```

pub use crate::gpu_builtins::*;
pub use crate::gpu_kernel;
pub use crate::runtime::{GpuBuffer, KaioDevice, KaioError, LaunchConfig, Result};
