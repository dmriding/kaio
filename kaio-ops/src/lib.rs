// Generated launch functions for matmul have many parameters (device + kernel args + grid).
#![allow(clippy::too_many_arguments)]
#![warn(missing_docs)]

//! Pre-built GPU operations for KAIO.
//!
//! Provides high-level functions for common GPU compute operations.
//! Users don't need to write kernels — just call the operation.
//!
//! # Example
//!
//! ```ignore
//! use kaio::prelude::*;
//! use kaio_ops::matmul;
//!
//! let device = KaioDevice::new(0)?;
//! let a = device.alloc_from(&a_data)?;
//! let b = device.alloc_from(&b_data)?;
//! let mut c = device.alloc_zeros::<f32>(m * n)?;
//! matmul(&device, &a, &b, &mut c, m, n, k)?;
//! ```

mod attention_kernel;
mod matmul_kernel;
mod tuner;

pub use attention_kernel::{attention, attention_causal, attention_flash, attention_flash_causal};
pub use matmul_kernel::matmul;
pub use tuner::{
    attention_auto, attention_auto_causal, matmul_auto, tune_attention, tune_attention_causal,
    tune_matmul,
};

// Expose naive kernel for benchmarking (not public API)
#[doc(hidden)]
pub use matmul_kernel::matmul_naive;
