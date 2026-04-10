//! # pyros-runtime
//!
//! CUDA runtime layer for the PYROS GPU kernel authoring framework. This
//! crate wraps [`cudarc`] to provide device management, typed device
//! buffers, PTX module loading, and a builder-style kernel launch API. It
//! is Layer 2 of PYROS, sitting on top of [`pyros-core`] (PTX emission).
//!
//! ## Quick start
//!
//! ```ignore
//! use pyros_runtime::{PyrosDevice, GpuBuffer};
//!
//! let device = PyrosDevice::new(0)?;
//! let buf = device.alloc_from(&[1.0f32, 2.0, 3.0])?;
//! let host = buf.to_host(&device)?;
//! assert_eq!(host, vec![1.0, 2.0, 3.0]);
//! ```
//!
//! ## GPU-gated tests
//!
//! Tests that require an NVIDIA GPU are `#[ignore]`-gated. Run them with:
//!
//! ```sh
//! cargo test -p pyros-runtime -- --ignored
//! ```
//!
//! Standard `cargo test --workspace` runs only host-side tests.
//!
//! [`cudarc`]: https://crates.io/crates/cudarc
//! [`pyros-core`]: https://crates.io/crates/pyros-core

#![warn(missing_docs)]

pub mod buffer;
pub mod device;
pub mod error;
pub mod module;

pub use buffer::GpuBuffer;
pub use cudarc::driver::LaunchConfig;
pub use cudarc::driver::PushKernelArg;
pub use device::{DeviceInfo, PyrosDevice};
pub use error::{PyrosError, Result};
pub use module::{PyrosFunction, PyrosModule};
