//! # kaio-runtime
//!
//! CUDA runtime layer for the KAIO GPU kernel authoring framework. This
//! crate wraps [`cudarc`] to provide device management, typed device
//! buffers, PTX module loading, and a builder-style kernel launch API. It
//! is Layer 2 of KAIO, sitting on top of [`kaio-core`] (PTX emission).
//!
//! ## Quick start
//!
//! ```ignore
//! use kaio_runtime::{KaioDevice, GpuBuffer};
//!
//! let device = KaioDevice::new(0)?;
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
//! cargo test -p kaio-runtime -- --ignored
//! ```
//!
//! Standard `cargo test --workspace` runs only host-side tests.
//!
//! [`cudarc`]: https://crates.io/crates/cudarc
//! [`kaio-core`]: https://crates.io/crates/kaio-core

#![warn(missing_docs)]

pub mod buffer;
pub mod device;
pub mod error;
pub mod module;

pub use buffer::GpuBuffer;
pub use cudarc::driver::LaunchConfig;
pub use cudarc::driver::PushKernelArg;
pub use device::{DeviceInfo, KaioDevice};
pub use error::{KaioError, Result};
pub use module::{KaioFunction, KaioModule};
