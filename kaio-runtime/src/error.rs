//! KAIO error types.

/// Errors that can occur during KAIO runtime operations.
#[derive(Debug, thiserror::Error)]
pub enum KaioError {
    /// A CUDA driver API call failed.
    #[error("CUDA driver error: {0}")]
    Cuda(#[from] cudarc::driver::DriverError),

    /// Device memory allocation failed.
    #[error("out of device memory: requested {requested} bytes")]
    OutOfMemory {
        /// Number of bytes requested.
        requested: usize,
    },

    /// Invalid kernel launch configuration.
    #[error("invalid kernel configuration: {0}")]
    InvalidConfig(String),

    /// No GPU device found at the given ordinal.
    #[error("device not found: ordinal {0}")]
    DeviceNotFound(usize),

    /// Failed to load a PTX module into the driver.
    #[error("PTX module load failed: {0}")]
    PtxLoad(String),
}

/// Convenience alias for `std::result::Result<T, KaioError>`.
pub type Result<T> = std::result::Result<T, KaioError>;
