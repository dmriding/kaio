//! PTX module — the top-level IR container.

use super::kernel::PtxKernel;

/// A complete PTX module containing version/target metadata and kernels.
///
/// Corresponds to a single `.ptx` file with a header and one or more
/// `.entry` kernel definitions.
#[derive(Debug, Clone)]
pub struct PtxModule {
    /// PTX ISA version (e.g. `"7.8"`).
    pub version: String,
    /// Target SM architecture (e.g. `"sm_89"`).
    pub target: String,
    /// Address size in bits (32 or 64).
    pub address_size: u32,
    /// Kernel definitions in this module.
    pub kernels: Vec<PtxKernel>,
}

impl PtxModule {
    /// Create a new module targeting the given SM architecture.
    ///
    /// Defaults: PTX version `8.7` (CUDA 12.8), address size `64`.
    pub fn new(target: &str) -> Self {
        Self {
            version: "8.7".to_string(),
            target: target.to_string(),
            address_size: 64,
            kernels: Vec::new(),
        }
    }

    /// Add a kernel to this module.
    pub fn add_kernel(&mut self, kernel: PtxKernel) {
        self.kernels.push(kernel);
    }
}
