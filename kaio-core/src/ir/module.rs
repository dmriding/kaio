//! PTX module — the top-level IR container.

use std::fmt;

use super::instruction::PtxInstruction;
use super::kernel::PtxKernel;
use crate::instr::MemoryOp;

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

    /// Parse the target string (e.g. `"sm_89"`) into a numeric SM
    /// version (e.g. `89`).
    ///
    /// Returns `None` if the target string is not a recognized
    /// `sm_NN` form (e.g. future targets, virtual architectures).
    /// [`validate`](Self::validate) tolerates unparseable targets by
    /// skipping the SM check — we'd rather let unusual targets through
    /// than block a user experimenting with a custom target string.
    fn parse_sm_target(&self) -> Option<u32> {
        self.target.strip_prefix("sm_").and_then(|s| s.parse().ok())
    }

    /// Validate that this module's target SM is high enough for every
    /// feature used by its kernels.
    ///
    /// Walks all kernel bodies looking for features that carry a minimum
    /// SM requirement — currently tensor-core operations and `cp.async`
    /// variants (both Ampere+ / SM 8.0). Returns [`ValidationError::SmTooLow`]
    /// on the **first** such mismatch with a human-readable description.
    ///
    /// This is a narrow **target-capability** check, not a semantic or
    /// dataflow pass. The goal is to surface clean errors at emit-time
    /// instead of cryptic ptxas messages downstream.
    pub fn validate(&self) -> Result<(), ValidationError> {
        let Some(target_sm) = self.parse_sm_target() else {
            // Unrecognized target (e.g. custom or virtual arch) — skip.
            return Ok(());
        };

        for kernel in &self.kernels {
            for instr in &kernel.body {
                if let Some((required, feature)) = instruction_sm_requirement(instr)
                    && target_sm < required
                {
                    return Err(ValidationError::SmTooLow {
                        required,
                        actual: target_sm,
                        feature,
                    });
                }
            }
        }
        Ok(())
    }
}

/// Return `Some((min_sm, feature_label))` if this instruction carries an SM
/// requirement, or `None` if it is SM-agnostic.
fn instruction_sm_requirement(instr: &PtxInstruction) -> Option<(u32, String)> {
    match instr {
        PtxInstruction::TensorCore(op) => Some((op.min_sm(), op.feature_label())),
        PtxInstruction::Memory(
            MemoryOp::CpAsyncCaSharedGlobal { .. }
            | MemoryOp::CpAsyncCommitGroup
            | MemoryOp::CpAsyncWaitGroup { .. },
        ) => Some((80, "cp.async".to_string())),
        _ => None,
    }
}

/// Errors returned by [`PtxModule::validate`].
///
/// Scope is intentionally narrow — target-capability checks only, no
/// semantic analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    /// A feature used by the module requires a higher SM target than
    /// the module declares.
    ///
    /// Example: a kernel containing `mma.sync.m16n8k16` in a module
    /// with `.target sm_70` would yield
    /// `required: 80, actual: 70, feature: "mma.sync.m16n8k16"`.
    SmTooLow {
        /// Minimum SM version required by the offending feature.
        required: u32,
        /// SM version parsed from the module's target string.
        actual: u32,
        /// Human-readable name of the offending feature.
        feature: String,
    },
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SmTooLow {
                required,
                actual,
                feature,
            } => {
                write!(
                    f,
                    "{feature} requires sm_{required}+, target is sm_{actual}"
                )
            }
        }
    }
}

impl std::error::Error for ValidationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fragment::{alloc_a_f16, alloc_b_f16, alloc_c};
    use crate::instr::{MemoryOp, MmaShape, TensorCoreOp};
    use crate::ir::{PtxInstruction, PtxKernel, Register, RegisterAllocator};
    use crate::types::{PtxType, RegKind};

    fn reg(kind: RegKind, index: u32, ptx_type: PtxType) -> Register {
        Register {
            kind,
            index,
            ptx_type,
        }
    }

    fn tc_kernel() -> PtxKernel {
        let mut alloc = RegisterAllocator::new();
        let mut k = PtxKernel::new("has_mma");
        k.push(PtxInstruction::TensorCore(TensorCoreOp::MmaSync {
            d: alloc_c(&mut alloc),
            a: alloc_a_f16(&mut alloc),
            b: alloc_b_f16(&mut alloc),
            c: alloc_c(&mut alloc),
            shape: MmaShape::M16N8K16,
            d_ty: PtxType::F32,
            a_ty: PtxType::F16,
            b_ty: PtxType::F16,
            c_ty: PtxType::F32,
        }));
        k
    }

    #[test]
    fn validate_rejects_mma_on_sm_70() {
        let mut module = PtxModule::new("sm_70");
        module.add_kernel(tc_kernel());
        let err = module.validate().unwrap_err();
        assert_eq!(
            err,
            ValidationError::SmTooLow {
                required: 80,
                actual: 70,
                feature: "mma.sync.m16n8k16".to_string(),
            }
        );
        assert_eq!(
            err.to_string(),
            "mma.sync.m16n8k16 requires sm_80+, target is sm_70"
        );
    }

    #[test]
    fn validate_accepts_mma_on_sm_80() {
        let mut module = PtxModule::new("sm_80");
        module.add_kernel(tc_kernel());
        assert!(module.validate().is_ok());
    }

    #[test]
    fn validate_accepts_mma_on_sm_89() {
        let mut module = PtxModule::new("sm_89");
        module.add_kernel(tc_kernel());
        assert!(module.validate().is_ok());
    }

    fn tc_int8_kernel() -> PtxKernel {
        use crate::fragment::{alloc_a_M16N8K32, alloc_b_M16N8K32, alloc_c_M16N8K32};
        let mut alloc = RegisterAllocator::new();
        let mut k = PtxKernel::new("has_mma_int8");
        k.push(PtxInstruction::TensorCore(TensorCoreOp::MmaSyncInt8 {
            d: alloc_c_M16N8K32(&mut alloc),
            a: alloc_a_M16N8K32(&mut alloc),
            b: alloc_b_M16N8K32(&mut alloc),
            c: alloc_c_M16N8K32(&mut alloc),
        }));
        k
    }

    #[test]
    fn validate_rejects_mma_int8_on_sm_70() {
        let mut module = PtxModule::new("sm_70");
        module.add_kernel(tc_int8_kernel());
        let err = module.validate().unwrap_err();
        assert_eq!(
            err,
            ValidationError::SmTooLow {
                required: 80,
                actual: 70,
                feature: "mma.sync.m16n8k32.s8.s8.s32".to_string(),
            }
        );
        assert_eq!(
            err.to_string(),
            "mma.sync.m16n8k32.s8.s8.s32 requires sm_80+, target is sm_70"
        );
    }

    #[test]
    fn validate_accepts_mma_int8_on_sm_80() {
        let mut module = PtxModule::new("sm_80");
        module.add_kernel(tc_int8_kernel());
        assert!(module.validate().is_ok());
    }

    #[test]
    fn validate_accepts_mma_int8_on_sm_89() {
        let mut module = PtxModule::new("sm_89");
        module.add_kernel(tc_int8_kernel());
        assert!(module.validate().is_ok());
    }

    #[test]
    fn validate_rejects_cp_async_on_sm_75() {
        let mut module = PtxModule::new("sm_75");
        let mut k = PtxKernel::new("has_cp_async");
        k.push(PtxInstruction::Memory(MemoryOp::new_cp_async_ca(
            reg(RegKind::R, 0, PtxType::U32),
            reg(RegKind::Rd, 0, PtxType::U64),
            16,
        )));
        module.add_kernel(k);
        let err = module.validate().unwrap_err();
        assert_eq!(
            err,
            ValidationError::SmTooLow {
                required: 80,
                actual: 75,
                feature: "cp.async".to_string(),
            }
        );
    }

    #[test]
    fn validate_accepts_scalar_kernel_on_sm_70() {
        // A module with no tensor-core or cp.async features should pass
        // validation even on sm_70.
        let mut module = PtxModule::new("sm_70");
        let k = PtxKernel::new("scalar_only");
        module.add_kernel(k);
        assert!(module.validate().is_ok());
    }

    #[test]
    fn validate_skips_unparseable_target() {
        // Don't block weird custom targets.
        let mut module = PtxModule::new("compute_90a");
        module.add_kernel(tc_kernel());
        assert!(module.validate().is_ok());
    }

    #[test]
    fn parse_sm_target() {
        let m = PtxModule::new("sm_89");
        assert_eq!(m.parse_sm_target(), Some(89));
        let m2 = PtxModule::new("sm_80");
        assert_eq!(m2.parse_sm_target(), Some(80));
        let m3 = PtxModule::new("compute_90a");
        assert_eq!(m3.parse_sm_target(), None);
    }
}
