//! PTX kernel parameter types.

use crate::types::PtxType;

/// A parameter to a PTX kernel function.
///
/// In PTX, kernel parameters are declared in the `.entry` signature:
/// - Scalar: `.param .u32 n` — a value passed by copy
/// - Pointer: `.param .u64 a_ptr` — a device memory address (always 64-bit)
///
/// For pointer params, `elem_type` records the type of data pointed to
/// (used by the code generator for load/store instruction types).
#[derive(Debug, Clone)]
pub enum PtxParam {
    /// A scalar value (e.g. `.param .u32 n`).
    Scalar {
        /// Parameter name.
        name: String,
        /// PTX type of the scalar value.
        ptx_type: PtxType,
    },
    /// A pointer to device memory (declared as `.param .u64 name`).
    Pointer {
        /// Parameter name.
        name: String,
        /// Type of the data pointed to (for codegen, not the param declaration).
        elem_type: PtxType,
    },
}

impl PtxParam {
    /// Create a scalar parameter.
    pub fn scalar(name: &str, ptx_type: PtxType) -> Self {
        Self::Scalar {
            name: name.to_string(),
            ptx_type,
        }
    }

    /// Create a pointer parameter.
    pub fn pointer(name: &str, elem_type: PtxType) -> Self {
        Self::Pointer {
            name: name.to_string(),
            elem_type,
        }
    }

    /// The parameter name.
    pub fn name(&self) -> &str {
        match self {
            Self::Scalar { name, .. } | Self::Pointer { name, .. } => name,
        }
    }

    /// PTX parameter declaration string (without trailing comma).
    ///
    /// - Scalar: `.param .u32 n`
    /// - Pointer: `.param .u64 a_ptr` (always 64-bit address)
    pub fn ptx_decl(&self) -> String {
        match self {
            Self::Scalar { name, ptx_type } => {
                format!(".param {} {}", ptx_type.ptx_suffix(), name)
            }
            Self::Pointer { name, .. } => {
                format!(".param .u64 {}", name)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_param() {
        let p = PtxParam::scalar("n", PtxType::U32);
        assert_eq!(p.name(), "n");
        assert!(matches!(
            p,
            PtxParam::Scalar {
                ptx_type: PtxType::U32,
                ..
            }
        ));
    }

    #[test]
    fn pointer_param() {
        let p = PtxParam::pointer("a_ptr", PtxType::F32);
        assert_eq!(p.name(), "a_ptr");
        assert!(matches!(
            p,
            PtxParam::Pointer {
                elem_type: PtxType::F32,
                ..
            }
        ));
    }

    #[test]
    fn scalar_ptx_decl() {
        let p = PtxParam::scalar("n", PtxType::U32);
        assert_eq!(p.ptx_decl(), ".param .u32 n");
    }

    #[test]
    fn pointer_ptx_decl() {
        let p = PtxParam::pointer("a_ptr", PtxType::F32);
        // Pointers always declared as .u64 regardless of elem_type
        assert_eq!(p.ptx_decl(), ".param .u64 a_ptr");
    }
}
