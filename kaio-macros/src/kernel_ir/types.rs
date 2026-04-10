//! Kernel-level type system for the `#[gpu_kernel]` macro.
//!
//! These types bridge `syn`'s AST to `kaio-core`'s PTX IR. They are
//! internal to the macro crate and never publicly exposed.

use proc_macro2::Span;

/// GPU-compatible types supported in kernel signatures and bodies.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)] // Variants used progressively across Sprints 2.1-2.8
pub enum KernelType {
    F32,
    F64,
    I32,
    U32,
    I64,
    U64,
    Bool,
    /// `&[T]` — read-only slice (becomes a `.param .u64` pointer in PTX).
    SliceRef(Box<KernelType>),
    /// `&mut [T]` — writable slice (becomes a `.param .u64` pointer in PTX).
    SliceMutRef(Box<KernelType>),
}

#[allow(dead_code)] // Methods used progressively across Sprints 2.2-2.8
impl KernelType {
    /// Returns the element type for slice types, or `None` for scalars.
    pub fn elem_type(&self) -> Option<&KernelType> {
        match self {
            KernelType::SliceRef(inner) | KernelType::SliceMutRef(inner) => Some(inner),
            _ => None,
        }
    }

    /// Returns `true` if this is a slice type (`&[T]` or `&mut [T]`).
    pub fn is_slice(&self) -> bool {
        matches!(self, KernelType::SliceRef(_) | KernelType::SliceMutRef(_))
    }

    /// Returns `true` if this is a scalar (non-slice) type.
    pub fn is_scalar(&self) -> bool {
        !self.is_slice()
    }

    /// Returns `true` if this is a mutable slice (`&mut [T]`).
    pub fn is_mut_slice(&self) -> bool {
        matches!(self, KernelType::SliceMutRef(_))
    }

    /// Size of the scalar type in bytes. Panics for slice types.
    pub fn size_bytes(&self) -> usize {
        match self {
            KernelType::F32 | KernelType::I32 | KernelType::U32 => 4,
            KernelType::F64 | KernelType::I64 | KernelType::U64 => 8,
            KernelType::Bool => 1,
            KernelType::SliceRef(_) | KernelType::SliceMutRef(_) => {
                panic!("size_bytes() called on slice type")
            }
        }
    }

    /// The `PtxType` variant name as a string, for codegen (e.g., `"F32"`, `"U64"`).
    pub fn ptx_type_token(&self) -> &'static str {
        match self {
            KernelType::F32 => "F32",
            KernelType::F64 => "F64",
            KernelType::I32 => "S32",
            KernelType::U32 => "U32",
            KernelType::I64 => "S64",
            KernelType::U64 => "U64",
            KernelType::Bool => "Pred",
            KernelType::SliceRef(_) | KernelType::SliceMutRef(_) => {
                panic!("ptx_type_token() called on slice type")
            }
        }
    }

    /// Human-readable name for error messages.
    pub fn display_name(&self) -> String {
        match self {
            KernelType::F32 => "f32".to_string(),
            KernelType::F64 => "f64".to_string(),
            KernelType::I32 => "i32".to_string(),
            KernelType::U32 => "u32".to_string(),
            KernelType::I64 => "i64".to_string(),
            KernelType::U64 => "u64".to_string(),
            KernelType::Bool => "bool".to_string(),
            KernelType::SliceRef(inner) => format!("&[{}]", inner.display_name()),
            KernelType::SliceMutRef(inner) => format!("&mut [{}]", inner.display_name()),
        }
    }
}

/// A parsed kernel parameter.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields used in Sprint 2.2+ lowering
pub struct KernelParam {
    /// Parameter name.
    pub name: String,
    /// Parameter type.
    pub ty: KernelType,
    /// Source span for error reporting.
    pub span: Span,
}

/// Configuration parsed from `#[gpu_kernel(block_size = N)]`.
#[derive(Debug, Clone)]
#[allow(dead_code)] // block_size_span used in Sprint 2.7 validation
pub struct KernelConfig {
    /// Thread block size (number of threads per block).
    pub block_size: u32,
    /// Source span of the block_size value for error reporting.
    pub block_size_span: Span,
}

/// A fully parsed kernel signature (before body parsing).
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields used in Sprint 2.6 codegen
pub struct KernelSignature {
    /// Kernel function name.
    pub name: String,
    /// Kernel parameters in declaration order.
    pub params: Vec<KernelParam>,
    /// Macro attribute configuration.
    pub config: KernelConfig,
    /// Source span of the function name.
    pub name_span: Span,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_type_size_bytes() {
        assert_eq!(KernelType::F32.size_bytes(), 4);
        assert_eq!(KernelType::F64.size_bytes(), 8);
        assert_eq!(KernelType::U32.size_bytes(), 4);
        assert_eq!(KernelType::I64.size_bytes(), 8);
        assert_eq!(KernelType::Bool.size_bytes(), 1);
    }

    #[test]
    fn kernel_type_ptx_token() {
        assert_eq!(KernelType::F32.ptx_type_token(), "F32");
        assert_eq!(KernelType::I32.ptx_type_token(), "S32");
        assert_eq!(KernelType::U64.ptx_type_token(), "U64");
    }

    #[test]
    fn slice_type_properties() {
        let slice = KernelType::SliceRef(Box::new(KernelType::F32));
        assert!(slice.is_slice());
        assert!(!slice.is_scalar());
        assert!(!slice.is_mut_slice());
        assert_eq!(slice.elem_type(), Some(&KernelType::F32));

        let mut_slice = KernelType::SliceMutRef(Box::new(KernelType::F64));
        assert!(mut_slice.is_slice());
        assert!(mut_slice.is_mut_slice());
    }

    #[test]
    fn display_names() {
        assert_eq!(KernelType::F32.display_name(), "f32");
        assert_eq!(
            KernelType::SliceRef(Box::new(KernelType::F32)).display_name(),
            "&[f32]"
        );
        assert_eq!(
            KernelType::SliceMutRef(Box::new(KernelType::U32)).display_name(),
            "&mut [u32]"
        );
    }
}
