//! PTX type system and Rust-to-PTX type mapping.

/// PTX data types corresponding to Rust primitives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PtxType {
    /// 32-bit float (`.f32`)
    F32,
    /// 64-bit float (`.f64`)
    F64,
    /// 32-bit signed integer (`.s32`)
    S32,
    /// 32-bit unsigned integer (`.u32`)
    U32,
    /// 64-bit signed integer (`.s64`)
    S64,
    /// 64-bit unsigned integer (`.u64`)
    U64,
    /// Predicate / boolean (`.pred`)
    Pred,
    // F16/BF16 intentionally omitted — deferred to Phase 3+ per master plan.
    // Adding them requires RegKind extensions (%h, %hb) and half-precision deps.
}

impl PtxType {
    /// Size of this type in bytes.
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::F32 | Self::S32 | Self::U32 => 4,
            Self::F64 | Self::S64 | Self::U64 => 8,
            Self::Pred => 1,
        }
    }

    /// PTX type suffix string (e.g. `.f32`, `.u64`).
    pub fn ptx_suffix(&self) -> &'static str {
        match self {
            Self::F32 => ".f32",
            Self::F64 => ".f64",
            Self::S32 => ".s32",
            Self::U32 => ".u32",
            Self::S64 => ".s64",
            Self::U64 => ".u64",
            Self::Pred => ".pred",
        }
    }

    /// Which register kind this type maps to.
    pub fn reg_kind(&self) -> RegKind {
        match self {
            Self::F32 => RegKind::F,
            Self::F64 => RegKind::Fd,
            Self::S32 | Self::U32 => RegKind::R,
            Self::S64 | Self::U64 => RegKind::Rd,
            Self::Pred => RegKind::P,
        }
    }
}

/// Register kind — determines the register name prefix in PTX.
///
/// Collapses signed/unsigned into the same prefix (both `i32` and `u32`
/// use `%r`, both `i64` and `u64` use `%rd`). This matches `nvcc` output
/// conventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegKind {
    /// `%r` — 32-bit integer (s32, u32)
    R,
    /// `%rd` — 64-bit integer (s64, u64)
    Rd,
    /// `%f` — 32-bit float (f32)
    F,
    /// `%fd` — 64-bit float (f64)
    Fd,
    /// `%p` — predicate (bool)
    P,
}

impl RegKind {
    /// PTX register name prefix.
    pub fn prefix(&self) -> &'static str {
        match self {
            Self::R => "%r",
            Self::Rd => "%rd",
            Self::F => "%f",
            Self::Fd => "%fd",
            Self::P => "%p",
        }
    }

    /// Index into the register allocator's counter array.
    pub(crate) fn counter_index(&self) -> usize {
        match self {
            Self::R => 0,
            Self::Rd => 1,
            Self::F => 2,
            Self::Fd => 3,
            Self::P => 4,
        }
    }
}

mod private {
    pub trait Sealed {}
}

/// Marker trait for Rust types that can be used in GPU kernels.
///
/// Sealed — only implemented for types PYROS knows about. Maps each Rust
/// type to its corresponding [`PtxType`].
pub trait GpuType: private::Sealed + Copy + 'static {
    /// The PTX type this Rust type maps to.
    const PTX_TYPE: PtxType;
}

macro_rules! impl_gpu_type {
    ($rust_ty:ty, $ptx_ty:expr) => {
        impl private::Sealed for $rust_ty {}
        impl GpuType for $rust_ty {
            const PTX_TYPE: PtxType = $ptx_ty;
        }
    };
}

impl_gpu_type!(f32, PtxType::F32);
impl_gpu_type!(f64, PtxType::F64);
impl_gpu_type!(i32, PtxType::S32);
impl_gpu_type!(u32, PtxType::U32);
impl_gpu_type!(i64, PtxType::S64);
impl_gpu_type!(u64, PtxType::U64);
impl_gpu_type!(bool, PtxType::Pred);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ptx_type_size_bytes() {
        assert_eq!(PtxType::F32.size_bytes(), 4);
        assert_eq!(PtxType::F64.size_bytes(), 8);
        assert_eq!(PtxType::S32.size_bytes(), 4);
        assert_eq!(PtxType::U32.size_bytes(), 4);
        assert_eq!(PtxType::S64.size_bytes(), 8);
        assert_eq!(PtxType::U64.size_bytes(), 8);
        assert_eq!(PtxType::Pred.size_bytes(), 1);
    }

    #[test]
    fn ptx_type_suffix() {
        assert_eq!(PtxType::F32.ptx_suffix(), ".f32");
        assert_eq!(PtxType::F64.ptx_suffix(), ".f64");
        assert_eq!(PtxType::S32.ptx_suffix(), ".s32");
        assert_eq!(PtxType::U32.ptx_suffix(), ".u32");
        assert_eq!(PtxType::S64.ptx_suffix(), ".s64");
        assert_eq!(PtxType::U64.ptx_suffix(), ".u64");
        assert_eq!(PtxType::Pred.ptx_suffix(), ".pred");
    }

    #[test]
    fn ptx_type_reg_kind() {
        // Signed and unsigned 32-bit both map to R
        assert_eq!(PtxType::S32.reg_kind(), RegKind::R);
        assert_eq!(PtxType::U32.reg_kind(), RegKind::R);
        // Signed and unsigned 64-bit both map to Rd
        assert_eq!(PtxType::S64.reg_kind(), RegKind::Rd);
        assert_eq!(PtxType::U64.reg_kind(), RegKind::Rd);
        // Floats
        assert_eq!(PtxType::F32.reg_kind(), RegKind::F);
        assert_eq!(PtxType::F64.reg_kind(), RegKind::Fd);
        // Predicate
        assert_eq!(PtxType::Pred.reg_kind(), RegKind::P);
    }

    #[test]
    fn reg_kind_prefix() {
        assert_eq!(RegKind::R.prefix(), "%r");
        assert_eq!(RegKind::Rd.prefix(), "%rd");
        assert_eq!(RegKind::F.prefix(), "%f");
        assert_eq!(RegKind::Fd.prefix(), "%fd");
        assert_eq!(RegKind::P.prefix(), "%p");
    }

    #[test]
    fn gpu_type_impls() {
        assert_eq!(<f32 as GpuType>::PTX_TYPE, PtxType::F32);
        assert_eq!(<f64 as GpuType>::PTX_TYPE, PtxType::F64);
        assert_eq!(<i32 as GpuType>::PTX_TYPE, PtxType::S32);
        assert_eq!(<u32 as GpuType>::PTX_TYPE, PtxType::U32);
        assert_eq!(<i64 as GpuType>::PTX_TYPE, PtxType::S64);
        assert_eq!(<u64 as GpuType>::PTX_TYPE, PtxType::U64);
        assert_eq!(<bool as GpuType>::PTX_TYPE, PtxType::Pred);
    }
}
