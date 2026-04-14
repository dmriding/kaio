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
    /// 16-bit float (`.f16`) — requires SM 5.3+
    F16,
    /// Brain float 16 (`.bf16`) — requires SM 8.0+
    ///
    /// Type-level support only in Sprint 6.1. Execution-path gating
    /// and hardware SM checks come in Sprint 6.5 (auto-tuner).
    BF16,
    /// 8-bit signed integer (`.s8`) — **marker / packed type only**.
    ///
    /// There is no scalar `.s8` register on NVIDIA GPUs. `s8` values
    /// live **packed four-per-`.b32`** inside register fragments and
    /// are addressed in bulk by memory operations (`ld/st.s8`) and by
    /// `mma.sync` instructions that declare `.s8` operand types in
    /// their suffix (e.g. `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`).
    ///
    /// Consequently:
    /// - `ptx_suffix()` → `.s8` (used in mma + memory-op suffixes)
    /// - `ptx_memory_suffix()` → `.s8` (PTX allows `ld.s8` / `st.s8`)
    /// - `reg_decl_type()` → `.b32` (s8 values share the `.b32` register
    ///   class; inventing a fake `.s8` register kind would lie about the
    ///   hardware reality)
    /// - `size_bytes()` → 1
    /// - `reg_kind()` → [`RegKind::R`] (the `%r` integer-register class)
    ///
    /// **Do not treat `S8` as a general scalar arithmetic type.** It is
    /// for memory loads/stores of packed bytes and for `mma.sync` operand
    /// suffix declarations only. Scalar integer arithmetic (add, sub,
    /// shift, etc.) continues to use [`S32`](Self::S32) / [`U32`](Self::U32);
    /// dequant inside a kernel casts `S8` → `S32` before any arithmetic,
    /// via shift-and-arith-shift sign-extension (see
    /// `examples/int8_dequant` for the reference pattern).
    ///
    /// Introduced in Sprint 7.1 for the `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`
    /// path (requires SM 8.0+).
    S8,
}

impl PtxType {
    /// Size of this type in bytes.
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::S8 | Self::Pred => 1,
            Self::F16 | Self::BF16 => 2,
            Self::F32 | Self::S32 | Self::U32 => 4,
            Self::F64 | Self::S64 | Self::U64 => 8,
        }
    }

    /// PTX type suffix string for arithmetic, cvt, mma, and register
    /// declarations (e.g. `.f32`, `.u64`, `.f16`, `.bf16`).
    pub fn ptx_suffix(&self) -> &'static str {
        match self {
            Self::F16 => ".f16",
            Self::BF16 => ".bf16",
            Self::F32 => ".f32",
            Self::F64 => ".f64",
            Self::S8 => ".s8",
            Self::S32 => ".s32",
            Self::U32 => ".u32",
            Self::S64 => ".s64",
            Self::U64 => ".u64",
            Self::Pred => ".pred",
        }
    }

    /// PTX type suffix for **memory operations** (`ld`, `st`, and
    /// `cp.async` variants).
    ///
    /// Differs from [`ptx_suffix`](Self::ptx_suffix) for half-precision
    /// types: the PTX spec (§8.7.9 "ld") lists the valid types as
    /// `{b8, b16, b32, b64, b128, u8, u16, u32, u64, s8, s16, s32,
    /// s64, f32, f64}` — `f16` and `bf16` are **not** valid type
    /// modifiers for scalar `ld`/`st`. To load a 16-bit half into an
    /// `.f16` register you emit `ld.global.b16 %h, [addr]`.
    ///
    /// For every non-half type this matches [`ptx_suffix`](Self::ptx_suffix).
    pub fn ptx_memory_suffix(&self) -> &'static str {
        match self {
            Self::F16 | Self::BF16 => ".b16",
            _ => self.ptx_suffix(),
        }
    }

    /// PTX type used in `.reg` declarations.
    ///
    /// Matches `nvcc` convention: integer registers use untyped `.b32`/`.b64`
    /// (the instruction carries signed/unsigned), while floats and predicates
    /// keep their typed declarations.
    pub fn reg_decl_type(&self) -> &'static str {
        match self {
            Self::F16 => ".f16",
            Self::BF16 => ".bf16",
            Self::F32 => ".f32",
            Self::F64 => ".f64",
            // S8 shares the `.b32` register class: s8 values live packed
            // four-per-register, there is no scalar `.s8` register.
            Self::S8 | Self::S32 | Self::U32 => ".b32",
            Self::S64 | Self::U64 => ".b64",
            Self::Pred => ".pred",
        }
    }

    /// Which register kind this type maps to.
    pub fn reg_kind(&self) -> RegKind {
        match self {
            Self::F16 => RegKind::H,
            Self::BF16 => RegKind::Hb,
            Self::F32 => RegKind::F,
            Self::F64 => RegKind::Fd,
            // S8 uses the `%r` class (same as S32/U32) because s8 values
            // live packed four-per-`.b32` register.
            Self::S8 | Self::S32 | Self::U32 => RegKind::R,
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
///
/// Models PTX register declaration classes directly, not higher-level
/// numeric families. This is intentional — see Phase 6.1 design notes.
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
    /// `%h` — 16-bit float (f16)
    H,
    /// `%hb` — brain float 16 (bf16)
    Hb,
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
            Self::H => "%h",
            Self::Hb => "%hb",
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
            Self::H => 5,
            Self::Hb => 6,
        }
    }
}

mod private {
    pub trait Sealed {}
}

/// Marker trait for Rust types that can be used in GPU kernels.
///
/// Sealed — only implemented for types KAIO knows about. Maps each Rust
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

impl_gpu_type!(half::f16, PtxType::F16);
impl_gpu_type!(half::bf16, PtxType::BF16);
impl_gpu_type!(f32, PtxType::F32);
impl_gpu_type!(f64, PtxType::F64);
impl_gpu_type!(i8, PtxType::S8);
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
        assert_eq!(PtxType::F16.size_bytes(), 2);
        assert_eq!(PtxType::BF16.size_bytes(), 2);
        assert_eq!(PtxType::F32.size_bytes(), 4);
        assert_eq!(PtxType::F64.size_bytes(), 8);
        assert_eq!(PtxType::S8.size_bytes(), 1);
        assert_eq!(PtxType::S32.size_bytes(), 4);
        assert_eq!(PtxType::U32.size_bytes(), 4);
        assert_eq!(PtxType::S64.size_bytes(), 8);
        assert_eq!(PtxType::U64.size_bytes(), 8);
        assert_eq!(PtxType::Pred.size_bytes(), 1);
    }

    #[test]
    fn ptx_type_suffix() {
        assert_eq!(PtxType::F16.ptx_suffix(), ".f16");
        assert_eq!(PtxType::BF16.ptx_suffix(), ".bf16");
        assert_eq!(PtxType::F32.ptx_suffix(), ".f32");
        assert_eq!(PtxType::F64.ptx_suffix(), ".f64");
        assert_eq!(PtxType::S8.ptx_suffix(), ".s8");
        assert_eq!(PtxType::S32.ptx_suffix(), ".s32");
        assert_eq!(PtxType::U32.ptx_suffix(), ".u32");
        assert_eq!(PtxType::S64.ptx_suffix(), ".s64");
        assert_eq!(PtxType::U64.ptx_suffix(), ".u64");
        assert_eq!(PtxType::Pred.ptx_suffix(), ".pred");
    }

    #[test]
    fn ptx_type_memory_suffix() {
        // Half types collapse to .b16 for memory ops (PTX ISA §8.7.9).
        assert_eq!(PtxType::F16.ptx_memory_suffix(), ".b16");
        assert_eq!(PtxType::BF16.ptx_memory_suffix(), ".b16");
        // All other types use their native suffix.
        assert_eq!(PtxType::F32.ptx_memory_suffix(), ".f32");
        assert_eq!(PtxType::F64.ptx_memory_suffix(), ".f64");
        // S8 uses .s8 for memory ops (PTX ISA lists s8 as valid for ld/st).
        assert_eq!(PtxType::S8.ptx_memory_suffix(), ".s8");
        assert_eq!(PtxType::S32.ptx_memory_suffix(), ".s32");
        assert_eq!(PtxType::U32.ptx_memory_suffix(), ".u32");
        assert_eq!(PtxType::S64.ptx_memory_suffix(), ".s64");
        assert_eq!(PtxType::U64.ptx_memory_suffix(), ".u64");
    }

    #[test]
    fn ptx_type_reg_decl_type() {
        // Half types keep typed declarations
        assert_eq!(PtxType::F16.reg_decl_type(), ".f16");
        assert_eq!(PtxType::BF16.reg_decl_type(), ".bf16");
        // Integers collapse to untyped bit-width (matches nvcc convention).
        // S8 shares the .b32 register class because s8 values are packed
        // four-per-register; there is no scalar .s8 register.
        assert_eq!(PtxType::S8.reg_decl_type(), ".b32");
        assert_eq!(PtxType::S32.reg_decl_type(), ".b32");
        assert_eq!(PtxType::U32.reg_decl_type(), ".b32");
        assert_eq!(PtxType::S64.reg_decl_type(), ".b64");
        assert_eq!(PtxType::U64.reg_decl_type(), ".b64");
        // Floats and predicates keep their typed declarations
        assert_eq!(PtxType::F32.reg_decl_type(), ".f32");
        assert_eq!(PtxType::F64.reg_decl_type(), ".f64");
        assert_eq!(PtxType::Pred.reg_decl_type(), ".pred");
    }

    #[test]
    fn ptx_type_reg_kind() {
        // Half types
        assert_eq!(PtxType::F16.reg_kind(), RegKind::H);
        assert_eq!(PtxType::BF16.reg_kind(), RegKind::Hb);
        // S8 shares the %r class with S32/U32 (packed four-per-b32 register).
        assert_eq!(PtxType::S8.reg_kind(), RegKind::R);
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
        assert_eq!(RegKind::H.prefix(), "%h");
        assert_eq!(RegKind::Hb.prefix(), "%hb");
    }

    #[test]
    fn gpu_type_impls() {
        assert_eq!(<half::f16 as GpuType>::PTX_TYPE, PtxType::F16);
        assert_eq!(<half::bf16 as GpuType>::PTX_TYPE, PtxType::BF16);
        assert_eq!(<f32 as GpuType>::PTX_TYPE, PtxType::F32);
        assert_eq!(<f64 as GpuType>::PTX_TYPE, PtxType::F64);
        assert_eq!(<i8 as GpuType>::PTX_TYPE, PtxType::S8);
        assert_eq!(<i32 as GpuType>::PTX_TYPE, PtxType::S32);
        assert_eq!(<u32 as GpuType>::PTX_TYPE, PtxType::U32);
        assert_eq!(<i64 as GpuType>::PTX_TYPE, PtxType::S64);
        assert_eq!(<u64 as GpuType>::PTX_TYPE, PtxType::U64);
        assert_eq!(<bool as GpuType>::PTX_TYPE, PtxType::Pred);
    }
}
