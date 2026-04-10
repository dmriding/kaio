//! PTX operand types (register references, immediates, special registers).

use std::fmt;

use super::register::Register;

/// An operand to a PTX instruction.
#[derive(Debug, Clone)]
pub enum Operand {
    /// A virtual register.
    Reg(Register),
    /// 32-bit signed integer immediate.
    ImmI32(i32),
    /// 32-bit unsigned integer immediate.
    ImmU32(u32),
    /// 64-bit signed integer immediate.
    ImmI64(i64),
    /// 64-bit unsigned integer immediate.
    ImmU64(u64),
    /// 32-bit float immediate.
    ImmF32(f32),
    /// 64-bit float immediate.
    ImmF64(f64),
    /// A PTX special register (`%tid.x`, `%ntid.x`, etc.).
    SpecialReg(SpecialReg),
    /// Address of a named shared memory allocation.
    ///
    /// Used with `Mov` to load a shared allocation's base address into a
    /// register: `mov.u32 %r0, sdata;`. Displays as the bare name.
    SharedAddr(String),
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reg(r) => write!(f, "{r}"),
            Self::ImmI32(v) => write!(f, "{v}"),
            Self::ImmU32(v) => write!(f, "{v}"),
            Self::ImmI64(v) => write!(f, "{v}"),
            Self::ImmU64(v) => write!(f, "{v}"),
            Self::ImmF32(v) => {
                // PTX requires decimal point for floats
                if v.fract() == 0.0 {
                    write!(f, "{v:.1}")
                } else {
                    write!(f, "{v}")
                }
            }
            Self::ImmF64(v) => {
                if v.fract() == 0.0 {
                    write!(f, "{v:.1}")
                } else {
                    write!(f, "{v}")
                }
            }
            Self::SpecialReg(sr) => write!(f, "{}", sr.ptx_name()),
            Self::SharedAddr(name) => write!(f, "{name}"),
        }
    }
}

/// PTX special registers for thread/block/grid indexing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpecialReg {
    /// `%tid.x` — thread index X
    TidX,
    /// `%tid.y` — thread index Y
    TidY,
    /// `%tid.z` — thread index Z
    TidZ,
    /// `%ntid.x` — block dimension X (threads per block)
    NtidX,
    /// `%ntid.y` — block dimension Y
    NtidY,
    /// `%ntid.z` — block dimension Z
    NtidZ,
    /// `%ctaid.x` — block/CTA index X
    CtaidX,
    /// `%ctaid.y` — block/CTA index Y
    CtaidY,
    /// `%ctaid.z` — block/CTA index Z
    CtaidZ,
    /// `%nctaid.x` — grid dimension X (blocks per grid)
    NctaidX,
    /// `%nctaid.y` — grid dimension Y
    NctaidY,
    /// `%nctaid.z` — grid dimension Z
    NctaidZ,
}

impl SpecialReg {
    /// The PTX name of this special register (e.g. `%tid.x`).
    pub fn ptx_name(&self) -> &'static str {
        match self {
            Self::TidX => "%tid.x",
            Self::TidY => "%tid.y",
            Self::TidZ => "%tid.z",
            Self::NtidX => "%ntid.x",
            Self::NtidY => "%ntid.y",
            Self::NtidZ => "%ntid.z",
            Self::CtaidX => "%ctaid.x",
            Self::CtaidY => "%ctaid.y",
            Self::CtaidZ => "%ctaid.z",
            Self::NctaidX => "%nctaid.x",
            Self::NctaidY => "%nctaid.y",
            Self::NctaidZ => "%nctaid.z",
        }
    }
}
