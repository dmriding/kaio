//! Helper functions for reading PTX special registers.
//!
//! Each function allocates a `u32` register and returns it alongside a
//! `Mov` instruction that reads from the corresponding special register.
//! The caller pushes the instruction into the kernel body and uses the
//! register in subsequent instructions.
//!
//! # Example
//!
//! ```ignore
//! let (tid, tid_instr) = special::tid_x(&mut alloc);
//! kernel.push(tid_instr);
//! // tid now holds %tid.x, usable in arithmetic
//! ```

use crate::ir::{Operand, PtxInstruction, Register, RegisterAllocator, SpecialReg};
use crate::types::PtxType;

/// Read `%tid.x` (thread index X) into a fresh register.
pub fn tid_x(alloc: &mut RegisterAllocator) -> (Register, PtxInstruction) {
    read_special(alloc, SpecialReg::TidX)
}

/// Read `%tid.y` (thread index Y) into a fresh register.
pub fn tid_y(alloc: &mut RegisterAllocator) -> (Register, PtxInstruction) {
    read_special(alloc, SpecialReg::TidY)
}

/// Read `%tid.z` (thread index Z) into a fresh register.
pub fn tid_z(alloc: &mut RegisterAllocator) -> (Register, PtxInstruction) {
    read_special(alloc, SpecialReg::TidZ)
}

/// Read `%ntid.x` (block dimension X / threads per block) into a fresh register.
pub fn ntid_x(alloc: &mut RegisterAllocator) -> (Register, PtxInstruction) {
    read_special(alloc, SpecialReg::NtidX)
}

/// Read `%ntid.y` (block dimension Y) into a fresh register.
pub fn ntid_y(alloc: &mut RegisterAllocator) -> (Register, PtxInstruction) {
    read_special(alloc, SpecialReg::NtidY)
}

/// Read `%ntid.z` (block dimension Z) into a fresh register.
pub fn ntid_z(alloc: &mut RegisterAllocator) -> (Register, PtxInstruction) {
    read_special(alloc, SpecialReg::NtidZ)
}

/// Read `%ctaid.x` (block/CTA index X) into a fresh register.
pub fn ctaid_x(alloc: &mut RegisterAllocator) -> (Register, PtxInstruction) {
    read_special(alloc, SpecialReg::CtaidX)
}

/// Read `%ctaid.y` (block/CTA index Y) into a fresh register.
pub fn ctaid_y(alloc: &mut RegisterAllocator) -> (Register, PtxInstruction) {
    read_special(alloc, SpecialReg::CtaidY)
}

/// Read `%ctaid.z` (block/CTA index Z) into a fresh register.
pub fn ctaid_z(alloc: &mut RegisterAllocator) -> (Register, PtxInstruction) {
    read_special(alloc, SpecialReg::CtaidZ)
}

/// Read `%nctaid.x` (grid dimension X / blocks per grid) into a fresh register.
pub fn nctaid_x(alloc: &mut RegisterAllocator) -> (Register, PtxInstruction) {
    read_special(alloc, SpecialReg::NctaidX)
}

/// Read `%nctaid.y` (grid dimension Y) into a fresh register.
pub fn nctaid_y(alloc: &mut RegisterAllocator) -> (Register, PtxInstruction) {
    read_special(alloc, SpecialReg::NctaidY)
}

/// Read `%nctaid.z` (grid dimension Z) into a fresh register.
pub fn nctaid_z(alloc: &mut RegisterAllocator) -> (Register, PtxInstruction) {
    read_special(alloc, SpecialReg::NctaidZ)
}

/// Internal: allocate a u32 register and create a mov from a special register.
fn read_special(alloc: &mut RegisterAllocator, sr: SpecialReg) -> (Register, PtxInstruction) {
    let reg = alloc.alloc(PtxType::U32);
    let instr = PtxInstruction::Mov {
        dst: reg,
        src: Operand::SpecialReg(sr),
        ty: PtxType::U32,
    };
    (reg, instr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::RegKind;

    #[test]
    fn special_tid_x() {
        let mut alloc = RegisterAllocator::new();
        let (reg, instr) = tid_x(&mut alloc);

        // Register should be %r0 (first u32 allocation)
        assert_eq!(reg.kind, RegKind::R);
        assert_eq!(reg.index, 0);
        assert_eq!(reg.ptx_type, PtxType::U32);

        // Instruction should be Mov from %tid.x
        match &instr {
            PtxInstruction::Mov { dst, src, ty } => {
                assert_eq!(*dst, reg);
                assert_eq!(*ty, PtxType::U32);
                match src {
                    Operand::SpecialReg(sr) => assert_eq!(*sr, SpecialReg::TidX),
                    other => panic!("expected SpecialReg, got {other:?}"),
                }
            }
            other => panic!("expected Mov, got {other:?}"),
        }
    }

    #[test]
    fn special_ctaid_x() {
        let mut alloc = RegisterAllocator::new();
        let (reg, instr) = ctaid_x(&mut alloc);

        assert_eq!(reg.kind, RegKind::R);
        assert_eq!(reg.index, 0);

        match &instr {
            PtxInstruction::Mov { src, .. } => match src {
                Operand::SpecialReg(sr) => assert_eq!(*sr, SpecialReg::CtaidX),
                other => panic!("expected SpecialReg, got {other:?}"),
            },
            other => panic!("expected Mov, got {other:?}"),
        }
    }
}
