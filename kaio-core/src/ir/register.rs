//! PTX register types and virtual register allocator.

use std::fmt;

use crate::types::{PtxType, RegKind};

/// A virtual PTX register with a kind prefix and unique index.
///
/// Register names follow PTX conventions: `%r0` (32-bit int), `%rd0`
/// (64-bit int), `%f0` (f32), `%fd0` (f64), `%p0` (predicate).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Register {
    /// Which register name prefix to use.
    pub kind: RegKind,
    /// Unique index within this kind.
    pub index: u32,
    /// The PTX type this register was declared with.
    pub ptx_type: PtxType,
}

impl Register {
    /// The PTX register name (e.g. `%r0`, `%fd3`, `%p1`).
    pub fn name(&self) -> String {
        format!("{}{}", self.kind.prefix(), self.index)
    }
}

impl fmt::Display for Register {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.kind.prefix(), self.index)
    }
}

/// Allocates virtual registers with unique names per kind prefix.
///
/// Each call to [`alloc`](Self::alloc) returns a [`Register`] with a
/// monotonically increasing index within its [`RegKind`]. The allocator
/// tracks all allocations so they can be emitted as `.reg` declarations
/// in the PTX kernel prelude.
#[derive(Debug)]
pub struct RegisterAllocator {
    counters: [u32; 7],
    allocated: Vec<Register>,
}

impl Default for RegisterAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl RegisterAllocator {
    /// Create a new allocator with all counters at zero.
    pub fn new() -> Self {
        Self {
            counters: [0; 7],
            allocated: Vec::new(),
        }
    }

    /// Allocate a fresh register for the given PTX type.
    pub fn alloc(&mut self, ptx_type: PtxType) -> Register {
        let kind = ptx_type.reg_kind();
        let idx = kind.counter_index();
        let index = self.counters[idx];
        self.counters[idx] += 1;
        let reg = Register {
            kind,
            index,
            ptx_type,
        };
        self.allocated.push(reg);
        reg
    }

    /// All registers allocated so far, in allocation order.
    pub fn allocated(&self) -> &[Register] {
        &self.allocated
    }

    /// Consume the allocator and return all allocated registers.
    pub fn into_allocated(self) -> Vec<Register> {
        self.allocated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequential_indices_within_kind() {
        let mut alloc = RegisterAllocator::new();
        let r0 = alloc.alloc(PtxType::S32);
        let r1 = alloc.alloc(PtxType::U32); // same RegKind::R
        let r2 = alloc.alloc(PtxType::S32);
        assert_eq!(r0.index, 0);
        assert_eq!(r1.index, 1);
        assert_eq!(r2.index, 2);
        assert_eq!(r0.name(), "%r0");
        assert_eq!(r1.name(), "%r1");
        assert_eq!(r2.name(), "%r2");
    }

    #[test]
    fn independent_counters_per_kind() {
        let mut alloc = RegisterAllocator::new();
        let r = alloc.alloc(PtxType::S32);
        let f = alloc.alloc(PtxType::F32);
        let rd = alloc.alloc(PtxType::U64);
        let p = alloc.alloc(PtxType::Pred);
        let fd = alloc.alloc(PtxType::F64);
        let h = alloc.alloc(PtxType::F16);
        let hb = alloc.alloc(PtxType::BF16);
        // All should be index 0 — independent counters
        assert_eq!(r.index, 0);
        assert_eq!(f.index, 0);
        assert_eq!(rd.index, 0);
        assert_eq!(p.index, 0);
        assert_eq!(fd.index, 0);
        assert_eq!(h.index, 0);
        assert_eq!(hb.index, 0);
        assert_eq!(r.name(), "%r0");
        assert_eq!(f.name(), "%f0");
        assert_eq!(rd.name(), "%rd0");
        assert_eq!(p.name(), "%p0");
        assert_eq!(fd.name(), "%fd0");
        assert_eq!(h.name(), "%h0");
        assert_eq!(hb.name(), "%hb0");
    }

    #[test]
    fn into_allocated_preserves_order() {
        let mut alloc = RegisterAllocator::new();
        let r0 = alloc.alloc(PtxType::F32);
        let r1 = alloc.alloc(PtxType::S32);
        let r2 = alloc.alloc(PtxType::F32);
        let regs = alloc.into_allocated();
        assert_eq!(regs.len(), 3);
        assert_eq!(regs[0], r0);
        assert_eq!(regs[1], r1);
        assert_eq!(regs[2], r2);
    }

    #[test]
    fn f16_bf16_register_allocation() {
        let mut alloc = RegisterAllocator::new();
        let h0 = alloc.alloc(PtxType::F16);
        let h1 = alloc.alloc(PtxType::F16);
        let hb0 = alloc.alloc(PtxType::BF16);
        let hb1 = alloc.alloc(PtxType::BF16);
        // f16 counters are independent from bf16
        assert_eq!(h0.index, 0);
        assert_eq!(h1.index, 1);
        assert_eq!(hb0.index, 0);
        assert_eq!(hb1.index, 1);
        assert_eq!(h0.name(), "%h0");
        assert_eq!(h1.name(), "%h1");
        assert_eq!(hb0.name(), "%hb0");
        assert_eq!(hb1.name(), "%hb1");
        // Verify kinds
        assert_eq!(h0.kind, RegKind::H);
        assert_eq!(hb0.kind, RegKind::Hb);
    }

    #[test]
    fn register_display() {
        let reg = Register {
            kind: RegKind::Fd,
            index: 7,
            ptx_type: PtxType::F64,
        };
        assert_eq!(format!("{reg}"), "%fd7");
    }
}
