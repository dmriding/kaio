//! PTX kernel — a single `.entry` function in a PTX module.

use super::instruction::PtxInstruction;
use super::param::PtxParam;
use super::register::Register;
use crate::instr::ArithOp;
use crate::instr::control::ControlOp;
use crate::instr::memory::MemoryOp;
use crate::instr::tensor_core::TensorCoreOp;
use crate::types::RegKind;

/// Shared memory declaration in a PTX kernel preamble.
///
/// Emitted as `.shared .align {align} .b8 {name}[{size_bytes}];` after
/// register declarations.
#[derive(Debug, Clone)]
pub struct SharedDecl {
    /// Name of the shared memory allocation (e.g., `"sdata"`).
    pub name: String,
    /// Alignment in bytes (4 for f32, 8 for f64).
    pub align: u32,
    /// Total allocation size in bytes.
    pub size_bytes: u32,
}

/// A PTX kernel function (`.visible .entry`).
///
/// Built by constructing parameters, allocating registers, and pushing
/// instructions. Call [`set_registers`](Self::set_registers) with the
/// allocator's output before emission so the kernel knows which `.reg`
/// declarations to emit.
#[derive(Debug, Clone)]
pub struct PtxKernel {
    /// Kernel entry point name.
    pub name: String,
    /// Declared parameters (in signature order).
    pub params: Vec<PtxParam>,
    /// Instruction body.
    pub body: Vec<PtxInstruction>,
    /// All registers used, for `.reg` declaration emission.
    pub registers: Vec<Register>,
    /// Shared memory declarations (emitted after register declarations).
    pub shared_decls: Vec<SharedDecl>,
}

impl PtxKernel {
    /// Create a new empty kernel with the given name.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            params: Vec::new(),
            body: Vec::new(),
            registers: Vec::new(),
            shared_decls: Vec::new(),
        }
    }

    /// Add a parameter to the kernel signature.
    pub fn add_param(&mut self, param: PtxParam) {
        self.params.push(param);
    }

    /// Append an instruction to the kernel body.
    pub fn push(&mut self, instr: PtxInstruction) {
        self.body.push(instr);
    }

    /// Set the register list (from [`super::register::RegisterAllocator::into_allocated`]).
    pub fn set_registers(&mut self, regs: Vec<Register>) {
        self.registers = regs;
    }

    /// Add a shared memory declaration to the kernel preamble.
    pub fn add_shared_decl(&mut self, decl: SharedDecl) {
        self.shared_decls.push(decl);
    }

    /// Compute structural statistics about this kernel's emitted PTX.
    ///
    /// Walks the instruction body and counts instruction types, registers
    /// by kind, and declared shared memory. Useful for inspection and
    /// comparison between kernel variants.
    ///
    /// These are **not** runtime profiling data — final hardware register
    /// allocation and occupancy may differ after CUDA driver compilation.
    pub fn stats(&self) -> KernelStats {
        let mut s = KernelStats::default();

        for instr in &self.body {
            match instr {
                PtxInstruction::Arith(op) => {
                    s.total_instructions += 1;
                    if matches!(op, ArithOp::Fma { .. }) {
                        s.fma += 1;
                    } else {
                        s.arith_other += 1;
                    }
                }
                PtxInstruction::Memory(op) => {
                    s.total_instructions += 1;
                    match op {
                        MemoryOp::LdGlobal { .. } => s.ld_global += 1,
                        MemoryOp::StGlobal { .. } => s.st_global += 1,
                        MemoryOp::LdShared { .. } => s.ld_shared += 1,
                        MemoryOp::StShared { .. } => s.st_shared += 1,
                        MemoryOp::CpAsyncCaSharedGlobal { .. } => s.cp_async += 1,
                        MemoryOp::CpAsyncCommitGroup => s.cp_async_commit += 1,
                        MemoryOp::CpAsyncWaitGroup { .. } => s.cp_async_wait += 1,
                        _ => {}
                    }
                }
                PtxInstruction::TensorCore(op) => {
                    s.total_instructions += 1;
                    match op {
                        TensorCoreOp::MmaSync { .. }
                        | TensorCoreOp::MmaSyncInt8 { .. }
                        | TensorCoreOp::MmaSyncBf16 { .. } => s.mma += 1,
                    }
                }
                PtxInstruction::Control(op) => {
                    s.total_instructions += 1;
                    match op {
                        ControlOp::BarSync { .. } => s.bar_sync += 1,
                        ControlOp::BraPred { .. } | ControlOp::Bra { .. } => s.branches += 1,
                        ControlOp::SetP { .. } => s.setp += 1,
                        _ => {}
                    }
                }
                PtxInstruction::Mov { .. } => {
                    s.total_instructions += 1;
                    s.mov += 1;
                }
                PtxInstruction::Cvt { .. } => {
                    s.total_instructions += 1;
                    s.cvt += 1;
                }
                PtxInstruction::MovPack { .. } => {
                    s.total_instructions += 1;
                    s.mov += 1;
                }
                PtxInstruction::Label(_) | PtxInstruction::Comment(_) => {}
            }
        }

        for reg in &self.registers {
            match reg.kind {
                RegKind::R => s.registers_r += 1,
                RegKind::Rd => s.registers_rd += 1,
                RegKind::F => s.registers_f += 1,
                RegKind::Fd => s.registers_fd += 1,
                RegKind::P => s.registers_p += 1,
                RegKind::H => s.registers_h += 1,
                RegKind::Hb => s.registers_hb += 1,
            }
        }

        s.shared_bytes = self.shared_decls.iter().map(|d| d.size_bytes).sum();

        s
    }
}

/// Structural statistics about a compiled kernel's emitted PTX.
///
/// These describe the instruction mix and declared resource usage in
/// KAIO's generated PTX — useful for inspection and comparison between
/// kernel variants, but **not** a substitute for runtime profiling.
/// Final hardware register allocation and occupancy may differ from
/// these counts after the CUDA driver's backend compilation (PTX → SASS).
#[derive(Debug, Default, PartialEq, Eq)]
pub struct KernelStats {
    /// Total instructions (excludes labels and comments).
    pub total_instructions: usize,
    /// `ld.global` count.
    pub ld_global: usize,
    /// `st.global` count.
    pub st_global: usize,
    /// `ld.shared` count.
    pub ld_shared: usize,
    /// `st.shared` count.
    pub st_shared: usize,
    /// `bar.sync` count.
    pub bar_sync: usize,
    /// `mma.sync` instruction count (all tensor-core shapes).
    pub mma: usize,
    /// `cp.async.ca.shared.global` instruction count.
    pub cp_async: usize,
    /// `cp.async.commit_group` instruction count.
    pub cp_async_commit: usize,
    /// `cp.async.wait_group` instruction count.
    pub cp_async_wait: usize,
    /// `fma` instruction count.
    pub fma: usize,
    /// Non-FMA arithmetic instructions (add, mul, sub, etc.).
    pub arith_other: usize,
    /// `mov` instruction count.
    pub mov: usize,
    /// `cvt` instruction count.
    pub cvt: usize,
    /// Branch instructions (`bra`, `@pred bra`).
    pub branches: usize,
    /// `setp` comparison-to-predicate instructions.
    pub setp: usize,
    /// `%r` registers (32-bit integer).
    pub registers_r: u32,
    /// `%rd` registers (64-bit integer).
    pub registers_rd: u32,
    /// `%f` registers (f32).
    pub registers_f: u32,
    /// `%fd` registers (f64).
    pub registers_fd: u32,
    /// `%p` registers (predicate).
    pub registers_p: u32,
    /// `%h` registers (f16).
    pub registers_h: u32,
    /// `%hb` registers (bf16).
    pub registers_hb: u32,
    /// Total declared shared memory in bytes.
    pub shared_bytes: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Operand;
    use crate::types::PtxType;

    fn reg(kind: RegKind, index: u32, ptx_type: PtxType) -> Register {
        Register {
            kind,
            index,
            ptx_type,
        }
    }

    #[test]
    fn stats_empty_kernel() {
        let kernel = PtxKernel::new("empty");
        let s = kernel.stats();
        assert_eq!(s, KernelStats::default());
    }

    #[test]
    fn stats_counts_instruction_types() {
        let mut kernel = PtxKernel::new("test");

        // 2 FMA
        for _ in 0..2 {
            kernel.push(PtxInstruction::Arith(ArithOp::Fma {
                dst: reg(RegKind::F, 0, PtxType::F32),
                a: Operand::Reg(reg(RegKind::F, 1, PtxType::F32)),
                b: Operand::Reg(reg(RegKind::F, 2, PtxType::F32)),
                c: Operand::Reg(reg(RegKind::F, 3, PtxType::F32)),
                ty: PtxType::F32,
            }));
        }
        // 1 Add (arith_other)
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: reg(RegKind::R, 0, PtxType::U32),
            lhs: Operand::Reg(reg(RegKind::R, 1, PtxType::U32)),
            rhs: Operand::ImmU32(1),
            ty: PtxType::U32,
        }));
        // 1 ld.global + 1 st.global
        kernel.push(PtxInstruction::Memory(MemoryOp::LdGlobal {
            dst: reg(RegKind::F, 0, PtxType::F32),
            addr: reg(RegKind::Rd, 0, PtxType::U64),
            ty: PtxType::F32,
        }));
        kernel.push(PtxInstruction::Memory(MemoryOp::StGlobal {
            addr: reg(RegKind::Rd, 0, PtxType::U64),
            src: reg(RegKind::F, 0, PtxType::F32),
            ty: PtxType::F32,
        }));
        // 1 ld.shared + 1 st.shared
        kernel.push(PtxInstruction::Memory(MemoryOp::LdShared {
            dst: reg(RegKind::F, 0, PtxType::F32),
            addr: reg(RegKind::R, 0, PtxType::U32),
            ty: PtxType::F32,
        }));
        kernel.push(PtxInstruction::Memory(MemoryOp::StShared {
            addr: reg(RegKind::R, 0, PtxType::U32),
            src: reg(RegKind::F, 0, PtxType::F32),
            ty: PtxType::F32,
        }));
        // 1 ld.param (memory, total-only)
        kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
            dst: reg(RegKind::Rd, 0, PtxType::U64),
            param_name: "p0".to_string(),
            ty: PtxType::U64,
        }));
        // 1 bar.sync
        kernel.push(PtxInstruction::Control(ControlOp::BarSync {
            barrier_id: 0,
        }));
        // 1 branch + 1 setp
        kernel.push(PtxInstruction::Control(ControlOp::BraPred {
            pred: reg(RegKind::P, 0, PtxType::Pred),
            target: "L0".to_string(),
            negate: false,
        }));
        kernel.push(PtxInstruction::Control(ControlOp::SetP {
            dst: reg(RegKind::P, 0, PtxType::Pred),
            cmp_op: crate::instr::control::CmpOp::Lt,
            lhs: Operand::Reg(reg(RegKind::R, 0, PtxType::U32)),
            rhs: Operand::ImmU32(10),
            ty: PtxType::U32,
        }));
        // 1 mov + 1 cvt
        kernel.push(PtxInstruction::Mov {
            dst: reg(RegKind::R, 0, PtxType::U32),
            src: Operand::ImmU32(0),
            ty: PtxType::U32,
        });
        kernel.push(PtxInstruction::Cvt {
            dst: reg(RegKind::F, 0, PtxType::F32),
            src: reg(RegKind::R, 0, PtxType::U32),
            dst_ty: PtxType::F32,
            src_ty: PtxType::U32,
        });
        // 1 ret
        kernel.push(PtxInstruction::Control(ControlOp::Ret));
        // Label + Comment — should not count
        kernel.push(PtxInstruction::Label("L0".to_string()));
        kernel.push(PtxInstruction::Comment("test".to_string()));

        let s = kernel.stats();
        // 2 fma + 1 add + 1 ld.global + 1 st.global + 1 ld.shared +
        // 1 st.shared + 1 ld.param + 1 bar.sync + 1 branch + 1 setp +
        // 1 mov + 1 cvt + 1 ret = 14
        assert_eq!(s.total_instructions, 14);
        assert_eq!(s.fma, 2);
        assert_eq!(s.arith_other, 1);
        assert_eq!(s.ld_global, 1);
        assert_eq!(s.st_global, 1);
        assert_eq!(s.ld_shared, 1);
        assert_eq!(s.st_shared, 1);
        assert_eq!(s.bar_sync, 1);
        assert_eq!(s.branches, 1);
        assert_eq!(s.setp, 1);
        assert_eq!(s.mov, 1);
        assert_eq!(s.cvt, 1);
    }

    #[test]
    fn stats_counts_registers_by_kind() {
        let mut kernel = PtxKernel::new("test");
        kernel.set_registers(vec![
            reg(RegKind::R, 0, PtxType::U32),
            reg(RegKind::R, 1, PtxType::S32),
            reg(RegKind::R, 2, PtxType::U32),
            reg(RegKind::Rd, 0, PtxType::U64),
            reg(RegKind::F, 0, PtxType::F32),
            reg(RegKind::F, 1, PtxType::F32),
            reg(RegKind::Fd, 0, PtxType::F64),
            reg(RegKind::P, 0, PtxType::Pred),
            reg(RegKind::P, 1, PtxType::Pred),
        ]);

        let s = kernel.stats();
        assert_eq!(s.registers_r, 3);
        assert_eq!(s.registers_rd, 1);
        assert_eq!(s.registers_f, 2);
        assert_eq!(s.registers_fd, 1);
        assert_eq!(s.registers_p, 2);
    }

    #[test]
    fn stats_counts_tensor_core_and_cp_async() {
        use crate::fragment::{alloc_a_f16, alloc_b_f16, alloc_c};
        use crate::instr::MmaShape;
        use crate::ir::RegisterAllocator;

        let mut alloc = RegisterAllocator::new();
        let mut kernel = PtxKernel::new("tc_stats_test");

        // 2 mma.sync
        for _ in 0..2 {
            kernel.push(PtxInstruction::TensorCore(
                crate::instr::TensorCoreOp::MmaSync {
                    d: alloc_c(&mut alloc),
                    a: alloc_a_f16(&mut alloc),
                    b: alloc_b_f16(&mut alloc),
                    c: alloc_c(&mut alloc),
                    shape: MmaShape::M16N8K16,
                    d_ty: PtxType::F32,
                    a_ty: PtxType::F16,
                    b_ty: PtxType::F16,
                    c_ty: PtxType::F32,
                },
            ));
        }

        // 3 cp.async loads, 1 commit, 1 wait
        let dst_shared = reg(RegKind::R, 0, PtxType::U32);
        let src_global = reg(RegKind::Rd, 0, PtxType::U64);
        for _ in 0..3 {
            kernel.push(PtxInstruction::Memory(MemoryOp::new_cp_async_ca(
                dst_shared, src_global, 16,
            )));
        }
        kernel.push(PtxInstruction::Memory(MemoryOp::CpAsyncCommitGroup));
        kernel.push(PtxInstruction::Memory(MemoryOp::CpAsyncWaitGroup { n: 0 }));

        let s = kernel.stats();
        assert_eq!(s.mma, 2);
        assert_eq!(s.cp_async, 3);
        assert_eq!(s.cp_async_commit, 1);
        assert_eq!(s.cp_async_wait, 1);
        // 2 mma + 3 cp.async + 1 commit + 1 wait = 7 total
        assert_eq!(s.total_instructions, 7);
    }

    #[test]
    fn stats_counts_shared_bytes() {
        let mut kernel = PtxKernel::new("test");
        kernel.add_shared_decl(SharedDecl {
            name: "tile_a".to_string(),
            align: 4,
            size_bytes: 4352, // 64 * 17 * 4
        });
        kernel.add_shared_decl(SharedDecl {
            name: "tile_b".to_string(),
            align: 4,
            size_bytes: 4160, // 16 * 65 * 4
        });

        let s = kernel.stats();
        assert_eq!(s.shared_bytes, 4352 + 4160);
    }
}
