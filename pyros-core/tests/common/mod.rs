//! Shared test helpers for pyros-core integration tests.

use pyros_core::emit::{Emit, PtxWriter};
use pyros_core::instr::control::{CmpOp, ControlOp};
use pyros_core::instr::memory::MemoryOp;
use pyros_core::instr::special;
use pyros_core::instr::{ArithOp, MadMode};
use pyros_core::ir::{Operand, PtxInstruction, PtxKernel, PtxModule, PtxParam, RegisterAllocator};
use pyros_core::types::PtxType;

/// Build the complete vector_add kernel IR and emit it to a PTX string.
///
/// This is the canonical Phase 1 test kernel — used by both the emission
/// test (`vector_add_emit.rs`) and the ptxas verification test
/// (`ptxas_verify.rs`).
pub fn build_vector_add_ptx() -> String {
    let mut alloc = RegisterAllocator::new();
    let mut kernel = PtxKernel::new("vector_add");

    kernel.add_param(PtxParam::pointer("a_ptr", PtxType::F32));
    kernel.add_param(PtxParam::pointer("b_ptr", PtxType::F32));
    kernel.add_param(PtxParam::pointer("c_ptr", PtxType::F32));
    kernel.add_param(PtxParam::scalar("n", PtxType::U32));

    // Load parameters
    let rd_a = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: rd_a,
        param_name: "a_ptr".to_string(),
        ty: PtxType::U64,
    }));
    let rd_b = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: rd_b,
        param_name: "b_ptr".to_string(),
        ty: PtxType::U64,
    }));
    let rd_c = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: rd_c,
        param_name: "c_ptr".to_string(),
        ty: PtxType::U64,
    }));
    let r_n = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: r_n,
        param_name: "n".to_string(),
        ty: PtxType::U32,
    }));

    // Global thread index
    let (r_ctaid, ctaid_instr) = special::ctaid_x(&mut alloc);
    kernel.push(ctaid_instr);
    let (r_ntid, ntid_instr) = special::ntid_x(&mut alloc);
    kernel.push(ntid_instr);
    let (r_tid, tid_instr) = special::tid_x(&mut alloc);
    kernel.push(tid_instr);

    let r_idx = alloc.alloc(PtxType::S32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mad {
        dst: r_idx,
        a: Operand::Reg(r_ctaid),
        b: Operand::Reg(r_ntid),
        c: Operand::Reg(r_tid),
        ty: PtxType::S32,
        mode: MadMode::Lo,
    }));

    // Bounds check
    let p_oob = alloc.alloc(PtxType::Pred);
    kernel.push(PtxInstruction::Control(ControlOp::SetP {
        dst: p_oob,
        cmp_op: CmpOp::Ge,
        lhs: Operand::Reg(r_idx),
        rhs: Operand::Reg(r_n),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Control(ControlOp::BraPred {
        pred: p_oob,
        target: "EXIT".to_string(),
        negate: false,
    }));

    // a[idx]
    let rd_a_global = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::CvtaToGlobal {
        dst: rd_a_global,
        src: rd_a,
    }));
    let rd_offset = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
        dst: rd_offset,
        lhs: Operand::Reg(r_idx),
        rhs: Operand::ImmU32(4),
        src_ty: PtxType::U32,
    }));
    let rd_a_addr = alloc.alloc(PtxType::S64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: rd_a_addr,
        lhs: Operand::Reg(rd_a_global),
        rhs: Operand::Reg(rd_offset),
        ty: PtxType::S64,
    }));
    let f_a = alloc.alloc(PtxType::F32);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdGlobal {
        dst: f_a,
        addr: rd_a_addr,
        ty: PtxType::F32,
    }));

    // b[idx]
    let rd_b_global = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::CvtaToGlobal {
        dst: rd_b_global,
        src: rd_b,
    }));
    let rd_b_addr = alloc.alloc(PtxType::S64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: rd_b_addr,
        lhs: Operand::Reg(rd_b_global),
        rhs: Operand::Reg(rd_offset),
        ty: PtxType::S64,
    }));
    let f_b = alloc.alloc(PtxType::F32);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdGlobal {
        dst: f_b,
        addr: rd_b_addr,
        ty: PtxType::F32,
    }));

    // c[idx] = a[idx] + b[idx]
    let f_c = alloc.alloc(PtxType::F32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: f_c,
        lhs: Operand::Reg(f_a),
        rhs: Operand::Reg(f_b),
        ty: PtxType::F32,
    }));

    // Store
    let rd_c_global = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::CvtaToGlobal {
        dst: rd_c_global,
        src: rd_c,
    }));
    let rd_c_addr = alloc.alloc(PtxType::S64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: rd_c_addr,
        lhs: Operand::Reg(rd_c_global),
        rhs: Operand::Reg(rd_offset),
        ty: PtxType::S64,
    }));
    kernel.push(PtxInstruction::Memory(MemoryOp::StGlobal {
        addr: rd_c_addr,
        src: f_c,
        ty: PtxType::F32,
    }));

    // EXIT + return
    kernel.push(PtxInstruction::Label("EXIT".to_string()));
    kernel.push(PtxInstruction::Control(ControlOp::Ret));

    kernel.set_registers(alloc.into_allocated());

    let mut module = PtxModule::new("sm_89");
    module.add_kernel(kernel);

    let mut w = PtxWriter::new();
    module.emit(&mut w).unwrap();
    w.finish()
}
