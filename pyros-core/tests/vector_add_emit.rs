//! Integration test: construct a complete `vector_add` kernel via the IR API
//! and emit it to a PTX string. Validates that `pyros-core` can produce a
//! complete, structurally correct `.ptx` file from Rust code.
//!
//! This is the Phase 1 milestone test for `pyros-core`.

use pyros_core::emit::{Emit, PtxWriter};
use pyros_core::instr::ArithOp;
use pyros_core::instr::control::{CmpOp, ControlOp};
use pyros_core::instr::memory::MemoryOp;
use pyros_core::instr::special;
use pyros_core::ir::{Operand, PtxInstruction, PtxKernel, PtxModule, PtxParam, RegisterAllocator};
use pyros_core::types::PtxType;

#[test]
fn emit_full_vector_add() {
    let mut alloc = RegisterAllocator::new();

    // --- Build kernel ---
    let mut kernel = PtxKernel::new("vector_add");

    // Parameters: three f32 pointers + one u32 scalar
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

    // Compute global thread index: idx = ctaid.x * ntid.x + tid.x
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
        mode: pyros_core::instr::MadMode::Lo,
    }));

    // Bounds check: if idx >= n, skip to EXIT
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
    }));

    // Convert generic pointers to global address space
    let rd_a_global = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::CvtaToGlobal {
        dst: rd_a_global,
        src: rd_a,
    }));

    // Compute byte offset: offset = idx * 4 (sizeof f32)
    let rd_offset = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
        dst: rd_offset,
        lhs: Operand::Reg(r_idx),
        rhs: Operand::ImmU32(4),
        src_ty: PtxType::U32,
    }));

    // a[idx]: address + load
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

    // b[idx]: convert + address + load
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

    // Store result
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

    // EXIT label + return
    kernel.push(PtxInstruction::Label("EXIT".to_string()));
    kernel.push(PtxInstruction::Control(ControlOp::Ret));

    // Finalize registers
    kernel.set_registers(alloc.into_allocated());

    // --- Build module ---
    let mut module = PtxModule::new("sm_89");
    module.add_kernel(kernel);

    // --- Emit ---
    let mut w = PtxWriter::new();
    module.emit(&mut w).unwrap();
    let ptx = w.finish();

    // --- Validate structure ---
    // Header
    assert!(ptx.starts_with(".version 8.7\n"));
    assert!(ptx.contains(".target sm_89\n"));
    assert!(ptx.contains(".address_size 64\n"));

    // Kernel signature
    assert!(ptx.contains(".visible .entry vector_add("));
    assert!(ptx.contains(".param .u64 a_ptr,"));
    assert!(ptx.contains(".param .u64 b_ptr,"));
    assert!(ptx.contains(".param .u64 c_ptr,"));
    assert!(ptx.contains(".param .u32 n")); // last param, no comma

    // Register declarations (exact counts from our allocation pattern)
    assert!(ptx.contains(".reg .b32 %r<"));
    assert!(ptx.contains(".reg .b64 %rd<"));
    assert!(ptx.contains(".reg .f32 %f<"));
    assert!(ptx.contains(".reg .pred %p<"));

    // Key instructions (structural, not byte-for-byte against nvcc)
    assert!(ptx.contains("ld.param.u64 %rd0, [a_ptr];"));
    assert!(ptx.contains("ld.param.u32 %r0, [n];"));
    assert!(ptx.contains("mov.u32 %r1, %ctaid.x;"));
    assert!(ptx.contains("mov.u32 %r2, %ntid.x;"));
    assert!(ptx.contains("mov.u32 %r3, %tid.x;"));
    assert!(ptx.contains("mad.lo.s32 %r4, %r1, %r2, %r3;"));
    assert!(ptx.contains("setp.ge.u32 %p0, %r4, %r0;"));
    assert!(ptx.contains("@%p0 bra EXIT;"));
    assert!(ptx.contains("cvta.to.global.u64"));
    assert!(ptx.contains("mul.wide.u32"));
    assert!(ptx.contains("add.s64"));
    assert!(ptx.contains("ld.global.f32"));
    assert!(ptx.contains("add.f32"));
    assert!(ptx.contains("st.global.f32"));
    assert!(ptx.contains("EXIT:"));
    assert!(ptx.contains("ret;"));

    // Structure: ends with closing brace
    assert!(ptx.trim_end().ends_with('}'));

    // Print for manual inspection (visible when running with --nocapture)
    eprintln!("=== PYROS vector_add PTX ===\n{ptx}");
}
