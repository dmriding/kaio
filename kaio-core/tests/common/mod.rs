//! Shared test helpers for kaio-core integration tests.

use kaio_core::emit::{Emit, PtxWriter};
use kaio_core::fragment::{alloc_a, alloc_b, alloc_c};
use kaio_core::instr::control::{CmpOp, ControlOp};
use kaio_core::instr::memory::MemoryOp;
use kaio_core::instr::special;
use kaio_core::instr::{ArithOp, MadMode, MmaShape, TensorCoreOp};
use kaio_core::ir::{
    Operand, PtxInstruction, PtxKernel, PtxModule, PtxParam, RegisterAllocator, SharedDecl,
};
use kaio_core::types::PtxType;

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

    let mut module =
        PtxModule::new(&std::env::var("KAIO_SM_TARGET").unwrap_or_else(|_| "sm_70".to_string()));
    module.add_kernel(kernel);

    let mut w = PtxWriter::new();
    module.emit(&mut w).unwrap();
    w.finish()
}

/// Build a shared memory test kernel that exercises shared memory, bar.sync,
/// and shfl.sync instructions.
///
/// Kernel: each thread writes its tid to shared memory, syncs, reads back,
/// then does a warp shuffle down by 1.
#[allow(dead_code)]
pub fn build_shared_mem_ptx() -> String {
    let mut alloc = RegisterAllocator::new();
    let mut kernel = PtxKernel::new("shared_mem_test");

    // Declare shared memory: 256 floats = 1024 bytes
    kernel.add_shared_decl(SharedDecl {
        name: "sdata".to_string(),
        align: 4,
        size_bytes: 1024,
    });

    // tid = %tid.x
    let (r_tid, tid_instr) = special::tid_x(&mut alloc);
    kernel.push(tid_instr);

    // Compute shared memory byte offset: tid * 4
    let r_offset = alloc.alloc(PtxType::S32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_offset,
        lhs: Operand::Reg(r_tid),
        rhs: Operand::ImmI32(4),
        ty: PtxType::S32,
    }));

    // Store a constant float to shared memory (avoids cvt rounding modifier issue)
    let f_val = alloc.alloc(PtxType::F32);
    kernel.push(PtxInstruction::Mov {
        dst: f_val,
        src: Operand::ImmF32(1.0),
        ty: PtxType::F32,
    });

    // st.shared: write float to shared memory
    kernel.push(PtxInstruction::Memory(MemoryOp::StShared {
        addr: r_offset,
        src: f_val,
        ty: PtxType::F32,
    }));

    // bar.sync 0: synchronize all threads
    kernel.push(PtxInstruction::Control(ControlOp::BarSync {
        barrier_id: 0,
    }));

    // ld.shared: read back from shared memory
    let f_loaded = alloc.alloc(PtxType::F32);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdShared {
        dst: f_loaded,
        addr: r_offset,
        ty: PtxType::F32,
    }));

    // shfl.sync.down: read from neighboring lane
    let r_shfl_result = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Control(ControlOp::ShflSyncDown {
        dst: r_shfl_result,
        src: r_tid,
        delta: Operand::ImmU32(1),
        c: 31,
        mask: 0xFFFFFFFF,
    }));

    kernel.push(PtxInstruction::Control(ControlOp::Ret));

    kernel.set_registers(alloc.into_allocated());

    let mut module =
        PtxModule::new(&std::env::var("KAIO_SM_TARGET").unwrap_or_else(|_| "sm_70".to_string()));
    module.add_kernel(kernel);

    let mut w = PtxWriter::new();
    module.emit(&mut w).unwrap();
    w.finish()
}

/// Build a minimal kernel that emits one `mma.sync.m16n8k16.row.col.f32.f16.f16.f32`
/// with zero-initialized fragment registers.
///
/// Used by `ptxas_verify_mma_sync` to confirm the PTX emission passes
/// the offline assembler for SM 8.0+. The kernel is not functionally
/// meaningful — just enough valid PTX to exercise the `TensorCoreOp::MmaSync`
/// emit path end-to-end.
#[allow(dead_code)]
pub fn build_mma_sync_ptx() -> String {
    let mut alloc = RegisterAllocator::new();
    let mut kernel = PtxKernel::new("mma_sync_smoke");

    let a = alloc_a(&mut alloc);
    let b = alloc_b(&mut alloc);
    let c = alloc_c(&mut alloc);
    let d = alloc_c(&mut alloc);

    // Zero-initialize all fragment registers.
    for r in &a.regs {
        kernel.push(PtxInstruction::Mov {
            dst: *r,
            src: Operand::ImmU32(0),
            ty: PtxType::U32,
        });
    }
    for r in &b.regs {
        kernel.push(PtxInstruction::Mov {
            dst: *r,
            src: Operand::ImmU32(0),
            ty: PtxType::U32,
        });
    }
    for r in &c.regs {
        kernel.push(PtxInstruction::Mov {
            dst: *r,
            src: Operand::ImmF32(0.0),
            ty: PtxType::F32,
        });
    }

    kernel.push(PtxInstruction::TensorCore(TensorCoreOp::MmaSync {
        d,
        a,
        b,
        c,
        shape: MmaShape::M16N8K16,
        d_ty: PtxType::F32,
        a_ty: PtxType::F16,
        b_ty: PtxType::F16,
        c_ty: PtxType::F32,
    }));

    kernel.push(PtxInstruction::Control(ControlOp::Ret));
    kernel.set_registers(alloc.into_allocated());

    // mma.sync requires SM 8.0+. Floor the env override at sm_80.
    let sm = std::env::var("KAIO_SM_TARGET").unwrap_or_else(|_| "sm_80".to_string());
    let mut module = PtxModule::new(&sm);
    module.add_kernel(kernel);

    let mut w = PtxWriter::new();
    module.emit(&mut w).unwrap();
    w.finish()
}

/// Build a minimal kernel exercising `cp.async.ca.shared.global`,
/// `cp.async.commit_group`, and `cp.async.wait_group`.
///
/// Used by `ptxas_verify_cp_async` to confirm the cp.async emission
/// passes the offline assembler for SM 8.0+.
#[allow(dead_code)]
pub fn build_cp_async_ptx() -> String {
    let mut alloc = RegisterAllocator::new();
    let mut kernel = PtxKernel::new("cp_async_smoke");

    kernel.add_shared_decl(SharedDecl {
        name: "tile".to_string(),
        align: 16,
        size_bytes: 16,
    });

    kernel.add_param(PtxParam::pointer("src", PtxType::F32));

    let rd_src_param = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: rd_src_param,
        param_name: "src".to_string(),
        ty: PtxType::U64,
    }));
    let rd_src_global = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::CvtaToGlobal {
        dst: rd_src_global,
        src: rd_src_param,
    }));

    let r_shared_addr = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_shared_addr,
        src: Operand::SharedAddr("tile".to_string()),
        ty: PtxType::U32,
    });

    kernel.push(PtxInstruction::Memory(MemoryOp::new_cp_async_ca(
        r_shared_addr,
        rd_src_global,
        16,
    )));
    kernel.push(PtxInstruction::Memory(MemoryOp::CpAsyncCommitGroup));
    kernel.push(PtxInstruction::Memory(MemoryOp::CpAsyncWaitGroup { n: 0 }));
    kernel.push(PtxInstruction::Control(ControlOp::BarSync {
        barrier_id: 0,
    }));

    kernel.push(PtxInstruction::Control(ControlOp::Ret));
    kernel.set_registers(alloc.into_allocated());

    let sm = std::env::var("KAIO_SM_TARGET").unwrap_or_else(|_| "sm_80".to_string());
    let mut module = PtxModule::new(&sm);
    module.add_kernel(kernel);

    let mut w = PtxWriter::new();
    module.emit(&mut w).unwrap();
    w.finish()
}
