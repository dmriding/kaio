//! Shared test helpers for kaio-core integration tests.

use kaio_core::emit::{Emit, PtxWriter};
use kaio_core::fragment::{
    alloc_a, alloc_b, alloc_c, load_fragment_a_m16n8k16_shared_row,
    load_fragment_b_m16n8k16_shared_col,
};
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
///
/// Sprint 6.10 D3: takes `sm` as an explicit argument. Callers pass the
/// SM target directly instead of mutating `KAIO_SM_TARGET`. `sm` must
/// be Ampere-or-better (sm_80+) — enforced by the caller, not here.
#[allow(dead_code)]
pub fn build_mma_sync_ptx(sm: &str) -> String {
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

    let mut module = PtxModule::new(sm);
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
///
/// Sprint 6.10 D3: takes `sm` as an explicit argument. Callers pass the
/// SM target directly; `sm` must be Ampere-or-better (sm_80+).
#[allow(dead_code)]
pub fn build_cp_async_ptx(sm: &str) -> String {
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

    let mut module = PtxModule::new(sm);
    module.add_kernel(kernel);

    let mut w = PtxWriter::new();
    module.emit(&mut w).unwrap();
    w.finish()
}

/// Build a minimal kernel exercising the **shared-source** fragment
/// load helpers: declares shared `tile_a` and `tile_b` allocations,
/// gets a `%tid.x`, calls `load_fragment_a_m16n8k16_shared_row` and
/// `load_fragment_b_m16n8k16_shared_col` (with zero C, fresh D), emits
/// one `mma.sync`, and returns.
///
/// Used by `ptxas_verify_mma_sync_shared` to confirm the shared-source
/// emission is structurally valid PTX for SM 8.0+. The kernel does no
/// initial tile population — ptxas does not care that the shared
/// memory is uninitialized; it only verifies the instruction syntax.
///
/// Sprint 6.10 D3: takes `sm` as an explicit argument. Callers pass the
/// SM target directly; `sm` must be Ampere-or-better (sm_80+).
#[allow(dead_code)]
pub fn build_mma_sync_shared_ptx(sm: &str) -> String {
    let mut alloc = RegisterAllocator::new();
    let mut kernel = PtxKernel::new("mma_sync_shared_smoke");

    // Declare shared tiles matching Sprint 6.3 sizes.
    kernel.add_shared_decl(SharedDecl {
        name: "tile_a".to_string(),
        align: 4,
        size_bytes: 512, // 16 × 16 fp16
    });
    kernel.add_shared_decl(SharedDecl {
        name: "tile_b".to_string(),
        align: 4,
        size_bytes: 256, // 16 × 8 fp16 column-major
    });

    // %tid.x
    let (r_tid, tid_instr) = special::tid_x(&mut alloc);
    kernel.push(tid_instr);

    // Shared base-offset registers for tile_a and tile_b.
    let r_tile_a = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_tile_a,
        src: Operand::SharedAddr("tile_a".to_string()),
        ty: PtxType::U32,
    });
    let r_tile_b = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_tile_b,
        src: Operand::SharedAddr("tile_b".to_string()),
        ty: PtxType::U32,
    });

    // Fragment loads from shared.
    let frag_a =
        load_fragment_a_m16n8k16_shared_row(&mut alloc, &mut kernel, r_tile_a, r_tid, 32, None);
    let frag_b =
        load_fragment_b_m16n8k16_shared_col(&mut alloc, &mut kernel, r_tile_b, r_tid, 32, None);

    // Zero C fragment.
    let frag_c = alloc_c(&mut alloc);
    for r in &frag_c.regs {
        kernel.push(PtxInstruction::Mov {
            dst: *r,
            src: Operand::ImmF32(0.0),
            ty: PtxType::F32,
        });
    }

    // Fresh D fragment.
    let frag_d = alloc_c(&mut alloc);

    // One mma.sync.
    kernel.push(PtxInstruction::TensorCore(TensorCoreOp::MmaSync {
        d: frag_d,
        a: frag_a,
        b: frag_b,
        c: frag_c,
        shape: MmaShape::M16N8K16,
        d_ty: PtxType::F32,
        a_ty: PtxType::F16,
        b_ty: PtxType::F16,
        c_ty: PtxType::F32,
    }));

    kernel.push(PtxInstruction::Control(ControlOp::Ret));
    kernel.set_registers(alloc.into_allocated());

    let mut module = PtxModule::new(sm);
    module.add_kernel(kernel);

    let mut w = PtxWriter::new();
    module.emit(&mut w).unwrap();
    w.finish()
}

/// Build a minimal kernel that exercises `MemoryOp::LdGlobalB128`:
/// loads a pointer parameter, converts to global-space, and issues
/// one `ld.global.v4.b32` into 4 freshly-allocated b32 registers.
///
/// Used by `ptxas_verify_ld_global_b128` (Sprint 6.7b Gate A) to
/// confirm the emitted vectorized-load instruction is accepted by
/// ptxas. The kernel doesn't do anything with the loaded values —
/// ptxas only verifies PTX syntax / ISA legality, not runtime
/// behavior. LDG.128 is not Ampere-gated (vectorized loads have
/// been part of PTX since sm_30-era); uses the base SM target.
#[allow(dead_code)]
pub fn build_ld_global_b128_ptx() -> String {
    let mut alloc = RegisterAllocator::new();
    let mut kernel = PtxKernel::new("ld_global_b128_smoke");

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

    let r0 = alloc.alloc(PtxType::U32);
    let r1 = alloc.alloc(PtxType::U32);
    let r2 = alloc.alloc(PtxType::U32);
    let r3 = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Memory(MemoryOp::new_ld_global_b128(
        [r0, r1, r2, r3],
        rd_src_global,
    )));

    kernel.push(PtxInstruction::Control(ControlOp::Ret));
    kernel.set_registers(alloc.into_allocated());

    let sm = std::env::var("KAIO_SM_TARGET").unwrap_or_else(|_| "sm_70".to_string());
    let mut module = PtxModule::new(&sm);
    module.add_kernel(kernel);

    let mut w = PtxWriter::new();
    module.emit(&mut w).unwrap();
    w.finish()
}

/// Build a minimal kernel exercising every Sprint 7.0 bitwise ArithOp variant
/// (And / Or / Xor / Shl / Shr signed / Shr unsigned / Not) so ptxas_verify
/// catches any malformed emit before the macro-level lowering lands in D2.
///
/// Takes `sm` as a parameter (D3-style clean contract) — no internal
/// `KAIO_SM_TARGET` mutation.
#[allow(dead_code)]
pub fn build_bitops_ptx(sm: &str) -> String {
    let mut alloc = RegisterAllocator::new();
    let mut kernel = PtxKernel::new("bitops_smoke");

    kernel.add_param(PtxParam::scalar("a", PtxType::U32));
    kernel.add_param(PtxParam::scalar("b", PtxType::U32));

    let r_a = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: r_a,
        param_name: "a".to_string(),
        ty: PtxType::U32,
    }));
    let r_b = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: r_b,
        param_name: "b".to_string(),
        ty: PtxType::U32,
    }));

    // and.b32
    let r_and = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::And {
        dst: r_and,
        lhs: Operand::Reg(r_a),
        rhs: Operand::Reg(r_b),
        ty: PtxType::U32,
    }));
    // or.b32
    let r_or = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Or {
        dst: r_or,
        lhs: Operand::Reg(r_a),
        rhs: Operand::Reg(r_b),
        ty: PtxType::U32,
    }));
    // xor.b32
    let r_xor = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Xor {
        dst: r_xor,
        lhs: Operand::Reg(r_a),
        rhs: Operand::Reg(r_b),
        ty: PtxType::U32,
    }));
    // shl.b32 with immediate shift
    let r_shl = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Shl {
        dst: r_shl,
        lhs: Operand::Reg(r_a),
        rhs: Operand::ImmU32(4),
        ty: PtxType::U32,
    }));
    // shr.u32 (logical right shift)
    let r_shr_u = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Shr {
        dst: r_shr_u,
        lhs: Operand::Reg(r_a),
        rhs: Operand::ImmU32(1),
        ty: PtxType::U32,
    }));
    // shr.s32 (arithmetic right shift) — signed type, same register space.
    // This is the AD2 canary: if ty: S32 silently becomes shr.u32, quant
    // dequant on signed INT8 will zero-extend negative packed values.
    let r_shr_s = alloc.alloc(PtxType::S32);
    kernel.push(PtxInstruction::Arith(ArithOp::Shr {
        dst: r_shr_s,
        lhs: Operand::Reg(r_a),
        rhs: Operand::ImmU32(1),
        ty: PtxType::S32,
    }));
    // not.b32
    let r_not = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Not {
        dst: r_not,
        src: Operand::Reg(r_a),
        ty: PtxType::U32,
    }));

    kernel.push(PtxInstruction::Control(ControlOp::Ret));
    kernel.set_registers(alloc.into_allocated());

    let mut module = PtxModule::new(sm);
    module.add_kernel(kernel);

    let mut w = PtxWriter::new();
    module.emit(&mut w).unwrap();
    w.finish()
}
