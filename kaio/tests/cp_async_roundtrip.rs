//! # Sprint 6.2 — cp.async primitive smoke test.
//!
//! Exercises `cp.async.ca.shared.global` + `cp.async.commit_group` +
//! `cp.async.wait_group 0` in a single-warp kernel:
//!
//! 1. A 4-element fp32 input buffer is copied global → shared via one
//!    16-byte `cp.async.ca` (lane 0 only).
//! 2. `commit_group` + `wait_group 0` make the copy visible.
//! 3. A `bar.sync` syncs the warp.
//! 4. Each of the 4 lanes (0..4) reads its element from shared memory
//!    and stores it back to a global output buffer.
//!
//! ## What this test proves
//!
//! - `cp.async.ca.shared.global` emission and operand ordering are correct
//! - The `[shared], [global], size` byte-count encoding works for size=16
//! - `commit_group` / `wait_group 0` sequencing is accepted and functional
//! - global → shared → global movement works at least once
//!
//! ## What this test does NOT prove
//!
//! - Pipeline overlap correctness
//! - Multi-stage commit-group numbering (`wait_group 1`, etc.)
//! - Double-buffer safety
//! - Performance — this is a correctness smoke test
//!
//! Those belong to Sprint 6.4. This test is a **primitive smoke test**,
//! not a pipeline validation.

use cudarc::driver::{LaunchConfig, PushKernelArg};
use kaio::prelude::*;
use kaio_core::emit::{Emit, PtxWriter};
use kaio_core::instr::control::{CmpOp, ControlOp};
use kaio_core::instr::memory::MemoryOp;
use kaio_core::instr::special;
use kaio_core::instr::{ArithOp, MadMode};
use kaio_core::ir::{
    Operand, PtxInstruction, PtxKernel, PtxModule, PtxParam, RegisterAllocator, SharedDecl,
};
use kaio_core::types::PtxType;

fn build_cp_async_roundtrip_module() -> PtxModule {
    let mut alloc = RegisterAllocator::new();
    let mut kernel = PtxKernel::new("cp_async_roundtrip");

    kernel.add_shared_decl(SharedDecl {
        name: "tile".to_string(),
        align: 16,
        size_bytes: 16, // 4 × fp32
    });

    kernel.add_param(PtxParam::pointer("src", PtxType::F32));
    kernel.add_param(PtxParam::pointer("dst", PtxType::F32));

    // Load & convert both pointers to global-space.
    let rd_src_param = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: rd_src_param,
        param_name: "src".to_string(),
        ty: PtxType::U64,
    }));
    let rd_dst_param = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: rd_dst_param,
        param_name: "dst".to_string(),
        ty: PtxType::U64,
    }));
    let rd_src = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::CvtaToGlobal {
        dst: rd_src,
        src: rd_src_param,
    }));
    let rd_dst = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::CvtaToGlobal {
        dst: rd_dst,
        src: rd_dst_param,
    }));

    // %r_tid
    let (r_tid, tid_instr) = special::tid_x(&mut alloc);
    kernel.push(tid_instr);

    // Shared base address.
    let r_shared_base = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_shared_base,
        src: Operand::SharedAddr("tile".to_string()),
        ty: PtxType::U32,
    });

    // Predicate: issue the cp.async only from lane 0.
    let p_is_lane0 = alloc.alloc(PtxType::Pred);
    kernel.push(PtxInstruction::Control(ControlOp::SetP {
        dst: p_is_lane0,
        cmp_op: CmpOp::Eq,
        lhs: Operand::Reg(r_tid),
        rhs: Operand::ImmU32(0),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Control(ControlOp::BraPred {
        pred: p_is_lane0,
        target: "NOT_LANE0".to_string(),
        negate: true,
    }));

    // Lane 0: issue one 16-byte cp.async from src[0..4] into tile[0..16].
    kernel.push(PtxInstruction::Memory(MemoryOp::new_cp_async_ca(
        r_shared_base,
        rd_src,
        16,
    )));
    kernel.push(PtxInstruction::Memory(MemoryOp::CpAsyncCommitGroup));
    kernel.push(PtxInstruction::Memory(MemoryOp::CpAsyncWaitGroup { n: 0 }));

    kernel.push(PtxInstruction::Label("NOT_LANE0".to_string()));
    // All lanes sync so the copied data is visible to every thread.
    kernel.push(PtxInstruction::Control(ControlOp::BarSync {
        barrier_id: 0,
    }));

    // Predicate: only lanes 0..4 do the write-back.
    let p_in_range = alloc.alloc(PtxType::Pred);
    kernel.push(PtxInstruction::Control(ControlOp::SetP {
        dst: p_in_range,
        cmp_op: CmpOp::Lt,
        lhs: Operand::Reg(r_tid),
        rhs: Operand::ImmU32(4),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Control(ControlOp::BraPred {
        pred: p_in_range,
        target: "EXIT".to_string(),
        negate: true,
    }));

    // shared_addr = tile + tid * 4
    let r_shared_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mad {
        dst: r_shared_off,
        a: Operand::Reg(r_tid),
        b: Operand::ImmU32(4),
        c: Operand::Reg(r_shared_base),
        ty: PtxType::U32,
        mode: MadMode::Lo,
    }));
    let f_val = alloc.alloc(PtxType::F32);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdShared {
        dst: f_val,
        addr: r_shared_off,
        ty: PtxType::F32,
    }));

    // dst_addr = dst + tid * 4  (u64 addr math)
    let rd_tid_bytes = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
        dst: rd_tid_bytes,
        lhs: Operand::Reg(r_tid),
        rhs: Operand::ImmU32(4),
        src_ty: PtxType::U32,
    }));
    let rd_dst_addr = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: rd_dst_addr,
        lhs: Operand::Reg(rd_dst),
        rhs: Operand::Reg(rd_tid_bytes),
        ty: PtxType::U64,
    }));
    kernel.push(PtxInstruction::Memory(MemoryOp::StGlobal {
        addr: rd_dst_addr,
        src: f_val,
        ty: PtxType::F32,
    }));

    kernel.push(PtxInstruction::Label("EXIT".to_string()));
    kernel.push(PtxInstruction::Control(ControlOp::Ret));
    kernel.set_registers(alloc.into_allocated());

    // cp.async requires SM 8.0+.
    let requested = std::env::var("KAIO_SM_TARGET").unwrap_or_else(|_| "sm_80".to_string());
    let sm = match requested
        .strip_prefix("sm_")
        .and_then(|s| s.parse::<u32>().ok())
    {
        Some(v) if v >= 80 => requested,
        _ => "sm_80".to_string(),
    };
    let mut module = PtxModule::new(&sm);
    module.add_kernel(kernel);
    module
}

/// Emit a `PtxModule` to PTX text for debug printing on failure paths.
fn emit_ptx_debug(module: &PtxModule) -> String {
    let mut w = PtxWriter::new();
    module.emit(&mut w).unwrap();
    w.finish()
}

#[test]
#[ignore] // requires NVIDIA GPU with SM 8.0+
fn cp_async_ca_roundtrip_4_floats() {
    let ptx_module = build_cp_async_roundtrip_module();

    let device = KaioDevice::new(0).expect("GPU required for this test");
    let info = device.info().expect("device info");
    let (major, _minor) = info.compute_capability;
    assert!(major >= 8, "cp.async requires SM 8.0+ (got sm_{major})");

    let module = device.load_module(&ptx_module).unwrap_or_else(|e| {
        eprintln!("=== PTX that failed to load ===\n{}", emit_ptx_debug(&ptx_module));
        panic!("load_module failed: {e}");
    });
    let func = module
        .function("cp_async_roundtrip")
        .expect("function handle lookup");

    let src = [1.5f32, -2.25, 3.75, 42.0];
    let buf_src = device.alloc_from(&src).expect("alloc src");
    let mut buf_dst = device.alloc_zeros::<f32>(4).expect("alloc dst");

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        device
            .stream()
            .launch_builder(func.inner())
            .arg(buf_src.inner())
            .arg(buf_dst.inner_mut())
            .launch(cfg)
    }
    .unwrap_or_else(|e| {
        eprintln!("=== PTX ===\n{}", emit_ptx_debug(&ptx_module));
        panic!("cp_async_roundtrip launch failed: {e}");
    });

    let out = buf_dst.to_host(&device).expect("dst roundtrip");
    assert_eq!(out, src, "cp.async roundtrip mismatch");
}
