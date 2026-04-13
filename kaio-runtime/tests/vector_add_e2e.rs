//! End-to-end test: construct vector_add kernel via KAIO IR, emit PTX,
//! load into the CUDA driver, launch on a real GPU, and verify results.
//!
//! This is the Phase 1 success gate. If this passes, the core thesis of
//! KAIO is proven: Rust IR → PTX text → GPU execution.

use kaio_core::emit::{Emit, PtxWriter};
use kaio_core::instr::control::{CmpOp, ControlOp};
use kaio_core::instr::memory::MemoryOp;
use kaio_core::instr::special;
use kaio_core::instr::{ArithOp, MadMode};
use kaio_core::ir::{Operand, PtxInstruction, PtxKernel, PtxModule, PtxParam, RegisterAllocator};
use kaio_core::types::PtxType;

use cudarc::driver::PushKernelArg;
use kaio_runtime::{KaioDevice, LaunchConfig};

/// Build the vector_add kernel IR and return it as a `PtxModule`.
///
/// This is the same IR construction as `kaio-core/tests/vector_add_emit.rs`,
/// extracted here so both the emission test and the E2E test share the same
/// kernel definition.
///
/// Sprint 6.10 D1a: migrated from `build_vector_add_ptx() -> String` to
/// return the `PtxModule` directly, matching the `load_module(&PtxModule)`
/// path used by internal TC kernels and the `#[gpu_kernel]` macro.
fn build_vector_add_module() -> PtxModule {
    let mut alloc = RegisterAllocator::new();
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

    // Global thread index: idx = ctaid.x * ntid.x + tid.x
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

    // a[idx]: convert pointer + compute address + load
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

    // EXIT + return
    kernel.push(PtxInstruction::Label("EXIT".to_string()));
    kernel.push(PtxInstruction::Control(ControlOp::Ret));

    kernel.set_registers(alloc.into_allocated());

    let sm = std::env::var("KAIO_SM_TARGET").unwrap_or_else(|_| "sm_70".to_string());
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
#[ignore] // requires NVIDIA GPU
fn vector_add_small() {
    let ptx_module = build_vector_add_module();

    // Load into driver
    let device = KaioDevice::new(0).expect("GPU required");
    let module = device.load_module(&ptx_module).unwrap_or_else(|e| {
        eprintln!("=== PTX that failed to load ===\n{}", emit_ptx_debug(&ptx_module));
        panic!("load_module failed: {e}");
    });
    let func = module.function("vector_add").unwrap_or_else(|e| {
        panic!("function('vector_add') failed: {e}");
    });

    // Allocate buffers
    let a_host = [1.0f32, 2.0, 3.0];
    let b_host = [4.0f32, 5.0, 6.0];
    let n: u32 = 3;

    let buf_a = device.alloc_from(&a_host).expect("alloc a");
    let buf_b = device.alloc_from(&b_host).expect("alloc b");
    let mut buf_out = device.alloc_zeros::<f32>(n as usize).expect("alloc out");

    // Launch kernel
    let cfg = LaunchConfig::for_num_elems(n);
    unsafe {
        device
            .stream()
            .launch_builder(func.inner())
            .arg(buf_a.inner())
            .arg(buf_b.inner())
            .arg(buf_out.inner_mut())
            .arg(&n)
            .launch(cfg)
    }
    .unwrap_or_else(|e| {
        eprintln!("=== PTX ===\n{}", emit_ptx_debug(&ptx_module));
        panic!("kernel launch failed: {e}");
    });

    // Verify
    let result = buf_out.to_host(&device).expect("to_host");
    assert_eq!(
        result,
        vec![5.0f32, 7.0, 9.0],
        "vector_add produced wrong results"
    );
}

#[test]
#[ignore] // requires NVIDIA GPU
fn vector_add_large() {
    let ptx_module = build_vector_add_module();

    let device = KaioDevice::new(0).expect("GPU required");
    let module = device.load_module(&ptx_module).unwrap_or_else(|e| {
        eprintln!("=== PTX that failed to load ===\n{}", emit_ptx_debug(&ptx_module));
        panic!("load_module failed: {e}");
    });
    let func = module.function("vector_add").unwrap_or_else(|e| {
        panic!("function('vector_add') failed: {e}");
    });

    // 10,000 elements — exercises multi-block launch
    let n: u32 = 10_000;
    let a_host: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
    let expected: Vec<f32> = a_host.iter().zip(&b_host).map(|(a, b)| a + b).collect();

    let buf_a = device.alloc_from(&a_host).expect("alloc a");
    let buf_b = device.alloc_from(&b_host).expect("alloc b");
    let mut buf_out = device.alloc_zeros::<f32>(n as usize).expect("alloc out");

    let cfg = LaunchConfig::for_num_elems(n);
    unsafe {
        device
            .stream()
            .launch_builder(func.inner())
            .arg(buf_a.inner())
            .arg(buf_b.inner())
            .arg(buf_out.inner_mut())
            .arg(&n)
            .launch(cfg)
    }
    .unwrap_or_else(|e| {
        eprintln!("=== PTX ===\n{}", emit_ptx_debug(&ptx_module));
        panic!("kernel launch failed: {e}");
    });

    let result = buf_out.to_host(&device).expect("to_host");
    assert_eq!(
        result, expected,
        "vector_add (10k elements) produced wrong results"
    );
}
