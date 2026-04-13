//! # Sprint 6.2 gate — single `mma.sync.m16n8k16` correctness.
//!
//! This is the gatekeeper GPU test for all of Phase 6. It builds a
//! warp-sized (32-thread) kernel that loads fixed known-value A, B, C
//! fragments from global memory, emits exactly one
//! `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`, and stores the
//! output D fragment back. The host compares D element-wise against a
//! hand-computed expected value, bit-exactly.
//!
//! ## Test inputs
//!
//! - **A** — 16×16 row-major fp16, `a[i,k] = i * 16 + k` (integers
//!   0..255, all exactly representable in fp16)
//! - **B** — 16×8 column-major fp16, `b[k,j] = j + 1` (column-varying
//!   1..8 so column-index errors in the B fragment mapping become
//!   visible — all-ones B would have hidden such bugs)
//! - **C** — 16×8 fp32 zeros
//! - **Expected D** — `d[i,j] = (j+1) * sum_k(a[i,k] * 1) = (j+1) * (256i + 120)`
//!
//! The expected values are integer-valued and exactly representable in
//! fp32 (max = 8 * 3960 = 31680), so the assertion is bit-exact.
//!
//! ## Failure diagnostics
//!
//! A wrong result can indicate:
//! - Fragment thread-data mapping (most likely — address math in
//!   `kaio-core/src/fragment.rs`)
//! - Operand ordering in `TensorCoreOp::MmaSync` emission
//! - Register packing (A/B should use `.b32`, not `.f16` — see
//!   [`kaio-core/src/fragment.rs`] docs)
//! - Output store layout (`store_fragment_c_m16n8k16_global_row`)
//!
//! If the test fails, do not "adjust expected values" or "loosen the
//! tolerance." Debug the root cause — silent corruption in tensor-core
//! kernels is the single most common Phase 6 failure mode.

use cudarc::driver::{LaunchConfig, PushKernelArg};
use half::f16;
use kaio::prelude::*;
use kaio_core::emit::{Emit, PtxWriter};
use kaio_core::fragment::{
    alloc_c, load_fragment_a_m16n8k16_global_row, load_fragment_b_m16n8k16_global_col,
    store_fragment_c_m16n8k16_global_row,
};
use kaio_core::instr::control::ControlOp;
use kaio_core::instr::memory::MemoryOp;
use kaio_core::instr::special;
use kaio_core::instr::{MmaShape, TensorCoreOp};
use kaio_core::ir::{Operand, PtxInstruction, PtxKernel, PtxModule, PtxParam, RegisterAllocator};
use kaio_core::types::PtxType;

/// Build the gate kernel PTX.
///
/// Parameters (in order): `a_ptr` (f16), `b_ptr` (f16), `d_ptr` (f32).
/// The C input is hardcoded to zeros inside the kernel so the host
/// doesn't need to ship a buffer of zeros.
fn build_mma_gate_ptx() -> String {
    let mut alloc = RegisterAllocator::new();
    let mut kernel = PtxKernel::new("mma_sync_gate");

    kernel.add_param(PtxParam::pointer("a_ptr", PtxType::F16));
    kernel.add_param(PtxParam::pointer("b_ptr", PtxType::F16));
    kernel.add_param(PtxParam::pointer("d_ptr", PtxType::F32));

    // Load param pointers (generic-space).
    let rd_a_param = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: rd_a_param,
        param_name: "a_ptr".to_string(),
        ty: PtxType::U64,
    }));
    let rd_b_param = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: rd_b_param,
        param_name: "b_ptr".to_string(),
        ty: PtxType::U64,
    }));
    let rd_d_param = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: rd_d_param,
        param_name: "d_ptr".to_string(),
        ty: PtxType::U64,
    }));

    // Convert to global-space addresses.
    let rd_a = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::CvtaToGlobal {
        dst: rd_a,
        src: rd_a_param,
    }));
    let rd_b = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::CvtaToGlobal {
        dst: rd_b,
        src: rd_b_param,
    }));
    let rd_d = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::CvtaToGlobal {
        dst: rd_d,
        src: rd_d_param,
    }));

    // Get %tid.x.
    let (r_tid, tid_instr) = special::tid_x(&mut alloc);
    kernel.push(tid_instr);

    // Load A and B fragments.
    let frag_a = load_fragment_a_m16n8k16_global_row(&mut alloc, &mut kernel, rd_a, r_tid);
    let frag_b = load_fragment_b_m16n8k16_global_col(&mut alloc, &mut kernel, rd_b, r_tid);

    // C fragment = zeros. Allocate and initialize with mov.f32 0.0.
    let frag_c = alloc_c(&mut alloc);
    for r in &frag_c.regs {
        kernel.push(PtxInstruction::Mov {
            dst: *r,
            src: Operand::ImmF32(0.0),
            ty: PtxType::F32,
        });
    }

    // D output fragment — fresh registers.
    let frag_d = alloc_c(&mut alloc);

    // mma.sync.
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

    // Store D back to global memory. 32-byte row stride = native 16×8 D
    // matrix (8 fp32 per row × 4 bytes).
    store_fragment_c_m16n8k16_global_row(&mut alloc, &mut kernel, rd_d, r_tid, frag_d, 32);

    kernel.push(PtxInstruction::Control(ControlOp::Ret));
    kernel.set_registers(alloc.into_allocated());

    // m16n8k16.f16 requires Ampere+. Floor at sm_80.
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

    let mut w = PtxWriter::new();
    module.emit(&mut w).unwrap();
    w.finish()
}

/// Build host-side A: 16×16 row-major, `a[i,k] = i * 16 + k`.
fn build_a_host() -> Vec<f16> {
    let mut a = Vec::with_capacity(256);
    for i in 0..16u32 {
        for k in 0..16u32 {
            a.push(f16::from_f32((i * 16 + k) as f32));
        }
    }
    a
}

/// Build host-side B: 16×8 column-major, `b[k,j] = j + 1`.
///
/// Column-major layout means column `j` occupies rows 0..15 contiguously
/// in memory starting at offset `j * 16`. The flat vec is:
/// `[1; 16]` concatenated with `[2; 16]`, ..., through `[8; 16]`.
fn build_b_host() -> Vec<f16> {
    let mut b = Vec::with_capacity(128);
    for j in 0..8u32 {
        for _k in 0..16u32 {
            b.push(f16::from_f32((j + 1) as f32));
        }
    }
    b
}

#[test]
#[ignore] // requires NVIDIA GPU with SM 8.0+
fn mma_sync_m16n8k16_fragment_gate() {
    let ptx = build_mma_gate_ptx();

    let device = KaioDevice::new(0).expect("GPU required for this test");

    // Verify the GPU is Ampere+ before we try to launch.
    let info = device.info().expect("device info");
    let (major, minor) = info.compute_capability;
    assert!(
        major >= 8,
        "mma.sync.m16n8k16 requires SM 8.0+ (got sm_{major}{minor})",
    );

    let module = device.load_ptx(&ptx).unwrap_or_else(|e| {
        eprintln!("=== PTX that failed to load ===\n{ptx}");
        panic!("load_ptx failed: {e}");
    });
    let func = module
        .function("mma_sync_gate")
        .expect("function handle lookup");

    let a_host = build_a_host();
    let b_host = build_b_host();

    let buf_a = device.alloc_from(&a_host).expect("alloc A");
    let buf_b = device.alloc_from(&b_host).expect("alloc B");
    let mut buf_d = device
        .alloc_zeros::<f32>(16 * 8)
        .expect("alloc D (16x8 fp32)");

    // Launch one warp (32 threads, 1 block).
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        device
            .stream()
            .launch_builder(func.inner())
            .arg(buf_a.inner())
            .arg(buf_b.inner())
            .arg(buf_d.inner_mut())
            .launch(cfg)
    }
    .unwrap_or_else(|e| {
        eprintln!("=== PTX ===\n{ptx}");
        panic!("mma_sync_gate launch failed: {e}");
    });

    let d_host = buf_d.to_host(&device).expect("D host roundtrip");
    assert_eq!(d_host.len(), 128);

    // Expected: d[i,j] = (j+1) * (256i + 120). Row-major, 8 columns.
    let mut mismatches: Vec<(u32, u32, f32, f32)> = Vec::new();
    for i in 0..16u32 {
        for j in 0..8u32 {
            let expected = ((j + 1) as f32) * ((256 * i + 120) as f32);
            let got = d_host[(i * 8 + j) as usize];
            if got.to_bits() != expected.to_bits() {
                mismatches.push((i, j, expected, got));
            }
        }
    }

    if !mismatches.is_empty() {
        let mut msg = format!(
            "mma.sync.m16n8k16 gate FAILED: {} of 128 elements mismatched (bit-exact).\n\
             First 10 mismatches (i, j, expected, got):\n",
            mismatches.len()
        );
        for (i, j, e, g) in mismatches.iter().take(10) {
            msg.push_str(&format!("  [{i:2},{j}] expected {e:8.1}, got {g:8.1}\n"));
        }
        msg.push_str(
            "\nThis is the Phase 6 gatekeeper — do NOT loosen the assertion.\n\
             Debug root cause in kaio-core/src/fragment.rs (thread-data layout)\n\
             or the TensorCoreOp::MmaSync emitter.\n",
        );
        panic!("{msg}");
    }
}
