//! # Sprint 7.1 D1b gate — `mma.sync.m16n8k32.s8.s8.s32` correctness.
//!
//! This test is the operand-layout correctness half of the Sprint 7.1
//! fork-decision gate. Companion to the D1a `ptxas_verify_mma_int8`
//! encoding-viability test in `kaio-core/tests/ptxas_verify.rs`.
//!
//! Builds a warp-sized (32-thread) kernel that:
//!   1. Loads A (16×32 i8 row-major) and B (32×8 i8 column-major) from
//!      global memory into mma fragment registers via the D1 fragment
//!      load helpers.
//!   2. Zeros the C accumulator (4 × `.s32` per thread).
//!   3. Emits one `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`.
//!   4. Stores the D fragment back (16×8 i32 row-major) via the D1 store
//!      helper.
//!
//! Host compares D element-wise against a hand-computed `i8 × i8 → i32`
//! reference. Bit-exact — `i8 × i8` fits in `i16` and summing 32 of them
//! fits in `i32` without overflow in any of the test patterns.
//!
//! ## Adversarial test matrix
//!
//! Each pattern is designed to expose a specific class of layout bug:
//!   - **identity-like** (A = i8-identity-ish, B = arbitrary) — catches
//!     fragment-indexing bugs
//!   - **all-ones** — isolates accumulator semantics
//!   - **alternating sign** — catches signed-vs-unsigned interpretation
//!   - **single-hot row** / **single-hot col** — isolates row/col
//!     semantics
//!   - **boundary values** (`i8::MIN`, `i8::MAX`, 0, ±1) at known
//!     positions — catches sign extension
//!   - **ascending byte sequence** — catches byte-within-u32
//!     interleaving bugs that random-ish patterns both miss
//!
//! If any test fails, do not loosen the assertion. Debug thread-data
//! layout in `kaio-core/src/fragment.rs` or operand packing in the
//! `TensorCoreOp::MmaSyncInt8` emit path. Silent corruption in INT8 mma
//! kernels silently produces wrong quantized weights — no loud failure.
//!
//! ## Scope
//!
//! This is **D1b correctness only** — encoding viability was confirmed in
//! D1a. Per-element register layout is the specific failure mode this
//! test is structured to catch; the D1a ptxas_verify test couldn't.

use cudarc::driver::{LaunchConfig, PushKernelArg};
use kaio::prelude::*;
use kaio_core::emit::{Emit, PtxWriter};
use kaio_core::fragment::{
    alloc_c_M16N8K32, load_fragment_a_m16n8k32_global_row, load_fragment_b_m16n8k32_global_col,
    store_fragment_c_m16n8k32_global_row,
};
use kaio_core::instr::TensorCoreOp;
use kaio_core::instr::control::ControlOp;
use kaio_core::instr::memory::MemoryOp;
use kaio_core::instr::special;
use kaio_core::ir::{Operand, PtxInstruction, PtxKernel, PtxModule, PtxParam, RegisterAllocator};
use kaio_core::types::PtxType;

const M: usize = 16;
const N: usize = 8;
const K: usize = 32;

/// Build the gate kernel PTX for `mma.sync.m16n8k32.s8.s8.s32`.
///
/// Parameters (in order): `a_ptr` (i8), `b_ptr` (i8), `d_ptr` (i32).
/// C input is hardcoded to zeros inside the kernel.
fn build_int8_gate_module(sm: &str) -> PtxModule {
    let mut alloc = RegisterAllocator::new();
    let mut kernel = PtxKernel::new("mma_int8_gate");

    kernel.add_param(PtxParam::pointer("a_ptr", PtxType::S8));
    kernel.add_param(PtxParam::pointer("b_ptr", PtxType::S8));
    kernel.add_param(PtxParam::pointer("d_ptr", PtxType::S32));

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

    let (r_tid, tid_instr) = special::tid_x(&mut alloc);
    kernel.push(tid_instr);

    let frag_a = load_fragment_a_m16n8k32_global_row(&mut alloc, &mut kernel, rd_a, r_tid);
    let frag_b = load_fragment_b_m16n8k32_global_col(&mut alloc, &mut kernel, rd_b, r_tid);

    // Zero C.
    let frag_c = alloc_c_M16N8K32(&mut alloc);
    for r in &frag_c.regs {
        kernel.push(PtxInstruction::Mov {
            dst: *r,
            src: Operand::ImmI32(0),
            ty: PtxType::S32,
        });
    }

    let frag_d = alloc_c_M16N8K32(&mut alloc);

    kernel.push(PtxInstruction::TensorCore(TensorCoreOp::MmaSyncInt8 {
        d: frag_d,
        a: frag_a,
        b: frag_b,
        c: frag_c,
    }));

    // Store D: 16×8 i32 row-major, row stride = 8 × 4 = 32 bytes.
    store_fragment_c_m16n8k32_global_row(&mut alloc, &mut kernel, rd_d, r_tid, frag_d, 32);

    kernel.push(PtxInstruction::Control(ControlOp::Ret));
    kernel.set_registers(alloc.into_allocated());

    let mut module = PtxModule::new(sm);
    module.add_kernel(kernel);
    module
}

fn emit_ptx_debug(module: &PtxModule) -> String {
    let mut w = PtxWriter::new();
    module.emit(&mut w).unwrap();
    w.finish()
}

/// CPU reference: `D = A * B` where A is 16×32 i8 row-major, B is 32×8 i8
/// column-major, D is 16×8 i32 row-major. Accumulates in i32.
fn cpu_reference(a: &[i8], b: &[i8]) -> Vec<i32> {
    assert_eq!(a.len(), M * K);
    assert_eq!(b.len(), K * N);
    let mut d = vec![0i32; M * N];
    for i in 0..M {
        for j in 0..N {
            let mut acc: i32 = 0;
            for k in 0..K {
                let a_ik = a[i * K + k] as i32;
                let b_kj = b[j * K + k] as i32; // column-major: col j starts at j*K
                acc += a_ik * b_kj;
            }
            d[i * N + j] = acc;
        }
    }
    d
}

/// Run the gate kernel with host-provided A and B, return the 16×8 i32
/// output from GPU. Panics on launch error.
fn run_int8_gate(a_host: &[i8], b_host: &[i8]) -> Vec<i32> {
    assert_eq!(a_host.len(), M * K);
    assert_eq!(b_host.len(), K * N);

    let device = KaioDevice::new(0).expect("GPU required for this test");
    let info = device.info().expect("device info");
    let (major, minor) = info.compute_capability;
    assert!(
        major >= 8,
        "mma.sync.m16n8k32.s8 requires SM 8.0+ (got sm_{major}{minor})",
    );
    let sm = format!("sm_{major}{minor}");

    let ptx_module = build_int8_gate_module(&sm);
    let module = device.load_module(&ptx_module).unwrap_or_else(|e| {
        eprintln!(
            "=== PTX that failed to load ===\n{}",
            emit_ptx_debug(&ptx_module)
        );
        panic!("load_module failed: {e}");
    });
    let func = module.function("mma_int8_gate").expect("function handle");

    let buf_a = device.alloc_from(a_host).expect("alloc A");
    let buf_b = device.alloc_from(b_host).expect("alloc B");
    let mut buf_d = device.alloc_zeros::<i32>(M * N).expect("alloc D");

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
        eprintln!("=== PTX ===\n{}", emit_ptx_debug(&ptx_module));
        panic!("mma_int8_gate launch failed: {e}");
    });

    buf_d.to_host(&device).expect("D host roundtrip")
}

/// Compare GPU output against CPU reference, bit-exact. Report the first
/// 10 mismatches with (i, j, expected, got) for diagnosis.
fn assert_bit_exact(pattern: &str, a_host: &[i8], b_host: &[i8]) {
    let d_gpu = run_int8_gate(a_host, b_host);
    let d_cpu = cpu_reference(a_host, b_host);

    let mut mismatches: Vec<(usize, usize, i32, i32)> = Vec::new();
    for i in 0..M {
        for j in 0..N {
            let idx = i * N + j;
            if d_gpu[idx] != d_cpu[idx] {
                mismatches.push((i, j, d_cpu[idx], d_gpu[idx]));
            }
        }
    }

    if !mismatches.is_empty() {
        let mut msg = format!(
            "D1b INT8 mma gate FAILED for pattern '{pattern}': {} of {} elements mismatched.\n\
             First 10 mismatches (i, j, expected, got):\n",
            mismatches.len(),
            M * N,
        );
        for (i, j, e, g) in mismatches.iter().take(10) {
            msg.push_str(&format!("  [{i:2},{j}] expected {e:12}, got {g:12}\n"));
        }
        msg.push_str(
            "\nThis is the Sprint 7.1 D1b gate. If it fails:\n\
             1. Check fragment thread-data mapping in kaio-core/src/fragment.rs\n\
                (m16n8k32 section) — byte-within-u32 order or row/col offsets wrong.\n\
             2. Check TensorCoreOp::MmaSyncInt8 operand ordering in tensor_core.rs.\n\
             3. If layout bugs, 48h pivot timebox before switching to DEQUANT-F16 fallback.\n",
        );
        panic!("{msg}");
    }
}

// ============================================================================
// Adversarial test patterns
// ============================================================================

#[test]
#[ignore] // requires NVIDIA GPU with SM 8.0+
fn int8_mma_ascending_byte_pattern() {
    // Ascending-byte pattern: catches byte-within-u32 interleaving bugs. If mma
    // interprets byte-0 of the first A register as K=8 instead of K=0,
    // the outputs will shift predictably. Ascending values make the shift
    // obvious at mismatch time.
    let mut a = vec![0i8; M * K];
    for i in 0..M {
        for k in 0..K {
            a[i * K + k] = (((i * K + k) % 127) as i8).wrapping_sub(0);
        }
    }
    let mut b = vec![0i8; K * N];
    for j in 0..N {
        for k in 0..K {
            b[j * K + k] = (k % 7 + 1) as i8; // 1..7 cycling
        }
    }
    assert_bit_exact("ascending-byte", &a, &b);
}

#[test]
#[ignore]
fn int8_mma_all_ones() {
    // Isolates accumulator semantics. Every D[i,j] must equal K = 32.
    let a = vec![1i8; M * K];
    let b = vec![1i8; K * N];
    assert_bit_exact("all-ones", &a, &b);
}

#[test]
#[ignore]
fn int8_mma_alternating_sign() {
    // Catches signed-vs-unsigned interpretation.
    // A[i,k] = +1 if (i+k) even, -1 if odd
    // B[k,j] = +1 if (k+j) even, -1 if odd
    // D[i,j] = sum_k (-1)^(i+k+k+j) = sum_k (-1)^(i+j) = K * (-1)^(i+j)
    let mut a = vec![0i8; M * K];
    for i in 0..M {
        for k in 0..K {
            a[i * K + k] = if (i + k) % 2 == 0 { 1 } else { -1 };
        }
    }
    let mut b = vec![0i8; K * N];
    for j in 0..N {
        for k in 0..K {
            b[j * K + k] = if (k + j) % 2 == 0 { 1 } else { -1 };
        }
    }
    assert_bit_exact("alternating-sign", &a, &b);
}

#[test]
#[ignore]
fn int8_mma_single_hot_row() {
    // A has one nonzero per row, B is all-ones. Exposes fragment A-row
    // semantics: if D row i doesn't equal the nonzero-column's row-sum,
    // row-indexing is wrong.
    let mut a = vec![0i8; M * K];
    for i in 0..M {
        let hot_col = i % K;
        a[i * K + hot_col] = ((i as i8) & 0x3F) + 1; // 1..64
    }
    let b = vec![1i8; K * N];
    assert_bit_exact("single-hot-row", &a, &b);
}

#[test]
#[ignore]
fn int8_mma_single_hot_col() {
    // B has one nonzero per column, A is all-ones. Exposes B-column
    // semantics and fragment B-reg ordering.
    let a = vec![1i8; M * K];
    let mut b = vec![0i8; K * N];
    for j in 0..N {
        let hot_row = (j * 4) % K;
        b[j * K + hot_row] = ((j as i8) & 0x3F) + 1;
    }
    assert_bit_exact("single-hot-col", &a, &b);
}

#[test]
#[ignore]
fn int8_mma_boundary_values() {
    // Exercises sign extension at the numeric boundaries. If signed-shr
    // somewhere collapsed to unsigned, i8::MIN would silently flip sign.
    let mut a = vec![0i8; M * K];
    for i in 0..M {
        for k in 0..K {
            // Cycle through {-128, -1, 0, 1, 127} at known positions.
            a[i * K + k] = match (i + k) % 5 {
                0 => i8::MIN, // -128
                1 => -1,
                2 => 0,
                3 => 1,
                _ => i8::MAX, // 127
            };
        }
    }
    // B = small symmetric integers to keep the accumulator in-range.
    let mut b = vec![0i8; K * N];
    for j in 0..N {
        for k in 0..K {
            b[j * K + k] = (((j as i32 + k as i32) % 5) - 2) as i8; // -2..2
        }
    }
    assert_bit_exact("boundary-values", &a, &b);
}

#[test]
#[ignore]
fn int8_mma_comprehensive() {
    // The f16 gate's strong-signal pattern adapted for i8: A contains
    // position-encoding data so any wrong fragment index produces a wrong
    // output magnitude, not just a permuted value.
    //
    // A[i,k] = (i * K + k - 256) clamped to i8 range, so values span the
    // full signed i8 space and encode (i, k) uniquely per position.
    let mut a = vec![0i8; M * K];
    for i in 0..M {
        for k in 0..K {
            let v = (i * K + k) as i32 - 256;
            a[i * K + k] = v.clamp(i8::MIN as i32, i8::MAX as i32) as i8;
        }
    }
    // B[k,j] = small column-varying values to make D magnitudes distinctive.
    let mut b = vec![0i8; K * N];
    for j in 0..N {
        for k in 0..K {
            b[j * K + k] = ((j as i8) + 1) * (if k % 2 == 0 { 1 } else { -1 });
        }
    }
    assert_bit_exact("comprehensive", &a, &b);
}
