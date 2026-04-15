//! D2.5 register-pressure skeleton for the tri-output QKV projection.
//!
//! Sprint 7.3 D2.5. This module is **test-only**. It exists to answer
//! one question before D3 authors the full kernel body: *will three
//! live `FragmentC` banks plus realistic fragment-A/B staging plus
//! scale and output-pointer state fit inside the 64-register
//! occupancy tier on sm_89?*
//!
//! The skeleton emits a minimal kernel that allocates every live
//! register at peak pressure for the Serial-fusion design:
//!
//! - **3 × 16 = 48 f32 FragmentC regs** per lane (all three projection
//!   accumulator banks, all 2×2 mma sub-tiles per warp quadrant, live
//!   simultaneously — these must persist across the K-loop in the real
//!   kernel, so ptxas cannot reuse them).
//! - **Fragment-A staging** — one warp-quadrant's worth of X-tile
//!   fragment registers (reused across projections in the real Design-S
//!   kernel, but allocated at peak).
//! - **Fragment-B staging** — one warp-quadrant's worth of per-projection
//!   dequant output registers (INT8 path: `cvt.rn.f16.s8` + `MovPack`;
//!   INT4 path: nibble-extract + `MovPack` — same envelope).
//! - **3 scalar f32 scale regs** (INT8 scales — INT4 folds scales into B
//!   so this is a slight over-count for INT4, but both variants share the
//!   skeleton).
//! - **3 × u64 output base pointers + row stride** (global addressing
//!   state).
//! - **tid / ctaid bookkeeping**.
//!
//! Real mma.sync emission is skipped — the register contention is
//! dominated by FragmentC liveness, not by the mma instruction itself.
//! Fragment-A/B registers are kept live with trivial `mov.b32 imm`
//! initializations, which mirrors what the real dequant chain produces
//! without requiring shared-memory loaders.
//!
//! # Pass criterion
//!
//! `ptxas --verbose --gpu-name sm_89`:
//!
//! - `Used N registers`, **N ≤ 64**
//! - `0 bytes spill stores, 0 bytes spill loads`
//!
//! If `N > 64`, the plan's Rollback #1 activates — drop
//! `MMAS_PER_WARP_N` from 2 to 1 (halving the output tile to 64×16) to
//! reclaim registers before D3 authors the full kernel body.
//!
//! The archived baseline (regs/thread, shared bytes, instruction count,
//! `.const` bytes) becomes the D3 regression detector.

#![cfg(test)]

use kaio_core::instr::control::ControlOp;
use kaio_core::instr::memory::MemoryOp;
use kaio_core::instr::{ArithOp, special};
use kaio_core::ir::{Operand, PtxInstruction, PtxKernel, Register, RegisterAllocator};
use kaio_core::types::PtxType;

use crate::store_out::emit_store_fragment_c_f32_to_f16_packed;

// Mirror the D1 tile constants — both qkv_project_int{4,8}_kernel modules export
// these with identical values after the W8A16 unification. We copy them rather
// than reach across modules to keep this skeleton independent of where D3 ends
// up living.
const MMAS_PER_WARP_M: u32 = 2;
const MMAS_PER_WARP_N: u32 = 2;
const SUBTILES_PER_BANK: u32 = MMAS_PER_WARP_M * MMAS_PER_WARP_N; // 4
const REGS_PER_FRAGMENT_C: u32 = 4; // m16n8k16 fragment-C: 4 f32 regs per lane
const PROJECTIONS: u32 = 3; // Q, K, V

// Fragment-A for m16n8k16 f16: 4 b32 regs per lane (holds 8 f16 values).
const REGS_PER_FRAGMENT_A: u32 = 4;
// Fragment-B for m16n8k16 f16: 2 b32 regs per lane (holds 4 f16 values).
const REGS_PER_FRAGMENT_B: u32 = 2;

/// Build the D2.5 tri-output register-pressure skeleton kernel PTX.
///
/// Returns a complete PTX module string targeting `sm`, ready to hand
/// to `ptxas --verbose` for the register / spill / shared / const
/// stats.
pub(crate) fn build_qkv_tri_output_skeleton_ptx(sm: &str) -> String {
    use kaio_core::emit::{Emit, PtxWriter};
    use kaio_core::ir::{PtxModule, PtxParam};

    let mut alloc = RegisterAllocator::new();
    let mut kernel = PtxKernel::new("qkv_tri_output_skeleton");

    // Kernel signature: three output pointers, three scalar scales, one row
    // stride. No weight / activation pointers — the skeleton does not do any
    // real loads; fragment-A/B are synthesized from immediates.
    kernel.add_param(PtxParam::pointer("q_out", PtxType::F16));
    kernel.add_param(PtxParam::pointer("k_out", PtxType::F16));
    kernel.add_param(PtxParam::pointer("v_out", PtxType::F16));
    kernel.add_param(PtxParam::scalar("scale_q", PtxType::F32));
    kernel.add_param(PtxParam::scalar("scale_k", PtxType::F32));
    kernel.add_param(PtxParam::scalar("scale_v", PtxType::F32));
    kernel.add_param(PtxParam::scalar("stride", PtxType::U32));

    // Load + cvta the three output pointers.
    let q_base = load_and_cvta_pointer(&mut alloc, &mut kernel, "q_out");
    let k_base = load_and_cvta_pointer(&mut alloc, &mut kernel, "k_out");
    let v_base = load_and_cvta_pointer(&mut alloc, &mut kernel, "v_out");

    // Load the three scalar scales.
    let scale_q = load_scalar_f32(&mut alloc, &mut kernel, "scale_q");
    let scale_k = load_scalar_f32(&mut alloc, &mut kernel, "scale_k");
    let scale_v = load_scalar_f32(&mut alloc, &mut kernel, "scale_v");

    // Load the row stride scalar.
    let row_stride = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: row_stride,
        param_name: "stride".to_string(),
        ty: PtxType::U32,
    }));

    // Read tid.x for lane id in store-out.
    let (tid_lane, tid_instr) = special::tid_x(&mut alloc);
    kernel.push(tid_instr);

    // --- Allocate + initialize all 48 FragmentC f32 regs per lane. ---
    //
    // Layout: PROJECTIONS × SUBTILES_PER_BANK × REGS_PER_FRAGMENT_C
    //       = 3 × 4 × 4 = 48 f32 regs per lane.
    //
    // Each register is `mov.f32`-initialized with a distinct immediate so
    // ptxas cannot fold them into a common constant (which would erase the
    // pressure we are trying to measure).
    let mut frag_c: Vec<[Register; 4]> =
        Vec::with_capacity((PROJECTIONS * SUBTILES_PER_BANK) as usize);
    for slot in 0..(PROJECTIONS * SUBTILES_PER_BANK) {
        let mut regs = [Register::placeholder(); 4];
        for (i, reg_slot) in regs
            .iter_mut()
            .enumerate()
            .take(REGS_PER_FRAGMENT_C as usize)
        {
            let r = alloc.alloc(PtxType::F32);
            kernel.push(PtxInstruction::Mov {
                dst: r,
                // Distinct immediates: 0.001..0.192 across the 48 regs. Keeps
                // every register distinct to the allocator without risking
                // INF / NaN that ptxas might fold.
                src: Operand::ImmF32(0.001 * (slot * REGS_PER_FRAGMENT_C + i as u32 + 1) as f32),
                ty: PtxType::F32,
            });
            *reg_slot = r;
        }
        frag_c.push(regs);
    }

    // --- Allocate + initialize FragmentA (shared across projections). ---
    //
    // 4 b32 regs per lane (each packing two f16). We initialize them via
    // `mov.b32 imm` — at the PTX level this is just a 32-bit load, which
    // approximates the downstream state of a `MovPack` of two `cvt.rn.f16.f32`
    // results without needing the full dequant chain.
    let frag_a: [Register; REGS_PER_FRAGMENT_A as usize] = core::array::from_fn(|i| {
        alloc_and_init_b32_imm(&mut alloc, &mut kernel, 0xDEADBEA0 + i as u32)
    });

    // --- Allocate + initialize FragmentB (per-projection staging). ---
    //
    // In the real Serial-fusion kernel this set of registers is reused
    // across Q/K/V; here we allocate one set at peak to simulate the
    // worst-case live envelope during any single projection's mma sweep.
    let frag_b: [Register; REGS_PER_FRAGMENT_B as usize] = core::array::from_fn(|i| {
        alloc_and_init_b32_imm(&mut alloc, &mut kernel, 0xFEEDCAFE + i as u32)
    });

    // --- Side-effect fragments A and B so ptxas cannot prune them. ---
    //
    // We XOR them into a single b32 "sink" register and drop the result
    // into a dummy store slot before the main store epilogue. This keeps
    // frag_a and frag_b registers live until after the frag_c banks are
    // fully populated — which is the peak-pressure point we are measuring.
    let sink = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: sink,
        src: Operand::ImmU32(0),
        ty: PtxType::U32,
    });
    let mut running = sink;
    for r in frag_a.iter().chain(frag_b.iter()) {
        let next = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Xor {
            dst: next,
            lhs: Operand::Reg(running),
            rhs: Operand::Reg(*r),
            ty: PtxType::U32,
        }));
        running = next;
    }
    // Stash the XOR into `q_base` at offset 0 so the whole chain is
    // observable. This one write has no correctness role — it just keeps
    // fragment A+B alive past FragmentC initialization.
    kernel.push(PtxInstruction::Memory(MemoryOp::StGlobal {
        addr: q_base,
        src: running,
        ty: PtxType::U32,
    }));

    // --- Tri-output store epilogue. ---
    //
    // For each of the 4 sub-tiles, emit the three store-out calls (one per
    // projection). This mirrors the exact access pattern the real kernel
    // will use in D3 after the K-loop terminates. All 48 FragmentC regs
    // are read; the helper handles cvt + MovPack + st.global.u32.
    let bases = [q_base, k_base, v_base];
    let scales = [scale_q, scale_k, scale_v];
    for sub in 0..SUBTILES_PER_BANK as usize {
        for proj in 0..PROJECTIONS as usize {
            let frag_idx = proj * SUBTILES_PER_BANK as usize + sub;
            emit_store_fragment_c_f32_to_f16_packed(
                &mut alloc,
                &mut kernel,
                &frag_c[frag_idx],
                bases[proj],
                row_stride,
                tid_lane,
                Some(scales[proj]),
            );
        }
    }

    kernel.push(PtxInstruction::Control(ControlOp::Ret));
    kernel.set_registers(alloc.into_allocated());

    let mut module = PtxModule::new(sm);
    module.add_kernel(kernel);
    let mut w = PtxWriter::new();
    module.emit(&mut w).unwrap();
    w.finish()
}

fn load_and_cvta_pointer(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    param_name: &str,
) -> Register {
    let rd_param = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: rd_param,
        param_name: param_name.to_string(),
        ty: PtxType::U64,
    }));
    let rd_global = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::CvtaToGlobal {
        dst: rd_global,
        src: rd_param,
    }));
    rd_global
}

fn load_scalar_f32(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    param_name: &str,
) -> Register {
    let r = alloc.alloc(PtxType::F32);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: r,
        param_name: param_name.to_string(),
        ty: PtxType::F32,
    }));
    r
}

fn alloc_and_init_b32_imm(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    imm: u32,
) -> Register {
    let r = alloc.alloc_packed_half2();
    kernel.push(PtxInstruction::Mov {
        dst: r,
        src: Operand::ImmU32(imm),
        ty: PtxType::U32,
    });
    r
}

// --- Register-allocator placeholder helper ------------------------------
//
// `Register` does not expose a public default / null value — the allocator
// is the single source of allocations. For our small fixed-size arrays
// of 4 f32 regs per FragmentC sub-tile we need a placeholder that's
// overwritten immediately during the initializer loop. Use any already-
// allocated register; the tiny shim below just borrows one.

trait RegisterPlaceholder {
    fn placeholder() -> Self;
}

impl RegisterPlaceholder for Register {
    fn placeholder() -> Self {
        // SAFETY: this value is *only* constructed as a fill before
        // being overwritten inside the initializer loop below. It is
        // never inserted into a PtxKernel or used as an operand.
        //
        // We construct an obviously-invalid RegKind/index pair; if this
        // ever leaks into emitted PTX a grep for `%f_PLACEHOLDER` would
        // make it visible. In practice the `iter_mut().take(N)` loop
        // replaces every entry before the array is consumed.
        Register {
            kind: kaio_core::types::RegKind::F,
            index: u32::MAX,
            ptx_type: PtxType::F32,
        }
    }
}

// ----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    /// Runs ptxas --verbose over the skeleton and parses the D3 regression
    /// baseline from stderr. `#[ignore]` so host-only CI stays green.
    ///
    /// Pass criterion: `Used N registers, N ≤ 64, 0 spills`.
    /// Archive: regs/thread, shared bytes, instruction count, `.const` bytes.
    #[test]
    #[ignore]
    fn ptxas_verify_qkv_tri_output_skeleton() {
        if Command::new("ptxas").arg("--version").output().is_err() {
            eprintln!("NOTE: ptxas not found in PATH — skipping D2.5 checkpoint");
            return;
        }

        let sm = std::env::var("KAIO_SM_TARGET").unwrap_or_else(|_| "sm_89".to_string());
        let ptx = build_qkv_tri_output_skeleton_ptx(&sm);
        let tmp = std::env::temp_dir().join("kaio_qkv_tri_output_skeleton.ptx");
        std::fs::write(&tmp, &ptx).expect("failed to write temp PTX");

        let output = Command::new("ptxas")
            .args(["--gpu-name", &sm, "--verbose"])
            .arg(tmp.to_str().unwrap())
            .output()
            .expect("failed to run ptxas");
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let _ = std::fs::remove_file(&tmp);

        eprintln!("--- ptxas --verbose stdout ({sm}) ---\n{stdout}");
        eprintln!("--- ptxas --verbose stderr ({sm}) ---\n{stderr}");

        assert!(
            output.status.success(),
            "ptxas FAILED on qkv tri-output skeleton ({sm}):\n{stderr}\n\
             === PTX (first 8000 chars) ===\n{}",
            &ptx[..ptx.len().min(8000)]
        );

        // Parse baseline numbers. ptxas --verbose writes its `info` lines to
        // stdout on Windows + CUDA 12.8 (other platforms may route to stderr).
        // Concatenate both streams so this test is portable across platforms.
        // On Ampere+ ptxas reports kernel params in `cmem[0]` — that's the
        // param-space constant bank, uniform-read across threads, no bank
        // pathology (which is what Gemini's D2.5 note wanted us to confirm).
        let combined = format!("{stdout}\n{stderr}");
        let regs = extract_metric(&combined, "Used ", " registers");
        let smem = extract_metric(&combined, ", used 0 barriers, ", " bytes smem");
        let cmem0 = extract_metric(&combined, ", used 0 barriers, ", " bytes cmem[0]");
        let spill_stores = extract_metric(&combined, "bytes stack frame, ", " bytes spill stores");
        let spill_loads = extract_metric(&combined, "spill stores, ", " bytes spill loads");

        eprintln!(
            "=== D2.5 ARCHIVED BASELINE ({sm}) ===\n\
             regs/thread     : {regs:?}\n\
             shared bytes    : {smem:?}\n\
             cmem[0] bytes   : {cmem0:?} (params, uniform read — no bank conflict)\n\
             spill stores    : {spill_stores:?}\n\
             spill loads     : {spill_loads:?}\n\
             PTX instr count : {}\n\
             user const bytes: {} (grep .const decls in emitted PTX)",
            count_instructions_in_ptx(&ptx),
            count_const_bytes_in_ptx(&ptx),
        );

        // Gate the pass/fail: register count must fit the 64-reg occupancy
        // tier and there must be zero spills.
        match regs {
            Some(n) => assert!(
                n <= 64,
                "D2.5 skeleton reports {n} registers > 64; plan Rollback #1 \
                 (drop MMAS_PER_WARP_N 2→1) should activate before D3."
            ),
            None => panic!(
                "Could not parse register count from ptxas output — update extract_metric \
                 pattern?\nstdout was:\n{stdout}\nstderr was:\n{stderr}"
            ),
        }
        assert_eq!(
            spill_stores,
            Some(0),
            "D2.5 skeleton has spill stores — register budget blown before D3."
        );
        assert_eq!(
            spill_loads,
            Some(0),
            "D2.5 skeleton has spill loads — register budget blown before D3."
        );

        eprintln!("D2.5 SKELETON GATE PASSED ({sm}): tri-output fits in 64-reg tier, zero spills.");
    }

    /// Parse a `<prefix><number><suffix>` pattern out of a multi-line string.
    fn extract_metric(haystack: &str, prefix: &str, suffix: &str) -> Option<u32> {
        for line in haystack.lines() {
            if let Some(after_prefix) = line.split(prefix).nth(1)
                && let Some(number_str) = after_prefix.split(suffix).next()
                && let Ok(n) = number_str.trim().parse::<u32>()
            {
                return Some(n);
            }
        }
        None
    }

    /// Rough instruction count: lines ending in `;` and not starting with `//`
    /// or `.` (directive). Not exact but good enough as a D3 regression baseline.
    fn count_instructions_in_ptx(ptx: &str) -> usize {
        ptx.lines()
            .map(str::trim)
            .filter(|l| {
                !l.is_empty()
                    && !l.starts_with("//")
                    && !l.starts_with('.')
                    && !l.starts_with('{')
                    && !l.starts_with('}')
                    && l.ends_with(';')
            })
            .count()
    }

    /// Count `.const`-space bytes declared by the emitted module. Looks for
    /// `.const .align ...` declarations at the module level. Expected to be
    /// 0 for the skeleton (all params land in `.param` space).
    fn count_const_bytes_in_ptx(ptx: &str) -> usize {
        // Naïve: sum byte-sizes out of `.const .align N .b8 <name>[M]` patterns.
        // If Gemini's `.const` concern materializes we'll see non-zero here.
        let mut total = 0usize;
        for line in ptx.lines().map(str::trim) {
            if !line.starts_with(".const") {
                continue;
            }
            if let Some(start) = line.rfind('[')
                && let Some(end) = line.rfind(']')
                && let Ok(n) = line[start + 1..end].parse::<usize>()
            {
                total += n;
            }
        }
        total
    }

    /// Compile-check: the skeleton module builds without panic for both
    /// default SMs. Host-side smoke, no ptxas dependency.
    #[test]
    fn skeleton_module_builds() {
        let ptx_80 = build_qkv_tri_output_skeleton_ptx("sm_80");
        let ptx_89 = build_qkv_tri_output_skeleton_ptx("sm_89");
        assert!(ptx_80.contains(".entry qkv_tri_output_skeleton"));
        assert!(ptx_89.contains(".entry qkv_tri_output_skeleton"));
        // Sanity: we should have emitted 12 packed-b32 stores for the
        // 4 sub-tiles × 3 projections × 2 stores-per-sub-tile = 24 stores,
        // plus the 1 "keep fragment A/B alive" sink store = 25 total.
        assert_eq!(
            ptx_89.matches("st.global.u32").count(),
            25,
            "expected 24 tri-store epilogue stores + 1 sink store = 25; \
             got\n{ptx_89}"
        );
    }
}
