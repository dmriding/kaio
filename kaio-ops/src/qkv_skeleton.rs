//! D1 register-pressure skeleton for the **Sprint 7.3.5 S+½P** tri-output QKV projection.
//!
//! Sprint 7.3.5 D1. This module is **test-only**. It exists to answer one
//! question before D2 authors the full kernel body: *will the S+½P peak
//! live set — three `FragmentC` banks, register-hoisted scales, two
//! ping-pong W-slot base pointers, in-flight cooperative-load address
//! registers, and frag-A staging — fit inside the 64-register occupancy
//! tier on sm_80 and sm_89?*
//!
//! # What makes this a S+½P skeleton vs the 7.3 D2.5 version
//!
//! The 7.3 Design-S skeleton (commit `94c5558`) modelled the tri-output
//! peak pressure for a single-W-slot serial-fusion kernel: 40 regs sm_89,
//! 32 regs sm_80, 0 spills. That's the baseline.
//!
//! Sprint 7.3.5 D1 models the S+½P peak, which adds:
//!
//! - **Two `tile_w` shared slots** separated by 64 B bank-phase padding
//!   (Design invariant #4 — non-multiple-of-128 stride avoids cross-warp
//!   SMEM bank-port contention during overlap; folklore-correct across
//!   Ampere/Ada).
//! - **`frag_A` register-hoist** across the K-loop iteration (Design
//!   invariant #1 — `tile_x` may be overwritten by `X_next` within the
//!   same K-tile because `frag_A` is in registers).
//! - **Scales register-hoist** from a runtime `k_tile_group` counter —
//!   INT4 only — replacing the 7.3 `tile_scales` shared slot (Design
//!   invariant #2 — eliminates the 7.3 bug-3 class categorically; cost
//!   ~6 regs).
//! - **Ping-pong slot pointers** (`cur_w_addr`, `next_w_addr`) updated
//!   against a runtime loop index (`ctaid.x`-derived, so ptxas cannot
//!   constant-fold — Design invariant #3).
//! - **In-flight cooperative-load address registers** — the overlapped
//!   `W_{P+1}` store-to-shared's address math must stay live through
//!   the current projection's mma epoch.
//! - **≥ 2 unrolled K-loop iterations** (3 modelled here: steady →
//!   steady → final) so `frag_C` accumulators carry values across
//!   back-edges and ptxas liveness analysis counts them at the peak
//!   rather than eliding them as dead-after-use (Opus R1b-2 / Opus
//!   R3b-3 — without cross-iteration modelling the skeleton reports
//!   62 regs and the real kernel spills at 68).
//!
//! The skeleton still skips real mma.sync emission — register contention
//! is dominated by frag_C liveness and the surrounding staging, not by
//! the mma instruction itself. Frag-A/B dequant chains are approximated
//! with `mov.b32 imm` + `ld.shared` + `fma.f32` sequences that have the
//! right register envelope and right back-edge data-flow.
//!
//! # Scale-hoist mechanism note (Gemini R3-2)
//!
//! The production kernel's INT4 scale-hoist path must use `ld.global.nc`
//! (read-only cache / TEX path) or a uniform-broadcast pattern to avoid
//! L1 thrash and GPR ballooning. This skeleton uses plain `ld.global`
//! because `kaio-core` does not yet expose `.nc`, and the *register
//! count* is what the skeleton is measuring — caching-behavior
//! differences don't change ptxas's register pressure report. A note in
//! D2 will move production emission to `.nc` or the uniform datapath.
//!
//! # Pass criteria
//!
//! `ptxas --verbose --gpu-name sm_{80,89}` on the emitted module:
//!
//! - `Used N registers`, **N ≤ 64** on both sm_80 and sm_89.
//! - `0 bytes spill stores, 0 bytes spill loads` on both architectures.
//!
//! **Escape-hatch-before-Rollback-#2 (Gemini R3-4, Opus R3b-2):** if
//! `N > 64`, first inspect the emitted SASS via `cuobjdump --dump-sass`
//! and check whether the 6 invariant base pointers (Q/K/V output, 3
//! scale bases) are already being sourced from uniform-broadcast
//! constant-bank loads (`ULDC` / `UPAK`) or from GPRs:
//!
//! - **Uniform loads already** → case (a), the escape hatch has no
//!   handle to grab; investigate other pressure sources. If that fails,
//!   Rollback #2 activates (INT8-only S+½P, INT4 stays Design S).
//! - **GPR reads** → case (b), re-emit with explicit `.const` / `.param`
//!   placement hints for the base pointers and re-run ptxas. If ≥ 2 regs
//!   reclaimed under 64 / 0 spills, proceed to D2.
//!
//! **Artifact retention (Codex R2-5):** archive both sm_80 and sm_89
//! `ptxas -v` outputs (pre- and post-escape-hatch, if applicable) in
//! the D1 commit message — these are the baseline for D5's full-kernel
//! comparison and for any future `cp.async`-contingency rework.

#![cfg(test)]

use kaio_core::instr::control::ControlOp;
use kaio_core::instr::memory::MemoryOp;
use kaio_core::instr::{ArithOp, special};
use kaio_core::ir::{Operand, PtxInstruction, PtxKernel, Register, RegisterAllocator, SharedDecl};
use kaio_core::types::PtxType;

use crate::store_out::emit_store_fragment_c_f32_to_f16_packed;

// Mirror the D1 tile constants — both qkv_project_int{4,8}_kernel modules export
// these with identical values after the W8A16 unification.
const MMAS_PER_WARP_M: u32 = 2;
const MMAS_PER_WARP_N: u32 = 1; // Post-Rollback-#1 (Sprint 7.3): halved from 2 to 1 on both variants.
const SUBTILES_PER_BANK: u32 = MMAS_PER_WARP_M * MMAS_PER_WARP_N; // 2
const REGS_PER_FRAGMENT_C: u32 = 4; // m16n8k16 fragment-C: 4 f32 regs per lane
const PROJECTIONS: u32 = 3; // Q, K, V

// Fragment-A for m16n8k16 f16: 4 b32 regs per lane (holds 8 f16 values).
const REGS_PER_FRAGMENT_A: u32 = 4;
// Fragment-B for m16n8k16 f16: 2 b32 regs per lane (holds 4 f16 values).
const REGS_PER_FRAGMENT_B: u32 = 2;

// ── Sprint 7.3.5 S+½P shared layout ────────────────────────────────────
//
// tile_x       : 2048 B  single slot, register-hoisted (frag_A lives in regs
//                across the K-tile, tile_x overwritten by X_next while frag_A
//                serves all three mma epochs — Design invariant #1)
// tile_w_slot0 : 512 B
// tile_w_pad   : 64 B   ── bank-phase padding (Design invariant #4);
//                stride(slot0 → slot1) = 512 + 64 = 576 B, non-multiple of
//                128 B, shifts bank mapping by 16 for maximum dispersal.
// tile_w_slot1 : 512 B
// Total        : 3136 B (well under sm_89's 100 KB/SM limit)
//
// In the production kernel this is emitted as one `tile_w` array of 1088 B
// (= 512 + 64 + 512) and per-slot base addresses are computed via offsets.
// The skeleton declares them as three separate `SharedDecl`s so the emitted
// PTX's `.shared` section is self-documenting; the register footprint is
// unchanged.
const TILE_X_BYTES: u32 = 2048;
const TILE_W_SLOT_BYTES: u32 = 512;
const TILE_W_PAD_BYTES: u32 = 64;
const SLOT_STRIDE_BYTES: u32 = TILE_W_SLOT_BYTES + TILE_W_PAD_BYTES; // 576

// Three unrolled K-loop iterations to model steady → steady → final
// transitions (Opus R3b-3 — 2-iteration unroll misses the back-edge
// transition between two steady-state iterations, which is a class of
// slot-mapping bug that lives specifically at that back-edge).
const K_ITERATIONS_MODELLED: u32 = 3;

// INT4 scales register-hoist cost — 3 projections × 2 scale regs per
// fragment-B (§2 pre-review clarification).
const SCALES_HOISTED_REGS: u32 = PROJECTIONS * REGS_PER_FRAGMENT_B; // 6

/// Build the Sprint 7.3.5 D1 S+½P tri-output register-pressure skeleton PTX.
///
/// Returns a complete PTX module string targeting `sm`, ready for
/// `ptxas --verbose` to report register count, spill bytes, shared bytes,
/// and `.const` bytes at the S+½P peak.
pub(crate) fn build_qkv_tri_output_skeleton_ptx(sm: &str) -> String {
    use kaio_core::emit::{Emit, PtxWriter};
    use kaio_core::ir::{PtxModule, PtxParam};

    let mut alloc = RegisterAllocator::new();
    let mut kernel = PtxKernel::new("qkv_sp_half_p_skeleton");

    // Kernel signature — includes the pointer set the real kernel will
    // receive. Having the 6 base pointers (Q/K/V output + 3 scale bases)
    // in the param list is what sets up the Opus R3b-2 SASS inspection
    // check: we want to observe whether ptxas places them in GPRs or
    // uniform-broadcasts them from constant bank 0.
    kernel.add_param(PtxParam::pointer("q_out", PtxType::F16));
    kernel.add_param(PtxParam::pointer("k_out", PtxType::F16));
    kernel.add_param(PtxParam::pointer("v_out", PtxType::F16));
    kernel.add_param(PtxParam::pointer("scale_q_base", PtxType::F32));
    kernel.add_param(PtxParam::pointer("scale_k_base", PtxType::F32));
    kernel.add_param(PtxParam::pointer("scale_v_base", PtxType::F32));
    kernel.add_param(PtxParam::scalar("stride", PtxType::U32));

    // ── Shared decls: 2-slot W layout with 64 B bank-phase padding ──
    kernel.add_shared_decl(SharedDecl {
        name: "tile_x".to_string(),
        align: 4,
        size_bytes: TILE_X_BYTES,
    });
    kernel.add_shared_decl(SharedDecl {
        name: "tile_w_slot0".to_string(),
        align: 4,
        size_bytes: TILE_W_SLOT_BYTES,
    });
    // The padding is emitted as a .shared .b8 array so the bank-phase
    // offset is observable in the compiled module. The production kernel
    // does the same conceptually — a single tile_w array with explicit
    // offset math. Either shape produces identical ptxas register
    // output.
    kernel.add_shared_decl(SharedDecl {
        name: "tile_w_pad".to_string(),
        align: 4,
        size_bytes: TILE_W_PAD_BYTES,
    });
    kernel.add_shared_decl(SharedDecl {
        name: "tile_w_slot1".to_string(),
        align: 4,
        size_bytes: TILE_W_SLOT_BYTES,
    });

    // ── Load + cvta the 6 base pointers ──
    let q_base = load_and_cvta_pointer(&mut alloc, &mut kernel, "q_out");
    let k_base = load_and_cvta_pointer(&mut alloc, &mut kernel, "k_out");
    let v_base = load_and_cvta_pointer(&mut alloc, &mut kernel, "v_out");
    let scale_q_base = load_and_cvta_pointer(&mut alloc, &mut kernel, "scale_q_base");
    let scale_k_base = load_and_cvta_pointer(&mut alloc, &mut kernel, "scale_k_base");
    let scale_v_base = load_and_cvta_pointer(&mut alloc, &mut kernel, "scale_v_base");

    // Load the row stride scalar.
    let row_stride = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: row_stride,
        param_name: "stride".to_string(),
        ty: PtxType::U32,
    }));

    // Read tid.x for the store-out lane id + ctaid.x for the runtime
    // k-tile index. Using ctaid.x as the per-block k-tile base ensures
    // ptxas cannot constant-fold the slot-selection or group-selection
    // math — we must emit real runtime arithmetic.
    let (tid_lane, tid_instr) = special::tid_x(&mut alloc);
    kernel.push(tid_instr);
    let (ctaid, ctaid_instr) = special::ctaid_x(&mut alloc);
    kernel.push(ctaid_instr);

    // tile_w slot base addresses — captured as .shared address symbols,
    // materialized into u32 registers. These two registers are the
    // ping-pong slot pointers, live across the K-loop.
    let tile_w_slot0_base = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: tile_w_slot0_base,
        src: Operand::SharedAddr("tile_w_slot0".to_string()),
        ty: PtxType::U32,
    });
    let tile_w_slot1_base = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: tile_w_slot1_base,
        src: Operand::SharedAddr("tile_w_slot1".to_string()),
        ty: PtxType::U32,
    });

    // tile_x shared base — for the frag_A hoist.
    let tile_x_base = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: tile_x_base,
        src: Operand::SharedAddr("tile_x".to_string()),
        ty: PtxType::U32,
    });

    // ── Allocate + initialize all 24 FragmentC f32 regs per lane ──
    //
    // Layout: PROJECTIONS × SUBTILES_PER_BANK × REGS_PER_FRAGMENT_C
    //       = 3 × 2 × 4 = 24 f32 regs per lane (post-Rollback-#1).
    //
    // Each register is `mov.f32`-initialized with a distinct immediate
    // so ptxas cannot fold them into a common constant. These form the
    // seed values for the cross-iteration back-edge chain below.
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
                src: Operand::ImmF32(0.001 * (slot * REGS_PER_FRAGMENT_C + i as u32 + 1) as f32),
                ty: PtxType::F32,
            });
            *reg_slot = r;
        }
        frag_c.push(regs);
    }

    // ── Modelled K-loop (3 iterations unrolled) ──
    //
    // Each iteration threads state through:
    //   frag_C[i]_{n+1} = fma(frag_A, frag_B_n, frag_C[i]_n)
    //
    // The back-edge from frag_C[i]_n to frag_C[i]_{n+1} forces ptxas to
    // carry all 24 frag_C regs live across the entire modelled loop body.
    // Iteration-local state (frag_A, frag_B, scales, slot pointers,
    // in-flight load address regs) is allocated fresh per iteration so
    // ptxas can see the peak simultaneous liveness window inside one
    // iteration's body.
    for iter in 0..K_ITERATIONS_MODELLED {
        emit_modelled_k_iteration(
            &mut alloc,
            &mut kernel,
            iter,
            ctaid,
            tile_x_base,
            tile_w_slot0_base,
            tile_w_slot1_base,
            scale_q_base,
            scale_k_base,
            scale_v_base,
            &mut frag_c,
        );
    }

    // ── Tri-output store epilogue ──
    //
    // For each of the SUBTILES_PER_BANK sub-tiles, emit the three
    // store-out calls (one per projection). All 24 FragmentC regs are
    // read here, closing the back-edge chain and forcing ptxas to
    // retain every register from initialization through the epilogue.
    let bases = [q_base, k_base, v_base];
    // Per-projection scales: load one scalar from each scale_base as a
    // stand-in for the scale-hoisted values. The store-out helper
    // accepts Option<Register> so we pass Some(scale) for the f32
    // multiply path.
    let epilogue_scales = [
        load_first_f32_from_base(&mut alloc, &mut kernel, scale_q_base),
        load_first_f32_from_base(&mut alloc, &mut kernel, scale_k_base),
        load_first_f32_from_base(&mut alloc, &mut kernel, scale_v_base),
    ];
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
                Some(epilogue_scales[proj]),
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

/// Emit one modelled K-loop iteration body.
///
/// The body allocates iteration-local state (frag-A, frag-B, scales,
/// ping-pong slot pointers, in-flight load address registers), threads
/// it through a chain of `fma.f32` updates on the shared `frag_c` grids,
/// and leaves the grids in a state consumed by the next iteration (or
/// the store-out epilogue).
///
/// `iter` is the iteration index — added to `ctaid` to give each
/// iteration a distinct runtime-varying k-tile index so ptxas cannot
/// CSE the slot-index or group-index math across iterations.
#[allow(clippy::too_many_arguments)]
fn emit_modelled_k_iteration(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    iter: u32,
    ctaid: Register,
    tile_x_base: Register,
    tile_w_slot0_base: Register,
    tile_w_slot1_base: Register,
    scale_q_base: Register,
    scale_k_base: Register,
    scale_v_base: Register,
    frag_c: &mut [[Register; 4]],
) {
    // k_tile_runtime = ctaid.x + iter (runtime-varying per iteration).
    let k_tile = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: k_tile,
        lhs: Operand::Reg(ctaid),
        rhs: Operand::ImmU32(iter),
        ty: PtxType::U32,
    }));

    // slot_idx = k_tile & 1 (runtime bit test — ptxas cannot fold).
    let slot_idx = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::And {
        dst: slot_idx,
        lhs: Operand::Reg(k_tile),
        rhs: Operand::ImmU32(1),
        ty: PtxType::U32,
    }));

    // next_slot_idx = (k_tile + 1) & 1 = slot_idx XOR 1.
    let next_slot_idx = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Xor {
        dst: next_slot_idx,
        lhs: Operand::Reg(slot_idx),
        rhs: Operand::ImmU32(1),
        ty: PtxType::U32,
    }));

    // Runtime ping-pong pointer selection:
    //   cur_w_addr  = (slot_idx == 0) ? slot0_base : slot1_base
    //   next_w_addr = (next_slot_idx == 0) ? slot0_base : slot1_base
    //
    // Model via: slot_offset = slot_idx * SLOT_STRIDE_BYTES, then
    // cur_w_addr = slot0_base + slot_offset.
    // (slot0_base + 576 happens to equal slot1_base because the shared
    // decls are contiguous with 64B pad, which is precisely the bank-
    // phase padding under test.)
    let cur_offset = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: cur_offset,
        lhs: Operand::Reg(slot_idx),
        rhs: Operand::ImmU32(SLOT_STRIDE_BYTES),
        ty: PtxType::U32,
    }));
    let cur_w_addr = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: cur_w_addr,
        lhs: Operand::Reg(tile_w_slot0_base),
        rhs: Operand::Reg(cur_offset),
        ty: PtxType::U32,
    }));

    let next_offset = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: next_offset,
        lhs: Operand::Reg(next_slot_idx),
        rhs: Operand::ImmU32(SLOT_STRIDE_BYTES),
        ty: PtxType::U32,
    }));
    let next_w_addr = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: next_w_addr,
        lhs: Operand::Reg(tile_w_slot1_base),
        rhs: Operand::Reg(next_offset),
        ty: PtxType::U32,
    }));
    // (We reference tile_w_slot1_base here even though cur_w_addr used
    // slot0_base — this keeps both symbols live across the iteration so
    // ptxas doesn't eliminate either, modelling the real kernel where
    // both slot addresses are reachable and reused across epochs.)

    // ── frag_A hoist: ld.shared from tile_x → 4 b32 regs ──
    //
    // The real kernel loads frag_A once per K-tile and reuses across
    // all 3 mma epochs. Each thread loads 4 b32 = 8 f16 values. We
    // model this with 4 ld.shared.b32 from staggered tile_x offsets;
    // ptxas sees the full frag_A register envelope live for the
    // duration of the iteration.
    let frag_a: [Register; REGS_PER_FRAGMENT_A as usize] = core::array::from_fn(|i| {
        let addr = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: addr,
            lhs: Operand::Reg(tile_x_base),
            rhs: Operand::ImmU32(i as u32 * 4),
            ty: PtxType::U32,
        }));
        let r = alloc.alloc_packed_half2();
        kernel.push(PtxInstruction::Memory(MemoryOp::LdShared {
            dst: r,
            addr,
            ty: PtxType::U32,
        }));
        r
    });

    // ── Scales register-hoist: 3 projections × 2 scale regs ──
    //
    // Runtime k_tile_group = k_tile >> 3 (group_size=128 at K_TILE=16,
    // so group boundary hits every 8 K-tiles). The shift is emitted as
    // a runtime instruction so ptxas cannot prove monotonic or cache
    // the group index across iterations.
    let k_tile_group = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Shr {
        dst: k_tile_group,
        lhs: Operand::Reg(k_tile),
        rhs: Operand::ImmU32(3),
        ty: PtxType::U32,
    }));

    // Per-projection scale load: offset = k_tile_group * sizeof(f32) * 2,
    // added to scale_{q,k,v}_base, loaded into 2 f32 regs per projection.
    // Six f32 regs total (Opus R3b-2: production will use ld.global.nc
    // or uniform broadcast; skeleton uses ld.global for the register
    // envelope — caching behavior doesn't affect ptxas count).
    let scale_offset = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
        dst: scale_offset,
        lhs: Operand::Reg(k_tile_group),
        rhs: Operand::ImmU32(8), // 2 f32 × 4 bytes
        src_ty: PtxType::U32,
    }));

    let hoist_scale_pair = |alloc: &mut RegisterAllocator,
                            kernel: &mut PtxKernel,
                            base: Register|
     -> [Register; REGS_PER_FRAGMENT_B as usize] {
        let addr = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: addr,
            lhs: Operand::Reg(base),
            rhs: Operand::Reg(scale_offset),
            ty: PtxType::U64,
        }));
        core::array::from_fn(|i| {
            let off_addr = alloc.alloc(PtxType::U64);
            kernel.push(PtxInstruction::Arith(ArithOp::Add {
                dst: off_addr,
                lhs: Operand::Reg(addr),
                rhs: Operand::ImmU32(i as u32 * 4),
                ty: PtxType::U64,
            }));
            let s = alloc.alloc(PtxType::F32);
            kernel.push(PtxInstruction::Memory(MemoryOp::LdGlobal {
                dst: s,
                addr: off_addr,
                ty: PtxType::F32,
            }));
            s
        })
    };
    let scales_q = hoist_scale_pair(alloc, kernel, scale_q_base);
    let scales_k = hoist_scale_pair(alloc, kernel, scale_k_base);
    let scales_v = hoist_scale_pair(alloc, kernel, scale_v_base);
    // SCALES_HOISTED_REGS constant matches what we emit here (3 × 2 = 6)
    // — exercised at compile time for documentation purposes.
    debug_assert_eq!(
        SCALES_HOISTED_REGS as usize,
        scales_q.len() + scales_k.len() + scales_v.len()
    );

    // ── Simulated in-flight cooperative W load ──
    //
    // In the real kernel, the overlapping load of W_{P+1} happens
    // concurrently with the current mma epoch. Per thread this is one
    // b32 load from global + one b32 store to tile_w at next_w_addr.
    // We model only the address math and the resulting load — keeping
    // the load-destination register and the store-address register
    // live simultaneously with the mma epoch below.
    let in_flight_global_offset = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
        dst: in_flight_global_offset,
        lhs: Operand::Reg(k_tile),
        rhs: Operand::ImmU32(16),
        src_ty: PtxType::U32,
    }));
    let in_flight_frag_b_q: [Register; REGS_PER_FRAGMENT_B as usize] = core::array::from_fn(|i| {
        let addr = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: addr,
            lhs: Operand::Reg(cur_w_addr),
            rhs: Operand::ImmU32(i as u32 * 4),
            ty: PtxType::U32,
        }));
        let r = alloc.alloc_packed_half2();
        kernel.push(PtxInstruction::Memory(MemoryOp::LdShared {
            dst: r,
            addr,
            ty: PtxType::U32,
        }));
        r
    });
    // The overlap store — one thread's b32 into next_w_addr. Having
    // next_w_addr live across the mma chain below matches the real
    // kernel's overlap window.
    let overlap_sink = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: overlap_sink,
        src: Operand::ImmU32(0xCAFEBABE),
        ty: PtxType::U32,
    });
    kernel.push(PtxInstruction::Memory(MemoryOp::StShared {
        addr: next_w_addr,
        src: overlap_sink,
        ty: PtxType::U32,
    }));

    // ── frag_C accumulator chain (back-edge across iterations) ──
    //
    // For each of 3 projections × SUBTILES_PER_BANK sub-tiles × 4 frag_C
    // regs per lane, emit:
    //     frag_c[i][r] = fma(frag_A_sink, frag_B_sink, frag_c[i][r])
    //
    // We first reduce frag_A and each projection's scales+frag_B into a
    // single f32 "contribution" value per projection, then FMA that into
    // every frag_C reg of that projection. This keeps frag_A, frag_B,
    // and scales simultaneously live until after all frag_C writes,
    // which is the peak liveness point.
    let frag_a_sink = reduce_b32_chain_to_f32(alloc, kernel, &frag_a);
    let frag_b_q_sink = reduce_b32_chain_to_f32(alloc, kernel, &in_flight_frag_b_q);
    // For K and V we simulate separate frag_B loads (in the real kernel,
    // the next K-tile's cooperative load fills a different slot each
    // epoch; here the register envelope is what matters).
    let frag_b_k: [Register; REGS_PER_FRAGMENT_B as usize] = core::array::from_fn(|i| {
        let addr = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: addr,
            lhs: Operand::Reg(next_w_addr),
            rhs: Operand::ImmU32(i as u32 * 4),
            ty: PtxType::U32,
        }));
        let r = alloc.alloc_packed_half2();
        kernel.push(PtxInstruction::Memory(MemoryOp::LdShared {
            dst: r,
            addr,
            ty: PtxType::U32,
        }));
        r
    });
    let frag_b_v: [Register; REGS_PER_FRAGMENT_B as usize] = core::array::from_fn(|i| {
        let addr = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: addr,
            lhs: Operand::Reg(cur_w_addr),
            rhs: Operand::ImmU32(i as u32 * 4 + 8),
            ty: PtxType::U32,
        }));
        let r = alloc.alloc_packed_half2();
        kernel.push(PtxInstruction::Memory(MemoryOp::LdShared {
            dst: r,
            addr,
            ty: PtxType::U32,
        }));
        r
    });
    let frag_b_k_sink = reduce_b32_chain_to_f32(alloc, kernel, &frag_b_k);
    let frag_b_v_sink = reduce_b32_chain_to_f32(alloc, kernel, &frag_b_v);

    // Fold each projection's scales into its frag_B contribution.
    let contribution_q = fma_scales_to_contrib(alloc, kernel, &scales_q, frag_b_q_sink);
    let contribution_k = fma_scales_to_contrib(alloc, kernel, &scales_k, frag_b_k_sink);
    let contribution_v = fma_scales_to_contrib(alloc, kernel, &scales_v, frag_b_v_sink);

    // Per-projection FMA chain over the sub-tiles belonging to that
    // projection — updates frag_C in place (back-edge for next iter).
    // Iterate via `iter_mut().enumerate()` and derive `proj` from the
    // flattened index; cleaner than triple-nested index loops.
    let contributions = [contribution_q, contribution_k, contribution_v];
    for (frag_idx, frag_c_bank) in frag_c.iter_mut().enumerate() {
        let proj = frag_idx / SUBTILES_PER_BANK as usize;
        let contrib = contributions[proj];
        for reg_slot in frag_c_bank.iter_mut() {
            let next = alloc.alloc(PtxType::F32);
            kernel.push(PtxInstruction::Arith(ArithOp::Fma {
                dst: next,
                a: Operand::Reg(frag_a_sink),
                b: Operand::Reg(contrib),
                c: Operand::Reg(*reg_slot),
                ty: PtxType::F32,
            }));
            *reg_slot = next;
        }
    }
}

/// Reduce an array of b32 registers into a single f32 value via an XOR
/// chain followed by a reinterpret. Used to collapse frag_A and frag_B
/// staging into one f32 we can FMA into the frag_C grids. Keeps every
/// input register live until the final reduction.
fn reduce_b32_chain_to_f32(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    regs: &[Register],
) -> Register {
    let mut running = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: running,
        src: Operand::ImmU32(0),
        ty: PtxType::U32,
    });
    for r in regs {
        let next = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Xor {
            dst: next,
            lhs: Operand::Reg(running),
            rhs: Operand::Reg(*r),
            ty: PtxType::U32,
        }));
        running = next;
    }
    // Convert u32 → f32 so the reduction feeds into fma.f32 below.
    // Value is meaningless (we're modelling register pressure, not
    // correctness) — the semantic conversion carries the register
    // dependency chain into the f32 fma grid just as well as a bit-
    // reinterpret would.
    let as_f32 = alloc.alloc(PtxType::F32);
    kernel.push(PtxInstruction::Cvt {
        dst: as_f32,
        src: running,
        dst_ty: PtxType::F32,
        src_ty: PtxType::U32,
    });
    as_f32
}

/// Fold a projection's 2 scale regs into its frag_B sink via one FMA,
/// producing a single f32 contribution value.
fn fma_scales_to_contrib(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    scales: &[Register; REGS_PER_FRAGMENT_B as usize],
    frag_b_sink: Register,
) -> Register {
    let tmp = alloc.alloc(PtxType::F32);
    kernel.push(PtxInstruction::Arith(ArithOp::Fma {
        dst: tmp,
        a: Operand::Reg(scales[0]),
        b: Operand::Reg(frag_b_sink),
        c: Operand::Reg(scales[1]),
        ty: PtxType::F32,
    }));
    tmp
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

/// Load a single f32 from the start of a base pointer, used in the
/// store-out epilogue to materialize a scalar scale (stand-in for the
/// register-hoisted scales which live in per-iteration scope).
fn load_first_f32_from_base(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    base: Register,
) -> Register {
    let r = alloc.alloc(PtxType::F32);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdGlobal {
        dst: r,
        addr: base,
        ty: PtxType::F32,
    }));
    r
}

// --- Register-allocator placeholder helper ------------------------------

trait RegisterPlaceholder {
    fn placeholder() -> Self;
}

impl RegisterPlaceholder for Register {
    fn placeholder() -> Self {
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

    /// Runs ptxas --verbose over the S+½P skeleton and parses the D1
    /// baseline from stderr. `#[ignore]` so host-only CI stays green.
    ///
    /// Pass criteria: `Used N registers, N ≤ 64, 0 spills` on both
    /// sm_80 and sm_89.
    ///
    /// Archive: regs/thread, shared bytes, instruction count, `.const`
    /// bytes — D1 commit message retains both sm_80 and sm_89 outputs.
    #[test]
    #[ignore]
    fn ptxas_verify_qkv_sp_half_p_skeleton() {
        if Command::new("ptxas").arg("--version").output().is_err() {
            eprintln!("NOTE: ptxas not found in PATH — skipping D1 checkpoint");
            return;
        }

        let sm = std::env::var("KAIO_SM_TARGET").unwrap_or_else(|_| "sm_89".to_string());
        let ptx = build_qkv_tri_output_skeleton_ptx(&sm);
        let tmp = std::env::temp_dir().join("kaio_qkv_sp_half_p_skeleton.ptx");
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
            "ptxas FAILED on S+½P skeleton ({sm}):\n{stderr}\n\
             === PTX (first 8000 chars) ===\n{}",
            &ptx[..ptx.len().min(8000)]
        );

        let combined = format!("{stdout}\n{stderr}");
        let regs = extract_metric(&combined, "Used ", " registers");
        let smem = extract_metric(&combined, ", used 0 barriers, ", " bytes smem");
        let cmem0 = extract_metric(&combined, ", used 0 barriers, ", " bytes cmem[0]");
        let spill_stores = extract_metric(&combined, "bytes stack frame, ", " bytes spill stores");
        let spill_loads = extract_metric(&combined, "spill stores, ", " bytes spill loads");

        eprintln!(
            "=== D1 S+½P ARCHIVED BASELINE ({sm}) ===\n\
             regs/thread     : {regs:?}\n\
             shared bytes    : {smem:?} (expected ~{} — tile_x + 2×tile_w + pad)\n\
             cmem[0] bytes   : {cmem0:?} (params, uniform-broadcast-friendly)\n\
             spill stores    : {spill_stores:?}\n\
             spill loads     : {spill_loads:?}\n\
             PTX instr count : {}\n\
             user const bytes: {} (grep .const decls in emitted PTX)",
            TILE_X_BYTES + 2 * TILE_W_SLOT_BYTES + TILE_W_PAD_BYTES,
            count_instructions_in_ptx(&ptx),
            count_const_bytes_in_ptx(&ptx),
        );

        // Gate pass/fail: register count must fit 64-reg occupancy tier
        // and there must be zero spills.
        match regs {
            Some(n) => assert!(
                n <= 64,
                "D1 S+½P skeleton reports {n} registers > 64 on {sm}; \
                 per plan: (1) SASS-inspect via `cuobjdump --dump-sass` to \
                 check whether base pointers are in GPRs (case b) vs uniform \
                 broadcasts (case a, nothing to reclaim); (2) if (b), re-emit \
                 with explicit .const/.param placement for base pointers; \
                 (3) if that reclaims ≥ 2 regs, proceed to D2. Otherwise \
                 Rollback #2 activates (INT8-only S+½P, INT4 stays Design S)."
            ),
            None => panic!(
                "Could not parse register count from ptxas output — update extract_metric \
                 pattern?\nstdout was:\n{stdout}\nstderr was:\n{stderr}"
            ),
        }
        assert_eq!(
            spill_stores,
            Some(0),
            "D1 S+½P skeleton has spill stores on {sm} — register budget blown."
        );
        assert_eq!(
            spill_loads,
            Some(0),
            "D1 S+½P skeleton has spill loads on {sm} — register budget blown."
        );

        eprintln!("D1 S+½P SKELETON GATE PASSED ({sm}): peak fits 64-reg tier, zero spills.");
    }

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

    fn count_const_bytes_in_ptx(ptx: &str) -> usize {
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
    /// default SMs. Host-side smoke, no ptxas dependency. Exercises the
    /// full emit path for the S+½P kernel shape.
    #[test]
    fn skeleton_module_builds() {
        let ptx_80 = build_qkv_tri_output_skeleton_ptx("sm_80");
        let ptx_89 = build_qkv_tri_output_skeleton_ptx("sm_89");
        assert!(ptx_80.contains(".entry qkv_sp_half_p_skeleton"));
        assert!(ptx_89.contains(".entry qkv_sp_half_p_skeleton"));

        // Shared decls reflect the 2-slot layout with 64 B bank-phase
        // padding — 3136 B total across 4 named slots.
        assert!(ptx_89.contains(".shared .align 4 .b8 tile_x[2048]"));
        assert!(ptx_89.contains(".shared .align 4 .b8 tile_w_slot0[512]"));
        assert!(ptx_89.contains(".shared .align 4 .b8 tile_w_pad[64]"));
        assert!(ptx_89.contains(".shared .align 4 .b8 tile_w_slot1[512]"));

        // Runtime slot-index math must be present (And + Xor + Mul
        // sequence), repeated K_ITERATIONS_MODELLED times. We check at
        // least one occurrence of each idiom per iteration.
        let and_count = ptx_89.matches("and.b32").count();
        assert!(
            and_count >= K_ITERATIONS_MODELLED as usize,
            "expected ≥ {} and.b32 ops for runtime slot-index math; got {}",
            K_ITERATIONS_MODELLED,
            and_count
        );

        // Fma chain for frag_C updates: per iteration, PROJECTIONS ×
        // SUBTILES_PER_BANK × REGS_PER_FRAGMENT_C = 3 × 2 × 4 = 24 FMAs
        // on frag_C, plus 3 FMAs folding scales into frag_B (one per
        // projection) = 27 FMAs per iteration × K_ITERATIONS_MODELLED.
        let expected_fma_min = (24 + PROJECTIONS) * K_ITERATIONS_MODELLED;
        let fma_count = ptx_89.matches("fma.rn.f32").count();
        assert!(
            fma_count >= expected_fma_min as usize,
            "expected ≥ {expected_fma_min} fma.rn.f32 ops for frag_C back-edge chain; got {fma_count}"
        );
    }
}
