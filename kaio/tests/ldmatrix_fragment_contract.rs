//! # Sprint 9.3 gate — ldmatrix fragment-A loader contract.
//!
//! Proves on hardware that [`load_fragment_a_m16n8k16_ldmatrix`]
//! produces **bit-identical** per-thread fragment registers to the
//! shipped [`load_fragment_a_m16n8k16_shared_row`] loader on the same
//! shared-tile data — equivalence to the loader being replaced, not to
//! a hand-derived layout, so the quadrant-order derivation in the
//! loader docs is itself under test. This gate runs **before** the
//! matmul_tc rewire commit; the rewire swaps loaders only because this
//! test says the swap is invisible.
//!
//! ## Coverage
//!
//! - **Both A stripes**: the matmul_tc inner loop calls the loader
//!   twice per warp with a `m_stripe * 16 * row_stride` base
//!   adjustment, so the contract is checked at stripe base 0 **and**
//!   stripe base +512 B (rows 16..31) — a wrong base handling cannot
//!   pass by symmetry.
//! - **Nontrivial tile contents**: element (row, col) holds
//!   `row * 16 + col` (0..511, all exact in fp16) — every element
//!   distinct, so any quadrant-order or offset mistake produces a
//!   visible mismatch instead of passing by coincidence.
//! - **Harness self-check**: the `ld.shared` loader's output is also
//!   spot-checked against the documented PTX ISA mapping at the
//!   warp corners (lane 0 reg0, lane 31 reg3), so an all-zeros or
//!   data-never-loaded failure cannot masquerade as equivalence.
//!
//! If the bit-equality fails, debug the per-lane address derivation in
//! `load_fragment_a_m16n8k16_ldmatrix` (row = lane & 15,
//! col_byte = lane & 16 → quadrant order [TL, BL, TR, BR]). Do not
//! touch the `ld.shared` loader — it is the reference.

use cudarc::driver::{LaunchConfig, PushKernelArg};
use half::f16;
use kaio::prelude::*;
use kaio_core::emit::{Emit, PtxWriter};
use kaio_core::fragment::{load_fragment_a_m16n8k16_ldmatrix, load_fragment_a_m16n8k16_shared_row};
use kaio_core::instr::control::ControlOp;
use kaio_core::instr::memory::MemoryOp;
use kaio_core::instr::special;
use kaio_core::instr::{ArithOp, MadMode};
use kaio_core::ir::{
    Operand, PtxInstruction, PtxKernel, PtxModule, PtxParam, RegisterAllocator, SharedDecl,
};
use kaio_core::types::PtxType;

/// Tile geometry: 32 rows × 16 fp16 cols (32-byte row stride) — two
/// 16-row A stripes, matching the matmul_tc `tile_a` layout.
const TILE_ROWS: usize = 32;
const TILE_COLS: usize = 16;
const ROW_STRIDE_BYTES: u32 = 32;
const STRIPE_BYTES: u32 = 16 * ROW_STRIDE_BYTES; // 16 rows per stripe
const TILE_BYTES: usize = TILE_ROWS * ROW_STRIDE_BYTES as usize;
/// Output: 2 stripes × 32 lanes × 4 fragment registers, u32 each.
const OUT_WORDS: usize = 2 * 32 * 4;

/// Build the contract kernel.
///
/// Params: `in_ptr` (f16 tile, row-major), `out_old` (u32), `out_new`
/// (u32). One warp. Cooperatively stages the tile into shared memory,
/// syncs, then for each stripe runs BOTH loaders on the same stripe
/// base and stores all four fragment registers of each to the
/// corresponding out buffer at `[(stripe*32 + lane)*4 + i]`.
fn build_contract_module() -> PtxModule {
    let mut alloc = RegisterAllocator::new();
    let mut kernel = PtxKernel::new("ldmatrix_contract");

    kernel.add_param(PtxParam::pointer("in_ptr", PtxType::F16));
    kernel.add_param(PtxParam::pointer("out_old", PtxType::U32));
    kernel.add_param(PtxParam::pointer("out_new", PtxType::U32));

    // ldmatrix row addresses must be 16-byte aligned at runtime; the
    // declaration alignment is what guarantees the tile base.
    kernel.add_shared_decl(SharedDecl {
        name: "tile".to_string(),
        align: 16,
        size_bytes: TILE_BYTES as u32,
    });

    // Params → global-space pointers.
    let mut global_ptr = |name: &str| {
        let rd_param = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
            dst: rd_param,
            param_name: name.to_string(),
            ty: PtxType::U64,
        }));
        let rd = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Memory(MemoryOp::CvtaToGlobal {
            dst: rd,
            src: rd_param,
        }));
        rd
    };
    let rd_in = global_ptr("in_ptr");
    let rd_old = global_ptr("out_old");
    let rd_new = global_ptr("out_new");

    let (r_tid, tid_instr) = special::tid_x(&mut alloc);
    kernel.push(tid_instr);

    let r_tile = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_tile,
        src: Operand::SharedAddr("tile".to_string()),
        ty: PtxType::U32,
    });

    // --- Cooperative global → shared stage: 256 u32 words, 8 per lane.
    let words = TILE_BYTES / 4;
    let iters = words / 32;
    for w in 0..iters as u32 {
        // word index = tid + w*32 (coalesced)
        let r_idx = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: r_idx,
            lhs: Operand::Reg(r_tid),
            rhs: Operand::ImmU32(w * 32),
            ty: PtxType::U32,
        }));
        // global byte offset (widened)
        let rd_off = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
            dst: rd_off,
            lhs: Operand::Reg(r_idx),
            rhs: Operand::ImmU32(4),
            src_ty: PtxType::U32,
        }));
        let rd_addr = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: rd_addr,
            lhs: Operand::Reg(rd_in),
            rhs: Operand::Reg(rd_off),
            ty: PtxType::U64,
        }));
        let r_val = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Memory(MemoryOp::LdGlobal {
            dst: r_val,
            addr: rd_addr,
            ty: PtxType::U32,
        }));
        // shared byte address = tile + idx*4
        let r_saddr = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Mad {
            dst: r_saddr,
            a: Operand::Reg(r_idx),
            b: Operand::ImmU32(4),
            c: Operand::Reg(r_tile),
            ty: PtxType::U32,
            mode: MadMode::Lo,
        }));
        kernel.push(PtxInstruction::Memory(MemoryOp::StShared {
            addr: r_saddr,
            src: r_val,
            ty: PtxType::U32,
        }));
    }
    kernel.push(PtxInstruction::Control(ControlOp::BarSync {
        barrier_id: 0,
    }));

    // Per-lane output byte offset within a stripe block: lane * 16.
    let rd_lane16 = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
        dst: rd_lane16,
        lhs: Operand::Reg(r_tid),
        rhs: Operand::ImmU32(16),
        src_ty: PtxType::U32,
    }));
    let rd_old_lane = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: rd_old_lane,
        lhs: Operand::Reg(rd_old),
        rhs: Operand::Reg(rd_lane16),
        ty: PtxType::U64,
    }));
    let rd_new_lane = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: rd_new_lane,
        lhs: Operand::Reg(rd_new),
        rhs: Operand::Reg(rd_lane16),
        ty: PtxType::U64,
    }));

    for stripe in 0..2u32 {
        // Stripe base in shared: tile + stripe * 512 (the matmul_tc
        // m_stripe * 16 * row_stride adjustment under test).
        let r_stripe_base = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: r_stripe_base,
            lhs: Operand::Reg(r_tile),
            rhs: Operand::ImmU32(stripe * STRIPE_BYTES),
            ty: PtxType::U32,
        }));

        let frag_old = load_fragment_a_m16n8k16_shared_row(
            &mut alloc,
            &mut kernel,
            r_stripe_base,
            r_tid,
            ROW_STRIDE_BYTES,
            None,
        );
        let frag_new = load_fragment_a_m16n8k16_ldmatrix(
            &mut alloc,
            &mut kernel,
            r_stripe_base,
            r_tid,
            ROW_STRIDE_BYTES,
        );

        for (rd_base, frag_regs) in [(rd_old_lane, &frag_old.regs), (rd_new_lane, &frag_new.regs)] {
            for (i, reg) in frag_regs.iter().enumerate() {
                // out[(stripe*32 + lane)*4 + i] — byte offset relative
                // to the lane base: stripe*512 + i*4.
                let rd_addr = alloc.alloc(PtxType::U64);
                kernel.push(PtxInstruction::Arith(ArithOp::Add {
                    dst: rd_addr,
                    lhs: Operand::Reg(rd_base),
                    rhs: Operand::ImmU64((stripe as u64) * 512 + (i as u64) * 4),
                    ty: PtxType::U64,
                }));
                kernel.push(PtxInstruction::Memory(MemoryOp::StGlobal {
                    addr: rd_addr,
                    src: *reg,
                    ty: PtxType::U32,
                }));
            }
        }
    }

    kernel.push(PtxInstruction::Control(ControlOp::Ret));
    kernel.set_registers(alloc.into_allocated());

    // ldmatrix floors at sm_75 (no mma in this kernel); honor a higher
    // requested target.
    let requested = std::env::var("KAIO_SM_TARGET").unwrap_or_else(|_| "sm_75".to_string());
    let sm = match requested
        .strip_prefix("sm_")
        .and_then(|s| s.parse::<u32>().ok())
    {
        Some(v) if v >= 75 => requested,
        _ => "sm_75".to_string(),
    };
    let mut module = PtxModule::new(&sm);
    module.add_kernel(kernel);
    module
}

fn emit_ptx_debug(module: &PtxModule) -> String {
    let mut w = PtxWriter::new();
    module.emit(&mut w).unwrap();
    w.finish()
}

/// Host tile: element (row, col) = row * 16 + col — 0..511, all exact
/// in fp16, every element distinct.
fn build_tile_host() -> Vec<f16> {
    let mut t = Vec::with_capacity(TILE_ROWS * TILE_COLS);
    for r in 0..TILE_ROWS {
        for c in 0..TILE_COLS {
            t.push(f16::from_f32((r * TILE_COLS + c) as f32));
        }
    }
    t
}

/// Pack two consecutive tile elements (row, col) and (row, col+1) the
/// way `ld.shared.b32` sees them in the row-major tile: low half =
/// even column.
fn packed(tile: &[f16], row: usize, col: usize) -> u32 {
    let lo = tile[row * TILE_COLS + col].to_bits() as u32;
    let hi = tile[row * TILE_COLS + col + 1].to_bits() as u32;
    lo | (hi << 16)
}

#[test]
#[ignore] // requires NVIDIA GPU with SM 7.5+
fn ldmatrix_fragment_a_contract_gate() {
    let ptx_module = build_contract_module();

    let device = KaioDevice::new(0).expect("GPU required for this test");
    let info = device.info().expect("device info");
    let (major, minor) = info.compute_capability;
    assert!(
        (major, minor) >= (7, 5),
        "ldmatrix requires SM 7.5+ (got sm_{major}{minor})",
    );

    let module = device.load_module(&ptx_module).unwrap_or_else(|e| {
        eprintln!(
            "=== PTX that failed to load ===\n{}",
            emit_ptx_debug(&ptx_module)
        );
        panic!("load_module failed: {e}");
    });
    let func = module
        .function("ldmatrix_contract")
        .expect("function handle lookup");

    let tile_host = build_tile_host();
    let buf_in = device.alloc_from(&tile_host).expect("alloc tile");
    let mut buf_old = device.alloc_zeros::<u32>(OUT_WORDS).expect("alloc out_old");
    let mut buf_new = device.alloc_zeros::<u32>(OUT_WORDS).expect("alloc out_new");

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        device
            .stream()
            .launch_builder(func.inner())
            .arg(buf_in.inner())
            .arg(buf_old.inner_mut())
            .arg(buf_new.inner_mut())
            .launch(cfg)
    }
    .unwrap_or_else(|e| {
        eprintln!("=== PTX ===\n{}", emit_ptx_debug(&ptx_module));
        panic!("ldmatrix_contract launch failed: {e}");
    });

    let old_host = buf_old.to_host(&device).expect("out_old roundtrip");
    let new_host = buf_new.to_host(&device).expect("out_new roundtrip");
    assert_eq!(old_host.len(), OUT_WORDS);
    assert_eq!(new_host.len(), OUT_WORDS);

    // --- Harness self-check: the reference loader's output matches the
    // documented mapping at the warp corners (so "both buffers stayed
    // zero / tile never staged" cannot pass as equivalence).
    // Lane 0 (group 0, tig 0): reg0 = (row 0, cols 0..1).
    // Lane 31 (group 7, tig 3): reg3 = (row 15, cols 14..15); stripe 1
    // shifts rows by +16.
    for (stripe, row_base) in [(0usize, 0usize), (1, 16)] {
        let lane0_reg0 = old_host[(stripe * 32) * 4];
        assert_eq!(
            lane0_reg0,
            packed(&tile_host, row_base, 0),
            "harness self-check failed: stripe {stripe} lane 0 reg0 — \
             reference loader output does not match the documented mapping",
        );
        let lane31_reg3 = old_host[(stripe * 32 + 31) * 4 + 3];
        assert_eq!(
            lane31_reg3,
            packed(&tile_host, row_base + 15, 14),
            "harness self-check failed: stripe {stripe} lane 31 reg3",
        );
    }

    // --- The contract: ldmatrix fragments bit-equal to the ld.shared
    // loader's fragments, every lane, every register, both stripes.
    let mut mismatches: Vec<(usize, usize, usize, u32, u32)> = Vec::new();
    for stripe in 0..2 {
        for lane in 0..32 {
            for i in 0..4 {
                let idx = (stripe * 32 + lane) * 4 + i;
                if old_host[idx] != new_host[idx] {
                    mismatches.push((stripe, lane, i, old_host[idx], new_host[idx]));
                }
            }
        }
    }

    if !mismatches.is_empty() {
        let mut msg = format!(
            "ldmatrix fragment-A contract FAILED: {} of {OUT_WORDS} registers \
             differ from the ld.shared reference loader.\n\
             First 10 (stripe, lane, reg, ld_shared, ldmatrix):\n",
            mismatches.len()
        );
        for (s, l, i, o, n) in mismatches.iter().take(10) {
            msg.push_str(&format!(
                "  stripe {s} lane {l:2} reg{i}: {o:#010x} vs {n:#010x}\n"
            ));
        }
        msg.push_str(
            "\nDebug the per-lane address derivation in \
             load_fragment_a_m16n8k16_ldmatrix (row = lane & 15, \
             col_byte = lane & 16 → matrix order [TL, BL, TR, BR]).\n\
             The ld.shared loader is the reference — do not touch it.\n",
        );
        panic!("{msg}");
    }
}
