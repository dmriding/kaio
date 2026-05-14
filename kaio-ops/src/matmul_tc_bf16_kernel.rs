// C3-only allow: the bf16 kernel module's symbols become live when C4
// adds the public host launch fn `matmul_tc_bf16` (plus the `pub use`
// re-export in `lib.rs`). Remove this attribute at C4.
#![allow(dead_code)]

//! Tensor-core matrix multiply — `m16n8k16.f32.bf16.bf16.f32`, multi-warp.
//!
//! bf16 × bf16 inputs with fp32 accumulation, using the dedicated
//! `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` instruction
//! (Sprint 9.1 D2.5 `TensorCoreOp::MmaSyncBf16` IR variant).
//!
//! Structurally a near-mirror of [`crate::matmul_tc_kernel`]'s sync f16
//! path — same block tile (64×64), same multi-warp layout
//! (4 warps × 32×32 sub-quadrants), same bank-conflict-padded Tile B
//! (col-stride 36 B per Sprint 6.7b), same fragment-loader (group_id, tig)
//! hoist (Sprint 6.7b D10). The only structural differences live in:
//!
//! - the fragment types (`FragmentA_BF16`, `FragmentB_BF16` instead of
//!   their `_F16` siblings),
//! - the fragment loaders (`load_fragment_{a,b}_m16n8k16_shared_*_bf16`),
//! - the mma op (`TensorCoreOp::MmaSyncBf16` instead of `MmaSync` with
//!   bf16 dtype tags), and
//! - the host parameter types (`GpuBuffer<half::bf16>` for A and B).
//!
//! Per Sprint 9.1 D4, the bf16 byte layout in shared memory is bit-identical
//! to f16 (two 16-bit values packed per `.b32` register), so the
//! `pub(crate)` shared-tile loaders and store helper from
//! [`crate::matmul_tc_kernel`] are reused as-is. The f16 dtype tag on
//! the B-tile loader's `ld.global.f16`/`st.shared.f16` pair is a
//! register-class label, not a value conversion — PTX memory ops do
//! not canonicalize across f16/bf16, so the bytes round-trip bit-perfect.
//!
//! C3 (Sprint 9.1 commit plan): kernel skeleton + module-build host
//! tests + the D4 cvt-free hot-path gate. The public host launch fn
//! `kaio_ops::matmul_tc_bf16` lands at C4 with the first correctness
//! test.

use crate::matmul_tc_kernel::{
    TILE_A_BYTES, TILE_A_ROW_STRIDE_BYTES, TILE_B_BYTES, TILE_B_COL_STRIDE_BYTES,
    emit_mw_load_tile_a_64x16, emit_mw_load_tile_b_16x64, emit_pre_zero_shared_tiles,
    emit_warp_quadrant_store,
};
use half::bf16;
use kaio::prelude::*;
use kaio_core::fragment::{
    FragmentA_BF16, FragmentB_BF16, FragmentC, alloc_c, load_fragment_a_m16n8k16_shared_row_bf16,
    load_fragment_b_m16n8k16_shared_col_bf16,
};
use kaio_core::instr::control::{CmpOp, ControlOp};
use kaio_core::instr::memory::MemoryOp;
use kaio_core::instr::special;
use kaio_core::instr::{ArithOp, MadMode, TensorCoreOp};
use kaio_core::ir::{
    Operand, PtxInstruction, PtxKernel, PtxModule, PtxParam, Register, RegisterAllocator,
    SharedDecl, SpecialReg,
};
use kaio_core::types::PtxType;

// --- mma.sync.m16n8k16 instance shape (used for validate constraint) ---
const BM: u32 = 16; // mma m dim
const BK: u32 = 16; // mma k dim (also K-tile granularity)

// --- Multi-warp block tiling (mirrors matmul_tc_kernel) ---
const BM_BLOCK: u32 = 64;
const BN_BLOCK: u32 = 64;
const WARP_QUAD_M: u32 = 32;
const WARP_QUAD_N: u32 = 32;
const MMAS_PER_WARP_M: u32 = WARP_QUAD_M / BM; // 2
const MMAS_PER_WARP_N: u32 = WARP_QUAD_N / 8; // 4 (BN of mma is 8)
const WARPS_PER_BLOCK: u32 = 4;

// --- Shared tile stride bytes (only used for warp-base offset math here) ---
const BYTES_PER_HALF: u32 = 2; // bf16 storage = 2 bytes per element
const BYTES_PER_F32: u32 = 4;

/// Validate dimension constraints for [`matmul_tc_bf16`] (sibling of
/// [`crate::matmul_tc_kernel::validate_dims_tc`] specialized for bf16
/// inputs).
///
/// Constraints mirror the f16 path:
/// - M, N may be any positive value — edge-tile predication handles
///   non-multiple-of-64 cases.
/// - K must be a multiple of 16 (the mma K-tile is structural; the
///   kernel does not pad K within a K-tile).
///
/// `pub(crate)` until C4 wires the public host API.
pub(crate) fn validate_dims_tc_bf16(
    a: &GpuBuffer<bf16>,
    b: &GpuBuffer<bf16>,
    c: &GpuBuffer<f32>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    if m == 0 || n == 0 || k == 0 {
        return Err(KaioError::InvalidConfig(
            "matmul_tc_bf16 dimensions must be non-zero".to_string(),
        ));
    }
    if !k.is_multiple_of(BK) {
        return Err(KaioError::InvalidConfig(format!(
            "matmul_tc_bf16: K must be a multiple of {BK} (got {k}). The mma.sync.m16n8k16 \
             instance shape requires K-tile size 16; K is not edge-padded inside a K-tile."
        )));
    }
    let mk = (m as usize) * (k as usize);
    let kn = (k as usize) * (n as usize);
    let mn = (m as usize) * (n as usize);
    if a.len() < mk {
        return Err(KaioError::InvalidConfig(format!(
            "matmul_tc_bf16: A buffer too small: need {mk} bf16 ({m}×{k}), got {}",
            a.len()
        )));
    }
    if b.len() < kn {
        return Err(KaioError::InvalidConfig(format!(
            "matmul_tc_bf16: B buffer too small: need {kn} bf16 ({k}×{n}), got {}",
            b.len()
        )));
    }
    if c.len() < mn {
        return Err(KaioError::InvalidConfig(format!(
            "matmul_tc_bf16: C buffer too small: need {mn} f32 ({m}×{n}), got {}",
            c.len()
        )));
    }
    Ok(())
}

/// Per-warp bf16 accumulation of one K-tile: 8 `mma.sync.m16n8k16.bf16`
/// calls in a 2 (m_stripe) × 4 (n_stripe) grid. Sibling of
/// [`crate::matmul_tc_kernel::emit_warp_quadrant_mma`] — same fragment-
/// loader hoist (`warp_group_tig`), same accumulator layout, only the
/// operand fragment types and mma variant differ.
///
/// The accumulator slice `&mut [FragmentC; 8]` is indexed as
/// `accs[m_stripe * MMAS_PER_WARP_N + n_stripe]`.
fn emit_warp_quadrant_mma_bf16(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    tile_a_warp_base_shared: Register, // u32
    tile_b_warp_base_shared: Register, // u32
    warp_lane: Register,               // u32 — tid_x ∈ [0, 32)
    warp_group_tig: (Register, Register),
    accs: &mut [FragmentC; 8],
) {
    // 2 FragmentA_BF16's, one per m_stripe.
    let mut frags_a: [Option<FragmentA_BF16>; MMAS_PER_WARP_M as usize] = [None, None];
    for m_stripe in 0..MMAS_PER_WARP_M {
        let row_off_bytes = m_stripe * BM * TILE_A_ROW_STRIDE_BYTES;
        let a_stripe_base = if row_off_bytes == 0 {
            tile_a_warp_base_shared
        } else {
            let r = alloc.alloc(PtxType::U32);
            kernel.push(PtxInstruction::Arith(ArithOp::Add {
                dst: r,
                lhs: Operand::Reg(tile_a_warp_base_shared),
                rhs: Operand::ImmU32(row_off_bytes),
                ty: PtxType::U32,
            }));
            r
        };
        let frag = load_fragment_a_m16n8k16_shared_row_bf16(
            alloc,
            kernel,
            a_stripe_base,
            warp_lane,
            TILE_A_ROW_STRIDE_BYTES,
            Some(warp_group_tig),
        );
        frags_a[m_stripe as usize] = Some(frag);
    }

    // 4 FragmentB_BF16's, one per n_stripe.
    let mut frags_b: [Option<FragmentB_BF16>; MMAS_PER_WARP_N as usize] = [None, None, None, None];
    for n_stripe in 0..MMAS_PER_WARP_N {
        let col_off_bytes = n_stripe * 8 * TILE_B_COL_STRIDE_BYTES;
        let b_stripe_base = if col_off_bytes == 0 {
            tile_b_warp_base_shared
        } else {
            let r = alloc.alloc(PtxType::U32);
            kernel.push(PtxInstruction::Arith(ArithOp::Add {
                dst: r,
                lhs: Operand::Reg(tile_b_warp_base_shared),
                rhs: Operand::ImmU32(col_off_bytes),
                ty: PtxType::U32,
            }));
            r
        };
        let frag = load_fragment_b_m16n8k16_shared_col_bf16(
            alloc,
            kernel,
            b_stripe_base,
            warp_lane,
            TILE_B_COL_STRIDE_BYTES,
            Some(warp_group_tig),
        );
        frags_b[n_stripe as usize] = Some(frag);
    }

    // 8 bf16 mma's: D[m,n] = A[m] * B[n] + D[m,n]
    for m_stripe in 0..MMAS_PER_WARP_M {
        let frag_a = frags_a[m_stripe as usize].unwrap();
        for n_stripe in 0..MMAS_PER_WARP_N {
            let frag_b = frags_b[n_stripe as usize].unwrap();
            let acc_idx = (m_stripe * MMAS_PER_WARP_N + n_stripe) as usize;
            let frag_d = accs[acc_idx];
            kernel.push(PtxInstruction::TensorCore(TensorCoreOp::MmaSyncBf16 {
                d: frag_d,
                a: frag_a,
                b: frag_b,
                c: frag_d,
            }));
            accs[acc_idx] = frag_d;
        }
    }
}

/// Build the IR module for `matmul_tc_bf16` targeting the given SM.
///
/// `sm` is a PTX target string such as `"sm_89"`. Sub-Ampere targets
/// are legal at build time; `PtxModule::validate()` inside
/// `KaioDevice::load_module` rejects them via `ValidationError::SmTooLow`.
pub(crate) fn build_matmul_tc_bf16_module(sm: &str) -> PtxModule {
    let mut alloc = RegisterAllocator::new();
    let mut kernel = PtxKernel::new("matmul_tc_bf16");

    kernel.add_param(PtxParam::pointer("a_ptr", PtxType::BF16));
    kernel.add_param(PtxParam::pointer("b_ptr", PtxType::BF16));
    kernel.add_param(PtxParam::pointer("d_ptr", PtxType::F32));
    kernel.add_param(PtxParam::scalar("m", PtxType::U32));
    kernel.add_param(PtxParam::scalar("n", PtxType::U32));
    kernel.add_param(PtxParam::scalar("k", PtxType::U32));

    kernel.add_shared_decl(SharedDecl {
        name: "tile_a".to_string(),
        align: 4,
        size_bytes: TILE_A_BYTES,
    });
    kernel.add_shared_decl(SharedDecl {
        name: "tile_b".to_string(),
        align: 4,
        size_bytes: TILE_B_BYTES,
    });

    // --- Load params + cvta to global ---
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
    let r_m = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: r_m,
        param_name: "m".to_string(),
        ty: PtxType::U32,
    }));
    let r_n = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: r_n,
        param_name: "n".to_string(),
        ty: PtxType::U32,
    }));
    let r_k = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: r_k,
        param_name: "k".to_string(),
        ty: PtxType::U32,
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

    // --- Special registers ---
    let (r_tid_x, tid_instr) = special::tid_x(&mut alloc);
    kernel.push(tid_instr);
    let r_warp_id = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_warp_id,
        src: Operand::SpecialReg(SpecialReg::TidY),
        ty: PtxType::U32,
    });

    let r_flat_tid = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mad {
        dst: r_flat_tid,
        a: Operand::Reg(r_warp_id),
        b: Operand::ImmU32(32),
        c: Operand::Reg(r_tid_x),
        ty: PtxType::U32,
        mode: MadMode::Lo,
    }));

    // Sprint 6.7b D10 hoist — group_id and tig once at kernel start.
    let r_hoisted_group_id = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: r_hoisted_group_id,
        lhs: Operand::Reg(r_tid_x),
        rhs: Operand::ImmU32(4),
        ty: PtxType::U32,
    }));
    let r_hoisted_tig = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Rem {
        dst: r_hoisted_tig,
        lhs: Operand::Reg(r_tid_x),
        rhs: Operand::ImmU32(4),
        ty: PtxType::U32,
    }));

    let r_bidx = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_bidx,
        src: Operand::SpecialReg(SpecialReg::CtaidX),
        ty: PtxType::U32,
    });
    let r_bidy = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_bidy,
        src: Operand::SpecialReg(SpecialReg::CtaidY),
        ty: PtxType::U32,
    });

    let r_block_row = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_block_row,
        lhs: Operand::Reg(r_bidy),
        rhs: Operand::ImmU32(BM_BLOCK),
        ty: PtxType::U32,
    }));
    let r_block_col = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_block_col,
        lhs: Operand::Reg(r_bidx),
        rhs: Operand::ImmU32(BN_BLOCK),
        ty: PtxType::U32,
    }));

    let r_k_bytes = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_k_bytes,
        lhs: Operand::Reg(r_k),
        rhs: Operand::ImmU32(BYTES_PER_HALF),
        ty: PtxType::U32,
    }));
    let r_n_bytes = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_n_bytes,
        lhs: Operand::Reg(r_n),
        rhs: Operand::ImmU32(BYTES_PER_HALF),
        ty: PtxType::U32,
    }));
    let r_n_f32_stride = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_n_f32_stride,
        lhs: Operand::Reg(r_n),
        rhs: Operand::ImmU32(BYTES_PER_F32),
        ty: PtxType::U32,
    }));

    let rd_a_row_off = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
        dst: rd_a_row_off,
        lhs: Operand::Reg(r_block_row),
        rhs: Operand::Reg(r_k_bytes),
        src_ty: PtxType::U32,
    }));
    let rd_a_block_base = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: rd_a_block_base,
        lhs: Operand::Reg(rd_a),
        rhs: Operand::Reg(rd_a_row_off),
        ty: PtxType::U64,
    }));

    let r_block_col_bytes = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_block_col_bytes,
        lhs: Operand::Reg(r_block_col),
        rhs: Operand::ImmU32(BYTES_PER_HALF),
        ty: PtxType::U32,
    }));
    let rd_block_col_bytes64 = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Cvt {
        dst: rd_block_col_bytes64,
        src: r_block_col_bytes,
        dst_ty: PtxType::U64,
        src_ty: PtxType::U32,
    });
    let rd_b_block_base = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: rd_b_block_base,
        lhs: Operand::Reg(rd_b),
        rhs: Operand::Reg(rd_block_col_bytes64),
        ty: PtxType::U64,
    }));

    // d_block_base = rd_d + block_row * N * 4 + block_col * 4 (folded in
    // at the store site for warp-quadrant absolute addressing — see
    // emit_warp_quadrant_store invocation below).

    // Shared tile bases.
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

    let r_warp_row_quad = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: r_warp_row_quad,
        lhs: Operand::Reg(r_warp_id),
        rhs: Operand::ImmU32(2),
        ty: PtxType::U32,
    }));
    let r_warp_col_quad = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Rem {
        dst: r_warp_col_quad,
        lhs: Operand::Reg(r_warp_id),
        rhs: Operand::ImmU32(2),
        ty: PtxType::U32,
    }));

    let r_warp_row_off_in_tile_a = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_warp_row_off_in_tile_a,
        lhs: Operand::Reg(r_warp_row_quad),
        rhs: Operand::ImmU32(WARP_QUAD_M * TILE_A_ROW_STRIDE_BYTES),
        ty: PtxType::U32,
    }));
    let r_tile_a_warp = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_tile_a_warp,
        lhs: Operand::Reg(r_tile_a),
        rhs: Operand::Reg(r_warp_row_off_in_tile_a),
        ty: PtxType::U32,
    }));

    let r_warp_col_off_in_tile_b = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_warp_col_off_in_tile_b,
        lhs: Operand::Reg(r_warp_col_quad),
        rhs: Operand::ImmU32(WARP_QUAD_N * TILE_B_COL_STRIDE_BYTES),
        ty: PtxType::U32,
    }));
    let r_tile_b_warp = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_tile_b_warp,
        lhs: Operand::Reg(r_tile_b),
        rhs: Operand::Reg(r_warp_col_off_in_tile_b),
        ty: PtxType::U32,
    }));

    emit_pre_zero_shared_tiles(
        &mut alloc,
        &mut kernel,
        r_tile_a,
        r_tile_b,
        r_flat_tid,
        TILE_A_BYTES,
        TILE_B_BYTES,
    );

    // 8 FragmentC accumulators, zeroed.
    let mut accs: [FragmentC; 8] = [
        alloc_c(&mut alloc),
        alloc_c(&mut alloc),
        alloc_c(&mut alloc),
        alloc_c(&mut alloc),
        alloc_c(&mut alloc),
        alloc_c(&mut alloc),
        alloc_c(&mut alloc),
        alloc_c(&mut alloc),
    ];
    for acc in &accs {
        for r in &acc.regs {
            kernel.push(PtxInstruction::Mov {
                dst: *r,
                src: Operand::ImmF32(0.0),
                ty: PtxType::F32,
            });
        }
    }

    let r_num_k_tiles = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: r_num_k_tiles,
        lhs: Operand::Reg(r_k),
        rhs: Operand::ImmU32(BK),
        ty: PtxType::U32,
    }));

    let r_k_tile = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_k_tile,
        src: Operand::ImmU32(0),
        ty: PtxType::U32,
    });

    kernel.push(PtxInstruction::Label("K_LOOP".to_string()));

    // A tile global source = a_block_base + k_tile * 16 * 2
    let r_k_tile_x_bk_bytes = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_k_tile_x_bk_bytes,
        lhs: Operand::Reg(r_k_tile),
        rhs: Operand::ImmU32(BK * BYTES_PER_HALF),
        ty: PtxType::U32,
    }));
    let rd_k_tile_x_bk_bytes64 = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Cvt {
        dst: rd_k_tile_x_bk_bytes64,
        src: r_k_tile_x_bk_bytes,
        dst_ty: PtxType::U64,
        src_ty: PtxType::U32,
    });
    let rd_a_tile_src = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: rd_a_tile_src,
        lhs: Operand::Reg(rd_a_block_base),
        rhs: Operand::Reg(rd_k_tile_x_bk_bytes64),
        ty: PtxType::U64,
    }));

    // B tile global source = b_block_base + k_tile * 16 * N * 2
    let r_k_tile_x_bk = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_k_tile_x_bk,
        lhs: Operand::Reg(r_k_tile),
        rhs: Operand::ImmU32(BK),
        ty: PtxType::U32,
    }));
    let rd_b_k_row_off = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
        dst: rd_b_k_row_off,
        lhs: Operand::Reg(r_k_tile_x_bk),
        rhs: Operand::Reg(r_n_bytes),
        src_ty: PtxType::U32,
    }));
    let rd_b_tile_src = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: rd_b_tile_src,
        lhs: Operand::Reg(rd_b_block_base),
        rhs: Operand::Reg(rd_b_k_row_off),
        ty: PtxType::U64,
    }));

    emit_mw_load_tile_a_64x16(
        &mut alloc,
        &mut kernel,
        rd_a_tile_src,
        r_tile_a,
        r_flat_tid,
        r_block_row,
        r_m,
        r_k_bytes,
        "",
    );
    emit_mw_load_tile_b_16x64(
        &mut alloc,
        &mut kernel,
        rd_b_tile_src,
        r_tile_b,
        r_flat_tid,
        r_block_col,
        r_n,
        r_n_bytes,
        "",
    );

    kernel.push(PtxInstruction::Control(ControlOp::BarSync {
        barrier_id: 0,
    }));

    emit_warp_quadrant_mma_bf16(
        &mut alloc,
        &mut kernel,
        r_tile_a_warp,
        r_tile_b_warp,
        r_tid_x,
        (r_hoisted_group_id, r_hoisted_tig),
        &mut accs,
    );

    kernel.push(PtxInstruction::Control(ControlOp::BarSync {
        barrier_id: 0,
    }));

    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_k_tile,
        lhs: Operand::Reg(r_k_tile),
        rhs: Operand::ImmU32(1),
        ty: PtxType::U32,
    }));
    let p_loop = alloc.alloc(PtxType::Pred);
    kernel.push(PtxInstruction::Control(ControlOp::SetP {
        dst: p_loop,
        cmp_op: CmpOp::Lt,
        lhs: Operand::Reg(r_k_tile),
        rhs: Operand::Reg(r_num_k_tiles),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Control(ControlOp::BraPred {
        pred: p_loop,
        target: "K_LOOP".to_string(),
        negate: false,
    }));

    // Per-warp output store. As in matmul_tc_kernel, fold warp_quadrant
    // start into r_warp_block_{row,col} so the store helper sees absolute
    // row/col bases relative to rd_d (not d_block_base).
    let r_wq_row_start = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_wq_row_start,
        lhs: Operand::Reg(r_warp_row_quad),
        rhs: Operand::ImmU32(WARP_QUAD_M),
        ty: PtxType::U32,
    }));
    let r_wq_col_start = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_wq_col_start,
        lhs: Operand::Reg(r_warp_col_quad),
        rhs: Operand::ImmU32(WARP_QUAD_N),
        ty: PtxType::U32,
    }));
    let r_warp_block_row = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_warp_block_row,
        lhs: Operand::Reg(r_block_row),
        rhs: Operand::Reg(r_wq_row_start),
        ty: PtxType::U32,
    }));
    let r_warp_block_col = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_warp_block_col,
        lhs: Operand::Reg(r_block_col),
        rhs: Operand::Reg(r_wq_col_start),
        ty: PtxType::U32,
    }));
    emit_warp_quadrant_store(
        &mut alloc,
        &mut kernel,
        rd_d,
        &accs,
        r_tid_x,
        0,
        0,
        r_warp_block_row,
        r_warp_block_col,
        r_m,
        r_n,
        r_n_f32_stride,
    );

    kernel.push(PtxInstruction::Control(ControlOp::Ret));
    kernel.set_registers(alloc.into_allocated());

    let mut module = PtxModule::new(sm);
    module.add_kernel(kernel);
    module
}

#[cfg(test)]
mod tests {
    use super::*;
    use kaio_core::emit::{Emit, PtxWriter};

    fn validate_dims_raw(
        m: u32,
        n: u32,
        k: u32,
        a_len: usize,
        b_len: usize,
        c_len: usize,
    ) -> Result<()> {
        // Mirror validate_dims_tc_bf16's body for length-only tests that
        // don't need a GpuBuffer (host-only).
        if m == 0 || n == 0 || k == 0 {
            return Err(KaioError::InvalidConfig(
                "matmul_tc_bf16 dimensions must be non-zero".to_string(),
            ));
        }
        if !k.is_multiple_of(BK) {
            return Err(KaioError::InvalidConfig(format!(
                "matmul_tc_bf16: K must be a multiple of {BK} (got {k}). The mma.sync.m16n8k16 \
                 instance shape requires K-tile size 16; K is not edge-padded inside a K-tile."
            )));
        }
        let mk = (m as usize) * (k as usize);
        let kn = (k as usize) * (n as usize);
        let mn = (m as usize) * (n as usize);
        if a_len < mk {
            return Err(KaioError::InvalidConfig(format!(
                "matmul_tc_bf16: A buffer too small: need {mk} bf16 ({m}×{k}), got {a_len}"
            )));
        }
        if b_len < kn {
            return Err(KaioError::InvalidConfig(format!(
                "matmul_tc_bf16: B buffer too small: need {kn} bf16 ({k}×{n}), got {b_len}"
            )));
        }
        if c_len < mn {
            return Err(KaioError::InvalidConfig(format!(
                "matmul_tc_bf16: C buffer too small: need {mn} f32 ({m}×{n}), got {c_len}"
            )));
        }
        Ok(())
    }

    #[test]
    fn validate_dims_bf16_rejects_zero() {
        let err = validate_dims_raw(0, 8, 16, 0, 128, 0).unwrap_err();
        assert!(matches!(err, KaioError::InvalidConfig(ref m) if m.contains("non-zero")));
    }

    #[test]
    fn validate_dims_bf16_accepts_non_divisible_m_n() {
        assert!(validate_dims_raw(17, 8, 16, 17 * 16, 16 * 8, 17 * 8).is_ok());
        assert!(validate_dims_raw(7, 5, 16, 7 * 16, 16 * 5, 7 * 5).is_ok());
        assert!(validate_dims_raw(1023, 1023, 1024, 1023 * 1024, 1024 * 1023, 1023 * 1023).is_ok());
    }

    #[test]
    fn validate_dims_bf16_rejects_k_not_multiple_of_16() {
        let err = validate_dims_raw(16, 8, 24, 1000, 1000, 1000).unwrap_err();
        assert!(
            matches!(err, KaioError::InvalidConfig(ref m) if m.contains("K must be a multiple of 16"))
        );
    }

    #[test]
    fn validate_dims_bf16_rejects_buffer_too_small() {
        let err = validate_dims_raw(16, 8, 16, 100, 1000, 1000).unwrap_err();
        assert!(matches!(err, KaioError::InvalidConfig(ref m) if m.contains("A buffer too small")));
    }

    #[test]
    fn validate_dims_bf16_accepts_valid_shapes() {
        assert!(validate_dims_raw(16, 8, 16, 256, 128, 128).is_ok());
        assert!(validate_dims_raw(64, 64, 64, 64 * 64, 64 * 64, 64 * 64).is_ok());
        assert!(validate_dims_raw(128, 8, 16, 128 * 16, 16 * 8, 128 * 8).is_ok());
    }

    fn emit_module_to_string(module: &PtxModule) -> String {
        let mut w = PtxWriter::new();
        module.emit(&mut w).unwrap();
        w.finish()
    }

    #[test]
    fn build_matmul_tc_bf16_module_produces_valid_structure() {
        let module = build_matmul_tc_bf16_module("sm_89");
        let ptx = emit_module_to_string(&module);

        assert!(ptx.contains(".visible .entry matmul_tc_bf16("));
        assert!(
            ptx.contains(".shared .align 4 .b8 tile_a[2048]"),
            "tile_a should be 2048 B (64×16 bf16; storage matches f16 path)"
        );
        assert!(
            ptx.contains(".shared .align 4 .b8 tile_b[2560]"),
            "tile_b should be 2560 B (Sprint 6.7b padded 64×36 + round-up tail)"
        );
        assert!(
            ptx.contains("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"),
            "expected dedicated MmaSyncBf16 mnemonic"
        );
        assert!(
            !ptx.contains("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"),
            "bf16 kernel must not emit any f16-tagged mma.sync"
        );
        assert!(ptx.contains("bar.sync"));

        let mma_count = ptx.matches("mma.sync.aligned.m16n8k16").count();
        assert_eq!(
            mma_count, 8,
            "expected 8 mma.sync emits per K-iter (2 m_stripes × 4 n_stripes per warp)"
        );

        assert!(
            ptx.contains("setp.lt.and.u32"),
            "edge-tile combined predicate emit missing"
        );
        let pred_store_count = ptx.matches("st.global.f32").count();
        assert_eq!(
            pred_store_count, 32,
            "expected 32 predicated st.global.f32 (8 fragments × 4 stores)"
        );
    }

    #[test]
    fn build_matmul_tc_bf16_module_declares_requested_sm_target() {
        let module_70 = build_matmul_tc_bf16_module("sm_70");
        assert!(emit_module_to_string(&module_70).contains(".target sm_70"));
        let module_89 = build_matmul_tc_bf16_module("sm_89");
        assert!(emit_module_to_string(&module_89).contains(".target sm_89"));
    }

    #[test]
    fn matmul_tc_bf16_module_rejects_sm_70_via_validate() {
        use kaio_core::ir::ValidationError;
        let module = build_matmul_tc_bf16_module("sm_70");
        let err = module
            .validate()
            .expect_err("matmul_tc_bf16 module at sm_70 must fail validation");
        match err {
            ValidationError::SmTooLow {
                required,
                actual,
                feature,
            } => {
                assert_eq!(required, 80);
                assert_eq!(actual, 70);
                assert!(
                    feature.contains("mma.sync"),
                    "feature string should name mma.sync; got: {feature}"
                );
            }
        }
    }

    #[test]
    fn matmul_tc_bf16_module_validates_at_sm_80_and_above() {
        for sm in ["sm_80", "sm_89", "sm_90"] {
            let module = build_matmul_tc_bf16_module(sm);
            module
                .validate()
                .unwrap_or_else(|e| panic!("{sm} should validate; got error: {e}"));
        }
    }

    /// Sprint 9.1 D4 resolution gate: the inner K-loop hot path must
    /// contain **no `cvt.*` instructions** between any `ld.shared.b32`
    /// fragment load and the nearest subsequent
    /// `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`. A `cvt`
    /// on that hot path would indicate accidental f16↔bf16 conversion
    /// inserted somewhere on the fragment-load → mma flow — exactly
    /// the precision bug class D4 was written to prevent.
    ///
    /// Acceptable `cvt`s elsewhere: host-side scale lowering, address
    /// widening (`cvt.u32.s32`, `cvt.u64.u32`), output-store cvt. None
    /// of these appear between an `ld.shared.b32` and the next
    /// `mma.sync.bf16` on the hot path.
    #[test]
    fn d4_gate_no_cvt_in_bf16_mma_hot_path() {
        let module = build_matmul_tc_bf16_module("sm_89");
        let ptx = emit_module_to_string(&module);

        // Find the K_LOOP region: from "K_LOOP:" label through the
        // bra.uni.pred back to K_LOOP. Everything inside that span is
        // the inner-loop body that runs once per K-tile.
        let loop_start = ptx
            .find("K_LOOP:")
            .expect("K_LOOP label must appear in emitted PTX");
        // The branch back to K_LOOP closes the loop body. Find the
        // last bra-pred that targets K_LOOP after the label.
        let loop_end_rel = ptx[loop_start..]
            .rfind("K_LOOP")
            .expect("matching back-edge to K_LOOP must exist");
        let loop_body = &ptx[loop_start..loop_start + loop_end_rel];

        // Walk the body once, tracking whether the most recent fragment
        // event was an ld.shared.b32 with no mma.sync since. A cvt that
        // appears in that "between" window fails the gate.
        let mut between_ld_and_mma = false;
        for line in loop_body.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("ld.shared.b32") || trimmed.starts_with("ld.shared.u32") {
                between_ld_and_mma = true;
            } else if trimmed.starts_with("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32") {
                between_ld_and_mma = false;
            } else if between_ld_and_mma && trimmed.starts_with("cvt.") {
                panic!(
                    "D4 GATE FAILED: cvt found between fragment ld.shared.b32 and mma.sync.bf16 \
                     in K_LOOP body — would indicate accidental precision conversion on the bf16 \
                     hot path.\nOffending line: `{line}`\n\n=== K_LOOP body ===\n{loop_body}"
                );
            }
        }
    }
}
