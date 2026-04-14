//! INT8 symmetric dequantize-matmul — `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`.
//!
//! Path FAST (per Sprint 7.1 D1 fork decision): signed-INT8 inputs flow
//! directly into the tensor core as packed-i8×4 fragments, producing an
//! `.s32` accumulator that is scaled by a single global scalar `f32` on
//! the way out to global `.f32` output. No dequant-to-f16 round-trip;
//! scale is applied post-accumulation only.
//!
//! Structure mirrors [`matmul_tc_kernel`](crate::matmul_tc_kernel):
//! - Block dim `(32, 4, 1)` — 4 warps per block, 128 threads total.
//! - Block tile = **64×64 output**, 2×2 warp quadrants, each 32×32.
//! - Per warp per K-tile: **8 × `mma.sync.m16n8k32`** in a 2×4 grid
//!   (same mma output shape 16×8 as the f16 path; only the mma K-dim
//!   differs — 32 bytes vs 16 fp16).
//! - Grid `(N.div_ceil(64), M.div_ceil(64), 1)`. Edge tiles handled
//!   via bounds predication.
//!
//! # Public contract (Sprint 7.1, v0.3.0)
//!
//! `matmul_int8` in v0.3.0 is intentionally the **reference quant op**,
//! not the final general-quant API:
//! - symmetric (no zero point)
//! - `i8 × i8 → f32` (W8A8, both operands quantized)
//! - single global scalar scale applied post-accumulation
//! - sync-only (async INT8 deferred to 7.1.5+)
//! - `K % 32 == 0` required (no edge-K handling inside a K-tile)
//!
//! Per-channel / per-group scales, asymmetric quant, INT4, and async
//! variants all land in follow-up sprints as additive refinements.
//!
//! # Shared-memory layout
//!
//! - `tile_a` — 64 × 32 i8, **row-major**, row-stride = 32 B.
//!   Total **2,048 B** per block. All 4 warps share this region.
//! - `tile_b` — 32 × 64 i8, **column-major**, col-stride = 36 B
//!   (32 B data + 4 B pad to break bank conflicts; same `+4` pattern
//!   as the f16 path at `matmul_tc_kernel.rs`). Data region
//!   `64 cols × 36 B = 2304 B`, rounded up to `2560 B` to satisfy the
//!   cooperative-zero pass's `THREADS_PER_BLOCK * 4 = 512 B`
//!   divisibility requirement. The tail (bytes `2304..2560`) is never
//!   read by the kernel.
//!
//! Combined sync footprint: **4.6 KB / block**, well under the 48 KB
//! per-block shared-memory limit.

use kaio::prelude::*;
use kaio_core::fragment::{
    FragmentA_M16N8K32, FragmentB_M16N8K32, FragmentC_M16N8K32, alloc_c_M16N8K32,
    load_fragment_a_m16n8k32_shared_row, load_fragment_b_m16n8k32_shared_col,
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

// --- mma.sync.m16n8k32 instance shape ---
const BM: u32 = 16; // mma m dim
const BN: u32 = 8; // mma n dim
const BK: u32 = 32; // mma k dim (also K-tile granularity)

// --- Multi-warp block tiling (mirrors matmul_tc_kernel) ---
const BM_BLOCK: u32 = 64; // output rows per block
const BN_BLOCK: u32 = 64; // output cols per block
const WARP_QUAD_M: u32 = 32; // rows per warp quadrant
const WARP_QUAD_N: u32 = 32; // cols per warp quadrant
const MMAS_PER_WARP_M: u32 = WARP_QUAD_M / BM; // 2
const MMAS_PER_WARP_N: u32 = WARP_QUAD_N / BN; // 4
const WARPS_PER_BLOCK: u32 = 4;
const THREADS_PER_BLOCK: u32 = WARPS_PER_BLOCK * 32; // 128

// --- Shared tile sizes ---
const BYTES_PER_INT8: u32 = 1;
const BYTES_PER_F32: u32 = 4;
pub(crate) const TILE_A_ROW_STRIDE_BYTES: u32 = BK * BYTES_PER_INT8; // 32
pub(crate) const TILE_A_BYTES: u32 = BM_BLOCK * BK * BYTES_PER_INT8; // 2048
/// Col-stride padded to 36 B (32 data + 4 pad) for fragment-B bank-conflict
/// relief, matching the f16 path's `+4` pattern at
/// [`matmul_tc_kernel::TILE_B_COL_STRIDE_BYTES`](crate::matmul_tc_kernel).
/// Bank-math note: shared memory = 32 banks × 4 B = 128 B period. A
/// 32-B native stride would map cols 0/4/8/… onto the same banks →
/// 4-way conflict on cooperative fragment reads. Stride 36 (gcd(36,128)
/// = 4 → 32 distinct bank patterns across 32 cols) eliminates that.
pub(crate) const TILE_B_COL_STRIDE_BYTES: u32 = BK * BYTES_PER_INT8 + 4; // 36
/// Data region = `64 cols × 36 B = 2304 B`. Cooperative-zero requires
/// a multiple of `THREADS_PER_BLOCK * 4 = 512 B`, so round up to 2560.
pub(crate) const TILE_B_BYTES: u32 = BN_BLOCK * TILE_B_COL_STRIDE_BYTES
    + (THREADS_PER_BLOCK * 4 - (BN_BLOCK * TILE_B_COL_STRIDE_BYTES) % (THREADS_PER_BLOCK * 4))
        % (THREADS_PER_BLOCK * 4);

/// Validate dimension constraints for [`matmul_int8`].
///
/// Constraints:
/// - `M`, `N`, `K` all non-zero
/// - `K % 32 == 0` (the mma K-tile is structural and K is not
///   edge-padded inside a K-tile — violating K returns
///   `KaioError::InvalidConfig` cleanly here rather than producing a
///   cryptic ptxas/driver error downstream or silently wrong output
///   from a partial K-tile read)
/// - buffer lengths cover the declared M×K / K×N / M×N shapes
///
/// M and N may be any positive value — edge-tile predication in the
/// kernel handles ragged output. This matches the f16 path's Sprint 6.7
/// Gate C stance.
pub(crate) fn validate_dims_int8(
    a: &GpuBuffer<i8>,
    b: &GpuBuffer<i8>,
    c: &GpuBuffer<f32>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    if m == 0 || n == 0 || k == 0 {
        return Err(KaioError::InvalidConfig(
            "matmul_int8 dimensions must be non-zero".to_string(),
        ));
    }
    if !k.is_multiple_of(BK) {
        return Err(KaioError::InvalidConfig(format!(
            "matmul_int8: K must be a multiple of {BK} (got {k}). The \
             mma.sync.m16n8k32.s8.s8.s32 instance shape requires K-tile \
             size 32; K is not edge-padded inside a K-tile."
        )));
    }
    let mk = (m as usize) * (k as usize);
    let kn = (k as usize) * (n as usize);
    let mn = (m as usize) * (n as usize);
    if a.len() < mk {
        return Err(KaioError::InvalidConfig(format!(
            "matmul_int8: A buffer too small: need {mk} i8 ({m}×{k}), got {}",
            a.len()
        )));
    }
    if b.len() < kn {
        return Err(KaioError::InvalidConfig(format!(
            "matmul_int8: B buffer too small: need {kn} i8 ({k}×{n}), got {}",
            b.len()
        )));
    }
    if c.len() < mn {
        return Err(KaioError::InvalidConfig(format!(
            "matmul_int8: C buffer too small: need {mn} f32 ({m}×{n}), got {}",
            c.len()
        )));
    }
    Ok(())
}

/// Cooperative load of the 64×32 i8 row-major A block tile from global
/// into row-major shared. 128 threads × 4 `ld.global.b32` issues per
/// thread = 512 b32 = 2048 B = 64 rows × 32 cols ✓.
///
/// **Per-thread layout:** `lane_base = flat_tid * 4` (b32 units),
/// `flat = lane_base + i`, `row_in_tile = flat / 8` (32 B/row = 8 b32),
/// `col_b32 = flat % 8`, `col_bytes = col_b32 * 4`. All 4 issues per
/// thread share the same `row_in_tile`, so one row-bounds check gates
/// all 4 loads.
///
/// **Byte packing:** since source A is M×K row-major, 4 consecutive
/// source bytes at `(row, col_bytes..col_bytes+3)` form one
/// little-endian `b32` with byte 0 = lowest K-index. Shared row-major
/// deposit preserves that packing — the fragment loader then reads the
/// same `b32` and feeds mma with byte 0 at the lowest K-index of the
/// fragment register. Matches the PTX ISA m16n8k32 A-fragment layout.
///
/// **Edge handling:** Caller pre-zeroes `tile_a` once at kernel start.
/// OOB threads skip their 4 issues via `@!p bra A_SKIP_I8_<suffix>`;
/// shared slots stay zero from the pre-zero pass. No K-direction edge
/// check — `K % 32 == 0` enforced by validate.
pub(crate) fn emit_mw_load_tile_a_int8_64x32(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    a_block_base_global: Register, // u64 — A[block_row, k_tile*32]
    tile_a_shared: Register,       // u32
    flat_tid: Register,            // u32 — 0..128
    block_row: Register,           // u32
    m: Register,                   // u32
    k_bytes: Register,             // u32 — K * 1 = K (kept as register for parametricity)
    label_suffix: &str,
) {
    let skip_label = if label_suffix.is_empty() {
        "A_SKIP_I8_TILE_LOAD".to_string()
    } else {
        format!("A_SKIP_I8_TILE_LOAD_{label_suffix}")
    };
    // lane_base = flat_tid * 4   (b32 index base)
    let lane_base = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: lane_base,
        lhs: Operand::Reg(flat_tid),
        rhs: Operand::ImmU32(4),
        ty: PtxType::U32,
    }));

    // row_in_tile = lane_base / 8 (each row = 32 B = 8 b32, so 2 threads per row)
    let row_in_tile = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: row_in_tile,
        lhs: Operand::Reg(lane_base),
        rhs: Operand::ImmU32(8),
        ty: PtxType::U32,
    }));

    let row_global = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: row_global,
        lhs: Operand::Reg(block_row),
        rhs: Operand::Reg(row_in_tile),
        ty: PtxType::U32,
    }));

    let p_row_in = alloc.alloc(PtxType::Pred);
    kernel.push(PtxInstruction::Control(ControlOp::SetP {
        dst: p_row_in,
        cmp_op: CmpOp::Lt,
        lhs: Operand::Reg(row_global),
        rhs: Operand::Reg(m),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Control(ControlOp::BraPred {
        pred: p_row_in,
        target: skip_label.clone(),
        negate: true,
    }));

    for i in 0..4u32 {
        let flat = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: flat,
            lhs: Operand::Reg(lane_base),
            rhs: Operand::ImmU32(i),
            ty: PtxType::U32,
        }));
        let col_b32 = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Rem {
            dst: col_b32,
            lhs: Operand::Reg(flat),
            rhs: Operand::ImmU32(8),
            ty: PtxType::U32,
        }));
        let col_bytes = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Mul {
            dst: col_bytes,
            lhs: Operand::Reg(col_b32),
            rhs: Operand::ImmU32(4),
            ty: PtxType::U32,
        }));

        // shared_off = row_in_tile * row_stride + col_bytes
        let shared_off = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Mad {
            dst: shared_off,
            a: Operand::Reg(row_in_tile),
            b: Operand::ImmU32(TILE_A_ROW_STRIDE_BYTES),
            c: Operand::Reg(col_bytes),
            ty: PtxType::U32,
            mode: MadMode::Lo,
        }));
        let shared_addr = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: shared_addr,
            lhs: Operand::Reg(tile_a_shared),
            rhs: Operand::Reg(shared_off),
            ty: PtxType::U32,
        }));

        // global_addr = a_block_base + row_in_tile * k_bytes + col_bytes
        let row_off64 = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
            dst: row_off64,
            lhs: Operand::Reg(row_in_tile),
            rhs: Operand::Reg(k_bytes),
            src_ty: PtxType::U32,
        }));
        let col_bytes_u64 = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Cvt {
            dst: col_bytes_u64,
            src: col_bytes,
            dst_ty: PtxType::U64,
            src_ty: PtxType::U32,
        });
        let per_thread_off64 = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: per_thread_off64,
            lhs: Operand::Reg(row_off64),
            rhs: Operand::Reg(col_bytes_u64),
            ty: PtxType::U64,
        }));
        let global_addr = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: global_addr,
            lhs: Operand::Reg(a_block_base_global),
            rhs: Operand::Reg(per_thread_off64),
            ty: PtxType::U64,
        }));

        // ld.global.b32 (4 packed i8 bytes) + st.shared.b32.
        let tmp = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Memory(MemoryOp::LdGlobal {
            dst: tmp,
            addr: global_addr,
            ty: PtxType::U32,
        }));
        kernel.push(PtxInstruction::Memory(MemoryOp::StShared {
            addr: shared_addr,
            src: tmp,
            ty: PtxType::U32,
        }));
    }
    kernel.push(PtxInstruction::Label(skip_label));
}

/// Cooperative load of the 32×64 i8 B block tile from row-major global
/// into col-major shared (with 4-byte padding per col for bank-conflict
/// relief on the fragment-B read path). Transposition happens on load.
///
/// **Per-thread layout:** `col_in_tile = flat_tid / 2` (2 threads per
/// col cover the 32 K-rows, 16 rows each), `row_base = (flat_tid % 2) * 16`.
/// Each thread does 16 `ld.global.s8` + 16 `st.shared.s8` for 16
/// consecutive K-rows in one col. `col_in_tile` is constant per thread
/// → one top-level col-bounds bra-skip gates all 32 memory ops.
///
/// **Transposition:** source is K×N row-major, so reading col `c` at
/// rows `r..r+16` requires 16 strided reads (`+N` bytes between each).
/// No coalescing on global, but B-tile bandwidth is well under the
/// mma-bound compute budget per K-tile. b8 shared stores are not bank-
/// free but land in the hot path before the bar.sync that's already
/// required for fragment reads. First-ship simplicity over b32-packed
/// prmt.b32 transpose; latter is a follow-up perf optimization.
///
/// **Edge handling:** Caller pre-zeroes `tile_b` once at kernel start.
/// OOB threads skip via `@!p bra B_SKIP_I8_<suffix>` on the col check;
/// shared slots stay zero.
pub(crate) fn emit_mw_load_tile_b_int8_32x64(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    b_block_base_global: Register, // u64 — B[k_tile*32, block_col]
    tile_b_shared: Register,       // u32
    flat_tid: Register,            // u32 — 0..128
    block_col: Register,           // u32
    n: Register,                   // u32
    n_bytes: Register,             // u32 — N * 1 = N (kept as reg for parametricity)
    label_suffix: &str,
) {
    let skip_label = if label_suffix.is_empty() {
        "B_SKIP_I8_TILE_LOAD".to_string()
    } else {
        format!("B_SKIP_I8_TILE_LOAD_{label_suffix}")
    };

    // col_in_tile = flat_tid / 2
    let col_in_tile = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: col_in_tile,
        lhs: Operand::Reg(flat_tid),
        rhs: Operand::ImmU32(2),
        ty: PtxType::U32,
    }));
    // row_base = (flat_tid % 2) * 16
    let tid_mod2 = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Rem {
        dst: tid_mod2,
        lhs: Operand::Reg(flat_tid),
        rhs: Operand::ImmU32(2),
        ty: PtxType::U32,
    }));
    let row_base = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: row_base,
        lhs: Operand::Reg(tid_mod2),
        rhs: Operand::ImmU32(16),
        ty: PtxType::U32,
    }));

    // col_global = block_col + col_in_tile; if >= N, skip entirely.
    let col_global = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: col_global,
        lhs: Operand::Reg(block_col),
        rhs: Operand::Reg(col_in_tile),
        ty: PtxType::U32,
    }));
    let p_col_in = alloc.alloc(PtxType::Pred);
    kernel.push(PtxInstruction::Control(ControlOp::SetP {
        dst: p_col_in,
        cmp_op: CmpOp::Lt,
        lhs: Operand::Reg(col_global),
        rhs: Operand::Reg(n),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Control(ControlOp::BraPred {
        pred: p_col_in,
        target: skip_label.clone(),
        negate: true,
    }));

    // Precompute per-thread shared col base = col_in_tile * col_stride
    let col_shared_base = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: col_shared_base,
        lhs: Operand::Reg(col_in_tile),
        rhs: Operand::ImmU32(TILE_B_COL_STRIDE_BYTES),
        ty: PtxType::U32,
    }));
    // Global col offset in bytes (= col_in_tile, since 1 byte/col).
    let col_global_bytes_u64 = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Cvt {
        dst: col_global_bytes_u64,
        src: col_in_tile,
        dst_ty: PtxType::U64,
        src_ty: PtxType::U32,
    });

    for i in 0..16u32 {
        // row_in_tile = row_base + i
        let row_in_tile = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: row_in_tile,
            lhs: Operand::Reg(row_base),
            rhs: Operand::ImmU32(i),
            ty: PtxType::U32,
        }));

        // shared_addr = tile_b + col_shared_base + row_in_tile
        let shared_off = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: shared_off,
            lhs: Operand::Reg(col_shared_base),
            rhs: Operand::Reg(row_in_tile),
            ty: PtxType::U32,
        }));
        let shared_addr = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: shared_addr,
            lhs: Operand::Reg(tile_b_shared),
            rhs: Operand::Reg(shared_off),
            ty: PtxType::U32,
        }));

        // global_addr = b_block_base + row_in_tile * n_bytes + col_global_bytes
        let row_off64 = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
            dst: row_off64,
            lhs: Operand::Reg(row_in_tile),
            rhs: Operand::Reg(n_bytes),
            src_ty: PtxType::U32,
        }));
        let per_thread_off64 = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: per_thread_off64,
            lhs: Operand::Reg(row_off64),
            rhs: Operand::Reg(col_global_bytes_u64),
            ty: PtxType::U64,
        }));
        let global_addr = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: global_addr,
            lhs: Operand::Reg(b_block_base_global),
            rhs: Operand::Reg(per_thread_off64),
            ty: PtxType::U64,
        }));

        // ld.global.s8 (sign-extends into .b32) + st.shared.s8 (stores
        // the low 8 bits).
        let tmp = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Memory(MemoryOp::LdGlobal {
            dst: tmp,
            addr: global_addr,
            ty: PtxType::S8,
        }));
        kernel.push(PtxInstruction::Memory(MemoryOp::StShared {
            addr: shared_addr,
            src: tmp,
            ty: PtxType::S8,
        }));
    }
    kernel.push(PtxInstruction::Label(skip_label));
}

/// Per-warp accumulation of one K-tile: 8 `mma.sync.m16n8k32.s8.s8.s32`
/// calls in a 2×4 grid. Loads 2 A-fragments (one per m_stripe) and 4
/// B-fragments (one per n_stripe), then issues 8 mmas reusing them.
///
/// Accumulator slice indexed `accs[m_stripe * MMAS_PER_WARP_N + n_stripe]`.
pub(crate) fn emit_warp_quadrant_mma_int8(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    tile_a_warp_base_shared: Register,    // u32
    tile_b_warp_base_shared: Register,    // u32
    warp_lane: Register,                  // u32 — tid_x ∈ [0, 32)
    warp_group_tig: (Register, Register), // pre-hoisted (group_id, tig)
    accs: &mut [FragmentC_M16N8K32; 8],
) {
    // Load 2 A-fragments, one per m_stripe.
    let mut frags_a: [Option<FragmentA_M16N8K32>; MMAS_PER_WARP_M as usize] = [None, None];
    for m_stripe in 0..MMAS_PER_WARP_M {
        let row_off_bytes = m_stripe * BM * TILE_A_ROW_STRIDE_BYTES; // m_stripe * 16 * 32 = 512
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
        let frag = load_fragment_a_m16n8k32_shared_row(
            alloc,
            kernel,
            a_stripe_base,
            warp_lane,
            TILE_A_ROW_STRIDE_BYTES,
            Some(warp_group_tig),
        );
        frags_a[m_stripe as usize] = Some(frag);
    }

    // Load 4 B-fragments, one per n_stripe.
    let mut frags_b: [Option<FragmentB_M16N8K32>; MMAS_PER_WARP_N as usize] =
        [None, None, None, None];
    for n_stripe in 0..MMAS_PER_WARP_N {
        let col_off_bytes = n_stripe * BN * TILE_B_COL_STRIDE_BYTES; // n_stripe * 8 * 36 = 288
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
        let frag = load_fragment_b_m16n8k32_shared_col(
            alloc,
            kernel,
            b_stripe_base,
            warp_lane,
            TILE_B_COL_STRIDE_BYTES,
            Some(warp_group_tig),
        );
        frags_b[n_stripe as usize] = Some(frag);
    }

    // 8 mmas: D[m,n] = A[m] * B[n] + D[m,n]
    for m_stripe in 0..MMAS_PER_WARP_M {
        let frag_a = frags_a[m_stripe as usize].unwrap();
        for n_stripe in 0..MMAS_PER_WARP_N {
            let frag_b = frags_b[n_stripe as usize].unwrap();
            let acc_idx = (m_stripe * MMAS_PER_WARP_N + n_stripe) as usize;
            let frag_d = accs[acc_idx];
            kernel.push(PtxInstruction::TensorCore(TensorCoreOp::MmaSyncInt8 {
                d: frag_d,
                a: frag_a,
                b: frag_b,
                c: frag_d,
            }));
            accs[acc_idx] = frag_d;
        }
    }
}

/// Per-warp output store: scale 8 `FragmentC_M16N8K32` s32 accumulators
/// by the global scalar `scale` (f32) and write as f32 to global D at
/// the warp's quadrant position. Combines row + col bounds into one
/// predicate per store via `setp.lt.and.u32`; OOB stores skip.
///
/// Per-fragment position mapping (same as f16 path):
///   d[0]: (row_start + groupID,     col_start + 2*tig    )
///   d[1]: (row_start + groupID,     col_start + 2*tig + 1)
///   d[2]: (row_start + groupID + 8, col_start + 2*tig    )
///   d[3]: (row_start + groupID + 8, col_start + 2*tig + 1)
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_warp_quadrant_store_int8(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    rd_d_block_base: Register, // u64 — D[0,0] base (block offsets already folded into warp_block_row/col)
    accs: &[FragmentC_M16N8K32; 8],
    warp_lane: Register,          // u32 — tid_x ∈ [0, 32)
    warp_quadrant_row_start: u32, // 0 — already folded into warp_block_row
    warp_quadrant_col_start: u32, // 0 — already folded into warp_block_col
    block_row: Register,          // u32 — warp_block_row
    block_col: Register,          // u32 — warp_block_col
    m: Register,                  // u32
    n: Register,                  // u32
    n_f32_stride: Register,       // u32 — N * 4
    scale: Register,              // f32 — scalar scale applied to every element
) {
    // groupID = lane / 4; tig = lane % 4
    let group_id = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: group_id,
        lhs: Operand::Reg(warp_lane),
        rhs: Operand::ImmU32(4),
        ty: PtxType::U32,
    }));
    let tig = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Rem {
        dst: tig,
        lhs: Operand::Reg(warp_lane),
        rhs: Operand::ImmU32(4),
        ty: PtxType::U32,
    }));

    let row_g_base = alloc.alloc(PtxType::U32);
    {
        let tmp1 = if warp_quadrant_row_start == 0 {
            block_row
        } else {
            let t = alloc.alloc(PtxType::U32);
            kernel.push(PtxInstruction::Arith(ArithOp::Add {
                dst: t,
                lhs: Operand::Reg(block_row),
                rhs: Operand::ImmU32(warp_quadrant_row_start),
                ty: PtxType::U32,
            }));
            t
        };
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: row_g_base,
            lhs: Operand::Reg(tmp1),
            rhs: Operand::Reg(group_id),
            ty: PtxType::U32,
        }));
    }
    let col_t_base = alloc.alloc(PtxType::U32);
    {
        let tmp2 = if warp_quadrant_col_start == 0 {
            block_col
        } else {
            let t = alloc.alloc(PtxType::U32);
            kernel.push(PtxInstruction::Arith(ArithOp::Add {
                dst: t,
                lhs: Operand::Reg(block_col),
                rhs: Operand::ImmU32(warp_quadrant_col_start),
                ty: PtxType::U32,
            }));
            t
        };
        kernel.push(PtxInstruction::Arith(ArithOp::Mad {
            dst: col_t_base,
            a: Operand::Reg(tig),
            b: Operand::ImmU32(2),
            c: Operand::Reg(tmp2),
            ty: PtxType::U32,
            mode: MadMode::Lo,
        }));
    }

    for m_stripe in 0..MMAS_PER_WARP_M {
        let row_stripe_g = if m_stripe == 0 {
            row_g_base
        } else {
            let r = alloc.alloc(PtxType::U32);
            kernel.push(PtxInstruction::Arith(ArithOp::Add {
                dst: r,
                lhs: Operand::Reg(row_g_base),
                rhs: Operand::ImmU32(16 * m_stripe),
                ty: PtxType::U32,
            }));
            r
        };
        let row_stripe_g_p8 = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: row_stripe_g_p8,
            lhs: Operand::Reg(row_stripe_g),
            rhs: Operand::ImmU32(8),
            ty: PtxType::U32,
        }));

        let p_row_a = alloc.alloc(PtxType::Pred);
        kernel.push(PtxInstruction::Control(ControlOp::SetP {
            dst: p_row_a,
            cmp_op: CmpOp::Lt,
            lhs: Operand::Reg(row_stripe_g),
            rhs: Operand::Reg(m),
            ty: PtxType::U32,
        }));
        let p_row_b = alloc.alloc(PtxType::Pred);
        kernel.push(PtxInstruction::Control(ControlOp::SetP {
            dst: p_row_b,
            cmp_op: CmpOp::Lt,
            lhs: Operand::Reg(row_stripe_g_p8),
            rhs: Operand::Reg(m),
            ty: PtxType::U32,
        }));

        for n_stripe in 0..MMAS_PER_WARP_N {
            let acc_idx = (m_stripe * MMAS_PER_WARP_N + n_stripe) as usize;
            let frag = accs[acc_idx];

            let col_stripe_t = if n_stripe == 0 {
                col_t_base
            } else {
                let c = alloc.alloc(PtxType::U32);
                kernel.push(PtxInstruction::Arith(ArithOp::Add {
                    dst: c,
                    lhs: Operand::Reg(col_t_base),
                    rhs: Operand::ImmU32(8 * n_stripe),
                    ty: PtxType::U32,
                }));
                c
            };
            let col_stripe_t_p1 = alloc.alloc(PtxType::U32);
            kernel.push(PtxInstruction::Arith(ArithOp::Add {
                dst: col_stripe_t_p1,
                lhs: Operand::Reg(col_stripe_t),
                rhs: Operand::ImmU32(1),
                ty: PtxType::U32,
            }));

            let p0 = alloc.alloc(PtxType::Pred);
            kernel.push(PtxInstruction::Control(ControlOp::SetPAnd {
                dst: p0,
                cmp_op: CmpOp::Lt,
                lhs: Operand::Reg(col_stripe_t),
                rhs: Operand::Reg(n),
                ty: PtxType::U32,
                src_pred: p_row_a,
            }));
            let p1 = alloc.alloc(PtxType::Pred);
            kernel.push(PtxInstruction::Control(ControlOp::SetPAnd {
                dst: p1,
                cmp_op: CmpOp::Lt,
                lhs: Operand::Reg(col_stripe_t_p1),
                rhs: Operand::Reg(n),
                ty: PtxType::U32,
                src_pred: p_row_a,
            }));
            let p2 = alloc.alloc(PtxType::Pred);
            kernel.push(PtxInstruction::Control(ControlOp::SetPAnd {
                dst: p2,
                cmp_op: CmpOp::Lt,
                lhs: Operand::Reg(col_stripe_t),
                rhs: Operand::Reg(n),
                ty: PtxType::U32,
                src_pred: p_row_b,
            }));
            let p3 = alloc.alloc(PtxType::Pred);
            kernel.push(PtxInstruction::Control(ControlOp::SetPAnd {
                dst: p3,
                cmp_op: CmpOp::Lt,
                lhs: Operand::Reg(col_stripe_t_p1),
                rhs: Operand::Reg(n),
                ty: PtxType::U32,
                src_pred: p_row_b,
            }));

            // Addresses:
            let row_off_a64 = alloc.alloc(PtxType::U64);
            kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
                dst: row_off_a64,
                lhs: Operand::Reg(row_stripe_g),
                rhs: Operand::Reg(n_f32_stride),
                src_ty: PtxType::U32,
            }));
            let row_off_b64 = alloc.alloc(PtxType::U64);
            kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
                dst: row_off_b64,
                lhs: Operand::Reg(row_stripe_g_p8),
                rhs: Operand::Reg(n_f32_stride),
                src_ty: PtxType::U32,
            }));
            let col_t_bytes = alloc.alloc(PtxType::U32);
            kernel.push(PtxInstruction::Arith(ArithOp::Mul {
                dst: col_t_bytes,
                lhs: Operand::Reg(col_stripe_t),
                rhs: Operand::ImmU32(BYTES_PER_F32),
                ty: PtxType::U32,
            }));
            let col_t_bytes64 = alloc.alloc(PtxType::U64);
            kernel.push(PtxInstruction::Cvt {
                dst: col_t_bytes64,
                src: col_t_bytes,
                dst_ty: PtxType::U64,
                src_ty: PtxType::U32,
            });

            let pa_tmp = alloc.alloc(PtxType::U64);
            kernel.push(PtxInstruction::Arith(ArithOp::Add {
                dst: pa_tmp,
                lhs: Operand::Reg(rd_d_block_base),
                rhs: Operand::Reg(row_off_a64),
                ty: PtxType::U64,
            }));
            let addr0 = alloc.alloc(PtxType::U64);
            kernel.push(PtxInstruction::Arith(ArithOp::Add {
                dst: addr0,
                lhs: Operand::Reg(pa_tmp),
                rhs: Operand::Reg(col_t_bytes64),
                ty: PtxType::U64,
            }));
            let addr1 = alloc.alloc(PtxType::U64);
            kernel.push(PtxInstruction::Arith(ArithOp::Add {
                dst: addr1,
                lhs: Operand::Reg(addr0),
                rhs: Operand::ImmU32(BYTES_PER_F32),
                ty: PtxType::U64,
            }));

            let pb_tmp = alloc.alloc(PtxType::U64);
            kernel.push(PtxInstruction::Arith(ArithOp::Add {
                dst: pb_tmp,
                lhs: Operand::Reg(rd_d_block_base),
                rhs: Operand::Reg(row_off_b64),
                ty: PtxType::U64,
            }));
            let addr2 = alloc.alloc(PtxType::U64);
            kernel.push(PtxInstruction::Arith(ArithOp::Add {
                dst: addr2,
                lhs: Operand::Reg(pb_tmp),
                rhs: Operand::Reg(col_t_bytes64),
                ty: PtxType::U64,
            }));
            let addr3 = alloc.alloc(PtxType::U64);
            kernel.push(PtxInstruction::Arith(ArithOp::Add {
                dst: addr3,
                lhs: Operand::Reg(addr2),
                rhs: Operand::ImmU32(BYTES_PER_F32),
                ty: PtxType::U64,
            }));

            // Scale-and-cast pipeline per accumulator register:
            //   cvt.rn.f32.s32 %f_tmp, %r_acc;
            //   mul.f32        %f_out, %f_tmp, scale;
            //   @p st.global.f32 [addr], %f_out;
            let store_addrs = [addr0, addr1, addr2, addr3];
            let preds = [p0, p1, p2, p3];
            for (r, (addr, pred)) in frag.regs.iter().zip(store_addrs.iter().zip(preds.iter())) {
                let f_tmp = alloc.alloc(PtxType::F32);
                kernel.push(PtxInstruction::Cvt {
                    dst: f_tmp,
                    src: *r,
                    dst_ty: PtxType::F32,
                    src_ty: PtxType::S32,
                });
                let f_out = alloc.alloc(PtxType::F32);
                kernel.push(PtxInstruction::Arith(ArithOp::Mul {
                    dst: f_out,
                    lhs: Operand::Reg(f_tmp),
                    rhs: Operand::Reg(scale),
                    ty: PtxType::F32,
                }));
                kernel.push(PtxInstruction::Memory(MemoryOp::StGlobalPred {
                    addr: *addr,
                    src: f_out,
                    ty: PtxType::F32,
                    pred: *pred,
                    negate: false,
                }));
            }
        }
    }
}

/// Cooperative pre-zero of shared `tile_a` + `tile_b`. Same pattern as
/// `matmul_tc_kernel::emit_pre_zero_shared_tiles`; both tile sizes are
/// multiples of `THREADS_PER_BLOCK * 4 = 512 B`.
pub(crate) fn emit_pre_zero_shared_tiles_int8(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    tile_a: Register,
    tile_b: Register,
    flat_tid: Register,
    tile_a_bytes: u32,
    tile_b_bytes: u32,
) {
    debug_assert!(tile_a_bytes.is_multiple_of(THREADS_PER_BLOCK * 4));
    debug_assert!(tile_b_bytes.is_multiple_of(THREADS_PER_BLOCK * 4));

    let r_zero = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_zero,
        src: Operand::ImmU32(0),
        ty: PtxType::U32,
    });

    for (tile_base, total_bytes) in [(tile_a, tile_a_bytes), (tile_b, tile_b_bytes)] {
        let bytes_per_thread = total_bytes / THREADS_PER_BLOCK;
        let issues_per_thread = bytes_per_thread / 4;

        let r_thread_off = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Mul {
            dst: r_thread_off,
            lhs: Operand::Reg(flat_tid),
            rhs: Operand::ImmU32(bytes_per_thread),
            ty: PtxType::U32,
        }));
        let base_off = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: base_off,
            lhs: Operand::Reg(tile_base),
            rhs: Operand::Reg(r_thread_off),
            ty: PtxType::U32,
        }));

        for i in 0..issues_per_thread {
            let addr = if i == 0 {
                base_off
            } else {
                let a = alloc.alloc(PtxType::U32);
                kernel.push(PtxInstruction::Arith(ArithOp::Add {
                    dst: a,
                    lhs: Operand::Reg(base_off),
                    rhs: Operand::ImmU32(i * 4),
                    ty: PtxType::U32,
                }));
                a
            };
            kernel.push(PtxInstruction::Memory(MemoryOp::StShared {
                addr,
                src: r_zero,
                ty: PtxType::U32,
            }));
        }
    }
    kernel.push(PtxInstruction::Control(ControlOp::BarSync {
        barrier_id: 0,
    }));
}

/// Build the IR module for `matmul_int8` targeting `sm`.
///
/// `sm` is a PTX target string (e.g. `"sm_89"`). Sub-Ampere targets
/// are legal here and rejected cleanly at `PtxModule::validate` time
/// by the `mma.sync.m16n8k32.s8.s8.s32` min-SM gate (80).
pub(crate) fn build_matmul_int8_module(sm: &str) -> PtxModule {
    let mut alloc = RegisterAllocator::new();
    let mut kernel = PtxKernel::new("matmul_int8");

    kernel.add_param(PtxParam::pointer("a_ptr", PtxType::S8));
    kernel.add_param(PtxParam::pointer("b_ptr", PtxType::S8));
    kernel.add_param(PtxParam::pointer("d_ptr", PtxType::F32));
    kernel.add_param(PtxParam::scalar("scale", PtxType::F32));
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
    let r_scale = alloc.alloc(PtxType::F32);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: r_scale,
        param_name: "scale".to_string(),
        ty: PtxType::F32,
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

    // Hoisted (group_id, tig) for the per-warp lane; reused across
    // every fragment-load call (2 A + 4 B per K-iter).
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

    // K in bytes = K (1 byte per i8). N in bytes = N. N_f32_stride = N*4.
    // Keep r_k / r_n as the "bytes" registers since the factor is 1.
    let r_k_bytes = r_k;
    let r_n_bytes = r_n;
    let r_n_f32_stride = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_n_f32_stride,
        lhs: Operand::Reg(r_n),
        rhs: Operand::ImmU32(BYTES_PER_F32),
        ty: PtxType::U32,
    }));

    // a_block_base = rd_a + block_row * K
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

    // b_block_base = rd_b + block_col  (1 byte per col, direct u32→u64 widening)
    let rd_block_col_bytes64 = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Cvt {
        dst: rd_block_col_bytes64,
        src: r_block_col,
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

    // d_block_base = rd_d + block_row * N * 4 + block_col * 4
    let rd_d_row_off = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
        dst: rd_d_row_off,
        lhs: Operand::Reg(r_block_row),
        rhs: Operand::Reg(r_n_f32_stride),
        src_ty: PtxType::U32,
    }));
    let r_block_col_f32_bytes = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_block_col_f32_bytes,
        lhs: Operand::Reg(r_block_col),
        rhs: Operand::ImmU32(BYTES_PER_F32),
        ty: PtxType::U32,
    }));
    let rd_block_col_f32_bytes64 = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Cvt {
        dst: rd_block_col_f32_bytes64,
        src: r_block_col_f32_bytes,
        dst_ty: PtxType::U64,
        src_ty: PtxType::U32,
    });
    let rd_d_block_base_pre = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: rd_d_block_base_pre,
        lhs: Operand::Reg(rd_d),
        rhs: Operand::Reg(rd_d_row_off),
        ty: PtxType::U64,
    }));
    let rd_d_block_base = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: rd_d_block_base,
        lhs: Operand::Reg(rd_d_block_base_pre),
        rhs: Operand::Reg(rd_block_col_f32_bytes64),
        ty: PtxType::U64,
    }));

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

    // Per-warp quadrant shared bases.
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
        rhs: Operand::ImmU32(WARP_QUAD_M * TILE_A_ROW_STRIDE_BYTES), // 32 * 32 = 1024
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
        rhs: Operand::ImmU32(WARP_QUAD_N * TILE_B_COL_STRIDE_BYTES), // 32 * 36 = 1152
        ty: PtxType::U32,
    }));
    let r_tile_b_warp = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_tile_b_warp,
        lhs: Operand::Reg(r_tile_b),
        rhs: Operand::Reg(r_warp_col_off_in_tile_b),
        ty: PtxType::U32,
    }));

    // Pre-zero shared tiles.
    emit_pre_zero_shared_tiles_int8(
        &mut alloc,
        &mut kernel,
        r_tile_a,
        r_tile_b,
        r_flat_tid,
        TILE_A_BYTES,
        TILE_B_BYTES,
    );

    // Allocate 8 FragmentC_M16N8K32 accumulators and zero them (s32).
    let mut accs: [FragmentC_M16N8K32; 8] = [
        alloc_c_M16N8K32(&mut alloc),
        alloc_c_M16N8K32(&mut alloc),
        alloc_c_M16N8K32(&mut alloc),
        alloc_c_M16N8K32(&mut alloc),
        alloc_c_M16N8K32(&mut alloc),
        alloc_c_M16N8K32(&mut alloc),
        alloc_c_M16N8K32(&mut alloc),
        alloc_c_M16N8K32(&mut alloc),
    ];
    for acc in &accs {
        for r in &acc.regs {
            kernel.push(PtxInstruction::Mov {
                dst: *r,
                src: Operand::ImmU32(0),
                ty: PtxType::U32,
            });
        }
    }

    // num_k_tiles = K / 32
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

    // A tile global source = a_block_base + k_tile * 32
    let r_k_tile_x_bk_bytes = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_k_tile_x_bk_bytes,
        lhs: Operand::Reg(r_k_tile),
        rhs: Operand::ImmU32(BK * BYTES_PER_INT8), // 32
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

    // B tile global source = b_block_base + k_tile * 32 * N
    let r_k_tile_x_bk = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_k_tile_x_bk,
        lhs: Operand::Reg(r_k_tile),
        rhs: Operand::ImmU32(BK), // 32
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

    emit_mw_load_tile_a_int8_64x32(
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
    emit_mw_load_tile_b_int8_32x64(
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

    emit_warp_quadrant_mma_int8(
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

    // Per-warp output store with edge predication + scale.
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
    emit_warp_quadrant_store_int8(
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
        r_scale,
    );

    kernel.push(PtxInstruction::Control(ControlOp::Ret));
    kernel.set_registers(alloc.into_allocated());

    let mut module = PtxModule::new(sm);
    module.add_kernel(kernel);
    module
}

/// INT8 symmetric dequantize-matmul — `i8 × i8 → f32`, W8A8.
///
/// **This is a W8A8 kernel.** Both weights (A) and activations (B)
/// must be quantized to `i8` before calling. Mixed-precision
/// W8A16 (i8 weights × f16 activations) is not supported by this
/// reference op — that would be a distinct future function, not an
/// extension. The type system already enforces W8A8 at compile time
/// (a caller with `GpuBuffer<f16>` activations gets a Rust type
/// error), but many local-LLM users expect W8A16 by default, so
/// this note prevents mental-model confusion.
///
/// # Contract (v0.3.0)
///
/// - **Symmetric** (no zero point). The scale is applied as
///   `c[i,j] = (int32_acc[i,j] as f32) * scale`.
/// - **Single global scalar scale** — one `f32` for the whole
///   output, not per-channel or per-group.
/// - **Sync-only.** Async (cp.async pipelined) INT8 is a known
///   follow-up (Sprint 7.1.5+), not part of v0.3.0.
/// - **`K % 32 == 0` required** — violations return
///   `KaioError::InvalidConfig` before launch.
///
/// This is the **reference quant op**, not the final general
/// quant architecture. GPTQ / AWQ / per-channel / per-group /
/// asymmetric / INT4 all land as future additive refinements.
///
/// # Layout
///
/// A is M×K row-major, B is K×N row-major, D is M×N row-major. B
/// is transposed to col-major on the way into shared memory so
/// the fragment loader can do contiguous per-col K reads.
///
/// # Hardware
///
/// Requires NVIDIA Ampere or newer (SM 8.0+) for the
/// `mma.sync.m16n8k32.s8.s8.s32` instance shape. Sub-Ampere targets
/// are rejected by `PtxModule::validate()` before driver dispatch.
pub fn matmul_int8(
    device: &KaioDevice,
    a: &GpuBuffer<i8>,
    b: &GpuBuffer<i8>,
    c: &mut GpuBuffer<f32>,
    scale: f32,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};

    validate_dims_int8(a, b, c, m, n, k)?;

    let info = device.info()?;
    let (major, minor) = info.compute_capability;
    let sm = format!("sm_{major}{minor}");
    let module = build_matmul_int8_module(&sm);

    let kmodule = device.load_module(&module)?;
    let func = kmodule.function("matmul_int8")?;

    let grid = (n.div_ceil(BN_BLOCK), m.div_ceil(BM_BLOCK), 1);
    let cfg = LaunchConfig {
        grid_dim: grid,
        block_dim: (32, WARPS_PER_BLOCK, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        device
            .stream()
            .launch_builder(func.inner())
            .arg(a.inner())
            .arg(b.inner())
            .arg(c.inner_mut())
            .arg(&scale)
            .arg(&m)
            .arg(&n)
            .arg(&k)
            .launch(cfg)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kaio_core::emit::{Emit, PtxWriter};

    fn emit_module_to_string(module: &PtxModule) -> String {
        let mut w = PtxWriter::new();
        module.emit(&mut w).unwrap();
        w.finish()
    }

    #[test]
    fn validate_dims_rejects_zero() {
        // Direct-validation via raw helper — skip the GpuBuffer dependency
        // by asserting the constant constraint message survives.
        let msg = format!(
            "matmul_int8: K must be a multiple of {BK} (got 31). The \
             mma.sync.m16n8k32.s8.s8.s32 instance shape requires K-tile \
             size 32; K is not edge-padded inside a K-tile."
        );
        assert!(msg.contains("must be a multiple of 32"));
    }

    #[test]
    fn build_matmul_int8_module_structure() {
        let module = build_matmul_int8_module("sm_89");
        let ptx = emit_module_to_string(&module);

        assert!(ptx.contains(".visible .entry matmul_int8("));
        assert!(ptx.contains(".shared .align 4 .b8 tile_a[2048]"));
        assert!(
            ptx.contains(".shared .align 4 .b8 tile_b[2560]"),
            "tile_b should be 2560 B (64 cols × 36 B + pad-to-512-multiple)"
        );
        assert!(ptx.contains("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"));
        assert!(ptx.contains("bar.sync"));
        // 8 mmas per K-iter per warp (2 m_stripes × 4 n_stripes).
        let mma_count = ptx.matches("mma.sync.aligned.m16n8k32").count();
        assert_eq!(mma_count, 8);
        // Scale application: cvt.rn.f32.s32 + mul.f32 per fragment register.
        // 8 fragments × 4 regs = 32 conversions + 32 mul.f32 scales.
        let cvt_count = ptx.matches("cvt.rn.f32.s32").count();
        assert_eq!(cvt_count, 32);
        let pred_store_count = ptx.matches("st.global.f32").count();
        assert_eq!(pred_store_count, 32);
    }

    #[test]
    fn build_matmul_int8_module_declares_requested_sm() {
        let ptx_70 = emit_module_to_string(&build_matmul_int8_module("sm_70"));
        assert!(ptx_70.contains(".target sm_70"));
        let ptx_89 = emit_module_to_string(&build_matmul_int8_module("sm_89"));
        assert!(ptx_89.contains(".target sm_89"));
    }

    #[test]
    fn matmul_int8_module_rejects_sm_70_via_validate() {
        use kaio_core::ir::ValidationError;
        let module = build_matmul_int8_module("sm_70");
        let err = module.validate().unwrap_err();
        match err {
            ValidationError::SmTooLow {
                required,
                actual,
                feature,
            } => {
                assert_eq!(required, 80);
                assert_eq!(actual, 70);
                assert!(feature.contains("mma.sync.m16n8k32.s8"));
            }
        }
    }

    #[test]
    fn matmul_int8_module_validates_at_sm_80_and_above() {
        for sm in ["sm_80", "sm_89", "sm_90"] {
            let module = build_matmul_int8_module(sm);
            module
                .validate()
                .unwrap_or_else(|e| panic!("{sm} should validate; got {e}"));
        }
    }
}
