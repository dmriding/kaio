//! Fused tri-output INT4 QKV projection — W4A16 (f16 activations,
//! packed signed-INT4 weights, f16 group scales).
//!
//! Sprint 7.3 contingent deliverable. Tri-output extension of
//! [`matmul_int4`][crate::matmul_int4] — same packing convention
//! (8 signed `s4` per `u32`, K-contiguous), same group-scale semantics
//! (`group_size = 128` fixed, f16 scales `[num_groups, N]` row-major),
//! same DEQUANT-F16 pipeline (`shr.s32` sign-extend → `cvt.rn.f32.s32`
//! → `cvt.rn.f16.f32` → scale-fold `mul.f16` → `MovPack` → fragment B).
//!
//! One kernel launch produces three f16 outputs (Q, K, V) from three
//! packed weight tensors and three separate scale tensors, sharing a
//! single load of X. `attention_tc`-ready outputs.
//!
//! # Contingent ship
//!
//! Per the Sprint 7.3 plan, INT4 is a **contingent second deliverable**
//! shipped after the `qkv_project_int8` MVS ship point at D4 if the
//! D2.5 register budget and D5/D6/D7 correctness gates stay clean.
//! Sprint may also ship INT8-only via rollback #5 if INT4 exposes
//! unexpected complexity in the tri-output + group-scale-reload
//! context.
//!
//! # Unified mma path with INT8
//!
//! Both `qkv_project_int4` and [`qkv_project_int8`][super::qkv_project_int8_kernel]
//! target the same `mma.sync.m16n8k16.f16.f16.f32` shape with
//! `K_TILE_SHARED = 16`. The shared store-out helper
//! ([`emit_store_fragment_c_f32_to_f16_packed`][super::store_out::emit_store_fragment_c_f32_to_f16_packed]
//! once wired) casts the three f32 fragment-C banks to f16 on store-out;
//! INT4 passes `scale = None` because the group scale is folded into
//! fragment B during dequant, not applied post-accumulation.
//!
//! # Register budget and tile shape
//!
//! Same envelope as INT8 (see `qkv_project_int8_kernel`): 4-warp block
//! with 64×32 output tile, 3 f32 fragment-C banks persistent across
//! the K-loop (48 regs/lane for accumulators). D2.5 checkpoint runs
//! the minimal skeleton through `ptxas -v` before the full kernel body
//! is authored.

use kaio::prelude::*;
use kaio_core::instr::ArithOp;
use kaio_core::instr::control::{CmpOp, ControlOp};
use kaio_core::instr::memory::MemoryOp;
use kaio_core::ir::{Operand, PtxInstruction, PtxKernel, Register, RegisterAllocator};
use kaio_core::types::PtxType;

// --- mma.sync.m16n8k16 instance shape (f16 inputs, f32 accumulator) ---
// Matches qkv_project_int8 and matmul_int4.
#[allow(dead_code)] // wired up in D5
pub(crate) const BM: u32 = 16; // mma m dim
#[allow(dead_code)] // wired up in D5
pub(crate) const BN: u32 = 8; // mma n dim

// --- Multi-warp block tiling (Rollback #1 mirrors qkv_project_int8) ---
//
// INT4 has identical fragment-C live state to INT8 (3 grids × 4 sub-tiles ×
// 4 f32 = 48 regs/lane at the original BN_BLOCK=32) plus extra register
// pressure from the group-scale reload + nibble-extract dequant chain.
// INT8 already overshot the 64-reg cliff at the original tile (80 regs);
// INT4 would be worse. Apply Rollback #1 preemptively: drop
// `MMAS_PER_WARP_N` 2 → 1, output tile becomes 64×16 per block. Same
// economic argument: more blocks launch, X-reuse still favors fusion.
#[allow(dead_code)] // wired up in D5
pub(crate) const BM_BLOCK: u32 = 64;
#[allow(dead_code)] // wired up in D5
pub(crate) const BN_BLOCK: u32 = 16; // Rollback #1 (was 32)
#[allow(dead_code)] // wired up in D5
pub(crate) const WARP_QUAD_M: u32 = 32;
#[allow(dead_code)] // wired up in D5
pub(crate) const WARP_QUAD_N: u32 = 8; // Rollback #1 (was 16)
#[allow(dead_code)] // wired up in D5
pub(crate) const MMAS_PER_WARP_M: u32 = WARP_QUAD_M / BM; // 2
#[allow(dead_code)] // wired up in D5
pub(crate) const MMAS_PER_WARP_N: u32 = WARP_QUAD_N / BN; // 1 (Rollback #1)
#[allow(dead_code)] // wired up in D5
pub(crate) const WARPS_PER_BLOCK: u32 = 4;
#[allow(dead_code)] // wired up in D5
pub(crate) const THREADS_PER_BLOCK: u32 = WARPS_PER_BLOCK * 32; // 128

// --- INT4 packing + group scales (mirrored from matmul_int4) ---
#[allow(dead_code)] // wired up in D5
pub(crate) const NIBBLES_PER_U32: u32 = 8;
#[allow(dead_code)] // wired up in D5
pub(crate) const GROUP_SIZE: u32 = 128;

// --- K-tile granularity (unified across both QKV variants) ---
#[allow(dead_code)] // wired up in D5
pub(crate) const K_TILE_SHARED: u32 = 16;

// --- Group-scale reload cadence ---
// group_idx = k_tile / K_TILE_GROUP_RATIO; reload every time k_tile crosses a group boundary.
#[allow(dead_code)] // wired up in D5
pub(crate) const K_TILE_GROUP_RATIO: u32 = GROUP_SIZE / K_TILE_SHARED; // 8 K-tiles per group

// --- Shared tile byte sizes ---

/// Bytes per element for f16 activations and f16 group scales.
const BYTES_PER_F16: u32 = 2;
/// Bytes per packed-INT4 word (8 nibbles per `u32`).
const BYTES_PER_B32: u32 = 4;

/// X tile in shared: `BM_BLOCK` rows × `K_TILE_SHARED` cols, f16 row-major.
/// 64 × 16 × 2 = 2048 B. Matches `matmul_int4_kernel::TILE_A_BYTES` so the
/// cooperative A-loader can be reused unchanged.
#[allow(dead_code)] // wired up in D5
pub(crate) const TILE_X_BYTES: u32 = BM_BLOCK * K_TILE_SHARED * BYTES_PER_F16; // 2048

/// X tile row stride in bytes.
#[allow(dead_code)] // wired up in D5
pub(crate) const TILE_X_ROW_STRIDE_BYTES: u32 = K_TILE_SHARED * BYTES_PER_F16; // 32

/// W_packed tile col-stride bytes — matches `matmul_int4_kernel::TILE_B_COL_STRIDE_BYTES`.
/// 2 packed u32s per K-tile col + 4 B padding = 12 B (bank-conflict avoidance,
/// see matmul_int4 D3 design note).
#[allow(dead_code)] // wired up in D5
pub(crate) const TILE_W_COL_STRIDE_BYTES: u32 = 12;

/// W_packed tile in shared: `BN_BLOCK` cols × `TILE_W_COL_STRIDE_BYTES` data,
/// padded to a 512-B multiple so the cooperative pre-zero pattern stays
/// 4-bytes-per-thread × 128 threads = 512 B per issue. After Rollback #1:
/// 16 × 12 = 192 B data, padded to 512 B. Same sizing approach as
/// `matmul_int4_kernel::TILE_B_BYTES` which lands at 1024 B with BN_BLOCK=64.
#[allow(dead_code)] // wired up in D5
pub(crate) const TILE_W_BYTES: u32 = {
    let data = BN_BLOCK * TILE_W_COL_STRIDE_BYTES;
    let chunk = THREADS_PER_BLOCK * 4;
    let pad = (chunk - data % chunk) % chunk;
    data + pad
}; // 512 (192 data + 320 pad)

/// Group-scale tile: one f16 per output column, BN_BLOCK f16 per tile.
/// Reloaded once per group transition (every 8 K-tiles). Tiny enough to fit
/// in a fraction of one warp's coalesced load. After Rollback #1: 16 × 2 = 32 B.
#[allow(dead_code)] // wired up in D5
pub(crate) const TILE_SCALES_BYTES: u32 = BN_BLOCK * BYTES_PER_F16; // 32

/// Pre-zero issue size — cooperative pre-zero writes one b32 per thread per
/// issue. `TILE_X_BYTES = 2048 B` = 4 issues per thread; `TILE_W_BYTES = 512 B`
/// = 1 issue per thread; `TILE_SCALES_BYTES = 32 B` = handled by predicating
/// the first 8 threads (32/4 = 8) — covered in the pre-zero helper.
#[allow(dead_code)]
const PRE_ZERO_BYTES_PER_ISSUE: u32 = THREADS_PER_BLOCK * 4; // 512

/// Cooperative load of the 64×16 f16 X block tile from global into shared.
/// Delegates to [`super::matmul_int4_kernel::emit_mw_load_tile_a_f16_64x16`]
/// — same shape, dtype, and row-major convention as `matmul_int4`'s A.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)] // wired up in D5
pub(crate) fn emit_mw_load_tile_x_f16_64x16(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    x_block_base_global: Register,
    tile_x_shared: Register,
    flat_tid: Register,
    block_row: Register,
    m: Register,
    k_bytes: Register,
    label_suffix: &str,
) {
    super::matmul_int4_kernel::emit_mw_load_tile_a_f16_64x16(
        alloc,
        kernel,
        x_block_base_global,
        tile_x_shared,
        flat_tid,
        block_row,
        m,
        k_bytes,
        label_suffix,
    );
}

/// Cooperative load of the packed-INT4 W tile from global into shared, sized
/// for `BN_BLOCK = 16` columns × 2 packed `u32` words per col.
///
/// # Tile shape
/// - 16 cols × 2 u32/col = 32 packed words = 128 B of real data per tile.
/// - Each thread issues 1 b32 if its `(col_in_tile, word_idx_local)` falls
///   inside the 32-word range; the remaining 96 threads idle through the
///   skip label.
///
/// # Per-thread layout
///
/// ```text
/// col_in_tile     = flat_tid / 2     // 0..64; valid only when < BN_BLOCK
/// word_idx_local  = flat_tid % 2     // 0 or 1 (the 2 u32s for the K-tile)
/// ```
///
/// # Edge handling
///
/// - **Tile predicate** (Rollback #1): `col_in_tile < BN_BLOCK = 16`. Threads
///   with `col_in_tile ≥ 16` skip — they cover the inactive half of the
///   per-thread layout and would otherwise read past the block's owned cols.
/// - **N-edge predicate**: `block_col + col_in_tile < N`. OOB cols fall
///   through to the pre-zeroed slot.
///
/// `w_packed_block_base_global` must already include both the block-column
/// offset (`block_col * (K/8) * 4` bytes, col-major INT4) and the per-K-tile
/// word offset (`k_tile * 2 * 4` bytes per col).
///
/// `label_suffix` differentiates the skip-label across the three
/// per-projection calls per K-tile; pass `"Q"` / `"K"` / `"V"`.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)] // wired up in D5
pub(crate) fn emit_mw_load_tile_w_packed_int4_2x16(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    w_packed_block_base_global: Register, // u64
    tile_w_shared: Register,              // u32
    flat_tid: Register,                   // u32 — 0..128
    block_col: Register,                  // u32
    n: Register,                          // u32
    k_words: Register,                    // u32 — K / NIBBLES_PER_U32 (rows of packed storage)
    label_suffix: &str,
) {
    let skip_label = if label_suffix.is_empty() {
        "WP_SKIP_I4_TILE_LOAD".to_string()
    } else {
        format!("WP_SKIP_I4_TILE_LOAD_{label_suffix}")
    };

    // col_in_tile = flat_tid / 2
    let col_in_tile = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: col_in_tile,
        lhs: Operand::Reg(flat_tid),
        rhs: Operand::ImmU32(2),
        ty: PtxType::U32,
    }));
    let word_idx_local = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Rem {
        dst: word_idx_local,
        lhs: Operand::Reg(flat_tid),
        rhs: Operand::ImmU32(2),
        ty: PtxType::U32,
    }));

    // Tile predicate: skip if col_in_tile >= BN_BLOCK (Rollback #1).
    let p_in_tile = alloc.alloc(PtxType::Pred);
    kernel.push(PtxInstruction::Control(ControlOp::SetP {
        dst: p_in_tile,
        cmp_op: CmpOp::Lt,
        lhs: Operand::Reg(col_in_tile),
        rhs: Operand::ImmU32(BN_BLOCK),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Control(ControlOp::BraPred {
        pred: p_in_tile,
        target: skip_label.clone(),
        negate: true,
    }));

    // N-edge predicate.
    let col_global = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: col_global,
        lhs: Operand::Reg(block_col),
        rhs: Operand::Reg(col_in_tile),
        ty: PtxType::U32,
    }));
    let p_n_edge = alloc.alloc(PtxType::Pred);
    kernel.push(PtxInstruction::Control(ControlOp::SetP {
        dst: p_n_edge,
        cmp_op: CmpOp::Lt,
        lhs: Operand::Reg(col_global),
        rhs: Operand::Reg(n),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Control(ControlOp::BraPred {
        pred: p_n_edge,
        target: skip_label.clone(),
        negate: true,
    }));

    // Shared offset = col_in_tile * TILE_W_COL_STRIDE_BYTES + word_idx_local * 4.
    let col_stride_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: col_stride_off,
        lhs: Operand::Reg(col_in_tile),
        rhs: Operand::ImmU32(TILE_W_COL_STRIDE_BYTES),
        ty: PtxType::U32,
    }));
    let word_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: word_off,
        lhs: Operand::Reg(word_idx_local),
        rhs: Operand::ImmU32(BYTES_PER_B32),
        ty: PtxType::U32,
    }));
    let shared_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: shared_off,
        lhs: Operand::Reg(col_stride_off),
        rhs: Operand::Reg(word_off),
        ty: PtxType::U32,
    }));
    let shared_addr = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: shared_addr,
        lhs: Operand::Reg(tile_w_shared),
        rhs: Operand::Reg(shared_off),
        ty: PtxType::U32,
    }));

    // Global offset (col-major packed): (col_in_tile * k_words + word_idx_local) * 4.
    let col_off_words = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: col_off_words,
        lhs: Operand::Reg(col_in_tile),
        rhs: Operand::Reg(k_words),
        ty: PtxType::U32,
    }));
    let total_word_idx = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: total_word_idx,
        lhs: Operand::Reg(col_off_words),
        rhs: Operand::Reg(word_idx_local),
        ty: PtxType::U32,
    }));
    let global_byte_off = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
        dst: global_byte_off,
        lhs: Operand::Reg(total_word_idx),
        rhs: Operand::ImmU32(BYTES_PER_B32),
        src_ty: PtxType::U32,
    }));
    let global_addr = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: global_addr,
        lhs: Operand::Reg(w_packed_block_base_global),
        rhs: Operand::Reg(global_byte_off),
        ty: PtxType::U64,
    }));

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

    kernel.push(PtxInstruction::Label(skip_label));
}

/// Cooperative load of the `BN_BLOCK = 16` f16 group-scale slice for the
/// current group transition. `scales[num_groups, N]` row-major.
///
/// Per block, on each group transition, reads `scales[g, block_col..+16]`
/// as 8 `.b32` words (2 f16 packed per b32). Lanes 0..7 cooperate; lanes
/// 8..127 idle through the skip label. N-edge predication drops loads where
/// `block_col + flat_tid*2 + 1 >= N` to avoid reading past N.
///
/// `scales_base_global` must be `&scales[g * N]` (row origin for the current
/// group — caller resolves `g = k_tile / K_TILE_GROUP_RATIO`).
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)] // wired up in D5
pub(crate) fn emit_cooperative_load_group_scales_int4(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    scales_base_global: Register, // u64
    tile_scales_shared: Register, // u32
    flat_tid: Register,           // u32 — 0..128
    block_col: Register,          // u32
    n: Register,                  // u32
    label_suffix: &str,
) {
    let skip_label = if label_suffix.is_empty() {
        "SCALES_SKIP_QKV_I4".to_string()
    } else {
        format!("SCALES_SKIP_QKV_I4_{label_suffix}")
    };
    // active_threads = BN_BLOCK / 2 = 8 (each thread loads one b32 = 2 f16).
    let active_threads = BN_BLOCK / 2;

    let p_active = alloc.alloc(PtxType::Pred);
    kernel.push(PtxInstruction::Control(ControlOp::SetP {
        dst: p_active,
        cmp_op: CmpOp::Lt,
        lhs: Operand::Reg(flat_tid),
        rhs: Operand::ImmU32(active_threads),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Control(ControlOp::BraPred {
        pred: p_active,
        target: skip_label.clone(),
        negate: true,
    }));

    // col_pair_idx = flat_tid (0..8); col_global = block_col + flat_tid * 2.
    let col_off_pair = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: col_off_pair,
        lhs: Operand::Reg(flat_tid),
        rhs: Operand::ImmU32(2),
        ty: PtxType::U32,
    }));
    let col_global = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: col_global,
        lhs: Operand::Reg(block_col),
        rhs: Operand::Reg(col_off_pair),
        ty: PtxType::U32,
    }));
    // N-edge: skip if col_global + 1 >= N (need 2 contiguous f16).
    let col_global_plus_one = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: col_global_plus_one,
        lhs: Operand::Reg(col_global),
        rhs: Operand::ImmU32(1),
        ty: PtxType::U32,
    }));
    let p_n_edge = alloc.alloc(PtxType::Pred);
    kernel.push(PtxInstruction::Control(ControlOp::SetP {
        dst: p_n_edge,
        cmp_op: CmpOp::Lt,
        lhs: Operand::Reg(col_global_plus_one),
        rhs: Operand::Reg(n),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Control(ControlOp::BraPred {
        pred: p_n_edge,
        target: skip_label.clone(),
        negate: true,
    }));

    // Shared addr = tile_scales + flat_tid * 4 (each thread writes one b32 pair).
    let shared_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: shared_off,
        lhs: Operand::Reg(flat_tid),
        rhs: Operand::ImmU32(BYTES_PER_B32),
        ty: PtxType::U32,
    }));
    let shared_addr = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: shared_addr,
        lhs: Operand::Reg(tile_scales_shared),
        rhs: Operand::Reg(shared_off),
        ty: PtxType::U32,
    }));

    // Global addr = scales_base + col_global * 2 (f16).
    let col_bytes = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: col_bytes,
        lhs: Operand::Reg(col_global),
        rhs: Operand::ImmU32(BYTES_PER_F16),
        ty: PtxType::U32,
    }));
    let col_bytes_u64 = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Cvt {
        dst: col_bytes_u64,
        src: col_bytes,
        dst_ty: PtxType::U64,
        src_ty: PtxType::U32,
    });
    let global_addr = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: global_addr,
        lhs: Operand::Reg(scales_base_global),
        rhs: Operand::Reg(col_bytes_u64),
        ty: PtxType::U64,
    }));

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

    kernel.push(PtxInstruction::Label(skip_label));
}

/// Cooperative pre-zero of the X + W_packed + scales shared tiles, then a
/// single `bar.sync 0`. X is 2048 B (4 issues/thread), W is 512 B (1 issue/
/// thread, padded), scales is 32 B (8 active threads, 1 issue each).
#[allow(dead_code)] // wired up in D5
pub(crate) fn emit_pre_zero_shared_tiles_qkv_int4(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    tile_x: Register,
    tile_w: Register,
    tile_scales: Register,
    flat_tid: Register,
) {
    debug_assert!(TILE_X_BYTES.is_multiple_of(PRE_ZERO_BYTES_PER_ISSUE));
    debug_assert!(TILE_W_BYTES.is_multiple_of(PRE_ZERO_BYTES_PER_ISSUE));

    let r_zero = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_zero,
        src: Operand::ImmU32(0),
        ty: PtxType::U32,
    });

    // X + W: full-occupancy 4-bytes-per-thread issues.
    for (tile_base, total_bytes) in [(tile_x, TILE_X_BYTES), (tile_w, TILE_W_BYTES)] {
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

    // Scales: predicated single-issue on the first TILE_SCALES_BYTES/4 threads.
    let scales_active = TILE_SCALES_BYTES / 4;
    let p_scales_active = alloc.alloc(PtxType::Pred);
    kernel.push(PtxInstruction::Control(ControlOp::SetP {
        dst: p_scales_active,
        cmp_op: CmpOp::Lt,
        lhs: Operand::Reg(flat_tid),
        rhs: Operand::ImmU32(scales_active),
        ty: PtxType::U32,
    }));
    let scales_skip = "PRE_ZERO_SCALES_SKIP_QKV_I4".to_string();
    kernel.push(PtxInstruction::Control(ControlOp::BraPred {
        pred: p_scales_active,
        target: scales_skip.clone(),
        negate: true,
    }));
    let r_scales_thread_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_scales_thread_off,
        lhs: Operand::Reg(flat_tid),
        rhs: Operand::ImmU32(4),
        ty: PtxType::U32,
    }));
    let scales_addr = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: scales_addr,
        lhs: Operand::Reg(tile_scales),
        rhs: Operand::Reg(r_scales_thread_off),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Memory(MemoryOp::StShared {
        addr: scales_addr,
        src: r_zero,
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Label(scales_skip));

    kernel.push(PtxInstruction::Control(ControlOp::BarSync {
        barrier_id: 0,
    }));
}

/// Per-warp-quadrant tri-output INT4 mma sweep for one projection.
///
/// Mirrors [`super::matmul_int4_kernel::emit_fragment_b_int4_per_lane`]'s
/// nibble-extract dequant + scale-fold chain (sign-extend via `shr.s32`,
/// `cvt` to f16, multiply by group-scale, `MovPack`), then runs the 2×1
/// inner mma grid into `frag_c` (Rollback #1: MMAS_PER_WARP_M=2,
/// MMAS_PER_WARP_N=1). Reuses the matmul_int4 dequant verbatim — INT4
/// fragment-B is projection-agnostic; only the A-tile's M offset and the
/// caller-provided frag_c grid differ across Q/K/V.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)] // wired up in D5
pub(crate) fn emit_warp_quadrant_mma_int4_per_projection(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    tile_x_shared: Register,
    tile_w_shared: Register,
    tile_scales_shared: Register,
    warp_quad_row_base_in_tile_x: Register, // 0 or 32
    warp_quad_col_base_in_tile_w: Register, // 0 or 8
    tid_x_in_warp: Register,
    frag_c_grid: &mut [[kaio_core::fragment::FragmentC; MMAS_PER_WARP_N as usize];
             MMAS_PER_WARP_M as usize],
) {
    use kaio_core::fragment::load_fragment_a_m16n8k16_shared_row;
    use kaio_core::instr::{MmaShape, TensorCoreOp};

    // Pre-compute the warp-quadrant byte row-shift into the shared X tile.
    let row_offset_bytes = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: row_offset_bytes,
        lhs: Operand::Reg(warp_quad_row_base_in_tile_x),
        rhs: Operand::ImmU32(TILE_X_ROW_STRIDE_BYTES),
        ty: PtxType::U32,
    }));

    for n_stripe in 0..MMAS_PER_WARP_N {
        // n_stripe_col_base = warp_quad_col_base + n_stripe * BN
        let n_stripe_col_base = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: n_stripe_col_base,
            lhs: Operand::Reg(warp_quad_col_base_in_tile_w),
            rhs: Operand::ImmU32(n_stripe * BN),
            ty: PtxType::U32,
        }));

        // Per-lane fragment B (nibble-extract + sign-extend + cvt + scale fold).
        let frag_b = super::matmul_int4_kernel::emit_fragment_b_int4_per_lane(
            alloc,
            kernel,
            tile_w_shared,
            tile_scales_shared,
            n_stripe_col_base,
            tid_x_in_warp,
        );

        for m_stripe in 0..MMAS_PER_WARP_M {
            let m_stripe_shift = alloc.alloc(PtxType::U32);
            kernel.push(PtxInstruction::Arith(ArithOp::Add {
                dst: m_stripe_shift,
                lhs: Operand::Reg(row_offset_bytes),
                rhs: Operand::ImmU32(m_stripe * BM * TILE_X_ROW_STRIDE_BYTES),
                ty: PtxType::U32,
            }));
            let shifted_tile_x = alloc.alloc(PtxType::U32);
            kernel.push(PtxInstruction::Arith(ArithOp::Add {
                dst: shifted_tile_x,
                lhs: Operand::Reg(tile_x_shared),
                rhs: Operand::Reg(m_stripe_shift),
                ty: PtxType::U32,
            }));
            let frag_a = load_fragment_a_m16n8k16_shared_row(
                alloc,
                kernel,
                shifted_tile_x,
                tid_x_in_warp,
                TILE_X_ROW_STRIDE_BYTES,
                None,
            );

            let frag_d = kaio_core::fragment::alloc_c(alloc);
            let frag_c_in = frag_c_grid[m_stripe as usize][n_stripe as usize];
            kernel.push(PtxInstruction::TensorCore(TensorCoreOp::MmaSync {
                d: frag_d,
                a: frag_a,
                b: frag_b,
                c: frag_c_in,
                shape: MmaShape::M16N8K16,
                d_ty: PtxType::F32,
                a_ty: PtxType::F16,
                b_ty: PtxType::F16,
                c_ty: PtxType::F32,
            }));
            for (c_reg, d_reg) in frag_c_in.regs.iter().zip(frag_d.regs.iter()) {
                kernel.push(PtxInstruction::Mov {
                    dst: *c_reg,
                    src: Operand::Reg(*d_reg),
                    ty: PtxType::F32,
                });
            }
        }
    }
}

/// Build the full IR module for `qkv_project_int4` targeting `sm`.
///
/// Sprint 7.3 D5 — INT4 contingent. Tri-output W4A16 kernel: f16 activations,
/// 3 packed-INT4 weight tensors (one per projection), 3 f16 group-scale
/// tensors (group_size = 128). Single launch produces three `f16` outputs
/// ready to feed `attention_tc`.
///
/// # Kernel signature (in launch order)
///
/// `x_ptr: *const f16, w_q_packed_ptr: *const u32, w_k_packed_ptr: *const u32,
///  w_v_packed_ptr: *const u32, scales_q_ptr: *const f16, scales_k_ptr,
///  scales_v_ptr, q_out_ptr, k_out_ptr, v_out_ptr, m, n, k`.
///
/// # Per-K-tile barrier cadence
///
/// 1 X-load sync + (1 W_P+scales sync + 1 W release sync) × 3 = **7 barriers
/// per K-tile** (Design S serial fusion). On group-transition K-tiles the
/// scale reload folds into the W_P load epoch — same barrier count, just
/// extra `ld.global` traffic on those iterations.
#[allow(dead_code)] // wired up in D6
pub(crate) fn build_qkv_project_int4_module(sm: &str) -> kaio_core::ir::PtxModule {
    use kaio_core::fragment::{FragmentC, alloc_c};
    use kaio_core::instr::MadMode;
    use kaio_core::ir::{PtxModule, PtxParam, SharedDecl};

    let mut alloc = RegisterAllocator::new();
    let mut kernel = PtxKernel::new("qkv_project_int4");

    // --- Kernel signature ---
    kernel.add_param(PtxParam::pointer("x_ptr", PtxType::F16));
    kernel.add_param(PtxParam::pointer("w_q_packed_ptr", PtxType::U32));
    kernel.add_param(PtxParam::pointer("w_k_packed_ptr", PtxType::U32));
    kernel.add_param(PtxParam::pointer("w_v_packed_ptr", PtxType::U32));
    kernel.add_param(PtxParam::pointer("scales_q_ptr", PtxType::F16));
    kernel.add_param(PtxParam::pointer("scales_k_ptr", PtxType::F16));
    kernel.add_param(PtxParam::pointer("scales_v_ptr", PtxType::F16));
    kernel.add_param(PtxParam::pointer("q_out_ptr", PtxType::F16));
    kernel.add_param(PtxParam::pointer("k_out_ptr", PtxType::F16));
    kernel.add_param(PtxParam::pointer("v_out_ptr", PtxType::F16));
    kernel.add_param(PtxParam::scalar("m", PtxType::U32));
    kernel.add_param(PtxParam::scalar("n", PtxType::U32));
    kernel.add_param(PtxParam::scalar("k", PtxType::U32));

    // --- Shared decls (Design S: single W slot + single scales slot reused) ---
    kernel.add_shared_decl(SharedDecl {
        name: "tile_x".to_string(),
        align: 4,
        size_bytes: TILE_X_BYTES,
    });
    kernel.add_shared_decl(SharedDecl {
        name: "tile_w".to_string(),
        align: 4,
        size_bytes: TILE_W_BYTES,
    });
    kernel.add_shared_decl(SharedDecl {
        name: "tile_scales".to_string(),
        align: 4,
        size_bytes: TILE_SCALES_BYTES,
    });

    // --- Load + cvta the 10 pointer params ---
    let load_and_cvta =
        |name: &str, alloc: &mut RegisterAllocator, kernel: &mut PtxKernel| -> Register {
            let rd = alloc.alloc(PtxType::U64);
            kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
                dst: rd,
                param_name: name.to_string(),
                ty: PtxType::U64,
            }));
            let rd_g = alloc.alloc(PtxType::U64);
            kernel.push(PtxInstruction::Memory(MemoryOp::CvtaToGlobal {
                dst: rd_g,
                src: rd,
            }));
            rd_g
        };
    let rd_x = load_and_cvta("x_ptr", &mut alloc, &mut kernel);
    let rd_w_q = load_and_cvta("w_q_packed_ptr", &mut alloc, &mut kernel);
    let rd_w_k = load_and_cvta("w_k_packed_ptr", &mut alloc, &mut kernel);
    let rd_w_v = load_and_cvta("w_v_packed_ptr", &mut alloc, &mut kernel);
    let rd_s_q = load_and_cvta("scales_q_ptr", &mut alloc, &mut kernel);
    let rd_s_k = load_and_cvta("scales_k_ptr", &mut alloc, &mut kernel);
    let rd_s_v = load_and_cvta("scales_v_ptr", &mut alloc, &mut kernel);
    let rd_q_out = load_and_cvta("q_out_ptr", &mut alloc, &mut kernel);
    let rd_k_out = load_and_cvta("k_out_ptr", &mut alloc, &mut kernel);
    let rd_v_out = load_and_cvta("v_out_ptr", &mut alloc, &mut kernel);

    // --- Load 3 dim scalars ---
    let load_scalar = |name: &str,
                       ty: PtxType,
                       alloc: &mut RegisterAllocator,
                       kernel: &mut PtxKernel|
     -> Register {
        let r = alloc.alloc(ty);
        kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
            dst: r,
            param_name: name.to_string(),
            ty,
        }));
        r
    };
    let r_m = load_scalar("m", PtxType::U32, &mut alloc, &mut kernel);
    let r_n = load_scalar("n", PtxType::U32, &mut alloc, &mut kernel);
    let r_k = load_scalar("k", PtxType::U32, &mut alloc, &mut kernel);

    // --- Derived dim scalars ---
    let r_k_bytes = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_k_bytes,
        lhs: Operand::Reg(r_k),
        rhs: Operand::ImmU32(BYTES_PER_F16),
        ty: PtxType::U32,
    }));
    let r_k_words = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: r_k_words,
        lhs: Operand::Reg(r_k),
        rhs: Operand::ImmU32(NIBBLES_PER_U32),
        ty: PtxType::U32,
    }));
    let r_n_f16_stride = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_n_f16_stride,
        lhs: Operand::Reg(r_n),
        rhs: Operand::ImmU32(BYTES_PER_F16),
        ty: PtxType::U32,
    }));

    // --- Thread / warp / block indices ---
    let (r_tid_x, tid_x_instr) = kaio_core::instr::special::tid_x(&mut alloc);
    kernel.push(tid_x_instr);
    let (r_tid_y, tid_y_instr) = kaio_core::instr::special::tid_y(&mut alloc);
    kernel.push(tid_y_instr);
    let (r_ntid_x, ntid_x_instr) = kaio_core::instr::special::ntid_x(&mut alloc);
    kernel.push(ntid_x_instr);
    let r_flat_tid = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mad {
        dst: r_flat_tid,
        a: Operand::Reg(r_tid_y),
        b: Operand::Reg(r_ntid_x),
        c: Operand::Reg(r_tid_x),
        ty: PtxType::U32,
        mode: MadMode::Lo,
    }));
    let r_warp_id = r_tid_y;
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
    let r_wq_row_idx = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_wq_row_idx,
        lhs: Operand::Reg(r_warp_row_quad),
        rhs: Operand::ImmU32(WARP_QUAD_M),
        ty: PtxType::U32,
    }));
    let r_wq_col_idx = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_wq_col_idx,
        lhs: Operand::Reg(r_warp_col_quad),
        rhs: Operand::ImmU32(WARP_QUAD_N),
        ty: PtxType::U32,
    }));

    let (r_ctaid_x, ctaid_x_instr) = kaio_core::instr::special::ctaid_x(&mut alloc);
    kernel.push(ctaid_x_instr);
    let (r_ctaid_y, ctaid_y_instr) = kaio_core::instr::special::ctaid_y(&mut alloc);
    kernel.push(ctaid_y_instr);
    let r_block_row = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_block_row,
        lhs: Operand::Reg(r_ctaid_y),
        rhs: Operand::ImmU32(BM_BLOCK),
        ty: PtxType::U32,
    }));
    let r_block_col = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_block_col,
        lhs: Operand::Reg(r_ctaid_x),
        rhs: Operand::ImmU32(BN_BLOCK),
        ty: PtxType::U32,
    }));

    // --- Shared base regs ---
    let r_tile_x = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_tile_x,
        src: Operand::SharedAddr("tile_x".to_string()),
        ty: PtxType::U32,
    });
    let r_tile_w = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_tile_w,
        src: Operand::SharedAddr("tile_w".to_string()),
        ty: PtxType::U32,
    });
    let r_tile_s = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_tile_s,
        src: Operand::SharedAddr("tile_scales".to_string()),
        ty: PtxType::U32,
    });

    // --- Block-level global origins ---
    // x_block_base = rd_x + block_row * k_bytes
    let rd_x_row_off = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
        dst: rd_x_row_off,
        lhs: Operand::Reg(r_block_row),
        rhs: Operand::Reg(r_k_bytes),
        src_ty: PtxType::U32,
    }));
    let rd_x_block_base = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: rd_x_block_base,
        lhs: Operand::Reg(rd_x),
        rhs: Operand::Reg(rd_x_row_off),
        ty: PtxType::U64,
    }));

    // w_packed col-major: w_block_base = rd_w + block_col * k_words * 4
    let r_b_col_off_words = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_b_col_off_words,
        lhs: Operand::Reg(r_block_col),
        rhs: Operand::Reg(r_k_words),
        ty: PtxType::U32,
    }));
    let rd_b_col_off = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
        dst: rd_b_col_off,
        lhs: Operand::Reg(r_b_col_off_words),
        rhs: Operand::ImmU32(BYTES_PER_B32),
        src_ty: PtxType::U32,
    }));
    let add_u64 =
        |alloc: &mut RegisterAllocator, kernel: &mut PtxKernel, base: Register, off: Register| {
            let r = alloc.alloc(PtxType::U64);
            kernel.push(PtxInstruction::Arith(ArithOp::Add {
                dst: r,
                lhs: Operand::Reg(base),
                rhs: Operand::Reg(off),
                ty: PtxType::U64,
            }));
            r
        };
    let rd_w_q_block_base = add_u64(&mut alloc, &mut kernel, rd_w_q, rd_b_col_off);
    let rd_w_k_block_base = add_u64(&mut alloc, &mut kernel, rd_w_k, rd_b_col_off);
    let rd_w_v_block_base = add_u64(&mut alloc, &mut kernel, rd_w_v, rd_b_col_off);

    // --- Allocate 3 frag_c grids (zero-init f32) ---
    let alloc_grid =
        |a: &mut RegisterAllocator| -> [[FragmentC; MMAS_PER_WARP_N as usize];
            MMAS_PER_WARP_M as usize] {
            core::array::from_fn(|_| core::array::from_fn(|_| alloc_c(a)))
        };
    let mut frag_c_q = alloc_grid(&mut alloc);
    let mut frag_c_k = alloc_grid(&mut alloc);
    let mut frag_c_v = alloc_grid(&mut alloc);
    for grid in [&frag_c_q, &frag_c_k, &frag_c_v] {
        for row in grid {
            for f in row {
                for r in &f.regs {
                    kernel.push(PtxInstruction::Mov {
                        dst: *r,
                        src: Operand::ImmF32(0.0),
                        ty: PtxType::F32,
                    });
                }
            }
        }
    }

    // --- Pre-zero shared tiles + bar.sync ---
    emit_pre_zero_shared_tiles_qkv_int4(
        &mut alloc,
        &mut kernel,
        r_tile_x,
        r_tile_w,
        r_tile_s,
        r_flat_tid,
    );

    // --- K-loop ---
    let r_num_k_tiles = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: r_num_k_tiles,
        lhs: Operand::Reg(r_k),
        rhs: Operand::ImmU32(K_TILE_SHARED),
        ty: PtxType::U32,
    }));
    let r_k_tile = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_k_tile,
        src: Operand::ImmU32(0),
        ty: PtxType::U32,
    });

    kernel.push(PtxInstruction::Label("K_LOOP_QKV_INT4".to_string()));

    // X tile global source = x_block_base + k_tile * 32 (K_TILE_SHARED * BYTES_PER_F16)
    let r_k_tile_x_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_k_tile_x_off,
        lhs: Operand::Reg(r_k_tile),
        rhs: Operand::ImmU32(K_TILE_SHARED * BYTES_PER_F16),
        ty: PtxType::U32,
    }));
    let rd_k_tile_x_off64 = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Cvt {
        dst: rd_k_tile_x_off64,
        src: r_k_tile_x_off,
        dst_ty: PtxType::U64,
        src_ty: PtxType::U32,
    });
    let rd_x_tile_src = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: rd_x_tile_src,
        lhs: Operand::Reg(rd_x_block_base),
        rhs: Operand::Reg(rd_k_tile_x_off64),
        ty: PtxType::U64,
    }));

    // W_P packed tile global offset = k_tile * (K_TILE_SHARED/8) * 4 = k_tile * 8
    let r_k_tile_b_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_k_tile_b_off,
        lhs: Operand::Reg(r_k_tile),
        rhs: Operand::ImmU32((K_TILE_SHARED / NIBBLES_PER_U32) * BYTES_PER_B32),
        ty: PtxType::U32,
    }));
    let rd_k_tile_b_off64 = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Cvt {
        dst: rd_k_tile_b_off64,
        src: r_k_tile_b_off,
        dst_ty: PtxType::U64,
        src_ty: PtxType::U32,
    });
    let rd_w_q_tile_src = add_u64(
        &mut alloc,
        &mut kernel,
        rd_w_q_block_base,
        rd_k_tile_b_off64,
    );
    let rd_w_k_tile_src = add_u64(
        &mut alloc,
        &mut kernel,
        rd_w_k_block_base,
        rd_k_tile_b_off64,
    );
    let rd_w_v_tile_src = add_u64(
        &mut alloc,
        &mut kernel,
        rd_w_v_block_base,
        rd_k_tile_b_off64,
    );

    // Current group id: g = k_tile / K_TILE_GROUP_RATIO.
    //
    // Design S has a single `tile_scales` shared slot reused across all three
    // projections per K-tile. Because each projection's mma reads its OWN
    // scales, the slot must be rewritten before each of Q/K/V — not just at
    // group boundaries. Reloading every K-tile adds 32 B/projection ×
    // 3 projections = 96 B global read per K-tile (vs 384 B/K-tile for the W
    // loads) and keeps the serial pipeline correct across the 8-K-tile group
    // span. Group-boundary optimization (load once per 8 K-tiles) would
    // require three separate `tile_scales_P` shared decls; deferred pending a
    // bench-driven decision (the bandwidth saving is small and shared-mem
    // footprint matters more than a handful of global ld.b32 issues).
    let r_current_group = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: r_current_group,
        lhs: Operand::Reg(r_k_tile),
        rhs: Operand::ImmU32(K_TILE_GROUP_RATIO),
        ty: PtxType::U32,
    }));
    let r_scales_row_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_scales_row_off,
        lhs: Operand::Reg(r_current_group),
        rhs: Operand::Reg(r_n),
        ty: PtxType::U32,
    }));
    let rd_scales_row_off64 = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
        dst: rd_scales_row_off64,
        lhs: Operand::Reg(r_scales_row_off),
        rhs: Operand::ImmU32(BYTES_PER_F16),
        src_ty: PtxType::U32,
    }));
    let rd_scales_q_base = add_u64(&mut alloc, &mut kernel, rd_s_q, rd_scales_row_off64);
    let rd_scales_k_base = add_u64(&mut alloc, &mut kernel, rd_s_k, rd_scales_row_off64);
    let rd_scales_v_base = add_u64(&mut alloc, &mut kernel, rd_s_v, rd_scales_row_off64);

    // X-load (once per K-tile, shared across all 3 projections).
    emit_mw_load_tile_x_f16_64x16(
        &mut alloc,
        &mut kernel,
        rd_x_tile_src,
        r_tile_x,
        r_flat_tid,
        r_block_row,
        r_m,
        r_k_bytes,
        "QKV",
    );
    kernel.push(PtxInstruction::Control(ControlOp::BarSync {
        barrier_id: 0,
    }));

    // Per-projection epoch: scales_P + W_P load → bar.sync → mma → bar.sync.
    //
    // Scales reload is unconditional (see note above on the single-slot design).
    // The three projections run serially within the K-tile; each loads its own
    // scales into `tile_scales` immediately before its mma reads them.
    for (w_tile_src, scales_base, label, frag_c) in [
        (rd_w_q_tile_src, rd_scales_q_base, "Q", &mut frag_c_q),
        (rd_w_k_tile_src, rd_scales_k_base, "K", &mut frag_c_k),
        (rd_w_v_tile_src, rd_scales_v_base, "V", &mut frag_c_v),
    ] {
        emit_cooperative_load_group_scales_int4(
            &mut alloc,
            &mut kernel,
            scales_base,
            r_tile_s,
            r_flat_tid,
            r_block_col,
            r_n,
            label,
        );

        emit_mw_load_tile_w_packed_int4_2x16(
            &mut alloc,
            &mut kernel,
            w_tile_src,
            r_tile_w,
            r_flat_tid,
            r_block_col,
            r_n,
            r_k_words,
            label,
        );
        kernel.push(PtxInstruction::Control(ControlOp::BarSync {
            barrier_id: 0,
        }));
        emit_warp_quadrant_mma_int4_per_projection(
            &mut alloc,
            &mut kernel,
            r_tile_x,
            r_tile_w,
            r_tile_s,
            r_wq_row_idx,
            r_wq_col_idx,
            r_tid_x,
            frag_c,
        );
        kernel.push(PtxInstruction::Control(ControlOp::BarSync {
            barrier_id: 0,
        }));
    }

    // k_tile += 1; if k_tile < num_k_tiles, jump back.
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
        target: "K_LOOP_QKV_INT4".to_string(),
        negate: false,
    }));

    // --- Store-out epilogue (reuses the shared D2 store helper) ---
    let r_warp_block_row = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_warp_block_row,
        lhs: Operand::Reg(r_block_row),
        rhs: Operand::Reg(r_wq_row_idx),
        ty: PtxType::U32,
    }));
    let r_warp_block_col = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_warp_block_col,
        lhs: Operand::Reg(r_block_col),
        rhs: Operand::Reg(r_wq_col_idx),
        ty: PtxType::U32,
    }));
    let rd_warp_row_off = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
        dst: rd_warp_row_off,
        lhs: Operand::Reg(r_warp_block_row),
        rhs: Operand::Reg(r_n_f16_stride),
        src_ty: PtxType::U32,
    }));
    let r_warp_col_off_bytes = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_warp_col_off_bytes,
        lhs: Operand::Reg(r_warp_block_col),
        rhs: Operand::ImmU32(BYTES_PER_F16),
        ty: PtxType::U32,
    }));
    let rd_warp_col_off = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Cvt {
        dst: rd_warp_col_off,
        src: r_warp_col_off_bytes,
        dst_ty: PtxType::U64,
        src_ty: PtxType::U32,
    });

    let make_warp_base =
        |alloc: &mut RegisterAllocator, kernel: &mut PtxKernel, p_g: Register| -> Register {
            let pre = alloc.alloc(PtxType::U64);
            kernel.push(PtxInstruction::Arith(ArithOp::Add {
                dst: pre,
                lhs: Operand::Reg(p_g),
                rhs: Operand::Reg(rd_warp_row_off),
                ty: PtxType::U64,
            }));
            let r = alloc.alloc(PtxType::U64);
            kernel.push(PtxInstruction::Arith(ArithOp::Add {
                dst: r,
                lhs: Operand::Reg(pre),
                rhs: Operand::Reg(rd_warp_col_off),
                ty: PtxType::U64,
            }));
            r
        };
    let rd_q_warp_base = make_warp_base(&mut alloc, &mut kernel, rd_q_out);
    let rd_k_warp_base = make_warp_base(&mut alloc, &mut kernel, rd_k_out);
    let rd_v_warp_base = make_warp_base(&mut alloc, &mut kernel, rd_v_out);

    let r_16_rows_bytes = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_16_rows_bytes,
        lhs: Operand::Reg(r_n_f16_stride),
        rhs: Operand::ImmU32(BM),
        ty: PtxType::U32,
    }));

    let projections = [
        (rd_q_warp_base, &frag_c_q),
        (rd_k_warp_base, &frag_c_k),
        (rd_v_warp_base, &frag_c_v),
    ];
    for m_stripe in 0..MMAS_PER_WARP_M {
        for n_stripe in 0..MMAS_PER_WARP_N {
            let sub_off_u32 = if m_stripe == 0 {
                let r = alloc.alloc(PtxType::U32);
                kernel.push(PtxInstruction::Mov {
                    dst: r,
                    src: Operand::ImmU32(n_stripe * BN * BYTES_PER_F16),
                    ty: PtxType::U32,
                });
                r
            } else {
                let r = alloc.alloc(PtxType::U32);
                kernel.push(PtxInstruction::Arith(ArithOp::Add {
                    dst: r,
                    lhs: Operand::Reg(r_16_rows_bytes),
                    rhs: Operand::ImmU32(n_stripe * BN * BYTES_PER_F16),
                    ty: PtxType::U32,
                }));
                r
            };
            let rd_sub_off = alloc.alloc(PtxType::U64);
            kernel.push(PtxInstruction::Cvt {
                dst: rd_sub_off,
                src: sub_off_u32,
                dst_ty: PtxType::U64,
                src_ty: PtxType::U32,
            });
            for (warp_base, grid) in &projections {
                let rd_subtile_base = alloc.alloc(PtxType::U64);
                kernel.push(PtxInstruction::Arith(ArithOp::Add {
                    dst: rd_subtile_base,
                    lhs: Operand::Reg(*warp_base),
                    rhs: Operand::Reg(rd_sub_off),
                    ty: PtxType::U64,
                }));
                // INT4 folds the group scale into fragment B during dequant —
                // pass `scale = None` to the store helper.
                crate::store_out::emit_store_fragment_c_f32_to_f16_packed(
                    &mut alloc,
                    &mut kernel,
                    &grid[m_stripe as usize][n_stripe as usize].regs,
                    rd_subtile_base,
                    r_n_f16_stride,
                    r_tid_x,
                    None,
                );
            }
        }
    }

    kernel.push(PtxInstruction::Control(ControlOp::Ret));
    kernel.set_registers(alloc.into_allocated());

    let mut module = PtxModule::new(sm);
    module.add_kernel(kernel);
    module
}

/// Validate shape + alignment preconditions for `qkv_project_int4`.
///
/// Sprint 7.3 D1. Enforces the **W4A16 MHA** contract:
///
/// - `M`, `N`, `K` all non-zero.
/// - `group_size == GROUP_SIZE` (=128) — non-128 group sizes are a
///   follow-up extension inherited from `matmul_int4`.
/// - `K % GROUP_SIZE == 0` (implies `K % K_TILE_SHARED == 0`).
/// - `N % 2 == 0` — store-out packs adjacent f16 output pairs into one `.b32`.
/// - `N_q == N_k == N_v == N` — v1 is strict MHA. Grouped-query attention
///   (GQA) is a follow-up op; users with GQA weights should call three
///   separate `matmul_int4`s.
/// - Buffer-size sanity:
///   - `x >= M * K`
///   - each of `w_q_packed`, `w_k_packed`, `w_v_packed` >= `N * (K / 8)` u32
///   - each of `scales_q`, `scales_k`, `scales_v` >= `(K / group_size) * N` f16
///   - each of `q_out`, `k_out`, `v_out` >= `M * N` f16
///
/// `M` and `N` may be any positive value — edge-tile predication in the
/// kernel handles ragged output (same posture as `matmul_int4`).
///
/// # Pointer distinctness
///
/// `q_out`, `k_out`, `v_out` must be **three distinct allocations**.
/// Same pathological-caller guard as `qkv_project_int8` — Rust's `&mut`
/// borrow rules prevent variable-level aliasing and KAIO does not
/// expose buffer-splitting APIs that could produce overlapping views.
#[allow(dead_code)] // wired up in D6
pub(crate) fn validate_dims_qkv_int4(
    x: &GpuBuffer<half::f16>,
    w_q_packed: &GpuBuffer<u32>,
    w_k_packed: &GpuBuffer<u32>,
    w_v_packed: &GpuBuffer<u32>,
    scales_q: &GpuBuffer<half::f16>,
    scales_k: &GpuBuffer<half::f16>,
    scales_v: &GpuBuffer<half::f16>,
    q_out: &GpuBuffer<half::f16>,
    k_out: &GpuBuffer<half::f16>,
    v_out: &GpuBuffer<half::f16>,
    m: u32,
    n: u32,
    k: u32,
    group_size: u32,
) -> Result<()> {
    if m == 0 || n == 0 || k == 0 {
        return Err(KaioError::InvalidConfig(
            "qkv_project_int4: M, N, K dimensions must be non-zero".to_string(),
        ));
    }
    if group_size != GROUP_SIZE {
        return Err(KaioError::InvalidConfig(format!(
            "qkv_project_int4: group_size must be {GROUP_SIZE} (got {group_size}). \
             Non-128 group sizes are deferred to a follow-up sprint."
        )));
    }
    if !k.is_multiple_of(GROUP_SIZE) {
        return Err(KaioError::InvalidConfig(format!(
            "qkv_project_int4: K must be a multiple of group_size={GROUP_SIZE} (got {k}). \
             Partial groups are not supported."
        )));
    }
    if !n.is_multiple_of(2) {
        return Err(KaioError::InvalidConfig(format!(
            "qkv_project_int4: N must be even (got {n}). The store-out path packs \
             adjacent f16 output pairs into one .b32; odd N would leave a ragged \
             last column that the current store epilogue does not handle."
        )));
    }

    let mk = (m as usize) * (k as usize);
    let packed_words = ((k as usize) / (NIBBLES_PER_U32 as usize)) * (n as usize);
    let num_groups = (k as usize) / (group_size as usize);
    let scales_cells = num_groups * (n as usize);
    let mn = (m as usize) * (n as usize);

    if x.len() < mk {
        return Err(KaioError::InvalidConfig(format!(
            "qkv_project_int4: X buffer too small: need {mk} f16 ({m}×{k}), got {}",
            x.len()
        )));
    }
    for (label, buf) in [
        ("W_Q_packed", w_q_packed),
        ("W_K_packed", w_k_packed),
        ("W_V_packed", w_v_packed),
    ] {
        if buf.len() < packed_words {
            return Err(KaioError::InvalidConfig(format!(
                "qkv_project_int4: {label} buffer too small: need {packed_words} u32 \
                 ({n} cols × {} K-words), got {}",
                (k as usize) / (NIBBLES_PER_U32 as usize),
                buf.len()
            )));
        }
    }
    for (label, buf) in [
        ("scales_q", scales_q),
        ("scales_k", scales_k),
        ("scales_v", scales_v),
    ] {
        if buf.len() < scales_cells {
            return Err(KaioError::InvalidConfig(format!(
                "qkv_project_int4: {label} buffer too small: need {scales_cells} f16 \
                 ({num_groups} groups × {n} cols), got {}",
                buf.len()
            )));
        }
    }
    for (label, buf) in [("Q_out", q_out), ("K_out", k_out), ("V_out", v_out)] {
        if buf.len() < mn {
            return Err(KaioError::InvalidConfig(format!(
                "qkv_project_int4: {label} buffer too small: need {mn} f16 ({m}×{n}), got {}",
                buf.len()
            )));
        }
    }
    Ok(())
}

/// Fused tri-output INT4 QKV projection — **W4A16** (f16 activations,
/// packed signed-INT4 weights, f16 group scales).
///
/// Sprint 7.3 contingent ship. One kernel launch produces three
/// `GpuBuffer<f16>` outputs (Q, K, V) ready to feed
/// [`attention_tc`][crate::attention_tc] from a shared activation `x`,
/// three packed-INT4 weight tensors, and three group-scale tensors.
/// Saves 2× global activation reads vs three separate
/// [`matmul_int4`][crate::matmul_int4] calls and amortizes kernel-launch
/// overhead — dominant at autoregressive-decode batch sizes.
///
/// # Packing convention (matches `matmul_int4`)
///
/// Each `w_*_packed` tensor has logical shape `[K/8, N]` stored
/// **col-major** as `u32`, with 8 signed `s4` nibbles packed K-contiguous
/// per word. For logical weight `B[k, n]`:
///
/// ```text
/// word_index   = (k / 8) + n * (K / 8)   // index into w_*_packed
/// nibble_index = k % 8                   // lane within the word
/// nibble_bits  = (w[word_index] >> (4 * nibble_index)) & 0xF
/// signed_value = sign_extend_from_4_bits(nibble_bits)  // in [-8, +7]
/// ```
///
/// See `examples/int4_matmul/` for a CPU reference packer that produces
/// this layout from f16 weights. **Not** a drop-in for AutoGPTQ /
/// exllama / GGUF formats.
///
/// # Group scales
///
/// Each `scales_*` tensor has shape `[K/group_size, N]` stored
/// **row-major** as `f16`. For group `g = k / group_size`,
/// `scales[g * N + n]` is the shared scale for all `group_size` weights
/// in column `n` over K-range `[g * group_size, (g + 1) * group_size)`.
/// `group_size = 128` is fixed in v1; `K % 128 == 0` is required.
///
/// # Contract (v0.4.0 unreleased)
///
/// - **`x`**: `f16` row-major `[M, K]`.
/// - **`w_q_packed`, `w_k_packed`, `w_v_packed`**: each `u32` packed-INT4
///   col-major `[K/8, N]`. Three distinct allocations.
/// - **`scales_q`, `scales_k`, `scales_v`**: each `f16` row-major
///   `[K/128, N]`. Three distinct allocations.
/// - **`q_out`, `k_out`, `v_out`**: each `f16` row-major `[M, N]`.
///   **MHA**: `N_q == N_k == N_v == N` enforced.
/// - **`group_size = 128`** required (parameterization is a follow-up).
/// - **`K % 128 == 0`** and **`N % 2 == 0`** required.
///
/// # Hardware
///
/// Requires NVIDIA Ampere or newer (SM 8.0+) for the
/// `mma.sync.m16n8k16.f16.f16.f32` instance shape.
///
/// # Out of scope
///
/// Asymmetric INT4 (zero-points), non-128 group sizes, GGUF / AWQ
/// packings, bias / activation fusion, GQA, async pipelining, and
/// auto-tuning — all are follow-up sprints. See
/// `docs/development/sprints/phase7/` for the roadmap.
#[allow(clippy::too_many_arguments)]
pub fn qkv_project_int4(
    device: &KaioDevice,
    x: &GpuBuffer<half::f16>,
    w_q_packed: &GpuBuffer<u32>,
    w_k_packed: &GpuBuffer<u32>,
    w_v_packed: &GpuBuffer<u32>,
    scales_q: &GpuBuffer<half::f16>,
    scales_k: &GpuBuffer<half::f16>,
    scales_v: &GpuBuffer<half::f16>,
    q_out: &mut GpuBuffer<half::f16>,
    k_out: &mut GpuBuffer<half::f16>,
    v_out: &mut GpuBuffer<half::f16>,
    m: u32,
    n: u32,
    k: u32,
    group_size: u32,
) -> Result<()> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};

    validate_dims_qkv_int4(
        x, w_q_packed, w_k_packed, w_v_packed, scales_q, scales_k, scales_v, q_out, k_out, v_out,
        m, n, k, group_size,
    )?;

    let info = device.info()?;
    let (major, minor) = info.compute_capability;
    let sm = format!("sm_{major}{minor}");
    let module = build_qkv_project_int4_module(&sm);

    let kmodule = device.load_module(&module)?;
    let func = kmodule.function("qkv_project_int4")?;

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
            .arg(x.inner())
            .arg(w_q_packed.inner())
            .arg(w_k_packed.inner())
            .arg(w_v_packed.inner())
            .arg(scales_q.inner())
            .arg(scales_k.inner())
            .arg(scales_v.inner())
            .arg(q_out.inner_mut())
            .arg(k_out.inner_mut())
            .arg(v_out.inner_mut())
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

    fn make_device() -> Result<KaioDevice> {
        // cudarc panics (not returns Err) inside its dlopen path when the
        // CUDA shared library is unavailable — e.g. host-only CI runners
        // without a driver. Catch that panic and surface DeviceNotFound so
        // the `let Ok(device) = ...` skip-guards below fire as intended.
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| KaioDevice::new(0))) {
            Ok(result) => result,
            Err(_) => Err(KaioError::DeviceNotFound(0)),
        }
    }

    #[test]
    fn validate_accepts_canonical_shape() {
        let Ok(device) = make_device() else {
            return;
        };
        let m = 64u32;
        let n = 64u32;
        let k = 128u32; // exactly one group
        let x = device.alloc_zeros::<half::f16>((m * k) as usize).unwrap();
        let w = device
            .alloc_zeros::<u32>(((k / NIBBLES_PER_U32) * n) as usize)
            .unwrap();
        let num_groups = k / GROUP_SIZE;
        let s = device
            .alloc_zeros::<half::f16>((num_groups * n) as usize)
            .unwrap();
        let o = device.alloc_zeros::<half::f16>((m * n) as usize).unwrap();
        assert!(
            validate_dims_qkv_int4(&x, &w, &w, &w, &s, &s, &s, &o, &o, &o, m, n, k, GROUP_SIZE)
                .is_ok()
        );
    }

    #[test]
    fn validate_rejects_non_default_group_size() {
        let Ok(device) = make_device() else {
            return;
        };
        let x = device.alloc_zeros::<half::f16>(1024).unwrap();
        let w = device.alloc_zeros::<u32>(1024).unwrap();
        let s = device.alloc_zeros::<half::f16>(1024).unwrap();
        let o = device.alloc_zeros::<half::f16>(1024).unwrap();
        let err = validate_dims_qkv_int4(&x, &w, &w, &w, &s, &s, &s, &o, &o, &o, 64, 64, 128, 64)
            .unwrap_err();
        match err {
            KaioError::InvalidConfig(msg) => assert!(msg.contains("group_size must be")),
            _ => panic!("expected InvalidConfig"),
        }
    }

    #[test]
    fn validate_rejects_k_not_multiple_of_group_size() {
        let Ok(device) = make_device() else {
            return;
        };
        let x = device.alloc_zeros::<half::f16>(1024).unwrap();
        let w = device.alloc_zeros::<u32>(1024).unwrap();
        let s = device.alloc_zeros::<half::f16>(1024).unwrap();
        let o = device.alloc_zeros::<half::f16>(1024).unwrap();
        // K=64 (< GROUP_SIZE=128) fails multiple-of check.
        let err = validate_dims_qkv_int4(
            &x, &w, &w, &w, &s, &s, &s, &o, &o, &o, 64, 64, 64, GROUP_SIZE,
        )
        .unwrap_err();
        match err {
            KaioError::InvalidConfig(msg) => assert!(msg.contains("K must be a multiple")),
            _ => panic!("expected InvalidConfig"),
        }
    }

    #[test]
    fn validate_rejects_small_packed_weight_buffer() {
        let Ok(device) = make_device() else {
            return;
        };
        let m = 64u32;
        let n = 64u32;
        let k = 128u32;
        let x = device.alloc_zeros::<half::f16>((m * k) as usize).unwrap();
        // Undersized packed W — should hold (k/8)*n=1024 u32 but we give 16.
        let w_bad = device.alloc_zeros::<u32>(16).unwrap();
        let w_ok = device
            .alloc_zeros::<u32>(((k / NIBBLES_PER_U32) * n) as usize)
            .unwrap();
        let s = device.alloc_zeros::<half::f16>((n) as usize).unwrap();
        let o = device.alloc_zeros::<half::f16>((m * n) as usize).unwrap();
        let err = validate_dims_qkv_int4(
            &x, &w_bad, &w_ok, &w_ok, &s, &s, &s, &o, &o, &o, m, n, k, GROUP_SIZE,
        )
        .unwrap_err();
        match err {
            KaioError::InvalidConfig(msg) => assert!(msg.contains("W_Q_packed buffer too small")),
            _ => panic!("expected InvalidConfig"),
        }
    }

    #[test]
    fn tile_constants_are_self_consistent() {
        assert_eq!(MMAS_PER_WARP_M * BM, WARP_QUAD_M);
        assert_eq!(MMAS_PER_WARP_N * BN, WARP_QUAD_N);
        assert_eq!(WARP_QUAD_M * 2, BM_BLOCK);
        assert_eq!(WARP_QUAD_N * 2, BN_BLOCK);
        assert_eq!(THREADS_PER_BLOCK, 128);
        // Group-scale reload cadence: 128 / 16 = 8 K-tiles per group.
        assert_eq!(K_TILE_GROUP_RATIO, 8);
        // Rollback #1: 64×16 per-block tile, MMAS_PER_WARP_N=1.
        assert_eq!(MMAS_PER_WARP_N, 1);
        assert_eq!(BN_BLOCK, 16);
        assert_eq!(WARP_QUAD_N, 8);
    }

    #[test]
    fn tile_byte_constants_match_shape() {
        assert_eq!(TILE_X_BYTES, 2048);
        assert_eq!(TILE_X_ROW_STRIDE_BYTES, 32);
        assert_eq!(TILE_W_COL_STRIDE_BYTES, 12);
        assert_eq!(TILE_W_BYTES, 512); // 192 data + 320 pad
        assert_eq!(TILE_SCALES_BYTES, 32); // 16 cols × 2 B
        assert_eq!(TILE_X_BYTES % PRE_ZERO_BYTES_PER_ISSUE, 0);
        assert_eq!(TILE_W_BYTES % PRE_ZERO_BYTES_PER_ISSUE, 0);
    }

    // --- D5 emit-level helper tests ----------------------------------

    use kaio_core::emit::{Emit, PtxWriter};

    fn fresh_kernel() -> (RegisterAllocator, PtxKernel) {
        (
            RegisterAllocator::new(),
            PtxKernel::new("qkv_int4_d5_smoke"),
        )
    }

    fn emit_text(kernel: &PtxKernel) -> String {
        let mut w = PtxWriter::new();
        kernel.emit(&mut w).unwrap();
        w.finish()
    }

    fn emit_module_to_string(module: &kaio_core::ir::PtxModule) -> String {
        let mut w = PtxWriter::new();
        module.emit(&mut w).unwrap();
        w.finish()
    }

    #[test]
    fn pre_zero_emits_correct_issue_count_for_three_tiles() {
        // X: 4 issues/thread, W: 1 issue/thread (full-occupancy) + 1 predicated
        // scales issue per active thread = 5 + 1 = 6 st.shared.u32 per emit.
        // Plus: 1 bar.sync at the end.
        let (mut alloc, mut kernel) = fresh_kernel();
        let tile_x = alloc.alloc(PtxType::U32);
        let tile_w = alloc.alloc(PtxType::U32);
        let tile_s = alloc.alloc(PtxType::U32);
        let flat_tid = alloc.alloc(PtxType::U32);
        emit_pre_zero_shared_tiles_qkv_int4(
            &mut alloc,
            &mut kernel,
            tile_x,
            tile_w,
            tile_s,
            flat_tid,
        );
        let ptx = emit_text(&kernel);
        // 4 X + 1 W + 1 (predicated) scales = 6 st.shared.u32 instructions emitted.
        assert_eq!(
            ptx.matches("st.shared.u32").count(),
            6,
            "expected 6 st.shared.u32 (4 X + 1 W + 1 scales); got:\n{ptx}"
        );
        assert_eq!(
            ptx.matches("bar.sync").count(),
            1,
            "expected 1 bar.sync at end of pre-zero; got:\n{ptx}"
        );
    }

    #[test]
    fn w_packed_loader_emits_in_tile_predicate_and_label_suffixes() {
        // Three calls with Q/K/V should produce three uniquely-labelled skip
        // targets and contain in-tile setp.lt + n-edge setp.lt for each.
        let (mut alloc, mut kernel) = fresh_kernel();
        let w_base = alloc.alloc(PtxType::U64);
        let tile_w = alloc.alloc(PtxType::U32);
        let flat_tid = alloc.alloc(PtxType::U32);
        let block_col = alloc.alloc(PtxType::U32);
        let n = alloc.alloc(PtxType::U32);
        let k_words = alloc.alloc(PtxType::U32);
        for label in ["Q", "K", "V"] {
            emit_mw_load_tile_w_packed_int4_2x16(
                &mut alloc,
                &mut kernel,
                w_base,
                tile_w,
                flat_tid,
                block_col,
                n,
                k_words,
                label,
            );
        }
        let ptx = emit_text(&kernel);
        for label in [
            "WP_SKIP_I4_TILE_LOAD_Q:",
            "WP_SKIP_I4_TILE_LOAD_K:",
            "WP_SKIP_I4_TILE_LOAD_V:",
        ] {
            assert!(ptx.contains(label), "missing label `{label}` in:\n{ptx}");
        }
        // Two predicates per call (tile + n-edge) × 3 calls = 6 setp.lt.u32.
        assert_eq!(
            ptx.matches("setp.lt.u32").count(),
            6,
            "expected 6 setp.lt.u32 (tile + n-edge per Q/K/V); got:\n{ptx}"
        );
    }

    #[test]
    fn group_scales_loader_active_thread_count_is_eight() {
        // Predicate gate: flat_tid < BN_BLOCK/2 = 8.
        let (mut alloc, mut kernel) = fresh_kernel();
        let scales_base = alloc.alloc(PtxType::U64);
        let tile_s = alloc.alloc(PtxType::U32);
        let flat_tid = alloc.alloc(PtxType::U32);
        let block_col = alloc.alloc(PtxType::U32);
        let n = alloc.alloc(PtxType::U32);
        emit_cooperative_load_group_scales_int4(
            &mut alloc,
            &mut kernel,
            scales_base,
            tile_s,
            flat_tid,
            block_col,
            n,
            "Q",
        );
        let ptx = emit_text(&kernel);
        assert!(
            ptx.contains(", 8;"),
            "expected active-thread immediate 8 in setp; got:\n{ptx}"
        );
        assert!(
            ptx.contains("SCALES_SKIP_QKV_I4_Q:"),
            "expected labeled skip target; got:\n{ptx}"
        );
    }

    #[test]
    fn warp_quad_mma_int4_emits_two_mmas_one_dequant_and_eight_copy_backs() {
        // Rollback #1: 2 m-stripes × 1 n-stripe = 2 mma.sync per call.
        // 1 fragment-B dequant per call (1 n-stripe), reused across m-stripes.
        // 2 mmas × 4 frag_d copy-back = 8 mov.f32.
        fn fresh_grid(
            alloc: &mut RegisterAllocator,
        ) -> [[kaio_core::fragment::FragmentC; MMAS_PER_WARP_N as usize]; MMAS_PER_WARP_M as usize]
        {
            core::array::from_fn(|_| core::array::from_fn(|_| kaio_core::fragment::alloc_c(alloc)))
        }
        let (mut alloc, mut kernel) = fresh_kernel();
        let tile_x = alloc.alloc(PtxType::U32);
        let tile_w = alloc.alloc(PtxType::U32);
        let tile_s = alloc.alloc(PtxType::U32);
        let warp_row = alloc.alloc(PtxType::U32);
        let warp_col = alloc.alloc(PtxType::U32);
        let tid = alloc.alloc(PtxType::U32);
        let mut frag_c = fresh_grid(&mut alloc);
        emit_warp_quadrant_mma_int4_per_projection(
            &mut alloc,
            &mut kernel,
            tile_x,
            tile_w,
            tile_s,
            warp_row,
            warp_col,
            tid,
            &mut frag_c,
        );
        let ptx = emit_text(&kernel);
        assert_eq!(
            ptx.matches("mma.sync").count(),
            2,
            "expected 2 mma.sync (2×1 sub-tiles); got:\n{ptx}"
        );
        // INT4 dequant per nibble: shr.s32 (sign-extend canary) + cvt.rn.f32.s32 +
        // cvt.rn.f16.f32 + mul.f16 (scale fold). 1 dequant call → 4 nibbles.
        // shr.s32 should appear 4 times (sign-extend correctness).
        assert_eq!(
            ptx.matches("shr.s32").count(),
            4,
            "expected 4 shr.s32 sign-extend instructions per dequant; got:\n{ptx}"
        );
        // 8 mov.f32 copy-backs (2 mmas × 4 dst regs).
        assert_eq!(
            ptx.matches("mov.f32").count(),
            8,
            "expected 8 mov.f32 copy-backs; got:\n{ptx}"
        );
    }

    // --- D5 module-level tests ---------------------------------------

    #[test]
    fn build_qkv_project_int4_module_structure() {
        let module = build_qkv_project_int4_module("sm_89");
        let ptx = emit_module_to_string(&module);

        assert!(ptx.contains(".visible .entry qkv_project_int4("));
        assert!(ptx.contains(".shared .align 4 .b8 tile_x[2048]"));
        assert!(ptx.contains(".shared .align 4 .b8 tile_w[512]"));
        assert!(ptx.contains(".shared .align 4 .b8 tile_scales[32]"));
        assert!(ptx.contains("K_LOOP_QKV_INT4:"));

        let sub_tiles_per_warp = (MMAS_PER_WARP_M * MMAS_PER_WARP_N) as usize;
        let projections = 3usize;
        let total_sub_tiles = sub_tiles_per_warp * projections;

        assert!(ptx.contains("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"));
        let mma_count = ptx.matches("mma.sync.aligned.m16n8k16").count();
        assert_eq!(
            mma_count, total_sub_tiles,
            "expected {total_sub_tiles} mma.sync"
        );

        // Store helper: each call → 2 packed b32 stores. INT4 passes scale=None,
        // so no per-store mul.f32 (mul.f16 scale fold happens INSIDE the
        // dequant chain via emit_unpack_s4_x2_scale_to_f16_pair).
        let st_count = ptx.matches("st.global.u32").count();
        assert_eq!(
            st_count,
            total_sub_tiles * 2,
            "expected {} packed stores",
            total_sub_tiles * 2
        );
        // cvt.rn.f16.f32 fires in TWO places:
        //   (1) per-nibble dequant inside emit_unpack_s4_x2_scale_to_f16_pair
        //       — 4 nibbles per per-projection call × 3 projections = 12 cvts.
        //   (2) per-frag_c store-out narrow — 4 regs × 6 sub-tiles = 24 cvts.
        // Total = 12 + 24 = 36.
        let dequant_cvts = projections * 4;
        let store_cvts = total_sub_tiles * 4;
        let expected_cvts = dequant_cvts + store_cvts;
        let cvt_narrow = ptx.matches("cvt.rn.f16.f32").count();
        assert_eq!(
            cvt_narrow, expected_cvts,
            "expected {expected_cvts} cvt.rn.f16.f32 ({dequant_cvts} dequant + {store_cvts} store)"
        );
    }

    #[test]
    fn build_qkv_project_int4_module_emits_seven_barriers_per_k_tile() {
        // Pre-zero (1) + per-K-tile (1 X + 2 per projection × 3 = 7) = 8 static.
        let module = build_qkv_project_int4_module("sm_89");
        let ptx = emit_module_to_string(&module);
        assert_eq!(
            ptx.matches("bar.sync").count(),
            8,
            "expected 8 bar.sync (1 pre-zero + 7 per K-tile)"
        );
    }

    #[test]
    fn build_qkv_project_int4_module_declares_requested_sm() {
        let ptx_70 = emit_module_to_string(&build_qkv_project_int4_module("sm_70"));
        assert!(ptx_70.contains(".target sm_70"));
        let ptx_89 = emit_module_to_string(&build_qkv_project_int4_module("sm_89"));
        assert!(ptx_89.contains(".target sm_89"));
    }

    #[test]
    fn build_qkv_project_int4_module_validates_at_sm_80_and_above() {
        for sm in ["sm_80", "sm_89", "sm_90"] {
            let module = build_qkv_project_int4_module(sm);
            module
                .validate()
                .unwrap_or_else(|e| panic!("{sm} should validate; got {e}"));
        }
    }

    #[test]
    fn build_qkv_project_int4_module_emits_distinct_mma_destination_regs() {
        // Disjoint-register canary across 3 frag_c grids (Sprint 7.2 sign-extend
        // canary's tri-output sibling). Catches accidental grid aliasing.
        use std::collections::HashSet;
        let expected = (MMAS_PER_WARP_M * MMAS_PER_WARP_N * 3 * 4) as usize;
        let module = build_qkv_project_int4_module("sm_89");
        let ptx = emit_module_to_string(&module);

        let mut all_dst_regs: Vec<u32> = Vec::new();
        for line in ptx.lines() {
            let line = line.trim();
            if !line.contains("mma.sync.aligned.m16n8k16") {
                continue;
            }
            let Some(open) = line.find('{') else { continue };
            let Some(close) = line[open..].find('}') else {
                continue;
            };
            for tok in line[open + 1..open + close].split(',') {
                let tok = tok.trim();
                if let Some(rest) = tok.strip_prefix("%f")
                    && let Ok(idx) = rest.parse::<u32>()
                {
                    all_dst_regs.push(idx);
                }
            }
        }
        assert_eq!(all_dst_regs.len(), expected);
        let unique: HashSet<u32> = all_dst_regs.iter().copied().collect();
        assert_eq!(
            unique.len(),
            expected,
            "frag_c grids alias — {} duplicates",
            all_dst_regs.len() - unique.len()
        );
    }

    /// D6 host-API smoke test: launches `qkv_project_int4` with canonical
    /// shapes (M=64, N=16, K=128 = exactly one group). Verifies module loads,
    /// kernel launches, and outputs are written without driver error.
    /// Correctness vs reference is the D7 e2e suite's job.
    #[test]
    fn qkv_project_int4_launches_without_error() {
        let Ok(device) = make_device() else {
            return;
        };
        let m = BM_BLOCK;
        let n = BN_BLOCK;
        let k = GROUP_SIZE; // exactly one group
        let x = device.alloc_zeros::<half::f16>((m * k) as usize).unwrap();
        let w_packed_len = ((k / NIBBLES_PER_U32) * n) as usize;
        let w_q = device.alloc_zeros::<u32>(w_packed_len).unwrap();
        let w_k = device.alloc_zeros::<u32>(w_packed_len).unwrap();
        let w_v = device.alloc_zeros::<u32>(w_packed_len).unwrap();
        let scales_len = ((k / GROUP_SIZE) * n) as usize;
        let s_q = device.alloc_zeros::<half::f16>(scales_len).unwrap();
        let s_k = device.alloc_zeros::<half::f16>(scales_len).unwrap();
        let s_v = device.alloc_zeros::<half::f16>(scales_len).unwrap();
        let mut q_out = device.alloc_zeros::<half::f16>((m * n) as usize).unwrap();
        let mut k_out = device.alloc_zeros::<half::f16>((m * n) as usize).unwrap();
        let mut v_out = device.alloc_zeros::<half::f16>((m * n) as usize).unwrap();
        qkv_project_int4(
            &device, &x, &w_q, &w_k, &w_v, &s_q, &s_k, &s_v, &mut q_out, &mut k_out, &mut v_out, m,
            n, k, GROUP_SIZE,
        )
        .expect("qkv_project_int4 launch failed on canonical shape");
        device.stream().synchronize().expect("device sync failed");
        let q_host = q_out.to_host(&device).expect("dtoh");
        let zero = half::f16::from_f32(0.0);
        assert!(
            q_host.iter().all(|h: &half::f16| *h == zero),
            "expected all-zero Q output for zero inputs"
        );
    }

    /// ptxas_verify offline gate for the full `qkv_project_int4` module.
    /// `#[ignore]` so host-only CI stays green; runs both sm_80 + sm_89 by
    /// default (override via `KAIO_SM_TARGET`). Pass criterion: ≤ 64 regs/
    /// thread, 0 spills.
    #[test]
    #[ignore]
    fn ptxas_verify_qkv_project_int4() {
        if std::process::Command::new("ptxas")
            .arg("--version")
            .output()
            .is_err()
        {
            eprintln!("NOTE: ptxas not found in PATH — skipping qkv_project_int4 verification");
            return;
        }

        let sms: Vec<String> = if let Ok(sm) = std::env::var("KAIO_SM_TARGET") {
            vec![sm]
        } else {
            vec!["sm_80".to_string(), "sm_89".to_string()]
        };

        for sm in &sms {
            let module = build_qkv_project_int4_module(sm);
            let ptx = emit_module_to_string(&module);
            let tmp = std::env::temp_dir().join(format!("kaio_qkv_project_int4_{sm}.ptx"));
            std::fs::write(&tmp, &ptx).expect("failed to write temp PTX");

            let output = std::process::Command::new("ptxas")
                .args(["--gpu-name", sm, "--verbose"])
                .arg(tmp.to_str().unwrap())
                .output()
                .expect("failed to run ptxas");
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let _ = std::fs::remove_file(&tmp);

            assert!(
                output.status.success(),
                "ptxas FAILED for qkv_project_int4 ({sm}):\nstdout: {stdout}\nstderr: {stderr}\n\n=== PTX (first 8000) ===\n{}",
                &ptx[..ptx.len().min(8000)]
            );

            let combined = format!("{stdout}\n{stderr}");
            let regs = parse_after(&combined, "Used ", " registers");
            let spill_stores = parse_after(&combined, "bytes stack frame, ", " bytes spill stores");
            let spill_loads = parse_after(&combined, "spill stores, ", " bytes spill loads");

            eprintln!(
                "=== qkv_project_int4 ptxas baseline ({sm}) ===\n\
                 regs/thread  : {regs:?}\n\
                 spill stores : {spill_stores:?}\n\
                 spill loads  : {spill_loads:?}"
            );

            match regs {
                Some(n) => assert!(
                    n <= 64,
                    "qkv_project_int4 ({sm}) reports {n} registers > 64 — Rollback #1 not enough; \
                     may need further sub-tile reduction or skeleton refactoring"
                ),
                None => {
                    panic!("could not parse register count\nstdout:\n{stdout}\nstderr:\n{stderr}")
                }
            }
            assert_eq!(
                spill_stores,
                Some(0),
                "qkv_project_int4 ({sm}) has spill stores"
            );
            assert_eq!(
                spill_loads,
                Some(0),
                "qkv_project_int4 ({sm}) has spill loads"
            );

            eprintln!("ptxas verification PASSED for qkv_project_int4 ({sm})");
        }
    }

    fn parse_after(haystack: &str, prefix: &str, suffix: &str) -> Option<u32> {
        for line in haystack.lines() {
            if let Some(after) = line.split(prefix).nth(1)
                && let Some(num) = after.split(suffix).next()
                && let Ok(n) = num.trim().parse::<u32>()
            {
                return Some(n);
            }
        }
        None
    }
}
