#![allow(dead_code)]
// WIP module — helpers + constants are wired up progressively across D4.1/D4.2/D4.3. Final sprint commit removes this.

//! INT4 symmetric dequantize-matmul — `mma.sync.m16n8k16.f16.f16.f32`
//! with fused unpack + sign-extend + cvt chain feeding fragment B.
//!
//! Sprint 7.2. Shipping in stages across D1–D8; this file currently
//! carries the D2 helper (`emit_unpack_s4_x8_scale_to_f16x8`) plus its
//! triple-layer sign-extend canary. D4 will add the full K-tile loop
//! + warp quadrant assembly.
//!
//! # DEQUANT-F16 path (why)
//!
//! No native `m16n8k?.s4.s4.s32` shape on sm_80+ — INT4 operands must
//! be dequantized before the mma. Group-scale quantization (one f16
//! scale per group of 128 K-elements, varying along K) additionally
//! forces pre-mma scale application, ruling out the "reuse INT8
//! kernel, apply post-accumulation scalar" shortcut that worked for
//! sprint 7.1's single-scalar INT8. Therefore:
//!
//! ```text
//! u32 (8 packed s4)
//!   └─ shl.b32 %tmp, %packed, (28 - 4i)    // sign bit of nibble i at MSB
//!   └─ shr.s32 %ext, %tmp, 28              // arithmetic shift → signed i32 in [-8, +7]
//!   └─ cvt.rn.f32.s32 %f, %ext
//!   └─ cvt.rn.f16.f32 %h_raw, %f
//!   └─ mul.f16 %h, %h_raw, %scale          // scale fold, pre-mma
//!   └─ mov.b32 %b32, {%h_lo, %h_hi}        // pack for fragment B feed
//! ```
//!
//! Sign-extend is the correctness-critical step. `shr.s32` (arithmetic
//! right shift) drags the sign bit across the upper 28 bits; an
//! accidental `shr.u32` would zero-extend, silently turning every
//! negative INT4 weight into a positive one. Triple-layer canary:
//! emit-level token assertion (this file), ptxas_verify offline,
//! GPU e2e boundary-value tests (D6).

use kaio::prelude::*;
use kaio_core::instr::control::{CmpOp, ControlOp};
use kaio_core::instr::memory::MemoryOp;
use kaio_core::instr::{ArithOp, MadMode};
use kaio_core::ir::{Operand, PtxInstruction, PtxKernel, Register, RegisterAllocator};
use kaio_core::types::PtxType;
use kaio_core::types::RegKind;

// --- mma.sync.m16n8k16 instance shape (f16 inputs, f32 accumulator) ---
const BM: u32 = 16; // mma m dim
const BN: u32 = 8; // mma n dim
const BK: u32 = 16; // mma k dim

// --- Multi-warp block tiling (mirrors matmul_int8 / matmul_tc) ---
const BM_BLOCK: u32 = 64; // output rows per block
const BN_BLOCK: u32 = 64; // output cols per block
const WARP_QUAD_M: u32 = 32; // rows per warp quadrant
const WARP_QUAD_N: u32 = 32; // cols per warp quadrant
const MMAS_PER_WARP_M: u32 = WARP_QUAD_M / BM; // 2
const MMAS_PER_WARP_N: u32 = WARP_QUAD_N / BN; // 4
const WARPS_PER_BLOCK: u32 = 4;
const THREADS_PER_BLOCK: u32 = WARPS_PER_BLOCK * 32; // 128

// --- INT4 packing + group scales ---
const NIBBLES_PER_U32: u32 = 8;
const GROUP_SIZE: u32 = 128;

// --- K-tile granularity (Codex round 3 lock) ---
const K_TILE_SHARED: u32 = 16;
const K_TILE_GROUP_RATIO: u32 = GROUP_SIZE / K_TILE_SHARED; // 8 — group transitions every 8 K-tiles

// --- Shared tile sizes ---
const BYTES_PER_F16: u32 = 2;
const BYTES_PER_B32: u32 = 4;
/// A tile: 64 rows × 16 f16 row-major. Row stride = 16 × 2 = 32 B.
pub(crate) const TILE_A_ROW_STRIDE_BYTES: u32 = K_TILE_SHARED * BYTES_PER_F16;
pub(crate) const TILE_A_BYTES: u32 = BM_BLOCK * TILE_A_ROW_STRIDE_BYTES; // 2048
/// B tile: 64 cols × 2 u32 col-major, padded to 12 B col-stride for
/// bank-conflict relief (same `+4` logic as `matmul_int8_kernel.rs`).
/// Natural 8 B stride would give gcd(8,128)=8 → 16 distinct bank
/// patterns → 4-way conflict. 12 B stride → gcd(12,128)=4 → 32
/// distinct patterns → conflict-free.
pub(crate) const TILE_B_PACKED_WORDS_PER_COL: u32 = K_TILE_SHARED / NIBBLES_PER_U32; // 2
pub(crate) const TILE_B_COL_STRIDE_BYTES: u32 = TILE_B_PACKED_WORDS_PER_COL * BYTES_PER_B32 + 4; // 12
/// Data region = 64 cols × 12 B = 768 B. Pre-zero requires a multiple of
/// `THREADS_PER_BLOCK * 4 = 512 B`, so round up to 1024 B. Tail bytes
/// (768..1024) are never read by the kernel — the cooperative B loader
/// addresses only the 12 B-stride col region.
pub(crate) const TILE_B_BYTES: u32 = BN_BLOCK * TILE_B_COL_STRIDE_BYTES
    + (THREADS_PER_BLOCK * 4 - (BN_BLOCK * TILE_B_COL_STRIDE_BYTES) % (THREADS_PER_BLOCK * 4))
        % (THREADS_PER_BLOCK * 4); // 1024
/// Group-scale tile: one f16 per output column, 64 f16 per tile.
/// Reloaded once per group transition (every 8 K-tiles).
pub(crate) const TILE_SCALES_BYTES: u32 = BN_BLOCK * BYTES_PER_F16; // 128

/// Validate shape + alignment preconditions for `matmul_int4`.
///
/// Constraints:
/// - M, N, K all non-zero.
/// - `K % GROUP_SIZE == 0` (also implies `K % K_TILE_SHARED == 0`).
/// - `group_size == GROUP_SIZE` (parameterization is a follow-up sprint).
/// - Buffer-size sanity: `a >= M*K`, `b_packed >= N*(K/8)`, `scales >= (K/group_size)*N`, `c >= M*N`.
///
/// M and N may be any positive value — edge-tile predication in the
/// kernel handles ragged output.
#[allow(dead_code)] // wired up in D5
pub(crate) fn validate_dims_int4(
    a: &GpuBuffer<half::f16>,
    b_packed: &GpuBuffer<u32>,
    scales: &GpuBuffer<half::f16>,
    c: &GpuBuffer<f32>,
    m: u32,
    n: u32,
    k: u32,
    group_size: u32,
) -> Result<()> {
    if m == 0 || n == 0 || k == 0 {
        return Err(KaioError::InvalidConfig(
            "matmul_int4: M, N, K dimensions must be non-zero".to_string(),
        ));
    }
    if group_size != GROUP_SIZE {
        return Err(KaioError::InvalidConfig(format!(
            "matmul_int4: group_size must be {GROUP_SIZE} (got {group_size}). Non-128 \
             group sizes are deferred to a follow-up sprint."
        )));
    }
    if !k.is_multiple_of(GROUP_SIZE) {
        return Err(KaioError::InvalidConfig(format!(
            "matmul_int4: K must be a multiple of group_size={GROUP_SIZE} (got {k}). \
             Partial groups are not supported."
        )));
    }

    let mk = (m as usize) * (k as usize);
    let packed_words = ((k as usize) / (NIBBLES_PER_U32 as usize)) * (n as usize);
    let num_groups = (k as usize) / (group_size as usize);
    let scales_cells = num_groups * (n as usize);
    let mn = (m as usize) * (n as usize);

    if a.len() < mk {
        return Err(KaioError::InvalidConfig(format!(
            "matmul_int4: A buffer too small: need {mk} f16 ({m}×{k}), got {}",
            a.len()
        )));
    }
    if b_packed.len() < packed_words {
        return Err(KaioError::InvalidConfig(format!(
            "matmul_int4: B packed buffer too small: need {packed_words} u32 \
             ({n} cols × {} K-words), got {}",
            (k as usize) / (NIBBLES_PER_U32 as usize),
            b_packed.len()
        )));
    }
    if scales.len() < scales_cells {
        return Err(KaioError::InvalidConfig(format!(
            "matmul_int4: scales buffer too small: need {scales_cells} f16 \
             ({num_groups} groups × {n} cols), got {}",
            scales.len()
        )));
    }
    if c.len() < mn {
        return Err(KaioError::InvalidConfig(format!(
            "matmul_int4: C buffer too small: need {mn} f32 ({m}×{n}), got {}",
            c.len()
        )));
    }
    Ok(())
}

/// Cooperative load of the 64×16 f16 A block tile from global (row-major)
/// into shared (row-major). 128 threads × 4 `ld.global.b32` = 512 b32 =
/// 1024 f16 = 64 × 16 ✓.
///
/// **Per-thread layout:** `lane_base = flat_tid * 4` (b32 units),
/// `flat = lane_base + i`, `row_in_tile = flat / 8` (32 B/row = 8 b32),
/// `col_b32 = flat % 8`, `col_bytes = col_b32 * 4`. 2 threads cooperate
/// per row, each loading 4 b32 = 8 f16 (half a row).
///
/// **Edge handling:** Caller pre-zeroes `tile_a` once at kernel start.
/// OOB threads (row_global >= M) skip all 4 issues via
/// `@!p bra A_SKIP_I4_<suffix>`; shared slots stay zero.
#[allow(dead_code)] // wired up in D4.3
pub(crate) fn emit_mw_load_tile_a_f16_64x16(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    a_block_base_global: Register, // u64 — A[block_row, k_tile*16]
    tile_a_shared: Register,       // u32
    flat_tid: Register,            // u32 — 0..128
    block_row: Register,           // u32
    m: Register,                   // u32
    k_bytes: Register,             // u32 — K * 2 (f16 is 2 bytes)
    label_suffix: &str,
) {
    let skip_label = if label_suffix.is_empty() {
        "A_SKIP_I4_TILE_LOAD".to_string()
    } else {
        format!("A_SKIP_I4_TILE_LOAD_{label_suffix}")
    };

    // lane_base = flat_tid * 4 (b32 units per thread)
    let lane_base = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: lane_base,
        lhs: Operand::Reg(flat_tid),
        rhs: Operand::ImmU32(4),
        ty: PtxType::U32,
    }));

    // row_in_tile = lane_base / 8  (each row = 8 b32 = 32 B)
    let row_in_tile = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: row_in_tile,
        lhs: Operand::Reg(lane_base),
        rhs: Operand::ImmU32(8),
        ty: PtxType::U32,
    }));

    // row_global = block_row + row_in_tile
    let row_global = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: row_global,
        lhs: Operand::Reg(block_row),
        rhs: Operand::Reg(row_in_tile),
        ty: PtxType::U32,
    }));

    // @!p bra A_SKIP  if row_global >= M
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
        // col_b32 = flat % 8
        let col_b32 = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Rem {
            dst: col_b32,
            lhs: Operand::Reg(flat),
            rhs: Operand::ImmU32(8),
            ty: PtxType::U32,
        }));
        // col_bytes = col_b32 * 4
        let col_bytes = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Mul {
            dst: col_bytes,
            lhs: Operand::Reg(col_b32),
            rhs: Operand::ImmU32(4),
            ty: PtxType::U32,
        }));

        // shared_off = row_in_tile * TILE_A_ROW_STRIDE_BYTES + col_bytes
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
        let per_thread_off = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: per_thread_off,
            lhs: Operand::Reg(row_off64),
            rhs: Operand::Reg(col_bytes_u64),
            ty: PtxType::U64,
        }));
        let global_addr = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: global_addr,
            lhs: Operand::Reg(a_block_base_global),
            rhs: Operand::Reg(per_thread_off),
            ty: PtxType::U64,
        }));

        // ld.global.b32 + st.shared.b32 (2 packed f16)
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

/// Cooperative load of the 2 × 64 packed-`u32` B block tile from global
/// (col-major, `b_packed[K/8, N]` where each u32 packs 8 signed INT4
/// nibbles K-contiguously) into shared (col-major, 12 B padded col-stride).
///
/// 128 threads × 1 `ld.global.u32` issue per thread = 128 u32 =
/// `TILE_B_PACKED_WORDS_PER_COL × BN_BLOCK` = 2 × 64 ✓.
///
/// **Per-thread layout:** `col_in_tile = flat_tid / 2` (2 threads per
/// col cover the 2 K-words in this K-tile), `word_idx_local = flat_tid % 2`
/// (0 or 1). One col-bounds check gates both issues per thread (col_in_tile
/// is constant per thread).
///
/// **Edge handling:** Caller pre-zeroes `tile_b` once at kernel start.
/// OOB threads (col_global >= N) skip via `@!p bra B_SKIP_I4_<suffix>`;
/// shared slots stay zero.
#[allow(dead_code)] // wired up in D4.3
pub(crate) fn emit_mw_load_tile_b_packed_2x64(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    b_packed_base_global: Register, // u64 — &b_packed[k_tile*2, block_col] region start
    tile_b_shared: Register,        // u32
    flat_tid: Register,             // u32 — 0..128
    block_col: Register,            // u32
    n: Register,                    // u32
    k_words: Register,              // u32 — K / 8 (rows of b_packed storage)
    label_suffix: &str,
) {
    let skip_label = if label_suffix.is_empty() {
        "B_SKIP_I4_TILE_LOAD".to_string()
    } else {
        format!("B_SKIP_I4_TILE_LOAD_{label_suffix}")
    };

    // col_in_tile = flat_tid / 2
    let col_in_tile = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: col_in_tile,
        lhs: Operand::Reg(flat_tid),
        rhs: Operand::ImmU32(2),
        ty: PtxType::U32,
    }));

    // word_idx_local = flat_tid % 2 (0 or 1)
    let word_idx_local = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Rem {
        dst: word_idx_local,
        lhs: Operand::Reg(flat_tid),
        rhs: Operand::ImmU32(2),
        ty: PtxType::U32,
    }));

    // col_global = block_col + col_in_tile
    let col_global = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: col_global,
        lhs: Operand::Reg(block_col),
        rhs: Operand::Reg(col_in_tile),
        ty: PtxType::U32,
    }));

    // @!p bra B_SKIP  if col_global >= N
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

    // shared_off = col_in_tile * TILE_B_COL_STRIDE_BYTES + word_idx_local * 4
    let col_stride_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: col_stride_off,
        lhs: Operand::Reg(col_in_tile),
        rhs: Operand::ImmU32(TILE_B_COL_STRIDE_BYTES),
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
        lhs: Operand::Reg(tile_b_shared),
        rhs: Operand::Reg(shared_off),
        ty: PtxType::U32,
    }));

    // global_addr = b_packed_base + (word_idx_local + col_in_tile * k_words) * 4
    // b_packed[K/8, N] col-major — col `n` stores (K/8) u32 contiguously.
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
        lhs: Operand::Reg(b_packed_base_global),
        rhs: Operand::Reg(global_byte_off),
        ty: PtxType::U64,
    }));

    // ld.global.u32 + st.shared.b32 (one packed-INT4 word)
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

/// Cooperative load of the 64 f16 group-scale slice for the current
/// group transition. `scales[num_groups, N]` row-major (the layout
/// chosen for contiguous-N access per group — see D4.1 note below).
///
/// Per block, at each group transition, reads `scales[g, block_col..block_col+64]`
/// as 32 `.b32` words (2 f16 packed per b32). Lanes 0..31 cooperate;
/// lanes 32..127 idle through the skip label.
///
/// **Per-thread layout:** `active = flat_tid < 32`, each active lane
/// loads `scales[g, block_col + flat_tid*2 : flat_tid*2+2]` (2 f16 =
/// 1 b32). N-edge predication drops loads where `block_col + flat_tid*2
/// >= N`; the OOB f16 slots in shared stay at their pre-zero value.
///
/// **Edge handling:** Caller pre-zeroes `tile_scales` at kernel start.
/// OOB threads (col >= N OR flat_tid >= 32) skip via
/// `@!p bra SCALES_SKIP_<suffix>`.
#[allow(dead_code)] // wired up in D4.3
pub(crate) fn emit_cooperative_load_group_scales_64(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    scales_base_global: Register, // u64 — &scales[g * N] region start
    tile_scales_shared: Register, // u32
    flat_tid: Register,           // u32 — 0..128
    block_col: Register,          // u32
    n: Register,                  // u32
    label_suffix: &str,
) {
    let skip_label = if label_suffix.is_empty() {
        "SCALES_SKIP_I4".to_string()
    } else {
        format!("SCALES_SKIP_I4_{label_suffix}")
    };

    // @!p_active bra SCALES_SKIP  if flat_tid >= 32
    let p_active = alloc.alloc(PtxType::Pred);
    kernel.push(PtxInstruction::Control(ControlOp::SetP {
        dst: p_active,
        cmp_op: CmpOp::Lt,
        lhs: Operand::Reg(flat_tid),
        rhs: Operand::ImmU32(32),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Control(ControlOp::BraPred {
        pred: p_active,
        target: skip_label.clone(),
        negate: true,
    }));

    // col_pair_local = flat_tid * 2 (first f16 in the b32 pair)
    let col_pair_local = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: col_pair_local,
        lhs: Operand::Reg(flat_tid),
        rhs: Operand::ImmU32(2),
        ty: PtxType::U32,
    }));

    // col_global = block_col + col_pair_local
    let col_global = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: col_global,
        lhs: Operand::Reg(block_col),
        rhs: Operand::Reg(col_pair_local),
        ty: PtxType::U32,
    }));

    // @!p bra SCALES_SKIP  if col_global >= N
    // (lanes whose col_pair_local would straddle N still load — second f16 may be OOB;
    //  pre-zeroed shared handles that since the tile_scales slot was zeroed.
    //  Simpler: gate on col_global >= N for the first f16, accept loading one OOB
    //  b16 in the pair. The OOB half of a partial pair is padded with zero
    //  via pre-zero.)
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

    // shared_off = flat_tid * 4 (32 b32 slots, 4 B each)
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

    // global_addr = scales_base + col_pair_local * 2 (f16 = 2 B)
    let col_bytes = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: col_bytes,
        lhs: Operand::Reg(col_pair_local),
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

    // ld.global.b32 + st.shared.b32 (one f16-pair word)
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

// Silence unused-import noise; RegKind will be used by D4.2's fragment B
// per-lane dequant.
#[allow(dead_code)]
const _: RegKind = RegKind::R;

// ============================================================================
// D4.2 — narrow 2-nibble unpack helper + fragment-B per-lane dequant + mma
// ============================================================================

/// Emit the unpack + sign-extend + dequant + pack chain for exactly
/// two nibbles of a packed `u32`, producing one `.b32` register
/// holding `{f16_lo, f16_hi}`.
///
/// Siblings with [`emit_unpack_s4_x8_scale_to_f16x8`] but only emits
/// two nibbles' worth of work. Used by the per-lane fragment-B dequant
/// path where each warp lane produces exactly 4 f16 values per
/// fragment B (2 nibbles × 2 u32 words).
///
/// `shl_count_lo` / `shl_count_hi` are the `shl.b32` shift counts
/// that place the target nibble's MSB at bit 31 — i.e.
/// `shl_count = 28 - 4 * nibble_index`. Passed as `Operand` so the
/// caller can use either an immediate (for a compile-time nibble
/// position) or a register (for a runtime per-lane `tig`).
fn emit_unpack_s4_x2_scale_to_f16_pair(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    packed: Register,
    scale_f16: Register,
    shl_count_lo: Operand,
    shl_count_hi: Operand,
) -> Register {
    let emit_nibble =
        |alloc: &mut RegisterAllocator, kernel: &mut PtxKernel, shl_count: Operand| -> Register {
            let tmp = alloc.alloc(PtxType::U32);
            kernel.push(PtxInstruction::Arith(ArithOp::Shl {
                dst: tmp,
                lhs: Operand::Reg(packed),
                rhs: shl_count,
                ty: PtxType::U32,
            }));
            let ext = alloc.alloc(PtxType::S32);
            kernel.push(PtxInstruction::Arith(ArithOp::Shr {
                dst: ext,
                lhs: Operand::Reg(tmp),
                rhs: Operand::ImmU32(28),
                ty: PtxType::S32,
            }));
            let f32_reg = alloc.alloc(PtxType::F32);
            kernel.push(PtxInstruction::Cvt {
                dst: f32_reg,
                src: ext,
                dst_ty: PtxType::F32,
                src_ty: PtxType::S32,
            });
            let h_raw = alloc.alloc(PtxType::F16);
            kernel.push(PtxInstruction::Cvt {
                dst: h_raw,
                src: f32_reg,
                dst_ty: PtxType::F16,
                src_ty: PtxType::F32,
            });
            let h = alloc.alloc(PtxType::F16);
            kernel.push(PtxInstruction::Arith(ArithOp::Mul {
                dst: h,
                lhs: Operand::Reg(h_raw),
                rhs: Operand::Reg(scale_f16),
                ty: PtxType::F16,
            }));
            h
        };

    let h_lo = emit_nibble(alloc, kernel, shl_count_lo);
    let h_hi = emit_nibble(alloc, kernel, shl_count_hi);

    let packed_pair = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::MovPack {
        dst: packed_pair,
        srcs: vec![h_lo, h_hi],
        ty: PtxType::U32,
    });
    packed_pair
}

/// Per-lane dequant for one `FragmentB_M16N8K16` at the current
/// warp-quadrant N-stripe. Reads 2 `u32` words + 1 `f16` scale from
/// shared and runs the narrow unpack helper twice.
///
/// Fragment B layout (PTX ISA §9.7.13.5.8.1 for m16n8k16.f16): lane
/// `l ∈ 0..32` owns col `l/4` and rows `{2*tig, 2*tig+1}` for reg[0] +
/// `{2*tig+8, 2*tig+9}` for reg[1], where `tig = l%4`. Per fragment-B
/// per lane: 4 nibbles at positions `(2*tig, 2*tig+1)` → shift counts
/// `(28 - 8*tig, 24 - 8*tig)` applied to both u32 words.
fn emit_fragment_b_int4_per_lane(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    tile_b_shared: Register,
    tile_scales_shared: Register,
    n_stripe_col_base: Register,
    lane_within_warp: Register,
) -> kaio_core::fragment::FragmentB {
    // group_id = lane / 4   (0..8, col within the 8-col fragment)
    let group_id = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: group_id,
        lhs: Operand::Reg(lane_within_warp),
        rhs: Operand::ImmU32(4),
        ty: PtxType::U32,
    }));
    // tig = lane % 4
    let tig = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Rem {
        dst: tig,
        lhs: Operand::Reg(lane_within_warp),
        rhs: Operand::ImmU32(4),
        ty: PtxType::U32,
    }));

    // col_abs = n_stripe_col_base + group_id  (col within the block tile, 0..64)
    let col_abs = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: col_abs,
        lhs: Operand::Reg(n_stripe_col_base),
        rhs: Operand::Reg(group_id),
        ty: PtxType::U32,
    }));

    // Load 2 u32 words from tile_b for col_abs.
    let col_stride_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: col_stride_off,
        lhs: Operand::Reg(col_abs),
        rhs: Operand::ImmU32(TILE_B_COL_STRIDE_BYTES),
        ty: PtxType::U32,
    }));
    let word0_addr = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: word0_addr,
        lhs: Operand::Reg(tile_b_shared),
        rhs: Operand::Reg(col_stride_off),
        ty: PtxType::U32,
    }));
    let word1_addr = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: word1_addr,
        lhs: Operand::Reg(word0_addr),
        rhs: Operand::ImmU32(BYTES_PER_B32),
        ty: PtxType::U32,
    }));

    let u32_lo = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdShared {
        dst: u32_lo,
        addr: word0_addr,
        ty: PtxType::U32,
    }));
    let u32_hi = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdShared {
        dst: u32_hi,
        addr: word1_addr,
        ty: PtxType::U32,
    }));

    // Load scale = tile_scales[col_abs] (f16, stride 2 B).
    let scale_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: scale_off,
        lhs: Operand::Reg(col_abs),
        rhs: Operand::ImmU32(BYTES_PER_F16),
        ty: PtxType::U32,
    }));
    let scale_addr = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: scale_addr,
        lhs: Operand::Reg(tile_scales_shared),
        rhs: Operand::Reg(scale_off),
        ty: PtxType::U32,
    }));
    let scale_f16 = alloc.alloc(PtxType::F16);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdShared {
        dst: scale_f16,
        addr: scale_addr,
        ty: PtxType::F16,
    }));

    // shl counts per tig: lo = 28 - 8*tig, hi = lo - 4 = 24 - 8*tig.
    let tig8 = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: tig8,
        lhs: Operand::Reg(tig),
        rhs: Operand::ImmU32(8),
        ty: PtxType::U32,
    }));
    let shl_count_lo = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Sub {
        dst: shl_count_lo,
        lhs: Operand::ImmU32(28),
        rhs: Operand::Reg(tig8),
        ty: PtxType::U32,
    }));
    let shl_count_hi = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Sub {
        dst: shl_count_hi,
        lhs: Operand::Reg(shl_count_lo),
        rhs: Operand::ImmU32(4),
        ty: PtxType::U32,
    }));

    let reg0 = emit_unpack_s4_x2_scale_to_f16_pair(
        alloc,
        kernel,
        u32_lo,
        scale_f16,
        Operand::Reg(shl_count_lo),
        Operand::Reg(shl_count_hi),
    );
    let reg1 = emit_unpack_s4_x2_scale_to_f16_pair(
        alloc,
        kernel,
        u32_hi,
        scale_f16,
        Operand::Reg(shl_count_lo),
        Operand::Reg(shl_count_hi),
    );

    kaio_core::fragment::FragmentB { regs: [reg0, reg1] }
}

/// Per-warp-quadrant mma pipeline: for each of `MMAS_PER_WARP_N=4`
/// N-stripes, per-lane dequant a fresh fragment B, then per-M-stripe
/// load fragment A from shared and run `mma.sync.m16n8k16.f16.f16.f32`,
/// accumulating into `frag_c_grid` in-place.
#[allow(clippy::too_many_arguments)] // all args are load-bearing kernel state
fn emit_warp_quadrant_mma_int4(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    tile_a_shared: Register,
    tile_b_shared: Register,
    tile_scales_shared: Register,
    warp_quad_row_base_in_tile_a: Register, // warp_quadrant_m_index * WARP_QUAD_M (0 or 32)
    warp_quad_col_base_in_tile_b: Register, // warp_quadrant_n_index * WARP_QUAD_N (0 or 32)
    tid_x_in_warp: Register,                // tid.x % 32
    frag_c_grid: &mut [[kaio_core::fragment::FragmentC; MMAS_PER_WARP_N as usize];
             MMAS_PER_WARP_M as usize],
) {
    use kaio_core::fragment::load_fragment_a_m16n8k16_shared_row;
    use kaio_core::instr::{MmaShape, TensorCoreOp};

    // Pre-compute the warp-quadrant row-offset in bytes (same across all n-stripes).
    let row_offset_bytes = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: row_offset_bytes,
        lhs: Operand::Reg(warp_quad_row_base_in_tile_a),
        rhs: Operand::ImmU32(TILE_A_ROW_STRIDE_BYTES),
        ty: PtxType::U32,
    }));

    for n_stripe in 0..MMAS_PER_WARP_N {
        // n_stripe_col_base = warp_quad_col_base + n_stripe * BN
        let n_stripe_col_base = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: n_stripe_col_base,
            lhs: Operand::Reg(warp_quad_col_base_in_tile_b),
            rhs: Operand::ImmU32(n_stripe * BN),
            ty: PtxType::U32,
        }));

        let frag_b = emit_fragment_b_int4_per_lane(
            alloc,
            kernel,
            tile_b_shared,
            tile_scales_shared,
            n_stripe_col_base,
            tid_x_in_warp,
        );

        for m_stripe in 0..MMAS_PER_WARP_M {
            // Each m-stripe points to a BM=16 rows slice inside the
            // warp quadrant (which is WARP_QUAD_M=32 rows). Shift the
            // tile_a base so the fragment loader reads the right 16 rows.
            let m_stripe_shift = alloc.alloc(PtxType::U32);
            kernel.push(PtxInstruction::Arith(ArithOp::Add {
                dst: m_stripe_shift,
                lhs: Operand::Reg(row_offset_bytes),
                rhs: Operand::ImmU32(m_stripe * BM * TILE_A_ROW_STRIDE_BYTES),
                ty: PtxType::U32,
            }));
            let shifted_tile_a = alloc.alloc(PtxType::U32);
            kernel.push(PtxInstruction::Arith(ArithOp::Add {
                dst: shifted_tile_a,
                lhs: Operand::Reg(tile_a_shared),
                rhs: Operand::Reg(m_stripe_shift),
                ty: PtxType::U32,
            }));

            let frag_a = load_fragment_a_m16n8k16_shared_row(
                alloc,
                kernel,
                shifted_tile_a,
                tid_x_in_warp,
                TILE_A_ROW_STRIDE_BYTES,
                None,
            );

            // Accumulate: frag_d = frag_a * frag_b + frag_c; then copy d→c.
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

/// Emit the unpack + sign-extend + dequant + pack chain for one `u32`
/// of packed signed INT4 weights, producing 4 `.b32` registers each
/// holding two packed f16 values ready for `mma.sync.m16n8k16.f16`
/// fragment B.
///
/// # Contract
///
/// **All 8 nibbles in `packed` are assumed to share the same
/// `scale_f16` value** — the caller is responsible for grouping `u32`
/// loads to group-scale boundaries. Under the Sprint 7.2 `AD3` + D3
/// layout (group_size = 128, K_tile_shared = 16), every `u32` carves
/// 8 contiguous K-elements entirely within one group, so this holds
/// by construction. A future sprint that relaxes group_size would
/// extend this signature, not rewrite it.
///
/// # PTX semantics
///
/// For each nibble `i ∈ 0..8`:
///
/// ```text
/// shl.b32 %tmp_i, %packed, (28 - 4*i);     // nibble-i MSB → bit 31
/// shr.s32 %ext_i, %tmp_i, 28;              // arithmetic: sign-extend → signed i32
/// cvt.rn.f32.s32 %f_i, %ext_i;             // int → f32
/// cvt.rn.f16.f32 %h_raw_i, %f_i;           // f32 → f16
/// mul.f16 %h_i, %h_raw_i, %scale_f16;      // scale fold
/// ```
///
/// The 8 dequanted f16 values are then packed low-high into 4 `.b32`
/// registers:
///
/// ```text
/// mov.b32 %b32_0, {%h_0, %h_1};
/// mov.b32 %b32_1, {%h_2, %h_3};
/// mov.b32 %b32_2, {%h_4, %h_5};
/// mov.b32 %b32_3, {%h_6, %h_7};
/// ```
///
/// # Parameters
///
/// - `alloc` — register allocator to draw intermediates from.
/// - `kernel` — kernel body to push instructions into.
/// - `packed` — `.b32` register holding 8 signed-INT4 nibbles, lane `i`
///   at bits `[4i..4i+4)` per the Sprint 7.2 packing convention.
/// - `scale_f16` — `.f16` register holding the group scale for all 8
///   nibbles.
///
/// # Returns
///
/// Four `.b32` registers suitable for direct use in a
/// `FragmentB_M16N8K16` feed (positions `[pair_01, pair_23, pair_45,
/// pair_67]`).
#[allow(dead_code)] // wired up in D4
pub(crate) fn emit_unpack_s4_x8_scale_to_f16x8(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    packed: Register,
    scale_f16: Register,
) -> [Register; 4] {
    let mut h_out: [Option<Register>; 8] = Default::default();

    for i in 0..8u32 {
        // shl.b32 %tmp, %packed, (28 - 4*i);
        let tmp = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Shl {
            dst: tmp,
            lhs: Operand::Reg(packed),
            rhs: Operand::ImmU32(28 - 4 * i),
            ty: PtxType::U32,
        }));
        // shr.s32 %ext, %tmp, 28;   // ARITHMETIC — sign-extends nibble
        let ext = alloc.alloc(PtxType::S32);
        kernel.push(PtxInstruction::Arith(ArithOp::Shr {
            dst: ext,
            lhs: Operand::Reg(tmp),
            rhs: Operand::ImmU32(28),
            ty: PtxType::S32,
        }));
        // cvt.rn.f32.s32 %f, %ext;
        let f32_reg = alloc.alloc(PtxType::F32);
        kernel.push(PtxInstruction::Cvt {
            dst: f32_reg,
            src: ext,
            dst_ty: PtxType::F32,
            src_ty: PtxType::S32,
        });
        // cvt.rn.f16.f32 %h_raw, %f;
        let h_raw = alloc.alloc(PtxType::F16);
        kernel.push(PtxInstruction::Cvt {
            dst: h_raw,
            src: f32_reg,
            dst_ty: PtxType::F16,
            src_ty: PtxType::F32,
        });
        // mul.f16 %h, %h_raw, %scale_f16;
        let h = alloc.alloc(PtxType::F16);
        kernel.push(PtxInstruction::Arith(ArithOp::Mul {
            dst: h,
            lhs: Operand::Reg(h_raw),
            rhs: Operand::Reg(scale_f16),
            ty: PtxType::F16,
        }));
        h_out[i as usize] = Some(h);
    }

    // Pack 8 f16 values into 4 .b32 pairs via the vector-pack mov form.
    let mut pairs = [Register {
        kind: kaio_core::types::RegKind::R,
        index: 0,
        ptx_type: PtxType::U32,
    }; 4];
    for p in 0..4usize {
        let lo = h_out[2 * p].expect("all 8 h_out entries populated above");
        let hi = h_out[2 * p + 1].expect("all 8 h_out entries populated above");
        let b32 = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::MovPack {
            dst: b32,
            srcs: vec![lo, hi],
            ty: PtxType::U32,
        });
        pairs[p] = b32;
    }
    pairs
}

/// Build a minimal kernel that exercises [`emit_unpack_s4_x8_scale_to_f16x8`]
/// once and stores the 4 packed `.b32` results to global memory.
///
/// Used by the ptxas_verify and emit-token canary tests in this
/// module. The kernel accepts:
/// - `packed: u32` scalar param — the packed nibbles to unpack
/// - `scale_bits: u16` scalar param — the f16 scale bit pattern
/// - `out: *mut u32` — 4 u32 output (packed f16 pairs)
///
/// Only two of the four params are wired (packed + out) — scale is
/// materialized via a trivial cvt to keep the emit minimal; this is
/// a structural canary, not a runtime kernel.
#[cfg(test)]
pub(crate) fn build_unpack_s4_smoke_ptx(sm: &str) -> String {
    use kaio_core::emit::{Emit, PtxWriter};
    use kaio_core::instr::control::ControlOp;
    use kaio_core::instr::memory::MemoryOp;
    use kaio_core::ir::{PtxModule, PtxParam};

    let mut alloc = RegisterAllocator::new();
    let mut kernel = PtxKernel::new("unpack_s4_smoke");

    kernel.add_param(PtxParam::scalar("packed", PtxType::U32));
    kernel.add_param(PtxParam::pointer("out", PtxType::U32));

    // Load params.
    let r_packed = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: r_packed,
        param_name: "packed".to_string(),
        ty: PtxType::U32,
    }));
    let rd_out_param = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: rd_out_param,
        param_name: "out".to_string(),
        ty: PtxType::U64,
    }));
    let rd_out_global = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::CvtaToGlobal {
        dst: rd_out_global,
        src: rd_out_param,
    }));

    // Materialize a scale of +1.0_f16 (0x3C00) via cvt from an immediate-loaded f32.
    let f_one = alloc.alloc(PtxType::F32);
    kernel.push(PtxInstruction::Mov {
        dst: f_one,
        src: Operand::ImmF32(1.0),
        ty: PtxType::F32,
    });
    let h_scale = alloc.alloc(PtxType::F16);
    kernel.push(PtxInstruction::Cvt {
        dst: h_scale,
        src: f_one,
        dst_ty: PtxType::F16,
        src_ty: PtxType::F32,
    });

    // Touch tid.x so the kernel isn't completely thread-invariant
    // (ptxas optimizes harder when it can prove uniform execution).
    let (_r_tid, tid_instr) = kaio_core::instr::special::tid_x(&mut alloc);
    kernel.push(tid_instr);

    // The actual unpack chain.
    let pairs = emit_unpack_s4_x8_scale_to_f16x8(&mut alloc, &mut kernel, r_packed, h_scale);

    // Store the 4 packed b32 results to global out[0..4].
    for (i, reg) in pairs.iter().enumerate() {
        let rd_addr = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: rd_addr,
            lhs: Operand::Reg(rd_out_global),
            rhs: Operand::ImmU32(4 * i as u32),
            ty: PtxType::U64,
        }));
        kernel.push(PtxInstruction::Memory(MemoryOp::StGlobal {
            addr: rd_addr,
            src: *reg,
            ty: PtxType::U32,
        }));
    }

    kernel.push(PtxInstruction::Control(ControlOp::Ret));
    kernel.set_registers(alloc.into_allocated());

    let mut module = PtxModule::new(sm);
    module.add_kernel(kernel);
    let mut w = PtxWriter::new();
    module.emit(&mut w).unwrap();
    w.finish()
}

/// Build a minimal kernel exercising all three D4.1 cooperative load
/// helpers (A tile, packed B tile, group scales) once, followed by a
/// `bar.sync 0`. Used by the D4.1 emit structure tests and
/// `ptxas_verify_matmul_int4_coop_loads`. Structural canary — tile
/// contents are whatever the kernel happens to read.
#[cfg(test)]
pub(crate) fn build_coop_loads_smoke_ptx(sm: &str) -> String {
    use kaio_core::emit::{Emit, PtxWriter};
    use kaio_core::ir::{PtxModule, PtxParam, SharedDecl};

    let mut alloc = RegisterAllocator::new();
    let mut kernel = PtxKernel::new("matmul_int4_coop_loads_smoke");

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
    kernel.add_shared_decl(SharedDecl {
        name: "tile_scales".to_string(),
        align: 4,
        size_bytes: TILE_SCALES_BYTES,
    });

    kernel.add_param(PtxParam::pointer("a_ptr", PtxType::F16));
    kernel.add_param(PtxParam::pointer("b_packed_ptr", PtxType::U32));
    kernel.add_param(PtxParam::pointer("scales_ptr", PtxType::F16));
    kernel.add_param(PtxParam::scalar("m", PtxType::U32));
    kernel.add_param(PtxParam::scalar("n", PtxType::U32));
    kernel.add_param(PtxParam::scalar("k", PtxType::U32));

    // Load pointer params.
    let rd_a_p = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: rd_a_p,
        param_name: "a_ptr".to_string(),
        ty: PtxType::U64,
    }));
    let rd_b_p = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: rd_b_p,
        param_name: "b_packed_ptr".to_string(),
        ty: PtxType::U64,
    }));
    let rd_s_p = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: rd_s_p,
        param_name: "scales_ptr".to_string(),
        ty: PtxType::U64,
    }));
    let rd_a_g = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::CvtaToGlobal {
        dst: rd_a_g,
        src: rd_a_p,
    }));
    let rd_b_g = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::CvtaToGlobal {
        dst: rd_b_g,
        src: rd_b_p,
    }));
    let rd_s_g = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::CvtaToGlobal {
        dst: rd_s_g,
        src: rd_s_p,
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
    // k_bytes = k * 2 (f16)
    let r_k_bytes = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_k_bytes,
        lhs: Operand::Reg(r_k),
        rhs: Operand::ImmU32(BYTES_PER_F16),
        ty: PtxType::U32,
    }));
    // k_words = k / 8
    let r_k_words = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: r_k_words,
        lhs: Operand::Reg(r_k),
        rhs: Operand::ImmU32(NIBBLES_PER_U32),
        ty: PtxType::U32,
    }));

    // Shared base registers.
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
    let r_tile_s = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_tile_s,
        src: Operand::SharedAddr("tile_scales".to_string()),
        ty: PtxType::U32,
    });

    // flat_tid = %tid.x (assuming 1D block for the smoke kernel — 128 threads).
    let (r_tid, tid_instr) = kaio_core::instr::special::tid_x(&mut alloc);
    kernel.push(tid_instr);

    // Synthesize block_row = ctaid.y * BM_BLOCK and block_col = ctaid.x * BN_BLOCK.
    let (r_cidy, cidy_instr) = kaio_core::instr::special::ctaid_y(&mut alloc);
    kernel.push(cidy_instr);
    let (r_cidx, cidx_instr) = kaio_core::instr::special::ctaid_x(&mut alloc);
    kernel.push(cidx_instr);
    let r_block_row = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_block_row,
        lhs: Operand::Reg(r_cidy),
        rhs: Operand::ImmU32(BM_BLOCK),
        ty: PtxType::U32,
    }));
    let r_block_col = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_block_col,
        lhs: Operand::Reg(r_cidx),
        rhs: Operand::ImmU32(BN_BLOCK),
        ty: PtxType::U32,
    }));

    // --- Exercise all three cooperative loaders once ---
    emit_mw_load_tile_a_f16_64x16(
        &mut alloc,
        &mut kernel,
        rd_a_g,
        r_tile_a,
        r_tid,
        r_block_row,
        r_m,
        r_k_bytes,
        "smoke",
    );

    emit_mw_load_tile_b_packed_2x64(
        &mut alloc,
        &mut kernel,
        rd_b_g,
        r_tile_b,
        r_tid,
        r_block_col,
        r_n,
        r_k_words,
        "smoke",
    );

    emit_cooperative_load_group_scales_64(
        &mut alloc,
        &mut kernel,
        rd_s_g,
        r_tile_s,
        r_tid,
        r_block_col,
        r_n,
        "smoke",
    );

    kernel.push(PtxInstruction::Control(ControlOp::BarSync {
        barrier_id: 0,
    }));
    kernel.push(PtxInstruction::Control(ControlOp::Ret));
    kernel.set_registers(alloc.into_allocated());

    let mut module = PtxModule::new(sm);
    module.add_kernel(kernel);
    let mut w = PtxWriter::new();
    module.emit(&mut w).unwrap();
    w.finish()
}

// ============================================================================
// D4.3 — pre-zero + full module builder
// ============================================================================

/// Cooperative pre-zero of tile_a, tile_b, and tile_scales, followed
/// by one `bar.sync`. Needed because the cooperative loaders use
/// bounds-skip branches that leave OOB slots untouched.
fn emit_pre_zero_shared_tiles_int4(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    tile_a: Register,
    tile_b: Register,
    tile_scales: Register,
    flat_tid: Register,
) {
    debug_assert!(TILE_A_BYTES.is_multiple_of(THREADS_PER_BLOCK * 4));
    debug_assert!(TILE_B_BYTES.is_multiple_of(THREADS_PER_BLOCK * 4));
    debug_assert!(
        TILE_SCALES_BYTES.is_multiple_of(THREADS_PER_BLOCK * 4)
            || TILE_SCALES_BYTES <= THREADS_PER_BLOCK * 4
    );

    let r_zero = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_zero,
        src: Operand::ImmU32(0),
        ty: PtxType::U32,
    });

    // Zero tile_a (2048 B, 128 threads × 16 B/thread = 4 b32/thread).
    for (tile_base, total_bytes) in [(tile_a, TILE_A_BYTES), (tile_b, TILE_B_BYTES)] {
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

    // Zero tile_scales (128 B = 32 b32 — only lanes 0..31 cooperate).
    let p_scale_active = alloc.alloc(PtxType::Pred);
    kernel.push(PtxInstruction::Control(ControlOp::SetP {
        dst: p_scale_active,
        cmp_op: CmpOp::Lt,
        lhs: Operand::Reg(flat_tid),
        rhs: Operand::ImmU32(TILE_SCALES_BYTES / 4),
        ty: PtxType::U32,
    }));
    let skip_scale_label = "SCALES_ZERO_SKIP".to_string();
    kernel.push(PtxInstruction::Control(ControlOp::BraPred {
        pred: p_scale_active,
        target: skip_scale_label.clone(),
        negate: true,
    }));
    let r_scale_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_scale_off,
        lhs: Operand::Reg(flat_tid),
        rhs: Operand::ImmU32(4),
        ty: PtxType::U32,
    }));
    let r_scale_addr = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_scale_addr,
        lhs: Operand::Reg(tile_scales),
        rhs: Operand::Reg(r_scale_off),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Memory(MemoryOp::StShared {
        addr: r_scale_addr,
        src: r_zero,
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Label(skip_scale_label));

    kernel.push(PtxInstruction::Control(ControlOp::BarSync {
        barrier_id: 0,
    }));
}

/// Build the full IR module for `matmul_int4` targeting `sm`.
///
/// Kernel signature (in order): `a_ptr: *const f16`, `b_packed_ptr:
/// *const u32`, `scales_ptr: *const f16`, `d_ptr: *mut f32`, `m, n, k`
/// (all u32). Block dim = `(32, 4, 1)` = 128 threads. Grid dim =
/// `(N/64, M/64, 1)` (caller responsibility at launch).
pub(crate) fn build_matmul_int4_module(sm: &str) -> kaio_core::ir::PtxModule {
    use kaio_core::fragment::{FragmentC, alloc_c};
    use kaio_core::ir::{PtxModule, PtxParam, SharedDecl};

    let mut alloc = RegisterAllocator::new();
    let mut kernel = PtxKernel::new("matmul_int4");

    kernel.add_param(PtxParam::pointer("a_ptr", PtxType::F16));
    kernel.add_param(PtxParam::pointer("b_packed_ptr", PtxType::U32));
    kernel.add_param(PtxParam::pointer("scales_ptr", PtxType::F16));
    kernel.add_param(PtxParam::pointer("d_ptr", PtxType::F32));
    kernel.add_param(PtxParam::scalar("m", PtxType::U32));
    kernel.add_param(PtxParam::scalar("n", PtxType::U32));
    kernel.add_param(PtxParam::scalar("k", PtxType::U32));

    // Shared decls.
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
    kernel.add_shared_decl(SharedDecl {
        name: "tile_scales".to_string(),
        align: 4,
        size_bytes: TILE_SCALES_BYTES,
    });

    // --- Load params ---
    let rd_a_p = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: rd_a_p,
        param_name: "a_ptr".to_string(),
        ty: PtxType::U64,
    }));
    let rd_b_p = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: rd_b_p,
        param_name: "b_packed_ptr".to_string(),
        ty: PtxType::U64,
    }));
    let rd_s_p = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: rd_s_p,
        param_name: "scales_ptr".to_string(),
        ty: PtxType::U64,
    }));
    let rd_d_p = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: rd_d_p,
        param_name: "d_ptr".to_string(),
        ty: PtxType::U64,
    }));
    let rd_a_g = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::CvtaToGlobal {
        dst: rd_a_g,
        src: rd_a_p,
    }));
    let rd_b_g = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::CvtaToGlobal {
        dst: rd_b_g,
        src: rd_b_p,
    }));
    let rd_s_g = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::CvtaToGlobal {
        dst: rd_s_g,
        src: rd_s_p,
    }));
    let rd_d_g = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::CvtaToGlobal {
        dst: rd_d_g,
        src: rd_d_p,
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

    // --- Derived scalars ---
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
    let r_n_f32_stride = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_n_f32_stride,
        lhs: Operand::Reg(r_n),
        rhs: Operand::ImmU32(4),
        ty: PtxType::U32,
    }));

    // --- Thread / block indices ---
    let (r_tid_x, tid_x_instr) = kaio_core::instr::special::tid_x(&mut alloc);
    kernel.push(tid_x_instr);
    let (r_tid_y, tid_y_instr) = kaio_core::instr::special::tid_y(&mut alloc);
    kernel.push(tid_y_instr);
    let (r_ntid_x, ntid_x_instr) = kaio_core::instr::special::ntid_x(&mut alloc);
    kernel.push(ntid_x_instr);
    // flat_tid = tid_y * blockDim.x + tid_x  (block is (32, 4, 1))
    let r_flat_tid = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mad {
        dst: r_flat_tid,
        a: Operand::Reg(r_tid_y),
        b: Operand::Reg(r_ntid_x),
        c: Operand::Reg(r_tid_x),
        ty: PtxType::U32,
        mode: MadMode::Lo,
    }));
    // warp_id = flat_tid / 32, lane = flat_tid % 32
    let r_warp_id = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: r_warp_id,
        lhs: Operand::Reg(r_flat_tid),
        rhs: Operand::ImmU32(32),
        ty: PtxType::U32,
    }));
    // warp_row_quad = warp_id / 2, warp_col_quad = warp_id % 2
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
    let r_wq_row_base = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_wq_row_base,
        lhs: Operand::Reg(r_warp_row_quad),
        rhs: Operand::ImmU32(WARP_QUAD_M),
        ty: PtxType::U32,
    }));
    let r_wq_col_base = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_wq_col_base,
        lhs: Operand::Reg(r_warp_col_quad),
        rhs: Operand::ImmU32(WARP_QUAD_N),
        ty: PtxType::U32,
    }));

    // block_row = ctaid.y * BM_BLOCK, block_col = ctaid.x * BN_BLOCK
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
    let r_tile_s = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_tile_s,
        src: Operand::SharedAddr("tile_scales".to_string()),
        ty: PtxType::U32,
    });

    // --- a_block_base, b_block_base, scales_block_base (block-level global origins) ---
    // a_block_base = a + block_row * k_bytes
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
        lhs: Operand::Reg(rd_a_g),
        rhs: Operand::Reg(rd_a_row_off),
        ty: PtxType::U64,
    }));

    // b_block_base = b + block_col * (k/8) * 4   (col-major, u32 stride)
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
    let rd_b_block_base = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: rd_b_block_base,
        lhs: Operand::Reg(rd_b_g),
        rhs: Operand::Reg(rd_b_col_off),
        ty: PtxType::U64,
    }));

    // --- Allocate + zero the per-warp C fragment grid (2 m-stripes × 4 n-stripes = 8) ---
    let mut frag_c_grid: [[FragmentC; MMAS_PER_WARP_N as usize]; MMAS_PER_WARP_M as usize] = [
        [
            alloc_c(&mut alloc),
            alloc_c(&mut alloc),
            alloc_c(&mut alloc),
            alloc_c(&mut alloc),
        ],
        [
            alloc_c(&mut alloc),
            alloc_c(&mut alloc),
            alloc_c(&mut alloc),
            alloc_c(&mut alloc),
        ],
    ];
    for row in &frag_c_grid {
        for frag in row {
            for reg in &frag.regs {
                kernel.push(PtxInstruction::Mov {
                    dst: *reg,
                    src: Operand::ImmF32(0.0),
                    ty: PtxType::F32,
                });
            }
        }
    }

    // --- Pre-zero shared tiles + bar.sync ---
    emit_pre_zero_shared_tiles_int4(
        &mut alloc,
        &mut kernel,
        r_tile_a,
        r_tile_b,
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

    kernel.push(PtxInstruction::Label("K_LOOP_I4".to_string()));

    // Compute per-K-tile global offsets.
    // A tile starts at a_block_base + k_tile * K_TILE_SHARED * BYTES_PER_F16.
    let r_k_tile_a_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_k_tile_a_off,
        lhs: Operand::Reg(r_k_tile),
        rhs: Operand::ImmU32(K_TILE_SHARED * BYTES_PER_F16),
        ty: PtxType::U32,
    }));
    let rd_k_tile_a_off64 = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Cvt {
        dst: rd_k_tile_a_off64,
        src: r_k_tile_a_off,
        dst_ty: PtxType::U64,
        src_ty: PtxType::U32,
    });
    let rd_a_tile_src = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: rd_a_tile_src,
        lhs: Operand::Reg(rd_a_block_base),
        rhs: Operand::Reg(rd_k_tile_a_off64),
        ty: PtxType::U64,
    }));
    // B tile (packed col-major) starts at b_block_base + k_tile * (K_TILE_SHARED/8) * 4 = k_tile * 8.
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
    let rd_b_tile_src = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: rd_b_tile_src,
        lhs: Operand::Reg(rd_b_block_base),
        rhs: Operand::Reg(rd_k_tile_b_off64),
        ty: PtxType::U64,
    }));

    // Group-scale reload on transitions (every K_TILE_GROUP_RATIO K-tiles).
    // If (k_tile % K_TILE_GROUP_RATIO) == 0, reload scales[g, block_col..+64].
    let r_group_phase = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Rem {
        dst: r_group_phase,
        lhs: Operand::Reg(r_k_tile),
        rhs: Operand::ImmU32(K_TILE_GROUP_RATIO),
        ty: PtxType::U32,
    }));
    let p_new_group = alloc.alloc(PtxType::Pred);
    kernel.push(PtxInstruction::Control(ControlOp::SetP {
        dst: p_new_group,
        cmp_op: CmpOp::Eq,
        lhs: Operand::Reg(r_group_phase),
        rhs: Operand::ImmU32(0),
        ty: PtxType::U32,
    }));
    let scales_load_skip = "SCALES_LOAD_SKIP_I4".to_string();
    kernel.push(PtxInstruction::Control(ControlOp::BraPred {
        pred: p_new_group,
        target: scales_load_skip.clone(),
        negate: true,
    }));
    // Current group id = k_tile / K_TILE_GROUP_RATIO.
    let r_current_group = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: r_current_group,
        lhs: Operand::Reg(r_k_tile),
        rhs: Operand::ImmU32(K_TILE_GROUP_RATIO),
        ty: PtxType::U32,
    }));
    // scales_base = scales + current_group * N * 2 (f16 = 2 B). Layout: [num_groups, N] row-major.
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
    let rd_scales_base = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: rd_scales_base,
        lhs: Operand::Reg(rd_s_g),
        rhs: Operand::Reg(rd_scales_row_off64),
        ty: PtxType::U64,
    }));
    emit_cooperative_load_group_scales_64(
        &mut alloc,
        &mut kernel,
        rd_scales_base,
        r_tile_s,
        r_flat_tid,
        r_block_col,
        r_n,
        "",
    );
    kernel.push(PtxInstruction::Label(scales_load_skip));

    // A tile + B tile cooperative loads.
    emit_mw_load_tile_a_f16_64x16(
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
    emit_mw_load_tile_b_packed_2x64(
        &mut alloc,
        &mut kernel,
        rd_b_tile_src,
        r_tile_b,
        r_flat_tid,
        r_block_col,
        r_n,
        r_k_words,
        "",
    );

    kernel.push(PtxInstruction::Control(ControlOp::BarSync {
        barrier_id: 0,
    }));

    // Warp quadrant mma pipeline.
    emit_warp_quadrant_mma_int4(
        &mut alloc,
        &mut kernel,
        r_tile_a,
        r_tile_b,
        r_tile_s,
        r_wq_row_base,
        r_wq_col_base,
        r_tid_x,
        &mut frag_c_grid,
    );

    kernel.push(PtxInstruction::Control(ControlOp::BarSync {
        barrier_id: 0,
    }));

    // k_tile += 1; if k_tile < num_k_tiles, loop.
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_k_tile,
        lhs: Operand::Reg(r_k_tile),
        rhs: Operand::ImmU32(1),
        ty: PtxType::U32,
    }));
    let p_k_loop = alloc.alloc(PtxType::Pred);
    kernel.push(PtxInstruction::Control(ControlOp::SetP {
        dst: p_k_loop,
        cmp_op: CmpOp::Lt,
        lhs: Operand::Reg(r_k_tile),
        rhs: Operand::Reg(r_num_k_tiles),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Control(ControlOp::BraPred {
        pred: p_k_loop,
        target: "K_LOOP_I4".to_string(),
        negate: false,
    }));

    // --- Store output ---
    // Flatten the 2x4 frag_c_grid into the 8-element flat layout that
    // matmul_tc_kernel::emit_warp_quadrant_store expects.
    // That helper indexes accs[m_stripe * MMAS_PER_WARP_N + n_stripe].
    let accs_flat: [FragmentC; 8] = [
        frag_c_grid[0][0],
        frag_c_grid[0][1],
        frag_c_grid[0][2],
        frag_c_grid[0][3],
        frag_c_grid[1][0],
        frag_c_grid[1][1],
        frag_c_grid[1][2],
        frag_c_grid[1][3],
    ];

    // Map (warp_row_quad, warp_col_quad) at runtime to the two static
    // warp_quadrant_(row|col)_start values by branching to specialized
    // store regions. This keeps the store helper happy with u32 constant
    // args while still supporting 4 warps.
    let label_store_end = "STORE_END_I4".to_string();
    for (quad_row, quad_col) in [(0u32, 0u32), (0u32, 1u32), (1u32, 0u32), (1u32, 1u32)]
        .iter()
        .copied()
    {
        let is_this_warp_label = format!("STORE_WARP_{quad_row}_{quad_col}_I4");
        let p_match = alloc.alloc(PtxType::Pred);
        let r_quad_row_imm = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Mov {
            dst: r_quad_row_imm,
            src: Operand::ImmU32(quad_row),
            ty: PtxType::U32,
        });
        kernel.push(PtxInstruction::Control(ControlOp::SetP {
            dst: p_match,
            cmp_op: CmpOp::Eq,
            lhs: Operand::Reg(r_warp_row_quad),
            rhs: Operand::Reg(r_quad_row_imm),
            ty: PtxType::U32,
        }));
        let r_quad_col_imm = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Mov {
            dst: r_quad_col_imm,
            src: Operand::ImmU32(quad_col),
            ty: PtxType::U32,
        });
        let p_col_match = alloc.alloc(PtxType::Pred);
        kernel.push(PtxInstruction::Control(ControlOp::SetP {
            dst: p_col_match,
            cmp_op: CmpOp::Eq,
            lhs: Operand::Reg(r_warp_col_quad),
            rhs: Operand::Reg(r_quad_col_imm),
            ty: PtxType::U32,
        }));
        let p_both = alloc.alloc(PtxType::Pred);
        kernel.push(PtxInstruction::Arith(ArithOp::And {
            dst: p_both,
            lhs: Operand::Reg(p_match),
            rhs: Operand::Reg(p_col_match),
            ty: PtxType::Pred,
        }));
        kernel.push(PtxInstruction::Control(ControlOp::BraPred {
            pred: p_both,
            target: is_this_warp_label.clone(),
            negate: true,
        }));

        crate::matmul_tc_kernel::emit_warp_quadrant_store(
            &mut alloc,
            &mut kernel,
            rd_d_g,
            &accs_flat,
            r_tid_x,
            quad_row * WARP_QUAD_M,
            quad_col * WARP_QUAD_N,
            r_block_row,
            r_block_col,
            r_m,
            r_n,
            r_n_f32_stride,
        );
        kernel.push(PtxInstruction::Control(ControlOp::Bra {
            target: label_store_end.clone(),
        }));
        kernel.push(PtxInstruction::Label(is_this_warp_label));
    }
    kernel.push(PtxInstruction::Label(label_store_end));

    kernel.push(PtxInstruction::Control(ControlOp::Ret));
    kernel.set_registers(alloc.into_allocated());

    let mut module = PtxModule::new(sm);
    module.add_kernel(kernel);
    module
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sign-extend emit canary (D2, R1 mitigation — IR / token layer).
    ///
    /// The most dangerous silent-correctness bug in INT4 dequant is a
    /// `shr.u32` where `shr.s32` is required — logical right shift
    /// zero-extends, so every negative INT4 weight becomes positive
    /// with no loud failure. This test asserts the emitted PTX contains
    /// exactly 8 `shr.s32` instructions (one per nibble) and zero
    /// `shr.u32` shifts of the packed word.
    #[test]
    fn unpack_s4_sign_extend_uses_arithmetic_shift() {
        let ptx = build_unpack_s4_smoke_ptx("sm_80");

        let shr_s32_count = ptx.matches("shr.s32").count();
        let shr_u32_count = ptx.matches("shr.u32").count();

        assert_eq!(
            shr_s32_count, 8,
            "expected exactly 8 `shr.s32` (sign-extend per nibble); found {shr_s32_count}\n\n\
             === emitted PTX ===\n{ptx}"
        );
        assert_eq!(
            shr_u32_count, 0,
            "expected zero `shr.u32` in the unpack chain (sign-extend requires arithmetic \
             right shift); found {shr_u32_count}\n\n=== emitted PTX ===\n{ptx}"
        );
    }

    /// Sign-extend emit canary — shift-count coverage (D2, R1 mitigation).
    ///
    /// Asserts the 8 `shl.b32` lane alignments cover the full set
    /// {0, 4, 8, 12, 16, 20, 24, 28} — one per nibble position. A bug
    /// that emitted the same shift count at two sites would leave a
    /// nibble unread (silent correctness failure at one lane) and be
    /// caught here.
    #[test]
    fn unpack_s4_shl_covers_all_nibble_positions() {
        let ptx = build_unpack_s4_smoke_ptx("sm_80");

        for shift in [0u32, 4, 8, 12, 16, 20, 24, 28] {
            let pattern = format!("shl.b32 %r{{any}}, %r{{any}}, {shift};");
            // Simpler exact-text search: `, {shift};`. Each shift value
            // appears exactly once as the immediate in a shl.b32.
            let needle = format!(", {shift};");
            let count = ptx
                .lines()
                .filter(|line| line.contains("shl.b32") && line.ends_with(&needle))
                .count();
            assert_eq!(
                count, 1,
                "expected exactly one `shl.b32 ..., {shift};` (nibble-position alignment); \
                 found {count}\n\npattern scanned: {pattern}\n\n=== emitted PTX ===\n{ptx}"
            );
        }
    }

    /// Verify the emitted PTX contains 4 `mov.b32 %dst, {%h_x, %h_y};`
    /// vector-pack instructions — one per output pair.
    #[test]
    fn unpack_s4_emits_four_packed_f16_pairs() {
        let ptx = build_unpack_s4_smoke_ptx("sm_80");

        let pack_count = ptx
            .lines()
            .filter(|line| line.contains("mov.b32") && line.contains('{') && line.contains('}'))
            .count();
        assert_eq!(
            pack_count, 4,
            "expected 4 `mov.b32 %b, {{%h_lo, %h_hi}};` pack instructions; found {pack_count}\n\n\
             === emitted PTX ===\n{ptx}"
        );
    }

    /// ptxas_verify for the D2 unpack helper (R1 mitigation — offline
    /// assembler layer). Runs `ptxas -arch=sm_80` over a minimal kernel
    /// that exercises `emit_unpack_s4_x8_scale_to_f16x8` exactly once
    /// and confirms the full chain assembles cleanly.
    ///
    /// Requires the CUDA toolkit (`ptxas` on PATH). `#[ignore]` so
    /// host-only runs don't fail — invoke via `cargo test -- --ignored`.
    #[test]
    #[ignore]
    fn ptxas_verify_unpack_s4() {
        let ptxas_check = std::process::Command::new("ptxas")
            .arg("--version")
            .output();
        if ptxas_check.is_err() {
            eprintln!("NOTE: ptxas not found in PATH — skipping PTX verification");
            return;
        }

        let sm = std::env::var("KAIO_SM_TARGET").unwrap_or_else(|_| "sm_80".to_string());
        let ptx = build_unpack_s4_smoke_ptx(&sm);
        let tmp = std::env::temp_dir().join("kaio_unpack_s4_verify.ptx");
        std::fs::write(&tmp, &ptx).expect("failed to write temp PTX");

        let output = std::process::Command::new("ptxas")
            .args(["--gpu-name", &sm])
            .arg(tmp.to_str().unwrap())
            .output()
            .expect("failed to run ptxas");
        let _ = std::fs::remove_file(&tmp);

        assert!(
            output.status.success(),
            "ptxas verification FAILED for unpack_s4 ({sm}):\nstdout: {}\nstderr: {}\n\n\
             === PTX ===\n{ptx}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );

        eprintln!("ptxas verification PASSED for unpack_s4 ({sm})");
    }

    // --- D4.1: cooperative load helpers emit structure canaries ---

    /// Structure canary: D4.1 smoke kernel must emit exactly one
    /// cooperative load cycle for each of A / B-packed / group-scales.
    /// Each cycle is `ld.global.*` + `st.shared.*` inside a bounds-gated
    /// `bra` region. This canary pins the counts so a refactor that
    /// silently drops a loader emits a visible test failure.
    #[test]
    fn coop_loads_emit_ld_st_pairs_match_per_helper_design() {
        let ptx = build_coop_loads_smoke_ptx("sm_80");

        // A loader: 4 b32 load+store pairs per thread (one unrolled loop).
        // B-packed loader: 1 b32 load+store pair per thread.
        // Scales loader: 1 b32 load+store pair per active lane.
        // Total: 4 + 1 + 1 = 6 `ld.global.u32` / `st.shared.b32` pair
        // emissions in the smoke kernel's linear instruction stream.
        let ld_count = ptx.matches("ld.global.u32").count();
        let st_count = ptx.matches("st.shared.u32").count();

        assert_eq!(
            ld_count, 6,
            "expected 6 `ld.global.u32` issues (A: 4, B: 1, scales: 1); found {ld_count}\n\n\
             === emitted PTX ===\n{ptx}"
        );
        assert_eq!(
            st_count, 6,
            "expected 6 `st.shared.b32` issues (A: 4, B: 1, scales: 1); found {st_count}\n\n\
             === emitted PTX ===\n{ptx}"
        );
    }

    /// Structure canary: each cooperative loader emits a bounds-gated
    /// `@!p bra <SKIP>` region with a labeled skip target. Three
    /// loaders × one skip region each = three labeled skip targets
    /// in the smoke kernel.
    #[test]
    fn coop_loads_emit_three_bounds_gated_skip_regions() {
        let ptx = build_coop_loads_smoke_ptx("sm_80");

        for needle in [
            "A_SKIP_I4_TILE_LOAD_smoke:",
            "B_SKIP_I4_TILE_LOAD_smoke:",
            "SCALES_SKIP_I4_smoke:",
        ] {
            assert!(
                ptx.contains(needle),
                "expected labeled skip region `{needle}` in smoke kernel\n\n\
                 === emitted PTX ===\n{ptx}"
            );
        }
    }

    /// ptxas_verify for all three D4.1 cooperative load helpers (R1/R6
    /// smoke gate). Runs `ptxas -arch=sm_80` over the D4.1 smoke
    /// kernel; confirms that the full A + B-packed + scales load chain
    /// with bar.sync assembles cleanly. `#[ignore]` so host-only runs
    /// don't fail — invoke via `cargo test -- --ignored`.
    #[test]
    #[ignore]
    fn ptxas_verify_matmul_int4_coop_loads() {
        let ptxas_check = std::process::Command::new("ptxas")
            .arg("--version")
            .output();
        if ptxas_check.is_err() {
            eprintln!("NOTE: ptxas not found in PATH — skipping PTX verification");
            return;
        }

        let sm = std::env::var("KAIO_SM_TARGET").unwrap_or_else(|_| "sm_80".to_string());
        let ptx = build_coop_loads_smoke_ptx(&sm);
        let tmp = std::env::temp_dir().join("kaio_matmul_int4_coop_loads_verify.ptx");
        std::fs::write(&tmp, &ptx).expect("failed to write temp PTX");

        let output = std::process::Command::new("ptxas")
            .args(["--gpu-name", &sm])
            .arg(tmp.to_str().unwrap())
            .output()
            .expect("failed to run ptxas");
        let _ = std::fs::remove_file(&tmp);

        assert!(
            output.status.success(),
            "ptxas verification FAILED for matmul_int4 coop loads ({sm}):\n\
             stdout: {}\nstderr: {}\n\n=== PTX ===\n{ptx}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );

        eprintln!("ptxas verification PASSED for matmul_int4 coop loads ({sm})");
    }

    // `validate_dims_int4` error-path coverage lands in the D6 integration
    // test suite (`kaio-ops/tests/matmul_int4_api.rs`), which mirrors
    // INT8's `matmul_api.rs` pattern — a real `KaioDevice` + `alloc_zeros`
    // buffers. The function itself ships in D4.1 so the test harness in
    // D6 can import and exercise it directly.

    // --- D4.3: full kernel module gate ---

    /// Structure canary: the full module must emit exactly 8 `mma.sync`
    /// instructions (2 m-stripes × 4 n-stripes) per warp quadrant —
    /// and the module emits the per-warp-quadrant block once, inside
    /// the K-loop. Count includes only the final assembly's mmas.
    #[test]
    fn module_emits_eight_mma_sync_per_warp_quadrant() {
        use kaio_core::emit::{Emit, PtxWriter};
        let module = build_matmul_int4_module("sm_80");
        let mut w = PtxWriter::new();
        module.emit(&mut w).unwrap();
        let ptx = w.finish();

        let mma_count = ptx.matches("mma.sync.aligned.m16n8k16").count();
        assert_eq!(
            mma_count,
            (MMAS_PER_WARP_M * MMAS_PER_WARP_N) as usize,
            "expected {} mma.sync.m16n8k16 ({MMAS_PER_WARP_M} m-stripes × {MMAS_PER_WARP_N} n-stripes); \
             found {mma_count}\n\n=== emitted PTX (first 4000 chars) ===\n{}",
            MMAS_PER_WARP_M * MMAS_PER_WARP_N,
            &ptx[..ptx.len().min(4000)]
        );
    }

    /// Structure canary: total `shr.s32` count in the full module.
    /// Every fragment-B dequant emits 2 `shr.s32` per lane per u32
    /// (inside the narrow 2-nibble helper). Per K-tile per warp: 4
    /// n-stripes × 2 u32 × 2 shr.s32 = 16 `shr.s32` call-sites
    /// emitted. The K-loop wraps this region once statically (runtime
    /// iteration is via a PTX branch, not unrolled), so the final
    /// module has exactly 16 `shr.s32` text occurrences from the
    /// fragment-B dequant pipeline.
    ///
    /// This canary would catch a regression where the narrow helper
    /// accidentally used `shr.u32` at one of the emission sites.
    #[test]
    fn module_shr_s32_count_matches_per_lane_dequant_shape() {
        use kaio_core::emit::{Emit, PtxWriter};
        let module = build_matmul_int4_module("sm_80");
        let mut w = PtxWriter::new();
        module.emit(&mut w).unwrap();
        let ptx = w.finish();

        let shr_s32 = ptx.matches("shr.s32").count();
        let shr_u32 = ptx.matches("shr.u32").count();
        // 4 n-stripes × 2 u32 × 2 nibbles = 16 shr.s32 inside the K-loop body.
        assert_eq!(
            shr_s32,
            16,
            "expected 16 `shr.s32` (4 n-stripes × 2 u32 × 2 nibbles per fragment-B dequant); \
             found {shr_s32}\n\n=== emitted PTX (first 4000 chars) ===\n{}",
            &ptx[..ptx.len().min(4000)]
        );
        assert_eq!(
            shr_u32, 0,
            "expected zero `shr.u32` in fragment-B dequant (R1 sign-extend canary for INT4 \
             weights); found {shr_u32}"
        );
    }

    /// ptxas_verify for the full `matmul_int4` module (final D4.3
    /// gate). Runs `ptxas -arch=sm_80` over the complete kernel
    /// (pre-zero + K-loop + fragment-B dequant + mma grid + store)
    /// and confirms the whole thing assembles cleanly. `#[ignore]`
    /// so host-only runs don't fail — invoke via `cargo test --
    /// --ignored`.
    #[test]
    #[ignore]
    fn ptxas_verify_matmul_int4() {
        use kaio_core::emit::{Emit, PtxWriter};

        let ptxas_check = std::process::Command::new("ptxas")
            .arg("--version")
            .output();
        if ptxas_check.is_err() {
            eprintln!("NOTE: ptxas not found in PATH — skipping PTX verification");
            return;
        }

        let sm = std::env::var("KAIO_SM_TARGET").unwrap_or_else(|_| "sm_80".to_string());
        let module = build_matmul_int4_module(&sm);
        let mut w = PtxWriter::new();
        module.emit(&mut w).unwrap();
        let ptx = w.finish();

        let tmp = std::env::temp_dir().join("kaio_matmul_int4_verify.ptx");
        std::fs::write(&tmp, &ptx).expect("failed to write temp PTX");

        let output = std::process::Command::new("ptxas")
            .args(["--gpu-name", &sm, "--verbose"])
            .arg(tmp.to_str().unwrap())
            .output()
            .expect("failed to run ptxas");
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        eprintln!("--- ptxas --verbose stdout ({sm}) ---\n{stdout}");
        eprintln!("--- ptxas --verbose stderr ({sm}) ---\n{stderr}");
        let _ = std::fs::remove_file(&tmp);

        assert!(
            output.status.success(),
            "ptxas verification FAILED for matmul_int4 full module ({sm}):\n\
             stdout: {}\nstderr: {}\n\n=== PTX (first 6000 chars) ===\n{}",
            String::from_utf8_lossy(&output.stdout),
            stderr,
            &ptx[..ptx.len().min(6000)]
        );

        eprintln!("ptxas verification PASSED for matmul_int4 full module ({sm})");
    }
}
