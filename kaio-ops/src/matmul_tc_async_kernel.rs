//! Tensor-core matrix multiply with cp.async double-buffering, multi-warp.
//!
//! Sprint 6.7 Gate B: layered on top of Gate A's multi-warp `matmul_tc`
//! (64×64 block tile, 4 warps, 32×32 quadrant per warp, 8 mma per
//! K-tile). Same shared-memory layout contract (A row-major in shared,
//! B column-major in shared), same fragment loaders, same per-warp 8-mma
//! accumulation, same edge-tile predication. Only the staging path
//! differs:
//!
//! - **A** is staged via `cp.async.ca.shared.global` (16 bytes per
//!   thread × 128 threads = 2,048 B = one full A buffer). Next K-tile's
//!   A is issued **before** the current K-tile's `mma.sync` work, so
//!   the memory fetch overlaps with compute.
//! - **B** stays synchronous — the row-major → column-major transpose
//!   during staging is a strided gather that `cp.async` cannot express
//!   (source must be contiguous bytes). Reuses Gate A's
//!   `emit_mw_load_tile_b_16x64` from `matmul_tc_kernel`.
//!
//! # Dimension constraint
//!
//! Identical to `matmul_tc`: `M % 16 == 0 && N % 8 == 0 && K % 16 == 0`.
//! Validation shared via [`validate_dims_tc`](crate::matmul_tc_kernel::validate_dims_tc).
//! Edge tiles inside the 64×64 block are handled by per-warp/per-thread
//! bra-skip on OOB — same mechanism as Gate A.
//!
//! # Shared-memory layout (2 buffers per tile, multi-warp sized)
//!
//! ```text
//! .shared .align 16 .b8 tile_a[4096];   // 2 × 2048 B, row-major per buffer
//! .shared .align 4  .b8 tile_b[4096];   // 2 × 2048 B, col-major per buffer
//! ```
//!
//! Per-buffer: `tile_a` 64×16 fp16 (2,048 B), `tile_b` 16×64 fp16
//! (2,048 B). Total shared per block: **8 KB** — well under the 48 KB
//! ceiling.
//!
//! `tile_a` is `align = 16` because `cp.async.ca.shared.global` with
//! `size = 16` requires 16-byte shared alignment. `tile_b` stays
//! `align = 4` — sync-store path, no cp.async writes there.
//!
//! Buffer selection at iteration `k_tile`: `cur = k_tile % 2`. Buffer
//! offsets computed via [`buffer_offsets`] each iter as
//! `(k_tile & 1) * buffer_size`.
//!
//! # Pipeline ordering
//!
//! ```text
//! preamble:
//!     pre-zero tile_a + tile_b (all 8 KB cooperatively)
//!     bar.sync
//!     cp.async A[0] → tile_a[0]; commit_group
//!     sync-store B[0] → tile_b[0]
//!
//! K_LOOP (k = 0 .. num_k_tiles):
//!     cp.async.wait_group 0               ; wait on A[k] to be resident
//!     bar.sync                             ; block-wide visibility
//!     if k + 1 < num_k_tiles:
//!         cp.async A[k+1] → tile_a[nxt]; commit_group
//!         sync-store B[k+1] → tile_b[nxt]
//!     emit_warp_quadrant_mma on tile_{a,b}[cur]   ; 8 mma per warp
//!     cur, nxt = nxt, cur (implicit via k_tile + 1)
//! ```
//!
//! **Loop-entry invariant:** `tile_{a,b}_cur` name the buffers that
//! have been *issued* (by the preamble at k=0 or by iter k-1). The
//! `wait_group 0` + `bar.sync` at the top of each iter is what makes
//! them *resident* and *visible* to the per-warp fragment loads.
//! `tile_{a,b}_nxt` name the free buffers, ready to receive the next
//! K-tile's issues.
//!
//! **Multi-warp safety vs Sprint 6.4:** the same "no trailing bar.sync
//! after mma" optimization holds — disjoint cur/nxt buffers + top-of-
//! iter `bar.sync` together fence reads from writes. With 4 warps now
//! all reading `tile_cur` simultaneously and all 4 warps' next-iter
//! cp.async issues going to `tile_nxt`, no inter-warp race exists
//! within a single K-iter, and the top-of-next-iter `bar.sync` covers
//! cross-warp visibility before the next mma reads.

use half::f16;
use kaio::prelude::*;
use kaio_core::fragment::FragmentC;
use kaio_core::instr::control::{CmpOp, ControlOp};
use kaio_core::instr::memory::MemoryOp;
use kaio_core::instr::special;
use kaio_core::instr::{ArithOp, MadMode};
use kaio_core::ir::{
    Operand, PtxInstruction, PtxKernel, PtxModule, PtxParam, Register, RegisterAllocator,
    SharedDecl, SpecialReg,
};
use kaio_core::types::PtxType;

use crate::matmul_tc_kernel::{
    emit_mw_load_tile_b_16x64, emit_pre_zero_shared_tiles, emit_warp_quadrant_mma,
    emit_warp_quadrant_store, validate_dims_tc,
};

// --- mma.sync.m16n8k16 instance shape (used for validate constraint
// reference; BM/BN are also encoded in the matmul_tc_kernel module's
// validate path which we share via `validate_dims_tc`) ---
const BK: u32 = 16;

// --- Multi-warp block tiling (Sprint 6.7) ---
const BM_BLOCK: u32 = 64;
const BN_BLOCK: u32 = 64;
const WARP_QUAD_M: u32 = 32;
const WARP_QUAD_N: u32 = 32;
const WARPS_PER_BLOCK: u32 = 4;

// --- Shared tile sizes (per buffer; 2 buffers each for double-buffering) ---
// Sprint 6.7b: tile constants are shared with the sync kernel (matmul_tc_kernel.rs)
// to guarantee the layout stored by the cooperative load is identical to what
// the fragment-B loader reads. Async kernel's Tile B per-buffer inherits the
// padded col stride (36 B) + rounded data size (2560 B).
const BYTES_PER_HALF: u32 = 2;
const BYTES_PER_F32: u32 = 4;
use crate::matmul_tc_kernel::{TILE_A_BYTES, TILE_A_ROW_STRIDE_BYTES, TILE_B_BYTES, TILE_B_COL_STRIDE_BYTES};
const TILE_A_TOTAL_BYTES: u32 = 2 * TILE_A_BYTES; // 4096 (2 buffers)
const TILE_B_TOTAL_BYTES: u32 = 2 * TILE_B_BYTES; // 2 × 2560 = 5120 per async (was 4096 pre-6.7b)

/// Pure helper: byte offsets of the (cur, nxt) buffer pair within
/// `tile_a` and `tile_b` at iteration `k_tile`.
///
/// Returns `(a_cur, a_nxt, b_cur, b_nxt)`. Sprint 6.7 multi-warp
/// buffer sizes: tile_a per-buffer = 2048 B, tile_b per-buffer = 2048 B.
/// Pure function so the toggle math is host-testable; the kernel
/// builder inlines equivalent runtime arithmetic via register operands.
#[allow(dead_code)]
pub(crate) fn buffer_offsets(k_tile: u32) -> (u32, u32, u32, u32) {
    let cur = k_tile & 1;
    let nxt = cur ^ 1;
    (
        cur * TILE_A_BYTES,
        nxt * TILE_A_BYTES,
        cur * TILE_B_BYTES,
        nxt * TILE_B_BYTES,
    )
}

/// Multi-warp cooperative async-load of the 64×16 fp16 row-major A
/// block tile via `cp.async.ca.shared.global` (size = 16). 128 threads
/// × 1 issue per thread = 2,048 B = full A buffer.
///
/// **Per-thread layout:** thread `t` writes 16 contiguous bytes (= 8
/// fp16 = a half-row of 8 cols) at:
/// - `row = t / 2`            (0..64 across all 128 threads ✓)
/// - `col_byte = (t % 2) * 16`
/// - `shared_off = row * 32 + col_byte`
/// - `global_off = row * K_bytes + col_byte`
///
/// **Edge handling:** Caller pre-zeros `tile_a` shared at kernel start.
/// OOB threads (`row_global = block_row + row >= M`) skip the cp.async
/// issue via `@!p bra A_ASYNC_SKIP_<suffix>` — the buffer slot stays
/// zero. No K-direction edge check (`K % 16 == 0` enforced by validate).
///
/// **Alignment safety:** shared dst is `align = 16` (decl) and
/// per-thread byte offset = `t * 16` so each thread's address is also
/// 16-byte aligned. Global src starts at `block_row * K_bytes + k_tile
/// * 32`; since `K % 16 == 0` and `BK = 16`, `K_bytes = 2K` is a
/// multiple of 32, and `t * 16` keeps each per-thread global address
/// 16-byte aligned. Same alignment contract as Sprint 6.4's single-warp
/// `emit_load_a_tile_async`.
///
/// Does **not** emit `cp.async.commit_group` — caller owns the commit
/// boundary so preamble and in-loop issues each commit independently.
///
/// `label_suffix` makes the bra-skip label unique per call site.
#[allow(clippy::too_many_arguments)]
fn emit_mw_load_tile_a_64x16_async(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    a_block_base_global: Register, // u64 — A[block_row, k_tile*16]
    tile_a_shared: Register,       // u32 — base of the chosen A buffer
    flat_tid: Register,            // u32 — 0..128
    block_row: Register,           // u32
    m: Register,                   // u32
    k_bytes: Register,             // u32 — K * 2
    label_suffix: &str,
) {
    let skip_label = format!("A_ASYNC_SKIP_{label_suffix}");

    // row = flat_tid / 2; col_byte = (flat_tid % 2) * 16
    let row = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: row,
        lhs: Operand::Reg(flat_tid),
        rhs: Operand::ImmU32(2),
        ty: PtxType::U32,
    }));
    let lane_mod2 = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Rem {
        dst: lane_mod2,
        lhs: Operand::Reg(flat_tid),
        rhs: Operand::ImmU32(2),
        ty: PtxType::U32,
    }));
    let col_byte = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: col_byte,
        lhs: Operand::Reg(lane_mod2),
        rhs: Operand::ImmU32(16),
        ty: PtxType::U32,
    }));

    // p_row_in = block_row + row < M; if false, skip cp.async.
    let row_global = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: row_global,
        lhs: Operand::Reg(block_row),
        rhs: Operand::Reg(row),
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

    // shared_addr = tile_a_shared + row * 32 + col_byte
    let shared_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mad {
        dst: shared_off,
        a: Operand::Reg(row),
        b: Operand::ImmU32(TILE_A_ROW_STRIDE_BYTES),
        c: Operand::Reg(col_byte),
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

    // global_addr = a_block_base + row * K_bytes + col_byte
    let row_off64 = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
        dst: row_off64,
        lhs: Operand::Reg(row),
        rhs: Operand::Reg(k_bytes),
        src_ty: PtxType::U32,
    }));
    let col_byte64 = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Cvt {
        dst: col_byte64,
        src: col_byte,
        dst_ty: PtxType::U64,
        src_ty: PtxType::U32,
    });
    let per_thread_off64 = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: per_thread_off64,
        lhs: Operand::Reg(row_off64),
        rhs: Operand::Reg(col_byte64),
        ty: PtxType::U64,
    }));
    let global_addr = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: global_addr,
        lhs: Operand::Reg(a_block_base_global),
        rhs: Operand::Reg(per_thread_off64),
        ty: PtxType::U64,
    }));

    // Issue cp.async.ca size=16. Caller commits the group.
    kernel.push(PtxInstruction::Memory(MemoryOp::new_cp_async_ca(
        shared_addr,
        global_addr,
        16,
    )));

    kernel.push(PtxInstruction::Label(skip_label));
}

/// Build the IR module for `matmul_tc_async` targeting the given SM.
///
/// Multi-warp 64×64 block tile, 4 warps × 32×32 quadrant via 8 mma per
/// K-tile, with double-buffered cp.async A staging and sync B staging.
/// Same staging contracts as Sprint 6.4's single-warp variant — see
/// the module docstring for the pipeline ordering.
pub(crate) fn build_matmul_tc_async_module(sm: &str) -> PtxModule {
    let mut alloc = RegisterAllocator::new();
    let mut kernel = PtxKernel::new("matmul_tc_async");

    kernel.add_param(PtxParam::pointer("a_ptr", PtxType::F16));
    kernel.add_param(PtxParam::pointer("b_ptr", PtxType::F16));
    kernel.add_param(PtxParam::pointer("d_ptr", PtxType::F32));
    kernel.add_param(PtxParam::scalar("m", PtxType::U32));
    kernel.add_param(PtxParam::scalar("n", PtxType::U32));
    kernel.add_param(PtxParam::scalar("k", PtxType::U32));

    // tile_a: align 16 (cp.async.ca size=16 dst alignment requirement).
    kernel.add_shared_decl(SharedDecl {
        name: "tile_a".to_string(),
        align: 16,
        size_bytes: TILE_A_TOTAL_BYTES,
    });
    kernel.add_shared_decl(SharedDecl {
        name: "tile_b".to_string(),
        align: 4,
        size_bytes: TILE_B_TOTAL_BYTES,
    });

    // --- Load params + cvta ---
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
    // flat_tid = tid_y * 32 + tid_x   (used for cooperative tile loads)
    let r_flat_tid = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mad {
        dst: r_flat_tid,
        a: Operand::Reg(r_warp_id),
        b: Operand::ImmU32(32),
        c: Operand::Reg(r_tid_x),
        ty: PtxType::U32,
        mode: MadMode::Lo,
    }));

    // Sprint 6.7b D10 hoist: compute fragment-layout (group_id, tig) ONCE at
    // kernel start, reuse across every emit_warp_quadrant_mma call. Saves
    // 6 × div/rem pairs per K-iter that the fragment loaders would otherwise
    // recompute internally (2 FragmentA + 4 FragmentB per K-iter).
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

    // block_row = bidy * BM_BLOCK; block_col = bidx * BN_BLOCK
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

    // K and N in bytes; N in fp32 stride.
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

    // a_block_base = rd_a + block_row * K_bytes
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

    // b_block_base = rd_b + block_col * 2
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

    // Shared symbolic bases.
    let r_tile_a_sym = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_tile_a_sym,
        src: Operand::SharedAddr("tile_a".to_string()),
        ty: PtxType::U32,
    });
    let r_tile_b_sym = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_tile_b_sym,
        src: Operand::SharedAddr("tile_b".to_string()),
        ty: PtxType::U32,
    });

    // Pre-zero shared (all 8 KB: tile_a 4096 + tile_b 4096).
    emit_pre_zero_shared_tiles(
        &mut alloc,
        &mut kernel,
        r_tile_a_sym,
        r_tile_b_sym,
        r_flat_tid,
        TILE_A_TOTAL_BYTES,
        TILE_B_TOTAL_BYTES,
    );

    // Per-warp shared offset within a buffer (warp_row_quad ∈ {0,1},
    // warp_col_quad ∈ {0,1}).
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
    // tile_a_warp_off (within a buffer) = warp_row_quad * 32 * TILE_A_ROW_STRIDE_BYTES
    let r_tile_a_warp_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_tile_a_warp_off,
        lhs: Operand::Reg(r_warp_row_quad),
        rhs: Operand::ImmU32(WARP_QUAD_M * TILE_A_ROW_STRIDE_BYTES),
        ty: PtxType::U32,
    }));
    let r_tile_b_warp_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_tile_b_warp_off,
        lhs: Operand::Reg(r_warp_col_quad),
        rhs: Operand::ImmU32(WARP_QUAD_N * TILE_B_COL_STRIDE_BYTES),
        ty: PtxType::U32,
    }));

    // 8 FragmentC accumulators, zeroed.
    let mut accs: [FragmentC; 8] = [
        kaio_core::fragment::alloc_c(&mut alloc),
        kaio_core::fragment::alloc_c(&mut alloc),
        kaio_core::fragment::alloc_c(&mut alloc),
        kaio_core::fragment::alloc_c(&mut alloc),
        kaio_core::fragment::alloc_c(&mut alloc),
        kaio_core::fragment::alloc_c(&mut alloc),
        kaio_core::fragment::alloc_c(&mut alloc),
        kaio_core::fragment::alloc_c(&mut alloc),
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

    // num_k_tiles = K / 16
    let r_num_k_tiles = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: r_num_k_tiles,
        lhs: Operand::Reg(r_k),
        rhs: Operand::ImmU32(BK),
        ty: PtxType::U32,
    }));

    // Helper closures (inline inside this builder; capture rd_*_block_base, r_*_bytes, r_tile_*_sym).
    let emit_a_tile_src =
        |alloc: &mut RegisterAllocator, kernel: &mut PtxKernel, k_tile: Register| -> Register {
            // a_tile_src = rd_a_block_base + k_tile * 32
            let k_x_32 = alloc.alloc(PtxType::U32);
            kernel.push(PtxInstruction::Arith(ArithOp::Mul {
                dst: k_x_32,
                lhs: Operand::Reg(k_tile),
                rhs: Operand::ImmU32(BK * BYTES_PER_HALF), // 32
                ty: PtxType::U32,
            }));
            let k_x_32_u64 = alloc.alloc(PtxType::U64);
            kernel.push(PtxInstruction::Cvt {
                dst: k_x_32_u64,
                src: k_x_32,
                dst_ty: PtxType::U64,
                src_ty: PtxType::U32,
            });
            let a_src = alloc.alloc(PtxType::U64);
            kernel.push(PtxInstruction::Arith(ArithOp::Add {
                dst: a_src,
                lhs: Operand::Reg(rd_a_block_base),
                rhs: Operand::Reg(k_x_32_u64),
                ty: PtxType::U64,
            }));
            a_src
        };

    let emit_b_tile_src =
        |alloc: &mut RegisterAllocator, kernel: &mut PtxKernel, k_tile: Register| -> Register {
            let k_x_bk = alloc.alloc(PtxType::U32);
            kernel.push(PtxInstruction::Arith(ArithOp::Mul {
                dst: k_x_bk,
                lhs: Operand::Reg(k_tile),
                rhs: Operand::ImmU32(BK), // 16
                ty: PtxType::U32,
            }));
            let b_k_row_off = alloc.alloc(PtxType::U64);
            kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
                dst: b_k_row_off,
                lhs: Operand::Reg(k_x_bk),
                rhs: Operand::Reg(r_n_bytes),
                src_ty: PtxType::U32,
            }));
            let b_src = alloc.alloc(PtxType::U64);
            kernel.push(PtxInstruction::Arith(ArithOp::Add {
                dst: b_src,
                lhs: Operand::Reg(rd_b_block_base),
                rhs: Operand::Reg(b_k_row_off),
                ty: PtxType::U64,
            }));
            b_src
        };

    // Compute buffer base (cur or nxt) for tile_a / tile_b.
    // toggle=false → cur = (k_tile & 1); toggle=true → nxt = 1 - cur.
    let emit_buf_base = |alloc: &mut RegisterAllocator,
                         kernel: &mut PtxKernel,
                         sym: Register,
                         k_tile: Register,
                         buf_bytes: u32,
                         toggle: bool|
     -> Register {
        let sel = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Rem {
            dst: sel,
            lhs: Operand::Reg(k_tile),
            rhs: Operand::ImmU32(2),
            ty: PtxType::U32,
        }));
        let sel_final = if toggle {
            let one_minus = alloc.alloc(PtxType::U32);
            kernel.push(PtxInstruction::Arith(ArithOp::Sub {
                dst: one_minus,
                lhs: Operand::ImmU32(1),
                rhs: Operand::Reg(sel),
                ty: PtxType::U32,
            }));
            one_minus
        } else {
            sel
        };
        let buf_off = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Mul {
            dst: buf_off,
            lhs: Operand::Reg(sel_final),
            rhs: Operand::ImmU32(buf_bytes),
            ty: PtxType::U32,
        }));
        let base = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: base,
            lhs: Operand::Reg(sym),
            rhs: Operand::Reg(buf_off),
            ty: PtxType::U32,
        }));
        base
    };

    // --- Preamble: issue A[0] async, sync-store B[0] ---
    let r_k_zero = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_k_zero,
        src: Operand::ImmU32(0),
        ty: PtxType::U32,
    });
    let tile_a_cur_pre = emit_buf_base(
        &mut alloc,
        &mut kernel,
        r_tile_a_sym,
        r_k_zero,
        TILE_A_BYTES,
        false,
    );
    let tile_b_cur_pre = emit_buf_base(
        &mut alloc,
        &mut kernel,
        r_tile_b_sym,
        r_k_zero,
        TILE_B_BYTES,
        false,
    );
    let a_tile_src_pre = emit_a_tile_src(&mut alloc, &mut kernel, r_k_zero);
    let b_tile_src_pre = emit_b_tile_src(&mut alloc, &mut kernel, r_k_zero);
    emit_mw_load_tile_a_64x16_async(
        &mut alloc,
        &mut kernel,
        a_tile_src_pre,
        tile_a_cur_pre,
        r_flat_tid,
        r_block_row,
        r_m,
        r_k_bytes,
        "PRE",
    );
    kernel.push(PtxInstruction::Memory(MemoryOp::CpAsyncCommitGroup));
    emit_mw_load_tile_b_16x64(
        &mut alloc,
        &mut kernel,
        b_tile_src_pre,
        tile_b_cur_pre,
        r_flat_tid,
        r_block_col,
        r_n,
        r_n_bytes,
        "PRE",
    );

    // --- K loop ---
    let r_k_tile = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_k_tile,
        src: Operand::ImmU32(0),
        ty: PtxType::U32,
    });

    kernel.push(PtxInstruction::Label("K_LOOP".to_string()));

    // Wait for A[k] to be resident.
    kernel.push(PtxInstruction::Memory(MemoryOp::CpAsyncWaitGroup { n: 0 }));
    kernel.push(PtxInstruction::Control(ControlOp::BarSync {
        barrier_id: 0,
    }));

    // has_next = (k_tile + 1) < num_k_tiles
    let r_k_tile_plus1 = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_k_tile_plus1,
        lhs: Operand::Reg(r_k_tile),
        rhs: Operand::ImmU32(1),
        ty: PtxType::U32,
    }));
    let p_has_next = alloc.alloc(PtxType::Pred);
    kernel.push(PtxInstruction::Control(ControlOp::SetP {
        dst: p_has_next,
        cmp_op: CmpOp::Lt,
        lhs: Operand::Reg(r_k_tile_plus1),
        rhs: Operand::Reg(r_num_k_tiles),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Control(ControlOp::BraPred {
        pred: p_has_next,
        target: "SKIP_NEXT_ISSUE".to_string(),
        negate: true,
    }));

    // Issue A[k+1] async + sync-store B[k+1] into the *nxt* buffers.
    let tile_a_nxt = emit_buf_base(
        &mut alloc,
        &mut kernel,
        r_tile_a_sym,
        r_k_tile,
        TILE_A_BYTES,
        true,
    );
    let tile_b_nxt = emit_buf_base(
        &mut alloc,
        &mut kernel,
        r_tile_b_sym,
        r_k_tile,
        TILE_B_BYTES,
        true,
    );
    let a_tile_src_nxt = emit_a_tile_src(&mut alloc, &mut kernel, r_k_tile_plus1);
    let b_tile_src_nxt = emit_b_tile_src(&mut alloc, &mut kernel, r_k_tile_plus1);
    emit_mw_load_tile_a_64x16_async(
        &mut alloc,
        &mut kernel,
        a_tile_src_nxt,
        tile_a_nxt,
        r_flat_tid,
        r_block_row,
        r_m,
        r_k_bytes,
        "ITER",
    );
    kernel.push(PtxInstruction::Memory(MemoryOp::CpAsyncCommitGroup));
    emit_mw_load_tile_b_16x64(
        &mut alloc,
        &mut kernel,
        b_tile_src_nxt,
        tile_b_nxt,
        r_flat_tid,
        r_block_col,
        r_n,
        r_n_bytes,
        "ITER",
    );

    kernel.push(PtxInstruction::Label("SKIP_NEXT_ISSUE".to_string()));

    // Compute on tile_{a,b}[cur]: per-warp shared base = sym + buf_cur_off + warp_off.
    let tile_a_cur = emit_buf_base(
        &mut alloc,
        &mut kernel,
        r_tile_a_sym,
        r_k_tile,
        TILE_A_BYTES,
        false,
    );
    let tile_b_cur = emit_buf_base(
        &mut alloc,
        &mut kernel,
        r_tile_b_sym,
        r_k_tile,
        TILE_B_BYTES,
        false,
    );
    let r_tile_a_warp_cur = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_tile_a_warp_cur,
        lhs: Operand::Reg(tile_a_cur),
        rhs: Operand::Reg(r_tile_a_warp_off),
        ty: PtxType::U32,
    }));
    let r_tile_b_warp_cur = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_tile_b_warp_cur,
        lhs: Operand::Reg(tile_b_cur),
        rhs: Operand::Reg(r_tile_b_warp_off),
        ty: PtxType::U32,
    }));

    // Per-warp 8-mma accumulation. Same helper as Gate A.
    emit_warp_quadrant_mma(
        &mut alloc,
        &mut kernel,
        r_tile_a_warp_cur,
        r_tile_b_warp_cur,
        r_tid_x,
        (r_hoisted_group_id, r_hoisted_tig),
        &mut accs,
    );

    // Advance k_tile and loop.
    kernel.push(PtxInstruction::Mov {
        dst: r_k_tile,
        src: Operand::Reg(r_k_tile_plus1),
        ty: PtxType::U32,
    });
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

    // --- Per-warp output store (same helper + same address protocol as Gate A) ---
    // Pass rd_d directly (NOT a per-block base) — warp_block_row /
    // warp_block_col are absolute, so store helper computes global
    // addresses from rd_d. (Gate A bug: double-counted block_row/col.)
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

/// Double-buffered tensor-core matmul — f16 × f16 → f32 with fp32
/// accumulation, cp.async-staged A tiles, multi-warp 64×64 block tile.
///
/// Sprint 6.7 Gate B: same 4-warp 32×32-quadrant restructure as
/// `matmul_tc`, with cp.async double-buffering on the A staging path.
/// Semantically equivalent to [`matmul_tc`](crate::matmul_tc) on the
/// same inputs; ships alongside it so Sprint 6.5's auto-tuner can
/// dispatch between the two based on profile data.
///
/// # Dimension constraint
///
/// Requires `M % 16 == 0`, `N % 8 == 0`, `K % 16 == 0`. Returns
/// [`KaioError::InvalidConfig`] otherwise. Sprint 6.7 Gate C will
/// relax M, N to "any positive value" (K%16 stays — mma K dim is
/// fixed) and promote this kernel from `#[doc(hidden)]` to stable
/// `pub` API.
///
/// # Hardware requirement
///
/// SM 8.0+ (Ampere) — required by both `mma.sync.m16n8k16` and
/// `cp.async.ca`. On a sub-Ampere device, returns
/// [`KaioError::Validation`] (wrapping
/// [`ValidationError::SmTooLow`](kaio_core::ir::ValidationError::SmTooLow))
/// via the `PtxModule::validate()` pass inside
/// [`KaioDevice::load_module`].
pub fn matmul_tc_async(
    device: &KaioDevice,
    a: &GpuBuffer<f16>,
    b: &GpuBuffer<f16>,
    c: &mut GpuBuffer<f32>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};

    validate_dims_tc(a, b, c, m, n, k)?;

    let info = device.info()?;
    let (major, minor) = info.compute_capability;
    let sm = format!("sm_{major}{minor}");
    let module = build_matmul_tc_async_module(&sm);

    let kmodule = device.load_module(&module)?;
    let func = kmodule.function("matmul_tc_async")?;

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

    /// Sprint 6.7 multi-warp buffer toggle.
    /// A per-buffer is 2048 B (unchanged in 6.7b). B per-buffer is 2560 B
    /// post-6.7b (col-stride padded 32→36, rounded up to multiple of 512
    /// for the cooperative pre-zero pass).
    #[test]
    fn buffer_offsets_toggle() {
        assert_eq!(buffer_offsets(0), (0, 2048, 0, 2560));
        assert_eq!(buffer_offsets(1), (2048, 0, 2560, 0));
        assert_eq!(buffer_offsets(2), (0, 2048, 0, 2560));
        assert_eq!(buffer_offsets(3), (2048, 0, 2560, 0));
    }

    #[test]
    fn build_matmul_tc_async_module_produces_valid_structure() {
        let module = build_matmul_tc_async_module("sm_89");
        let ptx = emit_module_to_string(&module);

        // Instruction presence.
        assert!(
            ptx.contains("cp.async.ca.shared.global"),
            "missing cp.async.ca"
        );
        assert!(
            ptx.contains("cp.async.commit_group"),
            "missing cp.async.commit_group"
        );
        assert!(
            ptx.contains("cp.async.wait_group"),
            "missing cp.async.wait_group"
        );
        assert!(
            ptx.contains("mma.sync.aligned.m16n8k16"),
            "missing mma.sync"
        );

        // Multi-warp shared sizing: 4 KB tile_a (2× 2048 B), 5 KB tile_b
        // (Sprint 6.7b: 2× 2560 B per buffer — padded col stride).
        assert!(
            ptx.contains(".shared .align 16 .b8 tile_a[4096]"),
            "tile_a should be 4096 B (2 buffers × 64×16 fp16)"
        );
        assert!(
            ptx.contains(".shared .align 4 .b8 tile_b[5120]"),
            "tile_b should be 5120 B (2 buffers × 2560 B padded col-stride)"
        );

        // 8 mma.sync per warp per K-tile (2 m_stripes × 4 n_stripes).
        let mma_count = ptx.matches("mma.sync.aligned.m16n8k16").count();
        assert_eq!(
            mma_count, 8,
            "expected 8 mma.sync per K-iter (multi-warp 2×4 stripe grid)"
        );

        // 2 cp.async.commit_group: preamble + in-loop.
        let commit_count = ptx.matches("cp.async.commit_group").count();
        assert_eq!(
            commit_count, 2,
            "expected two commit_groups (preamble + in-loop)"
        );

        // Edge-tile predication present.
        assert!(
            ptx.contains("setp.lt.and.u32"),
            "missing combined edge-tile predicate emit"
        );

        // 32 predicated st.global.f32 per warp output (8 fragments × 4 stores).
        let store_count = ptx.matches("st.global.f32").count();
        assert_eq!(store_count, 32, "expected 32 predicated st.global.f32");
    }

    #[test]
    fn build_matmul_tc_async_module_declares_requested_sm_target() {
        let module_70 = build_matmul_tc_async_module("sm_70");
        let ptx_70 = emit_module_to_string(&module_70);
        assert!(ptx_70.contains(".target sm_70"));

        let module_89 = build_matmul_tc_async_module("sm_89");
        let ptx_89 = emit_module_to_string(&module_89);
        assert!(ptx_89.contains(".target sm_89"));
    }

    #[test]
    fn matmul_tc_async_module_rejects_sm_70_via_validate() {
        use kaio_core::ir::ValidationError;

        let module = build_matmul_tc_async_module("sm_70");
        let err = module
            .validate()
            .expect_err("matmul_tc_async module at sm_70 must fail validation");
        match err {
            ValidationError::SmTooLow {
                required,
                actual,
                feature,
            } => {
                assert_eq!(required, 80);
                assert_eq!(actual, 70);
                assert!(
                    feature.contains("mma.sync") || feature.contains("cp.async"),
                    "unexpected feature name: {feature}"
                );
            }
        }
    }

    #[test]
    fn matmul_tc_async_module_validates_at_sm_80_and_above() {
        for sm in ["sm_80", "sm_89", "sm_90"] {
            let module = build_matmul_tc_async_module(sm);
            module
                .validate()
                .unwrap_or_else(|e| panic!("{sm} should validate; got error: {e}"));
        }
    }
}
