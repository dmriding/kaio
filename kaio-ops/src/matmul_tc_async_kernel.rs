//! Tensor-core matrix multiply with cp.async double-buffering
//! (Sprint 6.4).
//!
//! Layered on top of Sprint 6.3's `matmul_tc`: same dimension constraints,
//! same shared-memory layout contract (A row-major, B column-major), same
//! fragment loaders from `kaio-core::fragment`. The difference is the
//! staging path:
//!
//! - **A** is staged via `cp.async.ca.shared.global` (16 bytes per thread,
//!   one async issue per thread per K-tile). Next K-tile's A is issued
//!   **before** the current K-tile's `mma.sync`, so the memory fetch
//!   overlaps with compute.
//! - **B** stays synchronous — the row-major → column-major transpose
//!   during staging is a strided gather that `cp.async` cannot express
//!   (source must be contiguous bytes). Revisiting B with async staging
//!   is a Sprint 6.7 concern that will likely coincide with the multi-
//!   warp restructure. See `docs/development/sprints/phase6/sprint_6_4.md`
//!   §D3 for the full rationale.
//!
//! # Sprint 6.4 intent (correctness + pipeline-pattern soundness)
//!
//! Same one-warp-per-block, one 16×8 output tile per block structure as
//! 6.3. This is **not** a performance sprint — at 1 warp with 1 small
//! `mma.sync` per K-tile there's only ~16 cycles of compute to hide
//! ~100+ cycles of memory latency, so async may perform at or slightly
//! below sync on this workload. The goal is to prove the pipeline
//! **pattern** is correct and has a reusable skeleton for 6.7 to scale.
//!
//! # Dimension constraint
//!
//! Identical to `matmul_tc`: `M % 16 == 0 && N % 8 == 0 && K % 16 == 0`.
//! Validation is shared via [`validate_dims_tc`](crate::matmul_tc_kernel::validate_dims_tc).
//!
//! # Shared-memory layout (2 buffers per tile)
//!
//! ```text
//! .shared .align 16 .b8 tile_a[1024];   // 2 × 512 B, row-major per buffer
//! .shared .align 4  .b8 tile_b[512];    // 2 × 256 B, col-major per buffer
//! ```
//!
//! `tile_a` is `align = 16` because `cp.async.ca.shared.global` with
//! `size = 16` requires 16-byte shared alignment (PTX ISA §8.7.8.22.6).
//! `tile_b` stays `align = 4` — no cp.async writes there.
//!
//! Buffer selection at iteration `k_tile`: `cur = k_tile & 1`. Current
//! buffer offsets are computed via [`buffer_offsets`] each iter as
//! `(k_tile & 1) * buffer_size`. XOR-toggle is **not** used — it would
//! require the shared base to be size-aligned, which `SharedDecl` does
//! not guarantee.
//!
//! # Pipeline ordering
//!
//! ```text
//! preamble:
//!     cp.async A[0] → tile_a[0]; commit_group
//!     sync-store B[0] → tile_b[0]
//!
//! K_LOOP (k = 0 .. num_k_tiles):
//!     cp.async.wait_group 0               ; wait on A[k] to be resident
//!     bar.sync                             ; block-wide visibility
//!     if k + 1 < num_k_tiles:
//!         cp.async A[k+1] → tile_a[nxt]; commit_group
//!         sync-store B[k+1] → tile_b[nxt]
//!     load frag_a, frag_b from tile_{a,b}[cur]
//!     mma.sync                             ; D += A * B
//!     cur, nxt = nxt, cur
//! ```
//!
//! **Loop-entry invariant:** at the top of each `K_LOOP` iteration,
//! `tile_{a,b}_cur` name the tiles that have been *issued* (by the
//! preamble at k=0 or by iter k-1). The `wait_group 0` + `bar.sync` at
//! the top of the iter is what actually makes them *resident* and
//! *visible* to the fragment loads. `tile_{a,b}_nxt` name the free
//! buffers, ready to receive the next K-tile's issues.
//!
//! **No trailing `bar.sync` after `mma.sync`:** the mma reads from
//! `tile_{a,b}_cur`; the next iter's cp.async/sync-store writes go to
//! `tile_{a,b}_nxt`. Disjoint shared regions mean no race between
//! iterations — the `bar.sync` at the **top** of the next iter is what
//! fences the pending-A visibility. This differs from 6.3 intentionally:
//! 6.3's single-buffer design required a post-mma bar.sync to delimit
//! tile reuse; double-buffering removes that requirement.
//!
//! **B sync-store to `tile_b_nxt` is race-free by construction:** the
//! mma reads `tile_b_cur`, and the sync-store writes `tile_b_nxt`
//! (the other buffer). With one warp per block there is no
//! inter-thread visibility concern; in 6.7's multi-warp restructure
//! the top-of-iter `bar.sync` continues to handle that.

use half::f16;
use kaio::prelude::*;
use kaio_core::emit::{Emit, PtxWriter};
use kaio_core::fragment::{
    alloc_c, load_fragment_a_m16n8k16_shared_row, load_fragment_b_m16n8k16_shared_col,
};
use kaio_core::instr::control::{CmpOp, ControlOp};
use kaio_core::instr::memory::MemoryOp;
use kaio_core::instr::special;
use kaio_core::instr::{ArithOp, MadMode, MmaShape, TensorCoreOp};
use kaio_core::ir::{
    Operand, PtxInstruction, PtxKernel, PtxModule, PtxParam, Register, RegisterAllocator,
    SharedDecl, SpecialReg,
};
use kaio_core::types::PtxType;

use crate::matmul_tc_kernel::{emit_load_b_tile, validate_dims_tc};

const BM: u32 = 16;
const BN: u32 = 8;
const BK: u32 = 16;
const BYTES_PER_HALF: u32 = 2;
const BYTES_PER_F32: u32 = 4;
const TILE_A_ROW_STRIDE_BYTES: u32 = BK * BYTES_PER_HALF; // 32
const TILE_B_COL_STRIDE_BYTES: u32 = BK * BYTES_PER_HALF; // 32
const TILE_A_BYTES: u32 = BM * BK * BYTES_PER_HALF; // 512 per buffer
const TILE_B_BYTES: u32 = BK * BN * BYTES_PER_HALF; // 256 per buffer
const TILE_A_TOTAL_BYTES: u32 = 2 * TILE_A_BYTES; // 1024 (2 buffers)
const TILE_B_TOTAL_BYTES: u32 = 2 * TILE_B_BYTES; // 512  (2 buffers)

/// Pure helper: byte offsets of the (cur, nxt) buffer pair within
/// `tile_a` and `tile_b` at iteration `k_tile`.
///
/// Returns `(a_cur, a_nxt, b_cur, b_nxt)`, each a byte offset to add
/// to the corresponding `.shared` symbol base. Extracted as a pure
/// function so the toggle math is host-testable without a GPU — see
/// the `buffer_offsets_toggle` test in this module. The kernel
/// builder inlines equivalent arithmetic directly (it needs `Register`
/// operands, not `u32`s), so this helper is test-only in production
/// builds.
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

/// Stage the 16×16 A tile via `cp.async.ca.shared.global` (size = 16).
///
/// Per-thread: `flat_byte = lane * 16`, so each of 32 threads issues
/// exactly one 16-byte async copy. 32 × 16 = 512 bytes = one full A
/// tile.
///
/// Thread address layout:
/// - `row          = lane / 2`           (0..16)
/// - `col_pair_byte = (lane % 2) * 16`   (0 or 16)
/// - `shared_byte_off = row * 32 + col_pair_byte`
/// - `global_byte_off = row * K_bytes + col_pair_byte`
///   (the K-tile's `k_tile * 32` term is rolled into
///   `a_tile_src_global` by the caller, same pattern as
///   `emit_load_a_tile` in 6.3.)
///
/// SAFETY: shared dst and global src are both 16-byte aligned — see
/// sprint_6_4.md §D4. Invariant requires `BK = 16` (→ 32-byte row
/// stride) and `K % 16 == 0` (→ `K_bytes % 32 == 0`). Violating either
/// breaks this alignment contract silently.
///
/// Does **not** emit `cp.async.commit_group` — caller owns the commit
/// boundary so the preamble and the in-loop issue can each commit their
/// own group independently.
#[allow(clippy::too_many_arguments)]
fn emit_load_a_tile_async(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    a_tile_src_global: Register, // u64: start of this block's K-tile's A
    tile_a_shared: Register,     // u32: base of the chosen A buffer
    tid_x: Register,
    k_bytes: Register, // u32: K * 2
) {
    // row = lane / 2
    let row = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: row,
        lhs: Operand::Reg(tid_x),
        rhs: Operand::ImmU32(2),
        ty: PtxType::U32,
    }));
    // col_pair_byte = (lane % 2) * 16
    let lane_mod2 = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Rem {
        dst: lane_mod2,
        lhs: Operand::Reg(tid_x),
        rhs: Operand::ImmU32(2),
        ty: PtxType::U32,
    }));
    let col_pair_byte = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: col_pair_byte,
        lhs: Operand::Reg(lane_mod2),
        rhs: Operand::ImmU32(16),
        ty: PtxType::U32,
    }));

    // shared_addr = tile_a_shared + row * 32 + col_pair_byte
    let shared_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mad {
        dst: shared_off,
        a: Operand::Reg(row),
        b: Operand::ImmU32(TILE_A_ROW_STRIDE_BYTES),
        c: Operand::Reg(col_pair_byte),
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

    // global_addr = a_tile_src_global + row * K_bytes + col_pair_byte
    let row_off64 = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
        dst: row_off64,
        lhs: Operand::Reg(row),
        rhs: Operand::Reg(k_bytes),
        src_ty: PtxType::U32,
    }));
    let col_pair_byte64 = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Cvt {
        dst: col_pair_byte64,
        src: col_pair_byte,
        dst_ty: PtxType::U64,
        src_ty: PtxType::U32,
    });
    let per_thread_off64 = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: per_thread_off64,
        lhs: Operand::Reg(row_off64),
        rhs: Operand::Reg(col_pair_byte64),
        ty: PtxType::U64,
    }));
    let global_addr = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: global_addr,
        lhs: Operand::Reg(a_tile_src_global),
        rhs: Operand::Reg(per_thread_off64),
        ty: PtxType::U64,
    }));

    // SAFETY: shared dst and global src are 16-byte aligned.
    // See sprint_6_4.md §D4: requires BK = 16 (→ 32-byte rows) and
    // K % 16 == 0. Violating either breaks this invariant silently.
    kernel.push(PtxInstruction::Memory(MemoryOp::new_cp_async_ca(
        shared_addr,
        global_addr,
        16,
    )));
}

/// Build the IR kernel text for `matmul_tc_async`.
///
/// See module docstring for algorithm + pipeline diagram.
pub(crate) fn build_matmul_tc_async_ptx() -> String {
    let mut alloc = RegisterAllocator::new();
    let mut kernel = PtxKernel::new("matmul_tc_async");

    kernel.add_param(PtxParam::pointer("a_ptr", PtxType::F16));
    kernel.add_param(PtxParam::pointer("b_ptr", PtxType::F16));
    kernel.add_param(PtxParam::pointer("d_ptr", PtxType::F32));
    kernel.add_param(PtxParam::scalar("m", PtxType::U32));
    kernel.add_param(PtxParam::scalar("n", PtxType::U32));
    kernel.add_param(PtxParam::scalar("k", PtxType::U32));

    // tile_a: align 16 (required by cp.async.ca size=16 dst alignment).
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

    // --- Load params & convert pointers to global space ---
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
    let (r_tid, tid_instr) = special::tid_x(&mut alloc);
    kernel.push(tid_instr);
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

    // block_row = bidy * BM; block_col = bidx * BN
    let r_block_row = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_block_row,
        lhs: Operand::Reg(r_bidy),
        rhs: Operand::ImmU32(BM),
        ty: PtxType::U32,
    }));
    let r_block_col = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_block_col,
        lhs: Operand::Reg(r_bidx),
        rhs: Operand::ImmU32(BN),
        ty: PtxType::U32,
    }));

    // K and N in bytes (u32).
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

    // d_block_base = rd_d + block_row * N * 4 + block_col * 4
    let rd_d_row_off = alloc.alloc(PtxType::U64);
    let r_n_f32_stride_outer;
    {
        let r_n_f32_stride = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Mul {
            dst: r_n_f32_stride,
            lhs: Operand::Reg(r_n),
            rhs: Operand::ImmU32(BYTES_PER_F32),
            ty: PtxType::U32,
        }));
        kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
            dst: rd_d_row_off,
            lhs: Operand::Reg(r_block_row),
            rhs: Operand::Reg(r_n_f32_stride),
            src_ty: PtxType::U32,
        }));
        r_n_f32_stride_outer = r_n_f32_stride;
    }
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

    // Shared tile symbolic bases.
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

    // D accumulator — zeros.
    let frag_d = alloc_c(&mut alloc);
    for r in &frag_d.regs {
        kernel.push(PtxInstruction::Mov {
            dst: *r,
            src: Operand::ImmF32(0.0),
            ty: PtxType::F32,
        });
    }

    // num_k_tiles = k / BK
    let r_num_k_tiles = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: r_num_k_tiles,
        lhs: Operand::Reg(r_k),
        rhs: Operand::ImmU32(BK),
        ty: PtxType::U32,
    }));

    // Helper closure: compute `a_tile_src = a_block_base + k_tile * 32`
    // (global byte offset of A's K-tile start for this block).
    let emit_a_tile_src =
        |alloc: &mut RegisterAllocator, kernel: &mut PtxKernel, k_tile: Register| -> Register {
            let k_tile_x_32 = alloc.alloc(PtxType::U32);
            kernel.push(PtxInstruction::Arith(ArithOp::Mul {
                dst: k_tile_x_32,
                lhs: Operand::Reg(k_tile),
                rhs: Operand::ImmU32(BK * BYTES_PER_HALF), // 32
                ty: PtxType::U32,
            }));
            let k_tile_x_32_u64 = alloc.alloc(PtxType::U64);
            kernel.push(PtxInstruction::Cvt {
                dst: k_tile_x_32_u64,
                src: k_tile_x_32,
                dst_ty: PtxType::U64,
                src_ty: PtxType::U32,
            });
            let a_src = alloc.alloc(PtxType::U64);
            kernel.push(PtxInstruction::Arith(ArithOp::Add {
                dst: a_src,
                lhs: Operand::Reg(rd_a_block_base),
                rhs: Operand::Reg(k_tile_x_32_u64),
                ty: PtxType::U64,
            }));
            a_src
        };

    // Helper closure: compute `b_tile_src = b_block_base + k_tile * 16 * N_bytes`.
    let emit_b_tile_src =
        |alloc: &mut RegisterAllocator, kernel: &mut PtxKernel, k_tile: Register| -> Register {
            let k_tile_x_bk = alloc.alloc(PtxType::U32);
            kernel.push(PtxInstruction::Arith(ArithOp::Mul {
                dst: k_tile_x_bk,
                lhs: Operand::Reg(k_tile),
                rhs: Operand::ImmU32(BK),
                ty: PtxType::U32,
            }));
            let b_k_row_off = alloc.alloc(PtxType::U64);
            kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
                dst: b_k_row_off,
                lhs: Operand::Reg(k_tile_x_bk),
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

    // Helper closure: runtime base = sym + (k_tile & 1) * buf_size
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
            // nxt = (k_tile & 1) ^ 1  →  implement as 1 - sel
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
    // k_tile=0 → buffer 0 for both A and B.
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
    emit_load_a_tile_async(
        &mut alloc,
        &mut kernel,
        a_tile_src_pre,
        tile_a_cur_pre,
        r_tid,
        r_k_bytes,
    );
    kernel.push(PtxInstruction::Memory(MemoryOp::CpAsyncCommitGroup));
    emit_load_b_tile(
        &mut alloc,
        &mut kernel,
        b_tile_src_pre,
        tile_b_cur_pre,
        r_tid,
        r_n_bytes,
    );

    // --- K loop ---
    let r_k_tile = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_k_tile,
        src: Operand::ImmU32(0),
        ty: PtxType::U32,
    });

    kernel.push(PtxInstruction::Label("K_LOOP".to_string()));

    // Wait for the A cp.async for this iter to land.
    kernel.push(PtxInstruction::Memory(MemoryOp::CpAsyncWaitGroup { n: 0 }));
    kernel.push(PtxInstruction::Control(ControlOp::BarSync {
        barrier_id: 0,
    }));

    // has_next = k_tile + 1 < num_k_tiles
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

    // Issue A[k+1] → tile_a[nxt]; commit.
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
    emit_load_a_tile_async(
        &mut alloc,
        &mut kernel,
        a_tile_src_nxt,
        tile_a_nxt,
        r_tid,
        r_k_bytes,
    );
    kernel.push(PtxInstruction::Memory(MemoryOp::CpAsyncCommitGroup));
    emit_load_b_tile(
        &mut alloc,
        &mut kernel,
        b_tile_src_nxt,
        tile_b_nxt,
        r_tid,
        r_n_bytes,
    );

    kernel.push(PtxInstruction::Label("SKIP_NEXT_ISSUE".to_string()));

    // Compute on tile[cur].
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
    let frag_a = load_fragment_a_m16n8k16_shared_row(
        &mut alloc,
        &mut kernel,
        tile_a_cur,
        r_tid,
        TILE_A_ROW_STRIDE_BYTES,
    );
    let frag_b = load_fragment_b_m16n8k16_shared_col(
        &mut alloc,
        &mut kernel,
        tile_b_cur,
        r_tid,
        TILE_B_COL_STRIDE_BYTES,
    );
    kernel.push(PtxInstruction::TensorCore(TensorCoreOp::MmaSync {
        d: frag_d,
        a: frag_a,
        b: frag_b,
        c: frag_d,
        shape: MmaShape::M16N8K16,
        d_ty: PtxType::F32,
        a_ty: PtxType::F16,
        b_ty: PtxType::F16,
        c_ty: PtxType::F32,
    }));

    // k_tile = k_tile_plus1; continue if < num_k_tiles.
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

    // --- Store D to global (inline, matches 6.3's layout) ---
    let d_group_id = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: d_group_id,
        lhs: Operand::Reg(r_tid),
        rhs: Operand::ImmU32(4),
        ty: PtxType::U32,
    }));
    let d_tig = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Rem {
        dst: d_tig,
        lhs: Operand::Reg(r_tid),
        rhs: Operand::ImmU32(4),
        ty: PtxType::U32,
    }));

    let r_n_f32_stride = r_n_f32_stride_outer;
    let d_row_off32 = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: d_row_off32,
        lhs: Operand::Reg(d_group_id),
        rhs: Operand::Reg(r_n_f32_stride),
        ty: PtxType::U32,
    }));
    let d_base_off32 = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mad {
        dst: d_base_off32,
        a: Operand::Reg(d_tig),
        b: Operand::ImmU32(8),
        c: Operand::Reg(d_row_off32),
        ty: PtxType::U32,
        mode: MadMode::Lo,
    }));
    let r_eight_rows = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_eight_rows,
        lhs: Operand::Reg(r_n_f32_stride),
        rhs: Operand::ImmU32(8),
        ty: PtxType::U32,
    }));
    let d_base_plus_8r32 = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: d_base_plus_8r32,
        lhs: Operand::Reg(d_base_off32),
        rhs: Operand::Reg(r_eight_rows),
        ty: PtxType::U32,
    }));

    let emit_store = |alloc: &mut RegisterAllocator,
                      kernel: &mut PtxKernel,
                      base_off32: Register,
                      extra: u32,
                      src_reg: Register| {
        let off64 = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Cvt {
            dst: off64,
            src: base_off32,
            dst_ty: PtxType::U64,
            src_ty: PtxType::U32,
        });
        let addr = alloc.alloc(PtxType::U64);
        if extra == 0 {
            kernel.push(PtxInstruction::Arith(ArithOp::Add {
                dst: addr,
                lhs: Operand::Reg(rd_d_block_base),
                rhs: Operand::Reg(off64),
                ty: PtxType::U64,
            }));
        } else {
            let tmp = alloc.alloc(PtxType::U64);
            kernel.push(PtxInstruction::Arith(ArithOp::Add {
                dst: tmp,
                lhs: Operand::Reg(rd_d_block_base),
                rhs: Operand::Reg(off64),
                ty: PtxType::U64,
            }));
            kernel.push(PtxInstruction::Arith(ArithOp::Add {
                dst: addr,
                lhs: Operand::Reg(tmp),
                rhs: Operand::ImmU32(extra),
                ty: PtxType::U64,
            }));
        }
        kernel.push(PtxInstruction::Memory(MemoryOp::StGlobal {
            addr,
            src: src_reg,
            ty: PtxType::F32,
        }));
    };
    emit_store(&mut alloc, &mut kernel, d_base_off32, 0, frag_d.regs[0]);
    emit_store(&mut alloc, &mut kernel, d_base_off32, 4, frag_d.regs[1]);
    emit_store(&mut alloc, &mut kernel, d_base_plus_8r32, 0, frag_d.regs[2]);
    emit_store(&mut alloc, &mut kernel, d_base_plus_8r32, 4, frag_d.regs[3]);

    kernel.push(PtxInstruction::Control(ControlOp::Ret));
    kernel.set_registers(alloc.into_allocated());

    // cp.async + mma.sync.m16n8k16 require SM 8.0+.
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

/// Double-buffered tensor-core matmul — f16 × f16 → f32 with fp32
/// accumulation, cp.async-staged A tiles.
///
/// Semantically equivalent to [`matmul_tc`](crate::matmul_tc) on the
/// same inputs; differs only in the staging path. Ships alongside
/// `matmul_tc` so Sprint 6.5's auto-tuner can dispatch between the
/// two based on profile data.
///
/// # Dimension constraint
///
/// Requires `M % 16 == 0`, `N % 8 == 0`, `K % 16 == 0`. Returns
/// [`KaioError::InvalidConfig`] otherwise. Sprint 6.7 will relax this
/// with edge-tile bounds checking and also promote this kernel from
/// `#[doc(hidden)]` to stable `pub` API.
///
/// # Hardware requirement
///
/// SM 8.0+ (Ampere) — required by both `mma.sync.m16n8k16` and
/// `cp.async.ca`. Returns [`KaioError::InvalidConfig`] on lower SM.
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

    let ptx = build_matmul_tc_async_ptx();

    let info = device.info()?;
    let (major, _minor) = info.compute_capability;
    if major < 8 {
        return Err(KaioError::InvalidConfig(format!(
            "matmul_tc_async requires SM 8.0+ (cp.async + mma.sync.m16n8k16). \
             GPU compute capability is {}.{}",
            info.compute_capability.0, info.compute_capability.1,
        )));
    }

    let kmodule = device.load_ptx(&ptx).map_err(|e| {
        KaioError::PtxLoad(format!(
            "matmul_tc_async PTX load failed: {e}\n\n=== PTX ===\n{ptx}"
        ))
    })?;
    let func = kmodule.function("matmul_tc_async")?;

    let grid = (n.div_ceil(BN), m.div_ceil(BM), 1);
    let cfg = LaunchConfig {
        grid_dim: grid,
        block_dim: (32, 1, 1),
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

    /// Buffer-offset toggle unit test. Pure host — catches off-by-one
    /// bugs without needing a GPU or a full kernel build.
    #[test]
    fn buffer_offsets_toggle() {
        assert_eq!(buffer_offsets(0), (0, 512, 0, 256));
        assert_eq!(buffer_offsets(1), (512, 0, 256, 0));
        assert_eq!(buffer_offsets(2), (0, 512, 0, 256));
        assert_eq!(buffer_offsets(3), (512, 0, 256, 0));
    }

    /// Structural PTX check — instruction-centric, not label-centric.
    /// Label spellings can change without semantic regression; instruction
    /// presence and shared-decl sizing are the real invariants.
    #[test]
    fn build_matmul_tc_async_ptx_produces_valid_structure() {
        let ptx = build_matmul_tc_async_ptx();

        // Instruction presence — the semantic content.
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

        // Two distinct shared decls, sized correctly.
        assert!(
            ptx.contains(".shared .align 16 .b8 tile_a[1024]"),
            "tile_a decl wrong or missing"
        );
        assert!(
            ptx.contains(".shared .align 4 .b8 tile_b[512]"),
            "tile_b decl wrong or missing"
        );

        // Exactly one mma.sync in the kernel body (the inner-loop mma).
        let mma_count = ptx.matches("mma.sync.aligned.m16n8k16").count();
        assert_eq!(mma_count, 1, "expected exactly one mma.sync in loop body");

        // Exactly two cp.async.commit_group: preamble + loop-interior issue.
        let commit_count = ptx.matches("cp.async.commit_group").count();
        assert_eq!(
            commit_count, 2,
            "expected two commit_groups (preamble + in-loop)"
        );
    }
}
