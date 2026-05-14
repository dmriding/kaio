//! Tensor-core matrix multiply — `m16n8k16.f32.bf16.bf16.f32`, multi-warp,
//! cp.async double-buffered (Sprint 9.1.1).
//!
//! bf16 × bf16 inputs with fp32 accumulation, layered on top of Sprint
//! 6.7c's cp.async-pipelined `matmul_tc_async`. Structurally the
//! cross-product of [`crate::matmul_tc_async_kernel`] (f16 async, the
//! pipeline structure) and [`crate::matmul_tc_bf16_kernel`] (bf16 sync,
//! the precision-specific mma path):
//!
//! ```text
//!               f16              bf16
//! sync     matmul_tc        matmul_tc_bf16        (Sprint 9.1)
//! async    matmul_tc_async  matmul_tc_bf16_async  (Sprint 9.1.1)
//! ```
//!
//! Same 64×64 block tile, 4-warp 32×32 quadrant layout, padded Tile B
//! col-stride (Sprint 6.7b), 8 mma per K-tile, edge-tile predication
//! on M and N, fragment-loader `(group_id, tig)` hoist.
//!
//! - **A staging:** double-buffered via `cp.async.ca.shared.global`
//!   (size = 16). 128 threads × 1 issue per thread = 2,048 B = one
//!   full A buffer. The async A-tile loader
//!   [`crate::matmul_tc_async_kernel::emit_mw_load_tile_a_64x16_async`]
//!   is precision-agnostic at the byte level (the cp.async issue
//!   carries no dtype) — promoted to `pub(crate)` in Sprint 9.1.1 C0
//!   and reused here verbatim.
//! - **B staging:** synchronous — the row-major → column-major
//!   transpose during staging is a strided gather that `cp.async`
//!   cannot express. Reuses
//!   [`crate::matmul_tc_kernel::emit_mw_load_tile_b_16x64`] (also
//!   precision-agnostic at the storage-bit level; the `.f16`
//!   register-class label on the load/store moves raw 16-bit
//!   payloads without numeric conversion).
//! - **mma:** dedicated [`kaio_core::instr::TensorCoreOp::MmaSyncBf16`]
//!   via [`crate::matmul_tc_bf16_kernel::emit_warp_quadrant_mma_bf16`]
//!   (promoted to `pub(crate)` in Sprint 9.1.1 C0).
//!
//! # Dimension constraint
//!
//! Identical to [`crate::matmul_tc_bf16_kernel::matmul_tc_bf16`]:
//! `K % 16 == 0`. M and N may be any positive value (edge-tile
//! predication handles non-multiple-of-64). Validation shared via
//! [`crate::matmul_tc_bf16_kernel::validate_dims_tc_bf16`].
//!
//! # Shared-memory layout (2 buffers per tile, multi-warp sized)
//!
//! ```text
//! .shared .align 16 .b8 tile_a[4096];   // 2 × 2048 B, row-major per buffer
//! .shared .align 4  .b8 tile_b[5120];   // 2 × 2560 B padded col-stride
//! ```
//!
//! Per-buffer: `tile_a` 64×16 bf16 (2,048 B), `tile_b` 64×16 bf16
//! padded col-stride (2,560 B per Sprint 6.7b). Total shared per
//! block: **9 KB** — well under the 48 KB ceiling.
//!
//! # Pipeline ordering
//!
//! ```text
//! preamble:
//!     pre-zero tile_a + tile_b (9 KB cooperatively)
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
//!     emit_warp_quadrant_mma_bf16 on tile_{a,b}[cur]   ; 8 mma per warp
//!     cur, nxt = nxt, cur (implicit via k_tile + 1)
//! ```
//!
//! Same multi-warp safety argument as Sprint 6.7c: disjoint cur/nxt
//! buffers + top-of-iter `bar.sync` together fence reads from writes.
//!
//! Sprint 9.1.1 commit plan: C1 (this commit) ships the kernel module
//! builder + host validation tests + the D6 cvt-free hot-path gate.
//! C2 adds the public host launch fn [`matmul_tc_bf16_async`] and the
//! full 25-test correctness suite (`tests/matmul_tc_bf16_async_correctness.rs`).
//! C3 adds the bench + SC-2 perf-parity gate.

use crate::matmul_tc_async_kernel::emit_mw_load_tile_a_64x16_async;
use crate::matmul_tc_bf16_kernel::{emit_warp_quadrant_mma_bf16, validate_dims_tc_bf16};
use crate::matmul_tc_kernel::{
    TILE_A_BYTES, TILE_A_ROW_STRIDE_BYTES, TILE_B_BYTES, TILE_B_COL_STRIDE_BYTES,
    emit_mw_load_tile_b_16x64, emit_pre_zero_shared_tiles,
};
use half::bf16;
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

// --- mma.sync.m16n8k16 instance shape ---
const BK: u32 = 16;

// --- Multi-warp block tiling (mirrors matmul_tc_async_kernel) ---
const BM_BLOCK: u32 = 64;
const BN_BLOCK: u32 = 64;
const WARP_QUAD_M: u32 = 32;
const WARP_QUAD_N: u32 = 32;
pub(crate) const WARPS_PER_BLOCK: u32 = 4;

// --- Shared tile sizes (per buffer; 2 buffers each for double-buffering) ---
const BYTES_PER_HALF: u32 = 2; // bf16 storage = 2 bytes per element (same as f16)
const BYTES_PER_F32: u32 = 4;
const TILE_A_TOTAL_BYTES: u32 = 2 * TILE_A_BYTES; // 4096 (2 buffers)
const TILE_B_TOTAL_BYTES: u32 = 2 * TILE_B_BYTES; // 5120 (2 × 2560 padded col-stride)

/// Build the IR module for `matmul_tc_bf16_async` targeting the given SM.
///
/// Multi-warp 64×64 block tile, 4 warps × 32×32 quadrant via 8
/// `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` per K-tile,
/// with double-buffered cp.async A staging and sync B staging. Same
/// staging contracts as `matmul_tc_async`; see the module docstring
/// for the pipeline ordering.
///
/// `sm` is a PTX target string such as `"sm_89"`. Sub-Ampere targets
/// are legal at build time; `PtxModule::validate()` inside
/// `KaioDevice::load_module` rejects them via `ValidationError::SmTooLow`
/// (the combined `mma.sync.bf16` + `cp.async.ca` feature gate fires at
/// SM 8.0+).
pub(crate) fn build_matmul_tc_bf16_async_module(sm: &str) -> PtxModule {
    let mut alloc = RegisterAllocator::new();
    let mut kernel = PtxKernel::new("matmul_tc_bf16_async");

    kernel.add_param(PtxParam::pointer("a_ptr", PtxType::BF16));
    kernel.add_param(PtxParam::pointer("b_ptr", PtxType::BF16));
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

    // Pre-zero shared (all 9 KB: tile_a 4096 + tile_b 5120).
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

    // Helper closures — capture rd_*_block_base, r_*_bytes, r_tile_*_sym.
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

    // Per-warp 8-mma accumulation — bf16 mma variant.
    emit_warp_quadrant_mma_bf16(
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

    // --- Per-warp output store ---
    // Pass rd_d directly (NOT a per-block base) — warp_block_row /
    // warp_block_col are absolute, so store helper computes global
    // addresses from rd_d.
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
    crate::matmul_tc_kernel::emit_warp_quadrant_store(
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

/// Double-buffered tensor-core matmul — bf16 × bf16 → f32 with fp32
/// accumulation, cp.async-pipelined A staging, multi-warp 64×64 block tile.
///
/// Sprint 9.1.1 — the async sibling of
/// [`crate::matmul_tc_bf16`](crate::matmul_tc_bf16). Same multi-warp
/// 64×64 block tile structure, edge-tile predication on M and N (only
/// `K % 16 == 0` is enforced — the mma K-tile is structural), and same
/// `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` instance via
/// the dedicated [`kaio_core::instr::TensorCoreOp::MmaSyncBf16`] IR
/// variant.
///
/// The async path overlaps the K-loop's A-tile loads with the
/// previous iteration's mma compute via `cp.async.ca.shared.global`
/// double-buffering (Sprint 6.7c pattern applied to bf16). The B-tile
/// staging stays synchronous since the row-major → column-major
/// transpose is a strided gather `cp.async` cannot express.
///
/// Semantically equivalent to [`crate::matmul_tc_bf16`] on the same
/// inputs; ships alongside it so the future
/// `matmul_auto_tc_bf16` auto-tuner (Sprint 9.1.2) can dispatch
/// between the two based on profile data.
///
/// # Dimension constraint
///
/// Requires `K % 16 == 0`. M and N may be any positive value
/// (edge-tile predication handles non-multiple-of-64). Returns
/// [`KaioError::InvalidConfig`] otherwise. Validation shared with the
/// sync sibling via `matmul_tc_bf16_kernel::validate_dims_tc_bf16`
/// (private; see source).
///
/// # Hardware requirement
///
/// SM 8.0+ (Ampere) — required by both `mma.sync.bf16` and
/// `cp.async.ca`. Sub-Ampere targets are rejected by
/// [`PtxModule::validate()`](kaio_core::ir::PtxModule::validate) via
/// [`ValidationError::SmTooLow`](kaio_core::ir::ValidationError::SmTooLow)
/// before driver dispatch.
///
/// # Layout
///
/// A is M×K row-major, B is K×N row-major, D is M×N row-major. B is
/// transposed on the way into shared memory (column-major) by the
/// reused `pub(crate)` tile-B loader from `matmul_tc_kernel` (private
/// module; see source) — bf16 byte layout is bit-identical to f16 in
/// shared memory, so the same loader works for both precisions.
pub fn matmul_tc_bf16_async(
    device: &KaioDevice,
    a: &GpuBuffer<bf16>,
    b: &GpuBuffer<bf16>,
    c: &mut GpuBuffer<f32>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};

    validate_dims_tc_bf16(a, b, c, m, n, k)?;

    let info = device.info()?;
    let (major, minor) = info.compute_capability;
    let sm = format!("sm_{major}{minor}");
    let module = build_matmul_tc_bf16_async_module(&sm);

    let kmodule = device.load_module(&module)?;
    let func = kmodule.function("matmul_tc_bf16_async")?;

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

    #[test]
    fn build_matmul_tc_bf16_async_module_produces_valid_structure() {
        let module = build_matmul_tc_bf16_async_module("sm_89");
        let ptx = emit_module_to_string(&module);

        // Entry symbol present.
        assert!(ptx.contains(".visible .entry matmul_tc_bf16_async("));

        // Shared layout: tile_a 4096 (2 × 2048 B, align 16 for cp.async.ca),
        // tile_b 5120 (2 × 2560 B Sprint 6.7b padded col-stride).
        assert!(
            ptx.contains(".shared .align 16 .b8 tile_a[4096]"),
            "tile_a should be 4096 B with align 16 (cp.async.ca size=16 dst alignment)"
        );
        assert!(
            ptx.contains(".shared .align 4 .b8 tile_b[5120]"),
            "tile_b should be 5120 B (2 buffers × 2560 B padded col-stride per Sprint 6.7b)"
        );

        // cp.async issue/commit/wait structure. The async kernel has
        // EXACTLY TWO static cp.async.ca issue sites (preamble + in-loop
        // next-iter), matching the 2 commit_group instances. Same
        // structural count as the f16 async sibling.
        let cp_async_ca_count = ptx.matches("cp.async.ca.shared.global").count();
        assert_eq!(
            cp_async_ca_count, 2,
            "expected exactly 2 cp.async.ca.shared.global issue sites (preamble + in-loop), got {cp_async_ca_count}"
        );
        let commit_count = ptx.matches("cp.async.commit_group").count();
        assert_eq!(
            commit_count, 2,
            "expected exactly 2 cp.async.commit_group instances (matching cp.async.ca count)"
        );
        assert!(
            ptx.contains("cp.async.wait_group"),
            "missing cp.async.wait_group"
        );

        // No `.cg` cache-global variant — we deliberately use `.ca`
        // (cache-all) to match the f16 async kernel's choice.
        assert!(
            !ptx.contains("cp.async.cg.shared.global"),
            "matmul_tc_bf16_async must not emit cp.async.cg (we use .ca)"
        );

        // Dedicated bf16 mma mnemonic; no f16 mma tag.
        assert!(
            ptx.contains("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"),
            "expected dedicated MmaSyncBf16 mnemonic"
        );
        assert!(
            !ptx.contains("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"),
            "bf16 async kernel must not emit any f16-tagged mma.sync"
        );
        let mma_count = ptx.matches("mma.sync.aligned.m16n8k16").count();
        assert_eq!(
            mma_count, 8,
            "expected 8 mma.sync per K-iter (multi-warp 2×4 stripe grid)"
        );

        // Edge-tile predication present.
        assert!(
            ptx.contains("setp.lt.and.u32"),
            "missing combined edge-tile predicate emit"
        );

        // bar.sync for cross-warp visibility.
        assert!(ptx.contains("bar.sync"), "missing bar.sync");

        // 32 predicated st.global.f32 per warp output (8 fragments × 4 stores).
        let store_count = ptx.matches("st.global.f32").count();
        assert_eq!(
            store_count, 32,
            "expected 32 predicated st.global.f32 (8 fragments × 4 stores)"
        );
    }

    #[test]
    fn build_matmul_tc_bf16_async_module_declares_requested_sm_target() {
        let module_70 = build_matmul_tc_bf16_async_module("sm_70");
        assert!(emit_module_to_string(&module_70).contains(".target sm_70"));
        let module_89 = build_matmul_tc_bf16_async_module("sm_89");
        assert!(emit_module_to_string(&module_89).contains(".target sm_89"));
    }

    #[test]
    fn matmul_tc_bf16_async_module_rejects_sm_70_via_validate() {
        use kaio_core::ir::ValidationError;

        let module = build_matmul_tc_bf16_async_module("sm_70");
        let err = module
            .validate()
            .expect_err("matmul_tc_bf16_async module at sm_70 must fail validation");
        match err {
            ValidationError::SmTooLow {
                required,
                actual,
                feature,
            } => {
                assert_eq!(required, 80);
                assert_eq!(actual, 70);
                // Permissive `||` pattern matching the existing
                // matmul_tc_async test at matmul_tc_async_kernel.rs:1115
                // — don't introduce a stricter standard mid-sprint;
                // cross-precision test-consistency tightening is
                // out-of-scope here.
                assert!(
                    feature.contains("mma.sync") || feature.contains("cp.async"),
                    "unexpected feature name: {feature}"
                );
            }
            other => panic!("expected SmTooLow, got {other:?}"),
        }
    }

    #[test]
    fn matmul_tc_bf16_async_module_validates_at_sm_80_and_above() {
        for sm in ["sm_80", "sm_89", "sm_90"] {
            let module = build_matmul_tc_bf16_async_module(sm);
            module
                .validate()
                .unwrap_or_else(|e| panic!("{sm} should validate; got error: {e}"));
        }
    }

    /// Sprint 9.1.1 D6 resolution gate (port of 9.1's D4 cvt-free gate
    /// onto the async path): the inner K-loop hot path must contain
    /// **no `cvt.*` instructions** between any `ld.shared.b32` fragment
    /// load and the nearest subsequent
    /// `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`. A `cvt`
    /// on that hot path would indicate accidental f16↔bf16 conversion
    /// inserted somewhere on the fragment-load → mma flow — exactly
    /// the precision bug class 9.1's D4 gate was written to prevent.
    ///
    /// Why this matters more in async than sync: the async kernel has
    /// more in-loop arithmetic (buffer toggle math, has_next predicate,
    /// next-iter cp.async issue), which is more PTX surface where an
    /// accidental cvt could hide.
    ///
    /// Acceptable `cvt`s elsewhere: host-side scale lowering, address
    /// widening (`cvt.u32.s32`, `cvt.u64.u32`), output-store cvt. None
    /// of these appear between an `ld.shared.b32` and the next
    /// `mma.sync.bf16` on the hot path.
    #[test]
    fn d4_gate_no_cvt_in_bf16_mma_hot_path() {
        let module = build_matmul_tc_bf16_async_module("sm_89");
        let ptx = emit_module_to_string(&module);

        let loop_start = ptx
            .find("K_LOOP:")
            .expect("K_LOOP label must appear in emitted PTX");
        let loop_end_rel = ptx[loop_start..]
            .rfind("K_LOOP")
            .expect("matching back-edge to K_LOOP must exist");
        let loop_body = &ptx[loop_start..loop_start + loop_end_rel];

        let mut between_ld_and_mma = false;
        for line in loop_body.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("ld.shared.b32") || trimmed.starts_with("ld.shared.u32") {
                between_ld_and_mma = true;
            } else if trimmed.starts_with("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32") {
                between_ld_and_mma = false;
            } else if between_ld_and_mma && trimmed.starts_with("cvt.") {
                panic!(
                    "D6 GATE FAILED: cvt found between fragment ld.shared.b32 and mma.sync.bf16 \
                     in K_LOOP body — would indicate accidental precision conversion on the bf16 \
                     async hot path.\nOffending line: `{line}`\n\n=== K_LOOP body ===\n{loop_body}"
                );
            }
        }
    }
}
