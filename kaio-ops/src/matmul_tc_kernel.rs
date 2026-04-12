//! Tensor-core matrix multiply — `m16n8k16.f32.f16.f16.f32`.
//!
//! f16 × f16 inputs with fp32 accumulation, using the `mma.sync.m16n8k16`
//! primitive introduced in Sprint 6.2. IR-authored (not DSL) because
//! `mma.sync` is warp-collective and the `#[gpu_kernel]` proc macro
//! models per-thread scalar code only.
//!
//! # Sprint 6.3 intent (correctness-first)
//!
//! One warp per block, one 16×8 output tile per block, K-dimension
//! looped. Grid dimensions are `(ceil(N/8), ceil(M/16), 1)`. For a
//! 4096×4096×4096 matmul this spawns ~32k blocks each doing tiny work
//! — **deliberately slow**. Sprint 6.7 replaces this block structure
//! with multi-warp tiles to hit the 60%+ cuBLAS target.
//!
//! The kernel stages A and B into shared memory each K-tile, loads
//! fragments from shared via the Sprint 6.3 `*_shared_*` helpers in
//! `kaio-core::fragment`, executes one `mma.sync` accumulating into
//! the D fragment, and stores D to global at the end. No `cp.async`
//! (that is Sprint 6.4). No `ldmatrix`. No bank-conflict padding.
//!
//! # Dimension constraint
//!
//! Requires `M % 16 == 0 && N % 8 == 0 && K % 16 == 0`. Violations
//! return `KaioError::InvalidConfig`. Edge-tile handling comes in 6.7.
//!
//! # Shared-memory layout (internal contract)
//!
//! - `tile_a` — 16 × 16 fp16, **row-major**, 32-byte row stride = 512 B.
//!   Matches global A layout; staging load is a pure copy.
//! - `tile_b` — 16 × 8 fp16, **column-major**, 32-byte column stride = 256 B.
//!   Global B is row-major, so the staging load **transposes** during
//!   the copy. Column-major shared layout is required by the B-fragment
//!   loader (two adjacent fp16 per half2 = two consecutive rows of the
//!   same column).
//!
//! 6.4 and 6.7 must treat these layouts as intentional policy, not
//! implementation detail.

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

const BM: u32 = 16; // output tile rows per block
const BN: u32 = 8; // output tile cols per block
const BK: u32 = 16; // K tile size per iteration
const BYTES_PER_HALF: u32 = 2;
const BYTES_PER_F32: u32 = 4;
const TILE_A_ROW_STRIDE_BYTES: u32 = BK * BYTES_PER_HALF; // 32
const TILE_B_COL_STRIDE_BYTES: u32 = BK * BYTES_PER_HALF; // 32 (column-major: column stride = row count × elem)
const TILE_A_BYTES: u32 = BM * BK * BYTES_PER_HALF; // 512
const TILE_B_BYTES: u32 = BK * BN * BYTES_PER_HALF; // 256

/// Validate dimension constraints for [`matmul_tc`].
fn validate_dims_tc(
    a: &GpuBuffer<f16>,
    b: &GpuBuffer<f16>,
    c: &GpuBuffer<f32>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    if m == 0 || n == 0 || k == 0 {
        return Err(KaioError::InvalidConfig(
            "matmul_tc dimensions must be non-zero".to_string(),
        ));
    }
    if !m.is_multiple_of(BM) {
        return Err(KaioError::InvalidConfig(format!(
            "matmul_tc: M must be a multiple of {BM} (got {m}). Sprint 6.7 will relax."
        )));
    }
    if !n.is_multiple_of(BN) {
        return Err(KaioError::InvalidConfig(format!(
            "matmul_tc: N must be a multiple of {BN} (got {n}). Sprint 6.7 will relax."
        )));
    }
    if !k.is_multiple_of(BK) {
        return Err(KaioError::InvalidConfig(format!(
            "matmul_tc: K must be a multiple of {BK} (got {k}). Sprint 6.7 will relax."
        )));
    }
    let mk = (m as usize) * (k as usize);
    let kn = (k as usize) * (n as usize);
    let mn = (m as usize) * (n as usize);
    if a.len() < mk {
        return Err(KaioError::InvalidConfig(format!(
            "matmul_tc: A buffer too small: need {mk} f16 ({m}×{k}), got {}",
            a.len()
        )));
    }
    if b.len() < kn {
        return Err(KaioError::InvalidConfig(format!(
            "matmul_tc: B buffer too small: need {kn} f16 ({k}×{n}), got {}",
            b.len()
        )));
    }
    if c.len() < mn {
        return Err(KaioError::InvalidConfig(format!(
            "matmul_tc: C buffer too small: need {mn} f32 ({m}×{n}), got {}",
            c.len()
        )));
    }
    Ok(())
}

/// Stage the 16×16 A tile row-major → row-major (pure copy).
///
/// 32 threads × 4 × `ld.global.b32` + 4 × `st.shared.b32` = 128 half2
/// pairs = full 16×16 fp16 tile. See [matmul_tc_kernel.rs docstring §
/// "Shared-memory layout"] for invariants.
///
/// Per-thread: `flat_half2_idx = lane_id * 4 + i` (i ∈ 0..4),
/// `row = flat / 8`, `col_pair = flat % 8`,
/// `global_byte_off = row * (K * 2) + col_pair * 4`,
/// `shared_byte_off = row * 32 + col_pair * 4`.
#[allow(clippy::too_many_arguments)]
fn emit_load_a_tile(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    a_tile_src_global: Register, // u64
    tile_a_shared: Register,     // u32
    tid_x: Register,
    k_bytes: Register, // u32: K * 2 (pre-computed by caller)
) {
    let lane_base = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: lane_base,
        lhs: Operand::Reg(tid_x),
        rhs: Operand::ImmU32(4),
        ty: PtxType::U32,
    }));

    for i in 0..4u32 {
        // flat = lane_base + i
        let flat = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: flat,
            lhs: Operand::Reg(lane_base),
            rhs: Operand::ImmU32(i),
            ty: PtxType::U32,
        }));
        let row = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Div {
            dst: row,
            lhs: Operand::Reg(flat),
            rhs: Operand::ImmU32(8),
            ty: PtxType::U32,
        }));
        let col_pair = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Rem {
            dst: col_pair,
            lhs: Operand::Reg(flat),
            rhs: Operand::ImmU32(8),
            ty: PtxType::U32,
        }));
        let col_pair_bytes = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Mul {
            dst: col_pair_bytes,
            lhs: Operand::Reg(col_pair),
            rhs: Operand::ImmU32(4),
            ty: PtxType::U32,
        }));

        // shared_addr = tile_a_shared + row*32 + col_pair*4
        let shared_off = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Mad {
            dst: shared_off,
            a: Operand::Reg(row),
            b: Operand::ImmU32(TILE_A_ROW_STRIDE_BYTES),
            c: Operand::Reg(col_pair_bytes),
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

        // global_addr = a_tile_src_global + row * k_bytes + col_pair_bytes
        // row_off_u32 = row * k_bytes  (fits in u32 for reasonable K; safer
        // to use mul.wide to produce u64 directly)
        let row_off64 = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
            dst: row_off64,
            lhs: Operand::Reg(row),
            rhs: Operand::Reg(k_bytes),
            src_ty: PtxType::U32,
        }));
        let col_pair_bytes_u64 = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Cvt {
            dst: col_pair_bytes_u64,
            src: col_pair_bytes,
            dst_ty: PtxType::U64,
            src_ty: PtxType::U32,
        });
        let per_thread_off64 = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: per_thread_off64,
            lhs: Operand::Reg(row_off64),
            rhs: Operand::Reg(col_pair_bytes_u64),
            ty: PtxType::U64,
        }));
        let global_addr = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: global_addr,
            lhs: Operand::Reg(a_tile_src_global),
            rhs: Operand::Reg(per_thread_off64),
            ty: PtxType::U64,
        }));

        // ld.global.b32 tmp, [global_addr]
        let tmp = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Memory(MemoryOp::LdGlobal {
            dst: tmp,
            addr: global_addr,
            ty: PtxType::U32,
        }));
        // st.shared.b32 [shared_addr], tmp
        kernel.push(PtxInstruction::Memory(MemoryOp::StShared {
            addr: shared_addr,
            src: tmp,
            ty: PtxType::U32,
        }));
    }
}

/// Stage B: row-major global → column-major shared, per-thread 4 × f16 copy.
///
/// Per-thread: `flat_half_idx = lane_id * 4 + i` (i ∈ 0..4),
/// `col = flat / 16`, `row = flat % 16`,
/// `global_byte_off = row * (N * 2) + col * 2`,
/// `shared_byte_off = col * 32 + row * 2`.
fn emit_load_b_tile(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    b_tile_src_global: Register, // u64: first byte of this block's B tile at k_tile=0 row
    tile_b_shared: Register,     // u32: shared offset of tile_b
    tid_x: Register,
    n_bytes: Register, // u32: N * 2
) {
    let lane_base = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: lane_base,
        lhs: Operand::Reg(tid_x),
        rhs: Operand::ImmU32(4),
        ty: PtxType::U32,
    }));

    for i in 0..4u32 {
        let flat = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: flat,
            lhs: Operand::Reg(lane_base),
            rhs: Operand::ImmU32(i),
            ty: PtxType::U32,
        }));
        let col = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Div {
            dst: col,
            lhs: Operand::Reg(flat),
            rhs: Operand::ImmU32(16),
            ty: PtxType::U32,
        }));
        let row = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Rem {
            dst: row,
            lhs: Operand::Reg(flat),
            rhs: Operand::ImmU32(16),
            ty: PtxType::U32,
        }));

        // shared_addr = tile_b_shared + col*32 + row*2
        let row_bytes = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Mul {
            dst: row_bytes,
            lhs: Operand::Reg(row),
            rhs: Operand::ImmU32(BYTES_PER_HALF),
            ty: PtxType::U32,
        }));
        let shared_off = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Mad {
            dst: shared_off,
            a: Operand::Reg(col),
            b: Operand::ImmU32(TILE_B_COL_STRIDE_BYTES),
            c: Operand::Reg(row_bytes),
            ty: PtxType::U32,
            mode: MadMode::Lo,
        }));
        let shared_addr = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: shared_addr,
            lhs: Operand::Reg(tile_b_shared),
            rhs: Operand::Reg(shared_off),
            ty: PtxType::U32,
        }));

        // global_addr = b_tile_src_global + row * n_bytes + col * 2
        let row_global_off64 = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
            dst: row_global_off64,
            lhs: Operand::Reg(row),
            rhs: Operand::Reg(n_bytes),
            src_ty: PtxType::U32,
        }));
        let col_bytes = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Mul {
            dst: col_bytes,
            lhs: Operand::Reg(col),
            rhs: Operand::ImmU32(BYTES_PER_HALF),
            ty: PtxType::U32,
        }));
        let col_bytes64 = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Cvt {
            dst: col_bytes64,
            src: col_bytes,
            dst_ty: PtxType::U64,
            src_ty: PtxType::U32,
        });
        let per_thread_off64 = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: per_thread_off64,
            lhs: Operand::Reg(row_global_off64),
            rhs: Operand::Reg(col_bytes64),
            ty: PtxType::U64,
        }));
        let global_addr = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: global_addr,
            lhs: Operand::Reg(b_tile_src_global),
            rhs: Operand::Reg(per_thread_off64),
            ty: PtxType::U64,
        }));

        // Single-fp16 load/store (no U16 in PtxType; use F16).
        // global row-major → shared column-major per-thread 4 × f16 copy
        // (not a packed-b32 optimization oversight — the two values at
        // the same destination come from non-adjacent global rows).
        let tmp_h = alloc.alloc(PtxType::F16);
        kernel.push(PtxInstruction::Memory(MemoryOp::LdGlobal {
            dst: tmp_h,
            addr: global_addr,
            ty: PtxType::F16,
        }));
        kernel.push(PtxInstruction::Memory(MemoryOp::StShared {
            addr: shared_addr,
            src: tmp_h,
            ty: PtxType::F16,
        }));
    }
}

/// Build the IR kernel text for `matmul_tc`.
///
/// See module docstring for the kernel algorithm.
fn build_matmul_tc_ptx() -> String {
    let mut alloc = RegisterAllocator::new();
    let mut kernel = PtxKernel::new("matmul_tc");

    kernel.add_param(PtxParam::pointer("a_ptr", PtxType::F16));
    kernel.add_param(PtxParam::pointer("b_ptr", PtxType::F16));
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

    // --- Special registers: tid.x, ctaid.x, ctaid.y ---
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

    // block_row = bidy * BM (= 16); block_col = bidx * BN (= 8)
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

    // K and N in bytes (u32), reused inside the K loop.
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

    // a_block_row_base = rd_a + block_row * K * 2  (u64)
    //   points to A row (block_row, 0) — the top-left of this block's A rows
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

    // b_block_col_base = rd_b + block_col * 2  (u64)
    //   points to B column (0, block_col)
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
    {
        // r_n_f32_stride = N * 4
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

    // Shared tile base offsets (u32).
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

    // D accumulator — zeros.
    let frag_d = alloc_c(&mut alloc);
    for r in &frag_d.regs {
        kernel.push(PtxInstruction::Mov {
            dst: *r,
            src: Operand::ImmF32(0.0),
            ty: PtxType::F32,
        });
    }

    // num_k_tiles = k / 16  (in the runtime params, not at build time)
    let r_num_k_tiles = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: r_num_k_tiles,
        lhs: Operand::Reg(r_k),
        rhs: Operand::ImmU32(BK),
        ty: PtxType::U32,
    }));

    // K loop
    let r_k_tile = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_k_tile,
        src: Operand::ImmU32(0),
        ty: PtxType::U32,
    });

    kernel.push(PtxInstruction::Label("K_LOOP".to_string()));

    // A tile global source = a_block_base + k_tile * BK * 2
    //                      = a_block_base + k_tile * 32
    let r_k_tile_x_bk_bytes = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_k_tile_x_bk_bytes,
        lhs: Operand::Reg(r_k_tile),
        rhs: Operand::ImmU32(BK * BYTES_PER_HALF), // 32
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

    // B tile global source = b_block_base + k_tile * BK * N * 2
    //                      = b_block_base + k_tile * (16 * N * 2)
    // Compute k_tile * 16 first, then multiply by N*2.
    let r_k_tile_x_bk = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_k_tile_x_bk,
        lhs: Operand::Reg(r_k_tile),
        rhs: Operand::ImmU32(BK), // 16
        ty: PtxType::U32,
    }));
    // b_k_row_off_u64 = (k_tile * 16) * n_bytes
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

    // Stage A and B.
    emit_load_a_tile(
        &mut alloc,
        &mut kernel,
        rd_a_tile_src,
        r_tile_a,
        r_tid,
        r_k_bytes,
    );
    emit_load_b_tile(
        &mut alloc,
        &mut kernel,
        rd_b_tile_src,
        r_tile_b,
        r_tid,
        r_n_bytes,
    );

    // bar.sync — wait for all shared writes to complete.
    kernel.push(PtxInstruction::Control(ControlOp::BarSync {
        barrier_id: 0,
    }));

    // Fragment loads from shared.
    let frag_a = load_fragment_a_m16n8k16_shared_row(
        &mut alloc,
        &mut kernel,
        r_tile_a,
        r_tid,
        TILE_A_ROW_STRIDE_BYTES,
    );
    let frag_b = load_fragment_b_m16n8k16_shared_col(
        &mut alloc,
        &mut kernel,
        r_tile_b,
        r_tid,
        TILE_B_COL_STRIDE_BYTES,
    );

    // D = A * B + D  (D acts as both accumulator input and output)
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

    // bar.sync — block-wide fence before the next K tile overwrites
    // shared staging. Redundant in this single-warp kernel but keeps
    // the structure correct once 6.7 introduces multi-warp blocks.
    kernel.push(PtxInstruction::Control(ControlOp::BarSync {
        barrier_id: 0,
    }));

    // k_tile += 1 ; if k_tile < num_k_tiles, loop.
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

    // Store D to global at output[block_row..+16][block_col..+8].
    // Output matrix is M×N row-major fp32 → row stride = N * 4 bytes.
    //
    // store_fragment_c accepts row_stride_bytes as a u32 immediate baked
    // into the PTX, but our N is a runtime kernel param. The helper
    // can't parameterize over a runtime value without a signature
    // change. For Sprint 6.3 we emit the store inline here instead —
    // the helper would need a register-based stride variant; add later
    // as tech debt if 6.4+ kernels need it.
    //
    // Inline per-thread store (mirrors store_fragment_c layout):
    //   groupID = tid / 4, tig = tid % 4
    //   reg[0]: d[groupID,   2*tig    ] → block_base + groupID*N*4 + tig*8
    //   reg[1]: d[groupID,   2*tig + 1] → block_base + groupID*N*4 + tig*8 + 4
    //   reg[2]: d[groupID+8, 2*tig    ] → block_base + (groupID+8)*N*4 + tig*8
    //   reg[3]: d[groupID+8, 2*tig + 1] → block_base + (groupID+8)*N*4 + tig*8 + 4
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

    // n_f32_stride = N * 4 (reusing r_n)
    let r_n_f32_stride = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_n_f32_stride,
        lhs: Operand::Reg(r_n),
        rhs: Operand::ImmU32(BYTES_PER_F32),
        ty: PtxType::U32,
    }));

    // row_off_u32 = group_id * N * 4
    let d_row_off32 = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: d_row_off32,
        lhs: Operand::Reg(d_group_id),
        rhs: Operand::Reg(r_n_f32_stride),
        ty: PtxType::U32,
    }));
    // base_off = row_off + tig * 8
    let d_base_off32 = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mad {
        dst: d_base_off32,
        a: Operand::Reg(d_tig),
        b: Operand::ImmU32(8),
        c: Operand::Reg(d_row_off32),
        ty: PtxType::U32,
        mode: MadMode::Lo,
    }));
    // base_off_plus_8rows = base_off + 8 * N * 4
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

    // Build u64 addresses for all four output positions and store.
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

    // m16n8k16 requires SM 8.0+. Floor at sm_80.
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

/// Tensor-core matmul kernel — f16 × f16 → f32 with fp32 accumulation.
///
/// NOTE: single-warp-per-block for correctness validation (Sprint 6.3).
/// Production tiling + performance come in Sprint 6.7 — see
/// `docs/development/sprints/phase6/sprint_6_3.md` for the intent.
///
/// # Dimension constraint
///
/// Requires `M % 16 == 0`, `N % 8 == 0`, `K % 16 == 0`. Returns
/// [`KaioError::InvalidConfig`] otherwise. Sprint 6.7 will relax this
/// with edge-tile bounds checking.
///
/// # Layout
///
/// A is M×K row-major, B is K×N row-major, D is M×N row-major — the
/// standard convention. Internally B is transposed on the way into
/// shared memory (see the `matmul_tc_kernel` module docstring).
pub fn matmul_tc(
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

    let ptx = build_matmul_tc_ptx();

    // Runtime SM check via device.info() — early clean rejection if
    // someone tries to run this on pre-Ampere hardware. Technically
    // redundant with `PtxModule::validate()` (which would catch the
    // same case on the IR tree), but the IR tree is built and consumed
    // as a string inside `build_matmul_tc_ptx`. The device-info path is
    // simpler and avoids a round-trip. See tech-debt note on
    // `load_ptx(&str)` bypassing `validate()`.
    let info = device.info()?;
    let (major, _minor) = info.compute_capability;
    if major < 8 {
        return Err(KaioError::InvalidConfig(format!(
            "matmul_tc requires SM 8.0+ (Ampere). GPU compute capability is {}.{}",
            info.compute_capability.0, info.compute_capability.1,
        )));
    }

    let kmodule = device.load_ptx(&ptx).map_err(|e| {
        KaioError::PtxLoad(format!(
            "matmul_tc PTX load failed: {e}\n\n=== PTX ===\n{ptx}"
        ))
    })?;
    let func = kmodule.function("matmul_tc")?;

    let grid = (n.div_ceil(BN), m.div_ceil(BM), 1);
    let cfg = LaunchConfig {
        grid_dim: grid,
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0, // tile_a + tile_b are declared statically
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

    // --- Host-only validate_dims_tc tests (no GPU required) ---
    //
    // These don't need #[ignore] — they exercise the validation path
    // without touching the device. Cover each invalid-shape case plus
    // buffer-too-small.
    //
    // We synthesize empty GpuBuffers via a helper that fabricates a
    // dangling slice; validate_dims_tc only reads `len()`, so we can't
    // construct real buffers without a GPU. Instead we test validate_dims_tc
    // indirectly through a struct that mimics the interface.

    // Note: GpuBuffer requires a GPU to construct, so pure host tests
    // of validate_dims_tc aren't trivial. We provide a wrapper that
    // accepts raw lengths for host testing.
    fn validate_dims_raw(
        m: u32,
        n: u32,
        k: u32,
        a_len: usize,
        b_len: usize,
        c_len: usize,
    ) -> Result<()> {
        if m == 0 || n == 0 || k == 0 {
            return Err(KaioError::InvalidConfig(
                "matmul_tc dimensions must be non-zero".to_string(),
            ));
        }
        if !m.is_multiple_of(BM) {
            return Err(KaioError::InvalidConfig(format!(
                "matmul_tc: M must be a multiple of {BM} (got {m}). Sprint 6.7 will relax."
            )));
        }
        if !n.is_multiple_of(BN) {
            return Err(KaioError::InvalidConfig(format!(
                "matmul_tc: N must be a multiple of {BN} (got {n}). Sprint 6.7 will relax."
            )));
        }
        if !k.is_multiple_of(BK) {
            return Err(KaioError::InvalidConfig(format!(
                "matmul_tc: K must be a multiple of {BK} (got {k}). Sprint 6.7 will relax."
            )));
        }
        let mk = (m as usize) * (k as usize);
        let kn = (k as usize) * (n as usize);
        let mn = (m as usize) * (n as usize);
        if a_len < mk {
            return Err(KaioError::InvalidConfig(format!(
                "matmul_tc: A buffer too small: need {mk} f16 ({m}×{k}), got {a_len}"
            )));
        }
        if b_len < kn {
            return Err(KaioError::InvalidConfig(format!(
                "matmul_tc: B buffer too small: need {kn} f16 ({k}×{n}), got {b_len}"
            )));
        }
        if c_len < mn {
            return Err(KaioError::InvalidConfig(format!(
                "matmul_tc: C buffer too small: need {mn} f32 ({m}×{n}), got {c_len}"
            )));
        }
        Ok(())
    }

    #[test]
    fn validate_dims_rejects_zero() {
        let err = validate_dims_raw(0, 8, 16, 0, 128, 0).unwrap_err();
        assert!(matches!(err, KaioError::InvalidConfig(ref m) if m.contains("non-zero")));
    }

    #[test]
    fn validate_dims_rejects_m_not_multiple_of_16() {
        let err = validate_dims_raw(17, 8, 16, 1000, 1000, 1000).unwrap_err();
        assert!(
            matches!(err, KaioError::InvalidConfig(ref m) if m.contains("M must be a multiple of 16"))
        );
    }

    #[test]
    fn validate_dims_rejects_n_not_multiple_of_8() {
        let err = validate_dims_raw(16, 5, 16, 1000, 1000, 1000).unwrap_err();
        assert!(
            matches!(err, KaioError::InvalidConfig(ref m) if m.contains("N must be a multiple of 8"))
        );
    }

    #[test]
    fn validate_dims_rejects_k_not_multiple_of_16() {
        let err = validate_dims_raw(16, 8, 24, 1000, 1000, 1000).unwrap_err();
        assert!(
            matches!(err, KaioError::InvalidConfig(ref m) if m.contains("K must be a multiple of 16"))
        );
    }

    #[test]
    fn validate_dims_rejects_buffer_too_small_a() {
        // M=16, K=16 → need 256 f16 in A; give it 100
        let err = validate_dims_raw(16, 8, 16, 100, 1000, 1000).unwrap_err();
        assert!(
            matches!(err, KaioError::InvalidConfig(ref m) if m.contains("A buffer too small")),
            "got: {err:?}"
        );
    }

    #[test]
    fn validate_dims_accepts_valid_shapes() {
        assert!(validate_dims_raw(16, 8, 16, 256, 128, 128).is_ok());
        assert!(validate_dims_raw(64, 64, 64, 64 * 64, 64 * 64, 64 * 64).is_ok());
        assert!(validate_dims_raw(128, 8, 16, 128 * 16, 16 * 8, 128 * 8).is_ok());
    }

    #[test]
    fn build_matmul_tc_ptx_produces_valid_structure() {
        let ptx = build_matmul_tc_ptx();
        // Sanity: check for expected tokens.
        assert!(ptx.contains(".visible .entry matmul_tc("));
        assert!(ptx.contains(".shared .align 4 .b8 tile_a[512]"));
        assert!(ptx.contains(".shared .align 4 .b8 tile_b[256]"));
        assert!(ptx.contains("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"));
        assert!(ptx.contains("bar.sync"));
        // Has a K loop (label + branch)
        assert!(ptx.contains("K_LOOP:"));
        assert!(ptx.contains("bra K_LOOP"));
    }
}
