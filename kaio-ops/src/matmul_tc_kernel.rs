//! Tensor-core matrix multiply — `m16n8k16.f32.f16.f16.f32`, multi-warp.
//!
//! f16 × f16 inputs with fp32 accumulation, using the `mma.sync.m16n8k16`
//! primitive introduced in Sprint 6.2. IR-authored (not DSL) because
//! `mma.sync` is warp-collective and the `#[gpu_kernel]` proc macro
//! models per-thread scalar code only.
//!
//! # Sprint 6.7 layout (multi-warp + edge tiles)
//!
//! - Block dim `(32, 4, 1)` — 4 warps per block, 128 threads total.
//! - Block tile = **64×64 output**, with each warp owning a **32×32
//!   sub-quadrant** in a 2×2 grid. `warp_id = tid_y` ∈ [0, 4).
//! - Per warp per K-tile: **8 × `mma.sync.m16n8k16`** in a 2×4 grid
//!   (2 row-stripes × 4 col-stripes × 16×8 mma output).
//! - Grid `(N.div_ceil(64), M.div_ceil(64), 1)`. Edge tiles handled by
//!   per-thread bounds predication on global loads (zero-pad on OOB)
//!   and global stores (skip on OOB), so `M` and `N` need not be
//!   multiples of 64.
//! - K-tile size unchanged at 16 (mma's K dimension). `K % 16 == 0` is
//!   still required (Gate C lifts in 6.7).
//!
//! # Shared-memory layout (internal contract)
//!
//! - `tile_a` — 64 × 16 fp16, **row-major**, row stride = 32 B.
//!   Total **2,048 B** per block (sync). All 4 warps share this region.
//! - `tile_b` — 16 × 64 fp16, **column-major**, column stride = 32 B.
//!   Total **2,048 B** per block (sync). All 4 warps share this region.
//!
//! Combined sync footprint: **4 KB / block**. Sprint 6.7b's vectorized
//! loads and bank-conflict padding may reshape this.
//!
//! # Per-warp quadrant mapping
//!
//! ```text
//! warp 0 (tid_y=0): rows [ 0..32), cols [ 0..32)
//! warp 1 (tid_y=1): rows [ 0..32), cols [32..64)
//! warp 2 (tid_y=2): rows [32..64), cols [ 0..32)
//! warp 3 (tid_y=3): rows [32..64), cols [32..64)
//! ```
//!
//! Each warp's 32×32 quadrant is covered by 8 mma.sync outputs arranged
//! as a 2×4 grid of 16×8 sub-tiles (m_stripe ∈ [0,2), n_stripe ∈ [0,4)).
//!
//! # Edge-tile handling
//!
//! Per-thread row/col bounds checks predicate global memory accesses:
//! - **A loads:** `setp.lt.u32 p_row, row_global, M;` → predicated
//!   `ld.global.b32` (OOB threads write zero into shared via a default
//!   `mov.b32 reg, 0` before the load).
//! - **B loads:** same pattern with `setp.lt.u32 p_col, col_global, N;`.
//! - **Output stores:** per-fragment 4 stores, each with
//!   `setp.lt.and.u32` combining the row check with the col check —
//!   `@p st.global.f32` skips entirely when OOB.
//!
//! Zero-padded fragments contribute exactly zero to the fp32
//! accumulator (`0.0 * x + acc = acc`); hardware doesn't distinguish
//! padded zeros from real zeros. All-OOB warps degenerate to no-op
//! blocks. Standard CUTLASS/Triton/cuDNN approach.

use half::f16;
use kaio::prelude::*;
use kaio_core::fragment::{
    FragmentA_F16, FragmentB_F16, FragmentC, alloc_c, load_fragment_a_m16n8k16_shared_row,
    load_fragment_b_m16n8k16_shared_col,
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

// --- mma.sync.m16n8k16 instance shape (used for validate constraint) ---
const BM: u32 = 16; // mma m dim
const BN: u32 = 8; // mma n dim
const BK: u32 = 16; // mma k dim (also K-tile granularity)

// --- Multi-warp block tiling (Sprint 6.7) ---
const BM_BLOCK: u32 = 64; // output rows per block
const BN_BLOCK: u32 = 64; // output cols per block
const WARP_QUAD_M: u32 = 32; // rows per warp quadrant
const WARP_QUAD_N: u32 = 32; // cols per warp quadrant
const MMAS_PER_WARP_M: u32 = WARP_QUAD_M / BM; // 2 row-stripes per warp
const MMAS_PER_WARP_N: u32 = WARP_QUAD_N / BN; // 4 col-stripes per warp
const WARPS_PER_BLOCK: u32 = 4;
const THREADS_PER_BLOCK: u32 = WARPS_PER_BLOCK * 32; // 128

// --- Shared tile sizes ---
const BYTES_PER_HALF: u32 = 2;
const BYTES_PER_F32: u32 = 4;
pub(crate) const TILE_A_ROW_STRIDE_BYTES: u32 = BK * BYTES_PER_HALF; // 32
pub(crate) const TILE_A_BYTES: u32 = BM_BLOCK * BK * BYTES_PER_HALF; // 2048
/// Sprint 6.7b: col-stride padded to 36 B (32 B data + 4 B pad) to break
/// bank conflicts on fragment-B reads. Pre-pad (stride 32 = 8 banks) had
/// `group_id·8 + tig mod 32` collapse to only 16 distinct banks across a
/// warp → 2-way conflict on every fragment-B access. Stride 36 (s=9) gives
/// `group_id·9 + tig mod 32` with most banks 1-way and only 3 banks 2-way
/// — a large net reduction in serialization. Fragment B loader already
/// accepts this stride via its `col_stride_bytes` parameter; no loader
/// code changed.
pub(crate) const TILE_B_COL_STRIDE_BYTES: u32 = BK * BYTES_PER_HALF + 4; // 32 + 4 pad = 36
/// Data region is `64 cols × 36 B = 2304 B`. The pre-zero cooperative pass
/// requires the allocation to be a multiple of `THREADS_PER_BLOCK * 4 =
/// 512 B` so we round up to the next multiple. The tail beyond the data
/// region (byte `2304..2560`) is never read by the kernel — it only
/// exists to satisfy the cooperative-zero divisibility requirement.
pub(crate) const TILE_B_BYTES: u32 = BN_BLOCK * TILE_B_COL_STRIDE_BYTES
    + (THREADS_PER_BLOCK * 4 - (BN_BLOCK * TILE_B_COL_STRIDE_BYTES) % (THREADS_PER_BLOCK * 4))
        % (THREADS_PER_BLOCK * 4);

/// Validate dimension constraints for [`matmul_tc`].
///
/// `pub(crate)` so `matmul_tc_async_kernel` can reuse — the dim
/// constraints are identical.
///
/// Sprint 6.7 Gate C: M and N may be any positive value (edge-tile
/// predication in the kernel handles non-multiple-of-64 cases). K
/// remains a multiple of 16 — the mma K dim is structural and the
/// kernel does not pad K within a K-tile.
pub(crate) fn validate_dims_tc(
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
    if !k.is_multiple_of(BK) {
        return Err(KaioError::InvalidConfig(format!(
            "matmul_tc: K must be a multiple of {BK} (got {k}). The mma.sync.m16n8k16 \
             instance shape requires K-tile size 16; K is not edge-padded inside a K-tile."
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

/// Multi-warp cooperative load of the 64×16 fp16 row-major A block tile
/// from global memory into row-major shared. 128 threads × 4 b32 issues
/// per thread = 512 b32 = 1024 fp16 = 64 rows × 16 cols ✓.
///
/// **Per-thread layout:** `lane_base = flat_tid * 4`, `flat = lane_base + i`,
/// `row_in_tile = flat / 8`, `col_pair = flat % 8`. All 4 issues per
/// thread share the same `row_in_tile`, so one row-bounds check gates
/// all 4 loads for that thread.
///
/// **Edge handling:** Caller pre-zeroes `tile_a` shared memory once at
/// kernel start. OOB threads (`row_global >= M`) skip their 4 ld+st
/// pairs entirely via `@!p bra A_SKIP_TILE_LOAD_<suffix>` — the shared
/// slots stay zero from the pre-zero pass. No K-direction edge check —
/// `K % 16 == 0` is enforced by validate.
///
/// `label_suffix` makes the bra-skip label unique per call site.
/// Multi-call kernels (e.g. matmul_tc_async's preamble + per-iter)
/// must pass distinct suffixes to avoid duplicate-label ptxas errors.
/// Single-call kernels (matmul_tc) can pass `""`.
pub(crate) fn emit_mw_load_tile_a_64x16(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    a_block_base_global: Register, // u64 — A[block_row, k_tile*16]
    tile_a_shared: Register,       // u32
    flat_tid: Register,            // u32 — 0..128
    block_row: Register,           // u32
    m: Register,                   // u32
    k_bytes: Register,             // u32 — K * 2
    label_suffix: &str,
) {
    let skip_label = if label_suffix.is_empty() {
        "A_SKIP_TILE_LOAD".to_string()
    } else {
        format!("A_SKIP_TILE_LOAD_{label_suffix}")
    };
    // lane_base = flat_tid * 4   (half2 index base)
    let lane_base = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: lane_base,
        lhs: Operand::Reg(flat_tid),
        rhs: Operand::ImmU32(4),
        ty: PtxType::U32,
    }));

    // row_in_tile = lane_base / 8   (per thread; same for all 4 issues)
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

    // p_row_in = row_global < M; if false, skip all 4 ld+st pairs.
    // The pre-zero pass at kernel start ensures the OOB shared slots
    // already contain zero, so skipping the loads is safe.
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
        // flat = lane_base + i
        let flat = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: flat,
            lhs: Operand::Reg(lane_base),
            rhs: Operand::ImmU32(i),
            ty: PtxType::U32,
        }));
        // col_pair = flat % 8
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

        // shared_off = row_in_tile * TILE_A_ROW_STRIDE_BYTES + col_pair_bytes
        let shared_off = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Mad {
            dst: shared_off,
            a: Operand::Reg(row_in_tile),
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

        // global_addr = a_block_base + row_in_tile * k_bytes + col_pair_bytes
        let row_off64 = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
            dst: row_off64,
            lhs: Operand::Reg(row_in_tile),
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
            lhs: Operand::Reg(a_block_base_global),
            rhs: Operand::Reg(per_thread_off64),
            ty: PtxType::U64,
        }));

        // ld.global.b32 + st.shared.b32 (no predication needed at this
        // level — the bra-pred above skipped the whole block on OOB).
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

/// Multi-warp cooperative load of the 16×64 fp16 B block tile from
/// row-major global into column-major shared. 128 threads × 8 fp16
/// issues per thread = 1024 fp16 = 16 rows × 64 cols ✓.
///
/// **Per-thread layout:** `lane_base = flat_tid * 8`, `flat = lane_base + i`
/// for `i ∈ 0..8`. `col = flat / 16` (constant per thread = `flat_tid / 2`),
/// `row = flat % 16` (varies 0..16 across the thread's 8 issues).
///
/// **Edge handling:** Caller pre-zeroes `tile_b` shared memory once at
/// kernel start. OOB threads (`col_global >= N`) skip their 8 ld+st
/// pairs entirely via `@!p bra B_SKIP_TILE_LOAD_<suffix>` — the shared
/// slots stay zero from the pre-zero pass. No K-direction edge check —
/// `K % 16 == 0` is enforced by validate.
///
/// `label_suffix` makes the bra-skip label unique per call site; same
/// rules as `emit_mw_load_tile_a_64x16`.
pub(crate) fn emit_mw_load_tile_b_16x64(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    b_block_base_global: Register, // u64 — B[k_tile*16, block_col]
    tile_b_shared: Register,       // u32
    flat_tid: Register,            // u32 — 0..128
    block_col: Register,           // u32
    n: Register,                   // u32
    n_bytes: Register,             // u32 — N * 2
    label_suffix: &str,
) {
    let skip_label = if label_suffix.is_empty() {
        "B_SKIP_TILE_LOAD".to_string()
    } else {
        format!("B_SKIP_TILE_LOAD_{label_suffix}")
    };
    // lane_base = flat_tid * 8   (fp16 index base)
    let lane_base = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: lane_base,
        lhs: Operand::Reg(flat_tid),
        rhs: Operand::ImmU32(8),
        ty: PtxType::U32,
    }));

    // col_in_tile = lane_base / 16   (constant per thread; all 8 issues share)
    let col_in_tile = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: col_in_tile,
        lhs: Operand::Reg(lane_base),
        rhs: Operand::ImmU32(16),
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

    // p_col_in = col_global < N; if false, skip all 8 ld+st pairs.
    // The pre-zero pass at kernel start ensures the OOB shared slots
    // already contain zero, so skipping the loads is safe.
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

    // Pre-compute the per-thread shared col base offset (col_in_tile * 32).
    let col_shared_base = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: col_shared_base,
        lhs: Operand::Reg(col_in_tile),
        rhs: Operand::ImmU32(TILE_B_COL_STRIDE_BYTES),
        ty: PtxType::U32,
    }));

    // Pre-compute the per-thread global col offset bytes (col_in_tile * 2).
    let col_global_bytes = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: col_global_bytes,
        lhs: Operand::Reg(col_in_tile),
        rhs: Operand::ImmU32(BYTES_PER_HALF),
        ty: PtxType::U32,
    }));
    let col_global_bytes64 = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Cvt {
        dst: col_global_bytes64,
        src: col_global_bytes,
        dst_ty: PtxType::U64,
        src_ty: PtxType::U32,
    });

    for i in 0..8u32 {
        // flat = lane_base + i; row = flat % 16
        let flat = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: flat,
            lhs: Operand::Reg(lane_base),
            rhs: Operand::ImmU32(i),
            ty: PtxType::U32,
        }));
        let row = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Rem {
            dst: row,
            lhs: Operand::Reg(flat),
            rhs: Operand::ImmU32(16),
            ty: PtxType::U32,
        }));
        let row_bytes = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Mul {
            dst: row_bytes,
            lhs: Operand::Reg(row),
            rhs: Operand::ImmU32(BYTES_PER_HALF),
            ty: PtxType::U32,
        }));

        // shared_addr = tile_b_shared + col_shared_base + row_bytes
        let shared_off = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: shared_off,
            lhs: Operand::Reg(col_shared_base),
            rhs: Operand::Reg(row_bytes),
            ty: PtxType::U32,
        }));
        let shared_addr = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: shared_addr,
            lhs: Operand::Reg(tile_b_shared),
            rhs: Operand::Reg(shared_off),
            ty: PtxType::U32,
        }));

        // global_addr = b_block_base + row * n_bytes + col_global_bytes
        let row_global_off64 = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
            dst: row_global_off64,
            lhs: Operand::Reg(row),
            rhs: Operand::Reg(n_bytes),
            src_ty: PtxType::U32,
        }));
        let per_thread_off64 = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: per_thread_off64,
            lhs: Operand::Reg(row_global_off64),
            rhs: Operand::Reg(col_global_bytes64),
            ty: PtxType::U64,
        }));
        let global_addr = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: global_addr,
            lhs: Operand::Reg(b_block_base_global),
            rhs: Operand::Reg(per_thread_off64),
            ty: PtxType::U64,
        }));

        // ld.global.f16 + st.shared.f16 (no per-issue predication —
        // the bra-pred above skipped the whole block on OOB).
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
    kernel.push(PtxInstruction::Label(skip_label));
}

/// Per-warp accumulation of one K-tile: 8 mma.sync.m16n8k16 calls in a
/// 2 (m_stripe) × 4 (n_stripe) grid. Each mma covers a 16×8 sub-tile of
/// the warp's 32×32 quadrant. Loads 2 distinct A fragments (one per
/// m_stripe) and 4 distinct B fragments (one per n_stripe), then issues
/// 8 mma's reusing them.
///
/// The accumulator slice `&mut [FragmentC; 8]` is indexed as
/// `accs[m_stripe * MMAS_PER_WARP_N + n_stripe]`.
pub(crate) fn emit_warp_quadrant_mma(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    tile_a_warp_base_shared: Register, // u32 — tile_a + warp_row_quad * 32 * TILE_A_ROW_STRIDE_BYTES
    tile_b_warp_base_shared: Register, // u32 — tile_b + warp_col_quad * 32 * TILE_B_COL_STRIDE_BYTES
    warp_lane: Register,               // u32 — tid_x ∈ [0, 32)
    warp_group_tig: (Register, Register), // Sprint 6.7b D10 hoist: (group_id, tig) for this warp lane
    accs: &mut [FragmentC; 8],
) {
    // Load 2 FragmentA_F16's, one per m_stripe.
    let mut frags_a: [Option<FragmentA_F16>; MMAS_PER_WARP_M as usize] = [None, None];
    for m_stripe in 0..MMAS_PER_WARP_M {
        let row_off_bytes = m_stripe * BM * TILE_A_ROW_STRIDE_BYTES; // m_stripe * 16 * 32
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
        let frag = load_fragment_a_m16n8k16_shared_row(
            alloc,
            kernel,
            a_stripe_base,
            warp_lane,
            TILE_A_ROW_STRIDE_BYTES,
            Some(warp_group_tig),
        );
        frags_a[m_stripe as usize] = Some(frag);
    }

    // Load 4 FragmentB_F16's, one per n_stripe.
    let mut frags_b: [Option<FragmentB_F16>; MMAS_PER_WARP_N as usize] = [None, None, None, None];
    for n_stripe in 0..MMAS_PER_WARP_N {
        let col_off_bytes = n_stripe * BN * TILE_B_COL_STRIDE_BYTES; // n_stripe * 8 * padded_col_stride
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
        let frag = load_fragment_b_m16n8k16_shared_col(
            alloc,
            kernel,
            b_stripe_base,
            warp_lane,
            TILE_B_COL_STRIDE_BYTES,
            Some(warp_group_tig),
        );
        frags_b[n_stripe as usize] = Some(frag);
    }

    // 8 mma's: D[m,n] = A[m] * B[n] + D[m,n]
    for m_stripe in 0..MMAS_PER_WARP_M {
        let frag_a = frags_a[m_stripe as usize].unwrap();
        for n_stripe in 0..MMAS_PER_WARP_N {
            let frag_b = frags_b[n_stripe as usize].unwrap();
            let acc_idx = (m_stripe * MMAS_PER_WARP_N + n_stripe) as usize;
            let frag_d = accs[acc_idx];
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
            // mma.sync writes back to the same fragment; nothing to update.
            accs[acc_idx] = frag_d;
        }
    }
}

/// Per-warp output store: write 8 FragmentC's (32 × 4 fp32 per warp) to
/// global D matrix at the warp's quadrant position. Each store uses
/// `setp.lt.and.u32` to combine the row check (`row_global < M`) with
/// the col check (`col_global < N`) into one predicate, then
/// `@p st.global.f32 [addr], reg`. OOB stores skip entirely.
///
/// Per-fragment: 4 stores at positions
///   d[0]: (row_start + groupID,     col_start + 2*tig    )
///   d[1]: (row_start + groupID,     col_start + 2*tig + 1)
///   d[2]: (row_start + groupID + 8, col_start + 2*tig    )
///   d[3]: (row_start + groupID + 8, col_start + 2*tig + 1)
/// where groupID = lane / 4, tig = lane % 4.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_warp_quadrant_store(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    rd_d_block_base: Register, // u64 — D[block_row, block_col]
    accs: &[FragmentC; 8],
    warp_lane: Register,          // u32 — tid_x ∈ [0, 32)
    warp_quadrant_row_start: u32, // 0 or 32 (warp_row_quad * 32)
    warp_quadrant_col_start: u32, // 0 or 32 (warp_col_quad * 32)
    block_row: Register,          // u32
    block_col: Register,          // u32
    m: Register,                  // u32
    n: Register,                  // u32
    n_f32_stride: Register,       // u32 — N * 4
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

    // Per-thread row and col base contributions (constants per warp;
    // depends only on warp_lane and the constants warp_quadrant_*_start).
    //   row_g_base = block_row + warp_quadrant_row_start + groupID
    //   row_g_base_p8 = row_g_base + 8
    //   col_t_base = block_col + warp_quadrant_col_start + 2 * tig
    //   col_t_base_p1 = col_t_base + 1
    // For each (m_stripe, n_stripe), add 16*m_stripe to rows and 8*n_stripe to cols.
    let row_g_base = alloc.alloc(PtxType::U32);
    {
        // tmp1 = block_row + warp_quadrant_row_start
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
        // tmp2 = block_col + warp_quadrant_col_start
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
        // col_t_base = tmp2 + 2 * tig
        kernel.push(PtxInstruction::Arith(ArithOp::Mad {
            dst: col_t_base,
            a: Operand::Reg(tig),
            b: Operand::ImmU32(2),
            c: Operand::Reg(tmp2),
            ty: PtxType::U32,
            mode: MadMode::Lo,
        }));
    }

    // For each (m_stripe, n_stripe): emit 4 stores.
    for m_stripe in 0..MMAS_PER_WARP_M {
        // row_stripe_g = row_g_base + 16 * m_stripe
        // row_stripe_g_p8 = row_stripe_g + 8
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

        // p_row_a = row_stripe_g    < M
        // p_row_b = row_stripe_g_p8 < M
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

            // col_stripe_t = col_t_base + 8 * n_stripe
            // col_stripe_t_p1 = col_stripe_t + 1
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

            // Compute per-store predicates: combine row + col via setp.lt.and.u32
            //   p0 = (col_stripe_t    < N) AND p_row_a
            //   p1 = (col_stripe_t_p1 < N) AND p_row_a
            //   p2 = (col_stripe_t    < N) AND p_row_b
            //   p3 = (col_stripe_t_p1 < N) AND p_row_b
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

            // Compute the 4 store addresses.
            //   addr0 = base + row_stripe_g    * (N*4) + col_stripe_t    * 4
            //   addr1 = base + row_stripe_g    * (N*4) + col_stripe_t_p1 * 4   (= addr0 + 4)
            //   addr2 = base + row_stripe_g_p8 * (N*4) + col_stripe_t    * 4
            //   addr3 = base + row_stripe_g_p8 * (N*4) + col_stripe_t_p1 * 4   (= addr2 + 4)

            // row_offset_a = row_stripe_g * n_f32_stride (u32, fits since N <= ~1e6 in practice)
            // Use mul.wide.u32 to get u64 directly.
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

            // base_a = rd_d_block_base + row_off_a64 + col_t_bytes64
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

            // base_b = rd_d_block_base + row_off_b64 + col_t_bytes64
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

            // Predicated stores.
            kernel.push(PtxInstruction::Memory(MemoryOp::StGlobalPred {
                addr: addr0,
                src: frag.regs[0],
                ty: PtxType::F32,
                pred: p0,
                negate: false,
            }));
            kernel.push(PtxInstruction::Memory(MemoryOp::StGlobalPred {
                addr: addr1,
                src: frag.regs[1],
                ty: PtxType::F32,
                pred: p1,
                negate: false,
            }));
            kernel.push(PtxInstruction::Memory(MemoryOp::StGlobalPred {
                addr: addr2,
                src: frag.regs[2],
                ty: PtxType::F32,
                pred: p2,
                negate: false,
            }));
            kernel.push(PtxInstruction::Memory(MemoryOp::StGlobalPred {
                addr: addr3,
                src: frag.regs[3],
                ty: PtxType::F32,
                pred: p3,
                negate: false,
            }));
        }
    }
}

/// Cooperatively zero `tile_a` and `tile_b` shared regions across 128
/// threads, then bar.sync. Each thread writes a contiguous slice of
/// each tile via `st.shared.b32` of an immediate-zero register.
///
/// Per-tile size must be a multiple of `128 * 4 = 512` bytes (so each
/// thread does an integer number of `b32` stores). Sprint 6.7 sync tile
/// is 2048 B (128 × 16 → 4 stores/thread); async tile is 4096 B
/// (128 × 32 → 8 stores/thread).
///
/// `pub(crate)` — `matmul_tc_async_kernel` reuses this for its 4 KB
/// double-buffered tiles.
pub(crate) fn emit_pre_zero_shared_tiles(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    tile_a: Register,
    tile_b: Register,
    flat_tid: Register,
    tile_a_bytes: u32,
    tile_b_bytes: u32,
) {
    debug_assert!(
        tile_a_bytes.is_multiple_of(THREADS_PER_BLOCK * 4),
        "tile_a_bytes ({tile_a_bytes}) must be multiple of 128*4 (THREADS_PER_BLOCK * 4)"
    );
    debug_assert!(
        tile_b_bytes.is_multiple_of(THREADS_PER_BLOCK * 4),
        "tile_b_bytes ({tile_b_bytes}) must be multiple of 128*4"
    );

    let r_zero = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_zero,
        src: Operand::ImmU32(0),
        ty: PtxType::U32,
    });

    for (tile_base, total_bytes) in [(tile_a, tile_a_bytes), (tile_b, tile_b_bytes)] {
        let bytes_per_thread = total_bytes / THREADS_PER_BLOCK;
        let issues_per_thread = bytes_per_thread / 4;

        // base_off = tile_base + flat_tid * bytes_per_thread
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

/// Build the IR module for `matmul_tc` targeting the given SM.
///
/// `sm` is a PTX target string such as `"sm_89"` — the caller is
/// responsible for deriving it from `device.info()`. Passing a sub-Ampere
/// target (e.g. `sm_70`) is legal at build time; `PtxModule::validate()`
/// inside `KaioDevice::load_module` then rejects the module cleanly with
/// `ValidationError::SmTooLow`.
pub(crate) fn build_matmul_tc_module(sm: &str) -> PtxModule {
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
    // tid_x ∈ [0, 32) — lane within warp
    let (r_tid_x, tid_instr) = special::tid_x(&mut alloc);
    kernel.push(tid_instr);
    // tid_y ∈ [0, 4) — warp_id
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
    // kernel start and reuse across every emit_warp_quadrant_mma call. Saves
    // 6 × div/rem pairs per K-iter that the fragment loaders would otherwise
    // recompute internally (2 FragmentA_F16 + 4 FragmentB_F16 = 6 calls per K-iter).
    // group_id = tid_x / 4, tig = tid_x % 4.
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

    // K and N in bytes (for global addressing within tile loads).
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

    // a_block_row_base = rd_a + block_row * K * 2   (u64)
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

    // b_block_col_base = rd_b + block_col * 2   (u64)
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

    // Per-warp shared bases (warp_row_quad = warp_id / 2; warp_col_quad = warp_id % 2).
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

    // tile_a_warp_base = tile_a + warp_row_quad * (32 * TILE_A_ROW_STRIDE_BYTES)
    //                  = tile_a + warp_row_quad * 1024
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

    // tile_b_warp_base = tile_b + warp_col_quad * (32 * TILE_B_COL_STRIDE_BYTES)
    //                  = tile_b + warp_col_quad * 1024
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

    // Pre-zero shared tiles. tile_a + tile_b = 2048 + 2048 = 4096 B.
    emit_pre_zero_shared_tiles(
        &mut alloc,
        &mut kernel,
        r_tile_a,
        r_tile_b,
        r_flat_tid,
        TILE_A_BYTES,
        TILE_B_BYTES,
    );

    // Allocate 8 FragmentC accumulators, zero them.
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

    // num_k_tiles = K / 16
    let r_num_k_tiles = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: r_num_k_tiles,
        lhs: Operand::Reg(r_k),
        rhs: Operand::ImmU32(BK),
        ty: PtxType::U32,
    }));

    // K loop counter.
    let r_k_tile = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_k_tile,
        src: Operand::ImmU32(0),
        ty: PtxType::U32,
    });

    kernel.push(PtxInstruction::Label("K_LOOP".to_string()));

    // A tile global source = a_block_base + k_tile * 16 * 2 = a_block_base + k_tile * 32
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

    // B tile global source = b_block_base + k_tile * 16 * N * 2
    let r_k_tile_x_bk = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_k_tile_x_bk,
        lhs: Operand::Reg(r_k_tile),
        rhs: Operand::ImmU32(BK), // 16
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

    // Cooperative load of A and B tiles with edge predication.
    // Single call site each → empty label_suffix.
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

    // bar.sync — all warps see consistent shared state.
    kernel.push(PtxInstruction::Control(ControlOp::BarSync {
        barrier_id: 0,
    }));

    // Per-warp 8-mma accumulation.
    emit_warp_quadrant_mma(
        &mut alloc,
        &mut kernel,
        r_tile_a_warp,
        r_tile_b_warp,
        r_tid_x,
        (r_hoisted_group_id, r_hoisted_tig),
        &mut accs,
    );

    // bar.sync — fence before next K iteration overwrites shared.
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

    // Per-warp output store with edge predication. The warp's quadrant
    // start is determined at runtime by warp_id, so we compute
    // warp_quadrant_row_start = warp_row_quad * 32 inline rather than
    // passing a constant. For Gate A simplicity we do this via 4
    // per-warp store invocations (one per (warp_row_quad, warp_col_quad)
    // ∈ {0,1}²) gated by branches — wait, that's worse than passing
    // runtime warp_quadrant_*_start. Let's pass runtime values.
    //
    // Compute runtime warp_quadrant_row_start = warp_row_quad * WARP_QUAD_M
    //   and warp_quadrant_col_start = warp_col_quad * WARP_QUAD_N.
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
    // Replace block_row / block_col with the per-warp absolute row/col base
    // by adding wq_row_start / wq_col_start once.
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
    // Pass rd_d (not rd_d_block_base) — block_row/block_col are
    // already folded into r_warp_block_row/r_warp_block_col, so the
    // store helper computes absolute global addresses from rd_d.
    emit_warp_quadrant_store(
        &mut alloc,
        &mut kernel,
        rd_d,
        &accs,
        r_tid_x,
        0, // warp_quadrant_row_start: already folded into r_warp_block_row
        0, // warp_quadrant_col_start: already folded into r_warp_block_col
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

/// Tensor-core matmul kernel — f16 × f16 → f32 with fp32 accumulation.
///
/// Sprint 6.7 multi-warp restructure: 64×64 block tile, 4 warps in 2×2,
/// 32×32 quadrant per warp via 8 × `mma.sync.m16n8k16` per K-iteration.
/// Edge-tile predication on M and N — `M`/`N` need not be multiples of
/// 64 (only the mma shape `M%16 = N%8 = K%16 = 0` is enforced).
///
/// # Hardware
///
/// Requires NVIDIA Ampere or newer (SM 8.0+). Sub-Ampere targets are
/// rejected by `PtxModule::validate()` via `ValidationError::SmTooLow`
/// before driver dispatch.
///
/// # Layout
///
/// A is M×K row-major, B is K×N row-major, D is M×N row-major. B is
/// transposed on the way into shared memory (column-major) so the B
/// fragment loader can use single-half2 loads per K stripe.
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

    let info = device.info()?;
    let (major, minor) = info.compute_capability;
    let sm = format!("sm_{major}{minor}");
    let module = build_matmul_tc_module(&sm);

    let kmodule = device.load_module(&module)?;
    let func = kmodule.function("matmul_tc")?;

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

    // --- Host-only validate_dims_tc tests (Gate C: M, N can be any
    // positive value; only K is constrained to a multiple of 16) ---
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
        if !k.is_multiple_of(BK) {
            return Err(KaioError::InvalidConfig(format!(
                "matmul_tc: K must be a multiple of {BK} (got {k}). The mma.sync.m16n8k16 \
                 instance shape requires K-tile size 16; K is not edge-padded inside a K-tile."
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

    /// Gate C: M no longer needs to be a multiple of 16 — edge-tile
    /// predication handles ragged M. Validate accepts non-divisible M.
    #[test]
    fn validate_dims_accepts_non_divisible_m() {
        // 17 rows: edge-tile path inside the 64×64 block tile handles
        // the OOB rows. Buffer sizing follows the full M×K shape.
        assert!(validate_dims_raw(17, 8, 16, 17 * 16, 16 * 8, 17 * 8).is_ok());
        assert!(validate_dims_raw(7, 5, 16, 7 * 16, 16 * 5, 7 * 5).is_ok());
        assert!(validate_dims_raw(1023, 1023, 1024, 1023 * 1024, 1024 * 1023, 1023 * 1023).is_ok());
    }

    /// Gate C: N no longer needs to be a multiple of 8 — same reason.
    #[test]
    fn validate_dims_accepts_non_divisible_n() {
        assert!(validate_dims_raw(16, 5, 16, 16 * 16, 16 * 5, 16 * 5).is_ok());
        assert!(validate_dims_raw(64, 9, 16, 64 * 16, 16 * 9, 64 * 9).is_ok());
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
        let err = validate_dims_raw(16, 8, 16, 100, 1000, 1000).unwrap_err();
        assert!(
            matches!(err, KaioError::InvalidConfig(ref m) if m.contains("A buffer too small")),
            "got: {err:?}"
        );
    }

    #[test]
    fn validate_dims_accepts_valid_shapes() {
        // Original divisible shapes still pass.
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
    fn build_matmul_tc_module_produces_valid_structure() {
        // Sprint 6.7 multi-warp: 64×64 block tile, 4 warps, 8 mma per warp
        // per K-tile (2 m_stripes × 4 n_stripes). Tile A = 2048 B (64×16
        // fp16, row-major). Tile B = 2560 B per Sprint 6.7b (64×36 data
        // with col-stride padding for bank-conflict relief, rounded up to
        // next multiple of THREADS_PER_BLOCK*4 for cooperative-zero).
        let module = build_matmul_tc_module("sm_89");
        let ptx = emit_module_to_string(&module);

        assert!(ptx.contains(".visible .entry matmul_tc("));
        assert!(
            ptx.contains(".shared .align 4 .b8 tile_a[2048]"),
            "tile_a should be 2048 B (64×16 fp16)"
        );
        assert!(
            ptx.contains(".shared .align 4 .b8 tile_b[2560]"),
            "tile_b should be 2560 B (Sprint 6.7b: 64 cols × 36 B padded + round-up tail)"
        );
        assert!(ptx.contains("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"));
        assert!(ptx.contains("bar.sync"));

        let mma_count = ptx.matches("mma.sync.aligned.m16n8k16").count();
        assert_eq!(
            mma_count, 8,
            "expected 8 mma.sync emits per K-iter (2 m_stripes × 4 n_stripes per warp)"
        );

        // Edge-tile predication uses setp.lt.and.u32 and predicated stores.
        assert!(
            ptx.contains("setp.lt.and.u32"),
            "edge-tile combined predicate emit missing"
        );
        let pred_store_count = ptx.matches("st.global.f32").count();
        // 32 predicated stores per warp output (8 fragments × 4 stores).
        assert_eq!(
            pred_store_count, 32,
            "expected 32 predicated st.global.f32 per warp output (8 fragments × 4 stores)"
        );
    }

    #[test]
    fn build_matmul_tc_module_declares_requested_sm_target() {
        let module_70 = build_matmul_tc_module("sm_70");
        let ptx_70 = emit_module_to_string(&module_70);
        assert!(
            ptx_70.contains(".target sm_70"),
            "sm_70 should round-trip verbatim (no flooring)"
        );

        let module_89 = build_matmul_tc_module("sm_89");
        let ptx_89 = emit_module_to_string(&module_89);
        assert!(ptx_89.contains(".target sm_89"));
    }

    #[test]
    fn matmul_tc_module_rejects_sm_70_via_validate() {
        use kaio_core::ir::ValidationError;

        let module = build_matmul_tc_module("sm_70");
        let err = module
            .validate()
            .expect_err("matmul_tc module at sm_70 must fail validation");
        match err {
            ValidationError::SmTooLow {
                required,
                actual,
                feature,
            } => {
                assert_eq!(required, 80, "mma.sync.m16n8k16 requires sm_80");
                assert_eq!(actual, 70);
                assert!(
                    feature.contains("mma.sync"),
                    "feature string should name mma.sync; got: {feature}"
                );
            }
            other => panic!("expected SmTooLow, got {other:?}"),
        }
    }

    #[test]
    fn matmul_tc_module_validates_at_sm_80_and_above() {
        for sm in ["sm_80", "sm_89", "sm_90"] {
            let module = build_matmul_tc_module(sm);
            module
                .validate()
                .unwrap_or_else(|e| panic!("{sm} should validate; got error: {e}"));
        }
    }
}
