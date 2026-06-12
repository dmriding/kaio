//! Fused tri-output INT8 QKV projection — W8A16 (f16 activations, INT8
//! weights, scalar per-projection scales).
//!
//! Sprint 7.3 MVS. Single kernel launch produces three f16 outputs
//! (Q, K, V) ready to feed [`attention_tc`][crate::attention_tc] — saves
//! 2× global reads of the shared activation X compared to three
//! separate [`matmul_int8`][crate::matmul_int8][crate::matmul_int8] calls, and amortizes
//! kernel launch overhead (dominant at autoregressive-decode batch
//! sizes).
//!
//! # W8A16 vs W8A8
//!
//! [`matmul_int8`][crate::matmul_int8][crate::matmul_int8] (Sprint 7.1) is **W8A8** (`i8`
//! activations × `i8` weights, native `mma.sync.m16n8k32.s8.s8.s32`).
//! `qkv_project_int8` is **W8A16** (`f16` activations × `i8` weights,
//! `mma.sync.m16n8k16.f16.f16.f32` after per-weight `cvt.rn.f16.s8`).
//! The W8A16 contract lets the op drop in between f16-boundary layers
//! of a real LLM without forcing the caller to quantize X at every
//! attention block. Users who genuinely need W8A8 call `matmul_int8`
//! three times and manage their own activation quantization.
//!
//! # Unified mma path with INT4
//!
//! Both `qkv_project_int8` and [`qkv_project_int4`][super::qkv_project_int4_kernel]
//! target the same `mma.sync.m16n8k16.f16.f16.f32` shape with
//! `K_TILE_SHARED = 16`. Fragment-C layout and the store-out helper
//! ([`emit_store_fragment_c_f32_to_f16_packed`][super::store_out::emit_store_fragment_c_f32_to_f16_packed]
//! once wired) are shared — only fragment-B dequant differs
//! (INT8: `cvt.rn.f16.s8` + scalar scale; INT4: nibble-extract +
//! sign-extend + group-scale fold).
//!
//! # Tri-output design (Design S+½P — Sprint 7.3.5)
//!
//! Sprint 7.3 originally shipped Design S (serial fusion, single W
//! slot reused per projection, 7 `bar.sync` per K-tile). Sprint 7.3.5
//! rewrote the inner K-loop to **Design S+½P**: two `tile_w` shared
//! slots with a ping-pong index, overlapping the cooperative load of
//! `W_{P+1}` with the mma compute of `W_P`. Barrier cadence drops
//! from 7 to **4 per K-tile**.
//!
//! Per K-tile: cooperative-load `tile_x` once into shared (first
//! K-tile only — subsequent K-tiles overlap it with Epoch 3's
//! mma_V); hoist `frag_A` from `tile_x` to registers; pre-compute
//! `frag_A` once per K-tile and reuse it across all three projection
//! mma epochs (`tile_x` overwrite-after-hoist invariant); then for
//! each projection P ∈ {Q, K, V} in turn: load `W_{P+1}` into the
//! ping-pong NEXT slot while mma_P runs on the CURRENT slot, with
//! a single `bar.sync` between epochs. Three fragment-C banks
//! persist across the entire K-loop; scalar per-projection scales
//! apply at store-out, not pre-mma.
//!
//! # Register budget
//!
//! Sprint 7.3 D3.4 Rollback #1 dropped `MMAS_PER_WARP_N` from 2 to 1,
//! halving per-warp fragment-C live state (3 grids × 8 = 24 f32 regs
//! per lane, vs 48 at the original 64×32 tile). Per-block output
//! tile is 64×16 instead of 64×32.
//!
//! Sprint 7.3.5 Design S+½P adds ping-pong slot pointer tracking,
//! hoisted `frag_A` registers, and predicated overlap address math.
//! Measured `ptxas_verify` (sm_80 + sm_89, 0 spills both): sm_80 lands
//! at **64 regs** (at the full-occupancy cliff exactly — natural
//! allocation is 66 regs, ptxas compresses 2 under without spilling);
//! sm_89 lands at **56 regs** with 8 reg headroom. D1 register-budget
//! skeleton checkpoint models cross-iteration live ranges and clears
//! the design ahead of full kernel body work.

use kaio::prelude::*;
use kaio_core::instr::ArithOp;
use kaio_core::instr::control::{CmpOp, ControlOp};
use kaio_core::instr::memory::MemoryOp;
use kaio_core::ir::{Operand, PtxInstruction, PtxKernel, Register, RegisterAllocator};
use kaio_core::types::PtxType;

// --- mma.sync.m16n8k16 instance shape (f16 inputs, f32 accumulator) ---
// Matches the shape used by matmul_int4 and matmul_tc. Both QKV projection
// variants (INT8 and INT4) share this mma after the W8A16 switch.
#[allow(dead_code)] // wired up in D3
pub(crate) const BM: u32 = 16; // mma m dim
#[allow(dead_code)] // wired up in D3
pub(crate) const BN: u32 = 8; // mma n dim

// --- Multi-warp block tiling (halved N vs matmul_int4/int8 + Rollback #1) ---
//
// Sprint 7.3 D3.4 ptxas_verify reported 80 regs/thread (sm_80 + sm_89, 0 spills)
// for the 64×32 output tile with `MMAS_PER_WARP_N = 2`. Plan-authorized
// Rollback #1 fires: drop `MMAS_PER_WARP_N` from 2 to 1, halving per-warp
// fragment-C live state from 16 to 8 f32 regs (3 grids × 8 = 24 frag_c regs
// per lane vs. the prior 48). Per-block output tile becomes 64×16 instead of
// 64×32. More blocks launch but each block is lighter; X-reuse economics
// still favor the fused tri-output design over 3× standalone matmul_int8.
//
// `TILE_W_BYTES` is kept at 512 (16 K-rows × 32 N-cols, padded) so the
// pre-zero helper retains its 4-bytes-per-thread cooperative pattern; the
// W cooperative loader gates writes on the in-tile column predicate so
// only the first `BN_BLOCK = 16` cols receive real data. The remaining
// 16 cols stay pre-zeroed and are never read by fragment-B (which
// indexes col_abs < `BN_BLOCK`).
#[allow(dead_code)] // wired up in D3
pub(crate) const BM_BLOCK: u32 = 64; // output rows per block
#[allow(dead_code)] // wired up in D3
pub(crate) const BN_BLOCK: u32 = 16; // output cols per block (Rollback #1: 32 → 16)
#[allow(dead_code)] // wired up in D3
pub(crate) const WARP_QUAD_M: u32 = 32; // rows per warp quadrant
#[allow(dead_code)] // wired up in D3
pub(crate) const WARP_QUAD_N: u32 = 8; // cols per warp quadrant (Rollback #1: 16 → 8)
#[allow(dead_code)] // wired up in D3
pub(crate) const MMAS_PER_WARP_M: u32 = WARP_QUAD_M / BM; // 2
#[allow(dead_code)] // wired up in D3
pub(crate) const MMAS_PER_WARP_N: u32 = WARP_QUAD_N / BN; // 1 (Rollback #1: 2 → 1)
#[allow(dead_code)] // wired up in D3
pub(crate) const WARPS_PER_BLOCK: u32 = 4;
#[allow(dead_code)] // wired up in D3
pub(crate) const THREADS_PER_BLOCK: u32 = WARPS_PER_BLOCK * 32; // 128

// --- K-tile granularity (unified with matmul_int4 after W8A16 switch) ---
#[allow(dead_code)] // wired up in D3
pub(crate) const K_TILE_SHARED: u32 = 16;

// --- Shared tile byte sizes (D3.1) ---

/// Bytes per element for f16 activations and f16 dequanted B.
const BYTES_PER_F16: u32 = 2;

/// X tile in shared: `BM_BLOCK` rows × `K_TILE_SHARED` cols, f16 row-major.
/// 64 × 16 × 2 = 2048 B. Matches `matmul_int4_kernel::TILE_A_BYTES` so the
/// cooperative A-loader can be reused unchanged.
#[allow(dead_code)] // wired up in D3
pub(crate) const TILE_X_BYTES: u32 = BM_BLOCK * K_TILE_SHARED * BYTES_PER_F16; // 2048

/// X tile row stride in bytes (row-major f16, one b32 per two f16).
#[allow(dead_code)] // wired up in D3
pub(crate) const TILE_X_ROW_STRIDE_BYTES: u32 = K_TILE_SHARED * BYTES_PER_F16; // 32

/// W tile shared-memory column stride in bytes — fixed at **32** independent
/// of `BN_BLOCK`. After Sprint 7.3 Rollback #1 (`BN_BLOCK` shrunk 32 → 16),
/// the W slot is intentionally over-provisioned so the cooperative pre-zero
/// pattern (4 B per thread × 128 threads = 512 B) keeps its single-issue
/// shape. The W loader writes only cols `[0, BN_BLOCK)` per row; the
/// remaining cols stay pre-zeroed and are never read (fragment-B per-lane
/// indexing constrains `col_abs < BN_BLOCK`).
#[allow(dead_code)] // wired up in D3
pub(crate) const TILE_W_ROW_STRIDE_BYTES: u32 = 32;

/// W_P_i8 tile in shared: `K_TILE_SHARED` rows × `TILE_W_ROW_STRIDE_BYTES` cols, i8 row-major.
/// 16 × 32 × 1 = 512 B. See [`TILE_W_ROW_STRIDE_BYTES`] for the padding
/// rationale.
#[allow(dead_code)] // wired up in D3
pub(crate) const TILE_W_BYTES: u32 = K_TILE_SHARED * TILE_W_ROW_STRIDE_BYTES; // 512

// ── Sprint 7.3.5 S+½P bank-phase padding (Design invariant #4, R3-1) ──
//
// Two `tile_w` slots at 512 B each would place `slot1` at a 512 B offset
// from `slot0`. Since 512 is a multiple of 128 (one SMEM bank-line span),
// slot1's bank mapping is phase-aligned with slot0's — concurrent LDSM
// on slot0 and STS on slot1 during the overlap window can contend on the
// same bank ports and serialize, silently eroding the ILP gain from
// dropping barriers.
//
// The 64 B padding shifts slot1's bank mapping by 16 banks (maximum
// dispersal against the 32-bank file). 576 B = 512 + 64 is non-multiple
// of 128, satisfying invariant #4.
//
// Total Sprint 7.3.5 tile_w region: 512 + 64 + 512 = 1088 B. Shared
// budget post-S+½P: 2048 (tile_x) + 1088 = 3136 B, well under
// sm_89's 100 KB/SM limit.
#[allow(dead_code)]
pub(crate) const TILE_W_SLOT_PAD_BYTES: u32 = 64;
#[allow(dead_code)]
pub(crate) const TILE_W_SLOT_STRIDE_BYTES: u32 = TILE_W_BYTES + TILE_W_SLOT_PAD_BYTES; // 576
#[allow(dead_code)]
pub(crate) const TILE_W_SP_HALF_P_BYTES: u32 = TILE_W_BYTES + TILE_W_SLOT_PAD_BYTES + TILE_W_BYTES; // 1088

/// Pre-zero issue size — cooperative pre-zero writes one b32 per thread per
/// issue. Both tiles are required to be a multiple of `THREADS_PER_BLOCK * 4`
/// = 512 B. `TILE_X_BYTES = 2048 B` = 4 issues per thread; `TILE_W_BYTES =
/// 512 B` = 1 issue per thread. Asserted in debug builds by the pre-zero
/// helper.
#[allow(dead_code)]
const PRE_ZERO_BYTES_PER_ISSUE: u32 = THREADS_PER_BLOCK * 4; // 512

/// Cooperative load of the 64×16 f16 X block tile from global (row-major)
/// into shared (row-major). Delegates to the matmul_int4 loader because the
/// tile shape, f16 dtype, and row-major convention are identical; matmul_int4
/// is the canonical owner of the 64×16 f16 cooperative-load pattern.
///
/// The single `k_bytes` parameter is the K-dimension stride in bytes
/// (`K * 2` for f16) so the loader can compute per-row global offsets for
/// `M`-edge-tile predication without knowing the raw `K` value.
///
/// `label_suffix` differentiates the skip-label in kernels that call this
/// helper more than once; pass `""` for the default `A_SKIP_I4_TILE_LOAD`.
#[allow(dead_code)] // wired up in D3.3 / D3.4
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

/// Cooperative load of the 16×32 i8 W_P block tile from global (row-major)
/// into shared (row-major).
///
/// # Tile shape
/// - 16 K-rows × 32 N-cols = 512 bytes per tile per projection.
/// - 128 threads × 1 `ld.global.u32` issue = 512 bytes → exact single-issue fit.
///
/// # Per-thread layout
///
/// ```text
/// row_in_tile   = flat_tid / 8      // 0..16
/// col_group     = flat_tid % 8      // 0..8
/// col_start     = col_group * 4     // 0,4,8,...,28 (byte offset within row)
/// ```
///
/// Each thread reads 4 consecutive N-bytes from global at
/// `W_block[row_in_tile, col_start .. col_start+4]` and writes them to
/// shared at the identical `(row_in_tile, col_start)` offset — no
/// transpose. Fragment-B dequant (D3.2) does the per-lane `(K, N)`
/// address math when it reads the shared tile.
///
/// # Edge handling
///
/// - **N-direction edge:** caller is responsible for `block_col + col_start`
///   not overflowing `N`. The loader gates the 4-byte load on
///   `block_col + col_start + 3 < N`; if false, it skips the load and
///   relies on the pre-zero pass to leave safe zeros in shared.
/// - **K-direction edge:** `K % K_TILE_SHARED == 0` is enforced at
///   `validate_dims_qkv_int8`, so `row_in_tile < K_TILE_SHARED` always
///   indexes a valid global K row within the current K-tile iteration.
///   No K-edge branch inside the loader.
///
/// `w_block_base_global` must already include both the block-column offset
/// (`block_col`) and the per-K-tile offset (`k_tile * K_TILE_SHARED * N`).
/// `n_bytes` is `N * 1 = N` (kept as a register for parametricity).
///
/// `label_suffix` differentiates the skip-label in kernels that call this
/// helper more than once (Serial fusion calls it three times per K-tile —
/// once per projection); pass `"Q"` / `"K"` / `"V"` to keep PTX labels
/// unique.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)] // wired up in D3.3 / D3.4
pub(crate) fn emit_mw_load_tile_w_int8_16x32(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    w_block_base_global: Register, // u64 — W_P[k_tile * 16, block_col]
    tile_w_shared: Register,       // u32
    flat_tid: Register,            // u32 — 0..128
    block_col: Register,           // u32 — block's N-col start
    n: Register,                   // u32 — full N dimension
    n_bytes: Register,             // u32 — N * 1 (for edge-predicate math; kept parametric)
    label_suffix: &str,
) {
    let _ = n_bytes; // reserved for future ragged-tile handling along N
    let skip_label = if label_suffix.is_empty() {
        "W_SKIP_I8_TILE_LOAD".to_string()
    } else {
        format!("W_SKIP_I8_TILE_LOAD_{label_suffix}")
    };

    // row_in_tile = flat_tid / 8
    let row_in_tile = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: row_in_tile,
        lhs: Operand::Reg(flat_tid),
        rhs: Operand::ImmU32(8),
        ty: PtxType::U32,
    }));
    // col_group = flat_tid % 8
    let col_group = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Rem {
        dst: col_group,
        lhs: Operand::Reg(flat_tid),
        rhs: Operand::ImmU32(8),
        ty: PtxType::U32,
    }));
    // col_start = col_group * 4 (bytes within tile row; each thread handles 4 N-cols)
    let col_start = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: col_start,
        lhs: Operand::Reg(col_group),
        rhs: Operand::ImmU32(4),
        ty: PtxType::U32,
    }));

    // In-tile predicate (Rollback #1): the per-thread layout drives col_start
    // through {0,4,...,28} regardless of `BN_BLOCK`. After rollback to
    // `BN_BLOCK = 16`, threads with `col_start >= BN_BLOCK` would write past
    // the projection's owned columns into pre-zeroed padding (the bytes are
    // safe to clobber but the global read would be from beyond the block's
    // owned N range). Skip those threads entirely.
    let p_col_in_tile = alloc.alloc(PtxType::Pred);
    kernel.push(PtxInstruction::Control(ControlOp::SetP {
        dst: p_col_in_tile,
        cmp_op: CmpOp::Lt,
        lhs: Operand::Reg(col_start),
        rhs: Operand::ImmU32(BN_BLOCK),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Control(ControlOp::BraPred {
        pred: p_col_in_tile,
        target: skip_label.clone(),
        negate: true,
    }));

    // Edge predicate: skip if block_col + col_start + 3 >= N (any of the 4
    // N-bytes would be OOB). This keeps full-tile loads simple; ragged
    // columns fall through to the pre-zeroed shared slot.
    let col_global = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: col_global,
        lhs: Operand::Reg(block_col),
        rhs: Operand::Reg(col_start),
        ty: PtxType::U32,
    }));
    let col_global_plus_three = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: col_global_plus_three,
        lhs: Operand::Reg(col_global),
        rhs: Operand::ImmU32(3),
        ty: PtxType::U32,
    }));
    let p_col_in = alloc.alloc(PtxType::Pred);
    kernel.push(PtxInstruction::Control(ControlOp::SetP {
        dst: p_col_in,
        cmp_op: CmpOp::Lt,
        lhs: Operand::Reg(col_global_plus_three),
        rhs: Operand::Reg(n),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Control(ControlOp::BraPred {
        pred: p_col_in,
        target: skip_label.clone(),
        negate: true,
    }));

    // --- In-bounds path: read 4 bytes from global, write 4 bytes to shared. ---

    // Global offset = row_in_tile * n_bytes + col_start (bytes).
    //
    // Note: `w_block_base_global` already carries the `block_col` byte shift
    // (computed once per block in `build_qkv_project_int8_module`), so the
    // within-tile offset uses `col_start` not `col_global`. Adding
    // `col_global = block_col + col_start` here would double-count
    // `block_col` and corrupt every non-zero N-block (discovered during D7
    // multi-block e2e).
    let row_bytes_global = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: row_bytes_global,
        lhs: Operand::Reg(row_in_tile),
        rhs: Operand::Reg(n),
        ty: PtxType::U32,
    }));
    let global_off_u32 = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: global_off_u32,
        lhs: Operand::Reg(row_bytes_global),
        rhs: Operand::Reg(col_start),
        ty: PtxType::U32,
    }));
    let global_off_u64 = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Cvt {
        dst: global_off_u64,
        src: global_off_u32,
        dst_ty: PtxType::U64,
        src_ty: PtxType::U32,
    });
    let global_addr = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: global_addr,
        lhs: Operand::Reg(w_block_base_global),
        rhs: Operand::Reg(global_off_u64),
        ty: PtxType::U64,
    }));
    let loaded = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdGlobal {
        dst: loaded,
        addr: global_addr,
        ty: PtxType::U32,
    }));

    // Shared offset = row_in_tile * TILE_W_ROW_STRIDE_BYTES + col_start.
    let shared_row_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: shared_row_off,
        lhs: Operand::Reg(row_in_tile),
        rhs: Operand::ImmU32(TILE_W_ROW_STRIDE_BYTES),
        ty: PtxType::U32,
    }));
    let shared_off_within_tile = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: shared_off_within_tile,
        lhs: Operand::Reg(shared_row_off),
        rhs: Operand::Reg(col_start),
        ty: PtxType::U32,
    }));
    let shared_addr = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: shared_addr,
        lhs: Operand::Reg(tile_w_shared),
        rhs: Operand::Reg(shared_off_within_tile),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Memory(MemoryOp::StShared {
        addr: shared_addr,
        src: loaded,
        ty: PtxType::U32,
    }));

    kernel.push(PtxInstruction::Label(skip_label));
}

/// Cooperative pre-zero of the X + W shared tiles. Same pattern as
/// `matmul_int8_kernel::emit_pre_zero_shared_tiles_int8` — each thread
/// emits one `st.shared.u32` per `PRE_ZERO_BYTES_PER_ISSUE` chunk, and
/// the two tiles are handled back-to-back. Followed by a single
/// `bar.sync 0` so all threads observe the zeroed slots before the
/// cooperative loaders run.
///
/// `TILE_X_BYTES = 2048 B` = 4 issues per thread, `TILE_W_BYTES = 512 B`
/// = 1 issue per thread — totals 5 `st.shared.u32` per thread + 1 barrier.
#[allow(dead_code)] // wired up in D3.4
pub(crate) fn emit_pre_zero_shared_tiles_qkv_int8(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    tile_x: Register,
    tile_w: Register,
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
    kernel.push(PtxInstruction::Control(ControlOp::BarSync {
        barrier_id: 0,
    }));
}

/// Per-lane dequant of one `FragmentB_M16N8K16` for the INT8 W8A16 path.
///
/// Sprint 7.3 D3.2. Reads 4 `s8` bytes from the shared W tile, converts
/// each to `f16` via `cvt.rn.f16.s8`, and `MovPack`s them into 2 `.b32`
/// registers holding two packed f16 values each — the standard feed
/// format for the `mma.sync.m16n8k16.f16.f16.f32` B operand. No scale
/// is applied here; the INT8 variant folds its **scalar** per-projection
/// scale into the store-out chain (see
/// [`crate::store_out::emit_store_fragment_c_f32_to_f16_packed`]).
///
/// # Fragment-B layout (PTX ISA §9.7.13.5.8.1 for m16n8k16.f16)
///
/// Lane `l ∈ 0..32` owns fragment-B col `group_id = l / 4` (0..8) and
/// the four rows `{2*tig, 2*tig+1, 2*tig+8, 2*tig+9}` where `tig = l % 4`:
///
/// ```text
///   lane.reg[0] = pack( f16[row = 2*tig + 0, col = group_id],
///                       f16[row = 2*tig + 1, col = group_id] )
///   lane.reg[1] = pack( f16[row = 2*tig + 8, col = group_id],
///                       f16[row = 2*tig + 9, col = group_id] )
/// ```
///
/// # Shared tile assumption
///
/// The W tile has been cooperatively loaded in **row-major** layout (K
/// as row, N as column, stride = [`TILE_W_ROW_STRIDE_BYTES`] = 32 B)
/// by [`emit_mw_load_tile_w_int8_16x32`]. Per-lane byte addresses:
///
/// ```text
///   byte(k, n) = tile_w_shared + k * 32 + n
/// ```
///
/// Each lane issues 4 × `ld.shared.s8` at four `(k, n_col_abs)` positions,
/// one per K-row the lane owns.
///
/// # Why row-major shared
///
/// Alternative col-major shared would let each lane issue a single
/// `ld.shared.u32` (4 adjacent K-bytes in one word) instead of 4 `ld.s8`,
/// but would force the cooperative loader to transpose during the
/// `st.shared` epoch — 4 byte-wide stores per thread instead of one
/// 4-byte store. Design choice: keep the cooperative load simple and
/// coalesced (one b32 per thread) and pay the per-lane fragment-B read
/// cost in 4 byte loads. The INT8 dequant chain is already dominated by
/// the 4 `cvt.rn.f16.s8` and 2 `MovPack` instructions; 4 × 1-byte shared
/// loads add negligible marginal cost relative to the arithmetic.
#[allow(dead_code)] // wired up in D3.3
pub(crate) fn emit_fragment_b_int8_per_lane(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    tile_w_shared: Register,
    n_stripe_col_base: Register,
    lane_within_warp: Register,
) -> kaio_core::fragment::FragmentB_F16 {
    // group_id = lane / 4   (0..8 — N-col within the 8-col fragment)
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
    // col_abs = n_stripe_col_base + group_id (N-col in tile, 0..32)
    let col_abs = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: col_abs,
        lhs: Operand::Reg(n_stripe_col_base),
        rhs: Operand::Reg(group_id),
        ty: PtxType::U32,
    }));

    // two_tig = tig * 2  (K-row base for reg[0])
    let two_tig = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: two_tig,
        lhs: Operand::Reg(tig),
        rhs: Operand::ImmU32(2),
        ty: PtxType::U32,
    }));
    // two_tig_plus_8 = two_tig + 8  (K-row base for reg[1])
    let two_tig_plus_8 = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: two_tig_plus_8,
        lhs: Operand::Reg(two_tig),
        rhs: Operand::ImmU32(8),
        ty: PtxType::U32,
    }));

    // Load one s8 byte from shared at (k_row, col_abs) and cvt.rn.f16.s8
    // to an f16 register. Returns the f16 register holding the dequanted value.
    let load_one_s8_to_f16 =
        |alloc: &mut RegisterAllocator, kernel: &mut PtxKernel, k_row: Register| -> Register {
            // shared_off = k_row * TILE_W_ROW_STRIDE_BYTES + col_abs
            let row_off = alloc.alloc(PtxType::U32);
            kernel.push(PtxInstruction::Arith(ArithOp::Mul {
                dst: row_off,
                lhs: Operand::Reg(k_row),
                rhs: Operand::ImmU32(TILE_W_ROW_STRIDE_BYTES),
                ty: PtxType::U32,
            }));
            let total_off = alloc.alloc(PtxType::U32);
            kernel.push(PtxInstruction::Arith(ArithOp::Add {
                dst: total_off,
                lhs: Operand::Reg(row_off),
                rhs: Operand::Reg(col_abs),
                ty: PtxType::U32,
            }));
            let addr = alloc.alloc(PtxType::U32);
            kernel.push(PtxInstruction::Arith(ArithOp::Add {
                dst: addr,
                lhs: Operand::Reg(tile_w_shared),
                rhs: Operand::Reg(total_off),
                ty: PtxType::U32,
            }));
            // ld.shared.s8 %r_byte, [addr] — lands in the %r integer-register
            // class per the `PtxType::S8` doc (kaio-core/types.rs:38).
            let s8_reg = alloc.alloc(PtxType::S8);
            kernel.push(PtxInstruction::Memory(MemoryOp::LdShared {
                dst: s8_reg,
                addr,
                ty: PtxType::S8,
            }));
            // cvt.rn.f16.s8 %h, %r — sign-extend-then-convert, per ISA.
            let h = alloc.alloc(PtxType::F16);
            kernel.push(PtxInstruction::Cvt {
                dst: h,
                src: s8_reg,
                dst_ty: PtxType::F16,
                src_ty: PtxType::S8,
            });
            h
        };

    // k_row0 = two_tig, k_row1 = two_tig + 1 (reg[0] inputs)
    let k_row1 = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: k_row1,
        lhs: Operand::Reg(two_tig),
        rhs: Operand::ImmU32(1),
        ty: PtxType::U32,
    }));
    let h0 = load_one_s8_to_f16(alloc, kernel, two_tig);
    let h1 = load_one_s8_to_f16(alloc, kernel, k_row1);

    // k_row2 = two_tig + 8, k_row3 = two_tig + 9 (reg[1] inputs)
    let k_row3 = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: k_row3,
        lhs: Operand::Reg(two_tig_plus_8),
        rhs: Operand::ImmU32(1),
        ty: PtxType::U32,
    }));
    let h2 = load_one_s8_to_f16(alloc, kernel, two_tig_plus_8);
    let h3 = load_one_s8_to_f16(alloc, kernel, k_row3);

    // Pack adjacent-K f16 pairs into b32 registers for the mma feed.
    let reg0 = alloc.alloc_packed_half2();
    kernel.push(PtxInstruction::MovPack {
        dst: reg0,
        srcs: vec![h0, h1],
        ty: PtxType::U32,
    });
    let reg1 = alloc.alloc_packed_half2();
    kernel.push(PtxInstruction::MovPack {
        dst: reg1,
        srcs: vec![h2, h3],
        ty: PtxType::U32,
    });

    kaio_core::fragment::FragmentB_F16 { regs: [reg0, reg1] }
}

/// Per-warp-quadrant tri-output mma sweep for one projection.
///
/// Sprint 7.3 D3.3. For a single projection P ∈ {Q, K, V}, emits the
/// 2×2 inner grid of `mma.sync.m16n8k16.f16.f16.f32` accumulations over
/// the warp quadrant, reading f16 A-fragments from `tile_x_shared` and
/// dequanted f16 B-fragments from `tile_w_shared` (via
/// [`emit_fragment_b_int8_per_lane`]). Accumulates in-place into
/// `frag_c_grid` so the caller's per-projection fragment-C bank
/// persists across K-tile iterations.
///
/// # Structure
///
/// ```text
/// for n_stripe in 0..MMAS_PER_WARP_N (= 2):
///     frag_b = emit_fragment_b_int8_per_lane(...)     // reused across m_stripes
///     for m_stripe in 0..MMAS_PER_WARP_M (= 2):
///         frag_a = load_fragment_a_m16n8k16_shared_row(...)
///         frag_d = alloc_c(...)
///         mma.sync.m16n8k16.f16.f16.f32 frag_d = frag_a * frag_b + frag_c_grid[m][n]
///         frag_c_grid[m][n] = frag_d                  // mov.f32 × 4 (in-place update)
/// ```
///
/// The `mov.f32` copy-back after each mma reflects the kaio-core IR's
/// separate `d` and `c` operands — ptxas does its own liveness
/// analysis and typically folds the copy into register renaming.
/// Pattern matches `matmul_int4_kernel::emit_warp_quadrant_mma_int4`
/// (Sprint 7.2) exactly, just with INT8 fragment-B dequant instead of
/// INT4.
///
/// # Reuse across projections
///
/// This function is invoked **three times per warp per K-tile** by
/// the D3.4 module builder — once each for Q, K, V. Each call operates
/// on a distinct `frag_c_grid` (the per-projection accumulator bank)
/// but all three share the same `tile_x_shared` (X is loaded once per
/// K-tile and reused) and the same `tile_w_shared` slot (cooperatively
/// reloaded between projections by D3.4 with `bar.sync` gates).
///
/// # Arguments
///
/// - `tile_x_shared` — u32 register holding the shared base of the X tile.
/// - `tile_w_shared` — u32 register holding the shared base of the current
///   projection's W_P_i8 tile.
/// - `warp_quad_row_base_in_tile_x` — u32 register holding the warp's
///   M-row offset within the block tile (0 or 32 for the 2×2 warp grid).
/// - `warp_quad_col_base_in_tile_w` — u32 register holding the warp's
///   N-col offset within the block tile (0 or 16 for the 2×2 warp grid).
/// - `tid_x_in_warp` — u32 register holding `tid.x % 32`.
/// - `frag_c_grid` — `&mut [[FragmentC; MMAS_PER_WARP_N]; MMAS_PER_WARP_M]`,
///   the per-projection accumulator bank (2×2 = 4 fragments, each with 4
///   f32 regs per lane = 16 f32 regs per lane per projection).
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)] // wired up in D3.4
pub(crate) fn emit_warp_quadrant_mma_int8_per_projection(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    tile_x_shared: Register,
    tile_w_shared: Register,
    warp_quad_row_base_in_tile_x: Register,
    warp_quad_col_base_in_tile_w: Register,
    tid_x_in_warp: Register,
    frag_c_grid: &mut [[kaio_core::fragment::FragmentC; MMAS_PER_WARP_N as usize];
             MMAS_PER_WARP_M as usize],
) {
    use kaio_core::fragment::load_fragment_a_m16n8k16_shared_row;
    use kaio_core::instr::{MmaShape, TensorCoreOp};

    // Pre-compute the warp-quadrant row-offset in bytes. Same value is
    // reused across all n-stripes and m-stripes for the base shift into
    // the shared X tile — computed once outside the loop.
    let row_offset_bytes = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: row_offset_bytes,
        lhs: Operand::Reg(warp_quad_row_base_in_tile_x),
        rhs: Operand::ImmU32(TILE_X_ROW_STRIDE_BYTES),
        ty: PtxType::U32,
    }));

    for n_stripe in 0..MMAS_PER_WARP_N {
        // n_stripe_col_base = warp_quad_col_base + n_stripe * BN
        //   (BN = 8 cols per mma-N sub-tile; 2 stripes cover the 16-col warp quadrant)
        let n_stripe_col_base = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: n_stripe_col_base,
            lhs: Operand::Reg(warp_quad_col_base_in_tile_w),
            rhs: Operand::ImmU32(n_stripe * BN),
            ty: PtxType::U32,
        }));

        // Per-lane dequant fragment B once per n-stripe; reused across m-stripes.
        let frag_b = emit_fragment_b_int8_per_lane(
            alloc,
            kernel,
            tile_w_shared,
            n_stripe_col_base,
            tid_x_in_warp,
        );

        for m_stripe in 0..MMAS_PER_WARP_M {
            // Shift the tile_x base so the fragment-A loader reads the right
            // BM=16 rows within the warp quadrant's 32-row span.
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

            // mma.sync d = a * b + c; then copy d → c so the next K-tile
            // accumulates on top. Same idiom as matmul_int4.
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

/// Emit the 2×1 mma sweep for one projection using **pre-hoisted**
/// fragment-A registers.
///
/// Sprint 7.3.5 D2 — S+½P variant. Identical output to
/// [`emit_warp_quadrant_mma_int8_per_projection`] except frag_A is
/// loaded **once per m_stripe per K-tile** by the caller (before the
/// B1 barrier) instead of being re-loaded from `tile_x` on every
/// projection's mma. This is the mechanism that makes Design
/// invariant #1 ("`tile_x` overwrite-after-hoist") expressible: once
/// all threads have completed their frag_A ld.shared from `tile_x`
/// and cleared the B1 barrier, `tile_x` is safe to overwrite with
/// `X_next`, so the X-load for the next K-tile can overlap with the
/// current K-tile's mma_V epoch.
///
/// Caller contract:
///
/// - `frag_as[m_stripe]` holds the fragment-A registers loaded from
///   `tile_x` at the current K-tile's warp-quadrant-row offset for
///   m_stripe ∈ `{0, 1}`. The caller MUST fire `bar.sync` (B1)
///   between these loads and any subsequent `tile_x` overwrite.
/// - `tile_w_shared_slot` is the u32 shared base of **the current
///   projection's ping-pong slot** for this K-tile. Different calls
///   (Q vs K vs V) pass different slot bases per the K-loop's
///   ping-pong indexing.
/// - Everything else matches
///   [`emit_warp_quadrant_mma_int8_per_projection`].
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)] // wired up in D2 (Sprint 7.3.5)
pub(crate) fn emit_warp_quadrant_mma_int8_per_projection_hoisted(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    tile_w_shared_slot: Register,
    warp_quad_col_base_in_tile_w: Register,
    tid_x_in_warp: Register,
    frag_as: &[kaio_core::fragment::FragmentA_F16; MMAS_PER_WARP_M as usize],
    frag_c_grid: &mut [[kaio_core::fragment::FragmentC; MMAS_PER_WARP_N as usize];
             MMAS_PER_WARP_M as usize],
) {
    use kaio_core::instr::{MmaShape, TensorCoreOp};

    for n_stripe in 0..MMAS_PER_WARP_N {
        // n_stripe_col_base = warp_quad_col_base + n_stripe * BN
        let n_stripe_col_base = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: n_stripe_col_base,
            lhs: Operand::Reg(warp_quad_col_base_in_tile_w),
            rhs: Operand::ImmU32(n_stripe * BN),
            ty: PtxType::U32,
        }));

        // Per-lane dequant fragment B once per n-stripe; reused across m-stripes.
        let frag_b = emit_fragment_b_int8_per_lane(
            alloc,
            kernel,
            tile_w_shared_slot,
            n_stripe_col_base,
            tid_x_in_warp,
        );

        for m_stripe in 0..MMAS_PER_WARP_M {
            let frag_a = frag_as[m_stripe as usize];

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

/// Hoist both fragment-A sets (one per m_stripe) from `tile_x` at
/// K-tile start for the S+½P kernel.
///
/// Sprint 7.3.5 D2 — INV #1 helper. Invoked **once per K-tile**
/// immediately before the B1 barrier. The returned two `FragmentA_F16`
/// structs are used by all three projection mma calls (Q, K, V)
/// within the same K-tile; `tile_x` must stay stable between this
/// call and the B1 barrier, but may be overwritten by `X_next`
/// after B1 — that's precisely the overlap window Design invariant
/// #1 authorises.
///
/// # Emit-site invariant
///
/// The CALLER is responsible for emitting `bar.sync` (B1) AFTER
/// this function returns and BEFORE any instruction that writes to
/// `tile_x` in the current K-tile. See the D2 K-loop body for the
/// exact sequencing.
#[allow(dead_code)] // wired up in D2 (Sprint 7.3.5)
pub(crate) fn hoist_frag_as_for_warp_quadrant(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    tile_x_shared: Register,
    warp_quad_row_base_in_tile_x: Register,
    tid_x_in_warp: Register,
) -> [kaio_core::fragment::FragmentA_F16; MMAS_PER_WARP_M as usize] {
    use kaio_core::fragment::load_fragment_a_m16n8k16_shared_row;

    // Pre-compute the warp-quadrant row-offset in bytes.
    let row_offset_bytes = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: row_offset_bytes,
        lhs: Operand::Reg(warp_quad_row_base_in_tile_x),
        rhs: Operand::ImmU32(TILE_X_ROW_STRIDE_BYTES),
        ty: PtxType::U32,
    }));

    core::array::from_fn(|m_stripe| {
        let m_stripe_shift = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: m_stripe_shift,
            lhs: Operand::Reg(row_offset_bytes),
            rhs: Operand::ImmU32(m_stripe as u32 * BM * TILE_X_ROW_STRIDE_BYTES),
            ty: PtxType::U32,
        }));
        let shifted_tile_x = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: shifted_tile_x,
            lhs: Operand::Reg(tile_x_shared),
            rhs: Operand::Reg(m_stripe_shift),
            ty: PtxType::U32,
        }));
        load_fragment_a_m16n8k16_shared_row(
            alloc,
            kernel,
            shifted_tile_x,
            tid_x_in_warp,
            TILE_X_ROW_STRIDE_BYTES,
            None,
        )
    })
}

/// Cooperative pre-zero for the **S+½P** 3-slot layout (tile_x +
/// tile_w_slot0 + tile_w_slot1). Matches
/// [`emit_pre_zero_shared_tiles_qkv_int8`] but takes an extra slot
/// register; the 64 B bank-phase pad (declared separately in the
/// module but unused by any kernel path) is NOT zeroed — it's write-
/// only padding whose contents are never read.
#[allow(dead_code)] // wired up in D2 (Sprint 7.3.5)
pub(crate) fn emit_pre_zero_shared_tiles_qkv_int8_sp_half_p(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    tile_x: Register,
    tile_w_slot0: Register,
    tile_w_slot1: Register,
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

    for (tile_base, total_bytes) in [
        (tile_x, TILE_X_BYTES),
        (tile_w_slot0, TILE_W_BYTES),
        (tile_w_slot1, TILE_W_BYTES),
    ] {
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
    // Pre-zero requires a trailing barrier because cooperative pre-load
    // (X(0), W_Q(0)) uses a different thread-to-byte mapping than pre-
    // zero; without a barrier, thread A's pre-zero and thread B's pre-
    // load can race on the same address and the final value becomes
    // undefined. The plan's "1 + 4*N" barrier count assumed pre-zero
    // could fold into the pre-load barrier — in practice we accept one
    // additional setup barrier (total: 2 setup + 4 per K-tile) for
    // correctness on ragged M/N tiles.
    kernel.push(PtxInstruction::Control(ControlOp::BarSync {
        barrier_id: 0,
    }));
}

/// Build the full IR module for `qkv_project_int8` targeting `sm`.
///
/// Sprint 7.3 D3.4 — MVS module builder. Wires the D3.1 cooperative
/// loaders ([`emit_mw_load_tile_x_f16_64x16`] +
/// [`emit_mw_load_tile_w_int8_16x32`]), D3.2 fragment-B dequant
/// ([`emit_fragment_b_int8_per_lane`]), D3.3 per-projection mma sweep
/// ([`emit_warp_quadrant_mma_int8_per_projection`]), and the D2 shared
/// store-out helper
/// ([`emit_store_fragment_c_f32_to_f16_packed`][crate::store_out::emit_store_fragment_c_f32_to_f16_packed])
/// into a single PTX module producing three f16 output projections from
/// one shared activation X.
///
/// # Kernel signature (in launch order)
///
/// `x_ptr: *const f16, w_q_ptr: *const i8, w_k_ptr: *const i8,
///  w_v_ptr: *const i8, scale_q: f32, scale_k: f32, scale_v: f32,
///  q_out_ptr: *mut f16, k_out_ptr: *mut f16, v_out_ptr: *mut f16,
///  m: u32, n: u32, k: u32`.
///
/// # Block + grid
///
/// `block_dim = (32, 4, 1)` = 128 threads = 4 warps per block.
/// `grid_dim = (n / BN_BLOCK, m / BM_BLOCK, 1)` (caller responsibility
/// at launch).
///
/// # Per-K-tile barrier cadence (Design S — serial fusion)
///
/// 1 X-load sync + (1 W_P load sync + 1 W_P release sync) × 3 = **7
/// barriers per K-tile**. The release sync after V cannot fold into the
/// next K-tile's X-load sync because the X slot is being overwritten
/// and requires all prior mma_V reads to drain.
#[allow(dead_code)] // wired up in D4
pub(crate) fn build_qkv_project_int8_module(sm: &str) -> kaio_core::ir::PtxModule {
    use kaio_core::fragment::{FragmentC, alloc_c};
    use kaio_core::instr::MadMode;
    use kaio_core::ir::{PtxModule, PtxParam, SharedDecl};

    let mut alloc = RegisterAllocator::new();
    let mut kernel = PtxKernel::new("qkv_project_int8");

    // --- Kernel signature ---
    kernel.add_param(PtxParam::pointer("x_ptr", PtxType::F16));
    kernel.add_param(PtxParam::pointer("w_q_ptr", PtxType::S8));
    kernel.add_param(PtxParam::pointer("w_k_ptr", PtxType::S8));
    kernel.add_param(PtxParam::pointer("w_v_ptr", PtxType::S8));
    kernel.add_param(PtxParam::scalar("scale_q", PtxType::F32));
    kernel.add_param(PtxParam::scalar("scale_k", PtxType::F32));
    kernel.add_param(PtxParam::scalar("scale_v", PtxType::F32));
    kernel.add_param(PtxParam::pointer("q_out_ptr", PtxType::F16));
    kernel.add_param(PtxParam::pointer("k_out_ptr", PtxType::F16));
    kernel.add_param(PtxParam::pointer("v_out_ptr", PtxType::F16));
    kernel.add_param(PtxParam::scalar("m", PtxType::U32));
    kernel.add_param(PtxParam::scalar("n", PtxType::U32));
    kernel.add_param(PtxParam::scalar("k", PtxType::U32));

    // --- Shared decls (Sprint 7.3.5 S+½P: two W slots with bank-phase pad) ---
    //
    // tile_x stays single-slot (2048 B); `frag_A` is register-hoisted
    // across the K-loop iteration (Design invariant #1) so `tile_x` can
    // be overwritten by `X_next` while the hoisted frag_A serves all
    // three mma epochs.
    //
    // tile_w splits into slot0 (512 B) + pad (64 B, R3-1
    // Design invariant #4 — non-multiple-of-128 stride avoids cross-warp
    // SMEM bank-port contention during overlap) + slot1 (512 B). Ping-
    // pong indexing selects slot per K-tile × projection per the K-loop
    // body below.
    kernel.add_shared_decl(SharedDecl {
        name: "tile_x".to_string(),
        align: 4,
        size_bytes: TILE_X_BYTES,
    });
    kernel.add_shared_decl(SharedDecl {
        name: "tile_w_slot0".to_string(),
        align: 4,
        size_bytes: TILE_W_BYTES,
    });
    kernel.add_shared_decl(SharedDecl {
        name: "tile_w_pad".to_string(),
        align: 4,
        size_bytes: TILE_W_SLOT_PAD_BYTES,
    });
    kernel.add_shared_decl(SharedDecl {
        name: "tile_w_slot1".to_string(),
        align: 4,
        size_bytes: TILE_W_BYTES,
    });

    // --- Load + cvta the 7 pointer params ---
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
    let rd_w_q = load_and_cvta("w_q_ptr", &mut alloc, &mut kernel);
    let rd_w_k = load_and_cvta("w_k_ptr", &mut alloc, &mut kernel);
    let rd_w_v = load_and_cvta("w_v_ptr", &mut alloc, &mut kernel);
    let rd_q_out = load_and_cvta("q_out_ptr", &mut alloc, &mut kernel);
    let rd_k_out = load_and_cvta("k_out_ptr", &mut alloc, &mut kernel);
    let rd_v_out = load_and_cvta("v_out_ptr", &mut alloc, &mut kernel);

    // --- Load 3 scalar scales + 3 dim scalars ---
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
    let r_scale_q = load_scalar("scale_q", PtxType::F32, &mut alloc, &mut kernel);
    let r_scale_k = load_scalar("scale_k", PtxType::F32, &mut alloc, &mut kernel);
    let r_scale_v = load_scalar("scale_v", PtxType::F32, &mut alloc, &mut kernel);
    let r_m = load_scalar("m", PtxType::U32, &mut alloc, &mut kernel);
    let r_n = load_scalar("n", PtxType::U32, &mut alloc, &mut kernel);
    let r_k = load_scalar("k", PtxType::U32, &mut alloc, &mut kernel);

    // --- Derived dim scalars ---
    // k_bytes = K * 2  (X is f16)
    let r_k_bytes = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_k_bytes,
        lhs: Operand::Reg(r_k),
        rhs: Operand::ImmU32(BYTES_PER_F16),
        ty: PtxType::U32,
    }));
    // n_bytes = N * 1  (W is i8)
    let r_n_bytes = r_n;
    // n_f16_stride = N * 2  (output row stride for f16 row-major)
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
    // flat_tid = tid_y * blockDim.x + tid_x  (block = (32, 4, 1))
    let r_flat_tid = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mad {
        dst: r_flat_tid,
        a: Operand::Reg(r_tid_y),
        b: Operand::Reg(r_ntid_x),
        c: Operand::Reg(r_tid_x),
        ty: PtxType::U32,
        mode: MadMode::Lo,
    }));
    // warp_id = tid_y (one warp per row of the (32,4,1) block).
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
    // wq_row_idx = warp_row_quad * WARP_QUAD_M (0 or 32) — passed to mma helper as a row index.
    let r_wq_row_idx = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_wq_row_idx,
        lhs: Operand::Reg(r_warp_row_quad),
        rhs: Operand::ImmU32(WARP_QUAD_M),
        ty: PtxType::U32,
    }));
    // wq_col_idx = warp_col_quad * WARP_QUAD_N (0 or 16)
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

    // --- Shared base regs (Sprint 7.3.5 S+½P) ---
    let r_tile_x = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_tile_x,
        src: Operand::SharedAddr("tile_x".to_string()),
        ty: PtxType::U32,
    });
    let r_tile_w_slot0 = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_tile_w_slot0,
        src: Operand::SharedAddr("tile_w_slot0".to_string()),
        ty: PtxType::U32,
    });
    let r_tile_w_slot1 = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_tile_w_slot1,
        src: Operand::SharedAddr("tile_w_slot1".to_string()),
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

    // w_P_block_base = rd_w_P + block_col  (1 byte per i8, direct widening)
    let rd_block_col_64 = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Cvt {
        dst: rd_block_col_64,
        src: r_block_col,
        dst_ty: PtxType::U64,
        src_ty: PtxType::U32,
    });
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
    let rd_w_q_block_base = add_u64(&mut alloc, &mut kernel, rd_w_q, rd_block_col_64);
    let rd_w_k_block_base = add_u64(&mut alloc, &mut kernel, rd_w_k, rd_block_col_64);
    let rd_w_v_block_base = add_u64(&mut alloc, &mut kernel, rd_w_v, rd_block_col_64);

    // --- Allocate 3 frag_c grids (one per projection), zero-init f32 ---
    // 3 projections × 4 sub-tiles × 4 f32 regs = 48 f32 regs per lane (the
    // dominant register pressure measured at D2.5 — 32-40 regs/thread,
    // 24-32 headroom under the 64-reg cliff).
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

    // --- Pre-zero shared tiles (3-slot sp-half-p variant) + bar.sync ---
    emit_pre_zero_shared_tiles_qkv_int8_sp_half_p(
        &mut alloc,
        &mut kernel,
        r_tile_x,
        r_tile_w_slot0,
        r_tile_w_slot1,
        r_flat_tid,
    );

    // --- K-loop setup: compute num_k_tiles, init k_tile = 0 ---
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

    // --- Pre-loop: load X(0) + W_Q(0) into slot0, then B0 bar.sync ---
    //
    // S+½P invariant: at K-loop entry, `tile_x` holds X(k_tile) and the
    // "current" slot (slot[k_tile % 2]) holds W_Q(k_tile). For k_tile=0
    // that means tile_w_slot0 (since 0 & 1 == 0). Subsequent iterations
    // maintain the invariant via the overlap loads at the bottom of
    // each K-tile body.
    //
    // k_tile=0 X/W offsets are zero — no address math needed, the
    // block-base registers ARE the k_tile=0 tile sources.
    emit_mw_load_tile_x_f16_64x16(
        &mut alloc,
        &mut kernel,
        rd_x_block_base,
        r_tile_x,
        r_flat_tid,
        r_block_row,
        r_m,
        r_k_bytes,
        "QKV_PRELOAD",
    );
    emit_mw_load_tile_w_int8_16x32(
        &mut alloc,
        &mut kernel,
        rd_w_q_block_base,
        r_tile_w_slot0,
        r_flat_tid,
        r_block_col,
        r_n,
        r_n_bytes,
        "Q_PRELOAD",
    );
    // B0 — X(0) + W_Q(0) visible. Combined with the pre-zero barrier
    // emitted inside emit_pre_zero_shared_tiles_qkv_int8_sp_half_p,
    // total setup = 2 barriers (pre-zero + B0). Per-K-tile remains 4
    // (B1 + B2 + B3 + B4); see the K-loop body below.
    kernel.push(PtxInstruction::Control(ControlOp::BarSync {
        barrier_id: 0,
    }));

    // --- K-loop label ---
    kernel.push(PtxInstruction::Label(
        "K_LOOP_QKV_INT8_SP_HALF_P".to_string(),
    ));

    // ── Runtime ping-pong slot selection ──
    //   slot_idx    = k_tile & 1              (0 or 1)
    //   p_slot0     = (slot_idx == 0)
    //   cur_w_base  = p_slot0 ? slot0 : slot1 (CURRENT slot — at K-tile
    //                 entry holds W_Q(k_tile) per the worked-example
    //                 invariant)
    //   next_w_base = p_slot0 ? slot1 : slot0 (NEXT slot — target of the
    //                 W_K load below)
    let r_slot_idx = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::And {
        dst: r_slot_idx,
        lhs: Operand::Reg(r_k_tile),
        rhs: Operand::ImmU32(1),
        ty: PtxType::U32,
    }));
    let p_slot0 = alloc.alloc(PtxType::Pred);
    kernel.push(PtxInstruction::Control(ControlOp::SetP {
        dst: p_slot0,
        cmp_op: CmpOp::Eq,
        lhs: Operand::Reg(r_slot_idx),
        rhs: Operand::ImmU32(0),
        ty: PtxType::U32,
    }));
    let r_cur_w_base = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Selp {
        dst: r_cur_w_base,
        a: Operand::Reg(r_tile_w_slot0),
        b: Operand::Reg(r_tile_w_slot1),
        pred: p_slot0,
        ty: PtxType::U32,
    }));
    let r_next_w_base = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Selp {
        dst: r_next_w_base,
        a: Operand::Reg(r_tile_w_slot1),
        b: Operand::Reg(r_tile_w_slot0),
        pred: p_slot0,
        ty: PtxType::U32,
    }));

    // ── Hoist frag_A (2 m_stripes) from tile_x at K-tile start ──
    //
    // DESIGN INVARIANT #1 (tile_x overwrite-after-hoist, R2-3):
    // frag_A must be fully hoisted to registers before `tile_x` is
    // overwritten by the X_next cooperative load in Epoch 3 below.
    // All three projection mma calls read frag_A from these hoisted
    // registers — no projection re-reads `tile_x`. If a future change
    // reintroduces a per-projection `tile_x` read (e.g. by reverting
    // to the Design-S internal-load helper), the X_next overlap load
    // races the frag_A reload and produces nondeterministic output.
    //
    // See the k_tile=0 worked example (plan, step 10): any read from
    // tile_x after that step is a correctness bug.
    let frag_as =
        hoist_frag_as_for_warp_quadrant(&mut alloc, &mut kernel, r_tile_x, r_wq_row_idx, r_tid_x);

    // B1 — frag_A ld.shared completed on all threads; `tile_x` is now
    // safe to overwrite (DESIGN INVARIANT #1). The order here matters:
    // frag_A hoist comes BEFORE B1; any `tile_x` write after B1 (the
    // X_next overlap load in Epoch 3) is safe. Moving B1 after the
    // first W_K load — tempting for symmetry with B2/B3/B4 — would
    // race the tile_x overwrite against the frag_A hoist.
    kernel.push(PtxInstruction::Control(ControlOp::BarSync {
        barrier_id: 0,
    }));

    // ── Compute global addresses for W_K(k_tile), W_V(k_tile) ──
    let r_k_tile_rows = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_k_tile_rows,
        lhs: Operand::Reg(r_k_tile),
        rhs: Operand::ImmU32(K_TILE_SHARED),
        ty: PtxType::U32,
    }));
    let rd_w_row_off = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
        dst: rd_w_row_off,
        lhs: Operand::Reg(r_k_tile_rows),
        rhs: Operand::Reg(r_n_bytes),
        src_ty: PtxType::U32,
    }));
    let rd_w_k_tile_src = add_u64(&mut alloc, &mut kernel, rd_w_k_block_base, rd_w_row_off);
    let rd_w_v_tile_src = add_u64(&mut alloc, &mut kernel, rd_w_v_block_base, rd_w_row_off);

    // ── Epoch 1: load W_K(k_tile) → NEXT slot; mma_Q using CUR slot ──
    emit_mw_load_tile_w_int8_16x32(
        &mut alloc,
        &mut kernel,
        rd_w_k_tile_src,
        r_next_w_base,
        r_flat_tid,
        r_block_col,
        r_n,
        r_n_bytes,
        "K_EPOCH1",
    );
    emit_warp_quadrant_mma_int8_per_projection_hoisted(
        &mut alloc,
        &mut kernel,
        r_cur_w_base,
        r_wq_col_idx,
        r_tid_x,
        &frag_as,
        &mut frag_c_q,
    );
    // B2 — W_K(k_tile) visible in NEXT slot; mma_Q done (CUR slot safe
    // to overwrite for Epoch 2).
    kernel.push(PtxInstruction::Control(ControlOp::BarSync {
        barrier_id: 0,
    }));

    // ── Epoch 2: load W_V(k_tile) → CUR slot (overwrites W_Q); mma_K using NEXT slot ──
    emit_mw_load_tile_w_int8_16x32(
        &mut alloc,
        &mut kernel,
        rd_w_v_tile_src,
        r_cur_w_base,
        r_flat_tid,
        r_block_col,
        r_n,
        r_n_bytes,
        "V_EPOCH2",
    );
    emit_warp_quadrant_mma_int8_per_projection_hoisted(
        &mut alloc,
        &mut kernel,
        r_next_w_base,
        r_wq_col_idx,
        r_tid_x,
        &frag_as,
        &mut frag_c_k,
    );
    // B3 — W_V(k_tile) visible in CUR slot; mma_K done.
    kernel.push(PtxInstruction::Control(ControlOp::BarSync {
        barrier_id: 0,
    }));

    // ── Epoch 3: overlap [W_Q(k+1), X(k+1)] loads with mma_V ──
    //
    // Sets up NEXT K-tile's tile_x + next-slot state. After B4:
    //   tile_x          = X(k_tile + 1)
    //   slot[(k+1)%2]   = W_Q(k_tile+1)     ← overlap load destination
    //   slot[k_tile%2]  = W_V(k_tile)        ← still valid (mma_V reads
    //                                          it; B4 closes any
    //                                          lingering reads)
    //
    // On the LAST K-tile there is no k_tile+1 to preload, so both
    // overlap loads are skipped via a uniform branch on p_not_last.
    // mma_V still runs (the K-tile's final projection) and B4 still
    // fires — a no-op for the skipped loads but still gates mma_V
    // completion.
    let r_k_tile_plus_1 = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_k_tile_plus_1,
        lhs: Operand::Reg(r_k_tile),
        rhs: Operand::ImmU32(1),
        ty: PtxType::U32,
    }));
    let p_not_last = alloc.alloc(PtxType::Pred);
    kernel.push(PtxInstruction::Control(ControlOp::SetP {
        dst: p_not_last,
        cmp_op: CmpOp::Lt,
        lhs: Operand::Reg(r_k_tile_plus_1),
        rhs: Operand::Reg(r_num_k_tiles),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Control(ControlOp::BraPred {
        pred: p_not_last,
        target: "OVERLAP_SKIP_QKV_INT8_SP_HALF_P".to_string(),
        negate: true,
    }));
    // Inside "not last" branch: compute k_tile+1 offsets and issue
    // the two overlap loads.
    let r_k_next_x_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_k_next_x_off,
        lhs: Operand::Reg(r_k_tile_plus_1),
        rhs: Operand::ImmU32(K_TILE_SHARED * BYTES_PER_F16),
        ty: PtxType::U32,
    }));
    let rd_k_next_x_off64 = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Cvt {
        dst: rd_k_next_x_off64,
        src: r_k_next_x_off,
        dst_ty: PtxType::U64,
        src_ty: PtxType::U32,
    });
    let rd_x_next_tile_src = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: rd_x_next_tile_src,
        lhs: Operand::Reg(rd_x_block_base),
        rhs: Operand::Reg(rd_k_next_x_off64),
        ty: PtxType::U64,
    }));
    let r_k_next_rows = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_k_next_rows,
        lhs: Operand::Reg(r_k_tile_plus_1),
        rhs: Operand::ImmU32(K_TILE_SHARED),
        ty: PtxType::U32,
    }));
    let rd_w_next_row_off = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
        dst: rd_w_next_row_off,
        lhs: Operand::Reg(r_k_next_rows),
        rhs: Operand::Reg(r_n_bytes),
        src_ty: PtxType::U32,
    }));
    let rd_w_q_next_tile_src = add_u64(
        &mut alloc,
        &mut kernel,
        rd_w_q_block_base,
        rd_w_next_row_off,
    );

    // Overlap load 1: W_Q(k_tile+1) into NEXT slot.
    emit_mw_load_tile_w_int8_16x32(
        &mut alloc,
        &mut kernel,
        rd_w_q_next_tile_src,
        r_next_w_base,
        r_flat_tid,
        r_block_col,
        r_n,
        r_n_bytes,
        "Q_NEXT_OVERLAP",
    );
    // Overlap load 2: X(k_tile+1) into tile_x (safe because frag_A
    // was hoisted at K-tile start and B1 ensured all threads finished
    // reading tile_x before this point — INVARIANT #1).
    emit_mw_load_tile_x_f16_64x16(
        &mut alloc,
        &mut kernel,
        rd_x_next_tile_src,
        r_tile_x,
        r_flat_tid,
        r_block_row,
        r_m,
        r_k_bytes,
        "X_NEXT_OVERLAP",
    );
    // End of overlap-load block; label for @!p_not_last skip.
    kernel.push(PtxInstruction::Label(
        "OVERLAP_SKIP_QKV_INT8_SP_HALF_P".to_string(),
    ));

    // mma_V — runs unconditionally (after any overlap-load skip), reads
    // from CUR slot which holds W_V(k_tile) from Epoch 2.
    emit_warp_quadrant_mma_int8_per_projection_hoisted(
        &mut alloc,
        &mut kernel,
        r_cur_w_base,
        r_wq_col_idx,
        r_tid_x,
        &frag_as,
        &mut frag_c_v,
    );
    // B4 — closes the K-tile epoch: overlap loads visible (if issued),
    // mma_V done. Next K-tile's hoist+B1 sees tile_x = X(k+1) and
    // slot[(k+1)%2] = W_Q(k+1), restoring the K-tile-entry invariant.
    kernel.push(PtxInstruction::Control(ControlOp::BarSync {
        barrier_id: 0,
    }));

    // --- Loop back: k_tile += 1; if k_tile < num_k_tiles, jump ---
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
        target: "K_LOOP_QKV_INT8_SP_HALF_P".to_string(),
        negate: false,
    }));

    // --- Store-out epilogue ---
    // warp_block_row = block_row + r_wq_row_idx
    let r_warp_block_row = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_warp_block_row,
        lhs: Operand::Reg(r_block_row),
        rhs: Operand::Reg(r_wq_row_idx),
        ty: PtxType::U32,
    }));
    // warp_block_col = block_col + r_wq_col_idx
    let r_warp_block_col = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_warp_block_col,
        lhs: Operand::Reg(r_block_col),
        rhs: Operand::Reg(r_wq_col_idx),
        ty: PtxType::U32,
    }));
    // warp_row_off (u64) = warp_block_row * n_f16_stride
    let rd_warp_row_off = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
        dst: rd_warp_row_off,
        lhs: Operand::Reg(r_warp_block_row),
        rhs: Operand::Reg(r_n_f16_stride),
        src_ty: PtxType::U32,
    }));
    // warp_col_off (u64) = warp_block_col * 2
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

    // Per-projection warp base = rd_P_out + warp_row_off + warp_col_off (3 chains).
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

    // r_16_rows_bytes = BM * n_f16_stride (precomputed for m_stripe=1 sub-tiles).
    let r_16_rows_bytes = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_16_rows_bytes,
        lhs: Operand::Reg(r_n_f16_stride),
        rhs: Operand::ImmU32(BM),
        ty: PtxType::U32,
    }));

    // 4 sub-tiles × 3 projections = 12 store-out helper calls. Each call emits
    // 2 packed b32 stores (top + r+8 row), 4 cvt.rn.f16.f32, 2 mov.b32 packs,
    // and (with scale=Some) 4 mul.f32. Totals: 24 st.global.u32, 48 cvt, 24 packs,
    // 48 mul.f32 across the full epilogue.
    let projections = [
        (rd_q_warp_base, r_scale_q, &frag_c_q),
        (rd_k_warp_base, r_scale_k, &frag_c_k),
        (rd_v_warp_base, r_scale_v, &frag_c_v),
    ];
    for m_stripe in 0..MMAS_PER_WARP_M {
        for n_stripe in 0..MMAS_PER_WARP_N {
            // sub_off_bytes = m_stripe * (BM * n_f16_stride) + n_stripe * BN * 2
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
            for (warp_base, scale, grid) in &projections {
                let rd_subtile_base = alloc.alloc(PtxType::U64);
                kernel.push(PtxInstruction::Arith(ArithOp::Add {
                    dst: rd_subtile_base,
                    lhs: Operand::Reg(*warp_base),
                    rhs: Operand::Reg(rd_sub_off),
                    ty: PtxType::U64,
                }));
                crate::store_out::emit_store_fragment_c_f32_to_f16_packed(
                    &mut alloc,
                    &mut kernel,
                    &grid[m_stripe as usize][n_stripe as usize].regs,
                    rd_subtile_base,
                    r_n_f16_stride,
                    r_tid_x,
                    Some(*scale),
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

/// Validate shape + alignment preconditions for `qkv_project_int8`.
///
/// Sprint 7.3 D1. Enforces the **W8A16 MHA** contract:
///
/// - `M`, `N`, `K` all non-zero.
/// - `K % K_TILE_SHARED == 0` — mma K-tile is structural, no ragged tail inside a K-tile.
/// - `N % 2 == 0` — store-out packs adjacent f16 output pairs into one `.b32`.
/// - `N_q == N_k == N_v == N` — v1 is strict MHA. Grouped-query attention (GQA),
///   where `N_kv < N_q`, is a follow-up op (`qkv_project_gqa`); users with GQA
///   weights should call three separate `matmul_int{4,8}`s.
/// - Buffer-size sanity:
///   - `x >= M * K`
///   - each of `w_q_i8`, `w_k_i8`, `w_v_i8` >= `K * N`
///   - each of `q_out`, `k_out`, `v_out` >= `M * N`
///
/// `M` and `N` may be any positive value — edge-tile predication in the
/// kernel handles ragged output (same posture as `matmul_int4` / `matmul_int8`).
///
/// # Pointer distinctness
///
/// `q_out`, `k_out`, `v_out` must be **three distinct allocations**.
/// Aliasing (two outputs pointing into overlapping device memory) would
/// cause silent data corruption — the tri-output store epilogue writes
/// all three banks unconditionally, and overlapping writes race.
/// Rust's `&mut` borrow rules prevent accidental aliasing at the
/// variable level, and KAIO does not expose buffer-splitting APIs that
/// could produce overlapping `GpuBuffer` views, so this is a
/// pathological-caller guard enforced at the documentation level; a
/// cudarc-level device-ptr check is deferred (requires a `CudaStream`
/// argument, which validate is meant to run without).
#[allow(dead_code)] // wired up in D4
pub(crate) fn validate_dims_qkv_int8(
    x: &GpuBuffer<half::f16>,
    w_q_i8: &GpuBuffer<i8>,
    w_k_i8: &GpuBuffer<i8>,
    w_v_i8: &GpuBuffer<i8>,
    q_out: &GpuBuffer<half::f16>,
    k_out: &GpuBuffer<half::f16>,
    v_out: &GpuBuffer<half::f16>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    if m == 0 || n == 0 || k == 0 {
        return Err(KaioError::InvalidConfig(
            "qkv_project_int8: M, N, K dimensions must be non-zero".to_string(),
        ));
    }
    if !k.is_multiple_of(K_TILE_SHARED) {
        return Err(KaioError::InvalidConfig(format!(
            "qkv_project_int8: K must be a multiple of K_TILE_SHARED={K_TILE_SHARED} \
             (got {k}). The mma.sync.m16n8k16 instance shape requires K-tile size 16; \
             K is not edge-padded inside a K-tile."
        )));
    }
    if !n.is_multiple_of(2) {
        return Err(KaioError::InvalidConfig(format!(
            "qkv_project_int8: N must be even (got {n}). The store-out path packs \
             adjacent f16 output pairs into one .b32; odd N would leave a ragged \
             last column that the current store epilogue does not handle."
        )));
    }

    let mk = (m as usize) * (k as usize);
    let kn = (k as usize) * (n as usize);
    let mn = (m as usize) * (n as usize);

    if x.len() < mk {
        return Err(KaioError::InvalidConfig(format!(
            "qkv_project_int8: X buffer too small: need {mk} f16 ({m}×{k}), got {}",
            x.len()
        )));
    }
    for (label, buf) in [("W_Q", w_q_i8), ("W_K", w_k_i8), ("W_V", w_v_i8)] {
        if buf.len() < kn {
            return Err(KaioError::InvalidConfig(format!(
                "qkv_project_int8: {label} buffer too small: need {kn} i8 ({k}×{n}), got {}",
                buf.len()
            )));
        }
    }
    for (label, buf) in [("Q_out", q_out), ("K_out", k_out), ("V_out", v_out)] {
        if buf.len() < mn {
            return Err(KaioError::InvalidConfig(format!(
                "qkv_project_int8: {label} buffer too small: need {mn} f16 ({m}×{n}), got {}",
                buf.len()
            )));
        }
    }
    Ok(())
}

/// Fused tri-output INT8 QKV projection — **W8A16** (f16 activations,
/// INT8 weights, scalar per-projection scales).
///
/// Sprint 7.3 MVS ship deliverable. One kernel launch produces three
/// `GpuBuffer<f16>` outputs (Q, K, V) ready to feed
/// [`attention_tc`][crate::attention_tc] from a shared activation `x`
/// and three INT8 weight tensors. Saves 2× global activation reads vs
/// three separate [`matmul_int8`][crate::matmul_int8] calls and amortizes kernel-launch
/// overhead — dominant at autoregressive-decode batch sizes.
///
/// # W8A16 vs W8A8
///
/// **This is W8A16:** `f16` activations × `i8` weights, applying a
/// scalar per-projection scale at store-out via
/// `mma.sync.m16n8k16.f16.f16.f32` after per-weight `cvt.rn.f16.s8`.
/// [`matmul_int8`][crate::matmul_int8] (Sprint 7.1) is the **W8A8** (`i8` × `i8` →
/// `mma.sync.m16n8k32.s8.s8.s32`) reference op for users who genuinely
/// need int-only activations and manage their own activation
/// quantization. The Rust type system makes the distinction obvious at
/// compile time (`x: &GpuBuffer<half::f16>` here vs
/// `a: &GpuBuffer<i8>` for `matmul_int8`).
///
/// # Contract (v0.4.0 unreleased)
///
/// - **`x`**: `f16` row-major `[M, K]`.
/// - **`w_q`, `w_k`, `w_v`**: each `i8` row-major `[K, N]`. Three
///   distinct allocations (the store epilogue writes all three banks
///   unconditionally; aliasing two outputs is silent corruption).
/// - **`scale_q`, `scale_k`, `scale_v`**: scalar `f32`, one per
///   projection. Applied post-accumulation at store-out
///   (`out_P = (acc_P as f32) * scale_P` then `cvt.rn.f16.f32`).
/// - **`q_out`, `k_out`, `v_out`**: each `f16` row-major `[M, N]`.
///   **MHA**: `N_q == N_k == N_v == N` is enforced; grouped-query
///   attention (GQA) is a separate follow-up op (`qkv_project_gqa`).
///   Users with mismatched-N weights should call three separate
///   [`matmul_int8`][crate::matmul_int8] (W8A8) or compose three [`matmul_tc`][crate::matmul_tc] f16 ops.
/// - **`K % 16 == 0`** and **`N % 2 == 0`** required (validated
///   pre-launch). The store-out path packs adjacent f16 output pairs
///   into one `.b32`, so odd `N` would leave a ragged column the
///   epilogue does not handle.
///
/// # Hardware
///
/// Requires NVIDIA Ampere or newer (SM 8.0+) for the
/// `mma.sync.m16n8k16.f16.f16.f32` instance shape. Sub-Ampere targets
/// are rejected by `PtxModule::validate()` before driver dispatch.
///
/// # Out of scope
///
/// Bias / activation fusion (additive v2), GQA, async pipelining
/// (`cp.async`), auto-tuning, and the `kaio-candle` bridge — all are
/// follow-up sprints. See `docs/development/sprints/phase7/` for the
/// roadmap.
pub fn qkv_project_int8(
    device: &KaioDevice,
    x: &GpuBuffer<half::f16>,
    w_q: &GpuBuffer<i8>,
    w_k: &GpuBuffer<i8>,
    w_v: &GpuBuffer<i8>,
    scale_q: f32,
    scale_k: f32,
    scale_v: f32,
    q_out: &mut GpuBuffer<half::f16>,
    k_out: &mut GpuBuffer<half::f16>,
    v_out: &mut GpuBuffer<half::f16>,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};

    validate_dims_qkv_int8(x, w_q, w_k, w_v, q_out, k_out, v_out, m, n, k)?;

    let info = device.info()?;
    let (major, minor) = info.compute_capability;
    let sm = format!("sm_{major}{minor}");
    let module = build_qkv_project_int8_module(&sm);

    let kmodule = device.load_module(&module)?;
    let func = kmodule.function("qkv_project_int8")?;

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
            .arg(w_q.inner())
            .arg(w_k.inner())
            .arg(w_v.inner())
            .arg(&scale_q)
            .arg(&scale_k)
            .arg(&scale_v)
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
            return; // no GPU in CI host build — skip
        };
        let m = 64;
        let n = 64;
        let k = 128;
        let x = device.alloc_zeros::<half::f16>((m * k) as usize).unwrap();
        let w_q = device.alloc_zeros::<i8>((k * n) as usize).unwrap();
        let w_k = device.alloc_zeros::<i8>((k * n) as usize).unwrap();
        let w_v = device.alloc_zeros::<i8>((k * n) as usize).unwrap();
        let q_out = device.alloc_zeros::<half::f16>((m * n) as usize).unwrap();
        let k_out = device.alloc_zeros::<half::f16>((m * n) as usize).unwrap();
        let v_out = device.alloc_zeros::<half::f16>((m * n) as usize).unwrap();
        assert!(
            validate_dims_qkv_int8(&x, &w_q, &w_k, &w_v, &q_out, &k_out, &v_out, m, n, k).is_ok()
        );
    }

    #[test]
    fn validate_rejects_zero_dim() {
        let Ok(device) = make_device() else {
            return;
        };
        let x = device.alloc_zeros::<half::f16>(64).unwrap();
        let w = device.alloc_zeros::<i8>(64).unwrap();
        let o = device.alloc_zeros::<half::f16>(64).unwrap();
        let err = validate_dims_qkv_int8(&x, &w, &w, &w, &o, &o, &o, 0, 64, 16).unwrap_err();
        assert!(matches!(err, KaioError::InvalidConfig(_)));
    }

    #[test]
    fn validate_rejects_k_not_multiple_of_tile() {
        let Ok(device) = make_device() else {
            return;
        };
        let x = device.alloc_zeros::<half::f16>(1024).unwrap();
        let w = device.alloc_zeros::<i8>(1024).unwrap();
        let o = device.alloc_zeros::<half::f16>(1024).unwrap();
        // K=17 is not a multiple of K_TILE_SHARED=16
        let err = validate_dims_qkv_int8(&x, &w, &w, &w, &o, &o, &o, 16, 16, 17).unwrap_err();
        match err {
            KaioError::InvalidConfig(msg) => assert!(msg.contains("K_TILE_SHARED")),
            _ => panic!("expected InvalidConfig"),
        }
    }

    #[test]
    fn validate_rejects_odd_n() {
        let Ok(device) = make_device() else {
            return;
        };
        let x = device.alloc_zeros::<half::f16>(1024).unwrap();
        let w = device.alloc_zeros::<i8>(1024).unwrap();
        let o = device.alloc_zeros::<half::f16>(1024).unwrap();
        // N=7 is odd
        let err = validate_dims_qkv_int8(&x, &w, &w, &w, &o, &o, &o, 16, 7, 16).unwrap_err();
        match err {
            KaioError::InvalidConfig(msg) => assert!(msg.contains("N must be even")),
            _ => panic!("expected InvalidConfig"),
        }
    }

    #[test]
    fn validate_rejects_small_buffer() {
        let Ok(device) = make_device() else {
            return;
        };
        // Allocate smaller than M*K=64*32=2048; X undersized.
        let x = device.alloc_zeros::<half::f16>(128).unwrap();
        let w = device.alloc_zeros::<i8>(4096).unwrap();
        let o = device.alloc_zeros::<half::f16>(4096).unwrap();
        let err = validate_dims_qkv_int8(&x, &w, &w, &w, &o, &o, &o, 64, 64, 32).unwrap_err();
        match err {
            KaioError::InvalidConfig(msg) => assert!(msg.contains("X buffer too small")),
            _ => panic!("expected InvalidConfig"),
        }
    }

    #[test]
    fn tile_constants_are_self_consistent() {
        // Sanity: the derived MMAS_PER_WARP_* must multiply back to the warp quadrant.
        assert_eq!(MMAS_PER_WARP_M * BM, WARP_QUAD_M);
        assert_eq!(MMAS_PER_WARP_N * BN, WARP_QUAD_N);
        // 4 warps in a 2×2 grid cover the per-block output tile.
        assert_eq!(WARP_QUAD_M * 2, BM_BLOCK); // 2 warp rows × 32 = 64
        assert_eq!(WARP_QUAD_N * 2, BN_BLOCK); // 2 warp cols × 8 = 16 (Rollback #1)
        assert_eq!(THREADS_PER_BLOCK, 128);
        // Rollback #1: per-warp output tile is 32×8, MMAS_PER_WARP_N = 1.
        assert_eq!(MMAS_PER_WARP_N, 1);
        assert_eq!(WARP_QUAD_N, 8);
        assert_eq!(BN_BLOCK, 16);
    }

    #[test]
    fn tile_byte_constants_match_shape() {
        // X: 64 rows × 16 cols × 2 B per f16 = 2048 B.
        assert_eq!(TILE_X_BYTES, 2048);
        assert_eq!(TILE_X_ROW_STRIDE_BYTES, 32);
        // W: kept at 512 B (16 K-rows × 32 padded col-stride × 1 B per i8) so the
        // pre-zero pattern stays one-issue-per-thread after Rollback #1. Only the
        // first BN_BLOCK = 16 cols hold real data; the other 16 stay pre-zeroed.
        assert_eq!(TILE_W_BYTES, 512);
        assert_eq!(TILE_W_ROW_STRIDE_BYTES, 32);
        // Pre-zero requires both tiles to divide evenly into 512-byte issues.
        assert_eq!(TILE_X_BYTES % PRE_ZERO_BYTES_PER_ISSUE, 0);
        assert_eq!(TILE_W_BYTES % PRE_ZERO_BYTES_PER_ISSUE, 0);
        assert_eq!(TILE_X_BYTES / PRE_ZERO_BYTES_PER_ISSUE, 4); // 4 X-issues per thread
        assert_eq!(TILE_W_BYTES / PRE_ZERO_BYTES_PER_ISSUE, 1); // 1 W-issue per thread
    }

    // --- D3.1 cooperative-loader emit tests ----------------------------

    use kaio_core::emit::{Emit, PtxWriter};

    fn fresh_kernel() -> (RegisterAllocator, PtxKernel) {
        (
            RegisterAllocator::new(),
            PtxKernel::new("qkv_int8_d3_1_smoke"),
        )
    }

    fn emit_text(kernel: &PtxKernel) -> String {
        let mut w = PtxWriter::new();
        kernel.emit(&mut w).unwrap();
        w.finish()
    }

    #[test]
    fn pre_zero_emits_five_st_shared_u32_per_thread_plus_one_barrier() {
        // 4 issues for X (2048 B / 512 B per issue) + 1 for W = 5 per thread.
        let (mut alloc, mut kernel) = fresh_kernel();
        let tile_x = alloc.alloc(PtxType::U32);
        let tile_w = alloc.alloc(PtxType::U32);
        let flat_tid = alloc.alloc(PtxType::U32);
        emit_pre_zero_shared_tiles_qkv_int8(&mut alloc, &mut kernel, tile_x, tile_w, flat_tid);
        let ptx = emit_text(&kernel);
        assert_eq!(
            ptx.matches("st.shared.u32").count(),
            5,
            "expected 5 st.shared.u32 (4 X + 1 W); got:\n{ptx}"
        );
        assert_eq!(
            ptx.matches("bar.sync").count(),
            1,
            "expected 1 bar.sync at end of pre-zero; got:\n{ptx}"
        );
    }

    #[test]
    fn w_loader_emits_single_ld_global_and_single_st_shared() {
        let (mut alloc, mut kernel) = fresh_kernel();
        let w_base = alloc.alloc(PtxType::U64);
        let tile_w = alloc.alloc(PtxType::U32);
        let flat_tid = alloc.alloc(PtxType::U32);
        let block_col = alloc.alloc(PtxType::U32);
        let n = alloc.alloc(PtxType::U32);
        let n_bytes = alloc.alloc(PtxType::U32);
        emit_mw_load_tile_w_int8_16x32(
            &mut alloc,
            &mut kernel,
            w_base,
            tile_w,
            flat_tid,
            block_col,
            n,
            n_bytes,
            "Q",
        );
        let ptx = emit_text(&kernel);
        assert_eq!(
            ptx.matches("ld.global.u32").count(),
            1,
            "expected exactly 1 ld.global.u32 issue per thread (128 threads × 1 = full 512-B tile); got:\n{ptx}"
        );
        assert_eq!(
            ptx.matches("st.shared.u32").count(),
            1,
            "expected exactly 1 st.shared.u32 per thread; got:\n{ptx}"
        );
    }

    #[test]
    fn w_loader_emits_label_suffix_differentiation() {
        // Same kernel, three calls with Q / K / V suffixes — must not collide.
        let (mut alloc, mut kernel) = fresh_kernel();
        let w_base = alloc.alloc(PtxType::U64);
        let tile_w = alloc.alloc(PtxType::U32);
        let flat_tid = alloc.alloc(PtxType::U32);
        let block_col = alloc.alloc(PtxType::U32);
        let n = alloc.alloc(PtxType::U32);
        let n_bytes = alloc.alloc(PtxType::U32);
        for suffix in ["Q", "K", "V"] {
            emit_mw_load_tile_w_int8_16x32(
                &mut alloc,
                &mut kernel,
                w_base,
                tile_w,
                flat_tid,
                block_col,
                n,
                n_bytes,
                suffix,
            );
        }
        let ptx = emit_text(&kernel);
        for label in [
            "W_SKIP_I8_TILE_LOAD_Q:",
            "W_SKIP_I8_TILE_LOAD_K:",
            "W_SKIP_I8_TILE_LOAD_V:",
        ] {
            assert!(
                ptx.contains(label),
                "expected labeled skip target `{label}` in tri-call smoke; got:\n{ptx}"
            );
        }
    }

    #[test]
    fn w_loader_emits_bounds_gated_predicate_branch() {
        // The 4-byte load is gated on `block_col + col_start + 3 < N`.
        // PTX should emit a set-predicate + branch-negated-predicate pair.
        let (mut alloc, mut kernel) = fresh_kernel();
        let w_base = alloc.alloc(PtxType::U64);
        let tile_w = alloc.alloc(PtxType::U32);
        let flat_tid = alloc.alloc(PtxType::U32);
        let block_col = alloc.alloc(PtxType::U32);
        let n = alloc.alloc(PtxType::U32);
        let n_bytes = alloc.alloc(PtxType::U32);
        emit_mw_load_tile_w_int8_16x32(
            &mut alloc,
            &mut kernel,
            w_base,
            tile_w,
            flat_tid,
            block_col,
            n,
            n_bytes,
            "",
        );
        let ptx = emit_text(&kernel);
        // One setp.lt.u32 for the in-bounds check + one @!p bra to the skip.
        assert!(
            ptx.contains("setp.lt.u32"),
            "expected setp.lt.u32 for N-edge predicate; got:\n{ptx}"
        );
        assert!(
            ptx.contains("@!") && ptx.contains("bra W_SKIP_I8_TILE_LOAD"),
            "expected negated-predicate branch to the skip label; got:\n{ptx}"
        );
    }

    #[test]
    fn frag_b_int8_emits_four_cvt_f16_s8_and_two_mov_b32_packs() {
        // Per-lane chain: 4 × ld.shared.s8 → 4 × cvt.rn.f16.s8 → 2 × mov.b32 pack.
        // Plus address math: 4 × (mul row_stride) + 4 × (add col_abs) + 4 × (add base).
        let (mut alloc, mut kernel) = fresh_kernel();
        let tile_w = alloc.alloc(PtxType::U32);
        let n_stripe = alloc.alloc(PtxType::U32);
        let lane = alloc.alloc(PtxType::U32);
        let _frag = emit_fragment_b_int8_per_lane(&mut alloc, &mut kernel, tile_w, n_stripe, lane);
        let ptx = emit_text(&kernel);
        assert_eq!(
            ptx.matches("ld.shared.s8").count(),
            4,
            "expected 4 ld.shared.s8 per lane (2*tig, +1, +8, +9); got:\n{ptx}"
        );
        assert_eq!(
            ptx.matches("cvt.rn.f16.s8").count(),
            4,
            "expected 4 cvt.rn.f16.s8 per lane (one per loaded byte); got:\n{ptx}"
        );
        assert_eq!(
            ptx.matches("mov.b32").count(),
            2,
            "expected 2 mov.b32 packed pairs per lane (reg[0] and reg[1]); got:\n{ptx}"
        );
    }

    #[test]
    fn frag_b_int8_emits_no_mul_or_group_scale_load() {
        // INT8 W8A16 does NOT fold a scale into fragment B (scalar scale is
        // applied post-accumulation at store-out). There should be zero
        // multiplications and zero ld.shared.f16 reads in the emitted chain.
        let (mut alloc, mut kernel) = fresh_kernel();
        let tile_w = alloc.alloc(PtxType::U32);
        let n_stripe = alloc.alloc(PtxType::U32);
        let lane = alloc.alloc(PtxType::U32);
        let _ = emit_fragment_b_int8_per_lane(&mut alloc, &mut kernel, tile_w, n_stripe, lane);
        let ptx = emit_text(&kernel);
        // We expect mul.u32 for address math (row * stride, tig * 2) but
        // zero mul.f16 / mul.f32 (no scale fold).
        assert_eq!(
            ptx.matches("mul.f16").count(),
            0,
            "expected no mul.f16 (no pre-mma scale fold in INT8); got:\n{ptx}"
        );
        assert_eq!(
            ptx.matches("mul.f32").count(),
            0,
            "expected no mul.f32 (no pre-mma scale fold in INT8); got:\n{ptx}"
        );
        // And no ld.shared.b16 (no group-scale load — would be the f16 idiom).
        assert_eq!(
            ptx.matches("ld.shared.b16").count(),
            0,
            "expected no ld.shared.b16 (no group scale read in INT8); got:\n{ptx}"
        );
    }

    // --- D3.3 warp-quadrant mma sweep emit tests ----------------------

    /// Build a FragmentC grid by allocating fresh accumulator fragments.
    /// Returns the 2×2 grid so tests can pass it into the warp-quadrant
    /// helper and inspect the emitted mma destinations.
    fn fresh_frag_c_grid(
        alloc: &mut RegisterAllocator,
    ) -> [[kaio_core::fragment::FragmentC; MMAS_PER_WARP_N as usize]; MMAS_PER_WARP_M as usize]
    {
        core::array::from_fn(|_| core::array::from_fn(|_| kaio_core::fragment::alloc_c(alloc)))
    }

    #[test]
    fn warp_quad_mma_emits_two_mmas_one_frag_b_dequant() {
        // After Rollback #1: MMAS_PER_WARP_M × MMAS_PER_WARP_N = 2×1 = 2 mma.sync
        // per per-projection call. Fragment B is computed once per n-stripe
        // (1 n-stripe) and reused across the 2 m-stripes, so 1 dequant
        // chain per call → 1 × 4 = 4 cvt.rn.f16.s8.
        let (mut alloc, mut kernel) = fresh_kernel();
        let tile_x = alloc.alloc(PtxType::U32);
        let tile_w = alloc.alloc(PtxType::U32);
        let warp_row = alloc.alloc(PtxType::U32);
        let warp_col = alloc.alloc(PtxType::U32);
        let tid = alloc.alloc(PtxType::U32);
        let mut frag_c = fresh_frag_c_grid(&mut alloc);
        emit_warp_quadrant_mma_int8_per_projection(
            &mut alloc,
            &mut kernel,
            tile_x,
            tile_w,
            warp_row,
            warp_col,
            tid,
            &mut frag_c,
        );
        let ptx = emit_text(&kernel);
        assert_eq!(
            ptx.matches("mma.sync").count(),
            2,
            "expected 2 mma.sync per warp-quadrant projection (2×1 after Rollback #1); got:\n{ptx}"
        );
        assert_eq!(
            ptx.matches("cvt.rn.f16.s8").count(),
            4,
            "expected 4 cvt.rn.f16.s8 (1 n-stripe × 4 per dequant); got:\n{ptx}"
        );
        assert_eq!(
            ptx.matches("ld.shared.s8").count(),
            4,
            "expected 4 ld.shared.s8 (1 n-stripe × 4 per dequant); got:\n{ptx}"
        );
    }

    #[test]
    fn warp_quad_mma_emits_f32_copy_back_per_mma() {
        // After each mma, the per-lane frag_d registers (4 f32) are copied
        // back into frag_c via mov.f32 × 4. 2 mmas × 4 regs = 8 mov.f32
        // copy-backs per per-projection call (Rollback #1).
        let (mut alloc, mut kernel) = fresh_kernel();
        let tile_x = alloc.alloc(PtxType::U32);
        let tile_w = alloc.alloc(PtxType::U32);
        let warp_row = alloc.alloc(PtxType::U32);
        let warp_col = alloc.alloc(PtxType::U32);
        let tid = alloc.alloc(PtxType::U32);
        let mut frag_c = fresh_frag_c_grid(&mut alloc);
        emit_warp_quadrant_mma_int8_per_projection(
            &mut alloc,
            &mut kernel,
            tile_x,
            tile_w,
            warp_row,
            warp_col,
            tid,
            &mut frag_c,
        );
        let ptx = emit_text(&kernel);
        assert_eq!(
            ptx.matches("mov.f32").count(),
            8,
            "expected 8 mov.f32 (2 mmas × 4 copy-backs after Rollback #1); got:\n{ptx}"
        );
    }

    #[test]
    fn warp_quad_mma_uses_m16n8k16_f16_f32_shape() {
        // The mma.sync mnemonic should carry the full m16n8k16 signature
        // with f16.f16.f32 operand types.
        let (mut alloc, mut kernel) = fresh_kernel();
        let tile_x = alloc.alloc(PtxType::U32);
        let tile_w = alloc.alloc(PtxType::U32);
        let warp_row = alloc.alloc(PtxType::U32);
        let warp_col = alloc.alloc(PtxType::U32);
        let tid = alloc.alloc(PtxType::U32);
        let mut frag_c = fresh_frag_c_grid(&mut alloc);
        emit_warp_quadrant_mma_int8_per_projection(
            &mut alloc,
            &mut kernel,
            tile_x,
            tile_w,
            warp_row,
            warp_col,
            tid,
            &mut frag_c,
        );
        let ptx = emit_text(&kernel);
        assert!(
            ptx.contains("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"),
            "expected fully-qualified mma mnemonic for f32.f16.f16.f32; got:\n{ptx}"
        );
    }

    #[test]
    fn warp_quad_mma_preserves_frag_c_identity_in_place_accumulation() {
        // Disjoint-register canary at the per-projection level: the 16 f32
        // regs making up the frag_c_grid before the call must all appear as
        // mov.f32 *destinations* after the call (4 per mma × 4 mmas = 16),
        // proving the accumulation wrote back into the caller's grid.
        let (mut alloc, mut kernel) = fresh_kernel();
        let tile_x = alloc.alloc(PtxType::U32);
        let tile_w = alloc.alloc(PtxType::U32);
        let warp_row = alloc.alloc(PtxType::U32);
        let warp_col = alloc.alloc(PtxType::U32);
        let tid = alloc.alloc(PtxType::U32);
        let mut frag_c = fresh_frag_c_grid(&mut alloc);
        // Collect the frag_c_grid register IDs before emit.
        let frag_c_reg_ids: Vec<u32> = frag_c
            .iter()
            .flatten()
            .flat_map(|f| f.regs.iter().map(|r| r.index))
            .collect();
        // Rollback #1: 2 m-stripes × 1 n-stripe × 4 regs = 8 regs per grid.
        assert_eq!(
            frag_c_reg_ids.len(),
            (MMAS_PER_WARP_M * MMAS_PER_WARP_N * 4) as usize,
            "grid should hold MMAS_PER_WARP_M × MMAS_PER_WARP_N × 4 regs"
        );

        emit_warp_quadrant_mma_int8_per_projection(
            &mut alloc,
            &mut kernel,
            tile_x,
            tile_w,
            warp_row,
            warp_col,
            tid,
            &mut frag_c,
        );
        let ptx = emit_text(&kernel);
        // Each frag_c register must appear as a `mov.f32 %fN,` destination.
        for reg_idx in &frag_c_reg_ids {
            let needle = format!("mov.f32 %f{reg_idx},");
            assert!(
                ptx.contains(&needle),
                "expected copy-back into frag_c reg %f{reg_idx} but did not find `{needle}` in emitted PTX:\n{ptx}"
            );
        }
    }

    #[test]
    fn frag_b_int8_emits_tig_based_k_row_indexing() {
        // Fragment-B layout requires K-row offsets of `2*tig`, `2*tig+1`,
        // `2*tig+8`, `2*tig+9`. PTX address math should contain literal
        // +1, +8, +9 immediates in its computation of the per-byte k_row
        // values. (+8 comes from the `two_tig + 8` constant; +1 and the
        // implicit +0 handle the adjacent-K pairs inside reg[0] and reg[1].)
        let (mut alloc, mut kernel) = fresh_kernel();
        let tile_w = alloc.alloc(PtxType::U32);
        let n_stripe = alloc.alloc(PtxType::U32);
        let lane = alloc.alloc(PtxType::U32);
        let _ = emit_fragment_b_int8_per_lane(&mut alloc, &mut kernel, tile_w, n_stripe, lane);
        let ptx = emit_text(&kernel);
        // Constant +8 appears when computing `two_tig + 8`.
        assert!(
            ptx.contains(", 8;"),
            "expected immediate +8 in K-row address math; got:\n{ptx}"
        );
        // Row stride multiplier (32) — TILE_W_ROW_STRIDE_BYTES.
        assert!(
            ptx.contains(", 32;"),
            "expected row stride 32 in shared address math; got:\n{ptx}"
        );
    }

    // --- D3.4 full module tests --------------------------------------

    fn emit_module_to_string(module: &kaio_core::ir::PtxModule) -> String {
        let mut w = PtxWriter::new();
        module.emit(&mut w).unwrap();
        w.finish()
    }

    #[test]
    fn build_qkv_project_int8_module_structure() {
        let module = build_qkv_project_int8_module("sm_89");
        let ptx = emit_module_to_string(&module);

        // Entry name + 13 params + 4 shared decls (Sprint 7.3.5 S+½P).
        assert!(ptx.contains(".visible .entry qkv_project_int8("));
        assert!(ptx.contains(".shared .align 4 .b8 tile_x[2048]"));
        assert!(ptx.contains(".shared .align 4 .b8 tile_w_slot0[512]"));
        assert!(ptx.contains(".shared .align 4 .b8 tile_w_pad[64]"));
        assert!(ptx.contains(".shared .align 4 .b8 tile_w_slot1[512]"));

        // K-loop label + branch (S+½P variant).
        assert!(ptx.contains("K_LOOP_QKV_INT8_SP_HALF_P:"));
        assert!(ptx.contains("bra K_LOOP_QKV_INT8_SP_HALF_P"));
        // Overlap-skip label for the "if not last K-tile" predicate.
        assert!(ptx.contains("OVERLAP_SKIP_QKV_INT8_SP_HALF_P:"));

        // After Rollback #1: 6 mma.sync per K-tile (2 sub-tiles per warp × 3
        // projections), unified m16n8k16.f16.f16.f32 mnemonic (INT8 lives in
        // the per-warp dequant chain; the mma feed itself is f16).
        let sub_tiles_per_warp = (MMAS_PER_WARP_M * MMAS_PER_WARP_N) as usize;
        let projections = 3usize;
        let total_sub_tiles = sub_tiles_per_warp * projections;

        assert!(ptx.contains("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"));
        let mma_count = ptx.matches("mma.sync.aligned.m16n8k16").count();
        assert_eq!(
            mma_count, total_sub_tiles,
            "expected {total_sub_tiles} mma.sync ({sub_tiles_per_warp} sub-tiles × 3 projections)"
        );

        // Each store-out helper call → 2 packed b32 stores.
        let st_count = ptx.matches("st.global.u32").count();
        assert_eq!(
            st_count,
            total_sub_tiles * 2,
            "expected {} packed-f16 stores",
            total_sub_tiles * 2
        );
        // Each store-out helper call → 4 cvt.rn.f16.f32 (one per frag_c reg).
        let cvt_count = ptx.matches("cvt.rn.f16.f32").count();
        assert_eq!(
            cvt_count,
            total_sub_tiles * 4,
            "expected {} cvt.rn.f16.f32",
            total_sub_tiles * 4
        );
        // Each store-out helper call (scale=Some) → 4 mul.f32.
        let mul_count = ptx.matches("mul.f32").count();
        assert_eq!(
            mul_count,
            total_sub_tiles * 4,
            "expected {} mul.f32 from scale=Some across 3 projections",
            total_sub_tiles * 4
        );
    }

    #[test]
    fn build_qkv_project_int8_module_emits_four_barriers_per_k_tile_sp_half_p() {
        // Sprint 7.3.5 S+½P barrier cadence: 1 pre-zero + 1 pre-load
        // (B0) setup + 4 per K-tile (B1: post-hoist / INV #1; B2: post-
        // mma_Q; B3: post-mma_K; B4: post-mma_V + overlap loads).
        // Total static bar.sync in the emitted module for 1 K-tile =
        // 2 setup + 4 = 6 (vs Design S's 1 + 7 = 8). Savings scale with
        // K: per K-tile we drop 3 barriers (7 → 4).
        let module = build_qkv_project_int8_module("sm_89");
        let ptx = emit_module_to_string(&module);
        let bar_count = ptx.matches("bar.sync").count();
        assert_eq!(
            bar_count, 6,
            "expected 6 bar.sync (1 pre-zero + 1 B0 + 4 per K-tile, S+½P cadence); got {bar_count}"
        );
    }

    #[test]
    fn build_qkv_project_int8_module_declares_requested_sm() {
        let ptx_70 = emit_module_to_string(&build_qkv_project_int8_module("sm_70"));
        assert!(ptx_70.contains(".target sm_70"));
        let ptx_89 = emit_module_to_string(&build_qkv_project_int8_module("sm_89"));
        assert!(ptx_89.contains(".target sm_89"));
    }

    #[test]
    fn build_qkv_project_int8_module_validates_at_sm_80_and_above() {
        for sm in ["sm_80", "sm_89", "sm_90"] {
            let module = build_qkv_project_int8_module(sm);
            module
                .validate()
                .unwrap_or_else(|e| panic!("{sm} should validate; got {e}"));
        }
    }

    #[test]
    fn build_qkv_project_int8_module_rejects_sm_70_via_validate() {
        use kaio_core::ir::ValidationError;
        let module = build_qkv_project_int8_module("sm_70");
        let err = module.validate().unwrap_err();
        match err {
            ValidationError::SmTooLow {
                required,
                actual,
                feature,
            } => {
                assert_eq!(required, 80);
                assert_eq!(actual, 70);
                assert!(feature.contains("mma.sync.m16n8k16"));
            }
            other => panic!("expected SmTooLow, got {other:?}"),
        }
    }

    #[test]
    fn build_qkv_project_int8_module_emits_distinct_mma_destination_regs() {
        // Disjoint-register canary across the 3 frag_c grids: each mma.sync
        // instruction has a 4-register destination set; the union across all
        // grids and sub-tiles must contain no duplicates (= grids do not
        // alias). Catches the class of tri-output bugs where a buggy
        // alloc_grid pattern would reuse fragment-C register IDs across
        // projections, silently merging accumulators.
        use std::collections::HashSet;
        let expected = (MMAS_PER_WARP_M * MMAS_PER_WARP_N * 3 * 4) as usize;
        let module = build_qkv_project_int8_module("sm_89");
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
            let dst_str = &line[open + 1..open + close];
            for tok in dst_str.split(',') {
                let tok = tok.trim();
                if let Some(rest) = tok.strip_prefix("%f")
                    && let Ok(idx) = rest.parse::<u32>()
                {
                    all_dst_regs.push(idx);
                }
            }
        }
        assert_eq!(
            all_dst_regs.len(),
            expected,
            "expected {expected} mma destination regs (sub-tiles × 3 grids × 4 dst); got {}\n{:?}",
            all_dst_regs.len(),
            all_dst_regs
        );
        let unique: HashSet<u32> = all_dst_regs.iter().copied().collect();
        assert_eq!(
            unique.len(),
            expected,
            "expected {expected} DISTINCT mma dst regs; got {} (collision = grids alias)",
            unique.len()
        );
    }

    /// ptxas_verify offline gate for the full `qkv_project_int8` module.
    ///
    /// `#[ignore]` so host-only CI runs (no CUDA toolchain) stay green —
    /// invoke via `cargo test -- --ignored` on a machine with `ptxas` in
    /// PATH. Target SM overridable via `KAIO_SM_TARGET` env var; default
    /// runs both `sm_80` (compat floor) and `sm_89` (production target).
    #[test]
    #[ignore]
    fn ptxas_verify_qkv_project_int8() {
        if std::process::Command::new("ptxas")
            .arg("--version")
            .output()
            .is_err()
        {
            eprintln!("NOTE: ptxas not found in PATH — skipping qkv_project_int8 verification");
            return;
        }

        let sms: Vec<String> = if let Ok(sm) = std::env::var("KAIO_SM_TARGET") {
            vec![sm]
        } else {
            vec!["sm_80".to_string(), "sm_89".to_string()]
        };

        for sm in &sms {
            let module = build_qkv_project_int8_module(sm);
            let ptx = emit_module_to_string(&module);
            let tmp = std::env::temp_dir().join(format!("kaio_qkv_project_int8_{sm}.ptx"));
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
                "ptxas FAILED for qkv_project_int8 ({sm}):\n\
                 stdout: {stdout}\nstderr: {stderr}\n\n\
                 === PTX (first 8000 chars) ===\n{}",
                &ptx[..ptx.len().min(8000)]
            );

            // Combine stdout + stderr — Windows + CUDA 12.8 routes info lines to
            // stdout. Parse register count + spill bytes; gate at ≤ 64 regs and
            // 0 spills (matches the D2.5 baseline contract).
            let combined = format!("{stdout}\n{stderr}");
            let regs = parse_after(&combined, "Used ", " registers");
            let spill_stores = parse_after(&combined, "bytes stack frame, ", " bytes spill stores");
            let spill_loads = parse_after(&combined, "spill stores, ", " bytes spill loads");
            let smem = parse_after(&combined, ", used 0 barriers, ", " bytes smem");

            eprintln!(
                "=== qkv_project_int8 ptxas baseline ({sm}) ===\n\
                 regs/thread  : {regs:?}\n\
                 spill stores : {spill_stores:?}\n\
                 spill loads  : {spill_loads:?}\n\
                 shared bytes : {smem:?} (target ≈ 2560 = 2048 X + 512 W)"
            );

            match regs {
                Some(n) => assert!(
                    n <= 64,
                    "qkv_project_int8 ({sm}) reports {n} registers > 64; activate plan \
                     Rollback #1 (drop MMAS_PER_WARP_N 2→1)"
                ),
                None => panic!(
                    "could not parse register count from ptxas output\nstdout:\n{stdout}\nstderr:\n{stderr}"
                ),
            }
            assert_eq!(
                spill_stores,
                Some(0),
                "qkv_project_int8 ({sm}) has spill stores"
            );
            // Diagnostic: re-run with `-maxrregcount 128` to confirm the
            // default register count is the *natural* allocation, not
            // ptxas hitting an implicit cap and silently spilling under
            // it. If the uncapped number differs from the default, that
            // signals ptxas was being conservative by default and could
            // have used more registers.
            std::fs::write(&tmp, &ptx).expect("re-write temp PTX for diag");
            let diag_output = std::process::Command::new("ptxas")
                .args(["--gpu-name", sm, "--verbose"])
                .args(["--maxrregcount", "128"])
                .arg(tmp.to_str().unwrap())
                .output()
                .expect("failed to run ptxas diag");
            let diag_stdout = String::from_utf8_lossy(&diag_output.stdout);
            let diag_stderr = String::from_utf8_lossy(&diag_output.stderr);
            let _ = std::fs::remove_file(&tmp);
            let diag_combined = format!("{diag_stdout}\n{diag_stderr}");
            let diag_regs = parse_after(&diag_combined, "Used ", " registers");
            eprintln!(
                "  diag (maxrregcount=128) : regs/thread {diag_regs:?}  \
                 (if equal to default, allocation is natural)"
            );

            assert_eq!(
                spill_loads,
                Some(0),
                "qkv_project_int8 ({sm}) has spill loads"
            );

            eprintln!("ptxas verification PASSED for qkv_project_int8 ({sm})");
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

    /// D4 host-API smoke test: launches `qkv_project_int8` with canonical
    /// shapes (block-divisible M=64, N=16, K=16). Verifies the kernel loads
    /// and launches without driver error and produces non-NaN outputs.
    /// Correctness vs reference is the job of the D7 e2e suite.
    /// Skips silently when no GPU is available (host-only CI builds).
    #[test]
    fn qkv_project_int8_launches_without_error() {
        let Ok(device) = make_device() else {
            return;
        };
        let m = BM_BLOCK;
        let n = BN_BLOCK;
        let k = K_TILE_SHARED;
        let x = device.alloc_zeros::<half::f16>((m * k) as usize).unwrap();
        let w_q = device.alloc_zeros::<i8>((k * n) as usize).unwrap();
        let w_k = device.alloc_zeros::<i8>((k * n) as usize).unwrap();
        let w_v = device.alloc_zeros::<i8>((k * n) as usize).unwrap();
        let mut q_out = device.alloc_zeros::<half::f16>((m * n) as usize).unwrap();
        let mut k_out = device.alloc_zeros::<half::f16>((m * n) as usize).unwrap();
        let mut v_out = device.alloc_zeros::<half::f16>((m * n) as usize).unwrap();
        qkv_project_int8(
            &device, &x, &w_q, &w_k, &w_v, 1.0, 1.0, 1.0, &mut q_out, &mut k_out, &mut v_out, m, n,
            k,
        )
        .expect("qkv_project_int8 launch failed on canonical shape");
        device.stream().synchronize().expect("device sync failed");
        // Zero inputs → zero outputs. Sanity check that the kernel actually
        // ran and wrote to the outputs (rather than silently no-oping).
        let q_host = q_out.to_host(&device).expect("dtoh");
        let zero = half::f16::from_f32(0.0);
        assert!(
            q_host.iter().all(|h: &half::f16| *h == zero),
            "expected all-zero Q output for zero inputs"
        );
    }

    #[test]
    fn x_loader_wrapper_delegates_and_emits_load_cycle() {
        // The wrapper re-uses matmul_int4's loader; confirm it emits at
        // least one global load + shared store pair (the matmul_int4
        // loader issues 4 per thread for the 64×16 tile).
        let (mut alloc, mut kernel) = fresh_kernel();
        let x_base = alloc.alloc(PtxType::U64);
        let tile_x = alloc.alloc(PtxType::U32);
        let flat_tid = alloc.alloc(PtxType::U32);
        let block_row = alloc.alloc(PtxType::U32);
        let m = alloc.alloc(PtxType::U32);
        let k_bytes = alloc.alloc(PtxType::U32);
        emit_mw_load_tile_x_f16_64x16(
            &mut alloc,
            &mut kernel,
            x_base,
            tile_x,
            flat_tid,
            block_row,
            m,
            k_bytes,
            "qkv",
        );
        let ptx = emit_text(&kernel);
        let ld = ptx.matches("ld.global.u32").count();
        let st = ptx.matches("st.shared.u32").count();
        // matmul_int4's A loader issues 4 b32 per thread (64×16 tile / 128 threads / 4 B).
        assert_eq!(
            ld, 4,
            "expected matmul_int4 A-loader delegation to emit 4 ld.global.u32; got {ld}\n{ptx}"
        );
        assert_eq!(
            st, 4,
            "expected matmul_int4 A-loader delegation to emit 4 st.shared.u32; got {st}\n{ptx}"
        );
    }
}
