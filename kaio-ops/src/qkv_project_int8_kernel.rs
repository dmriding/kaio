//! Fused tri-output INT8 QKV projection — W8A16 (f16 activations, INT8
//! weights, scalar per-projection scales).
//!
//! Sprint 7.3 MVS. Single kernel launch produces three f16 outputs
//! (Q, K, V) ready to feed [`attention_tc`][crate::attention_tc] — saves
//! 2× global reads of the shared activation X compared to three
//! separate [`matmul_int8`][crate::matmul_int8] calls, and amortizes
//! kernel launch overhead (dominant at autoregressive-decode batch
//! sizes).
//!
//! # W8A16 vs W8A8
//!
//! [`matmul_int8`][crate::matmul_int8] (Sprint 7.1) is **W8A8** (`i8`
//! activations × `i8` weights, native `mma.sync.m16n8k32.s8.s8.s32`).
//! `qkv_project_int8` is **W8A16** (`f16` activations × `i8` weights,
//! `mma.sync.m16n8k16.f16.f16.f32` after per-weight `cvt.f16.s8`).
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
//! (INT8: `cvt.f16.s8` + scalar scale; INT4: nibble-extract +
//! sign-extend + group-scale fold).
//!
//! # Tri-output design (Serial fusion, Design S)
//!
//! Per K-tile iteration: cooperative-load X tile once into shared,
//! then sequentially for each projection P ∈ {Q, K, V}: load W_P i8
//! tile into a shared weight slot, dequant fragment B per-warp,
//! `mma` into a per-projection f32 accumulator bank. Three fragment-C
//! banks persist across the entire K-loop; scalar scales are applied
//! at store-out, not pre-mma.
//!
//! # Register budget
//!
//! Tripled fragment-C (3 × 16 f32 regs per lane = 48) is the dominant
//! live state. `MMAS_PER_WARP_N` is halved from `matmul_int4`'s 4 to 2
//! to stay inside the 64-reg occupancy cliff on sm_89 — output tile
//! becomes 64×32 per block. D2.5 register-pressure skeleton checkpoint
//! runs ahead of the full kernel body to catch any surprise.

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

// --- Multi-warp block tiling (halved N vs matmul_int4/int8 to fit tri-output accumulators) ---
#[allow(dead_code)] // wired up in D3
pub(crate) const BM_BLOCK: u32 = 64; // output rows per block
#[allow(dead_code)] // wired up in D3
pub(crate) const BN_BLOCK: u32 = 32; // output cols per block — halved vs standalone (see register budget)
#[allow(dead_code)] // wired up in D3
pub(crate) const WARP_QUAD_M: u32 = 32; // rows per warp quadrant
#[allow(dead_code)] // wired up in D3
pub(crate) const WARP_QUAD_N: u32 = 16; // cols per warp quadrant
#[allow(dead_code)] // wired up in D3
pub(crate) const MMAS_PER_WARP_M: u32 = WARP_QUAD_M / BM; // 2
#[allow(dead_code)] // wired up in D3
pub(crate) const MMAS_PER_WARP_N: u32 = WARP_QUAD_N / BN; // 2
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

/// W_P_i8 tile in shared: `K_TILE_SHARED` rows × `BN_BLOCK` cols, i8 row-major.
/// 16 × 32 × 1 = 512 B. Row-major in shared — the fragment-B dequant helper
/// (D3.2) reads per-lane i8 values at `(k_row, n_col)` offsets and produces
/// `cvt.f16.s8` + `MovPack` packed pairs. We pay no padding for bank-
/// conflict relief because the per-lane loads are 1-byte reads with fanout
/// across lanes, not the dense-stripe pattern that causes 32-bank contention.
#[allow(dead_code)] // wired up in D3
pub(crate) const TILE_W_BYTES: u32 = K_TILE_SHARED * BN_BLOCK; // 512

/// W tile row stride in bytes (row-major i8).
#[allow(dead_code)] // wired up in D3
pub(crate) const TILE_W_ROW_STRIDE_BYTES: u32 = BN_BLOCK; // 32

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

    // Global offset = row_in_tile * n_bytes + col_global (bytes).
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
        rhs: Operand::Reg(col_global),
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
/// each to `f16` via `cvt.f16.s8`, and `MovPack`s them into 2 `.b32`
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
/// the 4 `cvt.f16.s8` and 2 `MovPack` instructions; 4 × 1-byte shared
/// loads add negligible marginal cost relative to the arithmetic.
#[allow(dead_code)] // wired up in D3.3
pub(crate) fn emit_fragment_b_int8_per_lane(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    tile_w_shared: Register,
    n_stripe_col_base: Register,
    lane_within_warp: Register,
) -> kaio_core::fragment::FragmentB {
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

    // Load one s8 byte from shared at (k_row, col_abs) and cvt.f16.s8
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
            // cvt.f16.s8 %h, %r — sign-extend-then-convert, per ISA.
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

    kaio_core::fragment::FragmentB { regs: [reg0, reg1] }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_device() -> Result<KaioDevice> {
        KaioDevice::new(0)
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
        // And 4 warps × 32×16 quadrant cover the 64×32 block.
        assert_eq!(WARP_QUAD_M * 2, BM_BLOCK); // 2 warp rows × 32 = 64
        assert_eq!(WARP_QUAD_N * 2, BN_BLOCK); // 2 warp cols × 16 = 32
        assert_eq!(THREADS_PER_BLOCK, 128);
    }

    #[test]
    fn tile_byte_constants_match_shape() {
        // X: 64 rows × 16 cols × 2 B per f16 = 2048 B.
        assert_eq!(TILE_X_BYTES, 2048);
        assert_eq!(TILE_X_ROW_STRIDE_BYTES, 32);
        // W: 16 K-rows × 32 N-cols × 1 B per i8 = 512 B.
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
        // Per-lane chain: 4 × ld.shared.s8 → 4 × cvt.f16.s8 → 2 × mov.b32 pack.
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
            ptx.matches("cvt.f16.s8").count(),
            4,
            "expected 4 cvt.f16.s8 per lane (one per loaded byte); got:\n{ptx}"
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
