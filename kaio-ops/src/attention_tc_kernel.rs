//! Tensor-core scaled dot-product attention — fused `attention_tc`.
//!
//! A single-warp-per-block **fused monolithic** kernel that computes
//! `out = softmax(Q·Kᵀ / √d_k) · V` with two back-to-back `mma.sync.m16n8k16`
//! matmuls and an intra-kernel `cvt.rn.f16.f32` bridge between the
//! softmax output and the second matmul's f16 inputs.
//!
//! # Sprint 6.6 intent (correctness-first, internal preview only)
//!
//! Mirrors Sprint 6.3's philosophy: correctness-first, single-warp-per-
//! block, deliberately slow at realistic sizes, narrow shape contract.
//! Performance targets (multi-warp restructure, lifting the seq_k cap)
//! belong to Sprint 6.7 and Phase 7 (FlashAttention-TC with online
//! softmax). This kernel's job is to **prove** the TC-attention fused
//! pipeline — specifically the f32-accumulator → softmax → f16-input
//! bridge, which is the load-bearing architectural contract that any
//! eventually-performant TC attention implementation must honor.
//!
//! # Narrow contract (temporary, lifted in Phase 7)
//!
//! - Types: f16 Q, K, V → f32 output. No bf16 variant yet.
//! - Hardware: NVIDIA Ampere or newer (SM 8.0+, for `mma.sync.m16n8k16`).
//!   Sub-Ampere devices fail cleanly at `device.load_module()` with
//!   `KaioError::Validation(SmTooLow)` — not a runtime check in the
//!   host API.
//! - Shape: `seq_q % 16 == 0`, `seq_k % 16 == 0`, `d_k % 16 == 0`,
//!   `d_v % 8 == 0`. `seq_k ≤ 384` (shared-memory ceiling — Phase 7's
//!   FlashAttention-TC eliminates the scores matrix and this goes away).
//!
//! # Kernel body structure (Gate C — final)
//!
//! - Grid: `(seq_q / 16, 1, 1)`. Each block owns a 16-row slab of the
//!   output and the full seq_k × d_v computation for those rows.
//! - Block: `(32, 1, 1)` — one warp per block (6.3 precedent).
//! - Per block:
//!   1. Stage Q_tile (16 × d_k, row-major f16) to shared once.
//!   2. Loop over seq_k in 8-col chunks: compute matmul1 output (16×8
//!      scores tile) via d_k/16 inner mma.sync iterations, scale by
//!      `1/√d_k`, store to scores_tile shared (f32).
//!   3. Row-wise softmax over scores_tile, 16 rows serialized, warp
//!      shuffle reductions (bfly). Write f16 probs to probs_tile.
//!   4. Loop over d_v in 8-col chunks: compute matmul2 output (16×8)
//!      via seq_k/16 inner mma.sync iterations reading probs_tile (as
//!      A) + V chunks (as B). Store directly to global out.
//!
//! # Shared-memory layout (internal contract — 6.7 must preserve)
//!
//! - `tile_q` — 16 × d_k fp16, **row-major**, row stride = d_k·2 bytes.
//!   Sized for max d_k = 128 (= 4096 bytes static).
//! - `scores_tile` — 16 × seq_k fp32, row-major, row stride = seq_k·4.
//!   Sized for max seq_k = 384 (= 24576 bytes static).
//! - `probs_tile` — 16 × seq_k fp16, row-major, row stride = seq_k·2.
//!   Sized for max seq_k = 384 (= 12288 bytes static).
//! - `k_chunk` — 16 × 8 fp16, **column-major**, column stride = 16·2.
//!   Per-mma-iteration, 256 bytes. Matches matmul_tc's tile_b shape.
//! - `v_chunk` — 16 × 8 fp16, **column-major**, column stride = 16·2.
//!   Per-mma-iteration, 256 bytes. Same shape as k_chunk.
//!
//! Worst-case shared usage (seq_k = 384, d_k = 128, d_v = 64):
//! 4096 + 24576 + 12288 + 256 + 256 = **41,472 B ≈ 40.5 KB** within
//! the 48 KB static ceiling. Budget regression-gated by
//! `attention_tc_module_shared_bytes_under_ceiling` in the test file.
//!
//! # Public surface
//!
//! `attention_tc` and `attention_tc_causal` are `#[doc(hidden)] pub use`
//! in `lib.rs` — internal preview until Phase 7 lifts the divisibility
//! + seq_k constraints and adds `attention_flash_tc` (at which point
//! `attention_auto_tc` becomes the real user-facing dispatcher).

// During Gate A only a subset of the module is live. `allow(dead_code)`
// marks symbols that Gates B/C/causal will consume — the warnings are
// noise during incremental development.
#![allow(dead_code)]

use half::f16;
use kaio::prelude::*;
use kaio_core::fragment::{alloc_c, load_fragment_b_m16n8k16_shared_col};
use kaio_core::instr::control::{CmpOp, ControlOp};
use kaio_core::instr::memory::MemoryOp;
use kaio_core::instr::special;
use kaio_core::instr::{ArithOp, MadMode, MmaShape, TensorCoreOp};
use kaio_core::ir::{
    Operand, PtxInstruction, PtxKernel, PtxModule, PtxParam, Register, RegisterAllocator,
    SharedDecl, SpecialReg,
};
use kaio_core::types::PtxType;

// ---------------------------------------------------------------------
// Compile-time constants (max shape upper bounds — sized shared decls)
// ---------------------------------------------------------------------

const BM: u32 = 16; // Q-tile rows per block; M of mma1 / M of mma2
const BN: u32 = 8; // inner output chunk cols; N of mma.sync m16n8k16
const BK: u32 = 16; // inner K-chunk size; K of mma.sync m16n8k16

const MAX_D_K: u32 = 128;
const MAX_SEQ_K: u32 = 384;
const MAX_D_V: u32 = 128;

const BYTES_PER_HALF: u32 = 2;
const BYTES_PER_F32: u32 = 4;

// Shared decl sizes — sized for worst-case shapes; the runtime loop
// bounds use the actual seq_k / d_k / d_v, so we only "use" the region
// corresponding to the real shape on each launch.
const TILE_Q_BYTES: u32 = BM * MAX_D_K * BYTES_PER_HALF; // 4096
const SCORES_TILE_BYTES: u32 = BM * MAX_SEQ_K * BYTES_PER_F32; // 24576
const PROBS_TILE_BYTES: u32 = BM * MAX_SEQ_K * BYTES_PER_HALF; // 12288
const K_CHUNK_BYTES: u32 = BK * BN * BYTES_PER_HALF; // 256
const V_CHUNK_BYTES: u32 = BK * BN * BYTES_PER_HALF; // 256

/// The publicly-guaranteed shared-memory budget ceiling. Used by the
/// budget regression test to catch future alignment-padding or layout
/// drift that could silently push us over the 48 KB static shared
/// memory limit. 46 KB = 48 KB minus a ~2 KB safety margin.
pub(crate) const SHARED_MEMORY_CEILING_BYTES: u32 = 46 * 1024;

/// Sum of declared shared bytes (without extra alignment padding) —
/// used by the budget regression test to assert we stay under the
/// ceiling. Does NOT account for per-decl alignment padding that the
/// PTX emitter adds; for the current align=4 decls, the emitted size
/// equals this sum. If future decls use align=16 (cp.async), adjust
/// the test to compute emitted size instead of trusting this sum.
pub(crate) const DECLARED_SHARED_BYTES: u32 =
    TILE_Q_BYTES + SCORES_TILE_BYTES + PROBS_TILE_BYTES + K_CHUNK_BYTES + V_CHUNK_BYTES;

const _: () = assert!(
    DECLARED_SHARED_BYTES <= SHARED_MEMORY_CEILING_BYTES,
    "attention_tc shared-memory budget exceeds ceiling at build time"
);

// ---------------------------------------------------------------------
// Validation — host-side dim checks mirroring scalar attention's path.
// SM check is NOT here; `PtxModule::validate()` inside `load_module`
// catches sub-Ampere cleanly via `ValidationError::SmTooLow`.
// ---------------------------------------------------------------------

/// Validate the shape + buffer-size contract for `attention_tc` /
/// `attention_tc_causal`. `pub(crate)` for the tuner-equivalent (none
/// yet — 6.6 has one variant; a 2-way `attention_auto_tc` arrives in
/// Phase 7 when FlashAttention-TC joins as a sibling candidate).
#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_attention_tc_dims(
    q: &GpuBuffer<f16>,
    k: &GpuBuffer<f16>,
    v: &GpuBuffer<f16>,
    out: &GpuBuffer<f32>,
    seq_q: u32,
    seq_k: u32,
    d_k: u32,
    d_v: u32,
) -> Result<()> {
    if seq_q == 0 || seq_k == 0 || d_k == 0 || d_v == 0 {
        return Err(KaioError::InvalidConfig(
            "attention_tc dimensions must be non-zero".to_string(),
        ));
    }
    let why_constrained = "Either pad inputs to valid shapes, or use the \
                           existing f32 `attention` / `attention_flash` path instead.";
    if !seq_q.is_multiple_of(BM) {
        return Err(KaioError::InvalidConfig(format!(
            "attention_tc: seq_q must be a multiple of {BM} (got {seq_q}). {why_constrained}"
        )));
    }
    if !seq_k.is_multiple_of(BK) {
        return Err(KaioError::InvalidConfig(format!(
            "attention_tc: seq_k must be a multiple of {BK} (got {seq_k}). {why_constrained}"
        )));
    }
    if !d_k.is_multiple_of(BK) {
        return Err(KaioError::InvalidConfig(format!(
            "attention_tc: d_k must be a multiple of {BK} (got {d_k}). {why_constrained}"
        )));
    }
    if !d_v.is_multiple_of(BN) {
        return Err(KaioError::InvalidConfig(format!(
            "attention_tc: d_v must be a multiple of {BN} (got {d_v}). {why_constrained}"
        )));
    }
    if seq_k > MAX_SEQ_K {
        return Err(KaioError::InvalidConfig(format!(
            "attention_tc: seq_k must be ≤ {MAX_SEQ_K} (got {seq_k}). \
             The shared-memory scores buffer caps the sequence length; \
             Phase 7's FlashAttention-TC lifts this constraint. {why_constrained}"
        )));
    }
    if d_k > MAX_D_K {
        return Err(KaioError::InvalidConfig(format!(
            "attention_tc: d_k must be ≤ {MAX_D_K} (got {d_k}). {why_constrained}"
        )));
    }
    if d_v > MAX_D_V {
        return Err(KaioError::InvalidConfig(format!(
            "attention_tc: d_v must be ≤ {MAX_D_V} (got {d_v}). {why_constrained}"
        )));
    }

    let qk_elems = (seq_q as usize) * (d_k as usize);
    let k_elems = (seq_k as usize) * (d_k as usize);
    let v_elems = (seq_k as usize) * (d_v as usize);
    let out_elems = (seq_q as usize) * (d_v as usize);
    if q.len() < qk_elems {
        return Err(KaioError::InvalidConfig(format!(
            "attention_tc: Q buffer too small: need {qk_elems} f16 ({seq_q}×{d_k}), got {}",
            q.len()
        )));
    }
    if k.len() < k_elems {
        return Err(KaioError::InvalidConfig(format!(
            "attention_tc: K buffer too small: need {k_elems} f16 ({seq_k}×{d_k}), got {}",
            k.len()
        )));
    }
    if v.len() < v_elems {
        return Err(KaioError::InvalidConfig(format!(
            "attention_tc: V buffer too small: need {v_elems} f16 ({seq_k}×{d_v}), got {}",
            v.len()
        )));
    }
    if out.len() < out_elems {
        return Err(KaioError::InvalidConfig(format!(
            "attention_tc: out buffer too small: need {out_elems} f32 ({seq_q}×{d_v}), got {}",
            out.len()
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Shared tile-staging helpers
// ---------------------------------------------------------------------
//
// These differ from matmul_tc's emit_load_*_tile in two ways:
//   1. row strides are **runtime-valued** (shared decls are sized to
//      max, but the loop bounds address only the active sub-region
//      based on the actual d_k / seq_k / d_v). Strides are passed as
//      registers, not constants.
//   2. Global source is the per-block Q row-slab or the per-iteration
//      K/V column-chunk — addressing differs from matmul_tc's per-K-
//      tile Q/K slabs.
//
// Layout contracts (match matmul_tc for fragment loader compatibility):
//   - tile_q: 16 rows × d_k cols fp16, row-major, row stride = d_k*2
//   - k_chunk / v_chunk: 16 rows × 8 cols fp16, column-major,
//                        column stride = 16*2 = 32 bytes

/// Stage a 16×d_k Q row-slab from global to the `tile_q` shared region.
///
/// Runs once per block (Q is reused across all seq_k iterations).
/// Distributes the 16×d_k halves across 32 warp lanes as half2 pairs.
/// Requires `d_k % 16 == 0` (validated at host level). For d_k=16:
/// 16 rows × 8 half2 = 128 half2 pairs = 4 per lane. For d_k=128:
/// 16 rows × 64 half2 = 1024 half2 pairs = 32 per lane — we emit an
/// explicit loop rather than unroll.
///
/// Per-thread iteration: lane `tid` iterates `flat = tid, tid+32,
/// tid+64, ...` over `[0, total_half2 = 16 * d_k / 2)`. For each flat:
/// `row = flat / half2_per_row`, `col_half2 = flat % half2_per_row`,
/// where `half2_per_row = d_k / 2`. Global byte offset within the Q
/// row-slab: `row * d_k * 2 + col_half2 * 4`. Shared byte offset in
/// tile_q: identical.
fn emit_stage_q(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    q_rowslab_src_global: Register, // u64
    tile_q_shared: Register,        // u32
    tid_x: Register,                // u32
    d_k: Register,                  // u32 runtime value
) {
    // half2_per_row = d_k / 2
    let r_half2_per_row = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: r_half2_per_row,
        lhs: Operand::Reg(d_k),
        rhs: Operand::ImmU32(2),
        ty: PtxType::U32,
    }));
    // total_half2 = 16 * half2_per_row  (= 8 * d_k)
    let r_total_half2 = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_total_half2,
        lhs: Operand::Reg(r_half2_per_row),
        rhs: Operand::ImmU32(BM),
        ty: PtxType::U32,
    }));
    // d_k_bytes = d_k * 2
    let r_d_k_bytes = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_d_k_bytes,
        lhs: Operand::Reg(d_k),
        rhs: Operand::ImmU32(BYTES_PER_HALF),
        ty: PtxType::U32,
    }));

    // Loop: flat = tid; while flat < total_half2 { ...; flat += 32 }
    let r_flat = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_flat,
        src: Operand::Reg(tid_x),
        ty: PtxType::U32,
    });
    kernel.push(PtxInstruction::Label("STAGE_Q_LOOP".to_string()));

    // Guard: if flat >= total_half2, exit.
    let p_done = alloc.alloc(PtxType::Pred);
    kernel.push(PtxInstruction::Control(ControlOp::SetP {
        dst: p_done,
        cmp_op: CmpOp::Ge,
        lhs: Operand::Reg(r_flat),
        rhs: Operand::Reg(r_total_half2),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Control(ControlOp::BraPred {
        pred: p_done,
        target: "STAGE_Q_EXIT".to_string(),
        negate: false,
    }));

    // row = flat / half2_per_row ; col_half2 = flat % half2_per_row
    let r_row = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: r_row,
        lhs: Operand::Reg(r_flat),
        rhs: Operand::Reg(r_half2_per_row),
        ty: PtxType::U32,
    }));
    let r_col_half2 = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Rem {
        dst: r_col_half2,
        lhs: Operand::Reg(r_flat),
        rhs: Operand::Reg(r_half2_per_row),
        ty: PtxType::U32,
    }));

    // col_bytes = col_half2 * 4
    let r_col_bytes = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_col_bytes,
        lhs: Operand::Reg(r_col_half2),
        rhs: Operand::ImmU32(4),
        ty: PtxType::U32,
    }));

    // shared_off = row * d_k_bytes + col_bytes
    let r_shared_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mad {
        dst: r_shared_off,
        a: Operand::Reg(r_row),
        b: Operand::Reg(r_d_k_bytes),
        c: Operand::Reg(r_col_bytes),
        ty: PtxType::U32,
        mode: MadMode::Lo,
    }));
    let r_shared_addr = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_shared_addr,
        lhs: Operand::Reg(tile_q_shared),
        rhs: Operand::Reg(r_shared_off),
        ty: PtxType::U32,
    }));

    // global_off_u64 = row * d_k_bytes + col_bytes  (upcast to u64)
    let rd_row_off = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
        dst: rd_row_off,
        lhs: Operand::Reg(r_row),
        rhs: Operand::Reg(r_d_k_bytes),
        src_ty: PtxType::U32,
    }));
    let rd_col_bytes = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Cvt {
        dst: rd_col_bytes,
        src: r_col_bytes,
        dst_ty: PtxType::U64,
        src_ty: PtxType::U32,
    });
    let rd_per_thread_off = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: rd_per_thread_off,
        lhs: Operand::Reg(rd_row_off),
        rhs: Operand::Reg(rd_col_bytes),
        ty: PtxType::U64,
    }));
    let rd_global_addr = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: rd_global_addr,
        lhs: Operand::Reg(q_rowslab_src_global),
        rhs: Operand::Reg(rd_per_thread_off),
        ty: PtxType::U64,
    }));

    // Load half2 pair (b32) and store to shared.
    let r_tmp = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdGlobal {
        dst: r_tmp,
        addr: rd_global_addr,
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Memory(MemoryOp::StShared {
        addr: r_shared_addr,
        src: r_tmp,
        ty: PtxType::U32,
    }));

    // flat += 32
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_flat,
        lhs: Operand::Reg(r_flat),
        rhs: Operand::ImmU32(32),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Control(ControlOp::Bra {
        target: "STAGE_Q_LOOP".to_string(),
    }));
    kernel.push(PtxInstruction::Label("STAGE_Q_EXIT".to_string()));
}

/// Stage a 16×8 K-chunk from global to the `k_chunk` shared region,
/// transposing row-major global → column-major shared in-flight.
///
/// The K-chunk covers K rows `[k_chunk_idx*16, k_chunk_idx*16 + 16)`
/// × 8 cols `[n_chunk_idx*8, n_chunk_idx*8 + 8)` of the d_k × seq_k
/// matrix — wait, that's not right. Let me restate:
///
/// For matmul1 Q·Kᵀ with shape M=seq_q, N=seq_k, K=d_k: mma expects
/// A-fragment shape M×K = 16×16 (A is Q) and B-fragment shape K×N =
/// 16×8 (B is Kᵀ). Kᵀ[i,j] = K[j,i]. B-tile is 16 K-rows × 8 N-cols,
/// where "K-row" corresponds to d_k dimension and "N-col" to seq_k.
/// So from K (seq_k × d_k row-major), we need K[n_start:n_start+8,
/// k_start:k_start+16], transposed to become B[16 d_k rows × 8 seq_k
/// cols], then stored column-major in shared.
///
/// Per-thread: `flat = tid` iterates over 16×8 / 2 = 64 half2 pairs
/// (each thread handles 2 pairs via unroll). Flat addresses into the
/// d_k×seq_k "logical B tile": `d_row = flat / 4`, `s_col_half2 =
/// flat % 4`. Actual K source: `k_row = n_start + 2*s_col_half2{,+1}`,
/// `k_col = k_start + d_row`. Since two consecutive f16 values in the
/// half2 destination are from non-adjacent K rows (transpose), we can
/// NOT use a packed half2 load — must do per-fp16 loads, same as
/// matmul_tc's emit_load_b_tile.
///
/// Simplification: mirror matmul_tc's `emit_load_b_tile` pattern
/// exactly with N remapped to seq_k and B remapped to K. The helper
/// is already proven; we just call it with different base pointers
/// and strides.
fn emit_stage_k_chunk(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    k_chunk_src_global: Register, // u64: first byte of this chunk's K[n_start, k_start]
    tile_k_shared: Register,      // u32: k_chunk shared base
    tid_x: Register,
    d_k_bytes: Register, // u32: d_k * 2 (K's row stride)
) {
    // Lane distribution: each thread handles 4 f16 values (one per
    // the i∈0..4 unroll). flat = tid*4 + i ∈ [0, 128). Map to (col,
    // row) in the logical 16(d_k)×8(seq_k) transposed B tile:
    //   col = flat / 16  (which d_k row this represents → col in
    //                     column-major B)  — wait: in column-major
    //                     B, "col" of B has stride = 16 rows × 2 B.
    //   row = flat % 16
    // Global source: K[n_start + col, k_start + row] where col is the
    // seq_k coordinate and row is the d_k coordinate.
    //   global_byte_off = col * d_k_bytes + row * 2
    // Shared dest (column-major):
    //   shared_byte_off = col * 32 + row * 2
    //
    // Exactly matches matmul_tc's emit_load_b_tile with (M=seq_q's
    // "K" = d_k, N=seq_k). `k_chunk_src_global` already points at
    // K[n_start, k_start] — i.e. the first byte of the 8-row × 16-col
    // subregion of K in row-major global memory.
    let r_lane_base = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_lane_base,
        lhs: Operand::Reg(tid_x),
        rhs: Operand::ImmU32(4),
        ty: PtxType::U32,
    }));
    for i in 0..4u32 {
        let r_flat = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: r_flat,
            lhs: Operand::Reg(r_lane_base),
            rhs: Operand::ImmU32(i),
            ty: PtxType::U32,
        }));
        let r_col = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Div {
            dst: r_col,
            lhs: Operand::Reg(r_flat),
            rhs: Operand::ImmU32(16),
            ty: PtxType::U32,
        }));
        let r_row = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Rem {
            dst: r_row,
            lhs: Operand::Reg(r_flat),
            rhs: Operand::ImmU32(16),
            ty: PtxType::U32,
        }));

        // shared_off = col * 32 + row * 2
        let r_row_bytes = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Mul {
            dst: r_row_bytes,
            lhs: Operand::Reg(r_row),
            rhs: Operand::ImmU32(BYTES_PER_HALF),
            ty: PtxType::U32,
        }));
        let r_shared_off = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Mad {
            dst: r_shared_off,
            a: Operand::Reg(r_col),
            b: Operand::ImmU32(32), // column stride in column-major 16-row tile
            c: Operand::Reg(r_row_bytes),
            ty: PtxType::U32,
            mode: MadMode::Lo,
        }));
        let r_shared_addr = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: r_shared_addr,
            lhs: Operand::Reg(tile_k_shared),
            rhs: Operand::Reg(r_shared_off),
            ty: PtxType::U32,
        }));

        // global_off_u64 = col * d_k_bytes + row * 2
        let rd_row_global_off = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
            dst: rd_row_global_off,
            lhs: Operand::Reg(r_col),
            rhs: Operand::Reg(d_k_bytes),
            src_ty: PtxType::U32,
        }));
        let rd_row_bytes = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Cvt {
            dst: rd_row_bytes,
            src: r_row_bytes,
            dst_ty: PtxType::U64,
            src_ty: PtxType::U32,
        });
        let rd_per_thread_off = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: rd_per_thread_off,
            lhs: Operand::Reg(rd_row_global_off),
            rhs: Operand::Reg(rd_row_bytes),
            ty: PtxType::U64,
        }));
        let rd_global_addr = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: rd_global_addr,
            lhs: Operand::Reg(k_chunk_src_global),
            rhs: Operand::Reg(rd_per_thread_off),
            ty: PtxType::U64,
        }));
        let r_tmp_h = alloc.alloc(PtxType::F16);
        kernel.push(PtxInstruction::Memory(MemoryOp::LdGlobal {
            dst: r_tmp_h,
            addr: rd_global_addr,
            ty: PtxType::F16,
        }));
        kernel.push(PtxInstruction::Memory(MemoryOp::StShared {
            addr: r_shared_addr,
            src: r_tmp_h,
            ty: PtxType::F16,
        }));
    }
}

// ---------------------------------------------------------------------
// Gate A — matmul1-only dev kernel.
// Computes scores = (Q · Kᵀ) * inv_sqrt_dk and writes to global
// `scores_ptr`. No softmax, no second matmul, no causal mask.
// Deleted before the final Gate C commit; the test that exercises
// it is retained as a regression asset if it stays callable.
// ---------------------------------------------------------------------

/// Build the IR module for the Gate A dev kernel.
///
/// **Temporary.** This kernel exists purely to validate matmul1
/// correctness in isolation during 6.6 development. It is deleted
/// before the final commit — any future reference to this function
/// in production code is a regression.
#[doc(hidden)]
pub fn build_attention_tc_gate_a_module(sm: &str) -> PtxModule {
    let mut alloc = RegisterAllocator::new();
    let mut kernel = PtxKernel::new("attention_tc_gate_a");

    kernel.add_param(PtxParam::pointer("q_ptr", PtxType::F16));
    kernel.add_param(PtxParam::pointer("k_ptr", PtxType::F16));
    kernel.add_param(PtxParam::pointer("scores_ptr", PtxType::F32));
    kernel.add_param(PtxParam::scalar("seq_q", PtxType::U32));
    kernel.add_param(PtxParam::scalar("seq_k", PtxType::U32));
    kernel.add_param(PtxParam::scalar("d_k", PtxType::U32));
    kernel.add_param(PtxParam::scalar("inv_sqrt_dk", PtxType::F32));

    kernel.add_shared_decl(SharedDecl {
        name: "tile_q".to_string(),
        align: 4,
        size_bytes: TILE_Q_BYTES,
    });
    kernel.add_shared_decl(SharedDecl {
        name: "k_chunk".to_string(),
        align: 4,
        size_bytes: K_CHUNK_BYTES,
    });

    // ---- Load params + cvta to global ----
    let rd_q_param = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: rd_q_param,
        param_name: "q_ptr".to_string(),
        ty: PtxType::U64,
    }));
    let rd_k_param = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: rd_k_param,
        param_name: "k_ptr".to_string(),
        ty: PtxType::U64,
    }));
    let rd_scores_param = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: rd_scores_param,
        param_name: "scores_ptr".to_string(),
        ty: PtxType::U64,
    }));
    let r_seq_k = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: r_seq_k,
        param_name: "seq_k".to_string(),
        ty: PtxType::U32,
    }));
    let r_d_k = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: r_d_k,
        param_name: "d_k".to_string(),
        ty: PtxType::U32,
    }));
    let r_inv_sqrt_dk = alloc.alloc(PtxType::F32);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: r_inv_sqrt_dk,
        param_name: "inv_sqrt_dk".to_string(),
        ty: PtxType::F32,
    }));

    let rd_q = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::CvtaToGlobal {
        dst: rd_q,
        src: rd_q_param,
    }));
    let rd_k_global = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::CvtaToGlobal {
        dst: rd_k_global,
        src: rd_k_param,
    }));
    let rd_scores = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::CvtaToGlobal {
        dst: rd_scores,
        src: rd_scores_param,
    }));

    // ---- Special registers ----
    let (r_tid, tid_instr) = special::tid_x(&mut alloc);
    kernel.push(tid_instr);
    let r_bidx = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_bidx,
        src: Operand::SpecialReg(SpecialReg::CtaidX),
        ty: PtxType::U32,
    });

    // block_row = bidx * 16 (Q-row-slab)
    let r_block_row = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_block_row,
        lhs: Operand::Reg(r_bidx),
        rhs: Operand::ImmU32(BM),
        ty: PtxType::U32,
    }));

    // Stride bytes (reused).
    let r_d_k_bytes = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_d_k_bytes,
        lhs: Operand::Reg(r_d_k),
        rhs: Operand::ImmU32(BYTES_PER_HALF),
        ty: PtxType::U32,
    }));
    let r_seq_k_bytes_f32 = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_seq_k_bytes_f32,
        lhs: Operand::Reg(r_seq_k),
        rhs: Operand::ImmU32(BYTES_PER_F32),
        ty: PtxType::U32,
    }));

    // q_rowslab_base_global = rd_q + block_row * d_k * 2
    let rd_q_rowslab_off = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
        dst: rd_q_rowslab_off,
        lhs: Operand::Reg(r_block_row),
        rhs: Operand::Reg(r_d_k_bytes),
        src_ty: PtxType::U32,
    }));
    let rd_q_rowslab = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: rd_q_rowslab,
        lhs: Operand::Reg(rd_q),
        rhs: Operand::Reg(rd_q_rowslab_off),
        ty: PtxType::U64,
    }));

    // scores_rowslab_base_global = rd_scores + block_row * seq_k * 4
    let rd_scores_rowslab_off = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
        dst: rd_scores_rowslab_off,
        lhs: Operand::Reg(r_block_row),
        rhs: Operand::Reg(r_seq_k_bytes_f32),
        src_ty: PtxType::U32,
    }));
    let rd_scores_rowslab = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: rd_scores_rowslab,
        lhs: Operand::Reg(rd_scores),
        rhs: Operand::Reg(rd_scores_rowslab_off),
        ty: PtxType::U64,
    }));

    // Shared bases.
    let r_tile_q = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_tile_q,
        src: Operand::SharedAddr("tile_q".to_string()),
        ty: PtxType::U32,
    });
    let r_k_chunk = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_k_chunk,
        src: Operand::SharedAddr("k_chunk".to_string()),
        ty: PtxType::U32,
    });

    // ---- Stage Q once ----
    emit_stage_q(
        &mut alloc,
        &mut kernel,
        rd_q_rowslab,
        r_tile_q,
        r_tid,
        r_d_k,
    );
    kernel.push(PtxInstruction::Control(ControlOp::BarSync { barrier_id: 0 }));

    // num_n_chunks = seq_k / 8; num_k_chunks = d_k / 16
    let r_num_n_chunks = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: r_num_n_chunks,
        lhs: Operand::Reg(r_seq_k),
        rhs: Operand::ImmU32(BN),
        ty: PtxType::U32,
    }));
    let r_num_k_chunks = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: r_num_k_chunks,
        lhs: Operand::Reg(r_d_k),
        rhs: Operand::ImmU32(BK),
        ty: PtxType::U32,
    }));

    // Outer loop: n_chunk = 0 .. num_n_chunks
    let r_n_chunk = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_n_chunk,
        src: Operand::ImmU32(0),
        ty: PtxType::U32,
    });
    kernel.push(PtxInstruction::Label("ATTN_TC_A_N_LOOP".to_string()));

    // Reset FragmentC (scalar f32 accumulators) to 0.
    let frag_d = alloc_c(&mut alloc);
    for r in &frag_d.regs {
        kernel.push(PtxInstruction::Mov {
            dst: *r,
            src: Operand::ImmF32(0.0),
            ty: PtxType::F32,
        });
    }

    // k_chunk_src base for this n_chunk:
    //   K[n_chunk*8, 0]
    //   = rd_k_global + (n_chunk * 8) * d_k_bytes
    let r_n_start = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_n_start,
        lhs: Operand::Reg(r_n_chunk),
        rhs: Operand::ImmU32(BN),
        ty: PtxType::U32,
    }));
    let rd_n_start_off = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
        dst: rd_n_start_off,
        lhs: Operand::Reg(r_n_start),
        rhs: Operand::Reg(r_d_k_bytes),
        src_ty: PtxType::U32,
    }));
    let rd_k_n_base = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: rd_k_n_base,
        lhs: Operand::Reg(rd_k_global),
        rhs: Operand::Reg(rd_n_start_off),
        ty: PtxType::U64,
    }));

    // Inner K-chunk loop.
    let r_k_chunk_idx = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_k_chunk_idx,
        src: Operand::ImmU32(0),
        ty: PtxType::U32,
    });
    kernel.push(PtxInstruction::Label("ATTN_TC_A_K_LOOP".to_string()));

    // k_start_bytes = k_chunk_idx * 16 * 2 = k_chunk_idx * 32
    let r_k_start_bytes = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_k_start_bytes,
        lhs: Operand::Reg(r_k_chunk_idx),
        rhs: Operand::ImmU32(BK * BYTES_PER_HALF),
        ty: PtxType::U32,
    }));
    let rd_k_start_bytes = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Cvt {
        dst: rd_k_start_bytes,
        src: r_k_start_bytes,
        dst_ty: PtxType::U64,
        src_ty: PtxType::U32,
    });
    let rd_k_chunk_src = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: rd_k_chunk_src,
        lhs: Operand::Reg(rd_k_n_base),
        rhs: Operand::Reg(rd_k_start_bytes),
        ty: PtxType::U64,
    }));

    // Stage K chunk.
    emit_stage_k_chunk(
        &mut alloc,
        &mut kernel,
        rd_k_chunk_src,
        r_k_chunk,
        r_tid,
        r_d_k_bytes,
    );
    kernel.push(PtxInstruction::Control(ControlOp::BarSync { barrier_id: 0 }));

    // Load fragments.
    // FragmentA from tile_q at column offset k_chunk_idx * 16 * 2 = k_start_bytes.
    // Row stride of tile_q is d_k_bytes (runtime value).
    let r_frag_a_base = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_frag_a_base,
        lhs: Operand::Reg(r_tile_q),
        rhs: Operand::Reg(r_k_start_bytes),
        ty: PtxType::U32,
    }));
    // The fragment loader expects a `u32` row_stride_bytes immediate;
    // d_k_bytes is a runtime register. Work around by passing d_k_bytes
    // in a register path — we need a register-stride variant.
    //
    // For Gate A, d_k is a runtime param and we need the fragment load
    // to use a runtime stride. The existing loader only accepts a
    // compile-time u32 stride. Emit the load manually using the same
    // PTX pattern as load_fragment_a_m16n8k16_shared_row.
    let frag_a = emit_load_fragment_a_runtime_stride(
        &mut alloc,
        &mut kernel,
        r_frag_a_base,
        r_tid,
        r_d_k_bytes,
    );
    let frag_b = load_fragment_b_m16n8k16_shared_col(
        &mut alloc,
        &mut kernel,
        r_k_chunk,
        r_tid,
        32, // column stride of 16-row tile_k_chunk = 16 * 2 bytes
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

    kernel.push(PtxInstruction::Control(ControlOp::BarSync { barrier_id: 0 }));

    // k_chunk_idx += 1 ; if < num_k_chunks, loop.
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_k_chunk_idx,
        lhs: Operand::Reg(r_k_chunk_idx),
        rhs: Operand::ImmU32(1),
        ty: PtxType::U32,
    }));
    let p_k_loop = alloc.alloc(PtxType::Pred);
    kernel.push(PtxInstruction::Control(ControlOp::SetP {
        dst: p_k_loop,
        cmp_op: CmpOp::Lt,
        lhs: Operand::Reg(r_k_chunk_idx),
        rhs: Operand::Reg(r_num_k_chunks),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Control(ControlOp::BraPred {
        pred: p_k_loop,
        target: "ATTN_TC_A_K_LOOP".to_string(),
        negate: false,
    }));

    // Scale by inv_sqrt_dk.
    for r in &frag_d.regs {
        kernel.push(PtxInstruction::Arith(ArithOp::Mul {
            dst: *r,
            lhs: Operand::Reg(*r),
            rhs: Operand::Reg(r_inv_sqrt_dk),
            ty: PtxType::F32,
        }));
    }

    // Store FragmentC to scores_rowslab[0..16][n_chunk*8..n_chunk*8+8]
    // using the same lane layout as matmul_tc's inline store.
    //   reg[0]: scores[groupID   , 2*tig    ]
    //   reg[1]: scores[groupID   , 2*tig + 1]
    //   reg[2]: scores[groupID+8 , 2*tig    ]
    //   reg[3]: scores[groupID+8 , 2*tig + 1]
    let r_group_id = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: r_group_id,
        lhs: Operand::Reg(r_tid),
        rhs: Operand::ImmU32(4),
        ty: PtxType::U32,
    }));
    let r_tig = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Rem {
        dst: r_tig,
        lhs: Operand::Reg(r_tid),
        rhs: Operand::ImmU32(4),
        ty: PtxType::U32,
    }));

    // n_start_bytes_f32 = n_chunk * 8 * 4 = n_chunk * 32
    let r_n_start_bytes_f32 = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_n_start_bytes_f32,
        lhs: Operand::Reg(r_n_chunk),
        rhs: Operand::ImmU32(BN * BYTES_PER_F32),
        ty: PtxType::U32,
    }));
    // row_off_0 = groupID * seq_k_bytes_f32 + n_start_bytes_f32 + tig*8
    let r_row_off_g0_row = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_row_off_g0_row,
        lhs: Operand::Reg(r_group_id),
        rhs: Operand::Reg(r_seq_k_bytes_f32),
        ty: PtxType::U32,
    }));
    let r_tig_times_8 = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_tig_times_8,
        lhs: Operand::Reg(r_tig),
        rhs: Operand::ImmU32(8),
        ty: PtxType::U32,
    }));
    let r_row0_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_row0_off,
        lhs: Operand::Reg(r_row_off_g0_row),
        rhs: Operand::Reg(r_n_start_bytes_f32),
        ty: PtxType::U32,
    }));
    let r_row0_base = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_row0_base,
        lhs: Operand::Reg(r_row0_off),
        rhs: Operand::Reg(r_tig_times_8),
        ty: PtxType::U32,
    }));
    // row8_base = row0_base + 8 * seq_k_bytes_f32
    let r_eight_rows_f32 = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_eight_rows_f32,
        lhs: Operand::Reg(r_seq_k_bytes_f32),
        rhs: Operand::ImmU32(8),
        ty: PtxType::U32,
    }));
    let r_row8_base = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_row8_base,
        lhs: Operand::Reg(r_row0_base),
        rhs: Operand::Reg(r_eight_rows_f32),
        ty: PtxType::U32,
    }));

    // Store four f32 regs.
    let emit_store = |alloc: &mut RegisterAllocator,
                      kernel: &mut PtxKernel,
                      base_off32: Register,
                      extra: u32,
                      src_reg: Register| {
        let rd_off = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Cvt {
            dst: rd_off,
            src: base_off32,
            dst_ty: PtxType::U64,
            src_ty: PtxType::U32,
        });
        let rd_addr = if extra == 0 {
            let rd = alloc.alloc(PtxType::U64);
            kernel.push(PtxInstruction::Arith(ArithOp::Add {
                dst: rd,
                lhs: Operand::Reg(rd_scores_rowslab),
                rhs: Operand::Reg(rd_off),
                ty: PtxType::U64,
            }));
            rd
        } else {
            let rd_tmp = alloc.alloc(PtxType::U64);
            kernel.push(PtxInstruction::Arith(ArithOp::Add {
                dst: rd_tmp,
                lhs: Operand::Reg(rd_scores_rowslab),
                rhs: Operand::Reg(rd_off),
                ty: PtxType::U64,
            }));
            let rd = alloc.alloc(PtxType::U64);
            kernel.push(PtxInstruction::Arith(ArithOp::Add {
                dst: rd,
                lhs: Operand::Reg(rd_tmp),
                rhs: Operand::ImmU32(extra),
                ty: PtxType::U64,
            }));
            rd
        };
        kernel.push(PtxInstruction::Memory(MemoryOp::StGlobal {
            addr: rd_addr,
            src: src_reg,
            ty: PtxType::F32,
        }));
    };
    emit_store(&mut alloc, &mut kernel, r_row0_base, 0, frag_d.regs[0]);
    emit_store(&mut alloc, &mut kernel, r_row0_base, 4, frag_d.regs[1]);
    emit_store(&mut alloc, &mut kernel, r_row8_base, 0, frag_d.regs[2]);
    emit_store(&mut alloc, &mut kernel, r_row8_base, 4, frag_d.regs[3]);

    // n_chunk += 1 ; if < num_n_chunks, loop.
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_n_chunk,
        lhs: Operand::Reg(r_n_chunk),
        rhs: Operand::ImmU32(1),
        ty: PtxType::U32,
    }));
    let p_n_loop = alloc.alloc(PtxType::Pred);
    kernel.push(PtxInstruction::Control(ControlOp::SetP {
        dst: p_n_loop,
        cmp_op: CmpOp::Lt,
        lhs: Operand::Reg(r_n_chunk),
        rhs: Operand::Reg(r_num_n_chunks),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Control(ControlOp::BraPred {
        pred: p_n_loop,
        target: "ATTN_TC_A_N_LOOP".to_string(),
        negate: false,
    }));

    kernel.push(PtxInstruction::Control(ControlOp::Ret));
    kernel.set_registers(alloc.into_allocated());

    let mut module = PtxModule::new(sm);
    module.add_kernel(kernel);
    module
}

// ---------------------------------------------------------------------
// Runtime-stride FragmentA loader — mirrors
// `kaio_core::fragment::load_fragment_a_m16n8k16_shared_row`'s thread
// layout but accepts a runtime register for the row stride instead of
// a compile-time constant. Needed because tile_q's row stride is
// `d_k * 2` (d_k is a runtime kernel parameter).
//
// Per PTX ISA 9.7.13.5.8.1, each thread loads 4 packed-half2 b32
// registers:
//   a[0,1] : A[groupID,   2*tig .. 2*tig+1]
//   a[2,3] : A[groupID+8, 2*tig .. 2*tig+1]
//   a[4,5] : A[groupID,   2*tig+8 .. 2*tig+9]
//   a[6,7] : A[groupID+8, 2*tig+8 .. 2*tig+9]
// Pairs are contiguous in memory (same row, adjacent col) — one b32
// load per register.
// ---------------------------------------------------------------------

fn emit_load_fragment_a_runtime_stride(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    tile_base_shared: Register, // u32
    tid_x: Register,            // u32
    row_stride_bytes: Register, // u32 runtime value
) -> kaio_core::fragment::FragmentA {
    use kaio_core::fragment::FragmentA;

    let r_group_id = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: r_group_id,
        lhs: Operand::Reg(tid_x),
        rhs: Operand::ImmU32(4),
        ty: PtxType::U32,
    }));
    let r_tig = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Rem {
        dst: r_tig,
        lhs: Operand::Reg(tid_x),
        rhs: Operand::ImmU32(4),
        ty: PtxType::U32,
    }));
    // tig_bytes = tig * 4 (one half2 pair = 4 bytes)
    let r_tig_bytes = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_tig_bytes,
        lhs: Operand::Reg(r_tig),
        rhs: Operand::ImmU32(4),
        ty: PtxType::U32,
    }));

    // row0_off = group_id * row_stride_bytes + tig_bytes
    let r_row0_part = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_row0_part,
        lhs: Operand::Reg(r_group_id),
        rhs: Operand::Reg(row_stride_bytes),
        ty: PtxType::U32,
    }));
    let r_row0_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_row0_off,
        lhs: Operand::Reg(r_row0_part),
        rhs: Operand::Reg(r_tig_bytes),
        ty: PtxType::U32,
    }));

    // row8_off = row0_off + 8 * row_stride_bytes
    let r_eight_stride = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_eight_stride,
        lhs: Operand::Reg(row_stride_bytes),
        rhs: Operand::ImmU32(8),
        ty: PtxType::U32,
    }));
    let r_row8_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_row8_off,
        lhs: Operand::Reg(r_row0_part),
        rhs: Operand::Reg(r_eight_stride),
        ty: PtxType::U32,
    }));
    let r_row8_off2 = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_row8_off2,
        lhs: Operand::Reg(r_row8_off),
        rhs: Operand::Reg(r_tig_bytes),
        ty: PtxType::U32,
    }));

    let make_addr =
        |alloc: &mut RegisterAllocator, kernel: &mut PtxKernel, off32: Register, extra: u32| {
            let addr = alloc.alloc(PtxType::U32);
            let with_tile = alloc.alloc(PtxType::U32);
            kernel.push(PtxInstruction::Arith(ArithOp::Add {
                dst: with_tile,
                lhs: Operand::Reg(tile_base_shared),
                rhs: Operand::Reg(off32),
                ty: PtxType::U32,
            }));
            if extra == 0 {
                kernel.push(PtxInstruction::Mov {
                    dst: addr,
                    src: Operand::Reg(with_tile),
                    ty: PtxType::U32,
                });
            } else {
                kernel.push(PtxInstruction::Arith(ArithOp::Add {
                    dst: addr,
                    lhs: Operand::Reg(with_tile),
                    rhs: Operand::ImmU32(extra),
                    ty: PtxType::U32,
                }));
            }
            addr
        };

    let reg0 = alloc.alloc_packed_half2();
    let addr0 = make_addr(alloc, kernel, r_row0_off, 0);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdShared {
        dst: reg0,
        addr: addr0,
        ty: PtxType::U32,
    }));
    let reg1 = alloc.alloc_packed_half2();
    let addr1 = make_addr(alloc, kernel, r_row8_off2, 0);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdShared {
        dst: reg1,
        addr: addr1,
        ty: PtxType::U32,
    }));
    let reg2 = alloc.alloc_packed_half2();
    // +16 byte offset is 8 more fp16 columns (half2 pairs at 2*tig+8)
    let addr2 = make_addr(alloc, kernel, r_row0_off, 16);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdShared {
        dst: reg2,
        addr: addr2,
        ty: PtxType::U32,
    }));
    let reg3 = alloc.alloc_packed_half2();
    let addr3 = make_addr(alloc, kernel, r_row8_off2, 16);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdShared {
        dst: reg3,
        addr: addr3,
        ty: PtxType::U32,
    }));

    FragmentA {
        regs: [reg0, reg1, reg2, reg3],
    }
}

// ---------------------------------------------------------------------
// Gate A host API (dev-only — deleted before final commit)
// ---------------------------------------------------------------------

/// Gate A dev host entrypoint: runs `attention_tc_gate_a` on the device
/// and writes the scaled scores matrix (seq_q × seq_k f32) to `scores`.
///
/// **Temporary**. Exists only for Sprint 6.6 Gate A's in-isolation
/// matmul1 correctness test. Deleted before the final commit.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn attention_tc_gate_a(
    device: &KaioDevice,
    q: &GpuBuffer<f16>,
    k: &GpuBuffer<f16>,
    scores: &mut GpuBuffer<f32>,
    seq_q: u32,
    seq_k: u32,
    d_k: u32,
) -> Result<()> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};

    if seq_q == 0 || seq_k == 0 || d_k == 0 {
        return Err(KaioError::InvalidConfig(
            "attention_tc_gate_a dims must be non-zero".to_string(),
        ));
    }
    if !seq_q.is_multiple_of(BM) || !seq_k.is_multiple_of(BK) || !d_k.is_multiple_of(BK) {
        return Err(KaioError::InvalidConfig(format!(
            "attention_tc_gate_a: seq_q%16=seq_k%16=d_k%16=0 required \
             (got seq_q={seq_q}, seq_k={seq_k}, d_k={d_k})"
        )));
    }
    if seq_k > MAX_SEQ_K || d_k > MAX_D_K {
        return Err(KaioError::InvalidConfig(format!(
            "attention_tc_gate_a: seq_k ≤ {MAX_SEQ_K} and d_k ≤ {MAX_D_K} required"
        )));
    }
    let scores_len = (seq_q as usize) * (seq_k as usize);
    if scores.len() < scores_len {
        return Err(KaioError::InvalidConfig(format!(
            "attention_tc_gate_a: scores buffer too small: need {scores_len}, got {}",
            scores.len()
        )));
    }

    let info = device.info()?;
    let (major, minor) = info.compute_capability;
    let sm = format!("sm_{major}{minor}");
    let module = build_attention_tc_gate_a_module(&sm);
    let kmodule = device.load_module(&module)?;
    let func = kmodule.function("attention_tc_gate_a")?;

    let inv_sqrt_dk: f32 = 1.0f32 / (d_k as f32).sqrt();

    let cfg = LaunchConfig {
        grid_dim: (seq_q / BM, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0, // all decls are static
    };

    unsafe {
        device
            .stream()
            .launch_builder(func.inner())
            .arg(q.inner())
            .arg(k.inner())
            .arg(scores.inner_mut())
            .arg(&seq_q)
            .arg(&seq_k)
            .arg(&d_k)
            .arg(&inv_sqrt_dk)
            .launch(cfg)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------
// Gate B helpers — softmax, cvt bridge, probs store.
// These compose with the matmul1 body from Gate A but write into shared
// (scores_tile, probs_tile) rather than global, so the softmax phase
// has all 16 × seq_k f32 values resident before reducing per-row.
// ---------------------------------------------------------------------

/// Store a full FragmentC to `scores_tile` at the column offset
/// corresponding to the current `n_chunk`. Matches the PTX ISA
/// m16n8k16 C/D layout: each lane holds four scalars at
///   reg[0]: (group_id    , 2*tig    )
///   reg[1]: (group_id    , 2*tig + 1)
///   reg[2]: (group_id + 8, 2*tig    )
///   reg[3]: (group_id + 8, 2*tig + 1)
/// where group_id = tid/4, tig = tid%4.
#[allow(clippy::too_many_arguments)]
fn emit_store_fragment_c_to_scores_tile(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    frag_d: kaio_core::fragment::FragmentC,
    scores_tile_shared: Register,
    tid_x: Register,
    n_chunk: Register,
    seq_k_bytes_f32: Register,
) {
    let r_group_id = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: r_group_id,
        lhs: Operand::Reg(tid_x),
        rhs: Operand::ImmU32(4),
        ty: PtxType::U32,
    }));
    let r_tig = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Rem {
        dst: r_tig,
        lhs: Operand::Reg(tid_x),
        rhs: Operand::ImmU32(4),
        ty: PtxType::U32,
    }));
    let r_n_start_bytes_f32 = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_n_start_bytes_f32,
        lhs: Operand::Reg(n_chunk),
        rhs: Operand::ImmU32(BN * BYTES_PER_F32),
        ty: PtxType::U32,
    }));
    let r_row_off_g = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_row_off_g,
        lhs: Operand::Reg(r_group_id),
        rhs: Operand::Reg(seq_k_bytes_f32),
        ty: PtxType::U32,
    }));
    let r_tig8 = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_tig8,
        lhs: Operand::Reg(r_tig),
        rhs: Operand::ImmU32(8),
        ty: PtxType::U32,
    }));
    let r_row_off_tmp = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_row_off_tmp,
        lhs: Operand::Reg(r_row_off_g),
        rhs: Operand::Reg(r_n_start_bytes_f32),
        ty: PtxType::U32,
    }));
    let r_row0_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_row0_off,
        lhs: Operand::Reg(r_row_off_tmp),
        rhs: Operand::Reg(r_tig8),
        ty: PtxType::U32,
    }));
    let r_eight_rows = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_eight_rows,
        lhs: Operand::Reg(seq_k_bytes_f32),
        rhs: Operand::ImmU32(8),
        ty: PtxType::U32,
    }));
    let r_row8_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_row8_off,
        lhs: Operand::Reg(r_row0_off),
        rhs: Operand::Reg(r_eight_rows),
        ty: PtxType::U32,
    }));
    let emit_st = |alloc: &mut RegisterAllocator,
                   kernel: &mut PtxKernel,
                   base: Register,
                   extra: u32,
                   src_reg: Register| {
        let addr = alloc.alloc(PtxType::U32);
        let with_tile = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: with_tile,
            lhs: Operand::Reg(scores_tile_shared),
            rhs: Operand::Reg(base),
            ty: PtxType::U32,
        }));
        if extra == 0 {
            kernel.push(PtxInstruction::Mov {
                dst: addr,
                src: Operand::Reg(with_tile),
                ty: PtxType::U32,
            });
        } else {
            kernel.push(PtxInstruction::Arith(ArithOp::Add {
                dst: addr,
                lhs: Operand::Reg(with_tile),
                rhs: Operand::ImmU32(extra),
                ty: PtxType::U32,
            }));
        }
        kernel.push(PtxInstruction::Memory(MemoryOp::StShared {
            addr,
            src: src_reg,
            ty: PtxType::F32,
        }));
    };
    emit_st(alloc, kernel, r_row0_off, 0, frag_d.regs[0]);
    emit_st(alloc, kernel, r_row0_off, 4, frag_d.regs[1]);
    emit_st(alloc, kernel, r_row8_off, 0, frag_d.regs[2]);
    emit_st(alloc, kernel, r_row8_off, 4, frag_d.regs[3]);
}

/// Softmax over `scores_tile` (16 × seq_k f32 shared) writing f16
/// probs to `probs_tile`. Per-row serial (16 iterations). Each row
/// uses three warp-strided column sub-loops:
///   (1) max-reduce    → bfly reduce the local max
///   (2) exp + sum     → bfly reduce the local sum (stores exp values
///                       back into scores_tile for phase 3 re-reading)
///   (3) normalize+cvt → multiply by 1/sum, cvt to f16, store probs.
///
/// For `seq_k < 32`, lanes with `lane ≥ seq_k` never enter the strided
/// loop; their `local_max = -INF` and `local_sum = 0` are identity
/// elements for bfly reduction so the per-row result is correct.
fn emit_softmax_rows(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    scores_tile_shared: Register,
    probs_tile_shared: Register,
    tid_x: Register,
    seq_k: Register,
    seq_k_bytes_f32: Register,
    seq_k_bytes_f16: Register,
) {
    const LOG2_E: f32 = std::f32::consts::LOG2_E;
    const NEG_INF_F32: f32 = -3.4028235e38f32;

    let r_row_idx = alloc.alloc(PtxType::U32);
    let r_row_off_f32 = alloc.alloc(PtxType::U32);
    let r_row_off_f16 = alloc.alloc(PtxType::U32);
    let r_scores_row = alloc.alloc(PtxType::U32);
    let r_probs_row = alloc.alloc(PtxType::U32);

    let r_local_max = alloc.alloc(PtxType::F32);
    let r_local_sum = alloc.alloc(PtxType::F32);
    let r_tmp_shfl = alloc.alloc(PtxType::F32);

    let r_col = alloc.alloc(PtxType::U32);
    let r_col_bytes_f32 = alloc.alloc(PtxType::U32);
    let r_col_bytes_f16 = alloc.alloc(PtxType::U32);
    let r_addr_f32 = alloc.alloc(PtxType::U32);
    let r_addr_f16 = alloc.alloc(PtxType::U32);

    let r_val = alloc.alloc(PtxType::F32);
    let r_diff = alloc.alloc(PtxType::F32);
    let r_scaled = alloc.alloc(PtxType::F32);
    let r_ex = alloc.alloc(PtxType::F32);
    let r_inv_sum = alloc.alloc(PtxType::F32);
    let r_prob_f32 = alloc.alloc(PtxType::F32);
    let r_prob_f16 = alloc.alloc(PtxType::F16);

    kernel.push(PtxInstruction::Mov {
        dst: r_row_idx,
        src: Operand::ImmU32(0),
        ty: PtxType::U32,
    });
    kernel.push(PtxInstruction::Label("SOFTMAX_ROW_LOOP".to_string()));

    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_row_off_f32,
        lhs: Operand::Reg(r_row_idx),
        rhs: Operand::Reg(seq_k_bytes_f32),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_scores_row,
        lhs: Operand::Reg(scores_tile_shared),
        rhs: Operand::Reg(r_row_off_f32),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_row_off_f16,
        lhs: Operand::Reg(r_row_idx),
        rhs: Operand::Reg(seq_k_bytes_f16),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_probs_row,
        lhs: Operand::Reg(probs_tile_shared),
        rhs: Operand::Reg(r_row_off_f16),
        ty: PtxType::U32,
    }));

    // Phase 1: max reduction
    kernel.push(PtxInstruction::Mov {
        dst: r_local_max,
        src: Operand::ImmF32(NEG_INF_F32),
        ty: PtxType::F32,
    });
    kernel.push(PtxInstruction::Mov {
        dst: r_col,
        src: Operand::Reg(tid_x),
        ty: PtxType::U32,
    });
    kernel.push(PtxInstruction::Label("SOFTMAX_MAX_LOOP".to_string()));
    let p_max_done = alloc.alloc(PtxType::Pred);
    kernel.push(PtxInstruction::Control(ControlOp::SetP {
        dst: p_max_done,
        cmp_op: CmpOp::Ge,
        lhs: Operand::Reg(r_col),
        rhs: Operand::Reg(seq_k),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Control(ControlOp::BraPred {
        pred: p_max_done,
        target: "SOFTMAX_MAX_EXIT".to_string(),
        negate: false,
    }));
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_col_bytes_f32,
        lhs: Operand::Reg(r_col),
        rhs: Operand::ImmU32(BYTES_PER_F32),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_addr_f32,
        lhs: Operand::Reg(r_scores_row),
        rhs: Operand::Reg(r_col_bytes_f32),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Memory(MemoryOp::LdShared {
        dst: r_val,
        addr: r_addr_f32,
        ty: PtxType::F32,
    }));
    kernel.push(PtxInstruction::Arith(ArithOp::Max {
        dst: r_local_max,
        lhs: Operand::Reg(r_local_max),
        rhs: Operand::Reg(r_val),
        ty: PtxType::F32,
    }));
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_col,
        lhs: Operand::Reg(r_col),
        rhs: Operand::ImmU32(32),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Control(ControlOp::Bra {
        target: "SOFTMAX_MAX_LOOP".to_string(),
    }));
    kernel.push(PtxInstruction::Label("SOFTMAX_MAX_EXIT".to_string()));
    for mask in [16u32, 8, 4, 2, 1] {
        kernel.push(PtxInstruction::Control(ControlOp::ShflSyncBfly {
            dst: r_tmp_shfl,
            src: r_local_max,
            lane_mask: Operand::ImmU32(mask),
            c: 31,
            mask: 0xFFFFFFFF,
        }));
        kernel.push(PtxInstruction::Arith(ArithOp::Max {
            dst: r_local_max,
            lhs: Operand::Reg(r_local_max),
            rhs: Operand::Reg(r_tmp_shfl),
            ty: PtxType::F32,
        }));
    }

    // Phase 2: exp + sum
    kernel.push(PtxInstruction::Mov {
        dst: r_local_sum,
        src: Operand::ImmF32(0.0),
        ty: PtxType::F32,
    });
    kernel.push(PtxInstruction::Mov {
        dst: r_col,
        src: Operand::Reg(tid_x),
        ty: PtxType::U32,
    });
    kernel.push(PtxInstruction::Label("SOFTMAX_EXP_LOOP".to_string()));
    let p_exp_done = alloc.alloc(PtxType::Pred);
    kernel.push(PtxInstruction::Control(ControlOp::SetP {
        dst: p_exp_done,
        cmp_op: CmpOp::Ge,
        lhs: Operand::Reg(r_col),
        rhs: Operand::Reg(seq_k),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Control(ControlOp::BraPred {
        pred: p_exp_done,
        target: "SOFTMAX_EXP_EXIT".to_string(),
        negate: false,
    }));
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_col_bytes_f32,
        lhs: Operand::Reg(r_col),
        rhs: Operand::ImmU32(BYTES_PER_F32),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_addr_f32,
        lhs: Operand::Reg(r_scores_row),
        rhs: Operand::Reg(r_col_bytes_f32),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Memory(MemoryOp::LdShared {
        dst: r_val,
        addr: r_addr_f32,
        ty: PtxType::F32,
    }));
    kernel.push(PtxInstruction::Arith(ArithOp::Sub {
        dst: r_diff,
        lhs: Operand::Reg(r_val),
        rhs: Operand::Reg(r_local_max),
        ty: PtxType::F32,
    }));
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_scaled,
        lhs: Operand::Reg(r_diff),
        rhs: Operand::ImmF32(LOG2_E),
        ty: PtxType::F32,
    }));
    kernel.push(PtxInstruction::Arith(ArithOp::Ex2 {
        dst: r_ex,
        src: Operand::Reg(r_scaled),
    }));
    kernel.push(PtxInstruction::Memory(MemoryOp::StShared {
        addr: r_addr_f32,
        src: r_ex,
        ty: PtxType::F32,
    }));
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_local_sum,
        lhs: Operand::Reg(r_local_sum),
        rhs: Operand::Reg(r_ex),
        ty: PtxType::F32,
    }));
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_col,
        lhs: Operand::Reg(r_col),
        rhs: Operand::ImmU32(32),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Control(ControlOp::Bra {
        target: "SOFTMAX_EXP_LOOP".to_string(),
    }));
    kernel.push(PtxInstruction::Label("SOFTMAX_EXP_EXIT".to_string()));
    for mask in [16u32, 8, 4, 2, 1] {
        kernel.push(PtxInstruction::Control(ControlOp::ShflSyncBfly {
            dst: r_tmp_shfl,
            src: r_local_sum,
            lane_mask: Operand::ImmU32(mask),
            c: 31,
            mask: 0xFFFFFFFF,
        }));
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: r_local_sum,
            lhs: Operand::Reg(r_local_sum),
            rhs: Operand::Reg(r_tmp_shfl),
            ty: PtxType::F32,
        }));
    }

    // Phase 3: normalize + cvt
    kernel.push(PtxInstruction::Arith(ArithOp::Rcp {
        dst: r_inv_sum,
        src: Operand::Reg(r_local_sum),
    }));
    kernel.push(PtxInstruction::Mov {
        dst: r_col,
        src: Operand::Reg(tid_x),
        ty: PtxType::U32,
    });
    kernel.push(PtxInstruction::Label("SOFTMAX_NORM_LOOP".to_string()));
    let p_norm_done = alloc.alloc(PtxType::Pred);
    kernel.push(PtxInstruction::Control(ControlOp::SetP {
        dst: p_norm_done,
        cmp_op: CmpOp::Ge,
        lhs: Operand::Reg(r_col),
        rhs: Operand::Reg(seq_k),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Control(ControlOp::BraPred {
        pred: p_norm_done,
        target: "SOFTMAX_NORM_EXIT".to_string(),
        negate: false,
    }));
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_col_bytes_f32,
        lhs: Operand::Reg(r_col),
        rhs: Operand::ImmU32(BYTES_PER_F32),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_col_bytes_f16,
        lhs: Operand::Reg(r_col),
        rhs: Operand::ImmU32(BYTES_PER_HALF),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_addr_f32,
        lhs: Operand::Reg(r_scores_row),
        rhs: Operand::Reg(r_col_bytes_f32),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_addr_f16,
        lhs: Operand::Reg(r_probs_row),
        rhs: Operand::Reg(r_col_bytes_f16),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Memory(MemoryOp::LdShared {
        dst: r_val,
        addr: r_addr_f32,
        ty: PtxType::F32,
    }));
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_prob_f32,
        lhs: Operand::Reg(r_val),
        rhs: Operand::Reg(r_inv_sum),
        ty: PtxType::F32,
    }));
    kernel.push(PtxInstruction::Cvt {
        dst: r_prob_f16,
        src: r_prob_f32,
        dst_ty: PtxType::F16,
        src_ty: PtxType::F32,
    });
    kernel.push(PtxInstruction::Memory(MemoryOp::StShared {
        addr: r_addr_f16,
        src: r_prob_f16,
        ty: PtxType::F16,
    }));
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_col,
        lhs: Operand::Reg(r_col),
        rhs: Operand::ImmU32(32),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Control(ControlOp::Bra {
        target: "SOFTMAX_NORM_LOOP".to_string(),
    }));
    kernel.push(PtxInstruction::Label("SOFTMAX_NORM_EXIT".to_string()));

    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_row_idx,
        lhs: Operand::Reg(r_row_idx),
        rhs: Operand::ImmU32(1),
        ty: PtxType::U32,
    }));
    let p_row_loop = alloc.alloc(PtxType::Pred);
    kernel.push(PtxInstruction::Control(ControlOp::SetP {
        dst: p_row_loop,
        cmp_op: CmpOp::Lt,
        lhs: Operand::Reg(r_row_idx),
        rhs: Operand::ImmU32(BM),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Control(ControlOp::BraPred {
        pred: p_row_loop,
        target: "SOFTMAX_ROW_LOOP".to_string(),
        negate: false,
    }));
}

/// Copy `probs_tile` (16 × seq_k fp16 shared, row-major) → global
/// `probs_global_rowslab`. Gate B dev output path only; Gate C
/// consumes probs directly from shared in the second mma.sync.
fn emit_store_probs_to_global(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    probs_tile_shared: Register,
    probs_global_rowslab: Register,
    tid_x: Register,
    seq_k: Register,
    seq_k_bytes_f16: Register,
) {
    let r_total_half2 = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_total_half2,
        lhs: Operand::Reg(seq_k),
        rhs: Operand::ImmU32(8),
        ty: PtxType::U32,
    }));
    let r_half2_per_row = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: r_half2_per_row,
        lhs: Operand::Reg(seq_k),
        rhs: Operand::ImmU32(2),
        ty: PtxType::U32,
    }));

    let r_flat = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_flat,
        src: Operand::Reg(tid_x),
        ty: PtxType::U32,
    });
    kernel.push(PtxInstruction::Label("STORE_PROBS_LOOP".to_string()));
    let p_done = alloc.alloc(PtxType::Pred);
    kernel.push(PtxInstruction::Control(ControlOp::SetP {
        dst: p_done,
        cmp_op: CmpOp::Ge,
        lhs: Operand::Reg(r_flat),
        rhs: Operand::Reg(r_total_half2),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Control(ControlOp::BraPred {
        pred: p_done,
        target: "STORE_PROBS_EXIT".to_string(),
        negate: false,
    }));

    let r_row = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: r_row,
        lhs: Operand::Reg(r_flat),
        rhs: Operand::Reg(r_half2_per_row),
        ty: PtxType::U32,
    }));
    let r_col_half2 = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Rem {
        dst: r_col_half2,
        lhs: Operand::Reg(r_flat),
        rhs: Operand::Reg(r_half2_per_row),
        ty: PtxType::U32,
    }));
    let r_col_bytes = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_col_bytes,
        lhs: Operand::Reg(r_col_half2),
        rhs: Operand::ImmU32(4),
        ty: PtxType::U32,
    }));
    let r_shared_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mad {
        dst: r_shared_off,
        a: Operand::Reg(r_row),
        b: Operand::Reg(seq_k_bytes_f16),
        c: Operand::Reg(r_col_bytes),
        ty: PtxType::U32,
        mode: MadMode::Lo,
    }));
    let r_shared_addr = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_shared_addr,
        lhs: Operand::Reg(probs_tile_shared),
        rhs: Operand::Reg(r_shared_off),
        ty: PtxType::U32,
    }));
    let rd_row_off = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
        dst: rd_row_off,
        lhs: Operand::Reg(r_row),
        rhs: Operand::Reg(seq_k_bytes_f16),
        src_ty: PtxType::U32,
    }));
    let rd_col_bytes = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Cvt {
        dst: rd_col_bytes,
        src: r_col_bytes,
        dst_ty: PtxType::U64,
        src_ty: PtxType::U32,
    });
    let rd_per_thread_off = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: rd_per_thread_off,
        lhs: Operand::Reg(rd_row_off),
        rhs: Operand::Reg(rd_col_bytes),
        ty: PtxType::U64,
    }));
    let rd_global_addr = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: rd_global_addr,
        lhs: Operand::Reg(probs_global_rowslab),
        rhs: Operand::Reg(rd_per_thread_off),
        ty: PtxType::U64,
    }));
    let r_tmp = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdShared {
        dst: r_tmp,
        addr: r_shared_addr,
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Memory(MemoryOp::StGlobal {
        addr: rd_global_addr,
        src: r_tmp,
        ty: PtxType::U32,
    }));

    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_flat,
        lhs: Operand::Reg(r_flat),
        rhs: Operand::ImmU32(32),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Control(ControlOp::Bra {
        target: "STORE_PROBS_LOOP".to_string(),
    }));
    kernel.push(PtxInstruction::Label("STORE_PROBS_EXIT".to_string()));
}

// ---------------------------------------------------------------------
// Gate B — matmul1 + softmax + cvt dev kernel. Writes f16 probs
// (seq_q × seq_k) to global. Deleted before Gate C ships.
// ---------------------------------------------------------------------

/// Build the IR module for the Gate B dev kernel.
#[doc(hidden)]
pub fn build_attention_tc_gate_b_module(sm: &str) -> PtxModule {
    let mut alloc = RegisterAllocator::new();
    let mut kernel = PtxKernel::new("attention_tc_gate_b");

    kernel.add_param(PtxParam::pointer("q_ptr", PtxType::F16));
    kernel.add_param(PtxParam::pointer("k_ptr", PtxType::F16));
    kernel.add_param(PtxParam::pointer("probs_ptr", PtxType::F16));
    kernel.add_param(PtxParam::scalar("seq_q", PtxType::U32));
    kernel.add_param(PtxParam::scalar("seq_k", PtxType::U32));
    kernel.add_param(PtxParam::scalar("d_k", PtxType::U32));
    kernel.add_param(PtxParam::scalar("inv_sqrt_dk", PtxType::F32));

    kernel.add_shared_decl(SharedDecl {
        name: "tile_q".to_string(),
        align: 4,
        size_bytes: TILE_Q_BYTES,
    });
    kernel.add_shared_decl(SharedDecl {
        name: "k_chunk".to_string(),
        align: 4,
        size_bytes: K_CHUNK_BYTES,
    });
    kernel.add_shared_decl(SharedDecl {
        name: "scores_tile".to_string(),
        align: 4,
        size_bytes: SCORES_TILE_BYTES,
    });
    kernel.add_shared_decl(SharedDecl {
        name: "probs_tile".to_string(),
        align: 4,
        size_bytes: PROBS_TILE_BYTES,
    });

    // Params + cvta
    let rd_q_param = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: rd_q_param,
        param_name: "q_ptr".to_string(),
        ty: PtxType::U64,
    }));
    let rd_k_param = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: rd_k_param,
        param_name: "k_ptr".to_string(),
        ty: PtxType::U64,
    }));
    let rd_probs_param = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: rd_probs_param,
        param_name: "probs_ptr".to_string(),
        ty: PtxType::U64,
    }));
    let r_seq_k = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: r_seq_k,
        param_name: "seq_k".to_string(),
        ty: PtxType::U32,
    }));
    let r_d_k = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: r_d_k,
        param_name: "d_k".to_string(),
        ty: PtxType::U32,
    }));
    let r_inv_sqrt_dk = alloc.alloc(PtxType::F32);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: r_inv_sqrt_dk,
        param_name: "inv_sqrt_dk".to_string(),
        ty: PtxType::F32,
    }));

    let rd_q = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::CvtaToGlobal {
        dst: rd_q,
        src: rd_q_param,
    }));
    let rd_k_global = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::CvtaToGlobal {
        dst: rd_k_global,
        src: rd_k_param,
    }));
    let rd_probs = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::CvtaToGlobal {
        dst: rd_probs,
        src: rd_probs_param,
    }));

    let (r_tid, tid_instr) = special::tid_x(&mut alloc);
    kernel.push(tid_instr);
    let r_bidx = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_bidx,
        src: Operand::SpecialReg(SpecialReg::CtaidX),
        ty: PtxType::U32,
    });

    let r_block_row = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_block_row,
        lhs: Operand::Reg(r_bidx),
        rhs: Operand::ImmU32(BM),
        ty: PtxType::U32,
    }));

    let r_d_k_bytes = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_d_k_bytes,
        lhs: Operand::Reg(r_d_k),
        rhs: Operand::ImmU32(BYTES_PER_HALF),
        ty: PtxType::U32,
    }));
    let r_seq_k_bytes_f32 = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_seq_k_bytes_f32,
        lhs: Operand::Reg(r_seq_k),
        rhs: Operand::ImmU32(BYTES_PER_F32),
        ty: PtxType::U32,
    }));
    let r_seq_k_bytes_f16 = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_seq_k_bytes_f16,
        lhs: Operand::Reg(r_seq_k),
        rhs: Operand::ImmU32(BYTES_PER_HALF),
        ty: PtxType::U32,
    }));

    let rd_q_rowslab_off = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
        dst: rd_q_rowslab_off,
        lhs: Operand::Reg(r_block_row),
        rhs: Operand::Reg(r_d_k_bytes),
        src_ty: PtxType::U32,
    }));
    let rd_q_rowslab = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: rd_q_rowslab,
        lhs: Operand::Reg(rd_q),
        rhs: Operand::Reg(rd_q_rowslab_off),
        ty: PtxType::U64,
    }));
    let rd_probs_rowslab_off = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
        dst: rd_probs_rowslab_off,
        lhs: Operand::Reg(r_block_row),
        rhs: Operand::Reg(r_seq_k_bytes_f16),
        src_ty: PtxType::U32,
    }));
    let rd_probs_rowslab = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: rd_probs_rowslab,
        lhs: Operand::Reg(rd_probs),
        rhs: Operand::Reg(rd_probs_rowslab_off),
        ty: PtxType::U64,
    }));

    let r_tile_q = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_tile_q,
        src: Operand::SharedAddr("tile_q".to_string()),
        ty: PtxType::U32,
    });
    let r_k_chunk = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_k_chunk,
        src: Operand::SharedAddr("k_chunk".to_string()),
        ty: PtxType::U32,
    });
    let r_scores_tile = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_scores_tile,
        src: Operand::SharedAddr("scores_tile".to_string()),
        ty: PtxType::U32,
    });
    let r_probs_tile = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_probs_tile,
        src: Operand::SharedAddr("probs_tile".to_string()),
        ty: PtxType::U32,
    });

    emit_stage_q(
        &mut alloc,
        &mut kernel,
        rd_q_rowslab,
        r_tile_q,
        r_tid,
        r_d_k,
    );
    kernel.push(PtxInstruction::Control(ControlOp::BarSync { barrier_id: 0 }));

    let r_num_n_chunks = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: r_num_n_chunks,
        lhs: Operand::Reg(r_seq_k),
        rhs: Operand::ImmU32(BN),
        ty: PtxType::U32,
    }));
    let r_num_k_chunks = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: r_num_k_chunks,
        lhs: Operand::Reg(r_d_k),
        rhs: Operand::ImmU32(BK),
        ty: PtxType::U32,
    }));

    let r_n_chunk = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_n_chunk,
        src: Operand::ImmU32(0),
        ty: PtxType::U32,
    });
    kernel.push(PtxInstruction::Label("ATTN_TC_B_N_LOOP".to_string()));

    let frag_d = alloc_c(&mut alloc);
    for r in &frag_d.regs {
        kernel.push(PtxInstruction::Mov {
            dst: *r,
            src: Operand::ImmF32(0.0),
            ty: PtxType::F32,
        });
    }

    let r_n_start = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_n_start,
        lhs: Operand::Reg(r_n_chunk),
        rhs: Operand::ImmU32(BN),
        ty: PtxType::U32,
    }));
    let rd_n_start_off = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
        dst: rd_n_start_off,
        lhs: Operand::Reg(r_n_start),
        rhs: Operand::Reg(r_d_k_bytes),
        src_ty: PtxType::U32,
    }));
    let rd_k_n_base = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: rd_k_n_base,
        lhs: Operand::Reg(rd_k_global),
        rhs: Operand::Reg(rd_n_start_off),
        ty: PtxType::U64,
    }));

    let r_k_chunk_idx = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_k_chunk_idx,
        src: Operand::ImmU32(0),
        ty: PtxType::U32,
    });
    kernel.push(PtxInstruction::Label("ATTN_TC_B_K_LOOP".to_string()));

    let r_k_start_bytes = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_k_start_bytes,
        lhs: Operand::Reg(r_k_chunk_idx),
        rhs: Operand::ImmU32(BK * BYTES_PER_HALF),
        ty: PtxType::U32,
    }));
    let rd_k_start_bytes = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Cvt {
        dst: rd_k_start_bytes,
        src: r_k_start_bytes,
        dst_ty: PtxType::U64,
        src_ty: PtxType::U32,
    });
    let rd_k_chunk_src = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: rd_k_chunk_src,
        lhs: Operand::Reg(rd_k_n_base),
        rhs: Operand::Reg(rd_k_start_bytes),
        ty: PtxType::U64,
    }));
    emit_stage_k_chunk(
        &mut alloc,
        &mut kernel,
        rd_k_chunk_src,
        r_k_chunk,
        r_tid,
        r_d_k_bytes,
    );
    kernel.push(PtxInstruction::Control(ControlOp::BarSync { barrier_id: 0 }));

    let r_frag_a_base = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_frag_a_base,
        lhs: Operand::Reg(r_tile_q),
        rhs: Operand::Reg(r_k_start_bytes),
        ty: PtxType::U32,
    }));
    let frag_a = emit_load_fragment_a_runtime_stride(
        &mut alloc,
        &mut kernel,
        r_frag_a_base,
        r_tid,
        r_d_k_bytes,
    );
    let frag_b = load_fragment_b_m16n8k16_shared_col(
        &mut alloc,
        &mut kernel,
        r_k_chunk,
        r_tid,
        32,
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
    kernel.push(PtxInstruction::Control(ControlOp::BarSync { barrier_id: 0 }));

    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_k_chunk_idx,
        lhs: Operand::Reg(r_k_chunk_idx),
        rhs: Operand::ImmU32(1),
        ty: PtxType::U32,
    }));
    let p_k_loop = alloc.alloc(PtxType::Pred);
    kernel.push(PtxInstruction::Control(ControlOp::SetP {
        dst: p_k_loop,
        cmp_op: CmpOp::Lt,
        lhs: Operand::Reg(r_k_chunk_idx),
        rhs: Operand::Reg(r_num_k_chunks),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Control(ControlOp::BraPred {
        pred: p_k_loop,
        target: "ATTN_TC_B_K_LOOP".to_string(),
        negate: false,
    }));

    for r in &frag_d.regs {
        kernel.push(PtxInstruction::Arith(ArithOp::Mul {
            dst: *r,
            lhs: Operand::Reg(*r),
            rhs: Operand::Reg(r_inv_sqrt_dk),
            ty: PtxType::F32,
        }));
    }

    emit_store_fragment_c_to_scores_tile(
        &mut alloc,
        &mut kernel,
        frag_d,
        r_scores_tile,
        r_tid,
        r_n_chunk,
        r_seq_k_bytes_f32,
    );

    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_n_chunk,
        lhs: Operand::Reg(r_n_chunk),
        rhs: Operand::ImmU32(1),
        ty: PtxType::U32,
    }));
    let p_n_loop = alloc.alloc(PtxType::Pred);
    kernel.push(PtxInstruction::Control(ControlOp::SetP {
        dst: p_n_loop,
        cmp_op: CmpOp::Lt,
        lhs: Operand::Reg(r_n_chunk),
        rhs: Operand::Reg(r_num_n_chunks),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Control(ControlOp::BraPred {
        pred: p_n_loop,
        target: "ATTN_TC_B_N_LOOP".to_string(),
        negate: false,
    }));

    kernel.push(PtxInstruction::Control(ControlOp::BarSync { barrier_id: 0 }));

    emit_softmax_rows(
        &mut alloc,
        &mut kernel,
        r_scores_tile,
        r_probs_tile,
        r_tid,
        r_seq_k,
        r_seq_k_bytes_f32,
        r_seq_k_bytes_f16,
    );

    kernel.push(PtxInstruction::Control(ControlOp::BarSync { barrier_id: 0 }));

    emit_store_probs_to_global(
        &mut alloc,
        &mut kernel,
        r_probs_tile,
        rd_probs_rowslab,
        r_tid,
        r_seq_k,
        r_seq_k_bytes_f16,
    );

    kernel.push(PtxInstruction::Control(ControlOp::Ret));
    kernel.set_registers(alloc.into_allocated());

    let mut module = PtxModule::new(sm);
    module.add_kernel(kernel);
    module
}

/// Gate B dev host entrypoint — runs `attention_tc_gate_b` and writes
/// f16 probs (seq_q × seq_k) to `probs`. Deleted before final commit.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn attention_tc_gate_b(
    device: &KaioDevice,
    q: &GpuBuffer<f16>,
    k: &GpuBuffer<f16>,
    probs: &mut GpuBuffer<f16>,
    seq_q: u32,
    seq_k: u32,
    d_k: u32,
) -> Result<()> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};

    if seq_q == 0 || seq_k == 0 || d_k == 0 {
        return Err(KaioError::InvalidConfig(
            "attention_tc_gate_b dims must be non-zero".to_string(),
        ));
    }
    if !seq_q.is_multiple_of(BM) || !seq_k.is_multiple_of(BK) || !d_k.is_multiple_of(BK) {
        return Err(KaioError::InvalidConfig(format!(
            "attention_tc_gate_b: seq_q%16=seq_k%16=d_k%16=0 required \
             (got seq_q={seq_q}, seq_k={seq_k}, d_k={d_k})"
        )));
    }
    if seq_k > MAX_SEQ_K || d_k > MAX_D_K {
        return Err(KaioError::InvalidConfig(format!(
            "attention_tc_gate_b: seq_k ≤ {MAX_SEQ_K} and d_k ≤ {MAX_D_K} required"
        )));
    }
    let probs_len = (seq_q as usize) * (seq_k as usize);
    if probs.len() < probs_len {
        return Err(KaioError::InvalidConfig(format!(
            "attention_tc_gate_b: probs buffer too small: need {probs_len}, got {}",
            probs.len()
        )));
    }

    let info = device.info()?;
    let (major, minor) = info.compute_capability;
    let sm = format!("sm_{major}{minor}");
    let module = build_attention_tc_gate_b_module(&sm);
    let kmodule = device.load_module(&module)?;
    let func = kmodule.function("attention_tc_gate_b")?;

    let inv_sqrt_dk: f32 = 1.0f32 / (d_k as f32).sqrt();

    let cfg = LaunchConfig {
        grid_dim: (seq_q / BM, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        device
            .stream()
            .launch_builder(func.inner())
            .arg(q.inner())
            .arg(k.inner())
            .arg(probs.inner_mut())
            .arg(&seq_q)
            .arg(&seq_k)
            .arg(&d_k)
            .arg(&inv_sqrt_dk)
            .launch(cfg)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------
// Gate C helpers — V staging + second mma.sync + output store.
// ---------------------------------------------------------------------

/// Stage a V chunk: 16 seq_k rows × 8 d_v cols of row-major V global
/// → 16×8 column-major shared (B-fragment ready).
///
/// The flat → (row, col) mapping differs from K staging because V's
/// slice is 16×8 (row-major, row stride = d_v_bytes) vs K's slice
/// 8×16 (row-major, row stride = d_k_bytes, transposed when staging).
/// For V we iterate row ∈ 0..16 = flat/8, col ∈ 0..8 = flat%8, and
/// the global address is `row * d_v_bytes + col * 2`. Shared layout
/// is the same col-major 16×8 as k_chunk (`col * 32 + row * 2`).
fn emit_stage_v_chunk(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    v_chunk_src_global: Register, // u64: V + seq_k_start*d_v_bytes + d_v_start*2
    tile_v_shared: Register,      // u32
    tid_x: Register,
    d_v_bytes: Register, // u32 runtime: d_v * 2
) {
    let r_lane_base = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_lane_base,
        lhs: Operand::Reg(tid_x),
        rhs: Operand::ImmU32(4),
        ty: PtxType::U32,
    }));
    for i in 0..4u32 {
        let r_flat = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: r_flat,
            lhs: Operand::Reg(r_lane_base),
            rhs: Operand::ImmU32(i),
            ty: PtxType::U32,
        }));
        let r_row = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Div {
            dst: r_row,
            lhs: Operand::Reg(r_flat),
            rhs: Operand::ImmU32(8),
            ty: PtxType::U32,
        }));
        let r_col = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Rem {
            dst: r_col,
            lhs: Operand::Reg(r_flat),
            rhs: Operand::ImmU32(8),
            ty: PtxType::U32,
        }));

        // shared_off = col * 32 + row * 2
        let r_row_bytes = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Mul {
            dst: r_row_bytes,
            lhs: Operand::Reg(r_row),
            rhs: Operand::ImmU32(BYTES_PER_HALF),
            ty: PtxType::U32,
        }));
        let r_shared_off = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Mad {
            dst: r_shared_off,
            a: Operand::Reg(r_col),
            b: Operand::ImmU32(32),
            c: Operand::Reg(r_row_bytes),
            ty: PtxType::U32,
            mode: MadMode::Lo,
        }));
        let r_shared_addr = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: r_shared_addr,
            lhs: Operand::Reg(tile_v_shared),
            rhs: Operand::Reg(r_shared_off),
            ty: PtxType::U32,
        }));

        // global_off = row * d_v_bytes + col * 2
        let rd_row_global_off = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
            dst: rd_row_global_off,
            lhs: Operand::Reg(r_row),
            rhs: Operand::Reg(d_v_bytes),
            src_ty: PtxType::U32,
        }));
        let r_col_bytes = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Mul {
            dst: r_col_bytes,
            lhs: Operand::Reg(r_col),
            rhs: Operand::ImmU32(BYTES_PER_HALF),
            ty: PtxType::U32,
        }));
        let rd_col_bytes = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Cvt {
            dst: rd_col_bytes,
            src: r_col_bytes,
            dst_ty: PtxType::U64,
            src_ty: PtxType::U32,
        });
        let rd_per_thread_off = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: rd_per_thread_off,
            lhs: Operand::Reg(rd_row_global_off),
            rhs: Operand::Reg(rd_col_bytes),
            ty: PtxType::U64,
        }));
        let rd_global_addr = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: rd_global_addr,
            lhs: Operand::Reg(v_chunk_src_global),
            rhs: Operand::Reg(rd_per_thread_off),
            ty: PtxType::U64,
        }));
        let r_tmp_h = alloc.alloc(PtxType::F16);
        kernel.push(PtxInstruction::Memory(MemoryOp::LdGlobal {
            dst: r_tmp_h,
            addr: rd_global_addr,
            ty: PtxType::F16,
        }));
        kernel.push(PtxInstruction::Memory(MemoryOp::StShared {
            addr: r_shared_addr,
            src: r_tmp_h,
            ty: PtxType::F16,
        }));
    }
}

/// Emit the FragmentC → global output store for a 16×8 output tile,
/// matching matmul_tc's inline store: each lane writes four fp32
/// values at layout positions (group_id, 2*tig), (group_id, 2*tig+1),
/// (group_id+8, 2*tig), (group_id+8, 2*tig+1) within the current
/// 16×8 output-column chunk.
#[allow(clippy::too_many_arguments)]
fn emit_store_fragment_c_to_global_out(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    frag_d: kaio_core::fragment::FragmentC,
    out_rowslab_global: Register, // u64: out + block_row * d_v_bytes (f32)
    tid_x: Register,
    d_v_chunk: Register,   // u32: current d_v-column-chunk index
    d_v_bytes_f32: Register, // u32: d_v * 4 (output row stride)
) {
    let r_group_id = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: r_group_id,
        lhs: Operand::Reg(tid_x),
        rhs: Operand::ImmU32(4),
        ty: PtxType::U32,
    }));
    let r_tig = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Rem {
        dst: r_tig,
        lhs: Operand::Reg(tid_x),
        rhs: Operand::ImmU32(4),
        ty: PtxType::U32,
    }));

    // d_v_start_bytes_f32 = d_v_chunk * 8 * 4 = d_v_chunk * 32
    let r_d_v_start_bytes = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_d_v_start_bytes,
        lhs: Operand::Reg(d_v_chunk),
        rhs: Operand::ImmU32(BN * BYTES_PER_F32),
        ty: PtxType::U32,
    }));
    let r_row0_part = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_row0_part,
        lhs: Operand::Reg(r_group_id),
        rhs: Operand::Reg(d_v_bytes_f32),
        ty: PtxType::U32,
    }));
    let r_tig8 = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_tig8,
        lhs: Operand::Reg(r_tig),
        rhs: Operand::ImmU32(8),
        ty: PtxType::U32,
    }));
    let r_tmp0 = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_tmp0,
        lhs: Operand::Reg(r_row0_part),
        rhs: Operand::Reg(r_d_v_start_bytes),
        ty: PtxType::U32,
    }));
    let r_row0_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_row0_off,
        lhs: Operand::Reg(r_tmp0),
        rhs: Operand::Reg(r_tig8),
        ty: PtxType::U32,
    }));
    let r_eight_rows = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_eight_rows,
        lhs: Operand::Reg(d_v_bytes_f32),
        rhs: Operand::ImmU32(8),
        ty: PtxType::U32,
    }));
    let r_row8_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_row8_off,
        lhs: Operand::Reg(r_row0_off),
        rhs: Operand::Reg(r_eight_rows),
        ty: PtxType::U32,
    }));

    let emit_st = |alloc: &mut RegisterAllocator,
                   kernel: &mut PtxKernel,
                   base_off32: Register,
                   extra: u32,
                   src_reg: Register| {
        let rd_off = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Cvt {
            dst: rd_off,
            src: base_off32,
            dst_ty: PtxType::U64,
            src_ty: PtxType::U32,
        });
        let rd_addr = if extra == 0 {
            let rd = alloc.alloc(PtxType::U64);
            kernel.push(PtxInstruction::Arith(ArithOp::Add {
                dst: rd,
                lhs: Operand::Reg(out_rowslab_global),
                rhs: Operand::Reg(rd_off),
                ty: PtxType::U64,
            }));
            rd
        } else {
            let rd_tmp = alloc.alloc(PtxType::U64);
            kernel.push(PtxInstruction::Arith(ArithOp::Add {
                dst: rd_tmp,
                lhs: Operand::Reg(out_rowslab_global),
                rhs: Operand::Reg(rd_off),
                ty: PtxType::U64,
            }));
            let rd = alloc.alloc(PtxType::U64);
            kernel.push(PtxInstruction::Arith(ArithOp::Add {
                dst: rd,
                lhs: Operand::Reg(rd_tmp),
                rhs: Operand::ImmU32(extra),
                ty: PtxType::U64,
            }));
            rd
        };
        kernel.push(PtxInstruction::Memory(MemoryOp::StGlobal {
            addr: rd_addr,
            src: src_reg,
            ty: PtxType::F32,
        }));
    };
    emit_st(alloc, kernel, r_row0_off, 0, frag_d.regs[0]);
    emit_st(alloc, kernel, r_row0_off, 4, frag_d.regs[1]);
    emit_st(alloc, kernel, r_row8_off, 0, frag_d.regs[2]);
    emit_st(alloc, kernel, r_row8_off, 4, frag_d.regs[3]);
}

// ---------------------------------------------------------------------
// Gate C — full non-causal fused attention_tc.
// Final kernel: matmul1 → scale → softmax → cvt → matmul2 → output.
// ---------------------------------------------------------------------

/// Build the IR module for the final `attention_tc` kernel (Gate C
/// non-causal variant). Causal variant uses the same builder with
/// the `causal: bool` flag flipped (6.6b).
#[doc(hidden)]
pub fn build_attention_tc_module(sm: &str) -> PtxModule {
    let mut alloc = RegisterAllocator::new();
    let mut kernel = PtxKernel::new("attention_tc");

    kernel.add_param(PtxParam::pointer("q_ptr", PtxType::F16));
    kernel.add_param(PtxParam::pointer("k_ptr", PtxType::F16));
    kernel.add_param(PtxParam::pointer("v_ptr", PtxType::F16));
    kernel.add_param(PtxParam::pointer("out_ptr", PtxType::F32));
    kernel.add_param(PtxParam::scalar("seq_q", PtxType::U32));
    kernel.add_param(PtxParam::scalar("seq_k", PtxType::U32));
    kernel.add_param(PtxParam::scalar("d_k", PtxType::U32));
    kernel.add_param(PtxParam::scalar("d_v", PtxType::U32));
    kernel.add_param(PtxParam::scalar("inv_sqrt_dk", PtxType::F32));

    kernel.add_shared_decl(SharedDecl {
        name: "tile_q".to_string(),
        align: 4,
        size_bytes: TILE_Q_BYTES,
    });
    kernel.add_shared_decl(SharedDecl {
        name: "k_chunk".to_string(),
        align: 4,
        size_bytes: K_CHUNK_BYTES,
    });
    kernel.add_shared_decl(SharedDecl {
        name: "scores_tile".to_string(),
        align: 4,
        size_bytes: SCORES_TILE_BYTES,
    });
    kernel.add_shared_decl(SharedDecl {
        name: "probs_tile".to_string(),
        align: 4,
        size_bytes: PROBS_TILE_BYTES,
    });
    kernel.add_shared_decl(SharedDecl {
        name: "v_chunk".to_string(),
        align: 4,
        size_bytes: V_CHUNK_BYTES,
    });

    // Params + cvta
    let rd_q_param = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: rd_q_param,
        param_name: "q_ptr".to_string(),
        ty: PtxType::U64,
    }));
    let rd_k_param = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: rd_k_param,
        param_name: "k_ptr".to_string(),
        ty: PtxType::U64,
    }));
    let rd_v_param = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: rd_v_param,
        param_name: "v_ptr".to_string(),
        ty: PtxType::U64,
    }));
    let rd_out_param = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: rd_out_param,
        param_name: "out_ptr".to_string(),
        ty: PtxType::U64,
    }));
    let r_seq_k = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: r_seq_k,
        param_name: "seq_k".to_string(),
        ty: PtxType::U32,
    }));
    let r_d_k = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: r_d_k,
        param_name: "d_k".to_string(),
        ty: PtxType::U32,
    }));
    let r_d_v = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: r_d_v,
        param_name: "d_v".to_string(),
        ty: PtxType::U32,
    }));
    let r_inv_sqrt_dk = alloc.alloc(PtxType::F32);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: r_inv_sqrt_dk,
        param_name: "inv_sqrt_dk".to_string(),
        ty: PtxType::F32,
    }));

    let rd_q = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::CvtaToGlobal {
        dst: rd_q,
        src: rd_q_param,
    }));
    let rd_k_global = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::CvtaToGlobal {
        dst: rd_k_global,
        src: rd_k_param,
    }));
    let rd_v_global = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::CvtaToGlobal {
        dst: rd_v_global,
        src: rd_v_param,
    }));
    let rd_out = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::CvtaToGlobal {
        dst: rd_out,
        src: rd_out_param,
    }));

    let (r_tid, tid_instr) = special::tid_x(&mut alloc);
    kernel.push(tid_instr);
    let r_bidx = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_bidx,
        src: Operand::SpecialReg(SpecialReg::CtaidX),
        ty: PtxType::U32,
    });

    let r_block_row = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_block_row,
        lhs: Operand::Reg(r_bidx),
        rhs: Operand::ImmU32(BM),
        ty: PtxType::U32,
    }));

    // Strides (in bytes).
    let r_d_k_bytes = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_d_k_bytes,
        lhs: Operand::Reg(r_d_k),
        rhs: Operand::ImmU32(BYTES_PER_HALF),
        ty: PtxType::U32,
    }));
    let r_d_v_bytes = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_d_v_bytes,
        lhs: Operand::Reg(r_d_v),
        rhs: Operand::ImmU32(BYTES_PER_HALF),
        ty: PtxType::U32,
    }));
    let r_d_v_bytes_f32 = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_d_v_bytes_f32,
        lhs: Operand::Reg(r_d_v),
        rhs: Operand::ImmU32(BYTES_PER_F32),
        ty: PtxType::U32,
    }));
    let r_seq_k_bytes_f32 = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_seq_k_bytes_f32,
        lhs: Operand::Reg(r_seq_k),
        rhs: Operand::ImmU32(BYTES_PER_F32),
        ty: PtxType::U32,
    }));
    let r_seq_k_bytes_f16 = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_seq_k_bytes_f16,
        lhs: Operand::Reg(r_seq_k),
        rhs: Operand::ImmU32(BYTES_PER_HALF),
        ty: PtxType::U32,
    }));

    // Base global pointers adjusted for block_row.
    let rd_q_rowslab_off = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
        dst: rd_q_rowslab_off,
        lhs: Operand::Reg(r_block_row),
        rhs: Operand::Reg(r_d_k_bytes),
        src_ty: PtxType::U32,
    }));
    let rd_q_rowslab = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: rd_q_rowslab,
        lhs: Operand::Reg(rd_q),
        rhs: Operand::Reg(rd_q_rowslab_off),
        ty: PtxType::U64,
    }));
    let rd_out_rowslab_off = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
        dst: rd_out_rowslab_off,
        lhs: Operand::Reg(r_block_row),
        rhs: Operand::Reg(r_d_v_bytes_f32),
        src_ty: PtxType::U32,
    }));
    let rd_out_rowslab = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: rd_out_rowslab,
        lhs: Operand::Reg(rd_out),
        rhs: Operand::Reg(rd_out_rowslab_off),
        ty: PtxType::U64,
    }));

    // Shared bases.
    let r_tile_q = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_tile_q,
        src: Operand::SharedAddr("tile_q".to_string()),
        ty: PtxType::U32,
    });
    let r_k_chunk = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_k_chunk,
        src: Operand::SharedAddr("k_chunk".to_string()),
        ty: PtxType::U32,
    });
    let r_scores_tile = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_scores_tile,
        src: Operand::SharedAddr("scores_tile".to_string()),
        ty: PtxType::U32,
    });
    let r_probs_tile = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_probs_tile,
        src: Operand::SharedAddr("probs_tile".to_string()),
        ty: PtxType::U32,
    });
    let r_v_chunk = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_v_chunk,
        src: Operand::SharedAddr("v_chunk".to_string()),
        ty: PtxType::U32,
    });

    // Stage Q once.
    emit_stage_q(
        &mut alloc,
        &mut kernel,
        rd_q_rowslab,
        r_tile_q,
        r_tid,
        r_d_k,
    );
    kernel.push(PtxInstruction::Control(ControlOp::BarSync { barrier_id: 0 }));

    // ------------ MATMUL1: Q · Kᵀ · inv_sqrt_dk → scores_tile ------------

    let r_num_n_chunks = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: r_num_n_chunks,
        lhs: Operand::Reg(r_seq_k),
        rhs: Operand::ImmU32(BN),
        ty: PtxType::U32,
    }));
    let r_num_k_chunks = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: r_num_k_chunks,
        lhs: Operand::Reg(r_d_k),
        rhs: Operand::ImmU32(BK),
        ty: PtxType::U32,
    }));

    let r_n_chunk = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_n_chunk,
        src: Operand::ImmU32(0),
        ty: PtxType::U32,
    });
    kernel.push(PtxInstruction::Label("ATTN_TC_M1_N_LOOP".to_string()));

    let frag_s = alloc_c(&mut alloc);
    for r in &frag_s.regs {
        kernel.push(PtxInstruction::Mov {
            dst: *r,
            src: Operand::ImmF32(0.0),
            ty: PtxType::F32,
        });
    }

    let r_n_start = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_n_start,
        lhs: Operand::Reg(r_n_chunk),
        rhs: Operand::ImmU32(BN),
        ty: PtxType::U32,
    }));
    let rd_n_start_off = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
        dst: rd_n_start_off,
        lhs: Operand::Reg(r_n_start),
        rhs: Operand::Reg(r_d_k_bytes),
        src_ty: PtxType::U32,
    }));
    let rd_k_n_base = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: rd_k_n_base,
        lhs: Operand::Reg(rd_k_global),
        rhs: Operand::Reg(rd_n_start_off),
        ty: PtxType::U64,
    }));

    let r_k_chunk_idx = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_k_chunk_idx,
        src: Operand::ImmU32(0),
        ty: PtxType::U32,
    });
    kernel.push(PtxInstruction::Label("ATTN_TC_M1_K_LOOP".to_string()));

    let r_k_start_bytes = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_k_start_bytes,
        lhs: Operand::Reg(r_k_chunk_idx),
        rhs: Operand::ImmU32(BK * BYTES_PER_HALF),
        ty: PtxType::U32,
    }));
    let rd_k_start_bytes = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Cvt {
        dst: rd_k_start_bytes,
        src: r_k_start_bytes,
        dst_ty: PtxType::U64,
        src_ty: PtxType::U32,
    });
    let rd_k_chunk_src = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: rd_k_chunk_src,
        lhs: Operand::Reg(rd_k_n_base),
        rhs: Operand::Reg(rd_k_start_bytes),
        ty: PtxType::U64,
    }));
    emit_stage_k_chunk(
        &mut alloc,
        &mut kernel,
        rd_k_chunk_src,
        r_k_chunk,
        r_tid,
        r_d_k_bytes,
    );
    kernel.push(PtxInstruction::Control(ControlOp::BarSync { barrier_id: 0 }));

    let r_q_frag_base = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_q_frag_base,
        lhs: Operand::Reg(r_tile_q),
        rhs: Operand::Reg(r_k_start_bytes),
        ty: PtxType::U32,
    }));
    let frag_q = emit_load_fragment_a_runtime_stride(
        &mut alloc,
        &mut kernel,
        r_q_frag_base,
        r_tid,
        r_d_k_bytes,
    );
    let frag_kt = load_fragment_b_m16n8k16_shared_col(
        &mut alloc,
        &mut kernel,
        r_k_chunk,
        r_tid,
        32,
    );
    kernel.push(PtxInstruction::TensorCore(TensorCoreOp::MmaSync {
        d: frag_s,
        a: frag_q,
        b: frag_kt,
        c: frag_s,
        shape: MmaShape::M16N8K16,
        d_ty: PtxType::F32,
        a_ty: PtxType::F16,
        b_ty: PtxType::F16,
        c_ty: PtxType::F32,
    }));
    kernel.push(PtxInstruction::Control(ControlOp::BarSync { barrier_id: 0 }));

    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_k_chunk_idx,
        lhs: Operand::Reg(r_k_chunk_idx),
        rhs: Operand::ImmU32(1),
        ty: PtxType::U32,
    }));
    let p_k_loop = alloc.alloc(PtxType::Pred);
    kernel.push(PtxInstruction::Control(ControlOp::SetP {
        dst: p_k_loop,
        cmp_op: CmpOp::Lt,
        lhs: Operand::Reg(r_k_chunk_idx),
        rhs: Operand::Reg(r_num_k_chunks),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Control(ControlOp::BraPred {
        pred: p_k_loop,
        target: "ATTN_TC_M1_K_LOOP".to_string(),
        negate: false,
    }));

    // Scale.
    for r in &frag_s.regs {
        kernel.push(PtxInstruction::Arith(ArithOp::Mul {
            dst: *r,
            lhs: Operand::Reg(*r),
            rhs: Operand::Reg(r_inv_sqrt_dk),
            ty: PtxType::F32,
        }));
    }

    // Store scores tile chunk.
    emit_store_fragment_c_to_scores_tile(
        &mut alloc,
        &mut kernel,
        frag_s,
        r_scores_tile,
        r_tid,
        r_n_chunk,
        r_seq_k_bytes_f32,
    );

    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_n_chunk,
        lhs: Operand::Reg(r_n_chunk),
        rhs: Operand::ImmU32(1),
        ty: PtxType::U32,
    }));
    let p_n_loop = alloc.alloc(PtxType::Pred);
    kernel.push(PtxInstruction::Control(ControlOp::SetP {
        dst: p_n_loop,
        cmp_op: CmpOp::Lt,
        lhs: Operand::Reg(r_n_chunk),
        rhs: Operand::Reg(r_num_n_chunks),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Control(ControlOp::BraPred {
        pred: p_n_loop,
        target: "ATTN_TC_M1_N_LOOP".to_string(),
        negate: false,
    }));

    kernel.push(PtxInstruction::Control(ControlOp::BarSync { barrier_id: 0 }));

    // ------------ Softmax → probs_tile ------------

    emit_softmax_rows(
        &mut alloc,
        &mut kernel,
        r_scores_tile,
        r_probs_tile,
        r_tid,
        r_seq_k,
        r_seq_k_bytes_f32,
        r_seq_k_bytes_f16,
    );

    kernel.push(PtxInstruction::Control(ControlOp::BarSync { barrier_id: 0 }));

    // ------------ MATMUL2: probs · V → out ------------

    let r_num_dv_chunks = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: r_num_dv_chunks,
        lhs: Operand::Reg(r_d_v),
        rhs: Operand::ImmU32(BN),
        ty: PtxType::U32,
    }));
    let r_num_sk_chunks = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: r_num_sk_chunks,
        lhs: Operand::Reg(r_seq_k),
        rhs: Operand::ImmU32(BK),
        ty: PtxType::U32,
    }));

    let r_dv_chunk = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_dv_chunk,
        src: Operand::ImmU32(0),
        ty: PtxType::U32,
    });
    kernel.push(PtxInstruction::Label("ATTN_TC_M2_N_LOOP".to_string()));

    let frag_o = alloc_c(&mut alloc);
    for r in &frag_o.regs {
        kernel.push(PtxInstruction::Mov {
            dst: *r,
            src: Operand::ImmF32(0.0),
            ty: PtxType::F32,
        });
    }

    // d_v_start_bytes_f16 = dv_chunk * 8 * 2
    let r_dv_start_bytes_f16 = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_dv_start_bytes_f16,
        lhs: Operand::Reg(r_dv_chunk),
        rhs: Operand::ImmU32(BN * BYTES_PER_HALF),
        ty: PtxType::U32,
    }));
    let rd_dv_start_bytes_f16 = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Cvt {
        dst: rd_dv_start_bytes_f16,
        src: r_dv_start_bytes_f16,
        dst_ty: PtxType::U64,
        src_ty: PtxType::U32,
    });

    let r_sk_chunk = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Mov {
        dst: r_sk_chunk,
        src: Operand::ImmU32(0),
        ty: PtxType::U32,
    });
    kernel.push(PtxInstruction::Label("ATTN_TC_M2_K_LOOP".to_string()));

    // v_chunk source base: V + (sk_chunk * 16) * d_v_bytes + dv_chunk*8*2
    let r_sk_start = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_sk_start,
        lhs: Operand::Reg(r_sk_chunk),
        rhs: Operand::ImmU32(BK),
        ty: PtxType::U32,
    }));
    let rd_sk_row_off = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::MulWide {
        dst: rd_sk_row_off,
        lhs: Operand::Reg(r_sk_start),
        rhs: Operand::Reg(r_d_v_bytes),
        src_ty: PtxType::U32,
    }));
    let rd_v_base = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: rd_v_base,
        lhs: Operand::Reg(rd_v_global),
        rhs: Operand::Reg(rd_sk_row_off),
        ty: PtxType::U64,
    }));
    let rd_v_chunk_src = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: rd_v_chunk_src,
        lhs: Operand::Reg(rd_v_base),
        rhs: Operand::Reg(rd_dv_start_bytes_f16),
        ty: PtxType::U64,
    }));

    emit_stage_v_chunk(
        &mut alloc,
        &mut kernel,
        rd_v_chunk_src,
        r_v_chunk,
        r_tid,
        r_d_v_bytes,
    );
    kernel.push(PtxInstruction::Control(ControlOp::BarSync { barrier_id: 0 }));

    // probs fragment base: probs_tile + (sk_chunk * 16) * 2
    // (probs_tile has row stride = seq_k * 2 bytes; we address column
    // offset sk_chunk * 16 * 2 bytes within each row.)
    let r_sk_start_bytes_f16 = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_sk_start_bytes_f16,
        lhs: Operand::Reg(r_sk_chunk),
        rhs: Operand::ImmU32(BK * BYTES_PER_HALF),
        ty: PtxType::U32,
    }));
    let r_probs_frag_base = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_probs_frag_base,
        lhs: Operand::Reg(r_probs_tile),
        rhs: Operand::Reg(r_sk_start_bytes_f16),
        ty: PtxType::U32,
    }));
    let frag_p = emit_load_fragment_a_runtime_stride(
        &mut alloc,
        &mut kernel,
        r_probs_frag_base,
        r_tid,
        r_seq_k_bytes_f16,
    );
    let frag_v = load_fragment_b_m16n8k16_shared_col(
        &mut alloc,
        &mut kernel,
        r_v_chunk,
        r_tid,
        32,
    );
    kernel.push(PtxInstruction::TensorCore(TensorCoreOp::MmaSync {
        d: frag_o,
        a: frag_p,
        b: frag_v,
        c: frag_o,
        shape: MmaShape::M16N8K16,
        d_ty: PtxType::F32,
        a_ty: PtxType::F16,
        b_ty: PtxType::F16,
        c_ty: PtxType::F32,
    }));
    kernel.push(PtxInstruction::Control(ControlOp::BarSync { barrier_id: 0 }));

    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_sk_chunk,
        lhs: Operand::Reg(r_sk_chunk),
        rhs: Operand::ImmU32(1),
        ty: PtxType::U32,
    }));
    let p_sk_loop = alloc.alloc(PtxType::Pred);
    kernel.push(PtxInstruction::Control(ControlOp::SetP {
        dst: p_sk_loop,
        cmp_op: CmpOp::Lt,
        lhs: Operand::Reg(r_sk_chunk),
        rhs: Operand::Reg(r_num_sk_chunks),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Control(ControlOp::BraPred {
        pred: p_sk_loop,
        target: "ATTN_TC_M2_K_LOOP".to_string(),
        negate: false,
    }));

    emit_store_fragment_c_to_global_out(
        &mut alloc,
        &mut kernel,
        frag_o,
        rd_out_rowslab,
        r_tid,
        r_dv_chunk,
        r_d_v_bytes_f32,
    );

    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_dv_chunk,
        lhs: Operand::Reg(r_dv_chunk),
        rhs: Operand::ImmU32(1),
        ty: PtxType::U32,
    }));
    let p_dv_loop = alloc.alloc(PtxType::Pred);
    kernel.push(PtxInstruction::Control(ControlOp::SetP {
        dst: p_dv_loop,
        cmp_op: CmpOp::Lt,
        lhs: Operand::Reg(r_dv_chunk),
        rhs: Operand::Reg(r_num_dv_chunks),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Control(ControlOp::BraPred {
        pred: p_dv_loop,
        target: "ATTN_TC_M2_N_LOOP".to_string(),
        negate: false,
    }));

    kernel.push(PtxInstruction::Control(ControlOp::Ret));
    kernel.set_registers(alloc.into_allocated());

    let mut module = PtxModule::new(sm);
    module.add_kernel(kernel);
    module
}

/// Fused TC attention host API (non-causal). Sprint 6.6 internal
/// preview — promotes to stable `pub` at Phase 7 when FlashAttention-TC
/// lands and `attention_auto_tc` becomes the real user-facing
/// dispatcher.
#[allow(clippy::too_many_arguments)]
pub fn attention_tc(
    device: &KaioDevice,
    q: &GpuBuffer<f16>,
    k: &GpuBuffer<f16>,
    v: &GpuBuffer<f16>,
    out: &mut GpuBuffer<f32>,
    seq_q: u32,
    seq_k: u32,
    d_k: u32,
    d_v: u32,
) -> Result<()> {
    use cudarc::driver::{LaunchConfig, PushKernelArg};

    validate_attention_tc_dims(q, k, v, out, seq_q, seq_k, d_k, d_v)?;

    let info = device.info()?;
    let (major, minor) = info.compute_capability;
    let sm = format!("sm_{major}{minor}");
    let module = build_attention_tc_module(&sm);
    // ? propagates KaioError::Validation cleanly on pre-Ampere —
    // PtxModule::validate catches mma.sync's SM requirement before
    // any driver interaction. Distinct from KaioError::PtxLoad
    // which stays reserved for genuine ptxas / driver failures.
    let kmodule = device.load_module(&module)?;
    let func = kmodule.function("attention_tc")?;

    let inv_sqrt_dk: f32 = 1.0f32 / (d_k as f32).sqrt();

    let cfg = LaunchConfig {
        grid_dim: (seq_q / BM, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        device
            .stream()
            .launch_builder(func.inner())
            .arg(q.inner())
            .arg(k.inner())
            .arg(v.inner())
            .arg(out.inner_mut())
            .arg(&seq_q)
            .arg(&seq_k)
            .arg(&d_k)
            .arg(&d_v)
            .arg(&inv_sqrt_dk)
            .launch(cfg)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_a_module_builds_for_sm_89() {
        let module = build_attention_tc_gate_a_module("sm_89");
        // Trivial smoke: the module has a kernel named "attention_tc_gate_a".
        let emitted = {
            use kaio_core::emit::{Emit, PtxWriter};
            let mut w = PtxWriter::new();
            module.emit(&mut w).unwrap();
            w.finish()
        };
        assert!(
            emitted.contains(".entry attention_tc_gate_a"),
            "expected attention_tc_gate_a entry in emitted PTX"
        );
        assert!(
            emitted.contains("mma.sync.aligned.m16n8k16"),
            "expected mma.sync.m16n8k16 emission"
        );
    }

    #[test]
    fn gate_a_module_rejects_sm_70() {
        use kaio_core::ir::ValidationError;
        let module = build_attention_tc_gate_a_module("sm_70");
        let err = module.validate().expect_err("sm_70 should reject mma.sync");
        let ValidationError::SmTooLow {
            required, actual, ..
        } = err;
        assert_eq!(required, 80);
        assert_eq!(actual, 70);
    }
}
