//! Tensor-core fragment register containers for `mma.sync` operands.
//!
//! A "fragment" is the warp-level distributed slice of a matrix that
//! `mma.sync` consumes. For the `m16n8k16.f16.f16.f32` shape (the only
//! shape Phase 6 supports), each of the 32 threads in a warp holds a
//! fixed slice of the input matrices in registers:
//!
//! - **FragmentA** — 4 `.b32` packed-half2 registers per thread
//!   (8 fp16 values total)
//! - **FragmentB** — 2 `.b32` packed-half2 registers per thread
//!   (4 fp16 values total)
//! - **FragmentC / FragmentD** — 4 `.f32` registers per thread
//!
//! The thread-data mapping from matrix elements to fragment registers
//! is specified by NVIDIA in the PTX ISA (§9.7.13.5.8.1, "Matrix
//! Fragments for mma.m16n8k16 with floating point data types"). Helpers
//! that load/store fragments from global memory using this mapping are
//! defined later in this module.
//!
//! ## Why A/B use `.b32`, not `.f16`
//!
//! `mma.sync.m16n8k16.f16.f16` packs two fp16 values per 32-bit operand
//! register. nvcc emits `%r0..%r3` (the `.b32` register class) for the
//! A fragment, not `%h0..%h7` (the `.f16` class). The `.f16` class is
//! for single-value half arithmetic — it is not the register class
//! `mma.sync` wants. Sprint 6.1 added `%h`/`%hb` for completeness, but
//! fragments deliberately use `%r`.
//!
//! ## Design
//!
//! - Fragments are **pure register containers** — no invariant to
//!   protect, fixed register counts set by the PTX ISA. Fields are
//!   `pub` by design. Call sites wire `fragment_a.regs[0]` directly
//!   into `TensorCoreOp::MmaSync` operand positions.
//! - Load/store/alloc helpers are **free functions**, not methods. A
//!   future shared-memory-source load is a sibling function
//!   (`load_fragment_a_m16n8k16_shared_row`), not a method on the
//!   struct.
//! - No generics over shape — we support exactly `m16n8k16` today. A
//!   second shape gets a sibling type (`FragmentA_m16n8k8`), not
//!   `Fragment<Shape>`.

// ============================================================================
// 1. Fragment types — pure register bags
// ============================================================================

use crate::ir::{Register, RegisterAllocator};

/// A-matrix fragment for `mma.sync.m16n8k16.f16`.
///
/// Holds 4 × `.b32` packed-half2 registers per thread (8 fp16 values
/// per thread across the warp → 16×16 matrix in total).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FragmentA {
    /// The four `%r` (`.b32`) registers holding the thread's A-fragment
    /// slice. Each register packs two fp16 values. Layout is fixed by
    /// PTX ISA §9.7.13.5.8.1.
    pub regs: [Register; 4],
}

/// B-matrix fragment for `mma.sync.m16n8k16.f16`.
///
/// Holds 2 × `.b32` packed-half2 registers per thread (4 fp16 values
/// per thread across the warp → 16×8 matrix in total).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FragmentB {
    /// The two `%r` (`.b32`) registers holding the thread's B-fragment
    /// slice. Each register packs two fp16 values. Layout is fixed by
    /// PTX ISA §9.7.13.5.8.1.
    pub regs: [Register; 2],
}

/// C/D-matrix fragment for `mma.sync.m16n8k16.f32` (accumulator / output).
///
/// Holds 4 × `.f32` registers per thread (16×8 fp32 matrix total across
/// the warp). Used as both the input accumulator (`C`) and output (`D`)
/// of `mma.sync`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FragmentC {
    /// The four `%f` (`.f32`) registers. Layout is fixed by PTX ISA
    /// §9.7.13.5.8.1.
    pub regs: [Register; 4],
}

// ============================================================================
// 2. Fragment alloc helpers (free functions)
// ============================================================================

/// Allocate a fresh [`FragmentA`] — four packed-half2 `.b32` registers.
pub fn alloc_a(alloc: &mut RegisterAllocator) -> FragmentA {
    FragmentA {
        regs: [
            alloc.alloc_packed_half2(),
            alloc.alloc_packed_half2(),
            alloc.alloc_packed_half2(),
            alloc.alloc_packed_half2(),
        ],
    }
}

/// Allocate a fresh [`FragmentB`] — two packed-half2 `.b32` registers.
pub fn alloc_b(alloc: &mut RegisterAllocator) -> FragmentB {
    FragmentB {
        regs: [alloc.alloc_packed_half2(), alloc.alloc_packed_half2()],
    }
}

/// Allocate a fresh [`FragmentC`] — four `.f32` registers.
///
/// Used for both the `C` accumulator input and the `D` output of
/// `mma.sync` — the PTX spec treats them as the same fragment type.
pub fn alloc_c(alloc: &mut RegisterAllocator) -> FragmentC {
    use crate::types::PtxType;
    FragmentC {
        regs: [
            alloc.alloc(PtxType::F32),
            alloc.alloc(PtxType::F32),
            alloc.alloc(PtxType::F32),
            alloc.alloc(PtxType::F32),
        ],
    }
}

// ============================================================================
// 3. m16n8k16 global-source load / global-dest store helpers
// ============================================================================
//
// Thread-data layout (PTX ISA §9.7.13.5.8.1, "Matrix Fragments for
// mma.m16n8k16 with floating point data types"):
//
// Let `groupID = tid / 4` and `threadID_in_group = tid % 4` for the
// thread's `%tid.x` within a 32-thread warp.
//
// For a 16×16 row-major **A** matrix, each thread's 8 fp16 values are:
//   a[0] = A[groupID,   2*tig    ]      a[1] = A[groupID,   2*tig + 1]
//   a[2] = A[groupID+8, 2*tig    ]      a[3] = A[groupID+8, 2*tig + 1]
//   a[4] = A[groupID,   2*tig + 8]      a[5] = A[groupID,   2*tig + 9]
//   a[6] = A[groupID+8, 2*tig + 8]      a[7] = A[groupID+8, 2*tig + 9]
// Each pair `{a[2k], a[2k+1]}` lives in one packed-half2 `.b32` register
// and is contiguous in memory — loadable with a single `ld.global.b32`.
//
// For a 16×8 column-major **B** matrix, each thread's 4 fp16 values are:
//   b[0] = B[2*tig,     groupID]        b[1] = B[2*tig + 1, groupID]
//   b[2] = B[2*tig + 8, groupID]        b[3] = B[2*tig + 9, groupID]
// Again, each pair is contiguous in memory — one `ld.global.b32` per
// register.
//
// For a 16×8 row-major **C/D** matrix (fp32 accumulator / output), each
// thread's 4 fp32 values are:
//   d[0] = D[groupID,   2*tig    ]      d[1] = D[groupID,   2*tig + 1]
//   d[2] = D[groupID+8, 2*tig    ]      d[3] = D[groupID+8, 2*tig + 1]
// Each value is loaded/stored with a separate `ld.global.f32` /
// `st.global.f32` (no packing).

use crate::instr::{ArithOp, MemoryOp};
use crate::ir::{Operand, PtxInstruction, PtxKernel};
use crate::types::PtxType;

/// Byte stride between consecutive rows of the A (16×16, row-major, fp16)
/// or B (16×8, column-major, fp16) matrix — 32 bytes (16 × 2).
const FRAGMENT_HALF_ROW_STRIDE_BYTES: i32 = 32;
/// Byte stride between consecutive rows of the C/D (16×8, row-major, fp32)
/// matrix — 32 bytes (8 × 4).
const FRAGMENT_F32_ROW_STRIDE_BYTES: i32 = 32;

/// Compute `(groupID, threadID_in_group)` from a `tid_x` register.
///
/// Emits two instructions: `div.u32 gid, tid, 4` and `rem.u32 tig, tid, 4`.
/// Returns the two allocated `.u32` registers. Integer div/rem on a
/// constant power of two is accepted PTX — the driver lowers it to
/// shift/mask.
fn compute_group_thread_ids(
    alloc: &mut crate::ir::RegisterAllocator,
    kernel: &mut PtxKernel,
    tid_x: crate::ir::Register,
) -> (crate::ir::Register, crate::ir::Register) {
    let group_id = alloc.alloc(PtxType::U32);
    let thread_id_in_group = alloc.alloc(PtxType::U32);

    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: group_id,
        lhs: Operand::Reg(tid_x),
        rhs: Operand::ImmU32(4),
        ty: PtxType::U32,
    }));
    kernel.push(PtxInstruction::Arith(ArithOp::Rem {
        dst: thread_id_in_group,
        lhs: Operand::Reg(tid_x),
        rhs: Operand::ImmU32(4),
        ty: PtxType::U32,
    }));

    (group_id, thread_id_in_group)
}

/// Emit `addr_u64 = base + (row_off_u32 as u64) + extra_byte_offset`,
/// returning the final `.u64` address register.
///
/// The caller has already computed `row_off_u32` (which is
/// `row_index * row_stride_bytes + col_index * elem_bytes`, cast-free).
/// We widen to u64 and add the u64 matrix base pointer, plus an optional
/// compile-time byte offset for the second-column pair / next-row pair.
fn u64_addr_from_u32_offset(
    alloc: &mut crate::ir::RegisterAllocator,
    kernel: &mut PtxKernel,
    base: crate::ir::Register,
    offset_u32: crate::ir::Register,
    extra_bytes: u32,
) -> crate::ir::Register {
    let off64 = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Cvt {
        dst: off64,
        src: offset_u32,
        dst_ty: PtxType::U64,
        src_ty: PtxType::U32,
    });
    let addr = alloc.alloc(PtxType::U64);
    if extra_bytes == 0 {
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: addr,
            lhs: Operand::Reg(base),
            rhs: Operand::Reg(off64),
            ty: PtxType::U64,
        }));
    } else {
        // addr = base + off64 + extra_bytes, via a tmp.
        let tmp = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: tmp,
            lhs: Operand::Reg(base),
            rhs: Operand::Reg(off64),
            ty: PtxType::U64,
        }));
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: addr,
            lhs: Operand::Reg(tmp),
            rhs: Operand::ImmU32(extra_bytes),
            ty: PtxType::U64,
        }));
    }
    addr
}

/// Load one A-fragment for `mma.sync.m16n8k16.f16` from a 16×16 row-major
/// fp16 matrix in global memory.
///
/// Uses the canonical NVIDIA thread-data mapping (see module-level
/// comment). Emits:
///
/// - `div.u32` + `rem.u32` to split `tid_x` into `groupID` and
///   `threadID_in_group`
/// - Four `ld.global.b32` instructions, one per packed-half2 register,
///   at the per-thread fragment offsets
///
/// `matrix_base_global` must point to row 0 column 0 of A in
/// **global address space** (apply `cvta.to.global.u64` before calling
/// if the pointer came from `ld.param`).
pub fn load_fragment_a_m16n8k16_global_row(
    alloc: &mut crate::ir::RegisterAllocator,
    kernel: &mut PtxKernel,
    matrix_base_global: crate::ir::Register,
    tid_x: crate::ir::Register,
) -> FragmentA {
    let (group_id, tig) = compute_group_thread_ids(alloc, kernel, tid_x);

    // off0_u32 = groupID * 32 + threadID_in_group * 4
    //   (row stride = 32 bytes, each half2 = 4 bytes)
    let row_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: row_off,
        lhs: Operand::Reg(group_id),
        rhs: Operand::ImmU32(FRAGMENT_HALF_ROW_STRIDE_BYTES as u32),
        ty: PtxType::U32,
    }));
    let base_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mad {
        dst: base_off,
        a: Operand::Reg(tig),
        b: Operand::ImmU32(4),
        c: Operand::Reg(row_off),
        ty: PtxType::U32,
        mode: crate::instr::MadMode::Lo,
    }));

    // off1_u32 = base_off + 8 rows (= 8 * 32 = 256 bytes)
    let base_off_plus_8rows = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: base_off_plus_8rows,
        lhs: Operand::Reg(base_off),
        rhs: Operand::ImmU32((8 * FRAGMENT_HALF_ROW_STRIDE_BYTES) as u32),
        ty: PtxType::U32,
    }));

    // Addresses for reg[0..3]:
    //   reg[0]: groupID row, col 2*tig       → base_off
    //   reg[1]: groupID+8 row, col 2*tig     → base_off_plus_8rows
    //   reg[2]: groupID row, col 2*tig+8     → base_off + 16 (8 halves * 2 bytes)
    //   reg[3]: groupID+8 row, col 2*tig+8   → base_off_plus_8rows + 16
    let addr0 = u64_addr_from_u32_offset(alloc, kernel, matrix_base_global, base_off, 0);
    let addr1 = u64_addr_from_u32_offset(alloc, kernel, matrix_base_global, base_off_plus_8rows, 0);
    let addr2 = u64_addr_from_u32_offset(alloc, kernel, matrix_base_global, base_off, 16);
    let addr3 =
        u64_addr_from_u32_offset(alloc, kernel, matrix_base_global, base_off_plus_8rows, 16);

    let frag = alloc_a(alloc);
    for (reg, addr) in frag.regs.iter().zip([addr0, addr1, addr2, addr3]) {
        kernel.push(PtxInstruction::Memory(MemoryOp::LdGlobal {
            dst: *reg,
            addr,
            // Load as .b32 — the packed-half2 representation mma.sync expects.
            ty: PtxType::U32,
        }));
    }

    frag
}

/// Load one B-fragment for `mma.sync.m16n8k16.f16` from a 16×8
/// column-major fp16 matrix in global memory.
///
/// Column-major means column `j` occupies 16 consecutive fp16 values;
/// rows `2*tig` and `2*tig+1` of column `groupID` are therefore
/// contiguous in memory and loadable as a single `ld.global.b32`.
pub fn load_fragment_b_m16n8k16_global_col(
    alloc: &mut crate::ir::RegisterAllocator,
    kernel: &mut PtxKernel,
    matrix_base_global: crate::ir::Register,
    tid_x: crate::ir::Register,
) -> FragmentB {
    let (group_id, tig) = compute_group_thread_ids(alloc, kernel, tid_x);

    // Column stride = 16 fp16 elements = 32 bytes.
    // base_off = groupID * 32 + threadID_in_group * 4
    let col_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: col_off,
        lhs: Operand::Reg(group_id),
        rhs: Operand::ImmU32(FRAGMENT_HALF_ROW_STRIDE_BYTES as u32),
        ty: PtxType::U32,
    }));
    let base_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mad {
        dst: base_off,
        a: Operand::Reg(tig),
        b: Operand::ImmU32(4),
        c: Operand::Reg(col_off),
        ty: PtxType::U32,
        mode: crate::instr::MadMode::Lo,
    }));

    // reg[0]: column groupID, rows 2*tig and 2*tig+1 → base_off
    // reg[1]: column groupID, rows 2*tig+8 and 2*tig+9 → base_off + 16
    let addr0 = u64_addr_from_u32_offset(alloc, kernel, matrix_base_global, base_off, 0);
    let addr1 = u64_addr_from_u32_offset(alloc, kernel, matrix_base_global, base_off, 16);

    let frag = alloc_b(alloc);
    for (reg, addr) in frag.regs.iter().zip([addr0, addr1]) {
        kernel.push(PtxInstruction::Memory(MemoryOp::LdGlobal {
            dst: *reg,
            addr,
            ty: PtxType::U32,
        }));
    }

    frag
}

/// Store one C/D-fragment for `mma.sync.m16n8k16.f32` (fp32 accumulator /
/// output) to a 16×8 row-major fp32 matrix in global memory.
///
/// Emits 4 × `st.global.f32`, one per fragment register, at the
/// per-thread output positions.
pub fn store_fragment_c_m16n8k16_global_row(
    alloc: &mut crate::ir::RegisterAllocator,
    kernel: &mut PtxKernel,
    matrix_base_global: crate::ir::Register,
    tid_x: crate::ir::Register,
    fragment: FragmentC,
) {
    let (group_id, tig) = compute_group_thread_ids(alloc, kernel, tid_x);

    // Row stride = 8 fp32 = 32 bytes. Per-thread column pair occupies
    // 2 * 4 = 8 bytes starting at column 2*tig → tig * 8 bytes.
    // base_off = groupID * 32 + threadID_in_group * 8
    let row_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: row_off,
        lhs: Operand::Reg(group_id),
        rhs: Operand::ImmU32(FRAGMENT_F32_ROW_STRIDE_BYTES as u32),
        ty: PtxType::U32,
    }));
    let base_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mad {
        dst: base_off,
        a: Operand::Reg(tig),
        b: Operand::ImmU32(8),
        c: Operand::Reg(row_off),
        ty: PtxType::U32,
        mode: crate::instr::MadMode::Lo,
    }));

    // base_off_plus_8rows = base_off + 8 * 32 = +256 bytes
    let base_off_plus_8rows = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: base_off_plus_8rows,
        lhs: Operand::Reg(base_off),
        rhs: Operand::ImmU32((8 * FRAGMENT_F32_ROW_STRIDE_BYTES) as u32),
        ty: PtxType::U32,
    }));

    // Addresses for reg[0..3]:
    //   reg[0]: d[groupID,   2*tig    ] → base_off
    //   reg[1]: d[groupID,   2*tig + 1] → base_off + 4
    //   reg[2]: d[groupID+8, 2*tig    ] → base_off_plus_8rows
    //   reg[3]: d[groupID+8, 2*tig + 1] → base_off_plus_8rows + 4
    let addr0 = u64_addr_from_u32_offset(alloc, kernel, matrix_base_global, base_off, 0);
    let addr1 = u64_addr_from_u32_offset(alloc, kernel, matrix_base_global, base_off, 4);
    let addr2 = u64_addr_from_u32_offset(alloc, kernel, matrix_base_global, base_off_plus_8rows, 0);
    let addr3 = u64_addr_from_u32_offset(alloc, kernel, matrix_base_global, base_off_plus_8rows, 4);

    for (reg, addr) in fragment.regs.iter().zip([addr0, addr1, addr2, addr3]) {
        kernel.push(PtxInstruction::Memory(MemoryOp::StGlobal {
            addr,
            src: *reg,
            ty: PtxType::F32,
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{PtxType, RegKind};

    #[test]
    fn alloc_a_gives_four_b32_regs() {
        let mut a = RegisterAllocator::new();
        let frag = alloc_a(&mut a);
        for r in &frag.regs {
            assert_eq!(r.kind, RegKind::R);
            // alloc_packed_half2 tags ptx_type as U32 (b32 at PTX level)
            assert_eq!(r.ptx_type, PtxType::U32);
        }
        // Indices are sequential
        assert_eq!(frag.regs[0].index, 0);
        assert_eq!(frag.regs[1].index, 1);
        assert_eq!(frag.regs[2].index, 2);
        assert_eq!(frag.regs[3].index, 3);
    }

    #[test]
    fn alloc_b_gives_two_b32_regs() {
        let mut a = RegisterAllocator::new();
        let frag = alloc_b(&mut a);
        for r in &frag.regs {
            assert_eq!(r.kind, RegKind::R);
            assert_eq!(r.ptx_type, PtxType::U32);
        }
    }

    #[test]
    fn alloc_c_gives_four_f32_regs() {
        let mut a = RegisterAllocator::new();
        let frag = alloc_c(&mut a);
        for r in &frag.regs {
            assert_eq!(r.kind, RegKind::F);
            assert_eq!(r.ptx_type, PtxType::F32);
        }
    }

    #[test]
    fn load_fragment_a_emits_four_b32_loads() {
        use crate::ir::{PtxKernel, RegisterAllocator};
        let mut alloc = RegisterAllocator::new();
        let mut kernel = PtxKernel::new("test");
        let base = alloc.alloc(PtxType::U64);
        let tid = alloc.alloc(PtxType::U32);
        let frag = load_fragment_a_m16n8k16_global_row(&mut alloc, &mut kernel, base, tid);

        // Four fragment registers allocated.
        assert_eq!(frag.regs.len(), 4);

        // Count ld.global.b32 (= ld.global.u32) loads: there must be
        // exactly four — one per A-fragment packed-half2 register.
        let n_loads = kernel
            .body
            .iter()
            .filter(|instr| {
                matches!(
                    instr,
                    PtxInstruction::Memory(MemoryOp::LdGlobal {
                        ty: PtxType::U32,
                        ..
                    })
                )
            })
            .count();
        assert_eq!(n_loads, 4, "expected 4 ld.global.b32 for FragmentA");
    }

    #[test]
    fn load_fragment_b_emits_two_b32_loads() {
        use crate::ir::{PtxKernel, RegisterAllocator};
        let mut alloc = RegisterAllocator::new();
        let mut kernel = PtxKernel::new("test");
        let base = alloc.alloc(PtxType::U64);
        let tid = alloc.alloc(PtxType::U32);
        let frag = load_fragment_b_m16n8k16_global_col(&mut alloc, &mut kernel, base, tid);
        assert_eq!(frag.regs.len(), 2);

        let n_loads = kernel
            .body
            .iter()
            .filter(|instr| {
                matches!(
                    instr,
                    PtxInstruction::Memory(MemoryOp::LdGlobal {
                        ty: PtxType::U32,
                        ..
                    })
                )
            })
            .count();
        assert_eq!(n_loads, 2, "expected 2 ld.global.b32 for FragmentB");
    }

    #[test]
    fn store_fragment_c_emits_four_f32_stores() {
        use crate::ir::{PtxKernel, RegisterAllocator};
        let mut alloc = RegisterAllocator::new();
        let mut kernel = PtxKernel::new("test");
        let base = alloc.alloc(PtxType::U64);
        let tid = alloc.alloc(PtxType::U32);
        let frag = alloc_c(&mut alloc);
        store_fragment_c_m16n8k16_global_row(&mut alloc, &mut kernel, base, tid, frag);

        let n_stores = kernel
            .body
            .iter()
            .filter(|instr| {
                matches!(
                    instr,
                    PtxInstruction::Memory(MemoryOp::StGlobal {
                        ty: PtxType::F32,
                        ..
                    })
                )
            })
            .count();
        assert_eq!(n_stores, 4, "expected 4 st.global.f32 for FragmentC");
    }

    #[test]
    fn fragment_counters_independent() {
        // A and B both allocate from %r; C allocates from %f — indices
        // should be sequential within each kind, not collide across.
        let mut a = RegisterAllocator::new();
        let fa = alloc_a(&mut a);
        let fb = alloc_b(&mut a);
        let fc = alloc_c(&mut a);

        // A used %r0..%r3, B used %r4..%r5
        assert_eq!(fa.regs[0].index, 0);
        assert_eq!(fa.regs[3].index, 3);
        assert_eq!(fb.regs[0].index, 4);
        assert_eq!(fb.regs[1].index, 5);
        // C has its own %f counter starting at 0
        assert_eq!(fc.regs[0].index, 0);
        assert_eq!(fc.regs[3].index, 3);
    }
}
