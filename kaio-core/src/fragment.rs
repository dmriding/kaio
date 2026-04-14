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

// ----------------------------------------------------------------------------
// INT8 sibling types for `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`.
// ----------------------------------------------------------------------------
//
// Register counts match the fp16 m16n8k16 path by coincidence of the PTX ISA
// design (A = 4×.b32, B = 2×.b32, C/D = 4 accumulator), but the SEMANTICS
// differ:
//   - A/B registers hold packed i8x4 (four signed bytes per register), not
//     packed half2 (two fp16 values per register).
//   - C/D registers hold .s32 accumulator values, not .f32.
//
// Per AD3 in the Sprint 7.1 plan: new sibling types, no overloading of the
// f16 fragment infrastructure. Generic fragment types would hide the layout
// difference from readers; separate types document that the INT8 path is
// its own thing.
//
// Underscored names (`FragmentA_M16N8K32` etc.) mirror the existing design
// note's suggested shape-qualified naming. The `#[allow(non_camel_case_types)]`
// is isolated to these struct definitions so the convention deviation is
// visible in one place rather than drifting.

/// A-matrix fragment for `mma.sync.m16n8k32.s8`.
///
/// Holds 4 × `.b32` packed-i8x4 registers per thread (16 signed 8-bit values
/// per thread across the warp → 16×32 matrix in total).
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FragmentA_M16N8K32 {
    /// The four `%r` (`.b32`) registers holding the thread's A-fragment
    /// slice. Each register packs four signed i8 values. Layout is fixed
    /// by PTX ISA for the `.m16n8k32.s8` shape.
    pub regs: [Register; 4],
}

/// B-matrix fragment for `mma.sync.m16n8k32.s8`.
///
/// Holds 2 × `.b32` packed-i8x4 registers per thread (8 signed 8-bit values
/// per thread across the warp → 32×8 matrix in total).
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FragmentB_M16N8K32 {
    /// The two `%r` (`.b32`) registers holding the thread's B-fragment
    /// slice. Each register packs four signed i8 values.
    pub regs: [Register; 2],
}

/// C/D-matrix fragment for `mma.sync.m16n8k32.s32` (INT8 accumulator).
///
/// Holds 4 × `.s32` registers per thread (16×8 i32 matrix total across the
/// warp). Used as both the input accumulator (`C`) and output (`D`) of
/// the INT8 `mma.sync`. Distinct from [`FragmentC`] which holds `.f32`
/// for the fp16 path.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FragmentC_M16N8K32 {
    /// The four `%r` (`.s32`, declared `.b32` at register level) registers.
    pub regs: [Register; 4],
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

/// Allocate a fresh [`FragmentA_M16N8K32`] — four packed-i8x4 `.b32` registers.
#[allow(non_snake_case)]
pub fn alloc_a_M16N8K32(alloc: &mut RegisterAllocator) -> FragmentA_M16N8K32 {
    FragmentA_M16N8K32 {
        regs: [
            alloc.alloc_packed_int8x4(),
            alloc.alloc_packed_int8x4(),
            alloc.alloc_packed_int8x4(),
            alloc.alloc_packed_int8x4(),
        ],
    }
}

/// Allocate a fresh [`FragmentB_M16N8K32`] — two packed-i8x4 `.b32` registers.
#[allow(non_snake_case)]
pub fn alloc_b_M16N8K32(alloc: &mut RegisterAllocator) -> FragmentB_M16N8K32 {
    FragmentB_M16N8K32 {
        regs: [alloc.alloc_packed_int8x4(), alloc.alloc_packed_int8x4()],
    }
}

/// Allocate a fresh [`FragmentC_M16N8K32`] — four `.s32` accumulator registers.
///
/// Used for both the `C` accumulator input and the `D` output of the INT8
/// `mma.sync`. Distinct from [`alloc_c`] (which allocates `.f32` registers
/// for the fp16 path).
#[allow(non_snake_case)]
pub fn alloc_c_M16N8K32(alloc: &mut RegisterAllocator) -> FragmentC_M16N8K32 {
    use crate::types::PtxType;
    FragmentC_M16N8K32 {
        regs: [
            alloc.alloc(PtxType::S32),
            alloc.alloc(PtxType::S32),
            alloc.alloc(PtxType::S32),
            alloc.alloc(PtxType::S32),
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
/// output) to a **row-major fp32 matrix** in global memory.
///
/// The fragment represents a 16 × 8 output sub-tile. The caller
/// supplies the enclosing matrix's **row stride in bytes** via
/// `row_stride_bytes`:
///
/// - For a standalone 16 × 8 matrix, pass `32` (= 8 × `size_of::<f32>`).
/// - For a 16 × 8 tile inside a larger `M × N` fp32 matrix (e.g.
///   Sprint 6.3's `matmul_tc`), pass `N * 4`.
///
/// Emits 4 × `st.global.f32`, one per fragment register, at the
/// per-thread output positions.
pub fn store_fragment_c_m16n8k16_global_row(
    alloc: &mut crate::ir::RegisterAllocator,
    kernel: &mut PtxKernel,
    matrix_base_global: crate::ir::Register,
    tid_x: crate::ir::Register,
    fragment: FragmentC,
    row_stride_bytes: u32,
) {
    let (group_id, tig) = compute_group_thread_ids(alloc, kernel, tid_x);

    // Per-thread column pair occupies 2 * 4 = 8 bytes starting at
    // column 2*tig → tig * 8 bytes within the row.
    // base_off = groupID * row_stride_bytes + tig * 8
    let row_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: row_off,
        lhs: Operand::Reg(group_id),
        rhs: Operand::ImmU32(row_stride_bytes),
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

    // base_off_plus_8rows = base_off + 8 * row_stride_bytes
    let base_off_plus_8rows = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: base_off_plus_8rows,
        lhs: Operand::Reg(base_off),
        rhs: Operand::ImmU32(8 * row_stride_bytes),
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

// ============================================================================
// 4. m16n8k16 shared-source load helpers
// ============================================================================
//
// Sibling to the global-source helpers above. The NVIDIA thread-data
// mapping (PTX ISA §9.7.13.5.8.1) is defined in terms of **matrix
// positions**, independent of memory space — so the A-fragment values
// for thread `tid` are still:
//
//   reg[0] = half2( A[groupID,   2*tig    ], A[groupID,   2*tig + 1] )
//   reg[1] = half2( A[groupID+8, 2*tig    ], A[groupID+8, 2*tig + 1] )
//   reg[2] = half2( A[groupID,   2*tig + 8], A[groupID,   2*tig + 9] )
//   reg[3] = half2( A[groupID+8, 2*tig + 8], A[groupID+8, 2*tig + 9] )
//
// And the B-fragment values are:
//
//   reg[0] = half2( B[2*tig,     groupID], B[2*tig + 1, groupID] )
//   reg[1] = half2( B[2*tig + 8, groupID], B[2*tig + 9, groupID] )
//
// What changes vs the global-source helpers:
//
// - Base register is a 32-bit shared offset (`.u32`), not a 64-bit
//   global pointer (`.u64`). All address arithmetic stays in `%r`.
// - Opcode is `ld.shared.b32`, not `ld.global.b32`.
// - The tile may have a custom stride (caller-specified) — A might be
//   part of a larger shared tile with row stride ≠ 32 bytes; B
//   similarly with column stride ≠ 32 bytes. For Sprint 6.3 callers
//   pass `row_stride_bytes = col_stride_bytes = 32` (matching the
//   native mma shape), but the parameter is there for future work.
//
// Byte offset formulas (re-derived from the mapping above, with A
// row-major in shared and B column-major in shared):
//
// A (row-major, row stride = RS bytes):
//   reg[0] off = groupID * RS + 4*tig
//   reg[1] off = groupID * RS + 4*tig + 8*RS     (= reg[0] + 8 rows)
//   reg[2] off = groupID * RS + 4*tig + 16       (= reg[0] + 8 columns of fp16)
//   reg[3] off = groupID * RS + 4*tig + 8*RS + 16
//
// B (column-major, col stride = CS bytes):
//   reg[0] off = groupID * CS + 4*tig
//   reg[1] off = groupID * CS + 4*tig + 16       (= reg[0] + 8 rows of fp16)
//
// Implementation derivation: these formulas are algebraically identical
// in structure to the global-source helpers but expressed in u32
// arithmetic with a parameterized stride. `compute_group_thread_ids`
// is reused verbatim.

/// Compute an absolute `.u32` shared address from a `.u32` base offset
/// and a local `.u32` offset, optionally folding in a constant byte
/// offset.
///
/// Shared addresses are 32-bit on every SM ≥ 3.0 — no widening needed.
fn u32_shared_addr_from_offset(
    alloc: &mut crate::ir::RegisterAllocator,
    kernel: &mut PtxKernel,
    tile_base_shared: crate::ir::Register,
    offset_u32: crate::ir::Register,
    extra_bytes: u32,
) -> crate::ir::Register {
    let addr = alloc.alloc(PtxType::U32);
    if extra_bytes == 0 {
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: addr,
            lhs: Operand::Reg(tile_base_shared),
            rhs: Operand::Reg(offset_u32),
            ty: PtxType::U32,
        }));
    } else {
        let tmp = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: tmp,
            lhs: Operand::Reg(tile_base_shared),
            rhs: Operand::Reg(offset_u32),
            ty: PtxType::U32,
        }));
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: addr,
            lhs: Operand::Reg(tmp),
            rhs: Operand::ImmU32(extra_bytes),
            ty: PtxType::U32,
        }));
    }
    addr
}

/// Load one A-fragment for `mma.sync.m16n8k16.f16` from a 16×16
/// row-major fp16 tile in **shared** memory.
///
/// Emits `div.u32` + `rem.u32` to derive groupID / tig (unless
/// `group_tig_override` is `Some`), followed by four `ld.shared.b32`
/// at the per-thread fragment offsets derived above.
///
/// # Parameters
///
/// - `tile_base_shared` — `.u32` register holding the shared-memory
///   offset of the tile's row-0 column-0 element. Must already be
///   computed (typically via `mov.u32 %r, tile_symbol + block_offset`).
/// - `tid_x` — `.u32` register holding `%tid.x`. Unused when
///   `group_tig_override` is `Some`.
/// - `row_stride_bytes` — number of bytes between consecutive rows of
///   the A tile in shared memory. For the Sprint 6.3 kernel
///   (BM=16, BK=16, fp16), pass `32`. Future kernels with wider shared
///   tiles pass their actual row stride.
/// - `group_tig_override` — Sprint 6.7b D10 hoist. When `Some((g, t))`,
///   skip the internal `div.u32`/`rem.u32` emit and use the caller-
///   supplied `group_id` and `thread_id_in_group` registers instead.
///   Callers that invoke multiple fragment loads per warp per K-tile
///   (e.g. the multi-warp matmul_tc kernel's 2 FragmentA + 4 FragmentB
///   per K-iter) can compute these once at block start and pass them
///   here, saving 2 div/rem pairs per extra call. Pass `None` to keep
///   the pre-6.7b behaviour (loader computes them internally).
pub fn load_fragment_a_m16n8k16_shared_row(
    alloc: &mut crate::ir::RegisterAllocator,
    kernel: &mut PtxKernel,
    tile_base_shared: crate::ir::Register,
    tid_x: crate::ir::Register,
    row_stride_bytes: u32,
    group_tig_override: Option<(crate::ir::Register, crate::ir::Register)>,
) -> FragmentA {
    let (group_id, tig) = match group_tig_override {
        Some(pair) => pair,
        None => compute_group_thread_ids(alloc, kernel, tid_x),
    };

    // row_off = groupID * row_stride_bytes
    let row_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: row_off,
        lhs: Operand::Reg(group_id),
        rhs: Operand::ImmU32(row_stride_bytes),
        ty: PtxType::U32,
    }));

    // base_off = tig * 4 + row_off   (half2 = 4 bytes)
    let base_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mad {
        dst: base_off,
        a: Operand::Reg(tig),
        b: Operand::ImmU32(4),
        c: Operand::Reg(row_off),
        ty: PtxType::U32,
        mode: crate::instr::MadMode::Lo,
    }));

    // base_off_plus_8rows = base_off + 8 * row_stride_bytes
    let base_off_plus_8rows = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: base_off_plus_8rows,
        lhs: Operand::Reg(base_off),
        rhs: Operand::ImmU32(8 * row_stride_bytes),
        ty: PtxType::U32,
    }));

    // Absolute shared addresses for the four fragment registers:
    //   reg[0]: groupID   row, cols 2*tig   .. 2*tig+1  → base_off
    //   reg[1]: groupID+8 row, cols 2*tig   .. 2*tig+1  → base_off_plus_8rows
    //   reg[2]: groupID   row, cols 2*tig+8 .. 2*tig+9  → base_off + 16
    //   reg[3]: groupID+8 row, cols 2*tig+8 .. 2*tig+9  → base_off_plus_8rows + 16
    let addr0 = u32_shared_addr_from_offset(alloc, kernel, tile_base_shared, base_off, 0);
    let addr1 =
        u32_shared_addr_from_offset(alloc, kernel, tile_base_shared, base_off_plus_8rows, 0);
    let addr2 = u32_shared_addr_from_offset(alloc, kernel, tile_base_shared, base_off, 16);
    let addr3 =
        u32_shared_addr_from_offset(alloc, kernel, tile_base_shared, base_off_plus_8rows, 16);

    let frag = alloc_a(alloc);
    for (reg, addr) in frag.regs.iter().zip([addr0, addr1, addr2, addr3]) {
        kernel.push(PtxInstruction::Memory(MemoryOp::LdShared {
            dst: *reg,
            addr,
            // Load as .b32 — packed half2 representation.
            ty: PtxType::U32,
        }));
    }

    frag
}

/// Load one B-fragment for `mma.sync.m16n8k16.f16` from a 16×8
/// column-major fp16 tile in **shared** memory.
///
/// Column-major: column `j` occupies `col_stride_bytes` of contiguous
/// shared memory; within a column, consecutive fp16 rows are 2 bytes
/// apart. Two rows in the same column pack naturally into one `.b32`
/// half2 register via a single `ld.shared.b32`.
///
/// # Parameters
///
/// - `tile_base_shared` — `.u32` register holding the shared offset of
///   the tile's column-0 row-0 element.
/// - `tid_x` — `.u32` register holding `%tid.x`. Unused when
///   `group_tig_override` is `Some`.
/// - `col_stride_bytes` — number of bytes between consecutive columns
///   of the B tile in shared memory. Sprint 6.3 used `32` (BK=16 rows
///   × 2 bytes, tightly packed). Sprint 6.7b's multi-warp matmul_tc
///   passes `36` for bank-conflict relief — the extra 4-byte pad per
///   column breaks the 2-way bank conflict on fragment-B reads that
///   col stride 32 produces across a warp.
/// - `group_tig_override` — Sprint 6.7b D10 hoist. See docstring on
///   [`load_fragment_a_m16n8k16_shared_row`] for the rationale.
pub fn load_fragment_b_m16n8k16_shared_col(
    alloc: &mut crate::ir::RegisterAllocator,
    kernel: &mut PtxKernel,
    tile_base_shared: crate::ir::Register,
    tid_x: crate::ir::Register,
    col_stride_bytes: u32,
    group_tig_override: Option<(crate::ir::Register, crate::ir::Register)>,
) -> FragmentB {
    let (group_id, tig) = match group_tig_override {
        Some(pair) => pair,
        None => compute_group_thread_ids(alloc, kernel, tid_x),
    };

    // col_off = groupID * col_stride_bytes
    let col_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: col_off,
        lhs: Operand::Reg(group_id),
        rhs: Operand::ImmU32(col_stride_bytes),
        ty: PtxType::U32,
    }));

    // base_off = tig * 4 + col_off   (rows 2*tig and 2*tig+1 are adjacent,
    //                                  4 bytes / half2 pair)
    let base_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mad {
        dst: base_off,
        a: Operand::Reg(tig),
        b: Operand::ImmU32(4),
        c: Operand::Reg(col_off),
        ty: PtxType::U32,
        mode: crate::instr::MadMode::Lo,
    }));

    // reg[0]: column groupID, rows 2*tig   .. 2*tig+1  → base_off
    // reg[1]: column groupID, rows 2*tig+8 .. 2*tig+9  → base_off + 16
    //   (= +8 rows × 2 bytes each)
    let addr0 = u32_shared_addr_from_offset(alloc, kernel, tile_base_shared, base_off, 0);
    let addr1 = u32_shared_addr_from_offset(alloc, kernel, tile_base_shared, base_off, 16);

    let frag = alloc_b(alloc);
    for (reg, addr) in frag.regs.iter().zip([addr0, addr1]) {
        kernel.push(PtxInstruction::Memory(MemoryOp::LdShared {
            dst: *reg,
            addr,
            ty: PtxType::U32,
        }));
    }

    frag
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
        store_fragment_c_m16n8k16_global_row(&mut alloc, &mut kernel, base, tid, frag, 32);

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
    fn load_fragment_a_shared_emits_four_b32_shared_loads() {
        use crate::ir::{PtxKernel, RegisterAllocator};
        let mut alloc = RegisterAllocator::new();
        let mut kernel = PtxKernel::new("test");
        let base = alloc.alloc(PtxType::U32); // shared-memory offset
        let tid = alloc.alloc(PtxType::U32);
        let frag =
            load_fragment_a_m16n8k16_shared_row(&mut alloc, &mut kernel, base, tid, 32, None);
        assert_eq!(frag.regs.len(), 4);

        // Exactly four ld.shared.b32 (= ld.shared.u32) loads.
        let n_loads = kernel
            .body
            .iter()
            .filter(|instr| {
                matches!(
                    instr,
                    PtxInstruction::Memory(MemoryOp::LdShared {
                        ty: PtxType::U32,
                        ..
                    })
                )
            })
            .count();
        assert_eq!(n_loads, 4, "expected 4 ld.shared.b32 for FragmentA");

        // No ld.global in a shared-source load.
        let n_global = kernel
            .body
            .iter()
            .filter(|instr| matches!(instr, PtxInstruction::Memory(MemoryOp::LdGlobal { .. })))
            .count();
        assert_eq!(n_global, 0, "shared helper should not emit any ld.global");
    }

    #[test]
    fn load_fragment_b_shared_emits_two_b32_shared_loads() {
        use crate::ir::{PtxKernel, RegisterAllocator};
        let mut alloc = RegisterAllocator::new();
        let mut kernel = PtxKernel::new("test");
        let base = alloc.alloc(PtxType::U32);
        let tid = alloc.alloc(PtxType::U32);
        let frag =
            load_fragment_b_m16n8k16_shared_col(&mut alloc, &mut kernel, base, tid, 32, None);
        assert_eq!(frag.regs.len(), 2);

        let n_loads = kernel
            .body
            .iter()
            .filter(|instr| {
                matches!(
                    instr,
                    PtxInstruction::Memory(MemoryOp::LdShared {
                        ty: PtxType::U32,
                        ..
                    })
                )
            })
            .count();
        assert_eq!(n_loads, 2, "expected 2 ld.shared.b32 for FragmentB");
    }

    #[test]
    fn load_fragment_a_shared_respects_stride_parameter() {
        // Emit with a non-native stride and verify the generated Mul uses
        // that stride as the immediate, not the hardcoded 32. Catches a
        // regression where someone would hardcode 32 inside the helper.
        use crate::ir::{PtxKernel, RegisterAllocator};
        let mut alloc = RegisterAllocator::new();
        let mut kernel = PtxKernel::new("test");
        let base = alloc.alloc(PtxType::U32);
        let tid = alloc.alloc(PtxType::U32);
        let _ = load_fragment_a_m16n8k16_shared_row(&mut alloc, &mut kernel, base, tid, 128, None);

        // Expect a Mul with ImmU32(128) (row stride) and an Add with
        // ImmU32(1024) (8 * 128 = 8 rows' worth of bytes).
        let mut saw_stride_mul = false;
        let mut saw_eight_row_add = false;
        for instr in &kernel.body {
            if let PtxInstruction::Arith(ArithOp::Mul {
                rhs: Operand::ImmU32(128),
                ..
            }) = instr
            {
                saw_stride_mul = true;
            }
            if let PtxInstruction::Arith(ArithOp::Add {
                rhs: Operand::ImmU32(1024),
                ..
            }) = instr
            {
                saw_eight_row_add = true;
            }
        }
        assert!(
            saw_stride_mul,
            "shared A loader should multiply group_id by the caller-supplied row_stride_bytes"
        );
        assert!(
            saw_eight_row_add,
            "shared A loader should add 8*row_stride_bytes for the +8-rows address"
        );
    }

    #[test]
    fn load_fragment_b_shared_respects_stride_parameter() {
        use crate::ir::{PtxKernel, RegisterAllocator};
        let mut alloc = RegisterAllocator::new();
        let mut kernel = PtxKernel::new("test");
        let base = alloc.alloc(PtxType::U32);
        let tid = alloc.alloc(PtxType::U32);
        let _ = load_fragment_b_m16n8k16_shared_col(&mut alloc, &mut kernel, base, tid, 96, None);

        let mut saw_stride_mul = false;
        for instr in &kernel.body {
            if let PtxInstruction::Arith(ArithOp::Mul {
                rhs: Operand::ImmU32(96),
                ..
            }) = instr
            {
                saw_stride_mul = true;
            }
        }
        assert!(
            saw_stride_mul,
            "shared B loader should multiply group_id by the caller-supplied col_stride_bytes"
        );
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
