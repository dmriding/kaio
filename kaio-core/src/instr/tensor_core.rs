//! Tensor-core PTX operations.
//!
//! This module hosts the warp-collective tensor-core instructions. The
//! `mma.sync` compute variants all use Ampere+ (SM 8.0) shapes —
//! `m16n8k16` (fp16/bf16 × fp32 accumulate, the shape used by CUTLASS,
//! cuBLAS, and every production fp16 matmul since 2020) and `m16n8k32`
//! (s8 × s32 accumulate). The `ldmatrix` fragment loader is the one
//! Turing-capable (SM 7.5) op in the file — it is a memory-side
//! primitive that predates the Ampere mma shapes, so a module using
//! `ldmatrix` alone validates at sm_75 while any kernel that feeds it
//! into an mma still requires sm_80 via the mma's own gate.
//!
//! Earlier mma shapes (Volta `m8n8k4`, Turing `m16n8k8`) have different
//! fragment layouts and are out of scope.

use std::fmt;

use crate::emit::{Emit, PtxWriter};
use crate::fragment::{
    FragmentA_BF16, FragmentA_F16, FragmentA_M16N8K32, FragmentB_BF16, FragmentB_F16,
    FragmentB_M16N8K32, FragmentC, FragmentC_M16N8K32,
};
use crate::types::PtxType;

/// The shape of an `mma.sync` instruction.
///
/// Each variant corresponds to a distinct hardware tile geometry. Adding a
/// new shape requires a new enum variant **and** a new set of fragment types
/// because the per-thread register distribution is shape-dependent. See
/// [`crate::fragment`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmaShape {
    /// 16×16 × 16×8 → 16×8, fp16 / bf16 inputs with fp32 accumulate (Ampere+).
    M16N8K16,
    /// 16×32 × 32×8 → 16×8, signed int8 inputs with int32 accumulate (Ampere+).
    ///
    /// Used by `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`. K is 32,
    /// not 16 — twice the K-tile of the fp16 path. Introduced in
    /// Sprint 7.1 for INT8 dequantize-matmul.
    M16N8K32,
}

impl MmaShape {
    /// PTX shape token (e.g. `"m16n8k16"`).
    pub fn ptx_token(&self) -> &'static str {
        match self {
            Self::M16N8K16 => "m16n8k16",
            Self::M16N8K32 => "m16n8k32",
        }
    }

    /// Minimum SM version required to execute this shape.
    ///
    /// Used by [`crate::ir::PtxModule::validate`] to reject kernels that
    /// emit tensor-core ops against too-low a target SM.
    pub fn min_sm(&self) -> u32 {
        match self {
            Self::M16N8K16 => 80,
            Self::M16N8K32 => 80,
        }
    }
}

/// Destination registers for [`TensorCoreOp::LdMatrix`] — the `.num`
/// qualifier and the register count are inseparable by construction, so
/// a register-count/qualifier mismatch is unrepresentable.
///
/// Each destination register receives one 8×8 `.b16` matrix distributed
/// across the warp (per lane: one `.b32` word = two 16-bit elements).
/// No `.x1` arm yet — add it when a consumer exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LdMatrixDst {
    /// `.x2` — two 8×8 matrices into two `.b32` registers per thread.
    X2([crate::ir::Register; 2]),
    /// `.x4` — four 8×8 matrices into four `.b32` registers per thread.
    X4([crate::ir::Register; 4]),
}

impl LdMatrixDst {
    /// PTX `.num` token (`"x2"` / `"x4"`).
    pub fn num_token(&self) -> &'static str {
        match self {
            Self::X2(_) => "x2",
            Self::X4(_) => "x4",
        }
    }

    /// The destination registers as a slice, in matrix order.
    pub fn regs(&self) -> &[crate::ir::Register] {
        match self {
            Self::X2(regs) => regs,
            Self::X4(regs) => regs,
        }
    }
}

/// Tensor-core PTX instruction variants.
///
/// Warp-collective operations — `mma.sync` is executed cooperatively by
/// all 32 threads in a warp with a rigid NVIDIA-defined register layout,
/// and `ldmatrix` cooperatively loads fragment registers in that same
/// layout. See [`crate::fragment`] for the per-thread register
/// distribution.
#[derive(Debug, Clone)]
pub enum TensorCoreOp {
    /// Synchronous matrix-multiply-accumulate:
    /// `mma.sync.aligned.{shape}.row.col.{d_ty}.{a_ty}.{b_ty}.{c_ty}`
    /// `{d_regs}, {a_regs}, {b_regs}, {c_regs};`
    ///
    /// Computes `D = A * B + C` across the warp. A is row-major, B is
    /// column-major (the `.row.col` modifiers are fixed for the fp16
    /// `m16n8k16` form).
    ///
    /// Example emission:
    /// ```text
    /// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    ///     {%f4,%f5,%f6,%f7},
    ///     {%r0,%r1,%r2,%r3},
    ///     {%r4,%r5},
    ///     {%f0,%f1,%f2,%f3};
    /// ```
    MmaSync {
        /// Destination (D) fragment — `.f32` accumulator output.
        d: FragmentC,
        /// Input A fragment — `.b32` packed half2 registers.
        a: FragmentA_F16,
        /// Input B fragment — `.b32` packed half2 registers.
        b: FragmentB_F16,
        /// Input C fragment — `.f32` accumulator input.
        c: FragmentC,
        /// Matrix shape (currently only [`MmaShape::M16N8K16`]).
        shape: MmaShape,
        /// D element type (currently `F32`).
        d_ty: PtxType,
        /// A element type. Expected to be `PtxType::F16`; bf16 emission
        /// uses the dedicated [`MmaSyncBf16`](Self::MmaSyncBf16) variant.
        /// The enforcement boundary is at module-load time:
        /// [`PtxModule::validate`](crate::ir::PtxModule::validate)
        /// rejects `PtxType::BF16` here with
        /// [`ValidationError::MmaSyncBf16Rejected`](crate::ir::ValidationError::MmaSyncBf16Rejected),
        /// matching the convention used by the SM-target capability checks
        /// (also validated at module-load, not at raw emit). Direct
        /// `Emit::emit()` of a `MmaSync` carrying a BF16 dtype tag will
        /// still produce the corresponding PTX string — callers using
        /// the IR below the [`PtxModule`](crate::ir::PtxModule) boundary
        /// must invoke validation themselves.
        a_ty: PtxType,
        /// B element type. Same expectation and enforcement-boundary
        /// note as `a_ty`.
        b_ty: PtxType,
        /// C element type (currently `F32`).
        c_ty: PtxType,
    },
    /// INT8 matrix-multiply-accumulate:
    /// `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`
    /// `{d_regs}, {a_regs}, {b_regs}, {c_regs};`
    ///
    /// Computes `D = A * B + C` across the warp where A and B are signed
    /// 8-bit integer matrices (packed i8x4 into `.b32` fragment registers)
    /// and C/D are `.s32` accumulator matrices. A is row-major, B is
    /// column-major (`.row.col`).
    ///
    /// Shape is implicitly [`MmaShape::M16N8K32`]; element types are
    /// implicitly `.s32.s8.s8.s32`. Requires SM 8.0+ (Ampere or newer).
    ///
    /// Used as the fast-path compute primitive for `kaio_ops::matmul_int8`
    /// (Sprint 7.1). A dequant-to-f16 fallback uses the regular
    /// [`MmaSync`](Self::MmaSync) variant with `.f16.f16.f32` types.
    ///
    /// Example emission:
    /// ```text
    /// mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
    ///     {%r8,%r9,%r10,%r11},
    ///     {%r0,%r1,%r2,%r3},
    ///     {%r4,%r5},
    ///     {%r12,%r13,%r14,%r15};
    /// ```
    MmaSyncInt8 {
        /// Destination (D) fragment — four `.s32` accumulator registers.
        d: FragmentC_M16N8K32,
        /// Input A fragment — four `.b32` registers, each packing 4 signed i8.
        a: FragmentA_M16N8K32,
        /// Input B fragment — two `.b32` registers, each packing 4 signed i8.
        b: FragmentB_M16N8K32,
        /// Input C fragment — four `.s32` accumulator registers.
        c: FragmentC_M16N8K32,
    },
    /// bf16 matrix-multiply-accumulate:
    /// `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
    /// `{d_regs}, {a_regs}, {b_regs}, {c_regs};`
    ///
    /// Sibling of [`MmaSync`](Self::MmaSync) specialized for bf16 inputs.
    /// Operand fragments carry the bf16-distinguishing sibling types
    /// ([`FragmentA_BF16`] / [`FragmentB_BF16`]) so cross-precision wiring
    /// at call sites is a compile error rather than a silent dtype tag
    /// mismatch on `MmaSync`. The accumulator is the same `.f32`
    /// [`FragmentC`] used by the f16 path.
    ///
    /// Shape is implicitly [`MmaShape::M16N8K16`]; element types are
    /// implicitly `.f32.bf16.bf16.f32`. Requires SM 8.0+ (Ampere or newer).
    ///
    /// Introduced in Sprint 9.1 per the D2.5 sibling-IR-variant decision —
    /// mirrors the `MmaSyncInt8` precedent. The legacy route of constructing
    /// [`MmaSync`](Self::MmaSync) with `a_ty: BF16, b_ty: BF16` is rejected
    /// at module-load time by
    /// [`PtxModule::validate`](crate::ir::PtxModule::validate) with
    /// [`ValidationError::MmaSyncBf16Rejected`](crate::ir::ValidationError::MmaSyncBf16Rejected);
    /// callers that drive `Emit::emit()` directly without going through
    /// `PtxModule::validate` can still produce the legacy bf16 string,
    /// matching the SM-check convention. Public host-launch paths run
    /// validation before driver dispatch, so shipped kernels are
    /// protected. New bf16 mmas should go through this variant.
    ///
    /// Example emission:
    /// ```text
    /// mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
    ///     {%f4,%f5,%f6,%f7},
    ///     {%r0,%r1,%r2,%r3},
    ///     {%r4,%r5},
    ///     {%f0,%f1,%f2,%f3};
    /// ```
    MmaSyncBf16 {
        /// Destination (D) fragment — `.f32` accumulator output.
        d: FragmentC,
        /// Input A fragment — `.b32` packed bfloat2 registers.
        a: FragmentA_BF16,
        /// Input B fragment — `.b32` packed bfloat2 registers.
        b: FragmentB_BF16,
        /// Input C fragment — `.f32` accumulator input.
        c: FragmentC,
    },
    /// Warp-collective fragment load from shared memory:
    /// `ldmatrix.sync.aligned.m8n8.{num}{.trans}.shared.b16`
    /// `{d_regs}, [addr];`
    ///
    /// Cooperatively loads 2 or 4 (per `.num`) 8×8 matrices of 16-bit
    /// elements from shared memory, distributing them across the warp in
    /// exactly the per-thread layout `mma.sync.m16n8k16` expects for its
    /// fragment operands: within each loaded matrix, lane `t` receives
    /// the `.b32` word holding (row `t/4`, columns `2(t%4)..2(t%4)+1`).
    ///
    /// **Address-supply rule (per-lane operand):** every lane passes its
    /// own `addr` — the shared-space byte address of one 16-byte matrix
    /// row. Lanes `8i..8i+7` supply the row addresses of matrix `i`
    /// (rows 0..7 in order), and matrix `i` lands in destination
    /// register `i`. For `.x4` all 32 lanes' addresses are consumed; for
    /// `.x2` only lanes 0–15 (remaining lanes' operands are ignored).
    /// Each row address must be **16-byte aligned** — a runtime rule
    /// that ptxas does not statically check, so shared tiles feeding
    /// `ldmatrix` should declare `align: 16`.
    ///
    /// `.b16` is the element size, not a dtype — f16 and bf16 tiles load
    /// identically (the dtype distinction lives at the mma operand
    /// level). Future `.b8` / `.m16n16` forms have different geometry
    /// and get sibling variants, per the house specialized-variant
    /// style.
    ///
    /// Requires SM 7.5+ (Turing) — the one sub-Ampere tensor-core op;
    /// see the module docs. Introduced in Sprint 9.3.
    ///
    /// Example emission:
    /// ```text
    /// ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r0,%r1,%r2,%r3}, [%r4];
    /// ```
    LdMatrix {
        /// Destination registers — the `.num` qualifier and register
        /// count are bound together by [`LdMatrixDst`]. Registers must
        /// be `.b32`-class (`PtxType::U32`, the packed-pair convention
        /// of [`alloc_packed_half2`](crate::ir::RegisterAllocator::alloc_packed_half2));
        /// enforced at module-load by
        /// [`PtxModule::validate`](crate::ir::PtxModule::validate) with
        /// [`ValidationError::LdMatrixBadRegType`](crate::ir::ValidationError::LdMatrixBadRegType),
        /// matching the `MmaSync` bf16-rejection convention (raw
        /// `Emit::emit()` below the module boundary is unchecked).
        dst: LdMatrixDst,
        /// Per-lane shared-space byte address of this lane's matrix row
        /// (`.u32`, same addressing convention as `MemoryOp::LdShared`).
        /// Type enforced at module-load alongside `dst`.
        addr: crate::ir::Register,
        /// `.trans` — transpose each 8×8 matrix during the load (the
        /// fragment-B / column-major form).
        trans: bool,
    },
}

impl TensorCoreOp {
    /// Minimum SM version required to execute this op.
    pub fn min_sm(&self) -> u32 {
        match self {
            Self::MmaSync { shape, .. } => shape.min_sm(),
            Self::MmaSyncInt8 { .. } => MmaShape::M16N8K32.min_sm(),
            Self::MmaSyncBf16 { .. } => MmaShape::M16N8K16.min_sm(),
            // 75 flat across all m8n8.b16 forms (x1/x2/x4, ±trans) —
            // confirmed against the PTX ISA docs plus ptxas probes at
            // the Sprint 9.3 audit gate (`ptxas` itself reports
            // "Feature 'ldmatrix' requires .target sm_75 or higher").
            // The first sub-80 entry in this file; deliberate, see the
            // module docs.
            Self::LdMatrix { .. } => 75,
        }
    }

    /// Short human-readable label used in validation errors
    /// (e.g. `"mma.sync.m16n8k16"`).
    pub fn feature_label(&self) -> String {
        match self {
            Self::MmaSync { shape, .. } => format!("mma.sync.{}", shape.ptx_token()),
            Self::MmaSyncInt8 { .. } => {
                format!("mma.sync.{}.s8.s8.s32", MmaShape::M16N8K32.ptx_token())
            }
            Self::MmaSyncBf16 { .. } => {
                format!("mma.sync.{}.bf16.bf16.f32", MmaShape::M16N8K16.ptx_token())
            }
            Self::LdMatrix { dst, trans, .. } => {
                format!(
                    "ldmatrix.m8n8.{}{}",
                    dst.num_token(),
                    if *trans { ".trans" } else { "" }
                )
            }
        }
    }
}

/// Format a fragment register list as `{%x0,%x1,...}` (no surrounding
/// whitespace).
fn format_reg_list(regs: &[crate::ir::Register]) -> String {
    let joined = regs
        .iter()
        .map(|r| format!("{r}"))
        .collect::<Vec<_>>()
        .join(",");
    format!("{{{joined}}}")
}

impl Emit for TensorCoreOp {
    fn emit(&self, w: &mut PtxWriter) -> fmt::Result {
        match self {
            TensorCoreOp::MmaSync {
                d,
                a,
                b,
                c,
                shape,
                d_ty,
                a_ty,
                b_ty,
                c_ty,
            } => {
                // mma.sync.aligned.{shape}.row.col.{dty}.{aty}.{bty}.{cty}
                // For fp16/bf16 inputs with fp32 accumulate, .row.col is
                // the only valid operand-layout modifier.
                let mnemonic = format!(
                    "mma.sync.aligned.{}.row.col{}{}{}{}",
                    shape.ptx_token(),
                    d_ty.ptx_suffix(),
                    a_ty.ptx_suffix(),
                    b_ty.ptx_suffix(),
                    c_ty.ptx_suffix(),
                );
                let d_list = format_reg_list(&d.regs);
                let a_list = format_reg_list(&a.regs);
                let b_list = format_reg_list(&b.regs);
                let c_list = format_reg_list(&c.regs);
                w.instruction(
                    &mnemonic,
                    &[&d_list as &dyn fmt::Display, &a_list, &b_list, &c_list],
                )
            }
            TensorCoreOp::MmaSyncInt8 { d, a, b, c } => {
                // Full instruction: mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
                // The `.row.col` layout qualifiers are mandatory per PTX ISA —
                // A is row-major, B is col-major. Type suffix order is
                // {d_ty}.{a_ty}.{b_ty}.{c_ty} — s32 accumulator with s8 inputs.
                let mnemonic = "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32";
                let d_list = format_reg_list(&d.regs);
                let a_list = format_reg_list(&a.regs);
                let b_list = format_reg_list(&b.regs);
                let c_list = format_reg_list(&c.regs);
                w.instruction(
                    mnemonic,
                    &[&d_list as &dyn fmt::Display, &a_list, &b_list, &c_list],
                )
            }
            TensorCoreOp::MmaSyncBf16 { d, a, b, c } => {
                // Full instruction: mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
                // The .row.col layout qualifiers are mandatory; type-suffix
                // order is {d_ty}.{a_ty}.{b_ty}.{c_ty} — f32 accumulator
                // with bf16 inputs. Operand register byte layout is
                // identical to the f16 path (.b32 packed pairs); only the
                // mma operand dtype tag differs.
                let mnemonic = "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32";
                let d_list = format_reg_list(&d.regs);
                let a_list = format_reg_list(&a.regs);
                let b_list = format_reg_list(&b.regs);
                let c_list = format_reg_list(&c.regs);
                w.instruction(
                    mnemonic,
                    &[&d_list as &dyn fmt::Display, &a_list, &b_list, &c_list],
                )
            }
            TensorCoreOp::LdMatrix { dst, addr, trans } => {
                // Qualifier order per PTX ISA: .num before .trans before
                // .shared; `.b16` element size is the only supported form
                // at m8n8 (it is bits, not a dtype — serves f16 and bf16
                // alike).
                let mnemonic = format!(
                    "ldmatrix.sync.aligned.m8n8.{}{}.shared.b16",
                    dst.num_token(),
                    if *trans { ".trans" } else { "" },
                );
                let d_list = format_reg_list(dst.regs());
                let addr_operand = format!("[{addr}]");
                w.instruction(&mnemonic, &[&d_list as &dyn fmt::Display, &addr_operand])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fragment::{alloc_a_f16, alloc_b_f16, alloc_c};
    use crate::ir::RegisterAllocator;

    #[test]
    fn mma_shape_token_and_min_sm() {
        assert_eq!(MmaShape::M16N8K16.ptx_token(), "m16n8k16");
        assert_eq!(MmaShape::M16N8K16.min_sm(), 80);
        assert_eq!(MmaShape::M16N8K32.ptx_token(), "m16n8k32");
        assert_eq!(MmaShape::M16N8K32.min_sm(), 80);
    }

    #[test]
    fn emit_mma_sync_m16n8k16_f16_f32() {
        let mut alloc = RegisterAllocator::new();
        let a = alloc_a_f16(&mut alloc);
        let b = alloc_b_f16(&mut alloc);
        let c = alloc_c(&mut alloc);
        let d = alloc_c(&mut alloc);

        let op = TensorCoreOp::MmaSync {
            d,
            a,
            b,
            c,
            shape: MmaShape::M16N8K16,
            d_ty: PtxType::F32,
            a_ty: PtxType::F16,
            b_ty: PtxType::F16,
            c_ty: PtxType::F32,
        };

        let mut w = PtxWriter::new();
        w.indent();
        op.emit(&mut w).unwrap();
        let out = w.finish();

        // Check the full line — operand order: D, A, B, C.
        let expected = concat!(
            "    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 ",
            "{%f4,%f5,%f6,%f7}, {%r0,%r1,%r2,%r3}, {%r4,%r5}, {%f0,%f1,%f2,%f3};\n",
        );
        assert_eq!(out, expected);
    }

    #[test]
    fn emit_mma_sync_bf16_m16n8k16() {
        use crate::fragment::{alloc_a_bf16, alloc_b_bf16};
        let mut alloc = RegisterAllocator::new();
        let a = alloc_a_bf16(&mut alloc);
        let b = alloc_b_bf16(&mut alloc);
        let c = alloc_c(&mut alloc);
        let d = alloc_c(&mut alloc);

        let op = TensorCoreOp::MmaSyncBf16 { d, a, b, c };

        let mut w = PtxWriter::new();
        w.indent();
        op.emit(&mut w).unwrap();
        let out = w.finish();

        // Register layout mirrors the f16 path — alloc_a_bf16 and
        // alloc_b_bf16 reuse alloc_packed_half2, so:
        //   A = 4 × .b32  → %r0..%r3
        //   B = 2 × .b32  → %r4..%r5
        //   C = 4 × .f32  → %f0..%f3
        //   D = 4 × .f32  → %f4..%f7
        // Operand order in mma: D, A, B, C.
        let expected = concat!(
            "    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 ",
            "{%f4,%f5,%f6,%f7}, {%r0,%r1,%r2,%r3}, {%r4,%r5}, {%f0,%f1,%f2,%f3};\n",
        );
        assert_eq!(out, expected);
    }

    #[test]
    fn min_sm_and_feature_label_bf16() {
        use crate::fragment::{alloc_a_bf16, alloc_b_bf16};
        let mut alloc = RegisterAllocator::new();
        let op = TensorCoreOp::MmaSyncBf16 {
            d: alloc_c(&mut alloc),
            a: alloc_a_bf16(&mut alloc),
            b: alloc_b_bf16(&mut alloc),
            c: alloc_c(&mut alloc),
        };
        assert_eq!(op.min_sm(), 80);
        assert_eq!(op.feature_label(), "mma.sync.m16n8k16.bf16.bf16.f32");
    }

    #[test]
    fn min_sm_and_feature_label() {
        let mut alloc = RegisterAllocator::new();
        let op = TensorCoreOp::MmaSync {
            d: alloc_c(&mut alloc),
            a: alloc_a_f16(&mut alloc),
            b: alloc_b_f16(&mut alloc),
            c: alloc_c(&mut alloc),
            shape: MmaShape::M16N8K16,
            d_ty: PtxType::F32,
            a_ty: PtxType::F16,
            b_ty: PtxType::F16,
            c_ty: PtxType::F32,
        };
        assert_eq!(op.min_sm(), 80);
        assert_eq!(op.feature_label(), "mma.sync.m16n8k16");
    }

    #[test]
    fn emit_mma_sync_int8_m16n8k32() {
        use crate::fragment::{alloc_a_M16N8K32, alloc_b_M16N8K32, alloc_c_M16N8K32};
        let mut alloc = RegisterAllocator::new();
        let a = alloc_a_M16N8K32(&mut alloc);
        let b = alloc_b_M16N8K32(&mut alloc);
        let c = alloc_c_M16N8K32(&mut alloc);
        let d = alloc_c_M16N8K32(&mut alloc);

        let op = TensorCoreOp::MmaSyncInt8 { d, a, b, c };

        let mut w = PtxWriter::new();
        w.indent();
        op.emit(&mut w).unwrap();
        let out = w.finish();

        // Register layout:
        //   A = 4 × alloc_packed_int8x4  → %r0..%r3
        //   B = 2 × alloc_packed_int8x4  → %r4..%r5
        //   C = 4 × alloc(S32)           → %r6..%r9
        //   D = 4 × alloc(S32)           → %r10..%r13
        // All live in the %r class (S8/S32/U32 share RegKind::R).
        // Operand order in mma: D, A, B, C.
        let expected = concat!(
            "    mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 ",
            "{%r10,%r11,%r12,%r13}, {%r0,%r1,%r2,%r3}, {%r4,%r5}, {%r6,%r7,%r8,%r9};\n",
        );
        assert_eq!(out, expected);
    }

    #[test]
    fn int8_min_sm_and_feature_label() {
        use crate::fragment::{alloc_a_M16N8K32, alloc_b_M16N8K32, alloc_c_M16N8K32};
        let mut alloc = RegisterAllocator::new();
        let op = TensorCoreOp::MmaSyncInt8 {
            d: alloc_c_M16N8K32(&mut alloc),
            a: alloc_a_M16N8K32(&mut alloc),
            b: alloc_b_M16N8K32(&mut alloc),
            c: alloc_c_M16N8K32(&mut alloc),
        };
        assert_eq!(op.min_sm(), 80);
        assert_eq!(op.feature_label(), "mma.sync.m16n8k32.s8.s8.s32");
    }

    #[test]
    fn emit_ldmatrix_x4_shared_b16() {
        let mut alloc = RegisterAllocator::new();
        let dst = LdMatrixDst::X4([
            alloc.alloc_packed_half2(),
            alloc.alloc_packed_half2(),
            alloc.alloc_packed_half2(),
            alloc.alloc_packed_half2(),
        ]);
        let addr = alloc.alloc(PtxType::U32);

        let op = TensorCoreOp::LdMatrix {
            dst,
            addr,
            trans: false,
        };

        let mut w = PtxWriter::new();
        w.indent();
        op.emit(&mut w).unwrap();
        let out = w.finish();

        // Locked emission string (Sprint 9.3 D1) — destination list in
        // matrix order, bracketed per-lane shared address.
        let expected = concat!(
            "    ldmatrix.sync.aligned.m8n8.x4.shared.b16 ",
            "{%r0,%r1,%r2,%r3}, [%r4];\n",
        );
        assert_eq!(out, expected);
    }

    #[test]
    fn emit_ldmatrix_x2_trans_shared_b16() {
        let mut alloc = RegisterAllocator::new();
        let dst = LdMatrixDst::X2([alloc.alloc_packed_half2(), alloc.alloc_packed_half2()]);
        let addr = alloc.alloc(PtxType::U32);

        let op = TensorCoreOp::LdMatrix {
            dst,
            addr,
            trans: true,
        };

        let mut w = PtxWriter::new();
        w.indent();
        op.emit(&mut w).unwrap();
        let out = w.finish();

        // Locked emission string (Sprint 9.3 D1) — .trans sits between
        // .num and .shared per the ISA qualifier order.
        let expected = concat!(
            "    ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 ",
            "{%r0,%r1}, [%r2];\n",
        );
        assert_eq!(out, expected);
    }

    #[test]
    fn ldmatrix_min_sm_and_feature_label() {
        let mut alloc = RegisterAllocator::new();
        let x4 = TensorCoreOp::LdMatrix {
            dst: LdMatrixDst::X4([
                alloc.alloc_packed_half2(),
                alloc.alloc_packed_half2(),
                alloc.alloc_packed_half2(),
                alloc.alloc_packed_half2(),
            ]),
            addr: alloc.alloc(PtxType::U32),
            trans: false,
        };
        // Deliberate pin on 75 — the first sub-80 tensor-core op
        // (Sprint 9.3 audit: ptxas reports "requires .target sm_75 or
        // higher"). A later change "normalizing" this to 80 should have
        // to confront this assertion.
        assert_eq!(x4.min_sm(), 75);
        assert_eq!(x4.feature_label(), "ldmatrix.m8n8.x4");

        let x2t = TensorCoreOp::LdMatrix {
            dst: LdMatrixDst::X2([alloc.alloc_packed_half2(), alloc.alloc_packed_half2()]),
            addr: alloc.alloc(PtxType::U32),
            trans: true,
        };
        assert_eq!(x2t.min_sm(), 75);
        assert_eq!(x2t.feature_label(), "ldmatrix.m8n8.x2.trans");
    }

    #[test]
    fn ldmatrix_dst_accessors() {
        let mut alloc = RegisterAllocator::new();
        let r0 = alloc.alloc_packed_half2();
        let r1 = alloc.alloc_packed_half2();
        let x2 = LdMatrixDst::X2([r0, r1]);
        assert_eq!(x2.num_token(), "x2");
        assert_eq!(x2.regs(), &[r0, r1]);

        let r2 = alloc.alloc_packed_half2();
        let r3 = alloc.alloc_packed_half2();
        let x4 = LdMatrixDst::X4([r0, r1, r2, r3]);
        assert_eq!(x4.num_token(), "x4");
        assert_eq!(x4.regs().len(), 4);
    }

    #[test]
    fn tensor_core_via_ptx_instruction() {
        use crate::ir::PtxInstruction;
        let mut alloc = RegisterAllocator::new();
        let instr = PtxInstruction::TensorCore(TensorCoreOp::MmaSync {
            d: alloc_c(&mut alloc),
            a: alloc_a_f16(&mut alloc),
            b: alloc_b_f16(&mut alloc),
            c: alloc_c(&mut alloc),
            shape: MmaShape::M16N8K16,
            d_ty: PtxType::F32,
            a_ty: PtxType::F16,
            b_ty: PtxType::F16,
            c_ty: PtxType::F32,
        });
        let mut w = PtxWriter::new();
        w.indent();
        instr.emit(&mut w).unwrap();
        assert!(w.finish().contains("mma.sync.aligned.m16n8k16.row.col"));
    }
}
