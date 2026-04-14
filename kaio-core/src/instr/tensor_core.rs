//! Tensor-core PTX operations.
//!
//! This module hosts the warp-collective tensor-core instructions. Phase 6
//! supports a single shape — `m16n8k16` with fp16 inputs and fp32
//! accumulation — which is the Ampere+ (SM 8.0) shape used by CUTLASS,
//! cuBLAS, and every production fp16 matmul since 2020.
//!
//! Earlier shapes (Volta `m8n8k4`, Turing `m16n8k8`) have different
//! fragment layouts and are out of scope for Phase 6.

use std::fmt;

use crate::emit::{Emit, PtxWriter};
use crate::fragment::{
    FragmentA, FragmentA_M16N8K32, FragmentB, FragmentB_M16N8K32, FragmentC, FragmentC_M16N8K32,
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

/// Tensor-core PTX instruction variants.
///
/// Warp-collective operations — `mma.sync` is executed cooperatively by
/// all 32 threads in a warp with a rigid NVIDIA-defined register layout.
/// See [`crate::fragment`] for the per-thread register distribution.
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
        a: FragmentA,
        /// Input B fragment — `.b32` packed half2 registers.
        b: FragmentB,
        /// Input C fragment — `.f32` accumulator input.
        c: FragmentC,
        /// Matrix shape (currently only [`MmaShape::M16N8K16`]).
        shape: MmaShape,
        /// D element type (currently `F32`).
        d_ty: PtxType,
        /// A element type (currently `F16` or `BF16`).
        a_ty: PtxType,
        /// B element type (currently `F16` or `BF16`).
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
}

impl TensorCoreOp {
    /// Minimum SM version required to execute this op.
    pub fn min_sm(&self) -> u32 {
        match self {
            Self::MmaSync { shape, .. } => shape.min_sm(),
            Self::MmaSyncInt8 { .. } => MmaShape::M16N8K32.min_sm(),
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fragment::{alloc_a, alloc_b, alloc_c};
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
        let a = alloc_a(&mut alloc);
        let b = alloc_b(&mut alloc);
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
    fn emit_mma_sync_m16n8k16_bf16_f32() {
        let mut alloc = RegisterAllocator::new();
        let a = alloc_a(&mut alloc);
        let b = alloc_b(&mut alloc);
        let c = alloc_c(&mut alloc);
        let d = alloc_c(&mut alloc);

        let op = TensorCoreOp::MmaSync {
            d,
            a,
            b,
            c,
            shape: MmaShape::M16N8K16,
            d_ty: PtxType::F32,
            a_ty: PtxType::BF16,
            b_ty: PtxType::BF16,
            c_ty: PtxType::F32,
        };

        let mut w = PtxWriter::new();
        w.indent();
        op.emit(&mut w).unwrap();
        assert!(
            w.finish()
                .contains("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32")
        );
    }

    #[test]
    fn min_sm_and_feature_label() {
        let mut alloc = RegisterAllocator::new();
        let op = TensorCoreOp::MmaSync {
            d: alloc_c(&mut alloc),
            a: alloc_a(&mut alloc),
            b: alloc_b(&mut alloc),
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
    fn tensor_core_via_ptx_instruction() {
        use crate::ir::PtxInstruction;
        let mut alloc = RegisterAllocator::new();
        let instr = PtxInstruction::TensorCore(TensorCoreOp::MmaSync {
            d: alloc_c(&mut alloc),
            a: alloc_a(&mut alloc),
            b: alloc_b(&mut alloc),
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
