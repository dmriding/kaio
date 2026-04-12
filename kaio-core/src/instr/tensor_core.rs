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
use crate::fragment::{FragmentA, FragmentB, FragmentC};
use crate::types::PtxType;

/// The shape of an `mma.sync` instruction.
///
/// Exactly one variant today. Adding a new shape later means adding a
/// new enum variant **and** a new set of fragment types — the fragment
/// register count is shape-dependent. See [`crate::fragment`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmaShape {
    /// 16×16 × 16×8 → 16×8, fp16 inputs with fp32 accumulate (Ampere+).
    M16N8K16,
}

impl MmaShape {
    /// PTX shape token (e.g. `"m16n8k16"`).
    pub fn ptx_token(&self) -> &'static str {
        match self {
            Self::M16N8K16 => "m16n8k16",
        }
    }

    /// Minimum SM version required to execute this shape.
    ///
    /// Used by [`crate::ir::PtxModule::validate`] to reject kernels that
    /// emit tensor-core ops against too-low a target SM.
    pub fn min_sm(&self) -> u32 {
        match self {
            Self::M16N8K16 => 80,
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
}

impl TensorCoreOp {
    /// Minimum SM version required to execute this op.
    pub fn min_sm(&self) -> u32 {
        match self {
            Self::MmaSync { shape, .. } => shape.min_sm(),
        }
    }

    /// Short human-readable label used in validation errors
    /// (e.g. `"mma.sync.m16n8k16"`).
    pub fn feature_label(&self) -> String {
        match self {
            Self::MmaSync { shape, .. } => format!("mma.sync.{}", shape.ptx_token()),
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
