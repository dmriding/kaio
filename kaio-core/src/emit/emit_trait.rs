//! The `Emit` trait and implementations for all IR nodes.
//!
//! Individual instruction-category Emit impls are co-located with their types:
//! - `ArithOp` → `instr/arith.rs`
//! - `MemoryOp` → `instr/memory.rs`
//! - `ControlOp` → `instr/control.rs`
//!
//! This file contains the orchestration-level impls for `PtxModule`,
//! `PtxKernel`, and `PtxInstruction` (including Mov, Cvt, Label, Comment).

use std::fmt;

use super::writer::PtxWriter;
use crate::ir::{PtxInstruction, PtxKernel, PtxModule, Register};

/// Trait for emitting PTX text from an IR node.
///
/// Every IR type implements this. The writer handles indentation and
/// formatting; each `Emit` impl is responsible for producing the
/// content of its node type.
pub trait Emit {
    /// Write this node's PTX representation to the writer.
    fn emit(&self, w: &mut PtxWriter) -> fmt::Result;
}

// --- Module-level emission ---

impl Emit for PtxModule {
    fn emit(&self, w: &mut PtxWriter) -> fmt::Result {
        w.raw_line(&format!(".version {}", self.version))?;
        w.raw_line(&format!(".target {}", self.target))?;
        w.raw_line(&format!(".address_size {}", self.address_size))?;
        for kernel in &self.kernels {
            w.blank()?;
            kernel.emit(w)?;
        }
        Ok(())
    }
}

// --- Kernel-level emission ---

impl Emit for PtxKernel {
    fn emit(&self, w: &mut PtxWriter) -> fmt::Result {
        // 1. Kernel signature with parameters
        if self.params.is_empty() {
            w.raw_line(&format!(".visible .entry {}()", self.name))?;
        } else {
            w.raw_line(&format!(".visible .entry {}(", self.name))?;
            w.indent();
            for (i, param) in self.params.iter().enumerate() {
                let comma = if i < self.params.len() - 1 { "," } else { "" };
                w.line(&format!("{}{}", param.ptx_decl(), comma))?;
            }
            w.dedent();
            w.raw_line(")")?;
        }

        // 2. Opening brace
        w.raw_line("{")?;
        w.indent();

        // 3. Register declarations
        emit_reg_declarations(&self.registers, w)?;

        // 4. Shared memory declarations
        for decl in &self.shared_decls {
            w.line(&format!(
                ".shared .align {} .b8 {}[{}];",
                decl.align, decl.name, decl.size_bytes
            ))?;
        }

        // 5. Blank line between declarations and body
        w.blank()?;

        // 6. Instruction body
        for instr in &self.body {
            instr.emit(w)?;
        }

        // 7. Closing brace
        w.dedent();
        w.raw_line("}")?;
        Ok(())
    }
}

/// Emit `.reg` declarations grouped by register kind.
///
/// Uses the `<N>` syntax: `.reg .b32 %r<5>;` declares `%r0` through `%r4`.
/// Groups by [`RegKind`](crate::types::RegKind) using fixed-size arrays
/// indexed by `counter_index()` — no heap allocation, deterministic order.
fn emit_reg_declarations(registers: &[Register], w: &mut PtxWriter) -> fmt::Result {
    // Find max index per RegKind
    let mut max_idx: [Option<u32>; 5] = [None; 5];
    let mut decl_types: [&str; 5] = [""; 5];

    for reg in registers {
        let ci = reg.kind.counter_index();
        match max_idx[ci] {
            None => {
                max_idx[ci] = Some(reg.index);
                decl_types[ci] = reg.ptx_type.reg_decl_type();
            }
            Some(prev) if reg.index > prev => {
                max_idx[ci] = Some(reg.index);
            }
            _ => {}
        }
    }

    // Emit in counter_index order: R(0), Rd(1), F(2), Fd(3), P(4)
    let prefixes = ["%r", "%rd", "%f", "%fd", "%p"];
    for i in 0..5 {
        if let Some(max) = max_idx[i] {
            let count = max + 1;
            w.line(&format!(
                ".reg {} {}<{}>;",
                decl_types[i], prefixes[i], count
            ))?;
        }
    }
    Ok(())
}

// --- Instruction-level emission ---

impl Emit for PtxInstruction {
    fn emit(&self, w: &mut PtxWriter) -> fmt::Result {
        match self {
            Self::Arith(op) => op.emit(w),
            Self::Memory(op) => op.emit(w),
            Self::Control(op) => op.emit(w),
            Self::Mov { dst, src, ty } => {
                let mnemonic = format!("mov{}", ty.ptx_suffix());
                w.instruction(&mnemonic, &[dst as &dyn fmt::Display, src])
            }
            Self::Cvt {
                dst,
                src,
                dst_ty,
                src_ty,
            } => {
                let mnemonic = format!("cvt{}{}", dst_ty.ptx_suffix(), src_ty.ptx_suffix());
                w.instruction(&mnemonic, &[dst as &dyn fmt::Display, src])
            }
            Self::Label(name) => {
                // Labels are at column 0 — dedent, emit, re-indent.
                // dedent saturates at 0 (safe for edge cases).
                w.dedent();
                w.raw_line(&format!("{name}:"))?;
                w.indent();
                Ok(())
            }
            Self::Comment(text) => w.line(&format!("// {text}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Operand, PtxParam, RegisterAllocator, SpecialReg};
    use crate::types::{PtxType, RegKind};

    fn reg(kind: RegKind, index: u32, ptx_type: PtxType) -> Register {
        Register {
            kind,
            index,
            ptx_type,
        }
    }

    #[test]
    fn emit_mov_special_reg() {
        let mut w = PtxWriter::new();
        w.indent();
        let instr = PtxInstruction::Mov {
            dst: reg(RegKind::R, 0, PtxType::U32),
            src: Operand::SpecialReg(SpecialReg::TidX),
            ty: PtxType::U32,
        };
        instr.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    mov.u32 %r0, %tid.x;\n");
    }

    #[test]
    fn emit_mov_reg_to_reg() {
        let mut w = PtxWriter::new();
        w.indent();
        let instr = PtxInstruction::Mov {
            dst: reg(RegKind::F, 1, PtxType::F32),
            src: Operand::Reg(reg(RegKind::F, 0, PtxType::F32)),
            ty: PtxType::F32,
        };
        instr.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    mov.f32 %f1, %f0;\n");
    }

    #[test]
    fn emit_cvt() {
        let mut w = PtxWriter::new();
        w.indent();
        let instr = PtxInstruction::Cvt {
            dst: reg(RegKind::F, 0, PtxType::F32),
            src: reg(RegKind::R, 0, PtxType::S32),
            dst_ty: PtxType::F32,
            src_ty: PtxType::S32,
        };
        instr.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    cvt.f32.s32 %f0, %r0;\n");
    }

    #[test]
    fn emit_label_at_column_zero() {
        let mut w = PtxWriter::new();
        w.indent(); // simulate being inside a kernel body
        let instr = PtxInstruction::Label("EXIT".to_string());
        instr.emit(&mut w).unwrap();
        // Label should be at column 0, no indentation
        assert_eq!(w.finish(), "EXIT:\n");
    }

    #[test]
    fn emit_comment() {
        let mut w = PtxWriter::new();
        w.indent();
        let instr = PtxInstruction::Comment("bounds check".to_string());
        instr.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    // bounds check\n");
    }

    #[test]
    fn emit_module_header() {
        let module = PtxModule::new("sm_89");
        let mut w = PtxWriter::new();
        // Emit just the header (module with no kernels)
        module.emit(&mut w).unwrap();
        assert_eq!(
            w.finish(),
            ".version 8.7\n.target sm_89\n.address_size 64\n"
        );
    }

    #[test]
    fn emit_kernel_minimal() {
        let mut alloc = RegisterAllocator::new();
        let r0 = alloc.alloc(PtxType::U32);

        let mut kernel = PtxKernel::new("test_kernel");
        kernel.add_param(PtxParam::scalar("n", PtxType::U32));
        kernel.push(PtxInstruction::Mov {
            dst: r0,
            src: Operand::ImmU32(42),
            ty: PtxType::U32,
        });
        kernel.push(PtxInstruction::Control(crate::instr::ControlOp::Ret));
        kernel.set_registers(alloc.into_allocated());

        let mut w = PtxWriter::new();
        kernel.emit(&mut w).unwrap();
        let output = w.finish();

        // Validate structure
        assert!(output.contains(".visible .entry test_kernel("));
        assert!(output.contains(".param .u32 n"));
        assert!(output.contains(".reg .b32 %r<1>;"));
        assert!(output.contains("mov.u32 %r0, 42;"));
        assert!(output.contains("ret;"));
        assert!(output.starts_with(".visible .entry"));
        assert!(output.trim_end().ends_with('}'));
    }
}
