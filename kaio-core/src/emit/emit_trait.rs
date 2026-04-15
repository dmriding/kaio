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
use crate::types::PtxType;

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
    let mut max_idx: [Option<u32>; 7] = [None; 7];
    let mut decl_types: [&str; 7] = [""; 7];

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

    // Emit in counter_index order: R(0), Rd(1), F(2), Fd(3), P(4), H(5), Hb(6)
    let prefixes = ["%r", "%rd", "%f", "%fd", "%p", "%h", "%hb"];
    for i in 0..7 {
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
            Self::TensorCore(op) => op.emit(w),
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
                // PTX requires rounding modifiers for conversions involving floats.
                // KAIO emits .rn for all float-to-float cvt operations for
                // consistency and PTX validity, even where the conversion is
                // exact (e.g., f16→f32).
                let rounding = match (dst_ty, src_ty) {
                    // int → float (including half): round to nearest even
                    (
                        PtxType::F16 | PtxType::BF16 | PtxType::F32 | PtxType::F64,
                        PtxType::S32 | PtxType::U32 | PtxType::S64 | PtxType::U64,
                    ) => ".rn",
                    // float (including half) → int: round toward zero (matches Rust `as`)
                    (
                        PtxType::S32 | PtxType::U32 | PtxType::S64 | PtxType::U64,
                        PtxType::F16 | PtxType::BF16 | PtxType::F32 | PtxType::F64,
                    ) => ".rzi",
                    // float → float (any width, including half): round to nearest
                    (
                        PtxType::F16 | PtxType::BF16 | PtxType::F32 | PtxType::F64,
                        PtxType::F16 | PtxType::BF16 | PtxType::F32 | PtxType::F64,
                    ) => ".rn",
                    // int → int or same type: no rounding modifier
                    _ => "",
                };
                let mnemonic = format!(
                    "cvt{rounding}{}{}",
                    dst_ty.ptx_suffix(),
                    src_ty.ptx_suffix()
                );
                w.instruction(&mnemonic, &[dst as &dyn fmt::Display, src])
            }
            Self::MovPack { dst, srcs, ty } => {
                // mov.b{N} %dst, {%s0,%s1,...};
                //
                // The vector-pack form of `mov` requires the typeless `.b{N}`
                // suffix (PTX ISA 9.7.9.10) — `mov.u32 %r, {%h0, %h1};` is
                // rejected by ptxas. We derive `.b{N}` from the destination
                // type's byte width.
                let joined = srcs
                    .iter()
                    .map(|r| format!("{r}"))
                    .collect::<Vec<_>>()
                    .join(",");
                let src_list = format!("{{{joined}}}");
                let bits = ty.size_bytes() * 8;
                let mnemonic = format!("mov.b{bits}");
                w.instruction(&mnemonic, &[dst as &dyn fmt::Display, &src_list])
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
    fn emit_mov_shared_addr() {
        let mut w = PtxWriter::new();
        w.indent();
        let instr = PtxInstruction::Mov {
            dst: reg(RegKind::R, 0, PtxType::U32),
            src: Operand::SharedAddr("sdata".to_string()),
            ty: PtxType::U32,
        };
        instr.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    mov.u32 %r0, sdata;\n");
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
        assert_eq!(w.finish(), "    cvt.rn.f32.s32 %f0, %r0;\n");
    }

    #[test]
    fn emit_cvt_float_to_int() {
        let mut w = PtxWriter::new();
        w.indent();
        let instr = PtxInstruction::Cvt {
            dst: reg(RegKind::R, 0, PtxType::U32),
            src: reg(RegKind::F, 0, PtxType::F32),
            dst_ty: PtxType::U32,
            src_ty: PtxType::F32,
        };
        instr.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    cvt.rzi.u32.f32 %r0, %f0;\n");
    }

    #[test]
    fn emit_cvt_int_to_int() {
        let mut w = PtxWriter::new();
        w.indent();
        let instr = PtxInstruction::Cvt {
            dst: reg(RegKind::R, 0, PtxType::S32),
            src: reg(RegKind::R, 1, PtxType::U32),
            dst_ty: PtxType::S32,
            src_ty: PtxType::U32,
        };
        instr.emit(&mut w).unwrap();
        // No rounding modifier for int → int
        assert_eq!(w.finish(), "    cvt.s32.u32 %r0, %r1;\n");
    }

    #[test]
    fn emit_cvt_f32_to_f16() {
        let mut w = PtxWriter::new();
        w.indent();
        let instr = PtxInstruction::Cvt {
            dst: reg(RegKind::H, 0, PtxType::F16),
            src: reg(RegKind::F, 0, PtxType::F32),
            dst_ty: PtxType::F16,
            src_ty: PtxType::F32,
        };
        instr.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    cvt.rn.f16.f32 %h0, %f0;\n");
    }

    #[test]
    fn emit_cvt_f16_to_f32() {
        let mut w = PtxWriter::new();
        w.indent();
        let instr = PtxInstruction::Cvt {
            dst: reg(RegKind::F, 0, PtxType::F32),
            src: reg(RegKind::H, 0, PtxType::F16),
            dst_ty: PtxType::F32,
            src_ty: PtxType::F16,
        };
        instr.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    cvt.rn.f32.f16 %f0, %h0;\n");
    }

    #[test]
    fn emit_cvt_int_to_f16() {
        let mut w = PtxWriter::new();
        w.indent();
        let instr = PtxInstruction::Cvt {
            dst: reg(RegKind::H, 0, PtxType::F16),
            src: reg(RegKind::R, 0, PtxType::S32),
            dst_ty: PtxType::F16,
            src_ty: PtxType::S32,
        };
        instr.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    cvt.rn.f16.s32 %h0, %r0;\n");
    }

    #[test]
    fn emit_cvt_f16_to_int() {
        let mut w = PtxWriter::new();
        w.indent();
        let instr = PtxInstruction::Cvt {
            dst: reg(RegKind::R, 0, PtxType::U32),
            src: reg(RegKind::H, 0, PtxType::F16),
            dst_ty: PtxType::U32,
            src_ty: PtxType::F16,
        };
        instr.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    cvt.rzi.u32.f16 %r0, %h0;\n");
    }

    #[test]
    fn emit_cvt_bf16_to_f32() {
        let mut w = PtxWriter::new();
        w.indent();
        let instr = PtxInstruction::Cvt {
            dst: reg(RegKind::F, 0, PtxType::F32),
            src: reg(RegKind::Hb, 0, PtxType::BF16),
            dst_ty: PtxType::F32,
            src_ty: PtxType::BF16,
        };
        instr.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    cvt.rn.f32.bf16 %f0, %hb0;\n");
    }

    #[test]
    fn emit_reg_declarations_with_f16() {
        let regs = vec![
            reg(RegKind::F, 0, PtxType::F32),
            reg(RegKind::H, 0, PtxType::F16),
            reg(RegKind::H, 1, PtxType::F16),
            reg(RegKind::Hb, 0, PtxType::BF16),
        ];
        let mut w = PtxWriter::new();
        w.indent();
        emit_reg_declarations(&regs, &mut w).unwrap();
        let output = w.finish();
        assert!(output.contains(".reg .f32 %f<1>;"));
        assert!(output.contains(".reg .f16 %h<2>;"));
        assert!(output.contains(".reg .bf16 %hb<1>;"));
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
    fn emit_mov_pack_two_f16_into_b32() {
        let mut w = PtxWriter::new();
        w.indent();
        let instr = PtxInstruction::MovPack {
            dst: reg(RegKind::R, 7, PtxType::U32),
            srcs: vec![
                reg(RegKind::H, 3, PtxType::F16),
                reg(RegKind::H, 4, PtxType::F16),
            ],
            ty: PtxType::U32,
        };
        instr.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    mov.b32 %r7, {%h3,%h4};\n");
    }

    /// End-to-end emitter test: a mini f16 kernel proving half types flow
    /// through params, register declarations, loads, cvt, arithmetic, and stores.
    ///
    /// Kernel: load f16 → cvt to f32 → add 1.0 → cvt to f16 → store f16
    #[test]
    fn emit_kernel_f16_flow() {
        use crate::instr::{ArithOp, MemoryOp};

        let mut alloc = RegisterAllocator::new();
        // Registers: rd for pointers, h for f16, f for f32, r for tid
        let rd_in = alloc.alloc(PtxType::U64); // %rd0: input ptr
        let rd_out = alloc.alloc(PtxType::U64); // %rd1: output ptr
        let r_tid = alloc.alloc(PtxType::U32); // %r0: thread id
        let rd_off = alloc.alloc(PtxType::U64); // %rd2: byte offset
        let rd_addr_in = alloc.alloc(PtxType::U64); // %rd3: input addr
        let rd_addr_out = alloc.alloc(PtxType::U64); // %rd4: output addr
        let h_val = alloc.alloc(PtxType::F16); // %h0: loaded f16
        let f_val = alloc.alloc(PtxType::F32); // %f0: f32 value
        let f_one = alloc.alloc(PtxType::F32); // %f1: constant 1.0
        let f_sum = alloc.alloc(PtxType::F32); // %f2: result
        let h_out = alloc.alloc(PtxType::F16); // %h1: output f16

        let mut kernel = PtxKernel::new("f16_add_one");
        kernel.add_param(PtxParam::pointer("in_ptr", PtxType::F16));
        kernel.add_param(PtxParam::pointer("out_ptr", PtxType::F16));

        // Load params
        kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
            dst: rd_in,
            param_name: "in_ptr".to_string(),
            ty: PtxType::U64,
        }));
        kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
            dst: rd_out,
            param_name: "out_ptr".to_string(),
            ty: PtxType::U64,
        }));
        // Get tid
        kernel.push(PtxInstruction::Mov {
            dst: r_tid,
            src: Operand::SpecialReg(SpecialReg::TidX),
            ty: PtxType::U32,
        });
        // Compute byte offset (tid * 2 for f16)
        kernel.push(PtxInstruction::Cvt {
            dst: rd_off,
            src: r_tid,
            dst_ty: PtxType::U64,
            src_ty: PtxType::U32,
        });
        // addr_in = in_ptr + offset
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: rd_addr_in,
            lhs: Operand::Reg(rd_in),
            rhs: Operand::Reg(rd_off),
            ty: PtxType::U64,
        }));
        // Load f16 value
        kernel.push(PtxInstruction::Memory(MemoryOp::LdGlobal {
            dst: h_val,
            addr: rd_addr_in,
            ty: PtxType::F16,
        }));
        // Convert to f32
        kernel.push(PtxInstruction::Cvt {
            dst: f_val,
            src: h_val,
            dst_ty: PtxType::F32,
            src_ty: PtxType::F16,
        });
        // Add 1.0
        kernel.push(PtxInstruction::Mov {
            dst: f_one,
            src: Operand::ImmF32(1.0),
            ty: PtxType::F32,
        });
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: f_sum,
            lhs: Operand::Reg(f_val),
            rhs: Operand::Reg(f_one),
            ty: PtxType::F32,
        }));
        // Convert back to f16
        kernel.push(PtxInstruction::Cvt {
            dst: h_out,
            src: f_sum,
            dst_ty: PtxType::F16,
            src_ty: PtxType::F32,
        });
        // addr_out = out_ptr + offset
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: rd_addr_out,
            lhs: Operand::Reg(rd_out),
            rhs: Operand::Reg(rd_off),
            ty: PtxType::U64,
        }));
        // Store f16 result
        kernel.push(PtxInstruction::Memory(MemoryOp::StGlobal {
            addr: rd_addr_out,
            src: h_out,
            ty: PtxType::F16,
        }));
        kernel.push(PtxInstruction::Control(crate::instr::ControlOp::Ret));
        kernel.set_registers(alloc.into_allocated());

        let mut w = PtxWriter::new();
        kernel.emit(&mut w).unwrap();
        let output = w.finish();

        // Verify structure: params, register declarations, instructions
        assert!(output.contains(".param .u64 in_ptr"));
        assert!(output.contains(".param .u64 out_ptr"));
        assert!(output.contains(".reg .f16 %h<2>;"), "f16 reg declarations");
        assert!(output.contains(".reg .f32 %f<3>;"), "f32 reg declarations");
        // f16/bf16 loads and stores emit `.b16` — the valid ld/st type
        // modifier for 16-bit memory ops (PTX ISA §8.7.9). Register class
        // stays `.f16`. The `cvt` instruction still uses `.f16` / `.f32`
        // because it's a register-to-register conversion, not memory.
        assert!(output.contains("ld.global.b16 %h0"));
        assert!(output.contains("cvt.rn.f32.f16 %f0, %h0"));
        assert!(output.contains("cvt.rn.f16.f32 %h1, %f2"));
        assert!(output.contains("st.global.b16 [%rd4], %h1"));
    }

    #[test]
    fn emit_module_header() {
        let module = PtxModule::new("sm_70");
        let mut w = PtxWriter::new();
        // Emit just the header (module with no kernels)
        module.emit(&mut w).unwrap();
        assert_eq!(
            w.finish(),
            ".version 8.7\n.target sm_70\n.address_size 64\n"
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
