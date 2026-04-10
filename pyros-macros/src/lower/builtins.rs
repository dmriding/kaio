//! Built-in function registry and lowering.
//!
//! Maps kernel built-in function calls to PTX instruction generation.
//! Thread/block index functions map to `special::` helpers. Math functions
//! map to single PTX instructions or multi-instruction synthesis.

use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;

use crate::kernel_ir::KernelType;

use super::LoweringContext;

/// Lower a built-in function call.
///
/// Returns `(result_register, result_type, token_stream)`.
#[allow(dead_code)] // Called from lower/mod.rs::lower_expr
pub fn lower_builtin(
    ctx: &mut LoweringContext,
    name: &str,
    arg_regs: &[Ident],
    arg_types: &[KernelType],
    span: Span,
) -> syn::Result<(Ident, KernelType, TokenStream)> {
    match name {
        // --- Thread/block index builtins (no args, return U32) ---
        "thread_idx_x" => lower_special_reg(ctx, "tid_x", arg_regs, span),
        "thread_idx_y" => lower_special_reg(ctx, "tid_y", arg_regs, span),
        "thread_idx_z" => lower_special_reg(ctx, "tid_z", arg_regs, span),
        "block_idx_x" => lower_special_reg(ctx, "ctaid_x", arg_regs, span),
        "block_idx_y" => lower_special_reg(ctx, "ctaid_y", arg_regs, span),
        "block_idx_z" => lower_special_reg(ctx, "ctaid_z", arg_regs, span),
        "block_dim_x" => lower_special_reg(ctx, "ntid_x", arg_regs, span),
        "block_dim_y" => lower_special_reg(ctx, "ntid_y", arg_regs, span),
        "block_dim_z" => lower_special_reg(ctx, "ntid_z", arg_regs, span),
        "grid_dim_x" => lower_special_reg(ctx, "nctaid_x", arg_regs, span),
        "grid_dim_y" => lower_special_reg(ctx, "nctaid_y", arg_regs, span),
        "grid_dim_z" => lower_special_reg(ctx, "nctaid_z", arg_regs, span),

        // --- Direct math builtins (single PTX instruction) ---
        "sqrt" => lower_unary_math(ctx, "Sqrt", arg_regs, arg_types, span),
        "abs" => lower_abs(ctx, arg_regs, arg_types, span),
        "min" => lower_binary_math(ctx, "Min", arg_regs, arg_types, span),
        "max" => lower_binary_math(ctx, "Max", arg_regs, arg_types, span),

        // --- Synthesized math builtins (multi-instruction) ---
        "exp" => lower_exp(ctx, arg_regs, arg_types, span),
        "log" => lower_log(ctx, arg_regs, arg_types, span),
        "tanh" => lower_tanh(ctx, arg_regs, arg_types, span),

        _ => Err(syn::Error::new(
            span,
            format!(
                "unknown function `{name}` in GPU kernel. Available built-in functions: \
                 thread_idx_x, thread_idx_y, thread_idx_z, \
                 block_idx_x, block_idx_y, block_idx_z, \
                 block_dim_x, block_dim_y, block_dim_z, \
                 grid_dim_x, grid_dim_y, grid_dim_z, \
                 sqrt, abs, min, max, exp, log, tanh"
            ),
        )),
    }
}

/// Lower a thread/block special register read (no args, returns U32).
fn lower_special_reg(
    ctx: &mut LoweringContext,
    helper_name: &str,
    arg_regs: &[Ident],
    span: Span,
) -> syn::Result<(Ident, KernelType, TokenStream)> {
    if !arg_regs.is_empty() {
        return Err(syn::Error::new(
            span,
            format!(
                "thread/block index functions take no arguments, got {}",
                arg_regs.len()
            ),
        ));
    }

    let reg = ctx.fresh_reg();
    let instr_reg = ctx.fresh_reg();
    let helper = Ident::new(helper_name, Span::call_site());

    let tokens = quote! {
        let (#reg, #instr_reg) = special::#helper(&mut alloc);
        kernel.push(#instr_reg);
    };

    Ok((reg, KernelType::U32, tokens))
}

/// Lower a unary f32 math function (sqrt → Sqrt).
fn lower_unary_math(
    ctx: &mut LoweringContext,
    variant_name: &str,
    arg_regs: &[Ident],
    arg_types: &[KernelType],
    span: Span,
) -> syn::Result<(Ident, KernelType, TokenStream)> {
    if arg_regs.len() != 1 {
        return Err(syn::Error::new(
            span,
            format!("math function expects 1 argument, got {}", arg_regs.len()),
        ));
    }
    check_f32(&arg_types[0], span)?;

    let src = &arg_regs[0];
    let dst = ctx.fresh_reg();
    let variant = Ident::new(variant_name, Span::call_site());

    let tokens = quote! {
        let #dst = alloc.alloc(PtxType::F32);
        kernel.push(PtxInstruction::Arith(ArithOp::#variant {
            dst: #dst,
            src: Operand::Reg(#src),
        }));
    };

    Ok((dst, KernelType::F32, tokens))
}

/// Lower abs — works on any numeric type, not just f32.
fn lower_abs(
    ctx: &mut LoweringContext,
    arg_regs: &[Ident],
    arg_types: &[KernelType],
    span: Span,
) -> syn::Result<(Ident, KernelType, TokenStream)> {
    if arg_regs.len() != 1 {
        return Err(syn::Error::new(
            span,
            format!("abs() expects 1 argument, got {}", arg_regs.len()),
        ));
    }
    let ty = &arg_types[0];
    let src = &arg_regs[0];
    let dst = ctx.fresh_reg();
    let ptx_ty = ctx.ptx_type_tokens(ty);

    let tokens = quote! {
        let #dst = alloc.alloc(PtxType::#ptx_ty);
        kernel.push(PtxInstruction::Arith(ArithOp::Abs {
            dst: #dst,
            src: Operand::Reg(#src),
            ty: PtxType::#ptx_ty,
        }));
    };

    Ok((dst, ty.clone(), tokens))
}

/// Lower a binary math function (min/max).
fn lower_binary_math(
    ctx: &mut LoweringContext,
    variant_name: &str,
    arg_regs: &[Ident],
    arg_types: &[KernelType],
    span: Span,
) -> syn::Result<(Ident, KernelType, TokenStream)> {
    if arg_regs.len() != 2 {
        return Err(syn::Error::new(
            span,
            format!(
                "{variant_name}() expects 2 arguments, got {}",
                arg_regs.len()
            ),
        ));
    }
    let ty = &arg_types[0];
    let lhs = &arg_regs[0];
    let rhs = &arg_regs[1];
    let dst = ctx.fresh_reg();
    let ptx_ty = ctx.ptx_type_tokens(ty);
    let variant = Ident::new(variant_name, Span::call_site());

    let tokens = quote! {
        let #dst = alloc.alloc(PtxType::#ptx_ty);
        kernel.push(PtxInstruction::Arith(ArithOp::#variant {
            dst: #dst,
            lhs: Operand::Reg(#lhs),
            rhs: Operand::Reg(#rhs),
            ty: PtxType::#ptx_ty,
        }));
    };

    Ok((dst, ty.clone(), tokens))
}

/// Lower `exp(x)` = `2^(x * log2(e))` → mul + ex2 (2 instructions).
fn lower_exp(
    ctx: &mut LoweringContext,
    arg_regs: &[Ident],
    arg_types: &[KernelType],
    span: Span,
) -> syn::Result<(Ident, KernelType, TokenStream)> {
    if arg_regs.len() != 1 {
        return Err(syn::Error::new(span, "exp() expects 1 argument"));
    }
    check_f32(&arg_types[0], span)?;

    let src = &arg_regs[0];
    let log2e_reg = ctx.fresh_reg();
    let scaled = ctx.fresh_reg();
    let dst = ctx.fresh_reg();

    // LOG2_E = 1.442695 (f32)
    let log2e: f32 = std::f32::consts::LOG2_E;

    let tokens = quote! {
        // exp(x) = 2^(x * log2(e))
        let #log2e_reg = alloc.alloc(PtxType::F32);
        kernel.push(PtxInstruction::Mov {
            dst: #log2e_reg,
            src: Operand::ImmF32(#log2e),
            ty: PtxType::F32,
        });
        let #scaled = alloc.alloc(PtxType::F32);
        kernel.push(PtxInstruction::Arith(ArithOp::Mul {
            dst: #scaled,
            lhs: Operand::Reg(#src),
            rhs: Operand::Reg(#log2e_reg),
            ty: PtxType::F32,
        }));
        let #dst = alloc.alloc(PtxType::F32);
        kernel.push(PtxInstruction::Arith(ArithOp::Ex2 {
            dst: #dst,
            src: Operand::Reg(#scaled),
        }));
    };

    Ok((dst, KernelType::F32, tokens))
}

/// Lower `log(x)` = `log2(x) * ln(2)` → lg2 + mul (2 instructions).
fn lower_log(
    ctx: &mut LoweringContext,
    arg_regs: &[Ident],
    arg_types: &[KernelType],
    span: Span,
) -> syn::Result<(Ident, KernelType, TokenStream)> {
    if arg_regs.len() != 1 {
        return Err(syn::Error::new(span, "log() expects 1 argument"));
    }
    check_f32(&arg_types[0], span)?;

    let src = &arg_regs[0];
    let log2_result = ctx.fresh_reg();
    let ln2_reg = ctx.fresh_reg();
    let dst = ctx.fresh_reg();

    // LN_2 = 0.6931472 (f32)
    let ln2: f32 = std::f32::consts::LN_2;

    let tokens = quote! {
        // ln(x) = log2(x) * ln(2)
        let #log2_result = alloc.alloc(PtxType::F32);
        kernel.push(PtxInstruction::Arith(ArithOp::Lg2 {
            dst: #log2_result,
            src: Operand::Reg(#src),
        }));
        let #ln2_reg = alloc.alloc(PtxType::F32);
        kernel.push(PtxInstruction::Mov {
            dst: #ln2_reg,
            src: Operand::ImmF32(#ln2),
            ty: PtxType::F32,
        });
        let #dst = alloc.alloc(PtxType::F32);
        kernel.push(PtxInstruction::Arith(ArithOp::Mul {
            dst: #dst,
            lhs: Operand::Reg(#log2_result),
            rhs: Operand::Reg(#ln2_reg),
            ty: PtxType::F32,
        }));
    };

    Ok((dst, KernelType::F32, tokens))
}

/// Lower `tanh(x)` = `(exp(2x) - 1) / (exp(2x) + 1)` (6 instructions).
///
/// CRITICAL: `exp(2x)` is computed once and the register is reused for
/// both the numerator (`exp2x - 1`) and denominator (`exp2x + 1`).
fn lower_tanh(
    ctx: &mut LoweringContext,
    arg_regs: &[Ident],
    arg_types: &[KernelType],
    span: Span,
) -> syn::Result<(Ident, KernelType, TokenStream)> {
    if arg_regs.len() != 1 {
        return Err(syn::Error::new(span, "tanh() expects 1 argument"));
    }
    check_f32(&arg_types[0], span)?;

    let x = &arg_regs[0];

    // Step 1: two_x = x * 2.0
    let two_reg = ctx.fresh_reg();
    let two_x = ctx.fresh_reg();

    // Step 2: exp2x = exp(2x) via mul + ex2
    let log2e_reg = ctx.fresh_reg();
    let scaled = ctx.fresh_reg();
    let exp2x = ctx.fresh_reg();

    // Step 3: numerator = exp2x - 1, denominator = exp2x + 1, result = num / den
    let one_reg = ctx.fresh_reg();
    let numerator = ctx.fresh_reg();
    let denominator = ctx.fresh_reg();
    let dst = ctx.fresh_reg();

    let log2e: f32 = std::f32::consts::LOG2_E;

    let tokens = quote! {
        // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)

        // Step 1: two_x = x * 2.0
        let #two_reg = alloc.alloc(PtxType::F32);
        kernel.push(PtxInstruction::Mov {
            dst: #two_reg,
            src: Operand::ImmF32(2.0f32),
            ty: PtxType::F32,
        });
        let #two_x = alloc.alloc(PtxType::F32);
        kernel.push(PtxInstruction::Arith(ArithOp::Mul {
            dst: #two_x,
            lhs: Operand::Reg(#x),
            rhs: Operand::Reg(#two_reg),
            ty: PtxType::F32,
        }));

        // Step 2: exp2x = 2^(two_x * log2(e))
        let #log2e_reg = alloc.alloc(PtxType::F32);
        kernel.push(PtxInstruction::Mov {
            dst: #log2e_reg,
            src: Operand::ImmF32(#log2e),
            ty: PtxType::F32,
        });
        let #scaled = alloc.alloc(PtxType::F32);
        kernel.push(PtxInstruction::Arith(ArithOp::Mul {
            dst: #scaled,
            lhs: Operand::Reg(#two_x),
            rhs: Operand::Reg(#log2e_reg),
            ty: PtxType::F32,
        }));
        let #exp2x = alloc.alloc(PtxType::F32);
        kernel.push(PtxInstruction::Arith(ArithOp::Ex2 {
            dst: #exp2x,
            src: Operand::Reg(#scaled),
        }));

        // Step 3: (exp2x - 1) / (exp2x + 1)
        // exp2x register is reused for both numerator and denominator
        let #one_reg = alloc.alloc(PtxType::F32);
        kernel.push(PtxInstruction::Mov {
            dst: #one_reg,
            src: Operand::ImmF32(1.0f32),
            ty: PtxType::F32,
        });
        let #numerator = alloc.alloc(PtxType::F32);
        kernel.push(PtxInstruction::Arith(ArithOp::Sub {
            dst: #numerator,
            lhs: Operand::Reg(#exp2x),
            rhs: Operand::Reg(#one_reg),
            ty: PtxType::F32,
        }));
        let #denominator = alloc.alloc(PtxType::F32);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: #denominator,
            lhs: Operand::Reg(#exp2x),
            rhs: Operand::Reg(#one_reg),
            ty: PtxType::F32,
        }));
        let #dst = alloc.alloc(PtxType::F32);
        kernel.push(PtxInstruction::Arith(ArithOp::Div {
            dst: #dst,
            lhs: Operand::Reg(#numerator),
            rhs: Operand::Reg(#denominator),
            ty: PtxType::F32,
        }));
    };

    Ok((dst, KernelType::F32, tokens))
}

/// Check that a type is f32 (required for approx math instructions).
fn check_f32(ty: &KernelType, span: Span) -> syn::Result<()> {
    if *ty != KernelType::F32 {
        return Err(syn::Error::new(
            span,
            format!(
                "math function requires f32 argument, got {}",
                ty.display_name()
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_arg(ctx: &mut LoweringContext) -> (Vec<Ident>, Vec<KernelType>) {
        let reg = Ident::new("_pyros_r0", Span::call_site());
        (vec![reg], vec![KernelType::F32])
    }

    #[test]
    fn lower_thread_idx_x() {
        let mut ctx = LoweringContext::new();
        let (result, ty, tokens) =
            lower_builtin(&mut ctx, "thread_idx_x", &[], &[], Span::call_site()).unwrap();

        assert_eq!(ty, KernelType::U32);
        assert!(result.to_string().starts_with("_pyros_r"));
        let code = tokens.to_string();
        assert!(code.contains("special :: tid_x"));
    }

    #[test]
    fn lower_block_idx_x() {
        let mut ctx = LoweringContext::new();
        let (_, ty, tokens) =
            lower_builtin(&mut ctx, "block_idx_x", &[], &[], Span::call_site()).unwrap();

        assert_eq!(ty, KernelType::U32);
        let code = tokens.to_string();
        assert!(code.contains("special :: ctaid_x"));
    }

    #[test]
    fn lower_block_dim_x() {
        let mut ctx = LoweringContext::new();
        let (_, _, tokens) =
            lower_builtin(&mut ctx, "block_dim_x", &[], &[], Span::call_site()).unwrap();
        assert!(tokens.to_string().contains("special :: ntid_x"));
    }

    #[test]
    fn lower_sqrt() {
        let mut ctx = LoweringContext::new();
        let (regs, types) = f32_arg(&mut ctx);
        let (_, ty, tokens) =
            lower_builtin(&mut ctx, "sqrt", &regs, &types, Span::call_site()).unwrap();

        assert_eq!(ty, KernelType::F32);
        let code = tokens.to_string();
        assert!(code.contains("ArithOp :: Sqrt"));
    }

    #[test]
    fn lower_abs_f32() {
        let mut ctx = LoweringContext::new();
        let (regs, types) = f32_arg(&mut ctx);
        let (_, ty, tokens) =
            lower_builtin(&mut ctx, "abs", &regs, &types, Span::call_site()).unwrap();

        assert_eq!(ty, KernelType::F32);
        assert!(tokens.to_string().contains("ArithOp :: Abs"));
    }

    #[test]
    fn lower_min_max() {
        let mut ctx = LoweringContext::new();
        let r0 = Ident::new("_pyros_r0", Span::call_site());
        let r1 = Ident::new("_pyros_r1", Span::call_site());
        let regs = vec![r0, r1];
        let types = vec![KernelType::F32, KernelType::F32];

        let (_, _, tokens) =
            lower_builtin(&mut ctx, "min", &regs, &types, Span::call_site()).unwrap();
        assert!(tokens.to_string().contains("ArithOp :: Min"));

        let (_, _, tokens) =
            lower_builtin(&mut ctx, "max", &regs, &types, Span::call_site()).unwrap();
        assert!(tokens.to_string().contains("ArithOp :: Max"));
    }

    #[test]
    fn lower_exp_synthesized() {
        let mut ctx = LoweringContext::new();
        let (regs, types) = f32_arg(&mut ctx);
        let (_, ty, tokens) =
            lower_builtin(&mut ctx, "exp", &regs, &types, Span::call_site()).unwrap();

        assert_eq!(ty, KernelType::F32);
        let code = tokens.to_string();
        // exp(x) = 2^(x * log2(e)) → mul + ex2
        assert!(
            code.contains("ArithOp :: Mul"),
            "should have mul for scaling"
        );
        assert!(code.contains("ArithOp :: Ex2"), "should have ex2");
    }

    #[test]
    fn lower_log_synthesized() {
        let mut ctx = LoweringContext::new();
        let (regs, types) = f32_arg(&mut ctx);
        let (_, ty, tokens) =
            lower_builtin(&mut ctx, "log", &regs, &types, Span::call_site()).unwrap();

        assert_eq!(ty, KernelType::F32);
        let code = tokens.to_string();
        // ln(x) = log2(x) * ln(2) → lg2 + mul
        assert!(code.contains("ArithOp :: Lg2"), "should have lg2");
        assert!(
            code.contains("ArithOp :: Mul"),
            "should have mul for scaling"
        );
    }

    #[test]
    fn lower_tanh_synthesized() {
        let mut ctx = LoweringContext::new();
        let (regs, types) = f32_arg(&mut ctx);
        let (_, ty, tokens) =
            lower_builtin(&mut ctx, "tanh", &regs, &types, Span::call_site()).unwrap();

        assert_eq!(ty, KernelType::F32);
        let code = tokens.to_string();
        // tanh = (exp(2x) - 1) / (exp(2x) + 1)
        assert!(
            code.contains("ArithOp :: Ex2"),
            "should have ex2 for exp(2x)"
        );
        assert!(
            code.contains("ArithOp :: Sub"),
            "should have sub for numerator"
        );
        assert!(
            code.contains("ArithOp :: Add"),
            "should have add for denominator"
        );
        assert!(
            code.contains("ArithOp :: Div"),
            "should have div for result"
        );

        // exp2x register should appear in BOTH sub and add (reuse, not recompute)
        // Count occurrences of Ex2 — should be exactly 1
        let ex2_count = code.matches("ArithOp :: Ex2").count();
        assert_eq!(
            ex2_count, 1,
            "exp(2x) should be computed once, not twice (got {ex2_count} Ex2 instructions)"
        );
    }

    #[test]
    fn reject_unknown_builtin() {
        let mut ctx = LoweringContext::new();
        let err = lower_builtin(&mut ctx, "foobar", &[], &[], Span::call_site()).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("unknown function"));
        assert!(msg.contains("thread_idx_x")); // should list available functions
    }

    #[test]
    fn reject_args_on_thread_idx() {
        let mut ctx = LoweringContext::new();
        let reg = Ident::new("_pyros_r0", Span::call_site());
        let err = lower_builtin(
            &mut ctx,
            "thread_idx_x",
            &[reg],
            &[KernelType::U32],
            Span::call_site(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("no arguments"));
    }
}
