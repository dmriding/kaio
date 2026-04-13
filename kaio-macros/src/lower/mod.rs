//! Lowering pass: transform kernel IR into `TokenStream` fragments
//! that construct `kaio-core` IR at runtime.

pub mod arith;
pub mod builtins;
pub mod cast;
pub mod compare;
pub mod memory;
pub mod params;

use std::collections::HashMap;

use proc_macro2::{Ident, Span, TokenStream};
use quote::{format_ident, quote};

use crate::kernel_ir::KernelType;
use crate::kernel_ir::expr::{KernelExpr, UnaryOpKind};
use crate::kernel_ir::stmt::KernelStmt;

/// Context threaded through all lowering functions.
#[allow(dead_code)] // Used in Sprint 2.6 codegen; tested via lower/arith.rs and lower/mod.rs tests
pub struct LoweringContext {
    /// Monotonic counter for generating unique register variable names
    /// (`_kaio_r0`, `_kaio_r1`, ...) in the generated `build_ptx()` code.
    reg_counter: u32,
    /// Counter for generating unique label names (`IF_END_0`, `IF_ELSE_1`, ...).
    label_counter: u32,
    /// Variable-to-register mapping.
    /// Key: variable name, Value: (register Ident in generated code, type).
    /// Populated by parameter loading (Sprint 2.6) and let-binding lowering.
    pub locals: HashMap<String, (Ident, KernelType)>,
    /// Cached `cvta.to.global` results per pointer parameter.
    /// Key: param name, Value: register Ident holding the global address.
    /// One CvtaToGlobal per pointer, reused across multiple index accesses.
    pub global_addrs: HashMap<String, Ident>,
    /// Declared shared memory buffers.
    /// Key: buffer name, Value: (element type, element count).
    pub shared_arrays: HashMap<String, (KernelType, usize)>,
    /// Total block size (total threads per block). Set by codegen before
    /// body lowering. For 1D this is `block_size`, for 2D it is `x * y`.
    /// Needed by reductions to compute `num_warps = block_size / 32`.
    pub block_size: Option<u32>,
    /// Block size X dimension. `Some` for 2D kernels, `None` for 1D.
    /// Preserved so Sprint 4.3+ tile logic can access individual dimensions.
    pub block_size_x: Option<u32>,
    /// Block size Y dimension. `Some` for 2D kernels, `None` for 1D.
    pub block_size_y: Option<u32>,
    /// Whether reduction shared memory (`_kaio_reduce_smem`) has been allocated.
    /// Reused across multiple `block_reduce_*` calls in the same kernel.
    pub reduce_smem_allocated: bool,
}

#[allow(dead_code)]
impl LoweringContext {
    /// Create a new lowering context.
    pub fn new() -> Self {
        Self {
            reg_counter: 0,
            label_counter: 0,
            locals: HashMap::new(),
            global_addrs: HashMap::new(),
            shared_arrays: HashMap::new(),
            block_size: None,
            block_size_x: None,
            block_size_y: None,
            reduce_smem_allocated: false,
        }
    }

    /// Allocate a fresh register variable name for the generated code.
    pub fn fresh_reg(&mut self) -> Ident {
        let id = self.reg_counter;
        self.reg_counter += 1;
        format_ident!("_kaio_r{}", id)
    }

    /// Convert a `KernelType` to the `PtxType` variant name as an `Ident`
    /// for use in generated code (e.g., `F32`, `S32`, `U64`).
    pub fn ptx_type_tokens(&self, ty: &KernelType) -> Ident {
        Ident::new(ty.ptx_type_token(), Span::call_site())
    }

    /// Generate a unique label name (e.g., `"IF_END_0"`, `"IF_ELSE_3"`).
    pub fn fresh_label(&mut self, prefix: &str) -> String {
        let id = self.label_counter;
        self.label_counter += 1;
        format!("{prefix}_{id}")
    }
}

/// Recursively lower a `KernelExpr` to a `TokenStream` that builds IR.
///
/// Returns `(register_ident, result_type, token_stream)`:
/// - `register_ident`: the Ident of the register in generated code holding the result
/// - `result_type`: the `KernelType` of the expression
/// - `token_stream`: the generated Rust code that constructs the IR
///
/// For `Var` lookups, the token stream is empty (the register already exists).
/// For everything else, the token stream contains `alloc.alloc()` + `kernel.push()` calls.
#[allow(dead_code)] // Used in Sprint 2.6 codegen; tested in this module's tests
pub fn lower_expr(
    ctx: &mut LoweringContext,
    expr: &KernelExpr,
) -> syn::Result<(Ident, KernelType, TokenStream)> {
    match expr {
        // Variable reference: look up in locals, no codegen needed
        KernelExpr::Var(name, span) => {
            let (reg, ty) = ctx.locals.get(name).cloned().ok_or_else(|| {
                syn::Error::new(*span, format!("undefined variable `{name}` in GPU kernel"))
            })?;
            Ok((reg, ty, TokenStream::new()))
        }

        // Integer literal: allocate register, mov immediate
        KernelExpr::LitInt(value, ty, _span) => {
            let dst = ctx.fresh_reg();
            let ptx_ty = ctx.ptx_type_tokens(ty);

            // Choose the right Operand constructor for the type
            let operand = match ty {
                KernelType::I32 => {
                    let v = *value as i32;
                    quote! { Operand::ImmI32(#v) }
                }
                KernelType::U32 => {
                    let v = *value as u32;
                    quote! { Operand::ImmU32(#v) }
                }
                KernelType::I64 => {
                    let v = *value;
                    quote! { Operand::ImmI64(#v) }
                }
                KernelType::U64 => {
                    let v = *value as u64;
                    quote! { Operand::ImmU64(#v) }
                }
                _ => {
                    return Err(syn::Error::new(
                        Span::call_site(),
                        format!("integer literal cannot have type {}", ty.display_name()),
                    ));
                }
            };

            let tokens = quote! {
                let #dst = alloc.alloc(PtxType::#ptx_ty);
                kernel.push(PtxInstruction::Mov {
                    dst: #dst,
                    src: #operand,
                    ty: PtxType::#ptx_ty,
                });
            };
            Ok((dst, ty.clone(), tokens))
        }

        // Float literal: allocate register, mov immediate
        KernelExpr::LitFloat(value, ty, _span) => {
            let dst = ctx.fresh_reg();
            let ptx_ty = ctx.ptx_type_tokens(ty);

            let operand = match ty {
                KernelType::F32 => {
                    let v = *value as f32;
                    quote! { Operand::ImmF32(#v) }
                }
                KernelType::F64 => {
                    let v = *value;
                    quote! { Operand::ImmF64(#v) }
                }
                _ => {
                    return Err(syn::Error::new(
                        Span::call_site(),
                        format!("float literal cannot have type {}", ty.display_name()),
                    ));
                }
            };

            let tokens = quote! {
                let #dst = alloc.alloc(PtxType::#ptx_ty);
                kernel.push(PtxInstruction::Mov {
                    dst: #dst,
                    src: #operand,
                    ty: PtxType::#ptx_ty,
                });
            };
            Ok((dst, ty.clone(), tokens))
        }

        // Bool literal
        KernelExpr::LitBool(_value, span) => Err(syn::Error::new(
            *span,
            "boolean literals in expressions are not yet supported (use comparisons)",
        )),

        // Binary operation: lower both sides recursively, then lower the op
        KernelExpr::BinOp {
            op, lhs, rhs, span, ..
        } => {
            if op.is_arithmetic() {
                let (lhs_reg, lhs_ty, lhs_tokens) = lower_expr(ctx, lhs)?;
                let (rhs_reg, _rhs_ty, rhs_tokens) = lower_expr(ctx, rhs)?;
                // TODO (Sprint 2.7): type-check that lhs_ty == rhs_ty
                let (dst, op_tokens) = arith::lower_binop(ctx, op, &lhs_reg, &rhs_reg, &lhs_ty);
                let combined = quote! { #lhs_tokens #rhs_tokens #op_tokens };
                Ok((dst, lhs_ty, combined))
            } else if op.is_comparison() {
                let (lhs_reg, lhs_ty, lhs_tokens) = lower_expr(ctx, lhs)?;
                let (rhs_reg, _rhs_ty, rhs_tokens) = lower_expr(ctx, rhs)?;
                let (pred, cmp_tokens) =
                    compare::lower_comparison(ctx, op, &lhs_reg, &rhs_reg, &lhs_ty);
                let combined = quote! { #lhs_tokens #rhs_tokens #cmp_tokens };
                Ok((pred, KernelType::Bool, combined))
            } else if op.is_bitwise() {
                // Sprint 7.0 D2: bitwise binops. Shr signedness is preserved by
                // lhs_ty flowing through to ArithOp::Shr (see AD2 canary tests
                // in kaio-core/src/instr/arith.rs + lower/arith.rs).
                let (lhs_reg, lhs_ty, lhs_tokens) = lower_expr(ctx, lhs)?;
                let (rhs_reg, _rhs_ty, rhs_tokens) = lower_expr(ctx, rhs)?;
                if !lhs_ty.is_integer() {
                    return Err(syn::Error::new(
                        *span,
                        format!(
                            "bitwise operator {op:?} requires integer operands, got {}",
                            lhs_ty.display_name()
                        ),
                    ));
                }
                let (dst, op_tokens) = arith::lower_bitop(ctx, op, &lhs_reg, &rhs_reg, &lhs_ty);
                let combined = quote! { #lhs_tokens #rhs_tokens #op_tokens };
                Ok((dst, lhs_ty, combined))
            } else {
                // Logical &&/|| (AD4) land in D4.
                Err(syn::Error::new(
                    *span,
                    format!("operator {op:?} lowering not yet implemented"),
                ))
            }
        }

        // Unary negation / NOT
        KernelExpr::UnaryOp { op, expr, span } => match op {
            UnaryOpKind::Neg => {
                let (src_reg, src_ty, src_tokens) = lower_expr(ctx, expr)?;
                let (dst, neg_tokens) = arith::lower_neg(ctx, &src_reg, &src_ty);
                let combined = quote! { #src_tokens #neg_tokens };
                Ok((dst, src_ty, combined))
            }
            UnaryOpKind::Not => {
                // Sprint 7.0 D2 AD3: context-dispatch on source type.
                // - Integer (I32/U32/I64/U64) → bitwise not (emits not.b{size})
                // - Bool → logical not on predicate (emits not.pred)
                // Any other type is an error — `!` doesn't make sense on floats.
                let (src_reg, src_ty, src_tokens) = lower_expr(ctx, expr)?;
                if !src_ty.is_integer() && src_ty != KernelType::Bool {
                    return Err(syn::Error::new(
                        *span,
                        format!(
                            "unary `!` requires integer or bool operand, got {}",
                            src_ty.display_name()
                        ),
                    ));
                }
                let (dst, not_tokens) = arith::lower_not(ctx, &src_reg, &src_ty);
                let combined = quote! { #src_tokens #not_tokens };
                Ok((dst, src_ty, combined))
            }
        },

        // Parenthesized: just recurse
        KernelExpr::Paren(inner, _span) => lower_expr(ctx, inner),

        // Array index read: a[idx]
        KernelExpr::Index { array, index, span } => {
            // Check shared memory first
            if let Some((elem_ty, _count)) = ctx.shared_arrays.get(array).cloned() {
                let (idx_reg, _idx_ty, idx_tokens) = lower_expr(ctx, index)?;
                let (result, mem_tokens) =
                    memory::lower_shared_index_read(ctx, array, &idx_reg, &elem_ty);
                return Ok((result, elem_ty, quote! { #idx_tokens #mem_tokens }));
            }
            // Global memory path
            let (array_reg, array_ty) = ctx.locals.get(array).cloned().ok_or_else(|| {
                syn::Error::new(*span, format!("undefined array `{array}` in GPU kernel"))
            })?;
            let elem_ty = array_ty.elem_type().cloned().ok_or_else(|| {
                syn::Error::new(
                    *span,
                    format!(
                        "cannot index into `{array}`: type `{}` is not a slice",
                        array_ty.display_name()
                    ),
                )
            })?;
            let (idx_reg, _idx_ty, idx_tokens) = lower_expr(ctx, index)?;
            let (result, mem_tokens) =
                memory::lower_index_read(ctx, array, &array_reg, &idx_reg, &elem_ty);
            Ok((result, elem_ty, quote! { #idx_tokens #mem_tokens }))
        }
        // Built-in function call: thread_idx_x(), sqrt(x), etc.
        KernelExpr::BuiltinCall { name, args, span } => {
            let mut arg_regs = Vec::new();
            let mut arg_types = Vec::new();
            let mut arg_tokens = TokenStream::new();
            for arg in args {
                let (reg, ty, tokens) = lower_expr(ctx, arg)?;
                arg_regs.push(reg);
                arg_types.push(ty);
                arg_tokens.extend(tokens);
            }
            let (result, result_ty, builtin_tokens) =
                builtins::lower_builtin(ctx, name, &arg_regs, &arg_types, *span)?;
            Ok((result, result_ty, quote! { #arg_tokens #builtin_tokens }))
        }
        // Type cast: x as f32
        KernelExpr::Cast {
            expr, target_ty, ..
        } => {
            let (src_reg, src_ty, src_tokens) = lower_expr(ctx, expr)?;
            let (dst, cast_tokens) = cast::lower_cast(ctx, &src_reg, &src_ty, target_ty);
            Ok((dst, target_ty.clone(), quote! { #src_tokens #cast_tokens }))
        }
    }
}

/// Lower a sequence of kernel statements to a combined `TokenStream`.
#[allow(dead_code)] // Used in Sprint 2.6 codegen
pub fn lower_stmts(ctx: &mut LoweringContext, stmts: &[KernelStmt]) -> syn::Result<TokenStream> {
    let mut combined = TokenStream::new();
    for stmt in stmts {
        let tokens = lower_stmt(ctx, stmt)?;
        combined.extend(tokens);
    }
    Ok(combined)
}

/// Generate tokens that conditionally emit a PTX comment annotation.
///
/// The `_kaio_annotate` variable must be a bare identifier — it resolves
/// at runtime in the generated `build_ptx()` function, not at proc macro
/// expansion time. Same pattern as `kernel` and `alloc`.
fn annotation_tokens(description: &str) -> TokenStream {
    quote! {
        if _kaio_annotate {
            kernel.push(PtxInstruction::Comment(#description.to_string()));
        }
    }
}

/// Lower a single kernel statement to a `TokenStream`.
#[allow(dead_code)] // Used in Sprint 2.6 codegen; tested here
pub fn lower_stmt(ctx: &mut LoweringContext, stmt: &KernelStmt) -> syn::Result<TokenStream> {
    match stmt {
        // let x = expr; — lower value, register in locals
        KernelStmt::Let { name, value, .. } => {
            let ann = annotation_tokens(&format!("let {name}"));
            let (reg, ty, expr_tokens) = lower_expr(ctx, value)?;
            // Allows variable shadowing (e.g., reusing `i` in multiple loops).
            // Each `let` allocates a fresh register — the old register just
            // becomes unreferenced in subsequent code.
            // If the value expression reuses an existing register (e.g.,
            // `let i = tid` where tid is a Var lookup with empty tokens),
            // allocate a fresh register and copy the value. This prevents
            // the new variable from aliasing the source — critical for
            // `let mut i = tid; i += 1;` which must not corrupt `tid`.
            let (final_reg, final_tokens) = if expr_tokens.is_empty() {
                let new_reg = ctx.fresh_reg();
                let ptx_ty = ctx.ptx_type_tokens(&ty);
                let copy_tokens = quote! {
                    let #new_reg = alloc.alloc(PtxType::#ptx_ty);
                    kernel.push(PtxInstruction::Mov {
                        dst: #new_reg,
                        src: Operand::Reg(#reg),
                        ty: PtxType::#ptx_ty,
                    });
                };
                (new_reg, copy_tokens)
            } else {
                (reg, expr_tokens)
            };
            ctx.locals.insert(name.clone(), (final_reg, ty));
            Ok(quote! { #ann #final_tokens })
        }

        // if cond { then } [else { otherwise }]
        KernelStmt::If {
            condition,
            then_body,
            else_body,
            ..
        } => {
            let ann = annotation_tokens("if ...");
            // 1. Lower condition to predicate register
            let (pred_reg, _pred_ty, cond_tokens) = lower_expr(ctx, condition)?;

            // 2. Generate labels
            let has_else = else_body.is_some();
            let end_label = ctx.fresh_label("IF_END");
            let else_label = if has_else {
                Some(ctx.fresh_label("IF_ELSE"))
            } else {
                None
            };

            // 3. Branch: @!pred bra target (skip then-block when condition is false)
            let skip_target = else_label.as_deref().unwrap_or(&end_label);
            let skip_target_str = skip_target.to_string();
            let branch_tokens = quote! {
                kernel.push(PtxInstruction::Control(ControlOp::BraPred {
                    pred: #pred_reg,
                    target: #skip_target_str.to_string(),
                    negate: true,
                }));
            };

            // 4. Lower then-body
            let then_tokens = lower_stmts(ctx, then_body)?;

            // 5. If else: unconditional branch past else, else label, else body
            let else_tokens = if let Some(else_stmts) = else_body {
                let else_lbl = else_label.as_ref().unwrap();
                let end_lbl_str = end_label.clone();
                let else_body_tokens = lower_stmts(ctx, else_stmts)?;
                quote! {
                    kernel.push(PtxInstruction::Control(ControlOp::Bra {
                        target: #end_lbl_str.to_string(),
                    }));
                    kernel.push(PtxInstruction::Label(#else_lbl.to_string()));
                    #else_body_tokens
                }
            } else {
                TokenStream::new()
            };

            // 6. End label
            let end_label_tokens = quote! {
                kernel.push(PtxInstruction::Label(#end_label.to_string()));
            };

            Ok(quote! {
                #ann
                #cond_tokens
                #branch_tokens
                #then_tokens
                #else_tokens
                #end_label_tokens
            })
        }

        // Bare expression statement — only annotate bar_sync
        KernelStmt::Expr(expr, _span) => {
            let ann = if let KernelExpr::BuiltinCall { name, .. } = expr {
                if name == "bar_sync" {
                    annotation_tokens("bar_sync()")
                } else {
                    TokenStream::new()
                }
            } else {
                TokenStream::new()
            };
            let (_reg, _ty, tokens) = lower_expr(ctx, expr)?;
            Ok(quote! { #ann #tokens })
        }

        // x = expr — lower value, emit Mov to existing register
        KernelStmt::Assign {
            name, value, span, ..
        } => {
            let (existing_reg, existing_ty) = ctx.locals.get(name).cloned().ok_or_else(|| {
                syn::Error::new(
                    *span,
                    format!("cannot assign to undefined variable `{name}` in GPU kernel"),
                )
            })?;
            let (val_reg, _val_ty, val_tokens) = lower_expr(ctx, value)?;
            let ptx_ty = ctx.ptx_type_tokens(&existing_ty);
            let tokens = quote! {
                #val_tokens
                kernel.push(PtxInstruction::Mov {
                    dst: #existing_reg,
                    src: Operand::Reg(#val_reg),
                    ty: PtxType::#ptx_ty,
                });
            };
            Ok(tokens)
        }
        KernelStmt::IndexAssign {
            array,
            index,
            value,
            span,
        } => {
            let ann = annotation_tokens(&format!("{array}[...] = ..."));
            // Check shared memory first — always mutable
            if let Some((elem_ty, _count)) = ctx.shared_arrays.get(array).cloned() {
                let (idx_reg, _idx_ty, idx_tokens) = lower_expr(ctx, index)?;
                let (val_reg, _val_ty, val_tokens) = lower_expr(ctx, value)?;
                let store_tokens =
                    memory::lower_shared_index_write(ctx, array, &idx_reg, &val_reg, &elem_ty);
                return Ok(quote! { #ann #idx_tokens #val_tokens #store_tokens });
            }
            // Global memory path
            let (array_reg, array_ty) = ctx.locals.get(array).cloned().ok_or_else(|| {
                syn::Error::new(*span, format!("undefined array `{array}` in GPU kernel"))
            })?;
            // Must be &mut [T] for writes
            if !array_ty.is_mut_slice() {
                return Err(syn::Error::new(
                    *span,
                    format!(
                        "cannot write to immutable slice parameter `{array}`: \
                         declare as `&mut [T]`"
                    ),
                ));
            }
            let elem_ty = array_ty.elem_type().cloned().ok_or_else(|| {
                syn::Error::new(*span, "internal error: mut slice has no element type")
            })?;
            let (idx_reg, _idx_ty, idx_tokens) = lower_expr(ctx, index)?;
            let (val_reg, _val_ty, val_tokens) = lower_expr(ctx, value)?;
            let store_tokens =
                memory::lower_index_write(ctx, array, &array_reg, &idx_reg, &val_reg, &elem_ty);
            Ok(quote! { #ann #idx_tokens #val_tokens #store_tokens })
        }

        // shared_mem![T; N] — declare shared memory buffer
        KernelStmt::SharedMemDecl {
            name,
            elem_ty,
            count,
            span,
        } => {
            let ann = annotation_tokens(&format!(
                "shared_mem {name}: [{}; {count}]",
                elem_ty.display_name()
            ));
            if ctx.shared_arrays.contains_key(name) {
                return Err(syn::Error::new(
                    *span,
                    format!("shared memory buffer `{name}` already declared in this kernel"),
                ));
            }
            ctx.shared_arrays
                .insert(name.clone(), (elem_ty.clone(), *count));
            let size_bytes = (elem_ty.size_bytes() * count) as u32;
            let align = elem_ty.size_bytes() as u32;
            let ptx_name = name.clone();
            Ok(quote! {
                #ann
                kernel.add_shared_decl(SharedDecl {
                    name: #ptx_name.to_string(),
                    align: #align,
                    size_bytes: #size_bytes,
                });
            })
        }

        // for var in start..end { body }
        KernelStmt::For {
            var,
            start,
            end,
            body,
            span,
        } => {
            let ann = annotation_tokens(&format!("for {var}"));
            // 1. Lower end first (its type drives the counter type)
            let (end_reg, end_ty, end_tokens) = lower_expr(ctx, end)?;

            // 2. Lower start — coerce unsuffixed literals to match end type
            let coerced_start = coerce_literal_type(start, &end_ty);
            let start_expr = coerced_start.as_ref().unwrap_or(start);
            let (start_reg, start_ty, start_tokens) = lower_expr(ctx, start_expr)?;

            // Type check: start and end must have the same type
            if start_ty != end_ty {
                return Err(syn::Error::new(
                    *span,
                    format!(
                        "`for` loop range type mismatch: start is `{}` but end is `{}` \
                         — use explicit suffix (e.g., `0u32..n`)",
                        start_ty.display_name(),
                        end_ty.display_name()
                    ),
                ));
            }

            let counter_ty = end_ty;
            let ptx_ty = ctx.ptx_type_tokens(&counter_ty);

            // 3. Allocate counter register, init from start
            let counter_reg = ctx.fresh_reg();
            let init_tokens = quote! {
                let #counter_reg = alloc.alloc(PtxType::#ptx_ty);
                kernel.push(PtxInstruction::Mov {
                    dst: #counter_reg,
                    src: Operand::Reg(#start_reg),
                    ty: PtxType::#ptx_ty,
                });
            };

            // 4. Register loop var in locals
            let prev_local = ctx
                .locals
                .insert(var.clone(), (counter_reg.clone(), counter_ty.clone()));

            // 5. Generate labels
            let loop_start = ctx.fresh_label("LOOP_START");
            let loop_end = ctx.fresh_label("LOOP_END");

            // 6. Emit loop start label
            let start_label_tokens = quote! {
                kernel.push(PtxInstruction::Label(#loop_start.to_string()));
            };

            // 7. Bounds check: setp.ge counter, end → @pred bra LOOP_END
            let pred_reg = ctx.fresh_reg();
            let cmp_tokens = quote! {
                let #pred_reg = alloc.alloc(PtxType::Pred);
                kernel.push(PtxInstruction::Control(ControlOp::SetP {
                    dst: #pred_reg,
                    cmp_op: CmpOp::Ge,
                    lhs: Operand::Reg(#counter_reg),
                    rhs: Operand::Reg(#end_reg),
                    ty: PtxType::#ptx_ty,
                }));
                kernel.push(PtxInstruction::Control(ControlOp::BraPred {
                    pred: #pred_reg,
                    target: #loop_end.to_string(),
                    negate: false,
                }));
            };

            // 8. Lower body
            let body_tokens = lower_stmts(ctx, body)?;

            // 9. Increment counter in-place: add counter, counter, 1
            let imm_one = match &counter_ty {
                KernelType::I32 => quote! { Operand::ImmI32(1) },
                KernelType::U32 => quote! { Operand::ImmU32(1) },
                KernelType::I64 => quote! { Operand::ImmI64(1) },
                KernelType::U64 => quote! { Operand::ImmU64(1) },
                _ => {
                    return Err(syn::Error::new(
                        *span,
                        format!(
                            "`for` loop counter must be an integer type, got `{}`",
                            counter_ty.display_name()
                        ),
                    ));
                }
            };
            let inc_tokens = quote! {
                kernel.push(PtxInstruction::Arith(ArithOp::Add {
                    dst: #counter_reg,
                    lhs: Operand::Reg(#counter_reg),
                    rhs: #imm_one,
                    ty: PtxType::#ptx_ty,
                }));
            };

            // 10. Back-edge: bra LOOP_START
            let back_edge_tokens = quote! {
                kernel.push(PtxInstruction::Control(ControlOp::Bra {
                    target: #loop_start.to_string(),
                }));
            };

            // 11. Loop end label
            let end_label_tokens = quote! {
                kernel.push(PtxInstruction::Label(#loop_end.to_string()));
            };

            // 12. Remove loop var from locals (restore previous if shadowed)
            match prev_local {
                Some(prev) => {
                    ctx.locals.insert(var.clone(), prev);
                }
                None => {
                    ctx.locals.remove(var);
                }
            }

            Ok(quote! {
                #ann
                #end_tokens
                #start_tokens
                #init_tokens
                #start_label_tokens
                #cmp_tokens
                #body_tokens
                #inc_tokens
                #back_edge_tokens
                #end_label_tokens
            })
        }

        // while condition { body }
        KernelStmt::While {
            condition, body, ..
        } => {
            let ann = annotation_tokens("while ...");
            // 1. Generate labels
            let loop_start = ctx.fresh_label("LOOP_START");
            let loop_end = ctx.fresh_label("LOOP_END");

            // 2. Emit loop start label
            let start_label_tokens = quote! {
                kernel.push(PtxInstruction::Label(#loop_start.to_string()));
            };

            // 3. Lower condition → predicate
            let (pred_reg, _pred_ty, cond_tokens) = lower_expr(ctx, condition)?;

            // 4. Branch: @!pred bra LOOP_END (exit if condition is false)
            let branch_tokens = quote! {
                kernel.push(PtxInstruction::Control(ControlOp::BraPred {
                    pred: #pred_reg,
                    target: #loop_end.to_string(),
                    negate: true,
                }));
            };

            // 5. Lower body
            let body_tokens = lower_stmts(ctx, body)?;

            // 6. Back-edge: bra LOOP_START
            let back_edge_tokens = quote! {
                kernel.push(PtxInstruction::Control(ControlOp::Bra {
                    target: #loop_start.to_string(),
                }));
            };

            // 7. Loop end label
            let end_label_tokens = quote! {
                kernel.push(PtxInstruction::Label(#loop_end.to_string()));
            };

            Ok(quote! {
                #ann
                #start_label_tokens
                #cond_tokens
                #branch_tokens
                #body_tokens
                #back_edge_tokens
                #end_label_tokens
            })
        }
    }
}

/// If `expr` is an unsuffixed integer literal (`LitInt` with default type),
/// return a copy with its type changed to `target_ty`. This allows `0..n`
/// where `n: u32` to work without requiring `0u32..n`.
fn coerce_literal_type(expr: &KernelExpr, target_ty: &KernelType) -> Option<KernelExpr> {
    use crate::kernel_ir::expr::KernelExpr as KE;
    match expr {
        // Default-typed integer literals (unsuffixed) have type I32 from the parser.
        // Coerce to target if the target is also an integer type.
        KE::LitInt(value, KernelType::I32, span)
            if target_ty.is_integer() && *target_ty != KernelType::I32 =>
        {
            Some(KE::LitInt(*value, target_ty.clone(), *span))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_ir::expr::BinOpKind;

    #[test]
    fn lower_var_lookup() {
        let mut ctx = LoweringContext::new();
        let reg = Ident::new("_kaio_r5", Span::call_site());
        ctx.locals
            .insert("x".to_string(), (reg.clone(), KernelType::F32));

        let expr = KernelExpr::Var("x".to_string(), Span::call_site());
        let (result_reg, result_ty, tokens) = lower_expr(&mut ctx, &expr).unwrap();

        assert_eq!(result_reg.to_string(), "_kaio_r5");
        assert_eq!(result_ty, KernelType::F32);
        assert!(tokens.is_empty()); // No codegen for var lookup
    }

    #[test]
    fn lower_var_undefined() {
        let mut ctx = LoweringContext::new();
        let expr = KernelExpr::Var("nonexistent".to_string(), Span::call_site());
        let err = lower_expr(&mut ctx, &expr).unwrap_err();
        assert!(err.to_string().contains("undefined variable"));
    }

    #[test]
    fn lower_int_literal() {
        let mut ctx = LoweringContext::new();
        let expr = KernelExpr::LitInt(42, KernelType::I32, Span::call_site());
        let (dst, ty, tokens) = lower_expr(&mut ctx, &expr).unwrap();

        assert_eq!(ty, KernelType::I32);
        assert!(dst.to_string().starts_with("_kaio_r"));
        let code = tokens.to_string();
        assert!(code.contains("alloc . alloc"));
        assert!(code.contains("Mov"));
        assert!(code.contains("ImmI32"));
    }

    #[test]
    fn lower_float_literal() {
        let mut ctx = LoweringContext::new();
        let expr = KernelExpr::LitFloat(1.0, KernelType::F32, Span::call_site());
        let (_dst, ty, tokens) = lower_expr(&mut ctx, &expr).unwrap();

        assert_eq!(ty, KernelType::F32);
        let code = tokens.to_string();
        assert!(code.contains("ImmF32"));
    }

    #[test]
    fn lower_binop_add() {
        let mut ctx = LoweringContext::new();
        ctx.locals.insert(
            "a".to_string(),
            (Ident::new("_kaio_r0", Span::call_site()), KernelType::F32),
        );
        ctx.locals.insert(
            "b".to_string(),
            (Ident::new("_kaio_r1", Span::call_site()), KernelType::F32),
        );

        let expr = KernelExpr::BinOp {
            op: BinOpKind::Add,
            lhs: Box::new(KernelExpr::Var("a".to_string(), Span::call_site())),
            rhs: Box::new(KernelExpr::Var("b".to_string(), Span::call_site())),
            span: Span::call_site(),
        };
        let (dst, ty, tokens) = lower_expr(&mut ctx, &expr).unwrap();

        assert_eq!(ty, KernelType::F32);
        assert!(dst.to_string().starts_with("_kaio_r"));
        let code = tokens.to_string();
        assert!(code.contains("ArithOp :: Add"));
    }

    #[test]
    fn lower_nested_a_plus_b_times_c() {
        let mut ctx = LoweringContext::new();
        ctx.locals.insert(
            "a".to_string(),
            (Ident::new("_kaio_r0", Span::call_site()), KernelType::F32),
        );
        ctx.locals.insert(
            "b".to_string(),
            (Ident::new("_kaio_r1", Span::call_site()), KernelType::F32),
        );
        ctx.locals.insert(
            "c".to_string(),
            (Ident::new("_kaio_r2", Span::call_site()), KernelType::F32),
        );

        // a + b * c -> Add(a, Mul(b, c))
        let expr = KernelExpr::BinOp {
            op: BinOpKind::Add,
            lhs: Box::new(KernelExpr::Var("a".to_string(), Span::call_site())),
            rhs: Box::new(KernelExpr::BinOp {
                op: BinOpKind::Mul,
                lhs: Box::new(KernelExpr::Var("b".to_string(), Span::call_site())),
                rhs: Box::new(KernelExpr::Var("c".to_string(), Span::call_site())),
                span: Span::call_site(),
            }),
            span: Span::call_site(),
        };
        let (_dst, ty, tokens) = lower_expr(&mut ctx, &expr).unwrap();

        assert_eq!(ty, KernelType::F32);
        let code = tokens.to_string();
        // Mul should appear before Add (evaluation order: inner first)
        let mul_pos = code.find("ArithOp :: Mul").expect("should contain Mul");
        let add_pos = code.find("ArithOp :: Add").expect("should contain Add");
        assert!(
            mul_pos < add_pos,
            "Mul should be emitted before Add in evaluation order"
        );
    }

    #[test]
    fn lower_unary_neg() {
        let mut ctx = LoweringContext::new();
        ctx.locals.insert(
            "x".to_string(),
            (Ident::new("_kaio_r0", Span::call_site()), KernelType::F32),
        );

        let expr = KernelExpr::UnaryOp {
            op: UnaryOpKind::Neg,
            expr: Box::new(KernelExpr::Var("x".to_string(), Span::call_site())),
            span: Span::call_site(),
        };
        let (_dst, ty, tokens) = lower_expr(&mut ctx, &expr).unwrap();

        assert_eq!(ty, KernelType::F32);
        let code = tokens.to_string();
        assert!(code.contains("ArithOp :: Neg"));
    }

    #[test]
    fn lower_paren_recurses() {
        let mut ctx = LoweringContext::new();
        ctx.locals.insert(
            "x".to_string(),
            (Ident::new("_kaio_r0", Span::call_site()), KernelType::F32),
        );

        let expr = KernelExpr::Paren(
            Box::new(KernelExpr::Var("x".to_string(), Span::call_site())),
            Span::call_site(),
        );
        let (reg, ty, tokens) = lower_expr(&mut ctx, &expr).unwrap();

        assert_eq!(reg.to_string(), "_kaio_r0");
        assert_eq!(ty, KernelType::F32);
        assert!(tokens.is_empty());
    }

    // --- Sprint 2.3: Comparisons + If/Else ---

    #[test]
    fn lower_comparison_in_expr() {
        let mut ctx = LoweringContext::new();
        ctx.locals.insert(
            "x".to_string(),
            (Ident::new("_kaio_r0", Span::call_site()), KernelType::U32),
        );
        ctx.locals.insert(
            "n".to_string(),
            (Ident::new("_kaio_r1", Span::call_site()), KernelType::U32),
        );

        let expr = KernelExpr::BinOp {
            op: BinOpKind::Lt,
            lhs: Box::new(KernelExpr::Var("x".to_string(), Span::call_site())),
            rhs: Box::new(KernelExpr::Var("n".to_string(), Span::call_site())),
            span: Span::call_site(),
        };
        let (pred, ty, tokens) = lower_expr(&mut ctx, &expr).unwrap();

        assert_eq!(ty, KernelType::Bool);
        assert!(pred.to_string().starts_with("_kaio_r"));
        let code = tokens.to_string();
        assert!(code.contains("ControlOp :: SetP"));
        assert!(code.contains("CmpOp :: Lt"));
    }

    #[test]
    fn lower_let_registers_local() {
        let mut ctx = LoweringContext::new();
        ctx.locals.insert(
            "a".to_string(),
            (Ident::new("_kaio_r0", Span::call_site()), KernelType::F32),
        );
        ctx.locals.insert(
            "b".to_string(),
            (Ident::new("_kaio_r1", Span::call_site()), KernelType::F32),
        );

        let stmt = KernelStmt::Let {
            name: "x".to_string(),
            ty: None,
            value: KernelExpr::BinOp {
                op: BinOpKind::Add,
                lhs: Box::new(KernelExpr::Var("a".to_string(), Span::call_site())),
                rhs: Box::new(KernelExpr::Var("b".to_string(), Span::call_site())),
                span: Span::call_site(),
            },
            span: Span::call_site(),
        };
        let tokens = lower_stmt(&mut ctx, &stmt).unwrap();

        // "x" should now be in locals
        assert!(ctx.locals.contains_key("x"));
        let (reg, ty) = &ctx.locals["x"];
        assert_eq!(ty, &KernelType::F32);
        assert!(reg.to_string().starts_with("_kaio_r"));

        // Should have generated ArithOp::Add
        let code = tokens.to_string();
        assert!(code.contains("ArithOp :: Add"));
    }

    #[test]
    fn lower_if_simple() {
        let mut ctx = LoweringContext::new();
        ctx.locals.insert(
            "idx".to_string(),
            (Ident::new("_kaio_r0", Span::call_site()), KernelType::U32),
        );
        ctx.locals.insert(
            "n".to_string(),
            (Ident::new("_kaio_r1", Span::call_site()), KernelType::U32),
        );

        // if idx < n { (bare expression for now) }
        let stmt = KernelStmt::If {
            condition: KernelExpr::BinOp {
                op: BinOpKind::Lt,
                lhs: Box::new(KernelExpr::Var("idx".to_string(), Span::call_site())),
                rhs: Box::new(KernelExpr::Var("n".to_string(), Span::call_site())),
                span: Span::call_site(),
            },
            then_body: vec![],
            else_body: None,
            span: Span::call_site(),
        };
        let tokens = lower_stmt(&mut ctx, &stmt).unwrap();
        let code = tokens.to_string();

        // Should contain: SetP, BraPred with negate: true, Label
        assert!(code.contains("SetP"));
        assert!(code.contains("negate : true"));
        assert!(code.contains("IF_END_0"));
        assert!(code.contains("Label"));
    }

    #[test]
    fn lower_if_else() {
        let mut ctx = LoweringContext::new();
        ctx.locals.insert(
            "x".to_string(),
            (Ident::new("_kaio_r0", Span::call_site()), KernelType::U32),
        );
        ctx.locals.insert(
            "n".to_string(),
            (Ident::new("_kaio_r1", Span::call_site()), KernelType::U32),
        );

        let stmt = KernelStmt::If {
            condition: KernelExpr::BinOp {
                op: BinOpKind::Lt,
                lhs: Box::new(KernelExpr::Var("x".to_string(), Span::call_site())),
                rhs: Box::new(KernelExpr::Var("n".to_string(), Span::call_site())),
                span: Span::call_site(),
            },
            then_body: vec![],
            else_body: Some(vec![]),
            span: Span::call_site(),
        };
        let tokens = lower_stmt(&mut ctx, &stmt).unwrap();
        let code = tokens.to_string();

        // If/else should have: BraPred -> IF_ELSE, Bra -> IF_END, IF_ELSE label, IF_END label
        // Label allocation order: IF_END first (0), IF_ELSE second (1)
        assert!(code.contains("negate : true"));
        assert!(code.contains("IF_ELSE_1"));
        assert!(code.contains("IF_END_0"));
        // Unconditional branch to skip else
        assert!(code.contains("ControlOp :: Bra"));
    }

    #[test]
    fn fresh_labels_are_unique() {
        let mut ctx = LoweringContext::new();
        let l1 = ctx.fresh_label("IF_END");
        let l2 = ctx.fresh_label("IF_ELSE");
        let l3 = ctx.fresh_label("IF_END");
        assert_eq!(l1, "IF_END_0");
        assert_eq!(l2, "IF_ELSE_1");
        assert_eq!(l3, "IF_END_2");
    }

    // --- Sprint 2.4: Array Indexing ---

    #[test]
    fn lower_expr_index_read() {
        let mut ctx = LoweringContext::new();
        // Simulate a pointer param "a" loaded as SliceRef(F32)
        ctx.locals.insert(
            "a".to_string(),
            (
                Ident::new("_kaio_r0", Span::call_site()),
                KernelType::SliceRef(Box::new(KernelType::F32)),
            ),
        );
        ctx.locals.insert(
            "idx".to_string(),
            (Ident::new("_kaio_r1", Span::call_site()), KernelType::U32),
        );

        let expr = KernelExpr::Index {
            array: "a".to_string(),
            index: Box::new(KernelExpr::Var("idx".to_string(), Span::call_site())),
            span: Span::call_site(),
        };
        let (result, ty, tokens) = lower_expr(&mut ctx, &expr).unwrap();

        assert_eq!(ty, KernelType::F32); // result is the element type
        assert!(result.to_string().starts_with("_kaio_r"));
        let code = tokens.to_string();
        assert!(code.contains("CvtaToGlobal"));
        assert!(code.contains("MulWide"));
        assert!(code.contains("LdGlobal"));
    }

    #[test]
    fn lower_stmt_index_assign() {
        let mut ctx = LoweringContext::new();
        ctx.locals.insert(
            "out".to_string(),
            (
                Ident::new("_kaio_r0", Span::call_site()),
                KernelType::SliceMutRef(Box::new(KernelType::F32)),
            ),
        );
        ctx.locals.insert(
            "idx".to_string(),
            (Ident::new("_kaio_r1", Span::call_site()), KernelType::U32),
        );
        ctx.locals.insert(
            "val".to_string(),
            (Ident::new("_kaio_r2", Span::call_site()), KernelType::F32),
        );

        let stmt = KernelStmt::IndexAssign {
            array: "out".to_string(),
            index: KernelExpr::Var("idx".to_string(), Span::call_site()),
            value: KernelExpr::Var("val".to_string(), Span::call_site()),
            span: Span::call_site(),
        };
        let tokens = lower_stmt(&mut ctx, &stmt).unwrap();
        let code = tokens.to_string();

        assert!(code.contains("CvtaToGlobal"));
        assert!(code.contains("StGlobal"));
        assert!(code.contains("PtxType :: F32"));
    }

    #[test]
    fn reject_write_to_immutable_slice() {
        let mut ctx = LoweringContext::new();
        ctx.locals.insert(
            "a".to_string(),
            (
                Ident::new("_kaio_r0", Span::call_site()),
                KernelType::SliceRef(Box::new(KernelType::F32)), // &[f32], NOT &mut
            ),
        );
        ctx.locals.insert(
            "idx".to_string(),
            (Ident::new("_kaio_r1", Span::call_site()), KernelType::U32),
        );
        ctx.locals.insert(
            "val".to_string(),
            (Ident::new("_kaio_r2", Span::call_site()), KernelType::F32),
        );

        let stmt = KernelStmt::IndexAssign {
            array: "a".to_string(),
            index: KernelExpr::Var("idx".to_string(), Span::call_site()),
            value: KernelExpr::Var("val".to_string(), Span::call_site()),
            span: Span::call_site(),
        };
        let err = lower_stmt(&mut ctx, &stmt).unwrap_err();
        assert!(err.to_string().contains("immutable slice"));
    }

    #[test]
    fn reject_index_into_scalar() {
        let mut ctx = LoweringContext::new();
        ctx.locals.insert(
            "n".to_string(),
            (Ident::new("_kaio_r0", Span::call_site()), KernelType::U32),
        );
        ctx.locals.insert(
            "idx".to_string(),
            (Ident::new("_kaio_r1", Span::call_site()), KernelType::U32),
        );

        let expr = KernelExpr::Index {
            array: "n".to_string(),
            index: Box::new(KernelExpr::Var("idx".to_string(), Span::call_site())),
            span: Span::call_site(),
        };
        let err = lower_expr(&mut ctx, &expr).unwrap_err();
        assert!(err.to_string().contains("not a slice"));
    }

    // --- Sprint 3.1: Assign + Loops ---

    #[test]
    fn lower_assign() {
        let mut ctx = LoweringContext::new();
        let existing_reg = Ident::new("_kaio_r0", Span::call_site());
        ctx.locals
            .insert("x".to_string(), (existing_reg.clone(), KernelType::F32));
        ctx.locals.insert(
            "y".to_string(),
            (Ident::new("_kaio_r1", Span::call_site()), KernelType::F32),
        );

        let stmt = KernelStmt::Assign {
            name: "x".to_string(),
            value: KernelExpr::Var("y".to_string(), Span::call_site()),
            span: Span::call_site(),
        };
        let tokens = lower_stmt(&mut ctx, &stmt).unwrap();
        let code = tokens.to_string();

        // Should emit Mov to existing register
        assert!(code.contains("Mov"));
        assert!(code.contains("Operand :: Reg"));

        // x should still be in locals with the same register
        let (reg, _) = &ctx.locals["x"];
        assert_eq!(reg.to_string(), "_kaio_r0");
    }

    #[test]
    fn lower_assign_undefined_var() {
        let mut ctx = LoweringContext::new();
        let stmt = KernelStmt::Assign {
            name: "nonexistent".to_string(),
            value: KernelExpr::LitInt(0, KernelType::I32, Span::call_site()),
            span: Span::call_site(),
        };
        let err = lower_stmt(&mut ctx, &stmt).unwrap_err();
        assert!(err.to_string().contains("undefined variable"));
    }

    #[test]
    fn lower_for_loop() {
        let mut ctx = LoweringContext::new();
        ctx.locals.insert(
            "n".to_string(),
            (Ident::new("_kaio_r0", Span::call_site()), KernelType::U32),
        );

        let stmt = KernelStmt::For {
            var: "i".to_string(),
            start: KernelExpr::LitInt(0, KernelType::U32, Span::call_site()),
            end: KernelExpr::Var("n".to_string(), Span::call_site()),
            body: vec![],
            span: Span::call_site(),
        };
        let tokens = lower_stmt(&mut ctx, &stmt).unwrap();
        let code = tokens.to_string();

        // Should contain loop structure
        assert!(code.contains("LOOP_START_"));
        assert!(code.contains("LOOP_END_"));
        assert!(code.contains("SetP"));
        assert!(code.contains("CmpOp :: Ge"));
        assert!(code.contains("BraPred"));
        assert!(code.contains("ArithOp :: Add")); // increment
        assert!(code.contains("ControlOp :: Bra")); // back-edge

        // Loop var should be removed from locals after the loop
        assert!(!ctx.locals.contains_key("i"));
    }

    #[test]
    fn lower_for_loop_literal_coercion() {
        let mut ctx = LoweringContext::new();
        ctx.locals.insert(
            "n".to_string(),
            (Ident::new("_kaio_r0", Span::call_site()), KernelType::U32),
        );

        // for i in 0..n — start is unsuffixed LitInt (default I32), end is U32
        // Should coerce start literal to U32
        let stmt = KernelStmt::For {
            var: "i".to_string(),
            start: KernelExpr::LitInt(0, KernelType::I32, Span::call_site()),
            end: KernelExpr::Var("n".to_string(), Span::call_site()),
            body: vec![],
            span: Span::call_site(),
        };
        let result = lower_stmt(&mut ctx, &stmt);
        assert!(
            result.is_ok(),
            "literal coercion should allow 0..n where n: u32"
        );
    }

    #[test]
    fn lower_while_loop() {
        let mut ctx = LoweringContext::new();
        ctx.locals.insert(
            "x".to_string(),
            (Ident::new("_kaio_r0", Span::call_site()), KernelType::U32),
        );

        let stmt = KernelStmt::While {
            condition: KernelExpr::BinOp {
                op: BinOpKind::Gt,
                lhs: Box::new(KernelExpr::Var("x".to_string(), Span::call_site())),
                rhs: Box::new(KernelExpr::LitInt(0, KernelType::U32, Span::call_site())),
                span: Span::call_site(),
            },
            body: vec![],
            span: Span::call_site(),
        };
        let tokens = lower_stmt(&mut ctx, &stmt).unwrap();
        let code = tokens.to_string();

        // Should contain loop structure
        assert!(code.contains("LOOP_START_"));
        assert!(code.contains("LOOP_END_"));
        assert!(code.contains("negate : true")); // @!pred for while exit
        assert!(code.contains("ControlOp :: Bra")); // back-edge
    }

    #[test]
    fn lower_nested_loops_unique_labels() {
        let mut ctx = LoweringContext::new();
        ctx.locals.insert(
            "n".to_string(),
            (Ident::new("_kaio_r0", Span::call_site()), KernelType::U32),
        );
        ctx.locals.insert(
            "m".to_string(),
            (Ident::new("_kaio_r1", Span::call_site()), KernelType::U32),
        );

        let inner = KernelStmt::For {
            var: "j".to_string(),
            start: KernelExpr::LitInt(0, KernelType::U32, Span::call_site()),
            end: KernelExpr::Var("m".to_string(), Span::call_site()),
            body: vec![],
            span: Span::call_site(),
        };
        let outer = KernelStmt::For {
            var: "i".to_string(),
            start: KernelExpr::LitInt(0, KernelType::U32, Span::call_site()),
            end: KernelExpr::Var("n".to_string(), Span::call_site()),
            body: vec![inner],
            span: Span::call_site(),
        };
        let tokens = lower_stmt(&mut ctx, &outer).unwrap();
        let code = tokens.to_string();

        // Should have 4 unique labels (2 per loop)
        assert!(code.contains("LOOP_START_0"));
        assert!(code.contains("LOOP_END_1"));
        assert!(code.contains("LOOP_START_2"));
        assert!(code.contains("LOOP_END_3"));
    }
}
