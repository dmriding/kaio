//! Code generation — assemble the final `TokenStream` from lowered fragments.

mod launch_wrapper;
mod ptx_builder;

use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

use crate::kernel_ir::KernelSignature;
use crate::kernel_ir::stmt::KernelStmt;

/// Generate the complete kernel module from a parsed signature and body.
///
/// Produces:
/// ```ignore
/// mod kernel_name {
///     fn build_module(sm: &str) -> PtxModule { ... }
///     pub fn launch(device, params...) -> Result<(), KaioError> { ... }
/// }
/// ```
///
/// Sprint 6.10 D1a: the `PTX_CACHE: OnceLock<String>` cache is removed.
/// Each `launch()` call rebuilds the `PtxModule` fresh, using the device's
/// own compute capability as the SM target. Per-call rebuild cost is
/// microseconds (IR construction, not compilation); if it later becomes
/// hot, a cache-design sprint can re-introduce caching deliberately.
pub fn generate_kernel_module(
    sig: &KernelSignature,
    body: &[KernelStmt],
) -> syn::Result<TokenStream> {
    let mod_name = Ident::new(&sig.name, sig.name_span);

    let build_module_fn = ptx_builder::generate_build_module(sig, body)?;
    let launch_fn = launch_wrapper::generate_launch_fn(sig)?;

    Ok(quote! {
        #[allow(non_snake_case, unused_imports, dead_code)]
        mod #mod_name {
            #build_module_fn
            #launch_fn
        }
    })
}

#[cfg(test)]
mod tests {
    //! Host-level codegen regression tests for the lowering pipeline.
    //!
    //! These tests drive `parse_body` + `generate_kernel_module` end to end
    //! and inspect the emitted `TokenStream` for specific semantic patterns
    //! the lowering must produce. No GPU required.
    //!
    //! Added Sprint 6.10 (D2). Each test has a regression canary comment
    //! documenting the mutation it guards against.

    use super::*;
    use crate::kernel_ir::KernelConfig;
    use crate::parse::body::parse_body;
    use crate::parse::signature::parse_kernel_signature;
    use proc_macro2::Span;
    use quote::quote;
    use syn::ItemFn;

    fn dummy_config(block_size: u32) -> KernelConfig {
        KernelConfig {
            block_size,
            block_size_y: None,
            block_size_span: Span::call_site(),
        }
    }

    fn parse_kernel(tokens: proc_macro2::TokenStream) -> ItemFn {
        syn::parse2(tokens).expect("failed to parse test kernel")
    }

    #[test]
    fn shared_memory_lowering_emits_shared_addr_pattern() {
        // Regression canary: if shared memory lowering ever stops emitting
        //   Operand::SharedAddr("<name>".to_string())
        // as the base-address source for shared_mem![] access, or if it
        // switches to a raw pointer-arithmetic path that bypasses the
        // named-symbol scheme, this test fails.
        //
        // The SharedAddr pattern is load-bearing because shared memory
        // allocations are named .shared symbols in PTX, not anonymous
        // offsets. Losing the named-symbol path breaks cross-instruction
        // referencing and any future debugger/profiler integration.
        let func = parse_kernel(quote! {
            fn shared_kernel(n: u32) {
                let sdata = shared_mem![f32; 256];
                sdata[0] = 0.0f32;
            }
        });
        let sig = parse_kernel_signature(&func, dummy_config(256)).expect("signature should parse");
        let body = parse_body(&func.block).expect("body should parse");

        let module = generate_kernel_module(&sig, &body)
            .expect("codegen should succeed for valid shared_mem kernel");
        let output = module.to_string();

        assert!(
            output.contains("SharedAddr"),
            "expected Operand::SharedAddr(...) in shared-memory lowering output, \
             but did not find it. First 800 chars:\n{}",
            &output[..output.len().min(800)]
        );
    }

    /// Normalize the TokenStream `to_string()` output for snapshot comparison.
    ///
    /// `TokenStream::to_string` emits content with `proc_macro2`-internal
    /// spacing rules — collapsing consecutive whitespace (not just leading /
    /// trailing) lets the snapshot survive trivial formatting noise across
    /// Rust / proc-macro2 versions while still catching structural changes
    /// (register numbers, instruction ordering, allocation order).
    fn normalize_tokens(s: &str) -> String {
        s.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    /// Read-or-write snapshot helper. If `KAIO_UPDATE_SNAPSHOT=1` is set or
    /// the snapshot file does not yet exist, writes `actual` to the path and
    /// returns it unchanged (so the test trivially passes on that run).
    /// Otherwise reads the file and returns the expected-string for the
    /// caller to compare against.
    ///
    /// Sprint 7.1.5 D2.0: used to lock the pre-refactor TokenStream
    /// structure of `lower_block_reduce` before D2 factors out the warp-tree
    /// helper. Scoped as a refactor canary — if it becomes noisy from
    /// unrelated harmless drift in the future, relax into pattern
    /// assertions rather than fighting it forever.
    fn read_or_write_snapshot(path: &str, actual: &str) -> String {
        use std::path::PathBuf;
        let full_path: PathBuf = [env!("CARGO_MANIFEST_DIR"), path].iter().collect();
        let should_update = std::env::var("KAIO_UPDATE_SNAPSHOT").is_ok() || !full_path.exists();
        if should_update {
            if let Some(parent) = full_path.parent() {
                std::fs::create_dir_all(parent).expect("create snapshot dir");
            }
            std::fs::write(&full_path, actual).expect("write snapshot");
            return actual.to_string();
        }
        std::fs::read_to_string(&full_path).expect("read snapshot")
    }

    #[test]
    fn block_reduce_sum_f32_tokens_snapshot() {
        // Sprint 7.1.5 D2.0: pre-refactor canary for the `lower_block_reduce`
        // TokenStream. Captures the full generated `TokenStream` for a
        // minimal `block_reduce_sum(f32) -> f32` kernel. D2's helper
        // extraction must produce byte-identical output (after whitespace
        // normalization) — register allocation order, instruction ordering,
        // and shared-symbol naming all locked.
        //
        // If `KAIO_UPDATE_SNAPSHOT=1` is set, writes the snapshot instead of
        // comparing — use this to regenerate after an intentional change.
        let func = parse_kernel(quote! {
            fn snapshot_reduce(out: &mut [f32], n: u32) {
                let x = 1.0f32;
                let s = block_reduce_sum(x);
                out[0] = s;
            }
        });
        let sig = parse_kernel_signature(&func, dummy_config(256)).expect("signature");
        let body = parse_body(&func.block).expect("body");
        let module = generate_kernel_module(&sig, &body).expect("codegen");
        let actual = normalize_tokens(&module.to_string());
        let expected = normalize_tokens(&read_or_write_snapshot(
            "tests/snapshots/block_reduce_sum_f32.tokens.txt",
            &actual,
        ));
        assert_eq!(
            actual, expected,
            "block_reduce_sum(f32) TokenStream drifted vs snapshot. \
             If intentional, rerun with KAIO_UPDATE_SNAPSHOT=1 to regenerate."
        );
    }

    #[test]
    fn reduction_lowering_uses_named_symbol() {
        // Regression canary: if block_reduce_sum / block_reduce_max lowering
        // ever stops using the literal string "_kaio_reduce_smem" as the
        // shared-memory allocation name (e.g., switches to an anonymous
        // allocation, renames the symbol, or inlines the shared region
        // into a different layout), this test fails.
        //
        // The named symbol is required because the reduction lowering
        // performs multiple load/store operations against the same shared
        // region across warp rounds and cross-warp broadcast. All of those
        // touch the same SharedAddr by name. Losing the stable name breaks
        // the multi-phase reduction in a silent correctness-killing way.
        let func = parse_kernel(quote! {
            fn reduce_kernel(out: &mut [f32], n: u32) {
                let x = 1.0f32;
                let s = block_reduce_sum(x);
                out[0] = s;
            }
        });
        let sig = parse_kernel_signature(&func, dummy_config(256)).expect("signature should parse");
        let body = parse_body(&func.block).expect("body should parse");

        let module = generate_kernel_module(&sig, &body)
            .expect("codegen should succeed for valid reduction kernel");
        let output = module.to_string();

        assert!(
            output.contains("\"_kaio_reduce_smem\""),
            "expected \"_kaio_reduce_smem\" string literal in reduction lowering output, \
             but did not find it. First 800 chars:\n{}",
            &output[..output.len().min(800)]
        );
    }

    #[test]
    fn bitwise_and_lowers_to_arith_and() {
        // Regression canary (Sprint 7.0 D2): if bitwise `&` ever stops dispatching
        // through ArithOp::And (e.g. accidentally routes to ArithOp::Mul or collapses
        // into a logical `&&` path), this test fails. ArithOp::And is the only path
        // that produces `and.b32` / `and.b64` PTX — required for every bitmask
        // operation Phase 7.1+ quant kernels will rely on.
        let func = parse_kernel(quote! {
            fn bitand_kernel(a: u32, b: u32, out: &mut [u32], n: u32) {
                out[0] = a & b;
            }
        });
        let sig = parse_kernel_signature(&func, dummy_config(32)).expect("signature");
        let body = parse_body(&func.block).expect("body");

        let module = generate_kernel_module(&sig, &body).expect("codegen");
        let output = module.to_string();

        assert!(
            output.contains("ArithOp :: And"),
            "expected ArithOp::And in bitwise `&` lowering output, got:\n{}",
            &output[..output.len().min(800)]
        );
    }

    #[test]
    fn shr_signedness_preserved_in_codegen() {
        // Regression canary (Sprint 7.0 AD2): `i32 >> n` must carry PtxType::S32
        // all the way through the macro's emitted TokenStream. If it ever flips
        // to PtxType::U32 silently, quant INT8 dequantization on signed packed
        // values produces wrong weights without a loud error.
        //
        // The emitted TokenStream contains the constructor arguments for
        // ArithOp::Shr — one of those arguments must read `PtxType :: S32`.
        let func = parse_kernel(quote! {
            fn shr_kernel(a: i32, shift: u32, out: &mut [i32], n: u32) {
                out[0] = a >> shift;
            }
        });
        let sig = parse_kernel_signature(&func, dummy_config(32)).expect("signature");
        let body = parse_body(&func.block).expect("body");

        let module = generate_kernel_module(&sig, &body).expect("codegen");
        let output = module.to_string();

        assert!(
            output.contains("ArithOp :: Shr"),
            "expected ArithOp::Shr in `i32 >> n` lowering, got:\n{}",
            &output[..output.len().min(800)]
        );
        // The Shr constructor contains one `ty : PtxType :: S32` among several
        // PtxType tokens (also U32 for params, etc.) — assert S32 appears at all.
        assert!(
            output.contains("PtxType :: S32"),
            "expected PtxType::S32 somewhere in `i32 >> n` codegen \
             (so ArithOp::Shr emits shr.s32 / arithmetic shift), got:\n{}",
            &output[..output.len().min(1200)]
        );
    }

    #[test]
    fn if_condition_with_logical_and_uses_branch_direct() {
        // Regression canary (Sprint 7.0 D4): `if a && b { ... }` must use the
        // branch-direct path, NOT materialize an intermediate p_out register
        // via a Mov { PtxType::Pred, ... }. If the pattern detection in the
        // KernelStmt::If arm ever regresses, this kernel would emit
        //   mov.pred p_out, p_lhs
        //   mov.pred p_out, p_rhs
        // which — while still correct — defeats the point of the direct-branch
        // optimization and indicates the if-condition dispatch is broken.
        //
        // We assert the ABSENCE of `PtxType :: Pred` in a Mov statement
        // emitted by the logical lowering. Other Mov instructions in the
        // kernel (e.g. special-register reads for tid/ctaid) use PtxType::U32,
        // so this assertion is specific to the logical materialization path.
        let func = parse_kernel(quote! {
            fn and_if_kernel(a: u32, b: u32, out: &mut [u32], n: u32) {
                if a < n && b < n {
                    out[0] = 1;
                }
            }
        });
        let sig = parse_kernel_signature(&func, dummy_config(32)).expect("signature");
        let body = parse_body(&func.block).expect("body");

        let module = generate_kernel_module(&sig, &body).expect("codegen");
        let output = module.to_string();

        // Branch-direct path must NOT emit Mov { ty: PtxType::Pred } — that
        // only appears when the expression-position materializer ran.
        assert!(
            !output.contains("ty : PtxType :: Pred"),
            "if-condition `a && b` should use branch-direct form, but found \
             `ty : PtxType :: Pred` (the materialized expression-position \
             short-circuit path). Partial output:\n{}",
            &output[..output.len().min(1500)]
        );
        // Sanity: the short-circuit skip must still emit conditional branches.
        assert!(
            output.contains("ControlOp :: BraPred"),
            "expected at least one ControlOp::BraPred for if-condition `a && b`"
        );
    }

    #[test]
    fn logical_operator_rejects_non_bool_operand() {
        // Regression canary (Sprint 7.0 D4 error path): `&&` / `||` on integer
        // operands must produce a clear compile-time error, not silently accept
        // the kernel and emit wrong PTX. If `ensure_bool` ever stops guarding
        // this, a user writing `if count && flag` (meaning bitwise `&`) would
        // get confusing behavior.
        let func = parse_kernel(quote! {
            fn bad_logical_kernel(a: u32, b: u32, out: &mut [u32], n: u32) {
                // `a && b` is nonsense — both operands are u32, not bool.
                // This must error at codegen time, not silently lower.
                if a && b {
                    out[0] = 1;
                }
            }
        });
        let sig = parse_kernel_signature(&func, dummy_config(32)).expect("signature");
        let body = parse_body(&func.block).expect("body");

        let err = generate_kernel_module(&sig, &body)
            .expect_err("codegen must reject `u32 && u32` — logical ops require bool");
        let msg = err.to_string();
        assert!(
            msg.contains("&&") && msg.contains("bool"),
            "expected error mentioning && and bool, got: {msg}"
        );
    }

    #[test]
    fn if_condition_with_logical_or_uses_branch_direct_take_label() {
        // Regression canary (Sprint 7.0 D4, `||` if-condition path):
        // `if a || b { body }` must emit the branch-direct form with a
        // LOGICAL_OR_TAKE label — NOT materialize an intermediate p_out Mov.
        // The `||` path is structurally distinct from `&&` (short-circuits on
        // LHS true, not LHS false; uses a local TAKE label the body falls
        // through to). Losing either the TAKE label or the branch-direct
        // semantics silently reverts `||` to expression-position materialization.
        let func = parse_kernel(quote! {
            fn or_if_kernel(a: u32, b: u32, out: &mut [u32], n: u32) {
                if a < n || b < n {
                    out[0] = 1;
                }
            }
        });
        let sig = parse_kernel_signature(&func, dummy_config(32)).expect("signature");
        let body = parse_body(&func.block).expect("body");

        let module = generate_kernel_module(&sig, &body).expect("codegen");
        let output = module.to_string();

        assert!(
            output.contains("LOGICAL_OR_TAKE"),
            "expected LOGICAL_OR_TAKE_<n> label from `||` if-condition \
             branch-direct path, got:\n{}",
            &output[..output.len().min(1500)]
        );
        assert!(
            !output.contains("ty : PtxType :: Pred"),
            "if-condition `a || b` should use branch-direct form (no Mov \
             PtxType::Pred), got:\n{}",
            &output[..output.len().min(1500)]
        );
    }

    #[test]
    fn logical_or_in_expression_position_materializes_predicate() {
        // Regression canary (Sprint 7.0 D4 expression-position, `||` variant):
        // `let m = a || b;` must materialize the short-circuit result via the
        // LOGICAL_DONE label + mov.pred sequence, with the `||` short-circuit
        // branching on LHS true (negate=false in BraPred) rather than LHS false.
        let func = parse_kernel(quote! {
            fn or_expr_kernel(a: u32, b: u32, out: &mut [u32], n: u32) {
                let m = a < n || b < n;
                if m {
                    out[0] = 1;
                }
            }
        });
        let sig = parse_kernel_signature(&func, dummy_config(32)).expect("signature");
        let body = parse_body(&func.block).expect("body");

        let module = generate_kernel_module(&sig, &body).expect("codegen");
        let output = module.to_string();

        assert!(
            output.contains("LOGICAL_DONE"),
            "expected LOGICAL_DONE_<n> label from expression-position `||` \
             materialization, got:\n{}",
            &output[..output.len().min(1500)]
        );
        assert!(
            output.contains("ty : PtxType :: Pred"),
            "expected Mov with PtxType::Pred for materialized `||` short-circuit \
             result, got:\n{}",
            &output[..output.len().min(1500)]
        );
    }

    #[test]
    fn logical_and_in_expression_position_materializes_predicate() {
        // Regression canary (Sprint 7.0 D4 expression-position path): `let m = a && b;`
        // must materialize the short-circuit result into a .pred register via
        // the Mov { ty: PtxType::Pred, ... } sequence inside a LOGICAL_DONE label
        // block. If the expression-position path ever silently collapses to a
        // bitwise `and.b32` / `and.pred` (no branch), the RHS would always
        // evaluate — the Rust short-circuit contract would break silently.
        let func = parse_kernel(quote! {
            fn and_expr_kernel(a: u32, b: u32, out: &mut [u32], n: u32) {
                let m = a < n && b < n;
                if m {
                    out[0] = 1;
                }
            }
        });
        let sig = parse_kernel_signature(&func, dummy_config(32)).expect("signature");
        let body = parse_body(&func.block).expect("body");

        let module = generate_kernel_module(&sig, &body).expect("codegen");
        let output = module.to_string();

        // Materialized path must emit a LOGICAL_DONE label (fresh_label("LOGICAL_DONE")).
        assert!(
            output.contains("LOGICAL_DONE"),
            "expected LOGICAL_DONE_<n> label from expression-position `&&` \
             materialization, got:\n{}",
            &output[..output.len().min(1500)]
        );
        // And must emit Mov { ty: PtxType::Pred } for the p_out predicate.
        assert!(
            output.contains("ty : PtxType :: Pred"),
            "expected Mov with PtxType::Pred for materialized short-circuit \
             result, got:\n{}",
            &output[..output.len().min(1500)]
        );
    }
}
