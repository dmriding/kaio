//! Memory PTX operations.
//!
//! Contains load/store instructions for global and shared memory:
//! [`LdParam`](MemoryOp::LdParam), [`LdGlobal`](MemoryOp::LdGlobal),
//! [`StGlobal`](MemoryOp::StGlobal), [`LdShared`](MemoryOp::LdShared),
//! [`StShared`](MemoryOp::StShared), and
//! [`CvtaToGlobal`](MemoryOp::CvtaToGlobal).

use std::fmt;

use crate::emit::{Emit, PtxWriter};
use crate::ir::Register;
use crate::types::PtxType;

/// Memory PTX instruction variants.
///
/// Operand conventions:
/// - All addresses and values are [`Register`]s (not [`Operand`](crate::ir::Operand)).
///   You can't `ld.global` from an immediate address or `st.global` an immediate
///   value in PTX — those go through `mov` first.
/// - [`LdParam`](Self::LdParam) is the exception: it references a kernel parameter
///   by name (a `String`), not by register.
#[derive(Debug, Clone)]
pub enum MemoryOp {
    /// Load kernel parameter: `ld.param{ty} dst, [param_name];`
    ///
    /// References the parameter by name from the kernel signature.
    /// Example: `ld.param.u64 %rd1, [vector_add_param_0];`
    LdParam {
        /// Destination register.
        dst: Register,
        /// Parameter name from the kernel signature.
        param_name: String,
        /// PTX type of the parameter value.
        ty: PtxType,
    },
    /// Load from global memory: `ld.global{ty} dst, [addr];`
    ///
    /// The `addr` register holds the computed memory address.
    /// Example: `ld.global.f32 %f1, [%rd8];`
    LdGlobal {
        /// Destination register.
        dst: Register,
        /// Register holding the memory address.
        addr: Register,
        /// PTX type of the loaded value.
        ty: PtxType,
    },
    /// Predicated load from global memory: `@[!]{pred} ld.global{ty} dst, [addr];`
    ///
    /// Skips the load when the predicate evaluates false (or true when
    /// `negate` is set). Used for edge-tile bounds checking — the OOB
    /// thread's `dst` register is left unchanged, so callers typically
    /// pre-initialize `dst` to zero with `mov.b32 dst, 0` and then
    /// conditionally overwrite with a predicated load.
    ///
    /// Sprint 6.7 (multi-warp matmul_tc edge tiles) is the first user.
    /// Example: `@%p1 ld.global.u32 %r5, [%rd9];`
    LdGlobalPred {
        /// Destination register (unchanged when predicate is false).
        dst: Register,
        /// Register holding the memory address.
        addr: Register,
        /// PTX type of the loaded value.
        ty: PtxType,
        /// Predicate register controlling the load.
        pred: Register,
        /// When `true`, negate the predicate (`@!pred`).
        negate: bool,
    },
    /// 128-bit vectorized load from global memory:
    /// `ld.global.v4.b32 {%r_i, %r_j, %r_k, %r_l}, [addr];`
    ///
    /// Single-instruction 128-bit transfer into 4 independent b32
    /// destination registers. Halves (or more) the global-load
    /// instruction count vs scalar b32 loads for bandwidth-bound
    /// kernels. Requires the `addr` register to hold a **16-byte
    /// aligned global-space address** — unaligned access will fault
    /// at runtime; PTX does not catch this statically.
    ///
    /// Destinations are NOT required to be consecutive registers in
    /// the allocator — PTX `ld.global.v4.b32` accepts any 4 b32 regs
    /// in the vector brace list. In practice, allocating 4 regs in
    /// sequence produces consecutive indices, which is what callers
    /// typically do.
    ///
    /// No predicate variant in Sprint 6.7b — edge tiles stay on the
    /// existing [`LdGlobalPred`](Self::LdGlobalPred) scalar path. A
    /// future `LdGlobalB128Pred` would be additive.
    ///
    /// Sprint 6.7b (multi-warp matmul_tc Tile B fast path) is the
    /// first user. Construct via
    /// [`MemoryOp::new_ld_global_b128`](Self::new_ld_global_b128),
    /// which validates that all 4 destinations are b32-class registers.
    ///
    /// Example: `ld.global.v4.b32 {%r0, %r1, %r2, %r3}, [%rd8];`
    LdGlobalB128 {
        /// Four b32 destination registers — receive bytes 0-3, 4-7,
        /// 8-11, 12-15 of the loaded 128-bit value respectively.
        dsts: [Register; 4],
        /// Register holding a 16-B aligned global-space address.
        addr: Register,
    },
    /// Store to global memory: `st.global{ty} [addr], src;`
    ///
    /// **Operand order is reversed in PTX** — address comes first,
    /// value second. This matches PTX convention but is opposite to
    /// loads and arithmetic where `dst` is first.
    ///
    /// Example: `st.global.f32 [%rd10], %f3;`
    StGlobal {
        /// Register holding the memory address.
        addr: Register,
        /// Source register (value to store).
        src: Register,
        /// PTX type of the stored value.
        ty: PtxType,
    },
    /// Predicated store to global memory: `@[!]{pred} st.global{ty} [addr], src;`
    ///
    /// Skips the store when the predicate evaluates false (or true when
    /// `negate` is set). Used for edge-tile bounds checking on output
    /// writes — out-of-bounds threads simply don't store, leaving the
    /// destination memory untouched.
    ///
    /// Sprint 6.7 (multi-warp matmul_tc edge tiles) is the first user.
    /// Example: `@%p1 st.global.f32 [%rd11], %f4;`
    StGlobalPred {
        /// Register holding the memory address.
        addr: Register,
        /// Source register (value to store).
        src: Register,
        /// PTX type of the stored value.
        ty: PtxType,
        /// Predicate register controlling the store.
        pred: Register,
        /// When `true`, negate the predicate (`@!pred`).
        negate: bool,
    },
    /// Load from shared memory: `ld.shared{ty} dst, [addr];`
    ///
    /// Shared memory is block-scoped SRAM. The `addr` register holds the
    /// offset into the declared shared allocation.
    /// Example: `ld.shared.f32 %f0, [%r0];`
    LdShared {
        /// Destination register.
        dst: Register,
        /// Register holding the shared memory offset.
        addr: Register,
        /// PTX type of the loaded value.
        ty: PtxType,
    },
    /// Store to shared memory: `st.shared{ty} [addr], src;`
    ///
    /// **Operand order is reversed in PTX** — address first, value second
    /// (same convention as [`StGlobal`](Self::StGlobal)).
    /// Example: `st.shared.f32 [%r0], %f1;`
    StShared {
        /// Register holding the shared memory offset.
        addr: Register,
        /// Source register (value to store).
        src: Register,
        /// PTX type of the stored value.
        ty: PtxType,
    },
    /// Convert generic address to global: `cvta.to.global.u64 dst, src;`
    ///
    /// Always `.u64` (64-bit address space, matching `.address_size 64`).
    /// Required because `ld.param` returns generic-space pointers —
    /// `ld.global` needs global-space addresses.
    CvtaToGlobal {
        /// Destination register (global-space address).
        dst: Register,
        /// Source register (generic-space address from `ld.param`).
        src: Register,
    },
    /// Asynchronous global→shared copy, cache-at-all-levels variant:
    /// `cp.async.ca.shared.global [dst_shared], [src_global], size_bytes;`
    ///
    /// Issues a non-blocking transfer from global memory into shared
    /// memory without tying up registers. The copy is in-flight after
    /// this instruction; use [`CpAsyncCommitGroup`](Self::CpAsyncCommitGroup)
    /// to delimit a batch and [`CpAsyncWaitGroup`](Self::CpAsyncWaitGroup)
    /// to synchronize. Requires **SM 8.0+ (Ampere)**.
    ///
    /// `size_bytes` must be one of 4, 8, or 16 (validated at construction
    /// via [`MemoryOp::new_cp_async_ca`](Self::new_cp_async_ca)).
    ///
    /// Example: `cp.async.ca.shared.global [%r0], [%rd3], 16;`
    ///
    /// *Placement note:* cp.async lives in `MemoryOp` for Sprint 6.2
    /// because semantically it is a memory op. The commit/wait variants
    /// are pipeline-state operations and may relocate to a dedicated
    /// `PipelineOp` category in Sprint 6.4 once double-buffering patterns
    /// exercise the state machine.
    CpAsyncCaSharedGlobal {
        /// Register holding the shared-memory destination offset.
        dst_shared: Register,
        /// Register holding the global-memory source address (`.to.global`).
        src_global: Register,
        /// Copy size in bytes: must be 4, 8, or 16.
        size_bytes: u8,
    },
    /// Commit all pending `cp.async` operations into a new async group:
    /// `cp.async.commit_group;`
    ///
    /// Groups are numbered implicitly from 0 (most-recently committed)
    /// upward. Used in conjunction with
    /// [`CpAsyncWaitGroup`](Self::CpAsyncWaitGroup) to block until a
    /// specific group completes. Requires **SM 8.0+**.
    CpAsyncCommitGroup,
    /// Wait until at most `n` async copy groups remain in-flight:
    /// `cp.async.wait_group n;`
    ///
    /// `wait_group 0` waits for all outstanding groups to complete
    /// (the common one-stage-pipeline case). For double-buffered
    /// kernels, `wait_group 1` is used to block on the N-1'th group
    /// while issuing the N'th. Requires **SM 8.0+**.
    CpAsyncWaitGroup {
        /// Number of outstanding groups still permitted after this wait.
        n: u8,
    },
}

impl MemoryOp {
    /// Construct a [`CpAsyncCaSharedGlobal`](Self::CpAsyncCaSharedGlobal),
    /// validating the size byte count.
    ///
    /// # Panics
    ///
    /// Panics if `size_bytes` is not one of `4`, `8`, or `16` — the
    /// only sizes PTX accepts for `cp.async.ca`. PTX won't catch this
    /// until ptxas runs, and the error there is cryptic, so we fail
    /// loudly at construction time.
    pub fn new_cp_async_ca(dst_shared: Register, src_global: Register, size_bytes: u8) -> Self {
        assert!(
            matches!(size_bytes, 4 | 8 | 16),
            "cp.async.ca size must be 4, 8, or 16 bytes (got {size_bytes})"
        );
        Self::CpAsyncCaSharedGlobal {
            dst_shared,
            src_global,
            size_bytes,
        }
    }

    /// Construct an [`LdGlobalB128`](Self::LdGlobalB128), validating
    /// that all 4 destinations are b32-class registers.
    ///
    /// # Panics
    ///
    /// Panics if any destination register is not [`crate::types::RegKind::R`] (b32).
    /// `ld.global.v4.b32` requires 4× 32-bit-wide integer-class
    /// destinations; `.f` / `.rd` / `.h` / `.hb` / `.p` registers are
    /// invalid and ptxas's error message is cryptic. Fail loudly at
    /// construction.
    pub fn new_ld_global_b128(dsts: [Register; 4], addr: Register) -> Self {
        use crate::types::RegKind;
        for (i, d) in dsts.iter().enumerate() {
            assert!(
                d.kind == RegKind::R,
                "ld.global.v4.b32 destination {i} must be a b32 register (RegKind::R); got {:?}",
                d.kind
            );
        }
        Self::LdGlobalB128 { dsts, addr }
    }
}

impl Emit for MemoryOp {
    fn emit(&self, w: &mut PtxWriter) -> fmt::Result {
        match self {
            MemoryOp::LdParam {
                dst,
                param_name,
                ty,
            } => {
                let mnemonic = format!("ld.param{}", ty.ptx_memory_suffix());
                let addr = format!("[{param_name}]");
                w.instruction(&mnemonic, &[dst as &dyn fmt::Display, &addr])
            }
            MemoryOp::LdGlobal { dst, addr, ty } => {
                let mnemonic = format!("ld.global{}", ty.ptx_memory_suffix());
                let addr_str = format!("[{addr}]");
                w.instruction(&mnemonic, &[dst as &dyn fmt::Display, &addr_str])
            }
            MemoryOp::LdGlobalPred {
                dst,
                addr,
                ty,
                pred,
                negate,
            } => {
                let neg = if *negate { "!" } else { "" };
                w.line(&format!(
                    "@{neg}{pred} ld.global{} {dst}, [{addr}];",
                    ty.ptx_memory_suffix()
                ))
            }
            MemoryOp::LdGlobalB128 { dsts, addr } => {
                // ld.global.v4.b32 {d0, d1, d2, d3}, [addr];
                w.line(&format!(
                    "ld.global.v4.b32 {{{}, {}, {}, {}}}, [{addr}];",
                    dsts[0], dsts[1], dsts[2], dsts[3]
                ))
            }
            MemoryOp::StGlobal { addr, src, ty } => {
                let mnemonic = format!("st.global{}", ty.ptx_memory_suffix());
                let addr_str = format!("[{addr}]");
                // PTX store order: [address], source (reversed from load)
                w.instruction(&mnemonic, &[&addr_str as &dyn fmt::Display, src])
            }
            MemoryOp::StGlobalPred {
                addr,
                src,
                ty,
                pred,
                negate,
            } => {
                let neg = if *negate { "!" } else { "" };
                w.line(&format!(
                    "@{neg}{pred} st.global{} [{addr}], {src};",
                    ty.ptx_memory_suffix()
                ))
            }
            MemoryOp::LdShared { dst, addr, ty } => {
                let mnemonic = format!("ld.shared{}", ty.ptx_memory_suffix());
                let addr_str = format!("[{addr}]");
                w.instruction(&mnemonic, &[dst as &dyn fmt::Display, &addr_str])
            }
            MemoryOp::StShared { addr, src, ty } => {
                let mnemonic = format!("st.shared{}", ty.ptx_memory_suffix());
                let addr_str = format!("[{addr}]");
                w.instruction(&mnemonic, &[&addr_str as &dyn fmt::Display, src])
            }
            MemoryOp::CvtaToGlobal { dst, src } => {
                w.instruction("cvta.to.global.u64", &[dst as &dyn fmt::Display, src])
            }
            MemoryOp::CpAsyncCaSharedGlobal {
                dst_shared,
                src_global,
                size_bytes,
            } => {
                // cp.async.ca.shared.global [dst_shared], [src_global], size;
                let dst_str = format!("[{dst_shared}]");
                let src_str = format!("[{src_global}]");
                let sz = *size_bytes as u32;
                w.instruction(
                    "cp.async.ca.shared.global",
                    &[&dst_str as &dyn fmt::Display, &src_str, &sz],
                )
            }
            MemoryOp::CpAsyncCommitGroup => w.instruction("cp.async.commit_group", &[]),
            MemoryOp::CpAsyncWaitGroup { n } => {
                let n = *n as u32;
                w.instruction("cp.async.wait_group", &[&n as &dyn fmt::Display])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::RegKind;

    /// Helper to make a register without going through the allocator.
    fn reg(kind: RegKind, index: u32, ptx_type: PtxType) -> Register {
        Register {
            kind,
            index,
            ptx_type,
        }
    }

    // --- nvcc golden comparisons (byte-for-byte match against nvcc --ptx -arch=sm_89) ---

    #[test]
    fn emit_ld_param_u64() {
        // nvcc line 28: ld.param.u64 %rd1, [vector_add_param_0]
        let mut w = PtxWriter::new();
        w.indent();
        let op = MemoryOp::LdParam {
            dst: reg(RegKind::Rd, 1, PtxType::U64),
            param_name: "vector_add_param_0".to_string(),
            ty: PtxType::U64,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    ld.param.u64 %rd1, [vector_add_param_0];\n");
    }

    #[test]
    fn emit_ld_param_u32() {
        // nvcc line 31: ld.param.u32 %r2, [vector_add_param_3]
        let mut w = PtxWriter::new();
        w.indent();
        let op = MemoryOp::LdParam {
            dst: reg(RegKind::R, 2, PtxType::U32),
            param_name: "vector_add_param_3".to_string(),
            ty: PtxType::U32,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    ld.param.u32 %r2, [vector_add_param_3];\n");
    }

    #[test]
    fn emit_cvta_to_global() {
        // nvcc line 39: cvta.to.global.u64 %rd4, %rd1
        let mut w = PtxWriter::new();
        w.indent();
        let op = MemoryOp::CvtaToGlobal {
            dst: reg(RegKind::Rd, 4, PtxType::U64),
            src: reg(RegKind::Rd, 1, PtxType::U64),
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    cvta.to.global.u64 %rd4, %rd1;\n");
    }

    #[test]
    fn emit_ld_global_f32() {
        // nvcc line 44: ld.global.f32 %f1, [%rd8]
        let mut w = PtxWriter::new();
        w.indent();
        let op = MemoryOp::LdGlobal {
            dst: reg(RegKind::F, 1, PtxType::F32),
            addr: reg(RegKind::Rd, 8, PtxType::U64),
            ty: PtxType::F32,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    ld.global.f32 %f1, [%rd8];\n");
    }

    #[test]
    fn emit_ld_global_pred_b32() {
        // Sprint 6.7 edge-tile predicated load.
        let mut w = PtxWriter::new();
        w.indent();
        let op = MemoryOp::LdGlobalPred {
            dst: reg(RegKind::R, 5, PtxType::U32),
            addr: reg(RegKind::Rd, 9, PtxType::U64),
            ty: PtxType::U32,
            pred: reg(RegKind::P, 1, PtxType::Pred),
            negate: false,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    @%p1 ld.global.u32 %r5, [%rd9];\n");
    }

    #[test]
    fn emit_ld_global_pred_negated_b32() {
        let mut w = PtxWriter::new();
        w.indent();
        let op = MemoryOp::LdGlobalPred {
            dst: reg(RegKind::R, 5, PtxType::U32),
            addr: reg(RegKind::Rd, 9, PtxType::U64),
            ty: PtxType::U32,
            pred: reg(RegKind::P, 2, PtxType::Pred),
            negate: true,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    @!%p2 ld.global.u32 %r5, [%rd9];\n");
    }

    #[test]
    fn emit_ld_global_b128() {
        // Sprint 6.7b vectorized load — 128-bit global load into 4 b32 regs.
        let mut w = PtxWriter::new();
        w.indent();
        let op = MemoryOp::new_ld_global_b128(
            [
                reg(RegKind::R, 0, PtxType::U32),
                reg(RegKind::R, 1, PtxType::U32),
                reg(RegKind::R, 2, PtxType::U32),
                reg(RegKind::R, 3, PtxType::U32),
            ],
            reg(RegKind::Rd, 8, PtxType::U64),
        );
        op.emit(&mut w).unwrap();
        assert_eq!(
            w.finish(),
            "    ld.global.v4.b32 {%r0, %r1, %r2, %r3}, [%rd8];\n"
        );
    }

    #[test]
    fn emit_ld_global_b128_non_consecutive_regs() {
        // PTX accepts any 4 b32 regs in the brace list — not required
        // to be consecutive. Validates that emit just writes what it's given.
        let mut w = PtxWriter::new();
        w.indent();
        let op = MemoryOp::new_ld_global_b128(
            [
                reg(RegKind::R, 5, PtxType::U32),
                reg(RegKind::R, 9, PtxType::U32),
                reg(RegKind::R, 2, PtxType::U32),
                reg(RegKind::R, 14, PtxType::U32),
            ],
            reg(RegKind::Rd, 3, PtxType::U64),
        );
        op.emit(&mut w).unwrap();
        assert_eq!(
            w.finish(),
            "    ld.global.v4.b32 {%r5, %r9, %r2, %r14}, [%rd3];\n"
        );
    }

    #[test]
    fn ld_global_b128_emits_single_instruction() {
        // D1 promise: LDG.128 emits ONE PTX instruction, not four.
        let mut w = PtxWriter::new();
        w.indent();
        let op = MemoryOp::new_ld_global_b128(
            [
                reg(RegKind::R, 0, PtxType::U32),
                reg(RegKind::R, 1, PtxType::U32),
                reg(RegKind::R, 2, PtxType::U32),
                reg(RegKind::R, 3, PtxType::U32),
            ],
            reg(RegKind::Rd, 0, PtxType::U64),
        );
        op.emit(&mut w).unwrap();
        let out = w.finish();
        // Exactly one `ld.global.v4.b32` + exactly one newline = one line.
        assert_eq!(out.matches("ld.global").count(), 1);
        assert_eq!(out.matches('\n').count(), 1);
    }

    #[test]
    #[should_panic(expected = "ld.global.v4.b32 destination 0 must be a b32 register")]
    fn ld_global_b128_rejects_f32_destination() {
        MemoryOp::new_ld_global_b128(
            [
                reg(RegKind::F, 0, PtxType::F32), // wrong kind
                reg(RegKind::R, 1, PtxType::U32),
                reg(RegKind::R, 2, PtxType::U32),
                reg(RegKind::R, 3, PtxType::U32),
            ],
            reg(RegKind::Rd, 0, PtxType::U64),
        );
    }

    #[test]
    #[should_panic(expected = "ld.global.v4.b32 destination 2 must be a b32 register")]
    fn ld_global_b128_rejects_h_destination() {
        MemoryOp::new_ld_global_b128(
            [
                reg(RegKind::R, 0, PtxType::U32),
                reg(RegKind::R, 1, PtxType::U32),
                reg(RegKind::H, 0, PtxType::F16), // wrong kind (fp16)
                reg(RegKind::R, 3, PtxType::U32),
            ],
            reg(RegKind::Rd, 0, PtxType::U64),
        );
    }

    #[test]
    fn ld_global_b128_via_ptx_instruction() {
        use crate::ir::PtxInstruction;
        let mut w = PtxWriter::new();
        w.indent();
        let instr = PtxInstruction::Memory(MemoryOp::new_ld_global_b128(
            [
                reg(RegKind::R, 0, PtxType::U32),
                reg(RegKind::R, 1, PtxType::U32),
                reg(RegKind::R, 2, PtxType::U32),
                reg(RegKind::R, 3, PtxType::U32),
            ],
            reg(RegKind::Rd, 5, PtxType::U64),
        ));
        instr.emit(&mut w).unwrap();
        assert_eq!(
            w.finish(),
            "    ld.global.v4.b32 {%r0, %r1, %r2, %r3}, [%rd5];\n"
        );
    }

    #[test]
    fn emit_st_global_pred_f32() {
        // Sprint 6.7 edge-tile predicated store.
        let mut w = PtxWriter::new();
        w.indent();
        let op = MemoryOp::StGlobalPred {
            addr: reg(RegKind::Rd, 11, PtxType::U64),
            src: reg(RegKind::F, 4, PtxType::F32),
            ty: PtxType::F32,
            pred: reg(RegKind::P, 3, PtxType::Pred),
            negate: false,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    @%p3 st.global.f32 [%rd11], %f4;\n");
    }

    #[test]
    fn emit_st_global_f32() {
        // nvcc line 49: st.global.f32 [%rd10], %f3
        let mut w = PtxWriter::new();
        w.indent();
        let op = MemoryOp::StGlobal {
            addr: reg(RegKind::Rd, 10, PtxType::U64),
            src: reg(RegKind::F, 3, PtxType::F32),
            ty: PtxType::F32,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    st.global.f32 [%rd10], %f3;\n");
    }

    // --- Dispatch and ordering validation ---

    #[test]
    fn memory_via_ptx_instruction() {
        use crate::ir::PtxInstruction;

        let mut w = PtxWriter::new();
        w.indent();
        let instr = PtxInstruction::Memory(MemoryOp::LdGlobal {
            dst: reg(RegKind::F, 0, PtxType::F32),
            addr: reg(RegKind::Rd, 0, PtxType::U64),
            ty: PtxType::F32,
        });
        instr.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    ld.global.f32 %f0, [%rd0];\n");
    }

    // --- Shared memory ops ---

    #[test]
    fn emit_ld_shared_f32() {
        let mut w = PtxWriter::new();
        w.indent();
        let op = MemoryOp::LdShared {
            dst: reg(RegKind::F, 0, PtxType::F32),
            addr: reg(RegKind::R, 0, PtxType::U32),
            ty: PtxType::F32,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    ld.shared.f32 %f0, [%r0];\n");
    }

    #[test]
    fn emit_st_shared_f32() {
        let mut w = PtxWriter::new();
        w.indent();
        let op = MemoryOp::StShared {
            addr: reg(RegKind::R, 0, PtxType::U32),
            src: reg(RegKind::F, 1, PtxType::F32),
            ty: PtxType::F32,
        };
        op.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    st.shared.f32 [%r0], %f1;\n");
    }

    // --- Half-precision load/store (Sprint 6.1) ---

    #[test]
    fn emit_ld_global_f16() {
        let mut w = PtxWriter::new();
        w.indent();
        let op = MemoryOp::LdGlobal {
            dst: reg(RegKind::H, 0, PtxType::F16),
            addr: reg(RegKind::Rd, 0, PtxType::U64),
            ty: PtxType::F16,
        };
        op.emit(&mut w).unwrap();
        // PTX ISA §8.7.9: `ld`'s valid type set excludes f16/bf16 — must
        // use `.b16` for 16-bit loads into `.f16` registers.
        assert_eq!(w.finish(), "    ld.global.b16 %h0, [%rd0];\n");
    }

    #[test]
    fn emit_st_global_f16() {
        let mut w = PtxWriter::new();
        w.indent();
        let op = MemoryOp::StGlobal {
            addr: reg(RegKind::Rd, 0, PtxType::U64),
            src: reg(RegKind::H, 0, PtxType::F16),
            ty: PtxType::F16,
        };
        op.emit(&mut w).unwrap();
        // See ld.global counterpart — `.b16` is the memory-op form.
        assert_eq!(w.finish(), "    st.global.b16 [%rd0], %h0;\n");
    }

    #[test]
    fn emit_ld_shared_bf16() {
        let mut w = PtxWriter::new();
        w.indent();
        let op = MemoryOp::LdShared {
            dst: reg(RegKind::Hb, 0, PtxType::BF16),
            addr: reg(RegKind::R, 0, PtxType::U32),
            ty: PtxType::BF16,
        };
        op.emit(&mut w).unwrap();
        // Both f16 and bf16 use `.b16` in memory ops per PTX ISA.
        assert_eq!(w.finish(), "    ld.shared.b16 %hb0, [%r0];\n");
    }

    // --- cp.async (Sprint 6.2) ---

    #[test]
    fn emit_cp_async_ca_shared_global_16b() {
        let mut w = PtxWriter::new();
        w.indent();
        let op = MemoryOp::new_cp_async_ca(
            reg(RegKind::R, 0, PtxType::U32),  // shared offset
            reg(RegKind::Rd, 3, PtxType::U64), // global addr
            16,
        );
        op.emit(&mut w).unwrap();
        assert_eq!(
            w.finish(),
            "    cp.async.ca.shared.global [%r0], [%rd3], 16;\n"
        );
    }

    #[test]
    fn emit_cp_async_ca_size_4() {
        let mut w = PtxWriter::new();
        w.indent();
        let op = MemoryOp::new_cp_async_ca(
            reg(RegKind::R, 1, PtxType::U32),
            reg(RegKind::Rd, 4, PtxType::U64),
            4,
        );
        op.emit(&mut w).unwrap();
        assert_eq!(
            w.finish(),
            "    cp.async.ca.shared.global [%r1], [%rd4], 4;\n"
        );
    }

    #[test]
    fn emit_cp_async_ca_size_8() {
        let mut w = PtxWriter::new();
        w.indent();
        let op = MemoryOp::new_cp_async_ca(
            reg(RegKind::R, 2, PtxType::U32),
            reg(RegKind::Rd, 5, PtxType::U64),
            8,
        );
        op.emit(&mut w).unwrap();
        assert!(w.finish().ends_with("8;\n"));
    }

    #[test]
    #[should_panic(expected = "cp.async.ca size must be 4, 8, or 16 bytes")]
    fn cp_async_ca_rejects_bad_size() {
        // 12 is not a valid size — construction should panic.
        MemoryOp::new_cp_async_ca(
            reg(RegKind::R, 0, PtxType::U32),
            reg(RegKind::Rd, 0, PtxType::U64),
            12,
        );
    }

    #[test]
    fn emit_cp_async_commit_group() {
        let mut w = PtxWriter::new();
        w.indent();
        MemoryOp::CpAsyncCommitGroup.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    cp.async.commit_group;\n");
    }

    #[test]
    fn emit_cp_async_wait_group_zero() {
        let mut w = PtxWriter::new();
        w.indent();
        MemoryOp::CpAsyncWaitGroup { n: 0 }.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    cp.async.wait_group 0;\n");
    }

    #[test]
    fn emit_cp_async_wait_group_n() {
        let mut w = PtxWriter::new();
        w.indent();
        MemoryOp::CpAsyncWaitGroup { n: 3 }.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    cp.async.wait_group 3;\n");
    }

    #[test]
    fn cp_async_via_ptx_instruction() {
        use crate::ir::PtxInstruction;
        let mut w = PtxWriter::new();
        w.indent();
        let instr = PtxInstruction::Memory(MemoryOp::CpAsyncCommitGroup);
        instr.emit(&mut w).unwrap();
        assert_eq!(w.finish(), "    cp.async.commit_group;\n");
    }

    #[test]
    fn st_global_operand_order() {
        // Verify store has [addr], src order — NOT src, [addr]
        let mut w = PtxWriter::new();
        w.indent();
        let op = MemoryOp::StGlobal {
            addr: reg(RegKind::Rd, 0, PtxType::U64),
            src: reg(RegKind::F, 0, PtxType::F32),
            ty: PtxType::F32,
        };
        op.emit(&mut w).unwrap();
        let output = w.finish();
        // [%rd0] must appear BEFORE %f0
        let addr_pos = output.find("[%rd0]").expect("address not found");
        let src_pos = output.find("%f0").expect("source not found");
        assert!(
            addr_pos < src_pos,
            "store operand order wrong: address must come before source in PTX"
        );
    }
}
