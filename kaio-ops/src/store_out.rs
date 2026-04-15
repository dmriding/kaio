//! Shared fragment-C → global store helpers.
//!
//! Sprint 7.3 D2. Houses store-out emit helpers that are used by more
//! than one kernel module but do not belong to any single kernel's
//! internals. First inhabitant:
//! [`emit_store_fragment_c_f32_to_f16_packed`] — the store path used by
//! the fused tri-output QKV projection kernels (`qkv_project_int8`
//! W8A16 and `qkv_project_int4` W4A16), both of which accumulate in
//! f32 fragment-C and emit f16 outputs ready to feed
//! [`attention_tc`][crate::attention_tc].
//!
//! # Not an `attention_tc` refactor
//!
//! The original plan scoped D2 as a shared helper promoted out of
//! `attention_tc_kernel.rs`'s store-out path, with a TokenStream
//! snapshot canary proving byte-equivalent PTX emission. Execution
//! surfaced a premise miss: `attention_tc` outputs **f32** directly
//! (its `out_ptr` is `GpuBuffer<f32>`), and its inline store at
//! `attention_tc_kernel.rs:1318+` writes `st.global.f32` per
//! accumulator — no f32→f16 cast, no `MovPack`, no packed b32 store.
//! There is nothing to promote. The helper below ships standalone for
//! `qkv_project` use; if a future op needs an f16 attention output,
//! this helper is ready, but refactoring `attention_tc` itself is a
//! separate decision.

use kaio_core::instr::ArithOp;
use kaio_core::instr::memory::MemoryOp;
use kaio_core::ir::{Operand, PtxInstruction, PtxKernel, Register, RegisterAllocator};
use kaio_core::types::PtxType;

/// Emit fragment-C → packed-f16 global store for one `m16n8k16` output sub-tile.
///
/// # What it emits (per warp lane)
///
/// For each of the two packed f16 pairs in the fragment:
/// 1. (optional) `mul.f32 %tmp, frag_c[i], scale` — applies post-accumulation
///    scalar scale. Skipped when `scale` is `None` (INT4 path folds the scale
///    into fragment B during dequant).
/// 2. `cvt.rn.f16.f32 %h_i, %tmp_or_frag_c_i` — narrow to f16 with
///    round-to-nearest-even.
/// 3. `mov.b32 %packed, {%h_lo, %h_hi}` ([`PtxInstruction::MovPack`][kaio_core::ir::PtxInstruction::MovPack])
///    — pack two adjacent f16 values into one `.b32` register.
/// 4. `st.global.u32 [%addr], %packed` — one coalesced 4-byte store per
///    adjacent column pair. (The `.u32` vs `.b32` suffix is a framework-idiom
///    choice inherited from `matmul_int4`'s store path; both are valid PTX
///    for an untyped 4-byte store, and `ptxas` lowers them identically.)
///
/// # Fragment-C spatial layout (m16n8k16)
///
/// Each lane holds 4 f32 accumulators whose logical positions in the
/// 16×8 output tile are:
///
/// ```text
///   lane.reg[0]  →  (row = group_id,     col = 2*tig    )
///   lane.reg[1]  →  (row = group_id,     col = 2*tig + 1)
///   lane.reg[2]  →  (row = group_id + 8, col = 2*tig    )
///   lane.reg[3]  →  (row = group_id + 8, col = 2*tig + 1)
///
///   where  group_id = lane_id / 4   (range 0..8)
///          tig      = lane_id % 4   (range 0..4)
/// ```
///
/// Regs 0/1 are column-adjacent in the same row, as are regs 2/3. The
/// rows differ by exactly **8**, which is the "r+8 row offset" callers
/// need to understand. This is a property of the `m16n8k16` mma shape
/// — it is **not** projection-specific. The tri-output QKV kernels
/// invoke this helper three times per output sub-tile with only
/// `tile_base_global` (and, for INT8, `scale`) varying across Q/K/V;
/// the within-fragment geometry is identical because all three
/// projections use the same mma shape.
///
/// # Parameters
///
/// - `alloc` / `kernel` — IR building context.
/// - `frag_c` — the 4 f32 accumulators of one lane (pass the
///   `FragmentC::regs` array).
/// - `tile_base_global` — u64 register holding the global-memory
///   address of element `(row=0, col=0)` of **this specific `m16n8k16`
///   output sub-tile**. Callers are responsible for having added the
///   block-row, warp-row, and mma-M/N offsets ahead of this call.
/// - `row_stride_bytes` — u32 register holding the byte stride between
///   adjacent rows of the output (i.e. `N * 2` for f16 row-major
///   output).
/// - `tid_lane` — u32 register holding the warp lane id (0..32).
///   `group_id` and `tig` are derived internally.
/// - `scale` — when `Some(scale_f32_reg)`, each `frag_c[i]` is multiplied
///   by `scale_f32_reg` before the cvt. When `None`, the cvt consumes
///   `frag_c[i]` directly.
///
/// # Pre-conditions
///
/// - Output buffer row stride is **uniform across all three
///   projections** in the qkv kernels — enforced by `validate_dims_qkv_int{4,8}`
///   via `N_q == N_k == N_v`.
/// - Output N must be even — enforced by `validate_dims_qkv_int{4,8}`
///   — since this helper always stores two adjacent f16 columns per
///   `st.global.u32`.
/// - `tid_lane` must be the raw 0..32 lane id (not flat thread id).
/// - No edge-predication in v1: caller must ensure the stored sub-tile
///   is fully in-bounds. Ragged-tile predication is a follow-up if a
///   kernel ever drives it (qkv_project v1 enforces divisibility of M/N
///   by the block tile upstream in launch-config).
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)] // wired up in D3 (INT8) / D5 (INT4)
pub(crate) fn emit_store_fragment_c_f32_to_f16_packed(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    frag_c: &[Register; 4],
    tile_base_global: Register,
    row_stride_bytes: Register,
    tid_lane: Register,
    scale: Option<Register>,
) {
    // --- Address math: compute row0_off and row8_off in u32, then fold into tile_base ---
    let r_group_id = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Div {
        dst: r_group_id,
        lhs: Operand::Reg(tid_lane),
        rhs: Operand::ImmU32(4),
        ty: PtxType::U32,
    }));
    let r_tig = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Rem {
        dst: r_tig,
        lhs: Operand::Reg(tid_lane),
        rhs: Operand::ImmU32(4),
        ty: PtxType::U32,
    }));

    // row0_off_bytes = group_id * row_stride_bytes + tig * 4
    let r_group_row_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_group_row_off,
        lhs: Operand::Reg(r_group_id),
        rhs: Operand::Reg(row_stride_bytes),
        ty: PtxType::U32,
    }));
    let r_tig_col_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_tig_col_off,
        lhs: Operand::Reg(r_tig),
        // 2 f16s per packed pair × 2 bytes per f16 = 4 bytes between adjacent pairs.
        rhs: Operand::ImmU32(4),
        ty: PtxType::U32,
    }));
    let r_row0_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_row0_off,
        lhs: Operand::Reg(r_group_row_off),
        rhs: Operand::Reg(r_tig_col_off),
        ty: PtxType::U32,
    }));

    // row8_off_bytes = row0_off_bytes + 8 * row_stride_bytes
    let r_eight_rows = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Mul {
        dst: r_eight_rows,
        lhs: Operand::Reg(row_stride_bytes),
        rhs: Operand::ImmU32(8),
        ty: PtxType::U32,
    }));
    let r_row8_off = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: r_row8_off,
        lhs: Operand::Reg(r_row0_off),
        rhs: Operand::Reg(r_eight_rows),
        ty: PtxType::U32,
    }));

    // Convert both row offsets to u64 and add tile_base_global.
    let r_row0_addr = fold_u32_offset_into_u64_base(alloc, kernel, tile_base_global, r_row0_off);
    let r_row8_addr = fold_u32_offset_into_u64_base(alloc, kernel, tile_base_global, r_row8_off);

    // --- Per-pair emit: (reg0,reg1) → row0_addr, (reg2,reg3) → row8_addr ---
    emit_packed_pair_store(alloc, kernel, frag_c[0], frag_c[1], r_row0_addr, scale);
    emit_packed_pair_store(alloc, kernel, frag_c[2], frag_c[3], r_row8_addr, scale);
}

/// Fold a u32 byte offset into a u64 base pointer.
///
/// Two-step: `cvt.u64.u32` → `add.u64`. Mirrors the pattern used by
/// `attention_tc_kernel::emit_store_fragment_c_to_global_out`.
fn fold_u32_offset_into_u64_base(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    base_u64: Register,
    off_u32: Register,
) -> Register {
    let rd_off = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Cvt {
        dst: rd_off,
        src: off_u32,
        dst_ty: PtxType::U64,
        src_ty: PtxType::U32,
    });
    let rd_addr = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Arith(ArithOp::Add {
        dst: rd_addr,
        lhs: Operand::Reg(base_u64),
        rhs: Operand::Reg(rd_off),
        ty: PtxType::U64,
    }));
    rd_addr
}

/// Emit the scale (optional) + cvt + MovPack + st.global.u32 chain for
/// one adjacent pair of f32 accumulators.
fn emit_packed_pair_store(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    src_lo: Register,
    src_hi: Register,
    addr: Register,
    scale: Option<Register>,
) {
    let h_lo = scale_and_cvt_to_f16(alloc, kernel, src_lo, scale);
    let h_hi = scale_and_cvt_to_f16(alloc, kernel, src_hi, scale);
    let packed = alloc.alloc_packed_half2();
    kernel.push(PtxInstruction::MovPack {
        dst: packed,
        srcs: vec![h_lo, h_hi],
        ty: PtxType::U32,
    });
    kernel.push(PtxInstruction::Memory(MemoryOp::StGlobal {
        addr,
        src: packed,
        // 4-byte untyped store of the packed f16 pair. Emits
        // `st.global.u32` via the framework's default mapping — same
        // idiom as matmul_int4's packed-b32 store at
        // `matmul_int4_kernel.rs:1162`. `.u32` and `.b32` are
        // equivalent at the PTX layer for storing 4 untyped bytes.
        ty: PtxType::U32,
    }));
}

/// Apply an optional f32 scale to `src`, then `cvt.rn.f16.f32` to an f16.
fn scale_and_cvt_to_f16(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    src: Register,
    scale: Option<Register>,
) -> Register {
    let f_src = if let Some(scale_reg) = scale {
        let scaled = alloc.alloc(PtxType::F32);
        kernel.push(PtxInstruction::Arith(ArithOp::Mul {
            dst: scaled,
            lhs: Operand::Reg(src),
            rhs: Operand::Reg(scale_reg),
            ty: PtxType::F32,
        }));
        scaled
    } else {
        src
    };
    let h = alloc.alloc(PtxType::F16);
    kernel.push(PtxInstruction::Cvt {
        dst: h,
        src: f_src,
        dst_ty: PtxType::F16,
        src_ty: PtxType::F32,
    });
    h
}

/// Build a minimal smoke PTX module exercising
/// [`emit_store_fragment_c_f32_to_f16_packed`] once without scale and
/// once with scale. Purpose: `ptxas_verify` offline gate — confirms
/// the helper's emitted PTX is assembler-valid on real SM targets
/// (sm_80 baseline, sm_89 production). Not a correctness test.
///
/// Kernel signature:
///   - `out_ptr: .u64` (f16 output pointer)
///   - `stride: .u32`  (row stride in bytes)
///   - `scale: .f32`   (scalar multiplier for the scale=Some path)
///
/// Behavior per thread: reads `tid.x`, fabricates 4 f32 fragment-C
/// values via `mov.f32 <imm>`, calls the helper once with `scale=None`
/// and once with `scale=Some`, then `ret`. The two store chains write
/// to the same tile base — the second call overwrites the first,
/// which is fine for an assembler-validity gate.
#[cfg(test)]
pub(crate) fn build_store_out_smoke_ptx(sm: &str) -> String {
    use kaio_core::emit::{Emit, PtxWriter};
    use kaio_core::instr::control::ControlOp;
    use kaio_core::instr::memory::MemoryOp;
    use kaio_core::ir::{PtxModule, PtxParam};

    let mut alloc = RegisterAllocator::new();
    let mut kernel = PtxKernel::new("store_out_smoke");

    kernel.add_param(PtxParam::pointer("out_ptr", PtxType::F16));
    kernel.add_param(PtxParam::scalar("stride", PtxType::U32));
    kernel.add_param(PtxParam::scalar("scale", PtxType::F32));

    // Load params.
    let rd_out_param = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: rd_out_param,
        param_name: "out_ptr".to_string(),
        ty: PtxType::U64,
    }));
    let tile_base = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::CvtaToGlobal {
        dst: tile_base,
        src: rd_out_param,
    }));
    let row_stride = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: row_stride,
        param_name: "stride".to_string(),
        ty: PtxType::U32,
    }));
    let scale = alloc.alloc(PtxType::F32);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: scale,
        param_name: "scale".to_string(),
        ty: PtxType::F32,
    }));

    // Read tid.x into a lane-id register.
    let (tid_lane, tid_instr) = kaio_core::instr::special::tid_x(&mut alloc);
    kernel.push(tid_instr);

    // Fabricate 4 f32 fragment-C accumulators from distinct immediates.
    // Distinct values keep ptxas from folding them into a single
    // constant and erasing the store chain.
    let frag_c: [Register; 4] = core::array::from_fn(|i| {
        let r = alloc.alloc(PtxType::F32);
        kernel.push(PtxInstruction::Mov {
            dst: r,
            src: Operand::ImmF32(1.0 + i as f32),
            ty: PtxType::F32,
        });
        r
    });

    // Pass A: scale = None. Pure cvt + MovPack + st.global.u32 chain.
    emit_store_fragment_c_f32_to_f16_packed(
        &mut alloc,
        &mut kernel,
        &frag_c,
        tile_base,
        row_stride,
        tid_lane,
        None,
    );

    // Pass B: scale = Some. Adds the mul.f32 prefix to each pair emit.
    emit_store_fragment_c_f32_to_f16_packed(
        &mut alloc,
        &mut kernel,
        &frag_c,
        tile_base,
        row_stride,
        tid_lane,
        Some(scale),
    );

    kernel.push(PtxInstruction::Control(ControlOp::Ret));
    kernel.set_registers(alloc.into_allocated());

    let mut module = PtxModule::new(sm);
    module.add_kernel(kernel);
    let mut w = PtxWriter::new();
    module.emit(&mut w).unwrap();
    w.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use kaio_core::emit::{Emit, PtxWriter};

    fn emit_kernel_text(kernel: &PtxKernel) -> String {
        let mut w = PtxWriter::new();
        kernel.emit(&mut w).unwrap();
        w.finish()
    }

    fn fresh_kernel_with_scratch() -> (
        RegisterAllocator,
        PtxKernel,
        [Register; 4],
        Register,
        Register,
        Register,
    ) {
        let mut alloc = RegisterAllocator::new();
        let kernel = PtxKernel::new("store_out_test");
        let frag_c = [
            alloc.alloc(PtxType::F32),
            alloc.alloc(PtxType::F32),
            alloc.alloc(PtxType::F32),
            alloc.alloc(PtxType::F32),
        ];
        let tile_base = alloc.alloc(PtxType::U64);
        let row_stride = alloc.alloc(PtxType::U32);
        let tid_lane = alloc.alloc(PtxType::U32);
        (alloc, kernel, frag_c, tile_base, row_stride, tid_lane)
    }

    #[test]
    fn emit_without_scale_produces_expected_instruction_shape() {
        let (mut alloc, mut kernel, frag_c, tile_base, row_stride, tid_lane) =
            fresh_kernel_with_scratch();
        emit_store_fragment_c_f32_to_f16_packed(
            &mut alloc,
            &mut kernel,
            &frag_c,
            tile_base,
            row_stride,
            tid_lane,
            None,
        );
        let ptx = emit_kernel_text(&kernel);
        // Two adjacent-pair stores → exactly 2 st.global.u32 lines.
        assert_eq!(
            ptx.matches("st.global.u32").count(),
            2,
            "expected 2 packed stores, got:\n{ptx}"
        );
        // Two MovPack packs (one per adjacent pair).
        assert_eq!(
            ptx.matches("mov.b32").count(),
            2,
            "expected 2 mov.b32 packs, got:\n{ptx}"
        );
        // Four cvt.rn.f16.f32 (one per fragment-C register).
        assert_eq!(
            ptx.matches("cvt.rn.f16.f32").count(),
            4,
            "expected 4 cvt.rn.f16.f32, got:\n{ptx}"
        );
        // No scale mul when scale=None.
        assert_eq!(
            ptx.matches("mul.f32").count(),
            0,
            "expected 0 mul.f32 when scale=None, got:\n{ptx}"
        );
    }

    #[test]
    fn emit_with_scale_prepends_four_mul_f32() {
        let (mut alloc, mut kernel, frag_c, tile_base, row_stride, tid_lane) =
            fresh_kernel_with_scratch();
        let scale = alloc.alloc(PtxType::F32);
        emit_store_fragment_c_f32_to_f16_packed(
            &mut alloc,
            &mut kernel,
            &frag_c,
            tile_base,
            row_stride,
            tid_lane,
            Some(scale),
        );
        let ptx = emit_kernel_text(&kernel);
        // One mul.f32 per fragment-C register = 4.
        assert_eq!(
            ptx.matches("mul.f32").count(),
            4,
            "expected 4 mul.f32 when scale=Some, got:\n{ptx}"
        );
        // Still 2 st.global.u32 and 2 mov.b32 packs.
        assert_eq!(ptx.matches("st.global.u32").count(), 2);
        assert_eq!(ptx.matches("mov.b32").count(), 2);
        // Still 4 cvt.rn.f16.f32 (cvt consumes the scaled tmp, not the raw frag).
        assert_eq!(ptx.matches("cvt.rn.f16.f32").count(), 4);
    }

    #[test]
    fn emit_scale_chain_ordering_mul_before_cvt_before_movpack_before_store() {
        // Structural ordering check: for each pair, mul.f32 must appear
        // before cvt.rn.f16.f32, which must appear before mov.b32, which
        // must appear before st.global.u32. We check this by position of
        // the first occurrence of each mnemonic — sufficient for "the
        // pipeline is wired in the right direction."
        let (mut alloc, mut kernel, frag_c, tile_base, row_stride, tid_lane) =
            fresh_kernel_with_scratch();
        let scale = alloc.alloc(PtxType::F32);
        emit_store_fragment_c_f32_to_f16_packed(
            &mut alloc,
            &mut kernel,
            &frag_c,
            tile_base,
            row_stride,
            tid_lane,
            Some(scale),
        );
        let ptx = emit_kernel_text(&kernel);
        let mul_pos = ptx.find("mul.f32").expect("no mul.f32 emitted");
        let cvt_pos = ptx.find("cvt.rn.f16.f32").expect("no cvt emitted");
        let mov_pos = ptx.find("mov.b32").expect("no mov.b32 emitted");
        let st_pos = ptx.find("st.global.u32").expect("no st.global.u32 emitted");
        assert!(
            mul_pos < cvt_pos,
            "mul.f32 must precede cvt.rn.f16.f32\n{ptx}"
        );
        assert!(cvt_pos < mov_pos, "cvt must precede mov.b32\n{ptx}");
        assert!(
            mov_pos < st_pos,
            "mov.b32 must precede st.global.u32\n{ptx}"
        );
    }

    #[test]
    fn address_math_emits_row8_after_row0_with_eight_row_stride_multiplier() {
        // Sanity that we actually compute row8 as row0 + 8*stride and not
        // some other constant. The emitted PTX should contain a `mul.u32
        // ..., ..., 8;` and the row8 address should use the result.
        let (mut alloc, mut kernel, frag_c, tile_base, row_stride, tid_lane) =
            fresh_kernel_with_scratch();
        emit_store_fragment_c_f32_to_f16_packed(
            &mut alloc,
            &mut kernel,
            &frag_c,
            tile_base,
            row_stride,
            tid_lane,
            None,
        );
        let ptx = emit_kernel_text(&kernel);
        assert!(
            ptx.contains(", 8;"),
            "expected address math to contain a multiply-by-8 for the row+8 offset; got:\n{ptx}"
        );
        // And a multiply-by-4 for the tig * (2 f16 × 2 bytes) within-row offset.
        assert!(
            ptx.contains(", 4;"),
            "expected address math to contain a multiply-by-4 for the tig column offset; got:\n{ptx}"
        );
    }

    #[test]
    fn group_id_and_tig_come_from_tid_lane_divmod_4() {
        // The helper derives group_id = tid_lane / 4 and tig = tid_lane % 4.
        // Check that a div.u32 and rem.u32 appear with the tid_lane as
        // divisor's lhs.
        let (mut alloc, mut kernel, frag_c, tile_base, row_stride, tid_lane) =
            fresh_kernel_with_scratch();
        emit_store_fragment_c_f32_to_f16_packed(
            &mut alloc,
            &mut kernel,
            &frag_c,
            tile_base,
            row_stride,
            tid_lane,
            None,
        );
        let ptx = emit_kernel_text(&kernel);
        assert!(
            ptx.contains("div.u32"),
            "expected div.u32 for group_id;\n{ptx}"
        );
        assert!(ptx.contains("rem.u32"), "expected rem.u32 for tig;\n{ptx}");
    }

    /// ptxas_verify offline gate for the store-out smoke module.
    ///
    /// Runs `ptxas --gpu-name <sm>` over the module produced by
    /// [`build_store_out_smoke_ptx`]. `#[ignore]` so host-only CI runs
    /// (no CUDA toolchain) stay green — invoke via
    /// `cargo test -- --ignored` on a machine with `ptxas` in PATH.
    /// Target SM overridable via the `KAIO_SM_TARGET` env var; default
    /// `sm_80` matches the `matmul_int4` / `matmul_int8` compat floor.
    #[test]
    #[ignore]
    fn ptxas_verify_store_out() {
        let ptxas_check = std::process::Command::new("ptxas")
            .arg("--version")
            .output();
        if ptxas_check.is_err() {
            eprintln!("NOTE: ptxas not found in PATH — skipping PTX verification");
            return;
        }

        let sm = std::env::var("KAIO_SM_TARGET").unwrap_or_else(|_| "sm_80".to_string());
        let ptx = build_store_out_smoke_ptx(&sm);
        let tmp = std::env::temp_dir().join("kaio_store_out_verify.ptx");
        std::fs::write(&tmp, &ptx).expect("failed to write temp PTX");

        let output = std::process::Command::new("ptxas")
            .args(["--gpu-name", &sm])
            .arg(tmp.to_str().unwrap())
            .output()
            .expect("failed to run ptxas");
        let _ = std::fs::remove_file(&tmp);

        assert!(
            output.status.success(),
            "ptxas verification FAILED for store_out ({sm}):\n\
             stdout: {}\nstderr: {}\n\n=== PTX ===\n{ptx}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );

        eprintln!("ptxas verification PASSED for store_out ({sm})");
    }
}
