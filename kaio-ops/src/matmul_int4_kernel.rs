//! INT4 symmetric dequantize-matmul — `mma.sync.m16n8k16.f16.f16.f32`
//! with fused unpack + sign-extend + cvt chain feeding fragment B.
//!
//! Sprint 7.2. Shipping in stages across D1–D8; this file currently
//! carries the D2 helper (`emit_unpack_s4_x8_scale_to_f16x8`) plus its
//! triple-layer sign-extend canary. D4 will add the full K-tile loop
//! + warp quadrant assembly.
//!
//! # DEQUANT-F16 path (why)
//!
//! No native `m16n8k?.s4.s4.s32` shape on sm_80+ — INT4 operands must
//! be dequantized before the mma. Group-scale quantization (one f16
//! scale per group of 128 K-elements, varying along K) additionally
//! forces pre-mma scale application, ruling out the "reuse INT8
//! kernel, apply post-accumulation scalar" shortcut that worked for
//! sprint 7.1's single-scalar INT8. Therefore:
//!
//! ```text
//! u32 (8 packed s4)
//!   └─ shl.b32 %tmp, %packed, (28 - 4i)    // sign bit of nibble i at MSB
//!   └─ shr.s32 %ext, %tmp, 28              // arithmetic shift → signed i32 in [-8, +7]
//!   └─ cvt.rn.f32.s32 %f, %ext
//!   └─ cvt.rn.f16.f32 %h_raw, %f
//!   └─ mul.f16 %h, %h_raw, %scale          // scale fold, pre-mma
//!   └─ mov.b32 %b32, {%h_lo, %h_hi}        // pack for fragment B feed
//! ```
//!
//! Sign-extend is the correctness-critical step. `shr.s32` (arithmetic
//! right shift) drags the sign bit across the upper 28 bits; an
//! accidental `shr.u32` would zero-extend, silently turning every
//! negative INT4 weight into a positive one. Triple-layer canary:
//! emit-level token assertion (this file), ptxas_verify offline,
//! GPU e2e boundary-value tests (D6).

use kaio_core::instr::ArithOp;
use kaio_core::ir::{Operand, PtxInstruction, PtxKernel, Register, RegisterAllocator};
use kaio_core::types::PtxType;

/// Emit the unpack + sign-extend + dequant + pack chain for one `u32`
/// of packed signed INT4 weights, producing 4 `.b32` registers each
/// holding two packed f16 values ready for `mma.sync.m16n8k16.f16`
/// fragment B.
///
/// # Contract
///
/// **All 8 nibbles in `packed` are assumed to share the same
/// `scale_f16` value** — the caller is responsible for grouping `u32`
/// loads to group-scale boundaries. Under the Sprint 7.2 `AD3` + D3
/// layout (group_size = 128, K_tile_shared = 16), every `u32` carves
/// 8 contiguous K-elements entirely within one group, so this holds
/// by construction. A future sprint that relaxes group_size would
/// extend this signature, not rewrite it.
///
/// # PTX semantics
///
/// For each nibble `i ∈ 0..8`:
///
/// ```text
/// shl.b32 %tmp_i, %packed, (28 - 4*i);     // nibble-i MSB → bit 31
/// shr.s32 %ext_i, %tmp_i, 28;              // arithmetic: sign-extend → signed i32
/// cvt.rn.f32.s32 %f_i, %ext_i;             // int → f32
/// cvt.rn.f16.f32 %h_raw_i, %f_i;           // f32 → f16
/// mul.f16 %h_i, %h_raw_i, %scale_f16;      // scale fold
/// ```
///
/// The 8 dequanted f16 values are then packed low-high into 4 `.b32`
/// registers:
///
/// ```text
/// mov.b32 %b32_0, {%h_0, %h_1};
/// mov.b32 %b32_1, {%h_2, %h_3};
/// mov.b32 %b32_2, {%h_4, %h_5};
/// mov.b32 %b32_3, {%h_6, %h_7};
/// ```
///
/// # Parameters
///
/// - `alloc` — register allocator to draw intermediates from.
/// - `kernel` — kernel body to push instructions into.
/// - `packed` — `.b32` register holding 8 signed-INT4 nibbles, lane `i`
///   at bits `[4i..4i+4)` per the Sprint 7.2 packing convention.
/// - `scale_f16` — `.f16` register holding the group scale for all 8
///   nibbles.
///
/// # Returns
///
/// Four `.b32` registers suitable for direct use in a
/// `FragmentB_M16N8K16` feed (positions `[pair_01, pair_23, pair_45,
/// pair_67]`).
#[allow(dead_code)] // wired up in D4
pub(crate) fn emit_unpack_s4_x8_scale_to_f16x8(
    alloc: &mut RegisterAllocator,
    kernel: &mut PtxKernel,
    packed: Register,
    scale_f16: Register,
) -> [Register; 4] {
    let mut h_out: [Option<Register>; 8] = Default::default();

    for i in 0..8u32 {
        // shl.b32 %tmp, %packed, (28 - 4*i);
        let tmp = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::Arith(ArithOp::Shl {
            dst: tmp,
            lhs: Operand::Reg(packed),
            rhs: Operand::ImmU32(28 - 4 * i),
            ty: PtxType::U32,
        }));
        // shr.s32 %ext, %tmp, 28;   // ARITHMETIC — sign-extends nibble
        let ext = alloc.alloc(PtxType::S32);
        kernel.push(PtxInstruction::Arith(ArithOp::Shr {
            dst: ext,
            lhs: Operand::Reg(tmp),
            rhs: Operand::ImmU32(28),
            ty: PtxType::S32,
        }));
        // cvt.rn.f32.s32 %f, %ext;
        let f32_reg = alloc.alloc(PtxType::F32);
        kernel.push(PtxInstruction::Cvt {
            dst: f32_reg,
            src: ext,
            dst_ty: PtxType::F32,
            src_ty: PtxType::S32,
        });
        // cvt.rn.f16.f32 %h_raw, %f;
        let h_raw = alloc.alloc(PtxType::F16);
        kernel.push(PtxInstruction::Cvt {
            dst: h_raw,
            src: f32_reg,
            dst_ty: PtxType::F16,
            src_ty: PtxType::F32,
        });
        // mul.f16 %h, %h_raw, %scale_f16;
        let h = alloc.alloc(PtxType::F16);
        kernel.push(PtxInstruction::Arith(ArithOp::Mul {
            dst: h,
            lhs: Operand::Reg(h_raw),
            rhs: Operand::Reg(scale_f16),
            ty: PtxType::F16,
        }));
        h_out[i as usize] = Some(h);
    }

    // Pack 8 f16 values into 4 .b32 pairs via the vector-pack mov form.
    let mut pairs = [Register {
        kind: kaio_core::types::RegKind::R,
        index: 0,
        ptx_type: PtxType::U32,
    }; 4];
    for p in 0..4usize {
        let lo = h_out[2 * p].expect("all 8 h_out entries populated above");
        let hi = h_out[2 * p + 1].expect("all 8 h_out entries populated above");
        let b32 = alloc.alloc(PtxType::U32);
        kernel.push(PtxInstruction::MovPack {
            dst: b32,
            srcs: vec![lo, hi],
            ty: PtxType::U32,
        });
        pairs[p] = b32;
    }
    pairs
}

/// Build a minimal kernel that exercises [`emit_unpack_s4_x8_scale_to_f16x8`]
/// once and stores the 4 packed `.b32` results to global memory.
///
/// Used by the ptxas_verify and emit-token canary tests in this
/// module. The kernel accepts:
/// - `packed: u32` scalar param — the packed nibbles to unpack
/// - `scale_bits: u16` scalar param — the f16 scale bit pattern
/// - `out: *mut u32` — 4 u32 output (packed f16 pairs)
///
/// Only two of the four params are wired (packed + out) — scale is
/// materialized via a trivial cvt to keep the emit minimal; this is
/// a structural canary, not a runtime kernel.
#[cfg(test)]
pub(crate) fn build_unpack_s4_smoke_ptx(sm: &str) -> String {
    use kaio_core::emit::{Emit, PtxWriter};
    use kaio_core::instr::control::ControlOp;
    use kaio_core::instr::memory::MemoryOp;
    use kaio_core::ir::{PtxModule, PtxParam};

    let mut alloc = RegisterAllocator::new();
    let mut kernel = PtxKernel::new("unpack_s4_smoke");

    kernel.add_param(PtxParam::scalar("packed", PtxType::U32));
    kernel.add_param(PtxParam::pointer("out", PtxType::U32));

    // Load params.
    let r_packed = alloc.alloc(PtxType::U32);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: r_packed,
        param_name: "packed".to_string(),
        ty: PtxType::U32,
    }));
    let rd_out_param = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::LdParam {
        dst: rd_out_param,
        param_name: "out".to_string(),
        ty: PtxType::U64,
    }));
    let rd_out_global = alloc.alloc(PtxType::U64);
    kernel.push(PtxInstruction::Memory(MemoryOp::CvtaToGlobal {
        dst: rd_out_global,
        src: rd_out_param,
    }));

    // Materialize a scale of +1.0_f16 (0x3C00) via cvt from an immediate-loaded f32.
    let f_one = alloc.alloc(PtxType::F32);
    kernel.push(PtxInstruction::Mov {
        dst: f_one,
        src: Operand::ImmF32(1.0),
        ty: PtxType::F32,
    });
    let h_scale = alloc.alloc(PtxType::F16);
    kernel.push(PtxInstruction::Cvt {
        dst: h_scale,
        src: f_one,
        dst_ty: PtxType::F16,
        src_ty: PtxType::F32,
    });

    // Touch tid.x so the kernel isn't completely thread-invariant
    // (ptxas optimizes harder when it can prove uniform execution).
    let (_r_tid, tid_instr) = kaio_core::instr::special::tid_x(&mut alloc);
    kernel.push(tid_instr);

    // The actual unpack chain.
    let pairs = emit_unpack_s4_x8_scale_to_f16x8(&mut alloc, &mut kernel, r_packed, h_scale);

    // Store the 4 packed b32 results to global out[0..4].
    for (i, reg) in pairs.iter().enumerate() {
        let rd_addr = alloc.alloc(PtxType::U64);
        kernel.push(PtxInstruction::Arith(ArithOp::Add {
            dst: rd_addr,
            lhs: Operand::Reg(rd_out_global),
            rhs: Operand::ImmU32(4 * i as u32),
            ty: PtxType::U64,
        }));
        kernel.push(PtxInstruction::Memory(MemoryOp::StGlobal {
            addr: rd_addr,
            src: *reg,
            ty: PtxType::U32,
        }));
    }

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

    /// Sign-extend emit canary (D2, R1 mitigation — IR / token layer).
    ///
    /// The most dangerous silent-correctness bug in INT4 dequant is a
    /// `shr.u32` where `shr.s32` is required — logical right shift
    /// zero-extends, so every negative INT4 weight becomes positive
    /// with no loud failure. This test asserts the emitted PTX contains
    /// exactly 8 `shr.s32` instructions (one per nibble) and zero
    /// `shr.u32` shifts of the packed word.
    #[test]
    fn unpack_s4_sign_extend_uses_arithmetic_shift() {
        let ptx = build_unpack_s4_smoke_ptx("sm_80");

        let shr_s32_count = ptx.matches("shr.s32").count();
        let shr_u32_count = ptx.matches("shr.u32").count();

        assert_eq!(
            shr_s32_count, 8,
            "expected exactly 8 `shr.s32` (sign-extend per nibble); found {shr_s32_count}\n\n\
             === emitted PTX ===\n{ptx}"
        );
        assert_eq!(
            shr_u32_count, 0,
            "expected zero `shr.u32` in the unpack chain (sign-extend requires arithmetic \
             right shift); found {shr_u32_count}\n\n=== emitted PTX ===\n{ptx}"
        );
    }

    /// Sign-extend emit canary — shift-count coverage (D2, R1 mitigation).
    ///
    /// Asserts the 8 `shl.b32` lane alignments cover the full set
    /// {0, 4, 8, 12, 16, 20, 24, 28} — one per nibble position. A bug
    /// that emitted the same shift count at two sites would leave a
    /// nibble unread (silent correctness failure at one lane) and be
    /// caught here.
    #[test]
    fn unpack_s4_shl_covers_all_nibble_positions() {
        let ptx = build_unpack_s4_smoke_ptx("sm_80");

        for shift in [0u32, 4, 8, 12, 16, 20, 24, 28] {
            let pattern = format!("shl.b32 %r{{any}}, %r{{any}}, {shift};");
            // Simpler exact-text search: `, {shift};`. Each shift value
            // appears exactly once as the immediate in a shl.b32.
            let needle = format!(", {shift};");
            let count = ptx
                .lines()
                .filter(|line| line.contains("shl.b32") && line.ends_with(&needle))
                .count();
            assert_eq!(
                count, 1,
                "expected exactly one `shl.b32 ..., {shift};` (nibble-position alignment); \
                 found {count}\n\npattern scanned: {pattern}\n\n=== emitted PTX ===\n{ptx}"
            );
        }
    }

    /// Verify the emitted PTX contains 4 `mov.b32 %dst, {%h_x, %h_y};`
    /// vector-pack instructions — one per output pair.
    #[test]
    fn unpack_s4_emits_four_packed_f16_pairs() {
        let ptx = build_unpack_s4_smoke_ptx("sm_80");

        let pack_count = ptx
            .lines()
            .filter(|line| line.contains("mov.b32") && line.contains('{') && line.contains('}'))
            .count();
        assert_eq!(
            pack_count, 4,
            "expected 4 `mov.b32 %b, {{%h_lo, %h_hi}};` pack instructions; found {pack_count}\n\n\
             === emitted PTX ===\n{ptx}"
        );
    }

    /// ptxas_verify for the D2 unpack helper (R1 mitigation — offline
    /// assembler layer). Runs `ptxas -arch=sm_80` over a minimal kernel
    /// that exercises `emit_unpack_s4_x8_scale_to_f16x8` exactly once
    /// and confirms the full chain assembles cleanly.
    ///
    /// Requires the CUDA toolkit (`ptxas` on PATH). `#[ignore]` so
    /// host-only runs don't fail — invoke via `cargo test -- --ignored`.
    #[test]
    #[ignore]
    fn ptxas_verify_unpack_s4() {
        let ptxas_check = std::process::Command::new("ptxas")
            .arg("--version")
            .output();
        if ptxas_check.is_err() {
            eprintln!("NOTE: ptxas not found in PATH — skipping PTX verification");
            return;
        }

        let sm = std::env::var("KAIO_SM_TARGET").unwrap_or_else(|_| "sm_80".to_string());
        let ptx = build_unpack_s4_smoke_ptx(&sm);
        let tmp = std::env::temp_dir().join("kaio_unpack_s4_verify.ptx");
        std::fs::write(&tmp, &ptx).expect("failed to write temp PTX");

        let output = std::process::Command::new("ptxas")
            .args(["--gpu-name", &sm])
            .arg(tmp.to_str().unwrap())
            .output()
            .expect("failed to run ptxas");
        let _ = std::fs::remove_file(&tmp);

        assert!(
            output.status.success(),
            "ptxas verification FAILED for unpack_s4 ({sm}):\nstdout: {}\nstderr: {}\n\n\
             === PTX ===\n{ptx}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );

        eprintln!("ptxas verification PASSED for unpack_s4 ({sm})");
    }
}
