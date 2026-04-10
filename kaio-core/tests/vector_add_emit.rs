//! Integration test: construct a complete `vector_add` kernel via the IR API
//! and emit it to a PTX string. Validates that `kaio-core` can produce a
//! complete, structurally correct `.ptx` file from Rust code.
//!
//! This is the Phase 1 milestone test for `kaio-core`.

mod common;

#[test]
fn emit_full_vector_add() {
    let ptx = common::build_vector_add_ptx();

    // --- Validate structure ---
    // Header
    assert!(ptx.starts_with(".version 8.7\n"));
    assert!(ptx.contains(".target sm_89\n"));
    assert!(ptx.contains(".address_size 64\n"));

    // Kernel signature
    assert!(ptx.contains(".visible .entry vector_add("));
    assert!(ptx.contains(".param .u64 a_ptr,"));
    assert!(ptx.contains(".param .u64 b_ptr,"));
    assert!(ptx.contains(".param .u64 c_ptr,"));
    assert!(ptx.contains(".param .u32 n"));

    // Register declarations
    assert!(ptx.contains(".reg .b32 %r<"));
    assert!(ptx.contains(".reg .b64 %rd<"));
    assert!(ptx.contains(".reg .f32 %f<"));
    assert!(ptx.contains(".reg .pred %p<"));

    // Key instructions
    assert!(ptx.contains("ld.param.u64 %rd0, [a_ptr];"));
    assert!(ptx.contains("ld.param.u32 %r0, [n];"));
    assert!(ptx.contains("mov.u32 %r1, %ctaid.x;"));
    assert!(ptx.contains("mov.u32 %r2, %ntid.x;"));
    assert!(ptx.contains("mov.u32 %r3, %tid.x;"));
    assert!(ptx.contains("mad.lo.s32 %r4, %r1, %r2, %r3;"));
    assert!(ptx.contains("setp.ge.u32 %p0, %r4, %r0;"));
    assert!(ptx.contains("@%p0 bra EXIT;"));
    assert!(ptx.contains("cvta.to.global.u64"));
    assert!(ptx.contains("mul.wide.u32"));
    assert!(ptx.contains("add.s64"));
    assert!(ptx.contains("ld.global.f32"));
    assert!(ptx.contains("add.f32"));
    assert!(ptx.contains("st.global.f32"));
    assert!(ptx.contains("EXIT:"));
    assert!(ptx.contains("ret;"));

    // Structure
    assert!(ptx.trim_end().ends_with('}'));

    // Print for manual inspection
    eprintln!("=== KAIO vector_add PTX ===\n{ptx}");
}
