//! Belt-and-suspenders verification: emit PTX and verify it with
//! ptxas (NVIDIA's offline PTX assembler).
//!
//! **All tests are `#[ignore]`** — run them via `cargo test -- --ignored`
//! on a machine with the CUDA toolkit installed. They were previously
//! unmarked and would **soft-skip** (pass without running ptxas) when
//! ptxas wasn't in PATH, which meant default CI on Linux without the
//! CUDA toolkit reported green without doing any actual verification.
//! Marking them `#[ignore]` aligns the gate with Phase 6's reality:
//! Dave's Windows dev box with `--ignored` is the canonical verification
//! surface, not the stock GitHub Actions runner. The soft-skip below
//! stays as a secondary safety for anyone who runs `--ignored` on a
//! machine without ptxas.
//!
//! Uses `KAIO_SM_TARGET` env var (default `sm_70`) for portability across GPUs.

mod common;

/// SM target for ptxas verification. Reads KAIO_SM_TARGET env var,
/// defaults to sm_70 (Volta+, maximum compatibility).
fn sm_target() -> String {
    std::env::var("KAIO_SM_TARGET").unwrap_or_else(|_| "sm_70".to_string())
}

#[test]
#[ignore] // requires CUDA toolkit (ptxas) — run via `cargo test -- --ignored`
fn ptxas_verify_vector_add() {
    // Soft-skip: passes without verification if ptxas not in PATH
    let ptxas_check = std::process::Command::new("ptxas")
        .arg("--version")
        .output();

    if ptxas_check.is_err() {
        eprintln!("NOTE: ptxas not found in PATH — skipping PTX verification");
        eprintln!("      Install CUDA toolkit to enable this test");
        return;
    }

    let sm = sm_target();
    let ptx = common::build_vector_add_ptx();

    // Write PTX to temp file
    let tmp = std::env::temp_dir().join("kaio_vector_add_verify.ptx");
    std::fs::write(&tmp, &ptx).expect("failed to write temp PTX file");

    let output = std::process::Command::new("ptxas")
        .args(["--gpu-name", &sm])
        .arg(tmp.to_str().unwrap())
        .output()
        .expect("failed to run ptxas");

    // Clean up temp file (best effort)
    let _ = std::fs::remove_file(&tmp);

    assert!(
        output.status.success(),
        "ptxas verification FAILED:\nstdout: {}\nstderr: {}\n\n=== PTX ===\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
        ptx
    );

    eprintln!("ptxas verification PASSED for vector_add ({sm})");
}

/// SM target to use for PTX that requires features added in Ampere
/// (mma.sync.m16n8k16, cp.async). Floors `KAIO_SM_TARGET` at `sm_80`
/// — if the user sets `sm_70` (or nothing), we still need `sm_80` for
/// these features to ptxas-verify.
fn sm_target_ampere_or_better() -> String {
    let requested = sm_target();
    match requested
        .strip_prefix("sm_")
        .and_then(|s| s.parse::<u32>().ok())
    {
        Some(v) if v >= 80 => requested,
        _ => "sm_80".to_string(),
    }
}

#[test]
#[ignore] // requires CUDA toolkit (ptxas) — run via `cargo test -- --ignored`
fn ptxas_verify_mma_sync() {
    let ptxas_check = std::process::Command::new("ptxas")
        .arg("--version")
        .output();
    if ptxas_check.is_err() {
        eprintln!("NOTE: ptxas not found in PATH — skipping PTX verification");
        return;
    }

    let sm = sm_target_ampere_or_better();
    // Sprint 6.10 D3: sm is passed directly to the builder instead of
    // mutating KAIO_SM_TARGET. Removes the test-side process-global env
    // mutation that was a hygiene landmine under parallel test runners.
    let ptx = common::build_mma_sync_ptx(&sm);

    let tmp = std::env::temp_dir().join("kaio_mma_sync_verify.ptx");
    std::fs::write(&tmp, &ptx).expect("failed to write temp PTX file");

    let output = std::process::Command::new("ptxas")
        .args(["--gpu-name", &sm])
        .arg(tmp.to_str().unwrap())
        .output()
        .expect("failed to run ptxas");

    let _ = std::fs::remove_file(&tmp);

    assert!(
        output.status.success(),
        "ptxas verification FAILED for mma.sync ({sm}):\nstdout: {}\nstderr: {}\n\n=== PTX ===\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
        ptx
    );

    eprintln!("ptxas verification PASSED for mma.sync.m16n8k16.f16.f32 ({sm})");
}

#[test]
#[ignore] // requires CUDA toolkit (ptxas) — run via `cargo test -- --ignored`
fn ptxas_verify_mma_sync_shared() {
    let ptxas_check = std::process::Command::new("ptxas")
        .arg("--version")
        .output();
    if ptxas_check.is_err() {
        eprintln!("NOTE: ptxas not found in PATH — skipping PTX verification");
        return;
    }

    let sm = sm_target_ampere_or_better();
    // Sprint 6.10 D3: sm passed directly to builder; no env mutation.
    let ptx = common::build_mma_sync_shared_ptx(&sm);

    let tmp = std::env::temp_dir().join("kaio_mma_sync_shared_verify.ptx");
    std::fs::write(&tmp, &ptx).expect("failed to write temp PTX file");

    let output = std::process::Command::new("ptxas")
        .args(["--gpu-name", &sm])
        .arg(tmp.to_str().unwrap())
        .output()
        .expect("failed to run ptxas");

    let _ = std::fs::remove_file(&tmp);

    assert!(
        output.status.success(),
        "ptxas verification FAILED for mma.sync shared-source ({sm}):\nstdout: {}\nstderr: {}\n\n=== PTX ===\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
        ptx
    );

    eprintln!("ptxas verification PASSED for shared-source m16n8k16 fragment loaders ({sm})");
}

#[test]
#[ignore] // requires CUDA toolkit (ptxas) — run via `cargo test -- --ignored`
fn ptxas_verify_cp_async() {
    let ptxas_check = std::process::Command::new("ptxas")
        .arg("--version")
        .output();
    if ptxas_check.is_err() {
        eprintln!("NOTE: ptxas not found in PATH — skipping PTX verification");
        return;
    }

    let sm = sm_target_ampere_or_better();
    // Sprint 6.10 D3: sm passed directly to builder; no env mutation.
    let ptx = common::build_cp_async_ptx(&sm);

    let tmp = std::env::temp_dir().join("kaio_cp_async_verify.ptx");
    std::fs::write(&tmp, &ptx).expect("failed to write temp PTX file");

    let output = std::process::Command::new("ptxas")
        .args(["--gpu-name", &sm])
        .arg(tmp.to_str().unwrap())
        .output()
        .expect("failed to run ptxas");

    let _ = std::fs::remove_file(&tmp);

    assert!(
        output.status.success(),
        "ptxas verification FAILED for cp.async ({sm}):\nstdout: {}\nstderr: {}\n\n=== PTX ===\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
        ptx
    );

    eprintln!("ptxas verification PASSED for cp.async ({sm})");
}

#[test]
#[ignore] // requires CUDA toolkit (ptxas) — run via `cargo test -- --ignored`
fn ptxas_verify_ld_global_b128() {
    // Sprint 6.7b Gate A: verify the new LdGlobalB128 IR primitive emits
    // PTX that ptxas accepts. LDG.128 is not Ampere-gated — this uses the
    // base SM target (no forced sm_80).
    let ptxas_check = std::process::Command::new("ptxas")
        .arg("--version")
        .output();
    if ptxas_check.is_err() {
        eprintln!("NOTE: ptxas not found in PATH — skipping PTX verification");
        return;
    }

    let sm = sm_target();
    let ptx = common::build_ld_global_b128_ptx();

    let tmp = std::env::temp_dir().join("kaio_ld_global_b128_verify.ptx");
    std::fs::write(&tmp, &ptx).expect("failed to write temp PTX file");

    let output = std::process::Command::new("ptxas")
        .args(["--gpu-name", &sm])
        .arg(tmp.to_str().unwrap())
        .output()
        .expect("failed to run ptxas");

    let _ = std::fs::remove_file(&tmp);

    assert!(
        output.status.success(),
        "ptxas verification FAILED for ld.global.v4.b32 ({sm}):\nstdout: {}\nstderr: {}\n\n=== PTX ===\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
        ptx
    );

    eprintln!("ptxas verification PASSED for ld.global.v4.b32 ({sm})");
}

#[test]
#[ignore] // requires CUDA toolkit (ptxas) — run via `cargo test -- --ignored`
fn ptxas_verify_bitops() {
    // Sprint 7.0 D1: verify the 6 new bitwise ArithOp variants
    // (And / Or / Xor / Shl / Shr-signed / Shr-unsigned / Not) emit PTX
    // that ptxas accepts. Bitops are universally supported — no SM gate.
    let ptxas_check = std::process::Command::new("ptxas")
        .arg("--version")
        .output();
    if ptxas_check.is_err() {
        eprintln!("NOTE: ptxas not found in PATH — skipping PTX verification");
        return;
    }

    let sm = sm_target();
    let ptx = common::build_bitops_ptx(&sm);

    let tmp = std::env::temp_dir().join("kaio_bitops_verify.ptx");
    std::fs::write(&tmp, &ptx).expect("failed to write temp PTX file");

    let output = std::process::Command::new("ptxas")
        .args(["--gpu-name", &sm])
        .arg(tmp.to_str().unwrap())
        .output()
        .expect("failed to run ptxas");

    let _ = std::fs::remove_file(&tmp);

    assert!(
        output.status.success(),
        "ptxas verification FAILED for bitops ({sm}):\nstdout: {}\nstderr: {}\n\n=== PTX ===\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
        ptx
    );

    eprintln!("ptxas verification PASSED for bitops ({sm})");
}

#[test]
#[ignore] // requires CUDA toolkit (ptxas) — run via `cargo test -- --ignored`
fn ptxas_verify_mma_int8() {
    // Sprint 7.1 D1a pass gate: verify that
    // `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` passes the offline
    // assembler for SM 8.0+. This is the fork-decision gate for the sprint:
    // if ptxas accepts the emission, Path FAST (direct s8 mma) is viable;
    // if it rejects, D1 pivots to the DEQUANT-F16 fallback path.
    //
    // Note: ptxas accepting the instruction string only proves encoding
    // viability (D1a). Operand-layout correctness (D1b) is tested separately
    // via GPU round-trip against a CPU reference.
    let ptxas_check = std::process::Command::new("ptxas")
        .arg("--version")
        .output();
    if ptxas_check.is_err() {
        eprintln!("NOTE: ptxas not found in PATH — skipping PTX verification");
        return;
    }

    let sm = sm_target_ampere_or_better();
    let ptx = common::build_mma_int8_ptx(&sm);

    let tmp = std::env::temp_dir().join("kaio_mma_int8_verify.ptx");
    std::fs::write(&tmp, &ptx).expect("failed to write temp PTX file");

    let output = std::process::Command::new("ptxas")
        .args(["--gpu-name", &sm])
        .arg(tmp.to_str().unwrap())
        .output()
        .expect("failed to run ptxas");

    let _ = std::fs::remove_file(&tmp);

    assert!(
        output.status.success(),
        "ptxas verification FAILED for mma.sync.m16n8k32.s8.s8.s32 ({sm}):\nstdout: {}\nstderr: {}\n\n=== PTX ===\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
        ptx
    );

    eprintln!("ptxas verification PASSED for mma.sync.m16n8k32.s8.s8.s32 ({sm})");
}

#[test]
#[ignore] // requires CUDA toolkit (ptxas) — run via `cargo test -- --ignored`
fn ptxas_verify_mma_int8_shared() {
    // Sprint 7.1 D2: confirm that the shared-source INT8 fragment load
    // helpers (`load_fragment_{a,b}_m16n8k32_shared_{row,col}`) emit PTX
    // that ptxas accepts at sm_80+. Correctness (bit-exact vs CPU) for
    // the global-source path was already proved by the D1b adversarial
    // test matrix; this test is instruction-level-syntax only, same
    // pattern as the existing `ptxas_verify_mma_sync_shared`.
    let ptxas_check = std::process::Command::new("ptxas")
        .arg("--version")
        .output();
    if ptxas_check.is_err() {
        eprintln!("NOTE: ptxas not found in PATH — skipping PTX verification");
        return;
    }

    let sm = sm_target_ampere_or_better();
    let ptx = common::build_mma_int8_shared_ptx(&sm);

    let tmp = std::env::temp_dir().join("kaio_mma_int8_shared_verify.ptx");
    std::fs::write(&tmp, &ptx).expect("failed to write temp PTX file");

    let output = std::process::Command::new("ptxas")
        .args(["--gpu-name", &sm])
        .arg(tmp.to_str().unwrap())
        .output()
        .expect("failed to run ptxas");

    let _ = std::fs::remove_file(&tmp);

    assert!(
        output.status.success(),
        "ptxas verification FAILED for mma.sync.m16n8k32.s8 shared-source ({sm}):\nstdout: {}\nstderr: {}\n\n=== PTX ===\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
        ptx
    );

    eprintln!("ptxas verification PASSED for shared-source m16n8k32.s8 fragment loaders ({sm})");
}

#[test]
#[ignore] // requires CUDA toolkit (ptxas) — run via `cargo test -- --ignored`
fn ptxas_verify_shared_mem() {
    let ptxas_check = std::process::Command::new("ptxas")
        .arg("--version")
        .output();

    if ptxas_check.is_err() {
        eprintln!("NOTE: ptxas not found in PATH — skipping PTX verification");
        return;
    }

    let sm = sm_target();
    let ptx = common::build_shared_mem_ptx();

    let tmp = std::env::temp_dir().join("kaio_shared_mem_verify.ptx");
    std::fs::write(&tmp, &ptx).expect("failed to write temp PTX file");

    let output = std::process::Command::new("ptxas")
        .args(["--gpu-name", &sm])
        .arg(tmp.to_str().unwrap())
        .output()
        .expect("failed to run ptxas");

    let _ = std::fs::remove_file(&tmp);

    assert!(
        output.status.success(),
        "ptxas verification FAILED:\nstdout: {}\nstderr: {}\n\n=== PTX ===\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
        ptx
    );

    eprintln!("ptxas verification PASSED for shared_mem_test ({sm})");
}
