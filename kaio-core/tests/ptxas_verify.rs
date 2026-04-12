//! Belt-and-suspenders verification: emit vector_add PTX and verify it
//! with ptxas (NVIDIA's offline PTX assembler).
//!
//! Soft-skip: if ptxas is not in PATH, the test passes without verification.
//! This means CI without CUDA toolkit will show this test as "passed" even
//! though no verification occurred. The eprintln! output makes this visible
//! when running with --nocapture.
//!
//! Uses `KAIO_SM_TARGET` env var (default `sm_70`) for portability across GPUs.

mod common;

/// SM target for ptxas verification. Reads KAIO_SM_TARGET env var,
/// defaults to sm_70 (Volta+, maximum compatibility).
fn sm_target() -> String {
    std::env::var("KAIO_SM_TARGET").unwrap_or_else(|_| "sm_70".to_string())
}

#[test]
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
fn ptxas_verify_mma_sync() {
    let ptxas_check = std::process::Command::new("ptxas")
        .arg("--version")
        .output();
    if ptxas_check.is_err() {
        eprintln!("NOTE: ptxas not found in PATH — skipping PTX verification");
        return;
    }

    let sm = sm_target_ampere_or_better();
    // SAFETY: test binaries are single-threaded inside the test runner
    // for this scenario — set_var is fine here.
    unsafe { std::env::set_var("KAIO_SM_TARGET", &sm) };
    let ptx = common::build_mma_sync_ptx();

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
fn ptxas_verify_mma_sync_shared() {
    let ptxas_check = std::process::Command::new("ptxas")
        .arg("--version")
        .output();
    if ptxas_check.is_err() {
        eprintln!("NOTE: ptxas not found in PATH — skipping PTX verification");
        return;
    }

    let sm = sm_target_ampere_or_better();
    // SAFETY: test binaries are single-threaded inside the test runner
    // for this scenario — set_var is fine here.
    unsafe { std::env::set_var("KAIO_SM_TARGET", &sm) };
    let ptx = common::build_mma_sync_shared_ptx();

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
fn ptxas_verify_cp_async() {
    let ptxas_check = std::process::Command::new("ptxas")
        .arg("--version")
        .output();
    if ptxas_check.is_err() {
        eprintln!("NOTE: ptxas not found in PATH — skipping PTX verification");
        return;
    }

    let sm = sm_target_ampere_or_better();
    // SAFETY: see comment on ptxas_verify_mma_sync.
    unsafe { std::env::set_var("KAIO_SM_TARGET", &sm) };
    let ptx = common::build_cp_async_ptx();

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
