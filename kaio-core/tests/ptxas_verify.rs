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
