//! Compile-fail tests for `#[gpu_kernel]`.
//!
//! Verifies that invalid kernel code produces clear compile-time error messages.
//! These test the CF1-CF10 cases from `docs/success-criteria.md`.
//!
//! To regenerate `.stderr` files after changing error messages:
//! ```sh
//! TRYBUILD=overwrite cargo test -p pyros compile_fail
//! ```
//!
//! Note: `.stderr` files are platform-specific (path separators differ
//! between Windows and Linux). If adding Linux CI, consider platform-specific
//! stderr files or a looser matching strategy.

#[test]
fn compile_fail() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/*.rs");
}
