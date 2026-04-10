# Sprint 2.7 — Type Validation + Compile-Fail Tests

**Status:** Complete
**Commit:** `8dcfa52`

## Context

Validation logic is distributed across parse/ (signature types, attributes)
and lower/ (undefined variables, immutable writes, unknown builtins).
Sprint 2.7 adds trybuild compile-fail tests to verify these produce
correct compiler diagnostics, matching CF1-CF10 from success-criteria.md.

## Decisions

### No separate validate/ module

**Context:** The Phase 2 master plan proposed `validate/mod.rs`,
`validate/type_check.rs`, `validate/diagnostics.rs`. After building
Sprints 2.1-2.6, all validation is already inline where it belongs.

**Decision:** Skip the separate module. The validation is correctly
distributed: signature validation in `parse/signature.rs`, attribute
validation in `parse/attrs.rs`, construct rejection in `parse/body.rs`,
variable/type checks in `lower/mod.rs`, mutability in `lower/mod.rs`,
builtin names in `lower/builtins.rs`. Adding a second pass would walk
the AST twice for no benefit in Phase 2.

### trybuild .stderr regen command documented

**Context:** .stderr files are byte-for-byte comparisons of compiler
output, including paths and line numbers. They break when error messages
change or the Rust toolchain updates.

**Decision:** Document the regen command in the test file:
`TRYBUILD=overwrite cargo test -p pyros compile_fail`. The pinned
toolchain in `rust-toolchain.toml` helps stability. Note that Windows-
generated .stderr files may not match Linux output (path separators).

### CF4 adjusted: unknown function call instead of Vec::new()

**Context:** `Vec::new()` would parse as a two-segment path (`Vec::new`)
and produce "expected a variable name" rather than the intended "heap
allocation" error. The success-criteria CF4/CF5 intent is to test
rejection of non-built-in function calls.

**Decision:** Test with `unknown_function(data, idx)` which produces the
clear "unknown function" error listing all available builtins. This
matches the intent of CF4/CF5 without requiring a special-case detector.

## Scope

**In:** trybuild setup, 10 compile-fail test cases covering all CF1-CF10
from success-criteria.md plus generics.

**Out:** Formal type inference pass (deferred — type checking TODOs remain
in lower/mod.rs for lhs_ty == rhs_ty validation in binary ops).

## Results

Completed as planned. All 10 compile-fail tests generate clear, span-
accurate error messages.

**Files created:** 12
- `pyros/tests/compile_fail.rs` (trybuild runner)
- `pyros/tests/compile_fail/cf01_string_param.rs`
- `pyros/tests/compile_fail/cf02_return_type.rs`
- `pyros/tests/compile_fail/cf03_println.rs`
- `pyros/tests/compile_fail/cf04_unknown_call.rs`
- `pyros/tests/compile_fail/cf05_missing_block_size.rs`
- `pyros/tests/compile_fail/cf06_block_size_not_pow2.rs`
- `pyros/tests/compile_fail/cf07_block_size_too_large.rs`
- `pyros/tests/compile_fail/cf08_lifetime.rs`
- `pyros/tests/compile_fail/cf09_loop.rs`
- `pyros/tests/compile_fail/cf10_generics.rs`
- 10 `.stderr` files (auto-generated)

**Files modified:** 1
- `pyros/Cargo.toml` (trybuild dev-dep)

**Tests:** 163 total (72 pyros-core + 90 pyros-macros + 1 compile_fail), all passing.
- +1 trybuild test (runs 10 compile-fail cases internally)

**Quality gates:** `cargo fmt --check` clean, `cargo clippy -- -D warnings`
zero warnings, all tests pass.
