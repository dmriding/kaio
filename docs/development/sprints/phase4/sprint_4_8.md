# Sprint 4.8 — Polish, Integration Tests & Publish

**Status:** Complete
**Commit:** (pending)
**Date:** 2026-04-11
**Depends on:** All previous sprints

## Context

Final Phase 4 sprint. Close out with documentation updates, integration
test coverage, version bump, and crates.io publish preparation.

## Completed Items

### Version Bump (0.0.3 → 0.0.4) — Done

Workspace version bumped in root Cargo.toml. All inter-crate deps
updated to 0.0.4. Cargo.lock regenerated.

### kaio-ops Publish Readiness — Done

- Created `kaio-ops/README.md` with matmul example and benchmark table
- Added `readme` and `include` fields to Cargo.toml
- Added `#![warn(missing_docs)]` lint (all 5 crates now consistent)

### kaio-macros Metadata — Done

Added missing `keywords` and `categories` for crates.io discoverability.

### CHANGELOG.md — Done

Added `[0.0.4]` Phase 4 section with Added (11 items) and Fixed (4 items).
Previous `[Unreleased]` header changed to `[0.0.3]`.

### README Updates — Done

- Main README: feature table updated (matmul ✅, FMA ✅, 2D blocks ✅),
  kaio-ops added to crate table, roadmap Phase 4 checked
- kaio/README.md: status updated to Phase 4 complete, kaio-ops added
- kaio/src/lib.rs: module doc updated (Status, Crates sections)

### docs/phases.md — Done

Phase 4 section updated to reflect actual sprint structure and
deliverables (8 sprints with actual scope, not planned scope).

### Integration Tests — Done

- `api_matmul_rejects_zero_n` — validates N=0 rejection
- `api_matmul_rejects_zero_k` — validates K=0 rejection
- `cf11_2d_reduce_rejected` — compile-fail: 2D kernel + block_reduce
  produces clear error about TidX-only warp identity
- `cf12_fma_wrong_type` — compile-fail: fma(u32, f32, f32) rejected

### Publish Order Documentation — Done

Created `PUBLISH_ORDER.md` (gitignored) documenting dependency-order
publish sequence with dry-run-first protocol.

## Tests

+2 GPU validation tests (matmul zero-dim edge cases)
+2 compile-fail tests (12 total)

Total: 207 host tests + 43 GPU tests + 1 benchmark

## Files Modified

| File | Changes |
|------|---------|
| `Cargo.toml` | Version 0.0.3 → 0.0.4 |
| `kaio-runtime/Cargo.toml` | Dep version bump |
| `kaio/Cargo.toml` | Dep version bump |
| `kaio-ops/Cargo.toml` | Dep bump + readme + include |
| `kaio-macros/Cargo.toml` | Keywords + categories |
| `kaio-ops/README.md` | New |
| `kaio-ops/src/lib.rs` | `#![warn(missing_docs)]` |
| `CHANGELOG.md` | Phase 4 section |
| `README.md` | Feature table, crate table, roadmap |
| `kaio/README.md` | Status + kaio-ops |
| `kaio/src/lib.rs` | Module doc |
| `docs/phases.md` | Phase 4 status |
| `kaio-ops/tests/matmul_api.rs` | +2 validation tests |
| `kaio/tests/compile_fail/cf11_*.rs` | New |
| `kaio/tests/compile_fail/cf12_*.rs` | New |
| `.gitignore` | PUBLISH_ORDER.md |
