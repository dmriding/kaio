# Sprint 5.6 — CI/CD: Windows + Linux Matrix

**Status:** Done
**Branch:** phase5
**Goal:** Add Windows to CI, add doc build job.

## What Changed

`.github/workflows/ci.yml`:
- `check` job: unchanged (fmt + clippy, Ubuntu)
- `test` job: matrix `[ubuntu-latest, windows-latest]`
- `doc` job: new — `cargo doc --no-deps --workspace` (Ubuntu)

No Rust code changes. Config-only sprint.

## Why

- README claims Windows support — CI should verify it
- Phase 5 added attention kernels, auto-tuner with file I/O,
  serde deps — all should compile on Windows
- `cargo doc` catches broken intra-doc links

## Not Included

- macOS (not claimed, no demand)
- GPU tests in CI (no GPU runners)
- Release/publish automation
- Coverage reporting

## Files

| File | Change |
|------|--------|
| `.github/workflows/ci.yml` | Windows matrix + doc job |
| `docs/development/sprints/phase5/PHASE_5_LOG.md` | Updated |
