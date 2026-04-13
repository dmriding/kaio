# Sprint 1.0 — Workspace Restructure

**Commit:** `8e59f66`
**Status:** Complete

## Context

The name-reservation crate (`kaio v0.0.1`) was a single crate at the
workspace root — `Cargo.toml` had both `[workspace]` and `[package]`
sections, with `members = ["."]`. Phase 1 requires three crates
(`kaio`, `kaio-core`, `kaio-runtime`), so the repo needed
restructuring before any code sprints could begin.

## Decisions

### Virtual workspace vs umbrella-at-root

**Context:** Two common Rust workspace patterns exist:
1. *Umbrella at root* — root `Cargo.toml` is both workspace and package,
   sub-crates are siblings of `src/`. Used by serde, tokio.
2. *Virtual workspace* — root `Cargo.toml` is `[workspace]` only (no
   `[package]`), the umbrella crate is a subdirectory alongside sub-crates.

**Options considered:**
- **(A) Keep umbrella at root.** Simpler — no move needed. But when
  `kaio-core/` is added as a sibling, you end up with `src/` (the
  umbrella), `kaio-core/src/`, and `kaio-runtime/src/` at the same
  level. Confusing for contributors. Also doesn't match the crate
  structure diagram in `docs/index.md`.
- **(B) Virtual workspace.** Matches `docs/index.md`'s stated layout.
  Root is purely workspace config. Each crate gets its own clean directory.
  More files to move (one-time cost).

**Decision:** (B) Virtual workspace. Matches the design docs. Avoids the
`src/` ambiguity. One-time restructure cost is low (a few `git mv`s).

### Resolver version — 2 vs 3

**Context:** Edition 2024 defaults to resolver 3. Our name-reservation
crate used resolver 2 (the edition 2021 default).

**Decision:** Bumped to resolver 3 to match the edition. Resolver 3 has
subtler feature-unification behavior in workspaces but we verified with
`cargo build --workspace` that everything resolves correctly.

### cudarc CUDA version feature

**Context:** The original plan had `cudarc` with `features = ["driver",
"std", "dynamic-loading"]`. Build failed — cudarc 0.19.4's `build.rs`
panics if no `cuda-XXXXX` feature is selected (it needs to know which
FFI binding headers to generate).

**Options:**
- `cuda-version-from-build-system` — auto-detect from installed toolkit.
  Would break on machines without a CUDA toolkit (contradicts the
  dynamic-loading goal of no build-time CUDA requirement).
- `cuda-12080` — pin to CUDA 12.8 headers. CUDA 12.8 is installed on
  the dev machine. Combined with `dynamic-loading`, the binding ABI is
  set at build time but the actual driver is resolved at runtime. Can
  build without toolkit installed (untested — to be verified on a
  clean CI machine in Phase 5).

**Decision:** `cuda-12080`. Matches the installed toolkit exactly.
Documented the reasoning in a comment in root `Cargo.toml`.

### README / LICENSE placement

**Context:** `cargo publish -p kaio` packages only files inside the
crate directory. The workspace root has `README.md` and `LICENSE-*`
for GitHub browsing. The `kaio/` crate needs its own copies for the
crate tarball to be self-contained.

**Decision:** Copy (not symlink) README + LICENSE files into `kaio/`.
Symlinks work on Linux but are flaky on Windows. Duplication is cheap —
these are short, static files. Verified with `cargo package -p kaio
--allow-dirty --list` that the tarball contains the expected 8 files.

## Scope

**In:** Workspace restructure, toolchain pin, doc sync (MSRV, cudarc URL).
**Out:** Any functional code. Stub crates have empty `lib.rs` only.

## Results

Completed as planned. One deviation: cudarc CUDA version feature was
discovered during build (not in the original plan). Added `cuda-12080`
and documented.

**Files:** 7 created, 3 modified, 1 moved. All quality gates clean.
