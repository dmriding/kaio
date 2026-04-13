# Sprint 6.9 — v0.2.0 publish prep

**Status:** In progress (2026-04-13)
**Branch:** `phase6`
**Parent:** `9038182` — Sprint 6.7b closeout doc-sweep

---

## Goal

Tag, polish, and publish KAIO v0.2.0 to crates.io. v0.2.0 is the
flagship release closing out Phase 6 — tensor cores, async copies,
fp16 / bf16 types, `matmul_auto_tc` with the full tuner cache, and
three standalone showcase examples. Headline:
**92.5% of cuBLAS sgemm at 4096²** on the async TC matmul path.

---

## Scope

### Version bumps

- Workspace root `Cargo.toml`: `[workspace.package] version = "0.1.0"` → `"0.2.0"`.
- All 5 member crates inherit via `version.workspace = true` (no per-crate bumps needed).
- Inter-crate version constraints in `kaio`, `kaio-runtime`, `kaio-ops` dependency tables: `version = "0.1.0"` → `"0.2.0"` on every `kaio-core` / `kaio-runtime` / `kaio-macros` / `kaio` reference. These path-plus-version specs are required for `cargo publish` to work — cargo uses the `path` for local builds and the `version` for the published manifest.

### Example path-dep hardening

The three standalone showcase examples (`examples/fused_silu_gate`, `gelu_comparison`, `rms_norm`) each have a single dependency on `kaio`. Updated from bare `kaio = { path = "../../kaio" }` to `kaio = { path = "../../kaio", version = "0.2.0" }`:

- `path` keeps the examples buildable inside the repo against the local checkout.
- `version = "0.2.0"` means a user who copies an example directory out of the repo and deletes the path (or removes the entire line and re-adds `kaio = "0.2.0"`) gets a clean crates.io build against the published crate.
- The in-tree `cargo run --release` behaviour inside each example directory is unchanged — cargo resolves via `path` first when present.

### CHANGELOG finalisation

- `[Unreleased]` section rolled forward to `[0.2.0] — 2026-04-13 — Phase 6: Tensor Cores & Async Copies`.
- Added a v0.2.0 headline summary at the top of the release section (before the per-sprint breakdown) listing the four key shippable items: tensor-core matmul stack, fp16 / bf16 types, `PtxModule::validate()` + SM gating, three standalone examples.
- The dated release heading is the canonical pointer — all prior `[Unreleased]` content moves beneath it unchanged.

### Rustdoc polish on promoted public APIs

Three public APIs are v0.2.0-stable (Sprint 6.7 D7 promotion + 6.7b
closeout):

- `kaio_ops::matmul_tc` (sync variant)
- `kaio_ops::matmul_tc_async` (cp.async double-buffered variant)
- `kaio_ops::matmul_auto_tc` (auto-tuned dispatcher with cache)

The crate-level rustdoc for `kaio-ops` and the function-level rustdoc on
`matmul_auto_tc` both previously cited 79.9% / 85.1% (Sprint 6.7
numbers). Updated to reflect 82.3% / 92.5% (Sprint 6.7b final state)
with the 6.7b additions (bank-conflict padding, D10 hoist) explicitly
named in the performance commentary.

The size-heuristic fallback constant `ASYNC_FALLBACK_MAX_DIM_THRESHOLD`
has an inline rationale comment that referenced the pre-6.7b 6.5% async
win; updated to reflect the post-6.7b 12.4% gap. The threshold value
(3072) is unchanged — the measured crossover shape didn't move, only
the magnitude of the win on either side.

### Phase 6 sprint log wrap

- `PHASE_6_LOG.md`: 6.9 row added, marked in-progress then complete at tag time.
- `phase6_master_plan.md`: status line rolled from "6.1–6.8 + 6.7b complete; 6.9 pending" to "Phase 6 complete".
- Add a short "Phase 6 complete" summary block at the top of the master plan pointing at the v0.2.0 tag.

### Publish checklist (executed at tag time)

1. `cargo publish -p kaio-core --dry-run` → full publish
2. Wait for crates.io indexing (~30s)
3. `cargo publish -p kaio-runtime --dry-run` → full publish
4. `cargo publish -p kaio-macros --dry-run` → full publish
5. `cargo publish -p kaio --dry-run` → full publish
6. `cargo publish -p kaio-ops --dry-run` → full publish
7. `git tag v0.2.0 <commit>` + `git push --tags`
8. GitHub release with CHANGELOG 0.2.0 section as the body
9. Launch post / announcements

### Explicitly out of scope

- README.md rewrite / landing copy — handled separately with external feedback. The landing page gets the 92.5% headline and the positioning pass that a launch deserves; it's not mechanical doc work.
- env-var-based test isolation fix (adversarial-review finding #2 on 2026-04-13): deferred to a dedicated post-v0.2.0 hygiene sprint with the clean-fix path already scoped (parameterize SM target + cache path into helper functions, eliminating `set_var` calls entirely). Pre-existing from Phase 5, zero test failures in practice, fix is mechanical but not small. Tracked in `tech_debt.md`.

---

## Quality gates (pre-publish)

- `cargo fmt --all --check` — clean
- `cargo clippy --workspace --all-targets -- -D warnings` — clean
- `cargo test --workspace` — 286 host tests pass
- `cargo test --workspace -- --ignored` on RTX 4090 sm_89 — 148 GPU tests pass
- `cargo test -p kaio-core --test ptxas_verify -- --ignored` — 6/6 pass
- `cargo test -p kaio-ops --test matmul_tc_bench -- --ignored --nocapture` — 82.3% / 92.5% at 4096² confirmed
- `cargo doc --workspace --no-deps` — clean, no broken intra-doc links
- `cargo build --release` — clean across all 5 workspace crates
- Each example (`examples/fused_silu_gate`, `gelu_comparison`, `rms_norm`): `cargo run --release` from inside its directory produces PASS + timing output
- `cargo publish --dry-run -p <crate>` passes for each crate in publish order (core → runtime → macros → kaio → kaio-ops)

---

## Results

Populated at tag time.
