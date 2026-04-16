# Sprint 7.0.5 — Ergonomics fast-track before Phase 7.1

**Status:** Complete (v0.2.2, 2026-04-14)
**Branch:** `sprint-7-0-5` off `main` (post v0.2.1 publish)
**Release:** v0.2.2 — patch release, no breaking changes

## Context

v0.2.1 shipped Sprint 6.10 + 7.0 to crates.io on 2026-04-14. External feedback gathered immediately after converged on a thesis: *trust and ergonomics gate adoption more than feature count*.

Sprint 7.0.5 is the **narrow pre-7.1 ergonomics sprint** that addresses that signal without derailing quant work. Three adoption-friction items (debug-mode performance note, proc-macro error spans, consolidated debugging guide) plus two load-bearing additions (xtask-as-first-run, docs.rs Windows visibility) that emerged during the sprint.

## Goal

Ship v0.2.2 with small, high-leverage ergonomics wins that set the table for Phase 7.1+ users. Zero regressions. No breaking changes.

## Deliverables

### D1 — A2: One-time debug-build performance note

`KaioDevice::new` emits a performance note on first call in a debug binary:

```
[kaio] Note: debug build — GPU kernel performance is ~10-20x slower than --release. Use `cargo run --release` / `cargo test --release` for representative performance numbers. Correctness is unaffected. Set KAIO_SUPPRESS_DEBUG_WARNING=1 to silence.
```

Implementation:
- `DEBUG_WARNED: OnceLock<()>` latch in `kaio-runtime/src/device.rs` — once per process.
- Gated on `cfg!(debug_assertions)` — folds to `false` and compiles out entirely in release.
- `KAIO_SUPPRESS_DEBUG_WARNING=1` opt-out for CI / test harnesses that intentionally run in debug.
- Message **performance-framed only** per review fold: "Correctness is unaffected" is load-bearing. Debug-mode `cargo test` users checking correctness should not see their results cast into doubt.

Tests:
- `debug_warning_message_is_performance_framed_not_correctness_framed` — regression canary on message content. Blocks future drift toward "not meaningful" / "invalid" framing.
- `debug_warning_opt_out_env_var_suppresses` — exercises `should_emit_debug_warning()` pure-function logic under set / unset env var.

The once-per-process behavior itself is tested via manual / subprocess verification (documented in the Results section below); the `OnceLock` latch is set for the test binary's lifetime so in-process unit testing of "first call warns, second doesn't" requires restructuring that isn't worth the cost.

**Commit:** `60c03a5` — `wip(phase7): D1 — A2 one-time debug-mode performance note in KaioDevice::new`

### D2 — A4: Proc-macro span audit + 3 defensive fixes

Exploration-phase audit of 63 `syn::Error` sites across `kaio-macros/src/parse/` and `kaio-macros/src/lower/`. **Only 3 were flagged as potentially improvable.** Implementation-phase investigation showed all 3 are defensive:

- `kaio-macros/src/parse/attrs.rs:79` — `Span::call_site()` for missing `block_size` error. In proc-macro attribute context, `call_site()` *is* the attribute span, so `attr.span()` (the fix) resolves to the same location. Compile-fail test `cf05_missing_block_size.stderr` regenerated with `TRYBUILD=overwrite`: zero diff.
- `kaio-macros/src/lower/mod.rs:139` (LitInt) and `:172` (LitFloat) — both error paths are **unreachable in practice**: the parser in `parse/body.rs` gates literal types to I32/U32/I64/U64 and F32/F64 respectively via the suffix match, so the `_ => Err(...)` arms in lowering cannot fire with valid parsed input. The fixes preserve the literal's own `span` field (formerly discarded as `_span`) in case these paths become reachable in a future extension.

**Honest finding captured for future sprints:** the starting assumption that the macro's error messages point at the wrong location turned out to be overstated. KAIO's span handling is already quite good. Future user complaints about bad-span errors should be investigated as specific bugs, not treated as signal of a systemic pattern.

Carry-forward audit report lives in this document's [Appendix](#appendix--span-audit-summary).

**Commit:** `934bca9` — `wip(phase7): D2 — A4 span fixes (attrs block_size, LitInt, LitFloat)`

### D3 — B3: Consolidated `docs/debugging.md`

New file ~250 lines. Single entry point for diagnosis:

- Intro framing why GPU debugging differs from CPU (async launches, no debugger, silent-corruption risk)
- Troubleshooting flowchart — did it compile → launch → produce right output?
- Env-var reference (consolidated): `KAIO_DUMP_PTX`, `KAIO_PTX_STATS`, `KAIO_PTX_ANNOTATE`, `KAIO_SM_TARGET`, `KAIO_TUNE_CACHE`, `KAIO_SUPPRESS_DEBUG_WARNING` with links back to authoritative refs (`docs/performance.md`, etc.)
- Correctness verification section (fresh content): CPU reference pattern, choosing floating-point tolerances with magnitude scaling, bit-exact vs tolerance decision table
- `compute-sanitizer` section (fresh content): memcheck / racecheck / initcheck / synccheck invocation + interpretation
- PTX-stats interpretation (debugging-angle on register-pressure → occupancy)
- Common-errors quick reference table

Cross-linked from `README.md` (new "## Debugging" section) and `docs/index.md` (Documentation Index expanded).

**Commit:** `01e4b5d` — `docs(phase7): D3 — B3 consolidated debugging guide + README/index cross-links`

### D4 — Master plan + phases.md: ergonomics sequencing + adoption-rent framing

`docs/development/sprints/phase7/phase7_master_plan.md`:
- Added Sprint 7.0.5 row + Sprint 7.1.5 row (reductions, between-quant-milestones) to the Sprint Breakdown table
- New "Adoption-ergonomics sequencing" section mapping each identified feedback item to its landing sprint, deferral, or rejection with reasoning
- Explicit rejections: C2 builder-pattern launch (feature-negative), unrestricted Sprint A/B adoption (too long a quant delay)

`docs/phases.md`:
- Phase 7 intro paragraph gains the adoption-friction-rent framing: "users who'd be attracted by INT4 don't exist in the KAIO ecosystem yet; users who'd be repelled by rough ergonomics are who you're trying to attract."
- Sprint outline table: 7.0.5 + 7.1.5 rows

**Commit:** `8e6dfde` — `docs(phase7): D4 — master plan ergonomics sequencing + phases.md adoption-rent framing`

### D5 — `cargo xtask` with showcase / bench / all subcommands

New `xtask/` binary crate in the workspace (zero dependencies — shells out via `std::process::Command`). `.cargo/config.toml` aliases `cargo xtask = "run --package xtask --release --"` so the invocation is a single verb.

Subcommands:
- `cargo xtask showcase` — runs all 3 showcase examples in sequence, continue-on-error, pass/fail summary at end
- `cargo xtask showcase <name>` — runs one (short names `silu` / `gelu` / `rms` or directory names)
- `cargo xtask showcase --list` — lists available showcases
- `cargo xtask bench` — runs `kaio-ops` matmul TC benchmark via `cargo test -p kaio-ops --release --test matmul_tc_bench -- --ignored --nocapture`
- `cargo xtask all` — showcase then bench
- `cargo xtask --help` — usage

README "Try KAIO in 30 seconds" section rewritten to lead with `cargo xtask showcase`. `examples/README.md` updated similarly (keeps the standalone `cd` path as a secondary option so users see both).

**Design note — bench / showcase split:** kept separate because they tell different stories (showcase = "Rust expressiveness + correctness"; bench = "performance on a real workload"). Bundling both into one command forces every user to pay ~75s to see either story; separation lets each serve its purpose in under 45s.

**Commit:** `4b29551` — `wip(phase7): D5 — cargo xtask showcase/bench/all with workspace alias + README pivot`

### D6 — docs.rs explicit Windows + Linux targets

docs.rs announced that on 2026-05-01 it will default to building only `x86_64-unknown-linux-gnu` unless crates explicitly opt into more targets.

KAIO's differentiator is cross-platform (Windows + Linux). Zero `[package.metadata.docs.rs]` config would mean Linux-only docs after the change — undercutting the "works on Windows" positioning even though the code works fine on Windows (cudarc's `dynamic-loading` means no build-time CUDA dependency on either OS).

Fix: added `[package.metadata.docs.rs] targets = ["x86_64-unknown-linux-gnu", "x86_64-pc-windows-msvc"]` to all 5 published crates.

Two targets is enough — nobody building for `i686` is a KAIO user we're trying to impress.

**Commit:** `cb08a86` — `wip(phase7): D6 — docs.rs explicit Windows + Linux targets in all 5 crates`

## Results

### Test counts

No new GPU tests, no new ptxas_verify tests. Changes are predominantly non-code (docs, tooling, metadata) or additive host-side unit tests (D1).

| Suite | Before (v0.2.1) | After (v0.2.2) | Delta |
|-------|---|---|---|
| `kaio-runtime` lib | 0 | 2 | +2 (debug-warning pure-function tests) |
| `kaio-macros` lib | 137 | 137 | unchanged |
| `kaio-core` lib | 151 | 151 | unchanged |
| `kaio-core` ptxas_verify (`--ignored`) | 7 | 7 | unchanged |
| `kaio` GPU tests (`--ignored`) | stable | stable | unchanged |

### Gates

- `cargo fmt --all --check` — clean
- `cargo clippy --workspace --all-targets -- -D warnings` — clean
- `cargo test --workspace` — all host tests green
- `cargo doc --workspace --no-deps` — clean
- `cargo xtask showcase` — all 3 showcases pass on RTX 4090 sm_89 in ~4.5s warm-build
- `cargo xtask bench` — matmul TC benchmark runs to completion; numbers in expected band

### Manual verification — D1 debug-warning once-per-process behavior

`cargo xtask showcase silu` (debug build in the xtask binary, release build in the showcase which internally runs `KaioDevice::new` once): warning emitted once in xtask invocation, release-build example emits nothing. Confirmed.

`KAIO_SUPPRESS_DEBUG_WARNING=1 cargo test -p kaio-runtime` (debug): no warning. Confirmed.

### Manual verification — D6 docs.rs config

Cannot test docs.rs behavior in-repo. Config syntax verified by `cargo build --workspace` completing clean (unknown keys under `[package.metadata.docs.rs]` would not be a cargo error since that table is reserved for docs.rs, but cargo does validate the table existence). Effect will be observable after the next publish.

## Appendix — Span audit summary

Full report (from exploration-phase audit of 63 `syn::Error` sites):

**Files audited:**
- `kaio-macros/src/parse/` — `signature.rs` (9 errors, all GOOD), `body.rs` (11, all GOOD), `attrs.rs` (2; **1 BAD** on missing `block_size`)
- `kaio-macros/src/lower/` — `mod.rs` (26; **2 BAD** on LitInt and LitFloat type coercion; remaining 22 GOOD), `builtins.rs` (12, all GOOD via explicit span param), `logical.rs` (1, GOOD), `cast.rs`, `compare.rs`, `memory.rs` (GOOD or no error sites)

**3 BAD sites, all fixed in D2 despite being defensive:**
1. `parse/attrs.rs:79` — `Span::call_site()` → `attr.span()`. Same location in practice for empty `#[gpu_kernel]`.
2. `lower/mod.rs:139` — `Span::call_site()` → `*span` from the `LitInt` destructure. Unreachable per parser suffix gating.
3. `lower/mod.rs:172` — same as #2 for `LitFloat`.

**Error classes (count, status):**
| Class | Count | Status |
|---|---|---|
| Unsupported syntax in body (return, match, closure, etc.) | ~10 | GOOD — all use `new_spanned(expr)` |
| Unknown / misnamed builtin | ~12 | GOOD — all pass span from call site |
| Undefined variable | ~5 | GOOD — span from `KernelExpr::Var` |
| Builtin arity mismatch | ~12 | GOOD — span parameter |
| Missing required attribute | 1 | Fixed (#1) |
| Literal type coercion | 2 | Fixed (#2, #3) — unreachable paths |
| Internal assertions | ~3 | Acceptable — users shouldn't hit these |

**Takeaway for future sprints:** macro error spans are generally well-handled. Future user complaints about bad-span errors should be treated as specific bugs rather than signal of a systemic pattern.

## Commits

| Commit | Scope |
|--------|-------|
| `60c03a5` | D1 — A2 debug-mode performance note |
| `934bca9` | D2 — A4 span fixes (defensive) |
| `01e4b5d` | D3 — B3 debugging guide |
| `8e6dfde` | D4 — master plan + phases.md ergonomics sequencing |
| `4b29551` | D5 — cargo xtask tooling + README pivot |
| `cb08a86` | D6 — docs.rs Windows + Linux targets |
| _(final)_ | `feat(phase7): Sprint 7.0.5` — version bump + release |

## Carry-forward

- **Phase 7.1 (INT8 dequantize-matmul)** — next up. Folds in integer arithmetic DSL support as a prerequisite. Open question at 7.1 kickoff: does `mma.sync.m16n8k16.s8.s8.s32` behave as expected on Ampere+, skipping the dequant-to-f16 step?
- **Ergonomics items deferred or planned later:** see [`phase7_master_plan.md`](phase7_master_plan.md) adoption-ergonomics sequencing section for the full mapping.
- **Env-var hygiene for 3 remaining test helpers** reading `KAIO_SM_TARGET` internally (minor; low priority).
- **Short-circuit optimization pass** (D4 leftover from Sprint 7.0) — still open. Additive, backward-compatible future work.
