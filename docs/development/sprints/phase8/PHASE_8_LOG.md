# Phase 8 — Python Bindings + Pointer Syntax: Sprint Index

Quick-reference index for Phase 8 sprints. Each sprint gets a dedicated
doc in this directory with a post-delivery outline of what shipped.

Master plan: [../../phases.md](../../phases.md) §Phase 8

## Sprint Status

| Sprint | Scope | Status | Headline |
|---|---|---|---|
| [8.0](sprint_8_0.md) | Pointer syntax (RFC-0001) — `*mut [T]` / `*const [T]` in `#[gpu_kernel]` | ✅ Complete | Resolves #13 (rust-cuda DSL-soundness feedback); parser-only extension, zero IR/codegen/runtime change; v0.4.1 |
| [8.0.5](sprint_8_0_5.md) | Bench coverage extension — QKV + attention + norm/activation under `cargo xtask bench` | ✅ Complete | Seven bench harnesses covering the shipped public kernel families + showcase kernels; zero API / runtime change |
| 8.1 | PyO3 scaffold, Python module, NumPy interop | 📝 Planned | — |
| 8.2 | Expose core ops to Python (matmul f32 + INT8, attention, softmax, activations) | 📝 Planned | — |
| 8.3 | Python-side benchmarking + cross-validation against PyTorch | 📝 Planned | — |
| 8.4 | Documentation, examples, `pip install` packaging, Windows + Linux CI | 📝 Planned | — |

## Branch

Sprint 8.0 shipped on the `phase8` branch and merged to `main` via
PR #15 on 2026-04-24 (commit `e8b7ae6`), carrying v0.4.1 across the six
publishable crates (`kaio`, `kaio-core`, `kaio-macros`, `kaio-ops`,
`kaio-runtime`, plus `kaio-candle` 0.1.1).

Sprint 8.0.5 work lives on a dedicated branch off `main`; Phase 8.1+
PyO3 work will open its own branch when scoping begins.

## Key References

- **Phases roadmap:** [../../phases.md](../../phases.md) §Phase 8
- **RFC-0001 Pointer Syntax:** [../../rfcs/rfc-0001-pointer-syntax.md](../../rfcs/rfc-0001-pointer-syntax.md) — accepted + implemented in Sprint 8.0
- **Issue #13:** <https://github.com/dmriding/kaio/issues/13> — closed 2026-04-24 with the 0.4.1 release comment

## Phase 8 delta so far

Sprint 8.0 did not add a new op to `kaio-ops`. It extended the
`#[gpu_kernel]` parameter-syntax surface with `*const [T]` and
`*mut [T]` as primary forms, retained `&[T]` / `&mut [T]` as permanent
sugar (both lower to identical PTX), and migrated every user-facing
example + doctest + tutorial snippet to the pointer form. The
Phase-8.5 item originally planned for after the PyO3 work was absorbed
as this Sprint 8.0 so the syntax story lands before Python bindings
widen the public surface.

See [sprint_8_0.md](sprint_8_0.md) for the post-delivery outline.
