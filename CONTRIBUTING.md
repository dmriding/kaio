# Contributing to KAIO

## Development Process

KAIO is developed using AI-assisted coding tools (Claude Code, Opus 4.6)
for implementation, with all architecture decisions, API design, and code
review done by the maintainer. AI handles the mechanical work: writing
instruction emitters, generating test boilerplate, implementing spec'd
sprint plans. The human handles the hard parts: deciding what to build,
how the API should feel, what tradeoffs to make, and whether the output
is correct.

Every line of generated code is reviewed, tested against nvcc golden
output, validated with `ptxas --verify`, and executed on real GPU
hardware before merge. The test suite includes 200+ host tests and 24
GPU E2E tests with numerical accuracy validation against CPU reference
implementations. If it ships, it works.

## Building

```sh
cargo build --workspace
cargo test --workspace                  # host tests (no GPU)
cargo test --workspace -- --ignored     # GPU tests (requires NVIDIA GPU)
```

Requires Rust 1.94+ (pinned via `rust-toolchain.toml`).

## Quality Gates

All contributions must pass before merge:

- `cargo fmt --check` — zero formatting issues
- `cargo clippy --workspace -- -D warnings` — zero warnings
- `cargo test --workspace` — all host tests pass
- `cargo test --workspace -- --ignored` — all GPU tests pass (on
  contributor's hardware)

## Code Style

- Doc comments on public items (aiming for `#![deny(missing_docs)]` by v0.1.0)
- Every `unsafe` block has a `// SAFETY:` comment
- Conventional commits: `feat:`, `fix:`, `test:`, `docs:`, `refactor:`
- Tests inline in source files (`#[cfg(test)] mod tests`)
- GPU tests marked `#[ignore]` with comment `// requires NVIDIA GPU`
- `KAIO_SM_TARGET` env var controls PTX target (default `sm_70`)

## Architecture

See [docs/index.md](docs/index.md) for the four-layer architecture and
[docs/implementation.md](docs/implementation.md) for technical details.
Sprint-by-sprint development logs with architectural decision records are
in [docs/development/sprints/](docs/development/sprints/).

## License

By contributing, you agree that your contributions will be dual licensed
under MIT and Apache-2.0, consistent with the project's existing license.
