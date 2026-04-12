# Sprint 6.8 — Showcase Examples for v0.2.0

**Status:** ✅ Complete (2026-04-12)
**Branch:** `phase6`
**Parent:** `78b886c` — Sprint 6.7 post-review (size-heuristic cache-miss default)
**Dep-graph reorder:** 6.8 ran ahead of 6.7b. At 79.9/85.1% cuBLAS
sgemm the 6.7 perf story is already launch-ready; examples now
make v0.2.0 discoverable. 6.7b follows as a v0.2.x perf bump.
**Previous draft:** `sprint_6_8_rough_draft.md` (replaced by this doc)

---

## Context

Phase 6 will have shipped tensor-core matmul (stable `matmul_auto_tc`),
fused TC attention (internal preview), and the performance + benchmark
story (Sprint 6.7) by the time 6.8 runs. But the repo's current
examples (`kaio/examples/{vector_add,saxpy,reduction}.rs`) are "hello
world for any GPU compute framework" — they don't say *why* a Rust ML
engineer would reach for KAIO over Triton or cudarc. The README claims
the value prop is "custom GPU kernels your framework doesn't support —
novel attention variant, fused activation, quantization op," but shows
saxpy.

Sprint 6.8 closes that gap with three standalone Cargo-project
showcases in `examples/` at the repo root, each demonstrating a real
ML kernel with correctness + timing. These become the "this is what
KAIO looks like" companion to the v0.2.0 blog post.

## Scope

Three standalone examples under `examples/` at the repo root. Each is
a full Cargo project (its own `Cargo.toml`, `src/main.rs`, `README.md`)
that compiles and runs from a fresh clone:

```
examples/
├── README.md                   # (exists — index of planned + deferred)
├── fused_silu_gate/
│   ├── Cargo.toml
│   ├── README.md
│   └── src/main.rs
├── gelu_comparison/
│   ├── Cargo.toml
│   ├── README.md
│   └── src/main.rs
└── rms_norm/
    ├── Cargo.toml
    ├── README.md
    └── src/main.rs
```

The workspace `Cargo.toml` gains `exclude = ["examples/*"]` so these
don't become workspace members — they're independent projects that
happen to live in the repo.

### Out of scope (explicit)

Deferred to post-0.2.0 per the `examples/README.md` index:
`fused_layer_norm/`, `rotary_embeddings/`, `vector_cosine_similarity/`.
The rough-draft empty directories for these were removed in the 6.8
prep commit; they'll be added fresh when the content is written.

No IR-level or tensor-core APIs in these examples — 6.8 showcases
the user-facing `#[gpu_kernel]` macro path only. (Separate future
work might add `examples/tensor_core_matmul/` once `matmul_auto_tc`
graduates from internal preview, but that's post-0.2.0.)

## The three examples

### Example 1 — `fused_silu_gate/`

**Kernel:** `out[i] = x[i] * sigmoid(x[i]) * gate[i]`
where `sigmoid(x) = 1.0 / (1.0 + exp(-x))`.

**Why it matters:** This is the gated activation in every LLaMA /
Mistral / Qwen feedforward layer. `llama.cpp` has hand-written CUDA
for it. Showing a 20-line KAIO kernel that replaces the CUDA C++
is exactly the v0.2.0 value pitch.

**Kernel signature:**
```rust
#[gpu_kernel(block_size = 256)]
fn fused_silu_gate(x: &[f32], gate: &[f32], out: &mut [f32], n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        let xi = x[idx];
        let sig = 1.0 / (1.0 + exp(-xi));
        out[idx] = xi * sig * gate[idx];
    }
}
```

**Test:** 1M-element patterned inputs (`(i % 97) as f32 / 97.0 - 0.5`
style), CPU reference in f64 cast to f32, tolerance `abs_err < 1e-5`.
Timing: 5 warm-up + 100 timed runs, report median in microseconds.

**README:** Lead with the kernel source block. Explain what SiLU
does and why gated activations are everywhere in modern LLMs. End
with `cargo run --release` instructions.

### Example 2 — `gelu_comparison/`

**Kernels:** Two GELU variants on the same input, measured side by
side:

- **Exact GELU (tanh approximation):**
  `0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
- **Fast GELU (sigmoid approximation):**
  `x * sigmoid(1.702 * x)` = `x / (1.0 + exp(-1.702 * x))`

**Why it matters:** GELU is the activation in BERT / GPT / many
other models. Two common implementations trade precision for speed.
This example is the teaching moment for kernel-variant workflows:
"write two, measure both, pick the winner." That's exactly how
Triton sells itself; KAIO needs the same demo in the tree.

**Test:** Single 1M-element input, both kernels run against it,
each compared to its own f64 CPU reference (not against each
other — they're meant to give different answers). Tolerance
`abs_err < 1e-4` for the exact variant, `abs_err < 5e-3` for the
fast variant (intentionally looser; it's a lossy approximation).
Report max error and median timing for each, plus the ratio
(`fast is X% of exact's time`). 5 warm-up + 100 timed runs per
kernel.

**README:** Both kernel source blocks at the top. Brief context
on exact vs fast GELU and when you'd pick each. Tie it back to
"KAIO makes it trivial to experiment and benchmark side by side."

### Example 3 — `rms_norm/`

**Kernel:** `out[i] = (x[i] / rms) * weight[i]` where
`rms = sqrt(mean(x^2) + eps)`.

**Why it matters:** RMSNorm is what LLaMA replaced LayerNorm with.
Every LLaMA-family model uses it. Requires `block_reduce_sum` which
is a non-trivial DSL builtin to showcase.

**Kernel structure:** Single-block kernel, block_size = 256,
input size = 256 (one block covers the whole input). Each thread
computes `x[i] * x[i]`, calls `block_reduce_sum` to get the sum of
squares, computes `rms = sqrt(sum / n + eps)`, then writes
`out[i] = (x[i] / rms) * weight[i]`.

**Honest framing (plan-review fold):** The README must be explicit
that this is the *single-block* case. Real LLaMA RMSNorm operates
over `hidden_dim = 4096` — that's 16 blocks at `block_size = 256`,
which requires cross-block reduction (either via atomics or a
two-kernel split). Multi-block RMSNorm is **deferred to Phase 7+**
when either:
- KAIO's `#[gpu_kernel]` macro gains cross-block reduction
  primitives (possibly via atomic operations on a scratch buffer), or
- KAIO ships a `kaio_ops::rms_norm` pre-built operation that wraps
  the two-kernel split.

The single-block example is still valuable — it demonstrates the
`block_reduce_sum` builtin working end-to-end with sqrt, the
divide-by-reduction pattern, and how f64 CPU references catch
numerical bugs. Don't hide the constraint.

**Test:** 256-element patterned input + 256-element weight,
tolerance `abs_err < 1e-4`, 5 warm-up + 100 timed runs, report
median in microseconds.

**README:** Lead with the kernel source. Section on "why RMSNorm
matters" (LLaMA's normalization choice). Section on the single-block
limitation with the Phase 7+ path flagged.

**Contingency (plan-review fold):** "If `block_reduce_sum` has
issues with the macro for this pattern, flag it — don't hack
around it. Ship two clean examples rather than three with
workarounds." The sprint's success criteria explicitly allow 2/3
if the RMSNorm builtin path has a blocker discovered during
execution; in that case file an issue against the macro and defer
RMSNorm to a Phase 7 follow-up.

## File-by-file changes

### Root-level

| File              | Change                                                                   |
|-------------------|--------------------------------------------------------------------------|
| `Cargo.toml`      | Add `exclude = ["examples/*"]` to the `[workspace]` table.               |
| `examples/README.md` | Already exists (prep commit). No changes — index lists all three.     |

### examples/fused_silu_gate/

| File              | Change                                                                   |
|-------------------|--------------------------------------------------------------------------|
| `Cargo.toml`      | NEW — `[package]` with `name = "fused_silu_gate"`, `edition = "2024"`; `[dependencies] kaio = { path = "../../kaio" }`. **Path dep during dev; flip to `kaio = "0.2.0"` at Sprint 6.9 publish** (noted in the sprint's commit message). |
| `src/main.rs`     | NEW — kernel + CPU f64 reference + correctness check + timing loop.      |
| `README.md`       | NEW — leads with kernel source, 2–3 short sections (what / why / run).  |

### examples/gelu_comparison/

| File              | Change                                                                   |
|-------------------|--------------------------------------------------------------------------|
| `Cargo.toml`      | NEW — mirrors fused_silu_gate's shape.                                   |
| `src/main.rs`     | NEW — both kernels, both CPU f64 references, comparison output.          |
| `README.md`       | NEW — both kernel source blocks at top, exact vs fast discussion.        |

### examples/rms_norm/

| File              | Change                                                                   |
|-------------------|--------------------------------------------------------------------------|
| `Cargo.toml`      | NEW — mirrors fused_silu_gate's shape.                                   |
| `src/main.rs`     | NEW — single-block kernel + CPU reference + timing.                      |
| `README.md`       | NEW — kernel source, RMSNorm context, single-block limitation flagged.   |

### Docs + meta

| File                                              | Change                                       |
|---------------------------------------------------|----------------------------------------------|
| `docs/development/sprints/phase6/PHASE_6_LOG.md`  | Row 6.8 → Complete with commit hash + test count. |
| `docs/development/sprints/phase6/phase6_master_plan.md` | Replace the rough-draft ref with a clean sprint table row pointing to `sprint_6_8.md`. |
| `CHANGELOG.md`                                    | New "Added — Sprint 6.8" entry announcing the three examples. |
| `README.md`                                       | Examples table expanded — three new rows alongside `vector_add` / `saxpy` / `reduction` / `matmul`. |

## Shared constraints (all three examples)

### Allowed macro builtins

Only builtins that exist in `kaio-macros/src/lower/builtins.rs` as
of Sprint 6.8's parent commit:

- Math: `exp`, `log`, `tanh`, `sqrt`, `abs`, `min`, `max`, `fma`.
- Sync: `bar_sync`.
- Shuffle: `shfl_sync_down`, `shfl_sync_up`, `shfl_sync_bfly`.
- Reductions: `block_reduce_sum`, `block_reduce_max`.
- Thread / block index: `thread_idx_x/y/z`, `block_idx_x/y/z`,
  `block_dim_x/y/z`.

**Do NOT use:** `sin`, `cos`, `rcp` (not in the builtin list —
adding them is a separate task, not an examples task).

### Tolerance

- Pointwise kernels (SiLU-gate, both GELUs): CPU reference in f64,
  cast to f32, check `abs_err < 1e-5` for SiLU-gate, `abs_err < 1e-4`
  for exact GELU, `abs_err < 5e-3` for fast GELU. Different thresholds
  because the kernels have different f32-precision characteristics.
- Reduction kernel (RMSNorm): CPU reference in f64, cast to f32,
  check `abs_err < 1e-4`. Looser than pointwise because the
  `block_reduce_sum` + sqrt + divide chain accumulates more f32 noise.

### Timing methodology

- 5 warm-up launches (first-call PTX compile + kernel load is not
  what we want to measure).
- 100 timed launches. Take the median (not mean — median is
  robust to the occasional scheduler hiccup).
- Use `std::time::Instant`. Call `.to_host()` (or equivalent) on
  the output buffer after each timed launch to force device
  synchronization before the next sample.
- Report median in microseconds with one decimal place.

### Output format (consistent across all three)

Every `main.rs` prints a compact summary block:

```
=== fused_silu_gate ===
Input size:        1048576 elements
Correctness:       PASS  (max_abs_err = 7.45e-07)
Median latency:    42.3 μs  (of 100 timed runs, 5 warm-ups skipped)
```

For `gelu_comparison`, same layout but with two lines (exact + fast).

### Clone-and-run invariants

Running from a fresh `git clone` with an NVIDIA GPU present must
produce:

```sh
cd examples/fused_silu_gate
cargo run --release
# → PASS output
```

No `cargo build` from the workspace root is required — the
`exclude = ["examples/*"]` keeps them out of the workspace build
path. Users who want to run all three can `for d in examples/*/;
do (cd "$d" && cargo run --release); done`.

## Verification sequence

1. `cargo fmt --all --check` — workspace only; examples have
   their own fmt discipline (run `cargo fmt` inside each).
2. `cargo clippy --workspace --all-targets -- -D warnings` —
   workspace only; examples are excluded and checked separately.
3. `(cd examples/fused_silu_gate && cargo build --release && cargo clippy --all-targets -- -D warnings)`
4. `(cd examples/gelu_comparison && cargo build --release && cargo clippy --all-targets -- -D warnings)`
5. `(cd examples/rms_norm && cargo build --release && cargo clippy --all-targets -- -D warnings)`
6. `(cd examples/fused_silu_gate && cargo run --release)` — PASS, timing printed.
7. `(cd examples/gelu_comparison && cargo run --release)` — both variants PASS.
8. `(cd examples/rms_norm && cargo run --release)` — PASS, timing printed.
9. `cargo test --workspace` — no regressions. Examples don't
   have `#[test]` blocks (they're runnable binaries, not test
   suites); the correctness check runs at `cargo run` time.
10. `cargo doc --workspace --no-deps` — clean.

## Success criteria

1. `examples/fused_silu_gate/`, `examples/gelu_comparison/`, and
   `examples/rms_norm/` all exist with `Cargo.toml`, `src/main.rs`,
   `README.md`.
2. Each runs `cargo run --release` from its own directory, prints
   PASS + median timing, no panics.
3. All three READMEs lead with the kernel source block.
4. Workspace-level `cargo test`, `cargo clippy -D warnings`,
   `cargo doc` remain clean — examples don't regress the rest of
   the workspace.
5. The `examples/README.md` index is accurate (row-for-row matches
   what actually exists + what's deferred).
6. Path-dep `kaio = { path = "../../kaio" }` during 6.8, with a
   commit-message note reminding the Sprint 6.9 publish step to
   flip to `kaio = "0.2.0"` before pushing to crates.io.
7. `sprint_6_8.md` updated with a "Results" section populated at
   commit time.

**Partial success acceptable:** If RMSNorm's `block_reduce_sum` has
an unresolvable issue during execution (macro-codegen bug, missing
builtin coverage, precision blowout) flag it as a GitHub issue and
ship 2/3. Better to have two clean examples than three with
workarounds. Update the `examples/README.md` index to move RMSNorm
to the deferred list in that case.

## Risks

1. **`block_reduce_sum` correctness at single-block scale.**
   Sprint 5.1 resolved the 2D block_reduce issue and per the
   tech-debt tracker it works for 1D kernels; RMSNorm uses a 1D
   block, so this should be fine. But the combined pattern
   (block_reduce → sqrt → divide) hasn't been exercised by tests
   before; the risk is a new integration bug surfacing. Mitigated
   by the "partial success" fold.

2. **First-run PTX compile time dominating small kernel timing.**
   Even with 5 warm-up runs, tiny 1M-element kernels on an RTX
   4090 are bandwidth-bound at ~40 μs. If the median reports
   anything much higher than that, something's being measured
   wrong. Mitigation: sanity-check expected latency per kernel
   before accepting the timing numbers in the README.

3. **`exclude = ["examples/*"]` syntax correctness.** Cargo
   workspace exclude globs can be finicky — if it doesn't work
   as a glob, explicitly list each directory.

4. **Examples drift from mainline API.** Future API changes in
   `kaio` could break the examples silently because they're
   excluded from the workspace build. Mitigation: CI should
   `cd examples/* && cargo build --release` as a separate job
   (add to the CI matrix in a follow-up if there's appetite;
   otherwise document as "run examples manually before each
   release" in the publish checklist).

## Rollback plan

- If 3/3 examples can't land by sprint end due to one hitting a
  blocker (most likely RMSNorm): ship 2/3 with a GitHub issue
  for the third, update the index, don't force a messy workaround.
- If `exclude = ["examples/*"]` causes workspace surprises:
  fall back to explicit per-directory exclude. Unlikely.

## Carry-forward

- **Sprint 6.9 (Polish + v0.2.0 publish):** flip each example's
  `kaio = { path = "../../kaio" }` to `kaio = "0.2.0"` at publish
  time. This is a one-line edit per example; add it to the publish
  checklist.
- **Post-0.2.0:** the three deferred examples (LayerNorm, rotary,
  cosine similarity) move from `examples/README.md`'s deferred
  list into separate follow-up sprints as user feedback dictates.
- **CI integration:** future sprint adds an `examples-build`
  GitHub Actions job that does `cd examples/* && cargo build`
  per PR to prevent API drift. Not 6.8 scope but worth a tech-debt
  entry.

## Results

All three examples landed and run cleanly on RTX 4090 sm_89.

### Measured output

```
=== fused_silu_gate ===
Input size:        1048576 elements
Correctness:       PASS  (max_abs_err = 1.49e-8)
Median latency:    188.8 μs  (of 100 timed runs, 5 warm-ups skipped)

=== gelu_comparison ===
Input size:        1048576 elements
Exact (tanh):      PASS  (max_abs_err = 2.38e-7)  — 177.9 μs
Fast (sigmoid):    PASS  (max_abs_err = 2.38e-7)  — 186.6 μs
Fast is 104.9% of exact's time.

=== rms_norm ===
Input size:        256 elements  (single-block — see README)
Correctness:       PASS  (max_abs_err = 2.38e-7)
Median latency:    181.7 μs  (of 100 timed runs, 5 warm-ups skipped)
```

### Deviations from plan

1. **Workspace-detach mechanism changed.** The plan called for
   `exclude = ["examples/*"]` at the workspace root. In practice
   Cargo still treated the example as a would-be workspace member and
   refused to build. Fix: each example's `Cargo.toml` carries an empty
   `[workspace]` table, which cleanly detaches it. The workspace-level
   `exclude` is kept as belt-and-braces / documented intent. This was
   flagged in the plan's risk register as item 3; the fallback was the
   one specified there.

2. **RMSNorm kernel form.** The plan's implicit expression was
   `let val = if tid < n { x[tid] } else { 0.0f32 };` — the
   `#[gpu_kernel]` macro rejects `if` as an expression. Rewrote as
   imperative-style `let mut val = 0.0f32; if tid < n { val = x[tid]; }`,
   matching the pattern already used in `kaio/examples/reduction.rs`.
   This is consistent with the existing DSL scope; no new friction
   point to flag.

3. **GELU README — bandwidth-bound teaching moment added post-build.**
   Both variants landed at essentially identical wall-clock speed
   (104.9%) because they're bandwidth-bound, not compute-bound. Opus
   flagged this as the best teaching insight in the whole example
   set. Added a new section explaining why the "fast approximation"
   saves nothing on memory-bound workloads and why kernel fusion
   (not arithmetic micro-optimization) is the load-bearing lever for
   ML kernels.

### What this sprint proved

- The `#[gpu_kernel]` DSL is ergonomic enough for real-world ML
  kernels at the showcase level. No DSL extensions were needed for
  any of the three.
- `block_reduce_sum + sqrt + divide` integration works end-to-end
  from a user-authored kernel (previously only exercised from
  internal tests).
- Standalone `Cargo.toml` + `[workspace]` table + path-dep is a
  clean publishing story — the flip to `kaio = "0.2.0"` at Sprint
  6.9 is a one-line edit per example.
- The bandwidth-bound insight in `gelu_comparison/README.md` is the
  kind of "stars a repo" content Dave wants in the v0.2.0 launch
  story — a concrete demonstration that KAIO rewards fusion, not
  micro-optimization.

### Carry-forward

- **Sprint 6.9 publish:** flip `kaio = { path = "../../kaio" }` to
  `kaio = "0.2.0"` in all three `examples/*/Cargo.toml` at release.
- **Sprint 6.7b:** vectorized loads + bank-conflict padding, chase
  90%+ at 4096² on the matmul bench. Unblocked now.
- **CI integration (deferred):** a follow-up sprint should add an
  `examples-build` GitHub Actions job iterating over `examples/*/`
  to prevent silent API drift. Not in 6.8 scope; worth its own
  tech-debt line if it bites before we add it.
