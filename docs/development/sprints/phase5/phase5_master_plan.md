# Phase 5 Master Plan — Fused Attention & Community Release

**Status:** Planning
**Depends on:** Phase 4 complete (v0.0.4, commit `5cc2ca0`)

## Goal

Implement fused multi-head attention through KAIO's `#[gpu_kernel]`
macro, add it to `kaio-ops`, build an auto-tuning framework, and
ship v0.1.0 to crates.io with a community announcement.

This is the phase where KAIO transitions from "can do GPU kernels"
to "can do real ML primitives."

Target user experience:

```rust
use kaio_ops::attention;
use kaio::prelude::*;

let device = KaioDevice::new(0)?;
let q = device.alloc_from(&q_data)?;  // (batch, heads, seq_len, d_k)
let k = device.alloc_from(&k_data)?;
let v = device.alloc_from(&v_data)?;
let mut out = device.alloc_zeros::<f32>(batch * heads * seq_len * d_k)?;
attention(&device, &q, &k, &v, &mut out, batch, heads, seq_len, d_k)?;
```

## Key Architectural Decisions

### 1. 2D Block Reductions (Sprint 5.1 — HARD GATE)

Current: `block_reduce_sum/max` derive thread identity from
`thread_idx_x()` only. In 2D kernels, multiple rows alias the same
warp slots. Rejected at compile time.

Fix: compute linear tid = `thread_idx_x + block_dim_x * thread_idx_y`.
Pass linear tid through the reduction tree instead of raw
`thread_idx_x`. Affects `kaio-macros/src/lower/builtins.rs` lines
62-68.

**This is a core DSL change, not just a fix.** It changes a
foundational assumption (reductions assume 1D identity). Must verify
zero regression in all existing 1D reduction tests (softmax,
reduce_macro, shared_mem+reduce). Test both 1D and 2D reductions
side-by-side.

This unblocks every kernel that combines 2D blocks with reductions —
including attention.

### 2. Attention Strategy: Standard First, Then FlashAttention

Sprint 5.2-5.3: Standard attention — materialize full Q*K^T attention
matrix, apply mask, explicit softmax, multiply by V. Simple, testable,
correct baseline. Performance won't be great (O(n^2) memory) but
validates the algorithm and DSL expressiveness. **Treat as DSL stress
test** — document every friction point for Phase 6.

Sprint 5.4: FlashAttention — online softmax, tiled Q/K/V loads, no
full attention matrix materialization. O(n) memory. This is the
performance version.

**Why not jump straight to FlashAttention?** It's algorithmically
complex (online softmax with running max/sum correction). Having a
correct standard attention baseline lets us validate FlashAttention
output against it. Same pattern as Phase 4: naive matmul first (4.3),
optimized second (4.6).

**FlashAttention is a stretch goal for v0.1.0.** If it's not stable
by sprint end, Phase 5 still ships with standard attention. Standard
attention is the floor, not FlashAttention.

### 3. Auto-Tuner Scope

Not a general-purpose framework — a targeted block-size / tile-size
grid search for attention and matmul. Compile N kernel variants with
different configs, benchmark each, select best. Store results as a
simple JSON cache keyed by (kernel, GPU SM target, problem_size).
Cache key must include SM target — an sm_70 result is invalid for
sm_89.

**Scope lock:** grid search over fixed parameter sets only. No
heuristics, no adaptive logic, no dynamic tuning. Keep it dumb and
reliable. If scope creeps past this, stop and re-scope.

### 4. Validation Target

Attention output validated against CPU reference implementation.
No PyTorch dependency — too heavy. CPU reference is straightforward:
standard matmul + softmax + matmul, all in f32.

Tolerance: < 1e-3 absolute error, with per-element scaling for
larger values. Chained matmuls accumulate floating-point error —
at seq_len 512+ the tolerance must scale with magnitude, not be a
fixed constant. Don't chase phantom numerical bugs.

### 5. kaio-ops API

```rust
use kaio_ops::attention;

// Q: (batch, heads, seq_len, d_k), row-major f32
// K, V: same layout
// out: (batch, heads, seq_len, d_v)
attention(&device, &q, &k, &v, &mut out, batch, heads, seq_len, d_k)?;
```

Single-head first (Sprint 5.2-5.3), multi-head second (Sprint 5.4+).

**Ergonomics note:** 5 positional `u32` args in a row is an ergonomic
footgun. If the parameter list grows during implementation, introduce
an `AttentionConfig` struct. Don't prematurely abstract — only if we
hit > 5 params.

## Sprint Breakdown

| Sprint | Scope | Key Deliverable |
|--------|-------|-----------------|
| 5.1 | Tech debt: 2D reductions + DSL fixes | block_reduce in 2D kernels (HARD GATE). `&&`/`\|\|` operators and codegen regression tests are stretch goals — do not block 5.2 on them |
| 5.2 | Standard attention — forward pass | Q*K^T -> scale -> softmax -> *V, single-head, materialized. Treat as **DSL stress test** — document friction points |
| 5.3 | Masking + validation | Causal mask, padding mask, CPU reference validation suite. Deliverable includes: documented list of DSL friction points from 5.2-5.3 |
| 5.4 | FlashAttention — online softmax | Tiled attention, O(n) memory. Internally split: (a) online softmax validated in isolation, then (b) full tiled attention with output rescaling. **Stretch goal** — Phase 5 ships without it if unstable |
| 5.5 | Auto-tuner | Block/tile size grid search, benchmark runner, JSON cache |
| 5.6 | CI/CD + platform | Windows CI, Linux verification, GitHub Actions matrix |
| 5.7 | v0.1.0 prep | API stability review, cargo doc, publish dry-run |
| 5.8 | Community launch | Blog post, r/rust announcement, benchmark page |

## Dependency Graph

```
5.1 (2D reductions) -> 5.2 (standard attention) -> 5.3 (masking + validation)
                                                         |
                                                    5.4 (FlashAttention — stretch)
                                                         |
                                                    5.5 (auto-tuner)

5.6 (CI/CD) -- start immediately after 5.1. Attention kernels will
              introduce platform-specific issues. Catching Windows
              breaks early saves pain later.
5.7 (v0.1.0) -- depends on 5.1-5.6
5.8 (launch) -- depends on 5.7
```

## Tech Debt Addressed in Phase 5

**Resolved in 5.1:**
- block_reduce 2D support (high priority, HARD GATE for 5.2)
- `&&` / `||` logical operators (stretch goal — nice for attention
  verbosity but nested `if` works as fallback)
- Host-level codegen regression tests (stretch goal — CI hygiene)

**Deferred past Phase 5:**
- fma() f64 (no f64 attention kernels planned)
- Shared memory base register hoisting (perf, not correctness)
- Compound shared memory assignment (workaround exists)
- ArithOp::Shr optimization for reductions (minor perf)
- Matmul vectorized loads / double buffering (Phase 6)

## Success Criteria

1. `kaio_ops::attention()` produces correct output for single-head
   and multi-head attention (validated against CPU reference,
   < 1e-3 absolute error with magnitude-scaled tolerance)
2. Causal masking works correctly
3. FlashAttention variant uses O(seq_len) memory — **stretch goal**.
   Phase 5 ships with standard attention if FlashAttention is not
   stable
4. Auto-tuner selects optimal block size for at least attention
   and matmul
5. All 5 crates published at v0.1.0
6. CI runs on both Windows and Linux
7. Blog post and r/rust announcement published
8. DSL friction points from attention implementation documented
   (feeds Phase 6 DSL improvements)

## Key Risks

1. **FlashAttention complexity** — this is the hardest sprint in
   the entire project. Online softmax with running max/sum
   correction, tiled Q/K/V loads, output rescaling — Triton
   tutorials for this are 200+ lines of dense code. Mitigation:
   (a) standard attention baseline validates correctness,
   (b) Sprint 5.4 is internally split — online softmax validated
   in isolation before full tiled attention is attempted,
   (c) FlashAttention is explicitly a stretch goal.
2. **Register pressure** — attention kernels with Q/K/V tiles
   in registers could exceed GPU limits. Mitigation: start with
   small tile sizes, add KAIO_PTX_STATS monitoring.
3. **Performance expectations** — users may compare against
   FlashAttention-2 in CUDA (which uses tensor cores, async
   copies). Mitigation: document clearly that KAIO uses scalar
   PTX, not tensor core ops.
4. **Sprint 5.2/5.3 DSL friction** — standard attention will
   expose DSL weaknesses (indexing complexity, nested loops,
   memory pressure, verbosity). This is expected and valuable.
   Treat as a stress test, document findings for Phase 6.

## Review Context

- Phase 4 delivered: v0.0.4, 5 crates, 207 host + 43 GPU tests,
  matmul at 31% cuBLAS, PTX inspection tools, 4 runnable examples
- Sprint 4.9 (adoption polish): README rewrite, examples, patterns,
  limitations, feedback CTA
- Adoption review identified 7 barriers, all addressed in 4.9
- 2D block reduction is the only hard DSL blocker for attention
- Planning review: treat 5.1 as core DSL change, not just a fix;
  FlashAttention should not block v0.1.0; start CI early
- Adversarial review: treat 5.2/5.3 as DSL stress test; lock
  auto-tuner scope; standard attention is the real deliverable
