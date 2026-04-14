# Sprint 7.1 — INT8 dequantize-matmul (Phase 7's quant headline)

**Status:** In progress
**Branch:** `sprint-7-1` off `main` (post v0.2.2 publish)
**Release target:** v0.3.0 — first minor bump since Phase 6; new public op

## Context

Phase 7.1 is the quant-kernels headline that attracts new users. The DSL
completeness work shipped in Sprint 7.0 (bitwise ops, signed/unsigned shift
preservation, compound bitwise assign) was explicitly scoped to unlock dequant
work. Sprint 7.0.5 shipped the ergonomics surface so cold-start adopters don't
bounce on friction before reaching quant.

The rust-lang.org forum engagement is also relevant: the first external
feature request was explicitly about 4/5/6-bit quantization. The v0.2.2 reply
committed publicly to shipping **INT8 dequantize-matmul as the reference
template**, with the DSL supporting custom bit-width variants beyond that.
Sprint 7.1 delivers the INT8 half of that promise.

Dequant-only (DSL example) is already shipped at `examples/int8_dequant/`.
This sprint delivers the **matmul-fusion** half — the IR-authored
`kaio_ops::matmul_int8` — because reading packed INT8 into registers,
unpacking + dequantizing in a separate pass, then feeding f32 into matmul
destroys the point of quantization. Dequant must fuse with the tensor-core
inner loop: unpack → dequant → feed mma.sync in registers, no round-trip
through shared or global.

## Public contract for v0.3.0 `matmul_int8`

This paragraph is the reference for what 7.1 actually ships. It appears
verbatim in the `matmul_int8` rustdoc, the example README, and this log so
users, reviewers, and future sprints share the same expectations:

**`matmul_int8` in v0.3.0 is:**
- **symmetric** (no zero point)
- **int8 × int8 → f32** (both operands quantized; W8A8 only)
- **single global scalar scale** (one `f32` applied to the full output)
- **sync-only** (async INT8 matmul deferred to 7.1.5+)
- **`K % 32 == 0` required** (plus M/N constraints per fragment shape)
- **the first reference quant op, not the final general quant architecture**
  — GPTQ, AWQ, per-channel, per-group, asymmetric, INT4, W8A16 all come
  later as additive refinements, not as unmet expectations

## The primary unknown

**Does `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` behave as expected
on RTX 4090 (sm_89)?** D1 spike is specifically structured to answer this.
If the fast path is viable, INT8 flows straight into tensor-core; if not,
fall back to INT8 → f16 → existing `m16n8k16.f16.f16.f32` path. Either way
the sprint ships.

## Deliverables

D1 — `mma.sync.s8.s8.s32` spike (fork-decision gate, split into D1a encoding
viability + D1b layout correctness). Pending.

D2 — Full IR extensions for chosen path. Pending.

D3 — Fused dequant-matmul kernel in `kaio-ops` (IR-authored, sync-only, Path
FAST committed). Pending.

D4 — Public `matmul_int8` surface (direct-call only, K%32==0 validation,
W8A8 rustdoc warning). Pending.

D5 — Showcase example `examples/int8_matmul/` + xtask wiring. Pending.

D6 — Tests, docs, CHANGELOG, v0.3.0 release. Pending.

Full spec: see plan file at
`C:\Users\david\.claude\plans\scalable-giggling-cray.md` (planning-only doc,
not part of the repo).

## Review trail

| Round | Reviewer | Status |
|---|---|---|
| 1 | Owner + pre-plan Opus 4.6 | ✅ |
| 2 | Opus 4.6 full-plan | ✅ |
| 3 | Codex 5.4 | ✅ |
| 3.5 | Gemini 3.1 Pro (supplemental) | ✅ |
| 4 | Owner final sign-off | ✅ |

## Results

_To be filled in at sprint completion._
