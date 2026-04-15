# quantized_attention — end-to-end Phase 7 pipeline

Sprint 7.3 D9 showcase. Demonstrates the full Phase 7 quantization pipeline
shipping on `phase7-rest`:

```text
X [seq, d_model] f16
 │
 ├─► INT4 PATH:  qkv_project_int4 (fused) ──► Q, K, V f16 [seq, d_head]
 │                                                      │
 │                                                      ▼
 │                                            attention_tc ──► out_i4 f32 [seq, d_v]
 │
 └─► F16 REF :   3 × matmul_tc (one per proj) ─► Q', K', V' f16 [seq, d_head]
                                                      │
                                                      ▼
                                            attention_tc ──► out_ref f32 [seq, d_v]
```

## Run

```sh
cargo run --release
```

## What it reports

Three quality metrics on the final attention output vs the f16 reference:

- **Cosine similarity** — primary pass/fail. Plan D9 INT4 threshold: ≥ 0.98.
- **Max absolute error** — worst-case row outlier.
- **Mean relative error** — aggregate quality.

Also reports projection-stage cosine similarity (Q/K/V vs f16 reference)
for localization — if the final number is bad, this helps distinguish
whether quantization error is introduced at the projection step or
amplified by softmax.

## Pipeline honesty

Random f16 weights are a **worst case** for group-scale quantization
fidelity. Real trained LLM weights have much tighter group statistics
and land measurably better than these synthetic numbers. This example
demonstrates the *pipeline plumbing*, not target accuracy for any
specific quantization recipe.

## Why INT4 (not INT8)

Per the Sprint 7.3 plan D9: "INT4 is the more impressive demo; falls
back to INT8 in the README prose if INT4 is deferred." INT4 shipped, so
this example uses INT4. Swapping to INT8 is a ~15-line change: replace
`qkv_project_int4` with `qkv_project_int8`, drop the packing + scales
helpers, pass a scalar per-projection scale instead of a scales tensor.

## Shape

- `SEQ = 64` — sequence length / M for the projection matmul.
- `D_MODEL = 128` — input dim / K. Must be a multiple of `GROUP_SIZE = 128`.
- `D_HEAD = 64` — per-head dim / N. Must be even for the store-out path.

Tuning knobs are at the top of `src/main.rs`.

## Performance framing

The fused `qkv_project_int4` path wins **2.5–3.4× over three separate
`matmul_int4` calls at decode shapes (M ≤ 64)**, ties at mid prefill
(M=512), and slightly loses (~0.85×) at the largest prefill (M=2048).
See `docs/development/sprints/phase7/sprint_7_3.md` for the full bench
table. The showcase uses decode-scale shapes by default where fusion
wins cleanly.
