# Phase 3 Master Plan: Loops, Reductions & Softmax

**Status:** Planning
**Date:** 2026-04-11
**Depends on:** Phase 2 complete (commit `50a3ab0`)

---

## 1. Goal

Add the control flow and synchronization primitives needed for real GPU
algorithms. Users write loops, use shared memory, synchronize threads, and
call block-level reduction operations — culminating in a softmax kernel
validated against PyTorch.

```rust
use kaio::prelude::*;

#[gpu_kernel(block_size = 256)]
fn softmax_row(input: &[f32], output: &mut [f32], row_len: u32) {
    let tid = thread_idx_x();
    let bsize = block_dim_x();
    let row_offset = block_idx_x() * row_len;

    // Phase 3 features: loops, shared memory, reductions
    let sdata = shared_mem![f32; 256];
    let mut local_max = -3.402823e+38f32; // -FLT_MAX

    let mut i = tid;
    while i < row_len {
        let val = input[row_offset + i];
        if val > local_max { local_max = val; }
        i += bsize;
    }

    let row_max = block_reduce_max(local_max);

    let mut local_sum = 0.0f32;
    let mut i = tid;
    while i < row_len {
        local_sum += exp(input[row_offset + i] - row_max);
        i += bsize;
    }

    let row_sum = block_reduce_sum(local_sum);

    let mut i = tid;
    while i < row_len {
        output[row_offset + i] = exp(input[row_offset + i] - row_max) / row_sum;
        i += bsize;
    }
}
```

Phase 3 delivers: `for`/`while` loop support, shared memory (`shared_mem!`),
`bar_sync()`, warp shuffle (`shfl_sync_down/up/bfly`), block-level reductions
(`block_reduce_sum`, `block_reduce_max`), and softmax validated against PyTorch
at three matrix sizes.

---

## 2. Architecture Decisions

### 2.1 Compound assignment desugaring (`+=`, `-=`, `*=`, `/=`)

**Context:** The existing parser handles `Expr::Assign` (simple `=`) but NOT
`Expr::AssignOp` (compound assignment). Loops need `acc += value` and
`i += stride`. This is a prerequisite for Sprint 3.1.

**Decision:** Desugar compound assignment at parse time. When the parser
encounters `x += expr`, emit `KernelStmt::Assign { name: "x", value:
BinOp(Var("x"), Add, expr) }`. Same for `-=` (Sub), `*=` (Mul), `/=` (Div).
Array compound assignment (`arr[i] += expr`) desugars to
`KernelStmt::IndexAssign { array, index, value: BinOp(Index(array, index),
Add, expr) }`.

This reuses all existing lowering infrastructure — no new IR variants needed.

### 2.2 Loop lowering strategy

**Context:** `for i in start..end` and `while cond { body }` are the two
loop constructs needed. `loop`, `break`, `continue` remain unsupported.

**Decision:** Lower loops to the standard PTX branch + counter pattern:

```
for i in start..end:
  mov %counter, start
  LOOP_START_N:
  setp.ge.s32 %p, %counter, end
  @%p bra LOOP_END_N;
  // body (with %counter registered as 'i' in locals)
  add.s32 %counter, %counter, 1;
  bra LOOP_START_N;
  LOOP_END_N:

while condition:
  LOOP_START_N:
  // evaluate condition → %pred
  @!%pred bra LOOP_END_N;
  // body
  bra LOOP_START_N;
  LOOP_END_N:
```

The `label_counter` in `LoweringContext` is already monotonic, so nested
loops get unique labels with no collision.

**Supported `for` patterns:** Only `for ident in expr..expr` where both
bounds are integer expressions. `step_by()`, `break`, `continue`, reverse
ranges, inclusive ranges (`..=`) are deferred.

**`loop` remains rejected:** Without `break`/`continue`, bare `loop` is an
infinite loop. Keep the compile error but update the message to remove the
"available in Phase 3" text.

### 2.3 Shared memory model

**Context:** GPU shared memory is block-scoped SRAM accessible by all
threads in a block. PTX declares it in the kernel preamble and accesses
it with `ld.shared` / `st.shared`.

**Decision:** Expose via `shared_mem![T; N]` macro syntax inside kernels.

- **Declaration:** `let sdata = shared_mem![f32; 256];` — parsed as
  `Expr::Macro` in syn with path `shared_mem`, manually destructured to
  extract element type and count.
- **IR variant:** `KernelStmt::SharedDecl { name, elem_ty, count }`.
- **PTX emission:** `.shared .align A .b8 name[N*sizeof(T)];` in kernel
  preamble (new `shared_decls` field on `PtxKernel`).
- **Access:** `sdata[idx]` dispatches to `ld.shared` / `st.shared` instead
  of `ld.global` / `st.global`. The lowering context tracks shared arrays
  in a new `shared_arrays: HashMap<String, (KernelType, usize)>` field,
  separate from `locals`.
- **Addressing:** 32-bit offsets (`mul.lo.s32` for byte offset, `add.s32`
  for base) — no `cvta.to.global` needed for shared memory.
- **Compile-time reporting:** emit `eprintln!` diagnostic with shared memory
  usage in bytes. Emit a warning if total exceeds 48 KB (SM 8.9 default).

### 2.4 Shuffle instruction encoding

**Context:** The PTX `shfl.sync` instruction has a packed `c` operand that
encodes both the width and the clamp mode. This is the biggest technical
risk in Phase 3 — getting it wrong produces silent data corruption.

**Decision:** Read PTX ISA 8.7 Section 9.7.8 (Warp Shuffle Instructions)
before writing any emitter code. The `c` operand for `shfl.sync.down` is:
`((32 - width) << 8) | 0x1F` for clamp mode. Validate emitted instructions
against `ptxas --verify` before writing any tests.

**User-facing API:**
```rust
let down = shfl_sync_down(val, delta, width);  // implicit mask 0xFFFFFFFF
let up   = shfl_sync_up(val, delta, width);
let xor  = shfl_sync_bfly(val, lane_mask, width);
```

Three arguments, implicit full-warp mask. Explicit mask is deferred to a
future phase. Return type matches input value type (f32 → f32, u32 → u32).

### 2.5 Reduction auto-allocated shared memory

**Context:** `block_reduce_sum(val)` / `block_reduce_max(val)` are multi-
instruction expansions that need shared memory for cross-warp communication.
This shared memory is separate from any user-declared `shared_mem!` arrays.

**Decision:** Reductions auto-allocate shared memory with the naming
convention `_kaio_reduce_smem_N` (where N is a counter). This is tracked
separately in `LoweringContext` via an `auto_shared: Vec<SharedDecl>` field
and merged into `PtxKernel::shared_decls` during codegen. The leading
underscore prevents collision with user-declared names (which come from
Rust identifiers — always valid Rust idents, never starting with `_kaio_`).

Size: one `f32` or `u32` per warp (32 threads). For block_size=1024, that's
32 warps = 128 bytes. Negligible.

**Reduction algorithm:**
1. Warp-level tree reduction: 5 rounds of `shfl.sync.down` (delta 16, 8,
   4, 2, 1) + add/max
2. Thread 0 of each warp writes result to `_kaio_reduce_smem[warp_id]`
3. `bar.sync 0`
4. First warp loads from shared memory and performs another warp-level
   tree reduction
5. Result valid in thread 0 of the block

**Return semantic:** Result is valid in ALL threads (we broadcast via shared
memory + bar.sync after reduction completes). This is more ergonomic for
softmax where every thread needs `row_max` and `row_sum`.

---

## 3. New kaio-core Instructions

### MemoryOp additions

```rust
LdShared { dst: Register, addr: Register, ty: PtxType }
  → ld.shared.f32 %f0, [%r0];

StShared { addr: Register, src: Register, ty: PtxType }
  → st.shared.f32 [%r0], %f1;
```

### ControlOp additions

```rust
BarSync { barrier_id: u32 }
  → bar.sync 0;

ShflSyncDown { dst: Register, src: Register, delta: Operand, c: u32, mask: u32, ty: PtxType }
  → shfl.sync.down.b32 %r0, %r1, delta, c, 0xFFFFFFFF;

ShflSyncUp { dst: Register, src: Register, delta: Operand, c: u32, mask: u32, ty: PtxType }
  → shfl.sync.up.b32 %r0, %r1, delta, c, 0xFFFFFFFF;

ShflSyncBfly { dst: Register, src: Register, lane_mask: Operand, c: u32, mask: u32, ty: PtxType }
  → shfl.sync.bfly.b32 %r0, %r1, lane_mask, c, 0xFFFFFFFF;
```

Note: `c` is the pre-packed clamp/width operand per PTX ISA 8.7 Section 9.7.8.
The user-facing `width` parameter is converted to the packed `c` value during
lowering, not in kaio-core.

### PtxKernel additions

```rust
pub struct PtxKernel {
    // existing fields...
    pub shared_decls: Vec<SharedDecl>,
}

pub struct SharedDecl {
    pub name: String,
    pub align: u32,
    pub size_bytes: u32,
}
```

Emitted in kernel preamble after register declarations:
```ptx
.shared .align 4 .b8 sdata[1024];
```

---

## 4. New kaio-macros IR Variants

### KernelStmt additions

```rust
For {
    var: String,
    start: KernelExpr,
    end: KernelExpr,
    body: Vec<KernelStmt>,
}

While {
    condition: KernelExpr,
    body: Vec<KernelStmt>,
}

SharedDecl {
    name: String,
    elem_ty: KernelType,
    count: usize,
}
```

### New built-in functions

| Function | Arguments | Return | PTX |
|----------|-----------|--------|-----|
| `bar_sync()` | none | `()` | `bar.sync 0;` |
| `shfl_sync_down(val, delta, width)` | (f32/u32, u32, u32) | same as val | `shfl.sync.down.b32` |
| `shfl_sync_up(val, delta, width)` | (f32/u32, u32, u32) | same as val | `shfl.sync.up.b32` |
| `shfl_sync_bfly(val, mask, width)` | (f32/u32, u32, u32) | same as val | `shfl.sync.bfly.b32` |
| `block_reduce_sum(val)` | (f32) | f32 | multi-instruction |
| `block_reduce_max(val)` | (f32) | f32 | multi-instruction |

---

## 5. Sprint Plan

### Sprint 3.1: Loops + Compound Assignment
**Scope:** `for`/`while` loops end-to-end, `+=`/`-=`/`*=`/`/=` desugaring
**Layer:** kaio-macros only

Parse changes:
- Add `For`, `While` variants to `KernelStmt` (`kernel_ir/stmt.rs`)
- Parse `Expr::ForLoop` in `parse/body.rs`: extract `Pat::Ident`, validate
  `Expr::Range` with `start..end`, recurse body
- Parse `Expr::While` in `parse/body.rs`: recurse condition and body
- Remove for/while rejection (lines 238-245). Keep `loop` rejection,
  update message to remove "available in Phase 3"
- Parse `Expr::AssignOp` in `parse_stmt_from_expr()`: desugar `x += e`
  into `Assign { name, value: BinOp(Var(name), op, e) }` and `arr[i] += e`
  into `IndexAssign { array, index, value: BinOp(Index(array, i), op, e) }`

Lower changes:
- `lower_stmt` for `For`: lower start/end → init counter register → emit
  LOOP_START label → setp.ge → BraPred → register counter var in locals →
  lower body → add counter, 1 → Bra LOOP_START → LOOP_END label
- `lower_stmt` for `While`: emit LOOP_START → lower condition → BraPred
  to LOOP_END → lower body → Bra LOOP_START → LOOP_END label
- No changes needed for compound assignment (desugared at parse time)

Tests:
- Host: parse round-trip for for/while AST nodes, compound assignment
  desugaring
- Host: PTX emission contains loop labels and branch instructions
- GPU: `sum_0_to_n` kernel (for loop summing 0..N, verify against N*(N-1)/2)
- GPU: `while_converge` kernel (iterative halving until < threshold)
- GPU: `strided_sum` kernel (while loop with `i += stride`, verifies
  compound assignment works)
- Update CF09: keep `loop {}` rejection, update stderr to remove Phase 3 ref
- New CF: `for i in vec.iter()` → "only `start..end` ranges supported"

Dependencies: None
Risk: syn's `ExprForLoop` wraps `Pat` and `Expr` — need to destructure
`Pat::Ident` and `Expr::Range { start, end, limits: Half }` carefully.

---

### Sprint 3.2: Shared Memory + Barrier + Shuffle Instructions (kaio-core)
**Scope:** New PTX instruction variants in kaio-core only
**Layer:** kaio-core only

New instruction variants:
- `MemoryOp::LdShared { dst, addr, ty }` → `ld.shared.{ty} dst, [addr];`
- `MemoryOp::StShared { addr, src, ty }` → `st.shared.{ty} [addr], src;`
- `ControlOp::BarSync { barrier_id }` → `bar.sync {id};`
- `ControlOp::ShflSyncDown { dst, src, delta, c, mask, ty }` → `shfl.sync.down.b32 dst, src, delta, c, mask;`
- `ControlOp::ShflSyncUp { dst, src, delta, c, mask, ty }` → `shfl.sync.up.b32 ...`
- `ControlOp::ShflSyncBfly { dst, src, lane_mask, c, mask, ty }` → `shfl.sync.bfly.b32 ...`

PtxKernel changes:
- Add `shared_decls: Vec<SharedDecl>` field
- Emit `.shared .align A .b8 name[size];` after register declarations

**Critical:** Read PTX ISA 8.7 Section 9.7.8 (Warp Shuffle Instructions)
before writing any shfl emitter code. The `c` operand packs width and clamp
mode. Validate every emitted shfl instruction against `ptxas --verify` before
writing any tests. Getting this wrong produces silent lane-read corruption.

Tests:
- Unit: each new instruction variant emits correct PTX string
- Unit: shared decl appears in correct position in kernel preamble
- Unit: shfl.sync with various c values emits correct operand encoding
- Integration: hand-built shared memory kernel passes `ptxas --verify`
- Integration: hand-built shfl.sync kernel passes `ptxas --verify`

Dependencies: None (parallel with 3.1)
Risk: shfl.sync operand encoding — highest technical risk in Phase 3.

---

### Sprint 3.3: Shared Memory in Macro DSL
**Scope:** `shared_mem![T; N]` syntax, ld.shared/st.shared dispatch
**Layer:** kaio-macros

Parse changes:
- Add `SharedDecl` variant to `KernelStmt`
- Detect `Expr::Macro` with path `shared_mem` in `parse_stmt_from_expr`
- Manually parse inner token stream: expect `Type ; LitInt`
- Validate: T must be a supported scalar type, N must be a positive literal

Lower changes:
- Add `shared_arrays: HashMap<String, (KernelType, usize)>` to `LoweringContext`
- `lower_stmt` for `SharedDecl`: register in `shared_arrays`, emit
  `SharedDecl` to be collected into `PtxKernel::shared_decls`
- Modify `memory.rs`: when array name is in `shared_arrays`, use
  `ld.shared` / `st.shared` with 32-bit addressing (no cvta)
- Add `gpu_builtins` stub for `shared_mem!` macro (IDE autocomplete)

Codegen changes:
- `generate_build_ptx()` must collect shared decls and call
  `kernel.add_shared_decl()` before emission
- Compile-time reporting: `eprintln!` with total shared memory bytes
- Warning if total > 48 KB

Tests:
- Host: parse `shared_mem![f32; 256]` → correct SharedDecl
- Host: PTX contains `.shared` directive in kernel preamble
- GPU: write to shared, `bar_sync()`, read back — verify data survived
- CF: `shared_mem![String; 256]` → unsupported type
- CF: `shared_mem![f32; 0]` → size must be positive

Dependencies: 3.1 (loops needed for iteration over shared arrays), 3.2
(LdShared/StShared/BarSync instructions)

---

### Sprint 3.4: Barrier + Shuffle Built-in Functions
**Scope:** `bar_sync()` and `shfl_sync_*` as built-in functions
**Layer:** kaio-macros

Builtin additions (in `lower/builtins.rs`):
- `bar_sync()` → zero args, emits `ControlOp::BarSync { barrier_id: 0 }`.
  Handle as statement-expression (no meaningful return value).
- `shfl_sync_down(val, delta, width)` → three args, compute packed `c`
  value from `width`, emit `ControlOp::ShflSyncDown`. Return type matches
  `val` type (f32 → f32, u32 → u32).
- `shfl_sync_up(val, delta, width)` → same pattern
- `shfl_sync_bfly(val, lane_mask, width)` → same pattern
- Width-to-c packing: `c = ((32 - width) << 8) | 0x1F` for down/bfly,
  `c = width << 8` for up (per PTX ISA 8.7)

Update:
- `gpu_builtins` stubs for IDE autocomplete
- Error message list in `lower_builtin()` to include new functions

Tests:
- Host: parse and lower each new builtin
- GPU: `bar_sync` between shared memory writes and reads — producer-consumer
  pattern (success criterion 3.4)
- GPU: `shfl_sync_down` — each thread reads neighbor's value, verify
  lane mapping is correct
- GPU: `shfl_sync_bfly` — XOR exchange between lane pairs
- CF: `bar_sync(1, 2)` → wrong number of arguments
- CF: `shfl_sync_down(val, delta)` → wrong number of arguments

Dependencies: 3.2 (instructions), 3.3 (shared memory for bar_sync testing)

---

### Sprint 3.5: block_reduce_sum + block_reduce_max
**Scope:** Reduction primitives as multi-instruction built-in expansions
**Layer:** kaio-macros

Builtin additions:
- `block_reduce_sum(val: f32) -> f32` in `lower/builtins.rs`
- `block_reduce_max(val: f32) -> f32` in `lower/builtins.rs`

Multi-instruction expansion (per reduction call):
```
// Phase 1: Warp-level tree reduction (5 rounds)
shfl.sync.down val, val, 16, c, 0xFFFFFFFF
add.f32 val, val, shfl_result       // (or max.f32 for reduce_max)
shfl.sync.down val, val, 8, c, 0xFFFFFFFF
add.f32 val, val, shfl_result
// ... delta = 4, 2, 1

// Phase 2: Cross-warp via shared memory
// Thread 0 of each warp: st.shared warp_result to _kaio_reduce_smem[warp_id]
bar.sync 0

// Phase 3: First warp reduces across warps
// Thread tid < num_warps: ld.shared from _kaio_reduce_smem[tid]
// Warp-level tree reduction again
// Thread 0 writes final result to _kaio_reduce_smem[0]
bar.sync 0

// Phase 4: Broadcast — all threads read result from _kaio_reduce_smem[0]
ld.shared result, [_kaio_reduce_smem]
```

Auto-allocated shared memory:
- Naming convention: `_kaio_reduce_smem_N` (N = auto-increment counter)
- Tracked in `LoweringContext::auto_shared: Vec<SharedDecl>`
- Merged into `PtxKernel::shared_decls` during codegen alongside user
  shared memory declarations
- Size: `ceil(block_size / 32) * sizeof(f32)` — one element per warp
- Leading `_kaio_` prefix prevents collision with user names

Return semantic: Result broadcast to ALL threads via shared memory read
after final `bar.sync`. Every thread can use the reduction result.

Tests:
- GPU: `block_reduce_sum` with known values (e.g., each thread contributes
  tid+1, expected sum = block_size * (block_size+1) / 2)
- GPU: `block_reduce_max` with known values (each thread contributes tid,
  expected max = block_size - 1)
- GPU: verify ALL threads see the result (not just thread 0)
- GPU: edge cases — block_size=32 (single warp, no cross-warp phase),
  block_size=1024 (32 warps, max)
- Accuracy: f32 associativity differences acceptable

Dependencies: 3.3 (shared memory), 3.4 (bar_sync, shfl_sync)
Risk: Warp-level reduction with partial warps when block_size < 32. We
require block_size to be a power of 2 and >= 32 for reductions. Since
block_size is already required to be a power of 2 with max 1024, the
minimum for reductions is 32. Emit compile error if block_size < 32
and a reduction is used.

---

### Sprint 3.6: Softmax Kernel
**Scope:** Implement and validate row-wise softmax
**Layer:** kaio/tests — E2E kernel using all Phase 3 features

Kernel: row-wise softmax, single block per row.
```rust
#[gpu_kernel(block_size = 256)]
fn softmax_row(input: &[f32], output: &mut [f32], row_len: u32) {
    let tid = thread_idx_x();
    let bsize = block_dim_x();
    let row_offset = block_idx_x() * row_len;

    // Step 1: Find row max (strided loop + reduce)
    let mut local_max = -3.402823e+38f32;
    let mut i = tid;
    while i < row_len {
        let val = input[row_offset + i];
        if val > local_max { local_max = val; }
        i += bsize;
    }
    let row_max = block_reduce_max(local_max);

    // Step 2: Compute exp(x - max) and sum (strided loop + reduce)
    let mut local_sum = 0.0f32;
    let mut i = tid;
    while i < row_len {
        local_sum += exp(input[row_offset + i] - row_max);
        i += bsize;
    }
    let row_sum = block_reduce_sum(local_sum);

    // Step 3: Normalize (strided loop)
    let mut i = tid;
    while i < row_len {
        output[row_offset + i] = exp(input[row_offset + i] - row_max) / row_sum;
        i += bsize;
    }
}
```

Note: Uses `while` loops with manual stride (`i += bsize`), NOT `for` with
`step_by`. Sprint 3.1 only supports simple `for i in start..end` ranges.

Tests:
- GPU: softmax on 128-element row, compare to CPU reference
- GPU: softmax on 1024-element row (single block)
- GPU: multi-row softmax (batch of rows, one block per row)
- GPU: sum of output row ≈ 1.0 (softmax invariant)

Dependencies: 3.1 (while loops, +=), 3.5 (reductions), 3.4 (bar_sync)

---

### Sprint 3.7: PyTorch Validation + Accuracy Suite
**Scope:** Numerical accuracy testing at three matrix sizes
**Layer:** kaio/tests

Validation approach:
- Generate reference softmax outputs with Python/PyTorch script
- Hardcode known-good reference values for deterministic test inputs
- Compare KAIO output against reference at three sizes:
  - 128 x 128: max abs error < 1e-5
  - 1024 x 1024: max abs error < 1e-4
  - 4096 x 4096: max abs error < 1e-3
- Edge case tests:
  - All zeros input → uniform distribution (1/N for each element)
  - All same value → uniform distribution
  - Very large values → verify no inf/nan (numerical stability via max subtraction)

Reference CPU softmax:
```rust
fn cpu_softmax(row: &[f32]) -> Vec<f32> {
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = row.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}
```

Dependencies: 3.6 (softmax kernel)

---

### Sprint 3.8: Polish + Coverage + Docs
**Scope:** Coverage to 65%, documentation, final validation
**Layer:** all crates

Checklist:
- [ ] Run `cargo llvm-cov --workspace`, identify coverage gaps, add tests
- [ ] Coverage >= 65% workspace total
- [ ] All success criteria 3.1–3.10 verified
- [ ] Compile-fail tests updated (CF09 message, new CFs for Phase 3)
- [ ] `gpu_builtins` stubs complete for all new functions
- [ ] Sprint docs (sprint_3_1.md through sprint_3_8.md) with ADR traces
- [ ] PHASE_3_LOG.md sprint table filled in
- [ ] Zero clippy warnings, cargo fmt clean
- [ ] All GPU tests pass on RTX 4090

Dependencies: All prior sprints

---

## 6. Dependency Graph

```
3.1 (loops + +=) ─────────────┐
                               ├──→ 3.3 (shared mem DSL) → 3.4 (builtins) → 3.5 (reductions) → 3.6 (softmax) → 3.7 (PyTorch) → 3.8 (polish)
3.2 (shared mem + shfl core) ─┘
```

Sprints 3.1 and 3.2 have NO dependency on each other — 3.1 is pure
kaio-macros work, 3.2 is pure kaio-core work. They can execute in parallel.

---

## 7. Key Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **shfl.sync `c` operand encoding** | Silent lane-read corruption | Read PTX ISA 8.7 S9.7.8 before coding. Validate against ptxas. Sprint 3.2 has explicit spec-reading step. |
| **Range pattern parsing** | For-loop parsing fails | syn ExprForLoop wraps Pat + Expr; support only Pat::Ident + Expr::Range(Half). Reject all other patterns with clear errors. |
| **shared_mem! macro-in-macro** | Parse failure | syn sees Expr::Macro; pattern-match on path "shared_mem" and manually parse the inner token stream. |
| **Reduction correctness** | Wrong sum/max values | Require block_size >= 32 and power of 2. Test with known analytical results. Broadcast result to all threads via shared mem. |
| **f32 reduction ordering** | Numerical drift vs CPU | Tolerance thresholds in success criteria already account for non-associative fp addition. |
| **Compound assignment in index position** | `arr[i] += e` fails to parse | Handle in AssignOp parser alongside simple variable compound assignment. |

---

## 8. Success Criteria (from docs/success-criteria.md)

| # | Criterion | Sprint |
|---|-----------|--------|
| 3.1 | `for` loops compile and execute | 3.1 |
| 3.2 | `while` loops compile and execute | 3.1 |
| 3.3 | Shared memory allocation works | 3.3 |
| 3.4 | `bar.sync` prevents data races | 3.4 |
| 3.5 | `block_reduce_sum` correct | 3.5 |
| 3.6 | `block_reduce_max` correct | 3.5 |
| 3.7 | Softmax within PyTorch tolerance | 3.7 |
| 3.8 | Softmax edge cases | 3.7 |
| 3.9 | Shared memory usage reported at compile time | 3.3 |
| 3.10 | Warning if shared memory exceeds SM limit | 3.3 |
