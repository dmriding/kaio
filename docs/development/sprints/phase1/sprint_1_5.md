# Sprint 1.5 — PtxWriter + Full Module Emission

**Commit:** `d635721`
**Status:** Complete

## Context

Sprint 1.5 is the orchestration sprint. All individual instruction emitters
from sprints 1.2–1.4 are wired into a complete pipeline: PtxModule →
PtxKernel → register declarations → instruction body → labels. This is the
Phase 1 milestone for pyros-core — the crate can now produce a complete
`.ptx` file from an IR tree.

## Decisions

### Where to put Module/Kernel Emit impls — emit_trait.rs or co-located?

**Context:** Sprints 1.2–1.4 established the pattern of co-locating instruction
Emit impls with their types (ArithOp Emit in arith.rs, etc.). Should
PtxModule and PtxKernel follow the same pattern?

**Decision:** Keep in `emit_trait.rs`. Module and kernel emission is
orchestration logic — they call child Emit impls, manage indentation,
emit structural elements (headers, signatures, register declarations).
This is fundamentally different from leaf-level instruction emission
that produces a single PTX line. Keeping orchestration in emit_trait.rs
and leaf emission co-located is a clean separation of abstraction levels.

### raw_line() vs indent save/restore for labels and headers

**Context:** Labels need column 0 output. Module headers (`.version`,
`.target`) need column 0 output. PtxWriter::line() always applies
indentation. Options:
1. Add `raw_line()` — writes without any indentation
2. Save indent level, set to 0, write, restore
3. Track indent in the IR node and pass to writer

**Decision:** `raw_line()`. It's one method (5 lines), zero-risk, explicit
intent. Indent save/restore is fragile if code panics between save and
restore. Raw_line is also useful for the module header directives — they're
always at column 0 regardless of context.

### Register declaration format — individual vs range syntax

**Context:** PTX supports two register declaration forms:
1. Individual: `.reg .b32 %r0; .reg .b32 %r1;` (verbose)
2. Range: `.reg .b32 %r<5>;` (declares %r0 through %r4)

nvcc uses the range syntax.

**Decision:** Range syntax (`<N>`). Matches nvcc convention, more compact,
and trivially computed from the allocator's max index per kind. Implementation
uses fixed-size arrays indexed by `counter_index()` — no BTreeMap, no Ord
derivation needed on RegKind.

### Register declaration ordering

**Context:** nvcc orders declarations as P, F, R, Rd (pred, float, int32, int64).
Our implementation iterates by counter_index: R(0), Rd(1), F(2), Fd(3), P(4).

**Decision:** Keep our ordering. PTX register declaration order doesn't affect
correctness — ptxas accepts any order. Matching nvcc's order would require
a custom sort for no functional benefit. Deterministic order is sufficient.

### Kernel signature parameter formatting

**Context:** nvcc uses tabs and specific spacing for parameter declarations.
Our PtxWriter uses 4-space indentation. Parameters need commas after all but
the last.

**Decision:** Standard 4-space indentation, comma after all params except
the last. Indent/dedent around the param block so params are one level
deeper than the kernel signature. Closing `)` at column 0.

### Label indentation — dedent/indent dance vs raw_line

**Context:** Inside a kernel body, the writer is at indent level 1. Labels
need to be at column 0. Two approaches:
1. `raw_line()` directly (but then next instruction would also be at level 0
   unless we explicitly indent after)
2. `dedent() → raw_line() → indent()` (leaves indent state correct for
   subsequent instructions)

**Decision:** Option 2 — dedent/indent dance. After the label, the writer
must be back at indent level 1 for the next instruction (typically `ret;`
after a label). The dedent saturates at 0 (confirmed in Sprint 1.1), so
consecutive labels or labels at indent 0 are safe.

### PtxParam::ptx_decl() — new method on IR type

**Context:** Kernel emission needs to format parameter declarations. This
could be done inline in the Kernel Emit impl or as a method on PtxParam.

**Decision:** Method on PtxParam (`ptx_decl()`). The declaration format is
inherent to the parameter type — Scalar gets `.param {ty} {name}`, Pointer
gets `.param .u64 {name}` (always 64-bit). Putting it on PtxParam keeps
the knowledge co-located and tested independently.

### Integration test strategy — byte-for-byte vs structural

**Context:** The full vector_add test could compare byte-for-byte against
the nvcc golden file, or validate structural correctness (contains all
required elements in the right order).

**Decision:** Structural validation via `assert!(ptx.contains(...))`. Our
output legitimately differs from nvcc in register numbering (sequential
vs nvcc's optimized allocation), parameter names (descriptive vs
`vector_add_param_N`), whitespace (spaces vs tabs), and register declaration
order. Byte-for-byte comparison would be a false negative. The structural
checks validate every required PTX element is present.

## Scope

**In:** PtxModule Emit, PtxKernel Emit, PtxInstruction Mov/Cvt/Label/Comment
Emit, emit_reg_declarations helper, PtxWriter::raw_line(), PtxParam::ptx_decl(),
7 inline tests, 1 full vector_add integration test.

**Out:** Runtime code, GPU execution, ptxas verification, coverage measurement,
PtxWriter macro optimization.

## Results

Completed as planned. No deviations.

**The emitted vector_add PTX:**
```ptx
.version 8.7
.target sm_89
.address_size 64

.visible .entry vector_add(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 c_ptr,
    .param .u32 n
)
{
    .reg .b32 %r<5>;
    .reg .b64 %rd<10>;
    .reg .f32 %f<3>;
    .reg .pred %p<1>;

    ld.param.u64 %rd0, [a_ptr];
    ld.param.u64 %rd1, [b_ptr];
    ld.param.u64 %rd2, [c_ptr];
    ld.param.u32 %r0, [n];
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.s32 %r4, %r1, %r2, %r3;
    setp.ge.u32 %p0, %r4, %r0;
    @%p0 bra EXIT;
    cvta.to.global.u64 %rd3, %rd0;
    mul.wide.u32 %rd4, %r4, 4;
    add.s64 %rd5, %rd3, %rd4;
    ld.global.f32 %f0, [%rd5];
    cvta.to.global.u64 %rd6, %rd1;
    add.s64 %rd7, %rd6, %rd4;
    ld.global.f32 %f1, [%rd7];
    add.f32 %f2, %f0, %f1;
    cvta.to.global.u64 %rd8, %rd2;
    add.s64 %rd9, %rd8, %rd4;
    st.global.f32 [%rd9], %f2;
EXIT:
    ret;
}
```

**Tests:** 52 total (42 existing + 9 new unit + 1 integration), all passing.
**Files modified:** 3 (emit/emit_trait.rs, emit/writer.rs, ir/param.rs).
**Files created:** 1 (tests/vector_add_emit.rs).
