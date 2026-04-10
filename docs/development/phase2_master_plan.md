# Phase 2 Master Plan: `#[gpu_kernel]` Proc Macro DSL

**Status:** Planning
**Date:** 2026-04-10
**Depends on:** Phase 1 complete (commit `6c00418`)

---

## 1. Goal

Transform KAIO from an internal IR library into a user-facing GPU kernel framework. Users write:

```rust
use kaio::prelude::*;

#[gpu_kernel(block_size = 256)]
fn vector_add(a: &[f32], b: &[f32], out: &mut [f32], n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        out[idx] = a[idx] + b[idx];
    }
}

fn main() -> Result<()> {
    let device = KaioDevice::new(0)?;
    let a = device.alloc_from(&vec![1.0f32; 1024])?;
    let b = device.alloc_from(&vec![2.0f32; 1024])?;
    let mut out = device.alloc_zeros::<f32>(1024)?;
    vector_add::launch(&device, &a, &b, &mut out, 1024u32)?;
    let result = out.to_host(&device)?;
    Ok(())
}
```

Phase 2 delivers: `kaio-macros` crate, four working kernels through the macro
(`vector_add`, `saxpy`, `fused_relu`, `fused_gelu`), compile-fail diagnostics
via trybuild, and the `kaio::prelude` user-facing API.

---

## 2. Architecture Decision: Runtime IR Construction

**Decision:** The macro generates Rust code that builds `kaio-core` IR at
runtime, emits PTX via `PtxWriter`, and caches the result in
`OnceLock<String>`.

**Alternatives considered:**

| Option | Approach | Rejected because |
|---|---|---|
| A | Compile-time PTX string embedding | Proc-macro crates can't link against non-proc-macro crates at expansion time. Would require duplicating all of kaio-core's emission logic inside the macro crate. |
| C | const fn / build.rs hybrid | Rust const eval can't run `format!` or complex allocation. build.rs adds a separate compilation step for negligible benefit. |

**Why Option B wins:**
- Zero duplication: generated code calls the existing kaio-core IR builders
- Negligible cost: PTX construction + emission is microseconds, cached via `OnceLock`
- Debuggability: `KAIO_DUMP_PTX=1` can write generated PTX at runtime
- Future-proof: parameterized kernels (Phase 3+ loops with bounds) become trivial

---

## 3. Generated Code Shape

The macro expands `#[gpu_kernel] fn name(...)` into a Rust module:

```rust
mod vector_add {
    use std::sync::OnceLock;
    use kaio::core::emit::{Emit, PtxWriter};
    use kaio::core::instr::*;
    use kaio::core::ir::*;
    use kaio::core::types::PtxType;
    use kaio::runtime::{GpuBuffer, LaunchConfig, KaioDevice, KaioError};

    static PTX_CACHE: OnceLock<String> = OnceLock::new();

    fn build_ptx() -> String {
        let mut alloc = RegisterAllocator::new();
        let mut kernel = PtxKernel::new("vector_add");
        // ... IR construction matching the hand-built E2E pattern ...
        // ... (params, special regs, arithmetic, memory, control flow) ...
        kernel.set_registers(alloc.into_allocated());
        let mut module = PtxModule::new("sm_89");
        module.add_kernel(kernel);
        let mut w = PtxWriter::new();
        module.emit(&mut w).unwrap();
        let ptx = w.finish();
        if std::env::var("KAIO_DUMP_PTX").is_ok() {
            eprintln!("=== KAIO PTX: vector_add ===\n{ptx}");
        }
        ptx
    }

    pub fn launch(
        device: &KaioDevice,
        a: &GpuBuffer<f32>,
        b: &GpuBuffer<f32>,
        out: &mut GpuBuffer<f32>,
        n: u32,
    ) -> Result<(), KaioError> {
        let ptx = PTX_CACHE.get_or_init(build_ptx);
        let module = device.load_ptx(ptx)?;
        let func = module.function("vector_add")?;
        let cfg = LaunchConfig::for_num_elems(n);
        // SAFETY: kernel signature matches params (enforced by macro codegen),
        // buffers are valid device pointers (enforced by GpuBuffer construction),
        // launch config within device limits (enforced by LaunchConfig).
        unsafe {
            device.stream().launch_builder(func.inner())
                .arg(a.inner()).arg(b.inner()).arg(out.inner_mut()).arg(&n)
                .launch(cfg)
        }?;
        Ok(())
    }
}
```

### Argument marshaling

| Rust param type | Launch wrapper param | PTX param | `.arg()` call |
|---|---|---|---|
| `&[T]` | `&GpuBuffer<T>` | `.param .u64 name_ptr` | `.arg(buf.inner())` |
| `&mut [T]` | `&mut GpuBuffer<T>` | `.param .u64 name_ptr` | `.arg(buf.inner_mut())` |
| `u32` | `u32` | `.param .u32 name` | `.arg(&val)` |
| `f32` | `f32` | `.param .f32 name` | `.arg(&val)` |
| (other scalars) | (same type) | (corresponding PTX type) | `.arg(&val)` |

### Module load caching

The `OnceLock` caches the PTX string. The `CudaModule` is re-loaded on each
`launch()` call. This is intentional for Phase 2:
- cudarc's CUDA driver caches loaded modules internally per context
- A second `OnceLock<KaioModule>` would tie it to a specific `KaioDevice`,
  complicating multi-device use
- No correctness impact; tight-loop benchmarks would see overhead but
  typical use (one launch per kernel invocation) is fine
- Optimize in Phase 3+ only if profiling shows load overhead matters

---

## 4. Crate Structure: `kaio-macros`

```
kaio-macros/
  Cargo.toml              # proc-macro = true; deps: syn 2 (full), quote 1, proc-macro2 1
  src/
    lib.rs                 # #[proc_macro_attribute] pub fn gpu_kernel
    parse/
      mod.rs
      attrs.rs             # parse #[gpu_kernel(block_size = N)]
      signature.rs         # fn signature -> KernelSignature
      body.rs              # fn body -> Vec<KernelStmt>
    kernel_ir/
      mod.rs
      types.rs             # KernelType, KernelParam, KernelConfig
      expr.rs              # KernelExpr enum
      stmt.rs              # KernelStmt enum
    lower/
      mod.rs               # orchestrator + LoweringContext
      params.rs            # params -> PtxParam + ld.param codegen
      arith.rs             # arithmetic -> ArithOp codegen
      compare.rs           # comparisons + if/else -> SetP + branch codegen
      memory.rs            # array index -> address calc + ld/st codegen
      builtins.rs          # built-in fn registry + lowering
      cast.rs              # `as` casts -> Cvt codegen
    codegen/
      mod.rs               # assemble final TokenStream
      ptx_builder.rs       # build_ptx() function body
      launch_wrapper.rs    # launch() function + module shell
    validate/
      mod.rs
      type_check.rs        # type inference + validation
      diagnostics.rs       # error formatting with spans
```

### Data flow

```
User source (#[gpu_kernel] fn ...)
       |
  [syn parse]           lib.rs parses ItemFn via syn
       |
  [parse/*]             Extract KernelSignature + lower body to Vec<KernelStmt>
       |
  [validate/*]          Type-check expressions, reject unsupported constructs
       |
  [kernel_ir/*]         Our intermediate representation (not syn types)
       |
  [lower/*]             Transform KernelIR -> TokenStream fragments calling kaio-core
       |
  [codegen/*]           Assemble final TokenStream: mod { build_ptx + launch }
       |
  TokenStream output
```

---

## 5. Kernel IR (Intermediate Representation)

Types inside `kaio-macros` that bridge syn's AST to the generated kaio-core
API calls. Never publicly exposed.

### KernelType

```rust
enum KernelType {
    F32, F64, I32, U32, I64, U64, Bool,
    SliceRef(Box<KernelType>),      // &[T] — read-only pointer
    SliceMutRef(Box<KernelType>),   // &mut [T] — writable pointer
}
```

### KernelExpr

```rust
enum KernelExpr {
    LitInt(i64, KernelType, Span),
    LitFloat(f64, KernelType, Span),
    LitBool(bool, Span),
    Var(String, Span),
    BinOp { op: BinOpKind, lhs: Box<KernelExpr>, rhs: Box<KernelExpr>, span: Span },
    UnaryOp { op: UnaryOpKind, expr: Box<KernelExpr>, span: Span },
    Index { array: String, index: Box<KernelExpr>, span: Span },
    BuiltinCall { name: String, args: Vec<KernelExpr>, span: Span },
    Cast { expr: Box<KernelExpr>, target_ty: KernelType, span: Span },
    Paren(Box<KernelExpr>, Span),
}

enum BinOpKind {
    Add, Sub, Mul, Div, Rem,         // arithmetic
    Lt, Le, Gt, Ge, Eq, Ne,          // comparison -> Bool
    BitAnd, BitOr, BitXor, Shl, Shr, // bitwise
    And, Or,                          // logical
}

enum UnaryOpKind { Neg, Not }
```

### KernelStmt

```rust
enum KernelStmt {
    Let { name: String, ty: Option<KernelType>, value: KernelExpr, span: Span },
    Assign { name: String, value: KernelExpr, span: Span },
    IndexAssign { array: String, index: KernelExpr, value: KernelExpr, span: Span },
    If { condition: KernelExpr, then_body: Vec<KernelStmt>,
         else_body: Option<Vec<KernelStmt>>, span: Span },
    Expr(KernelExpr, Span),
}
```

### KernelDef (top level)

```rust
struct KernelConfig { block_size: u32 }
struct KernelParam { name: String, ty: KernelType, span: Span }
struct KernelSignature { name: String, params: Vec<KernelParam>, config: KernelConfig }
struct KernelDef { sig: KernelSignature, body: Vec<KernelStmt> }
```

### Type inference

`HashMap<String, KernelType>` built during validation:
- Parameters: type from signature
- `let x = expr;`: type = inferred from expr
- `let x: T = expr;`: type = T, verify expr produces T
- Binary arithmetic: both sides same numeric type; result = that type
- Comparison: both sides same numeric type; result = Bool
- Index `arr[idx]`: result = element type of slice; idx must be integer
- Builtin call: return type from registry
- Cast `expr as T`: result = T; numeric-to-numeric only
- Literals: `42` defaults to I32; `42u32` is U32; `1.0` defaults to F32

---

## 6. New kaio-core Instructions Required

### Arithmetic (Sprint 2.2)

| Variant | PTX | Shape |
|---|---|---|
| `ArithOp::Sub` | `sub{ty} dst, lhs, rhs;` | Same as Add |
| `ArithOp::Mul` | `mul.lo{ty}` (int) / `mul{ty}` (float) | Same as Add; int needs `.lo` |
| `ArithOp::Div` | `div{ty} dst, lhs, rhs;` | Same as Add |
| `ArithOp::Rem` | `rem{ty} dst, lhs, rhs;` | Same as Add |
| `ArithOp::Neg` | `neg{ty} dst, src;` | Unary |

### Math (Sprint 2.5)

| Variant | PTX | Notes |
|---|---|---|
| `ArithOp::Abs` | `abs{ty} dst, src;` | All numeric types |
| `ArithOp::Min` | `min{ty} dst, lhs, rhs;` | |
| `ArithOp::Max` | `max{ty} dst, lhs, rhs;` | |
| `ArithOp::Sqrt` | `sqrt.approx.f32 dst, src;` | f32 only Phase 2 |
| `ArithOp::Ex2` | `ex2.approx.f32 dst, src;` | For exp() synthesis |
| `ArithOp::Lg2` | `lg2.approx.f32 dst, src;` | For log() synthesis |
| `ArithOp::Rcp` | `rcp.approx.f32 dst, src;` | Reciprocal |
| `ArithOp::Fma` | `fma.rn{ty} dst, a, b, c;` | Fused multiply-add |

### Control flow (Sprint 2.3)

| Change | PTX | Notes |
|---|---|---|
| `BraPred` gains `negate: bool` | `@!%p1 bra target;` | Deferred from Sprint 1.4. Natural if/else lowering: `setp.lt` + `@!pred bra SKIP` reads as "skip when NOT less-than" |

Everything else already exists: Add, Mad, MulWide, all MemoryOps, all
ControlOps (SetP, BraPred, Bra, Ret), all SpecialRegs, all CmpOps,
Mov, Cvt, Label, Comment.

---

## 7. If/Else Lowering Pattern

### ADR: Predicate negation vs condition inversion

**Context:** Sprint 1.4 deferred `@!pred` (predicate negation) as not needed
for `vector_add`'s bounds check pattern. Phase 2 `if/else` lowering needs to
conditionally skip code blocks.

**Options:**
- **(A) Invert the comparison:** `if x < n` -> `setp.ge %p, x, n` + `@%p bra SKIP`.
  Works but the generated PTX is confusing — the SetP comparison is the
  opposite of what the source code says.
- **(B) Add `@!pred`:** `if x < n` -> `setp.lt %p, x, n` + `@!%p bra SKIP`.
  The SetP matches the source operator. The branch reads naturally:
  "skip when NOT less-than."

**Decision:** (B). Add a `negate: bool` field to `BraPred`. When true, emit
`@!{pred} bra {target};`. This is the simplest change (one field, one format
character in the emit match arm) and makes generated PTX readable. Every
developer who runs `KAIO_DUMP_PTX=1` will see comparisons that match
their source code.

**Implementation:** In `kaio-core/src/instr/control.rs`:

```rust
BraPred {
    pred: Register,
    target: String,
    negate: bool,  // NEW: when true, emit @!pred instead of @pred
}
```

Emit change: `@{neg}{pred} bra {target};` where `neg = if negate { "!" } else { "" }`.

Existing call sites (`vector_add` E2E test) pass `negate: false` — no
behavior change.

### Lowering pattern

**`if cond { then_body }`:**
1. Lower `cond` -> SetP with source-code-matching comparison -> predicate `%p`
2. `@!%p bra IF_END_{n}` (skip then-body when predicate is false)
3. Emit then-body instructions
4. `IF_END_{n}:` label

**`if cond { then_body } else { else_body }`:**
1. Lower `cond` -> SetP -> predicate `%p`
2. `@!%p bra IF_ELSE_{n}` (skip to else when predicate is false)
3. Emit then-body
4. `bra IF_END_{n}` (skip else-body)
5. `IF_ELSE_{n}:` label
6. Emit else-body
7. `IF_END_{n}:` label

---

## 8. Lowering Context

The lowering pass threads a context through all functions:

```rust
struct LoweringContext {
    params: HashMap<String, ParamInfo>,        // kernel params -> PTX param names + load regs
    locals: HashMap<String, LocalInfo>,        // let bindings -> register idents
    global_addrs: HashMap<String, Ident>,      // pointer params -> cvta'd address register
    types: HashMap<String, KernelType>,        // type inference table
    label_counter: u32,                        // unique label generation
    reg_counter: u32,                          // unique register variable names in TokenStream
}
```

Register naming in generated code: `let _kaio_r{N} = alloc.alloc(PtxType::...);`
with monotonic counter. These are Rust variable names in the generated
`build_ptx()` — the actual PTX register names come from `RegisterAllocator`.

---

## 9. Built-in Function Registry

### Thread/block builtins (no-argument, return `u32`)

| Kernel function | Maps to | PTX |
|---|---|---|
| `thread_idx_x/y/z()` | `special::tid_x/y/z()` | `mov.u32 %r, %tid.x/y/z` |
| `block_idx_x/y/z()` | `special::ctaid_x/y/z()` | `mov.u32 %r, %ctaid.x/y/z` |
| `block_dim_x/y/z()` | `special::ntid_x/y/z()` | `mov.u32 %r, %ntid.x/y/z` |
| `grid_dim_x/y/z()` | `special::nctaid_x/y/z()` | `mov.u32 %r, %nctaid.x/y/z` |

### Math builtins

| Kernel function | Implementation | Notes |
|---|---|---|
| `sqrt(x)` | `sqrt.approx.f32` | Direct PTX instruction |
| `exp(x)` | `mul(x, LOG2_E)` then `ex2.approx.f32` | exp(x) = 2^(x * log2(e)) |
| `log(x)` | `lg2.approx.f32` then `mul(result, LN_2)` | ln(x) = log2(x) * ln(2) |
| `tanh(x)` | Synthesized from exp | tanh(x) = (exp(2x)-1)/(exp(2x)+1) |
| `abs(x)` | `abs{ty}` | Direct PTX instruction |
| `min(x,y)` | `min{ty}` | Direct PTX instruction |
| `max(x,y)` | `max{ty}` | Direct PTX instruction |

### IDE stubs

`kaio/src/gpu_builtins.rs` exports real Rust functions that panic at runtime:

```rust
/// Returns the thread index in the X dimension. Only valid inside #[gpu_kernel].
pub fn thread_idx_x() -> u32 { panic!("thread_idx_x() can only be called inside a #[gpu_kernel] function") }
```

The macro recognizes calls by name and replaces them — stubs never execute
in generated code. Purpose: rust-analyzer autocomplete + docs + type checking.

---

## 10. Prelude

```rust
// kaio/src/prelude.rs
pub use crate::runtime::{KaioDevice, GpuBuffer, LaunchConfig, KaioError, Result};
pub use crate::gpu_kernel;
pub use crate::gpu_builtins::*;
```

Intentionally excludes: `kaio::core::*` (internal IR), `KaioModule`,
`KaioFunction` (hidden behind launch wrapper).

---

## 11. Error Handling

**Approach:** `syn::Error::new_spanned(tokens, message)` throughout. No
`proc_macro_error` dependency. Multi-error accumulation via `combine()`.

**Pattern:**
```rust
#[proc_macro_attribute]
pub fn gpu_kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    match gpu_kernel_impl(attr.into(), item.into()) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}
```

**Error categories:** See `docs/success-criteria.md` Phase 2 compile-fail cases
CF1-CF10 for the required minimum.

---

## 12. Design Decisions Not Revisited

These carry forward from Phase 1:
- Virtual workspace structure (kaio, kaio-core, kaio-runtime, +kaio-macros)
- Register allocator: 5-counter model (%r, %rd, %f, %fd, %p)
- PTX ISA 8.7, `.target sm_89`, `.address_size 64` (sm_89 hardcoded for Phase 2)
- Register declarations: `.b32`/`.b64` untyped (matches nvcc)
- Emit trait returns `fmt::Result`
- cudarc 0.19 with dynamic-loading

New for Phase 2:
- **No `a + b*c -> mad` fusion.** The NVIDIA PTX JIT does this. Adding it in
  the macro lowering creates complexity for zero measurable benefit.
- **Module load per launch** (not cached). cudarc's driver layer caches
  internally. Optimize only if profiling shows overhead.
- **SM target hardcoded** to sm_89. Parameterizing is a one-line change to
  `PtxModule::new()` when needed.

---

## 13. Coverage Targets

Per `docs/success-criteria.md` Phase 2:

| Crate | Target |
|---|---|
| `kaio-core` | 70% |
| `kaio-runtime` | 60% |
| `kaio-macros` | 65% |
| **Workspace** | **>=60%** |

Proc macro code is inherently harder to cover (expansion tests don't show as
line coverage in the macro crate). These targets are realistic, not
aspirational. Don't let a coverage number block shipping.

---

## 14. Sprint Breakdown

| Sprint | Scope | Risk | Depends on |
|---|---|---|---|
| 2.1 | Macro skeleton + fn signature parsing | Low | — |
| 2.2 | Expression lowering: arithmetic + new ArithOps in core | Medium | 2.1 |
| 2.3 | Comparisons + if/else + `@!pred` support | Medium | 2.2 |
| 2.4 | Array indexing + memory access | **High** | 2.2 |
| 2.5 | Built-in functions (thread/block index, math) | Medium | 2.2 |
| 2.6 | Launch wrapper + full pipeline connection | **High** | 2.1-2.5 |
| 2.7 | Type validation + compile-fail tests | Medium | 2.1-2.2 |
| 2.8 | E2E kernel tests (vector_add, saxpy, fused_relu, fused_gelu) | Medium-High | 2.1-2.7 |

Sprint details with per-sprint deliverables, files, and tests are tracked in
the individual sprint docs under `docs/development/sprints/sprint_2_*.md`.

---

## 15. Risk Mitigation

### Sprint 2.4 (array indexing) — highest risk

The address calculation pattern (`cvta -> mul.wide -> add.s64 -> ld.global`)
is the dependency for every E2E kernel. A wrong `sizeof(T)` or missing `cvta`
produces garbage data, not a clear error.

**Mitigation:** Sprint 2.4 includes a standalone address calculation unit test
that constructs just the address math IR, emits PTX, and validates the
instruction sequence (byte offset values, register types) before any
kernel-level testing. This catches bugs hours earlier than waiting for Sprint
2.8 E2E failures.

### Sprint 2.6 (full pipeline) — integration risk

Everything comes together: parsing, lowering, codegen, emission. TokenStream
assembly with `quote!` is complex and hard to debug.

**Mitigation:**
- `cargo expand` during development to inspect generated code
- `KAIO_DUMP_EXPANSION=1` env var check in the macro to print expansion
- Snapshot tests on generated TokenStream
- Build incrementally: get vector_add through the pipeline first, then
  validate other kernels

### General: proc macro debugging

**Mitigation:**
- Extensive unit tests on each lowering function in isolation
- `trybuild` compile-fail tests catch regression early
- `KAIO_DUMP_PTX=1` for runtime PTX inspection
- Golden PTX comparison: macro-generated vector_add vs hand-built E2E

---

## 16. Success Gate

Phase 2 is complete when all criteria in `docs/success-criteria.md` Phase 2
pass: four kernels running on GPU through the macro, 10+ compile-fail tests,
numerical accuracy within tolerance, coverage targets met, zero clippy
warnings, all code formatted.
