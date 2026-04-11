# KAIO — Testing Strategy

## Philosophy

KAIO is a compiler and runtime. Users trust it with their GPU compute workloads.
A kernel that silently produces wrong results destroys trust permanently.
Testing exists to guarantee: **if it compiles, it runs correctly.**

Three layers, each catching different failure modes.

---

## Layer 1: PTX Emission Correctness

**What we're testing:** Does kaio-core emit valid, semantically correct PTX?

### 1a. Instruction Emit Tests (existing)

Every ArithOp, MemoryOp, ControlOp variant has inline tests comparing
emitted strings against nvcc golden output. These are the foundation.

**Gap to close:** Systematic coverage of all (instruction × type) combinations.

```
# Example: exhaustive ArithOp::Add coverage
add.f32, add.f64, add.s32, add.u32, add.s64, add.u64
sub.f32, sub.f64, sub.s32, sub.u32, sub.s64, sub.u64
mul.f32, mul.f64, mul.lo.s32, mul.lo.u32, mul.lo.s64, mul.lo.u64
...
```

Add a `#[test] fn exhaustive_arith_emit()` that iterates all valid
(op, type) pairs and asserts the mnemonic is well-formed. Not comparing
against nvcc — just verifying the string matches the pattern
`"{op}{modifier}{type_suffix} {dst}, {src...};\n"`.

### 1b. ptxas Verification (existing)

Emit full kernel PTX, shell out to `ptxas --gpu-name sm_89`. If ptxas
accepts it, the PTX is syntactically valid. Self-skips if ptxas not in PATH.

**Extend to:** Every E2E kernel from Sprint 2.8 should have a ptxas verify
test in addition to the GPU execution test. ptxas verification doesn't
need a GPU — it can run in CI on machines without NVIDIA hardware.

### 1c. Module-Level Emission Tests

Full PtxModule emission producing a complete `.ptx` file. Verify:
- Header (.version, .target, .address_size)
- Kernel signature with correct parameter declarations
- Register declarations with correct `<N>` counts
- Label indentation (column 0)
- No dangling registers (every register used in body is declared)

---

## Layer 2: Macro Expansion Correctness

**What we're testing:** Does #[gpu_kernel] produce the right IR construction code?

### 2a. Lowering Unit Tests (existing)

Individual lowering functions tested with constructed KernelExpr/KernelStmt
inputs, verifying the TokenStream contains expected IR builder calls.

### 2b. Snapshot Tests (future)

Planned: `cargo expand` or `macrotest` to capture full macro expansions
as golden files. Not yet implemented.

### 2c. Compile-Fail Tests

`trybuild` tests verifying that invalid kernel code produces clear
compile-time errors. 10+ cases covering:
- Invalid parameter types
- Non-unit return types
- Unsupported language constructs
- Missing block_size
- Invalid block_size values
- Write to immutable slice
- Undefined variables
- Type mismatches

Each test has a `.stderr` file with the expected error message.

---

## Layer 3: GPU Execution Correctness

**What we're testing:** Does the kernel produce correct results on real hardware?

### 3a. Deterministic E2E Tests (Sprint 2.8, existing pattern)

Fixed inputs, known expected outputs. Assert exact equality (for integer
ops) or tolerance (for float ops).

```rust
// vector_add: exact f32 equality
assert_eq!(result, vec![5.0f32, 7.0, 9.0]);

// fused_gelu: tolerance
for (got, expected) in result.iter().zip(&cpu_reference) {
    assert!((got - expected).abs() < 1e-4,
        "gelu mismatch: got {got}, expected {expected}");
}
```

### 3b. Property-Based Tests (add post-Phase 2)

Use `proptest` or `quickcheck` to generate random inputs and verify
kernel output against CPU reference implementation.

```rust
proptest! {
    #[test]
    #[ignore] // requires GPU
    fn vector_add_random(
        a in prop::collection::vec(any::<f32>(), 1..10000),
        b in prop::collection::vec(any::<f32>(), 1..10000),
    ) {
        let n = a.len().min(b.len());
        let a = &a[..n];
        let b = &b[..n];
        let expected: Vec<f32> = a.iter().zip(b).map(|(x, y)| x + y).collect();

        let result = run_kernel_vector_add(a, b);
        assert_eq!(result, expected);
    }
}
```

This catches edge cases:
- **NaN propagation:** NaN + anything = NaN
- **Infinity handling:** large float + large float = inf
- **Negative zero:** -0.0 + 0.0 = 0.0 (IEEE 754)
- **Denormals:** very small floats near zero
- **Single element arrays:** n=1, grid=1 block
- **Non-power-of-2 sizes:** n=997, n=10001

### 3c. Boundary Tests (add in Sprint 2.8)

Specific edge cases that random testing might miss:

```rust
// Empty array
test_kernel_with_n(0);        // should not crash, no-op

// Single element
test_kernel_with_n(1);        // single thread

// Exactly one block
test_kernel_with_n(256);      // block_size boundary

// One more than one block
test_kernel_with_n(257);      // exercises multi-block + partial block

// Large, non-aligned
test_kernel_with_n(100_003);  // prime number, no alignment

// Very large
test_kernel_with_n(1_000_000); // stress test memory + multi-block
```

### 3d. Numerical Accuracy Tests

For math builtins (sqrt, exp, log, tanh), compare GPU results against
f64 CPU reference values cast down to f32. Track both absolute and
relative error.

```rust
fn test_math_accuracy<F: Fn(f32) -> f32>(
    kernel_fn: &str,
    cpu_ref: F,
    inputs: &[f32],
    max_abs_error: f32,
    max_rel_error: f32,
) {
    let gpu_results = run_elementwise_kernel(kernel_fn, inputs);
    for (i, (got, &input)) in gpu_results.iter().zip(inputs).enumerate() {
        let expected = cpu_ref(input);
        let abs_err = (got - expected).abs();
        let rel_err = if expected != 0.0 { abs_err / expected.abs() } else { abs_err };
        assert!(abs_err <= max_abs_error && rel_err <= max_rel_error,
            "input[{i}]={input}: got {got}, expected {expected}, \
             abs_err={abs_err}, rel_err={rel_err}");
    }
}
```

Test inputs should cover the full float range:
- Normal range: -100.0 to 100.0
- Near zero: -1e-6 to 1e-6
- Large values: ±1e30 (test overflow behavior)
- Special values: 0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN

Accuracy targets (from success-criteria.md):

| Function | Max Abs Error | Max Rel Error |
|----------|--------------|---------------|
| sqrt     | 1e-6         | 1e-5          |
| exp      | 1e-5         | 1e-4          |
| log      | 1e-5         | 1e-4          |
| tanh     | 1e-5         | 1e-4          |
| gelu     | 1e-4         | 1e-3          |

---

## Test Infrastructure

### GPU Test Fixture

Shared device initialization across all GPU tests to avoid repeated
CUDA context creation:

```rust
// tests/common/gpu.rs (or test_utils crate)
use std::sync::OnceLock;
use kaio_runtime::KaioDevice;

static DEVICE: OnceLock<KaioDevice> = OnceLock::new();
pub fn device() -> &'static KaioDevice {
    DEVICE.get_or_init(|| KaioDevice::new(0).expect("GPU required"))
}
```

### PTX Debug Output on Failure

Every GPU test should capture the emitted PTX and print it on failure:

```rust
let ptx = build_ptx();
let module = device.load_ptx(&ptx).unwrap_or_else(|e| {
    eprintln!("=== PTX that failed ===\n{ptx}");
    panic!("load_ptx failed: {e}");
});
```

This is already done in the vector_add E2E test. Maintain this pattern
for all GPU tests.

### Test Organization

```
kaio-core/
  src/**/*.rs                    # inline unit tests (#[cfg(test)])
  tests/
    vector_add_emit.rs           # full kernel emission test
    ptxas_verify.rs              # ptxas offline verification
    common/mod.rs                # shared test helpers

kaio-macros/
  src/**/*.rs                    # inline lowering unit tests

kaio-runtime/
  tests/
    vector_add_e2e.rs            # Phase 1 E2E (IR API → GPU)

kaio/ (umbrella)
  tests/
    compile_fail.rs              # trybuild harness
    compile_fail/                # 10 compile-fail test cases (CF1–CF10)
    vector_add_macro.rs          # macro-generated E2E kernels
    saxpy_macro.rs
    fused_relu_macro.rs
    fused_gelu_macro.rs
    loops_macro.rs               # for/while loop GPU tests
    shared_mem_macro.rs           # shared memory GPU tests
    reduce_macro.rs              # block_reduce_sum/max GPU tests
    softmax_macro.rs             # 10-test softmax accuracy suite
```

### Running Tests

```bash
# All host-only tests (no GPU needed, fast) — 200+ tests
cargo test --workspace

# GPU tests (requires NVIDIA GPU) — 24 tests
cargo test --workspace -- --ignored

# Single crate
cargo test -p kaio-core
cargo test -p kaio-macros
cargo test -p kaio

# Compile-fail tests only
cargo test -p kaio --test compile_fail

# Coverage
cargo llvm-cov --workspace --ignore-filename-regex "tests/"

# Inspect generated PTX
KAIO_DUMP_PTX=1 cargo test --workspace
```

---

## Coverage Targets

| Phase   | Workspace | kaio-core | kaio-macros | kaio-runtime |
|---------|-----------|------------|--------------|---------------|
| Phase 1 | 60%       | 70%        | —            | 60%           |
| Phase 2 | 65%       | 70%        | 65%          | 60%           |
| Phase 3 | 65%       | 70%        | 65%          | 60%           |
| Phase 5 | 70%       | 75%        | 70%          | 65%           |

Coverage is a signal, not a goal. 100% coverage doesn't mean correct code.
60% coverage with the right tests beats 90% coverage that only tests
happy paths.

---

## Pre-Release Checklist (Phase 5)

Before `cargo publish`:

- [ ] All host tests pass on Linux and Windows
- [ ] All GPU tests pass on RTX 4090 (SM 8.9)
- [ ] All GPU tests pass on at least one other GPU arch (SM 7.5+ ideally)
- [ ] ptxas verification passes for all example kernels
- [ ] Coverage ≥70% workspace
- [ ] No clippy warnings
- [ ] cargo fmt clean
- [ ] cargo doc clean, no broken links
- [ ] All trybuild compile-fail tests pass
- [ ] Boundary tests pass (n=0 through n=1M)
- [ ] Math accuracy within tolerance for all builtins
- [ ] KAIO_DUMP_PTX works and produces readable output
- [ ] README example compiles and runs
