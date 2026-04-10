# PYROS — Implementation Details

## Layer 1: PTX Codegen (`pyros-core`)

### What This Layer Does

Provides Rust types and an API for programmatically constructing valid PTX assembly. Think of it as a "PTX builder" — structured code generation where each PTX instruction is a Rust type with an `emit()` method that produces valid PTX text.

### Key Components

#### 1.1 PTX IR (Intermediate Representation)

A Rust-native IR that models PTX programs. This is NOT Triton-IR or LLVM-IR — it's a lightweight, purpose-built representation.

```
PtxModule
├── version (e.g., ".version 7.8")
├── target (e.g., ".target sm_89")
├── address_size (e.g., ".address_size 64")
└── kernels: Vec<PtxKernel>
    ├── name: String
    ├── params: Vec<PtxParam>
    ├── registers: Vec<PtxRegister>
    ├── body: Vec<PtxInstruction>
    └── metadata: KernelMetadata
```

#### 1.2 PTX Type System

Map Rust types to PTX types:

| Rust Type | PTX Type | Register Kind |
|-----------|----------|---------------|
| `f32` | `.f32` | `%f` (float) |
| `f64` | `.f64` | `%fd` (double) |
| `f16` | `.f16` | `%h` (half) |
| `bf16` | `.bf16` | `%hb` (bfloat) |
| `i32` | `.s32` | `%r` (signed int) |
| `u32` | `.u32` | `%ru` (unsigned int) |
| `i64` | `.s64` | `%rd` (signed long) |
| `u64` | `.u64` | `%rd` (unsigned long) |
| `bool` | `.pred` | `%p` (predicate) |

#### 1.3 Instruction Catalog

PTX instructions to implement, ordered by priority:

**Priority 1 — Arithmetic (required for any kernel):**
- `add`, `sub`, `mul`, `div`, `rem` (integer + float variants)
- `mad` (multiply-add, fused — critical for performance)
- `fma` (fused multiply-add for floats)
- `neg`, `abs`
- `min`, `max`

**Priority 2 — Memory (required for any useful kernel):**
- `ld.global` / `st.global` (global memory load/store)
- `ld.param` (kernel parameter load)
- `ld.shared` / `st.shared` (shared memory — needed from Phase 4)
- `mov` (register move)
- `cvt` (type conversion)

**Priority 3 — Control Flow:**
- `setp` (set predicate / comparison)
- `@%p bra` (predicated branch)
- `bra` (unconditional branch)
- `ret` (return)
- `bar.sync` (barrier synchronization)

**Priority 4 — Special Registers:**
- `%tid.x`, `%tid.y`, `%tid.z` (thread index)
- `%ctaid.x`, `%ctaid.y`, `%ctaid.z` (block/CTA index)
- `%ntid.x`, `%ntid.y`, `%ntid.z` (block dimension)
- `%nctaid.x`, `%nctaid.y`, `%nctaid.z` (grid dimension)

**Priority 5 — Math Functions (for activation functions):**
- `ex2` (2^x, base for exp)
- `lg2` (log base 2)
- `rcp` (reciprocal)
- `sqrt`, `rsqrt`
- `sin`, `cos`
- `tanh` (if available on target SM, otherwise synthesized)

**Priority 6 — Advanced (Phase 4+):**
- `atom` (atomic operations)
- `red` (reduction)
- `shfl` (warp shuffle)
- `mma` (matrix multiply-accumulate, tensor core ops)
- `cp.async` (async memory copy)

#### 1.4 Register Allocation

Simple linear register allocator for v0.1:
- Assign virtual registers during IR construction
- Map to physical register names during emission
- Track register pressure per kernel (report warnings if approaching SM limits)

No need for sophisticated graph coloring — PTX is a virtual ISA and the NVIDIA assembler (`ptxas`) handles physical register allocation.

#### 1.5 PTX Emission

Each IR node implements `emit(&self, writer: &mut PtxWriter) -> Result<()>`:
- `PtxWriter` manages indentation, label generation, and comment insertion
- Output is valid `.ptx` text that passes `ptxas` validation
- Optional: emit human-readable comments alongside instructions for debugging

### Testing Strategy (Layer 1)

- **Unit tests:** Each instruction type emits syntactically valid PTX (validate with `ptxas --verify`)
- **Round-trip tests:** Build IR → emit PTX → parse back (optional, lower priority)
- **Golden file tests:** Known-good PTX output compared against emitted output

---

## Layer 2: Runtime (`pyros-runtime`)

### What This Layer Does

Handles all interaction with the GPU: loading compiled PTX modules, allocating device memory, transferring data between host and device, launching kernels, and synchronizing.

### Foundation: cudarc

[`cudarc`](https://github.com/chelsea0x3b/cudarc) provides safe Rust bindings to the CUDA driver API. PYROS builds on top of this rather than writing raw FFI bindings. (Note: the canonical crates.io release is now the `chelsea0x3b/cudarc` fork; earlier docs referenced `coreylowman/cudarc`.)

Key `cudarc` capabilities PYROS uses:
- `CudaDevice` — device handle, context management
- `CudaSlice<T>` — device memory allocation with type safety
- `CudaModule` / `CudaFunction` — PTX module loading and kernel launch
- Host ↔ device memory transfers

### Key Components

#### 2.1 Kernel Registry

```rust
pub struct KernelRegistry {
    device: Arc<CudaDevice>,
    modules: HashMap<String, CudaModule>,
}

impl KernelRegistry {
    /// Load a PTX module from a string (compile-time generated)
    pub fn load_ptx(&mut self, name: &str, ptx: &str) -> Result<()>;

    /// Launch a registered kernel
    pub fn launch(
        &self,
        name: &str,
        grid: Grid,
        block: Block,
        args: &[KernelArg],
    ) -> Result<()>;
}
```

#### 2.2 Memory Management

```rust
/// Type-safe device buffer
pub struct GpuBuffer<T: GpuType> {
    inner: CudaSlice<T>,
    len: usize,
}

impl<T: GpuType> GpuBuffer<T> {
    pub fn from_host(device: &CudaDevice, data: &[T]) -> Result<Self>;
    pub fn to_host(&self) -> Result<Vec<T>>;
    pub fn len(&self) -> usize;
    pub fn zeroed(device: &CudaDevice, len: usize) -> Result<Self>;
}
```

#### 2.3 Launch Configuration

```rust
pub struct Grid {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

pub struct Block {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

/// Helper to compute grid dimensions from data size and block size
pub fn compute_grid(data_len: usize, block_size: u32) -> Grid;
```

#### 2.4 Device Discovery & Capability Querying

```rust
pub struct DeviceInfo {
    pub name: String,
    pub compute_capability: (u32, u32),  // e.g., (8, 9) for SM 8.9
    pub total_memory: usize,
    pub sm_count: u32,
    pub max_threads_per_block: u32,
    pub max_shared_memory_per_block: usize,
    pub warp_size: u32,
}

pub fn enumerate_devices() -> Result<Vec<DeviceInfo>>;
```

### Testing Strategy (Layer 2)

- **Integration tests:** Allocate memory, copy to device, copy back, verify
- **Kernel launch tests:** Load handwritten PTX, launch, verify output
- **Multi-device tests:** Enumerate devices, verify capability reporting (single GPU is fine — test the API, not multi-GPU)
- **Error handling tests:** Invalid PTX, out-of-memory, invalid launch configs

---

## Layer 3: Proc Macro DSL (`pyros-macros`)

### What This Layer Does

Provides the `#[gpu_kernel]` attribute macro that transforms Rust-like function syntax into PTX codegen + runtime launch glue. This is the user-facing API — the thing developers actually interact with.

### User-Facing API

```rust
use pyros::prelude::*;

#[gpu_kernel(block_size = 256)]
fn vector_add(a: &[f32], b: &[f32], out: &mut [f32], n: u32) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < n {
        out[idx] = a[idx] + b[idx];
    }
}

fn main() -> Result<()> {
    let device = PyrosDevice::new(0)?;
    let a = device.alloc_from(&vec![1.0f32; 1024])?;
    let b = device.alloc_from(&vec![2.0f32; 1024])?;
    let mut out = device.alloc_zeroed::<f32>(1024)?;

    vector_add::launch(&device, &a, &b, &mut out, 1024u32)?;

    let result = out.to_host()?;
    assert_eq!(result[0], 3.0);
    Ok(())
}
```

### Macro Expansion

The `#[gpu_kernel]` macro:

1. Parses the function body using `syn`
2. Validates types (only `GpuType` types allowed as parameters)
3. Transforms the AST into PYROS IR (Layer 1)
4. Emits PTX via Layer 1 at compile time
5. Generates a launch wrapper function that:
   - Accepts host-side typed arguments
   - Computes grid/block dimensions
   - Loads the PTX module (lazy, cached)
   - Launches the kernel via Layer 2
   - Returns `Result<()>`

### Supported Rust Subset (v0.1)

**Supported:**
- Arithmetic operators: `+`, `-`, `*`, `/`, `%`
- Comparison operators: `<`, `>`, `<=`, `>=`, `==`, `!=`
- `if` / `else` (compiles to predicated instructions or branches)
- `let` bindings (register allocation)
- Array indexing: `a[idx]` (compiles to `ld.global` / `st.global`)
- Built-in functions: `thread_idx_x()`, `block_idx_x()`, `block_dim_x()`, etc.
- Math functions: `sqrt()`, `abs()`, `min()`, `max()`, `exp()`, `log()`, `tanh()`
- Type casting via `as` keyword

**Not Supported (v0.1):**
- Loops (`for`, `while`) — added in v0.2
- Structs, enums, traits
- Function calls within kernels (inlining only)
- Closures
- Heap allocation
- String operations

### Testing Strategy (Layer 3)

- **Compile-time tests:** Macro expands without errors for valid kernels
- **Compile-fail tests:** Invalid kernel signatures produce clear error messages (use `trybuild`)
- **End-to-end tests:** Macro-generated kernels produce correct output
- **PTX inspection tests:** Verify emitted PTX matches expected patterns

---

## Layer 4: Block-Level Operations (`pyros-ops`)

### What This Layer Does

Provides higher-level abstractions for operating on blocks/tiles of data rather than individual elements. This is where PYROS approaches Triton's core innovation — the programmer thinks in blocks, the framework handles thread mapping and shared memory.

### Key Abstractions

```rust
#[gpu_kernel(block_size = 128)]
fn fused_softmax(input: &[f32], output: &mut [f32], n_cols: u32) {
    let row = block_idx_x();
    let col_offset = thread_idx_x();

    // Block-level load — each thread in the block loads one element
    let block = block_load(input, row * n_cols, n_cols);

    // Block-level reduction — max across the block
    let row_max = block_reduce_max(block);

    // Element-wise within block
    let shifted = block - row_max;
    let exp_val = exp(shifted);

    // Block-level reduction — sum
    let row_sum = block_reduce_sum(exp_val);

    // Normalize and store
    let result = exp_val / row_sum;
    block_store(output, row * n_cols, result);
}
```

### Block Operations to Implement

| Operation | Description | Underlying Mechanism |
|-----------|-------------|---------------------|
| `block_load` | Load a contiguous block from global to shared memory | Coalesced `ld.global` + `st.shared` |
| `block_store` | Store a block from shared/registers to global memory | Coalesced `st.global` |
| `block_reduce_sum` | Sum reduction across a thread block | Shared memory + warp shuffle |
| `block_reduce_max` | Max reduction across a thread block | Shared memory + warp shuffle |
| `block_reduce_min` | Min reduction across a thread block | Shared memory + warp shuffle |
| `block_dot` | Block-level matrix multiply (tiled) | Shared memory tiling + `fma` |
| `block_broadcast` | Broadcast scalar to all threads in block | Shared memory broadcast |

### Shared Memory Management

The framework automatically:
- Calculates required shared memory per kernel based on block operations used
- Inserts `bar.sync` instructions at correct synchronization points
- Manages shared memory bank conflict avoidance (padding)
- Reports shared memory usage at compile time (warn if exceeding SM limits)

### Testing Strategy (Layer 4)

- **Correctness tests:** Every block operation validated against CPU reference implementation
- **Numerical accuracy tests:** Compare against PyTorch output with tolerance thresholds
- **Performance tests:** Benchmark against cuBLAS/cuDNN equivalents (not required to match, but track regression)
- **Shared memory tests:** Verify correct synchronization (race condition detection)

---

## Cross-Cutting Concerns

### Error Handling

PYROS uses a unified error type across all layers:

```rust
#[derive(Debug, thiserror::Error)]
pub enum PyrosError {
    #[error("CUDA error: {0}")]
    Cuda(#[from] cudarc::driver::DriverError),

    #[error("PTX compilation error: {0}")]
    PtxCompilation(String),

    #[error("Invalid kernel configuration: {0}")]
    InvalidConfig(String),

    #[error("Device not found: {0}")]
    DeviceNotFound(String),

    #[error("Out of device memory: requested {requested} bytes, available {available} bytes")]
    OutOfMemory { requested: usize, available: usize },
}
```

### Logging & Diagnostics

- `PYROS_LOG=debug` — print emitted PTX to stderr
- `PYROS_LOG=trace` — print IR transformations
- `PYROS_DUMP_PTX=1` — write `.ptx` files to disk alongside the binary
- `PYROS_PROFILE=1` — print kernel launch times and memory transfer times

### Platform-Specific Notes

**Windows:**
- CUDA driver API (`nvcuda.dll`) is shipped with the NVIDIA display driver — no CUDA toolkit install required for runtime
- `ptxas.exe` is needed for ahead-of-time compilation to cubin (optional, PTX JIT works without it)
- `build.rs` must detect CUDA installation via registry keys (`HKLM\SOFTWARE\NVIDIA Corporation\GPU Computing Toolkit\CUDA`)

**Linux:**
- CUDA driver API (`libcuda.so`) is shipped with the NVIDIA driver
- `ptxas` available via CUDA toolkit or as standalone download
- `build.rs` detects CUDA via `nvidia-smi` and `/usr/local/cuda`
