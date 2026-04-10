# Phase 1 — PTX Foundation: Sprint Index

Quick-reference index for Phase 1 sprints. Each sprint has a dedicated
doc in [sprints/](sprints/) with full reasoning traces for every decision.

## Sprint Status

| Sprint | Scope | Status | Commit | Tests |
|---|---|---|---|---|
| [1.0](sprints/sprint_1_0.md) | Workspace restructure | Done | `8e59f66` | — |
| [1.1](sprints/sprint_1_1.md) | Types + IR skeleton + register allocator | Done | `b2dac92` | 20 |
| [1.2](sprints/sprint_1_2.md) | Arithmetic instructions (Add, Mad, MulWide) | Done | `279c799` | +7 = 27 |
| [1.3](sprints/sprint_1_3.md) | Memory instructions (LdParam, LdGlobal, StGlobal, CvtaToGlobal) | Done | `23db945` | +7 = 34 |
| [1.4](sprints/sprint_1_4.md) | Control flow + special registers (SetP, Bra, BraPred, Ret) | Done | pending | +8 = 42 |
| [1.5](sprints/sprint_1_5.md) | PtxWriter + full module emission | Pending | — | — |
| [1.6](sprints/sprint_1_6.md) | Runtime device + buffers | Pending | — | — |
| [1.7](sprints/sprint_1_7.md) | Runtime launch + vector_add E2E | Pending | — | — |
| [1.8](sprints/sprint_1_8.md) | Testing + coverage + docs polish | Pending | — | — |

## Key Validations

- **nvcc golden PTX:** `pyros-core/tests/golden/nvcc_vector_add_sm89.ptx` —
  compiled with `nvcc --ptx -arch=sm_89` from CUDA 12.8 on RTX 4090. All
  instruction emitters validated byte-for-byte against this file.
- **cudarc smoke test:** `pyros-runtime` `#[ignore]` test confirms
  `CudaContext::new(0)` + host↔device roundtrip works on the RTX 4090.
- **PTX ISA version:** `.version 8.7` (CUDA 12.8 = PTX ISA 8.7, not 7.8).
