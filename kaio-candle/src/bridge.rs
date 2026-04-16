//! Bridge primitives between candle's `CudaStorage` / `CudaDevice` and
//! KAIO's `GpuBuffer<T>` / `KaioDevice`.
//!
//! Implementation lands in D2. This module is a stub at D1 so the module
//! tree compiles end-to-end and the rest of the D1 scaffolding can exercise
//! the `#[cfg(feature = "cuda")]` path.
