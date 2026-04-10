//! # pyros-runtime
//!
//! CUDA runtime layer for the PYROS GPU kernel authoring framework. This
//! crate wraps [`cudarc`] to provide device management, typed device
//! buffers, PTX module loading, and a builder-style kernel launch API. It
//! is Layer 2 of PYROS, sitting on top of [`pyros-core`] (PTX emission).
//!
//! **Status:** Phase 1 scaffolding. No public API yet.
//!
//! [`cudarc`]: https://crates.io/crates/cudarc
//! [`pyros-core`]: https://crates.io/crates/pyros-core

#![warn(missing_docs)]

#[cfg(test)]
mod tests {
    #[test]
    #[ignore] // requires NVIDIA GPU
    fn cudarc_smoke_test() {
        let ctx = cudarc::driver::CudaContext::new(0).expect("CudaContext::new(0) failed");
        let stream = ctx.default_stream();

        let host_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let dev_buf = stream
            .clone_htod(&host_data)
            .expect("host-to-device transfer failed");
        let result: Vec<f32> = stream
            .clone_dtoh(&dev_buf)
            .expect("device-to-host transfer failed");

        assert_eq!(result, host_data, "roundtrip data mismatch");
    }
}
