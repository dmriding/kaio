//! Python wrapper around `GpuBuffer<T>` plus shape + dtype metadata.
//!
//! `kaio.Tensor` owns a `TensorStorage` enum variant carrying a concrete
//! `GpuBuffer<T>`; the enum is exhaustive-match-checked at every
//! dispatch site (no runtime downcast). Future dtype additions (`i8` for
//! INT8 matmul, packed `u32` for INT4) will extend the enum as
//! two-line additions, and the compiler will name every site that needs
//! a new match arm.
//!
//! Tensors hold a clone of their `Device`'s `Arc<KaioDevice>`. A Tensor
//! cannot outlive the CUDA context it was created under — Rust's
//! reference counting enforces that without lifetime annotations.

use std::sync::Arc;

use half::f16;
use kaio_rs::prelude::{GpuBuffer, KaioDevice};
use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyType};

use crate::device::Device;
use crate::errors::{kaio_err, map_kaio_err};

/// Storage variants for `kaio.Tensor`. Extend with new arms in 8.2 for
/// INT8 + packed-INT4 support; the compiler will surface every site
/// that needs a new branch.
pub enum TensorStorage {
    F16(GpuBuffer<f16>),
    F32(GpuBuffer<f32>),
}

impl TensorStorage {
    /// Dtype name matching NumPy's convention (`"float16"`, `"float32"`).
    fn dtype_name(&self) -> &'static str {
        match self {
            TensorStorage::F16(_) => "float16",
            TensorStorage::F32(_) => "float32",
        }
    }
}

/// A GPU-resident tensor with shape + dtype metadata.
///
/// Construct via `kaio.Tensor.from_numpy(device, numpy_array)`; convert
/// back to a NumPy array via `.to_numpy()`. Between those boundaries,
/// Tensors are passed by handle — multiple ops chain on device without
/// round-tripping through CPU.
#[pyclass(module = "kaio", name = "Tensor")]
pub struct Tensor {
    pub(crate) device: Arc<KaioDevice>,
    pub(crate) storage: TensorStorage,
    pub(crate) shape: Vec<usize>,
}

#[pymethods]
impl Tensor {
    /// Construct a Tensor from a NumPy array by copying host data to GPU.
    ///
    /// Rejects non-C-contiguous arrays with a clear error — wrap with
    /// `np.ascontiguousarray(x)` on the Python side. Supported dtypes
    /// in 8.1 are `float16` and `float32`; extending to `int8` +
    /// packed-`uint32` happens in 8.2.
    #[classmethod]
    fn from_numpy(
        _cls: &Bound<'_, PyType>,
        device: &Device,
        array: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let flags = array.getattr("flags")?;
        let c_contig: bool = flags.getattr("c_contiguous")?.extract()?;
        if !c_contig {
            return Err(kaio_err(
                "from_numpy requires a C-contiguous array; use np.ascontiguousarray(x) on the Python side",
            ));
        }

        let dtype_name: String = array.getattr("dtype")?.getattr("name")?.extract()?;
        let device_arc = Arc::clone(&device.inner);

        match dtype_name.as_str() {
            "float16" => {
                let arr: PyReadonlyArrayDyn<'_, f16> = array.extract()?;
                let shape: Vec<usize> = arr.shape().to_vec();
                let slice = arr.as_slice()?;
                let buffer = device_arc.alloc_from(slice).map_err(map_kaio_err)?;
                Ok(Self {
                    device: device_arc,
                    storage: TensorStorage::F16(buffer),
                    shape,
                })
            }
            "float32" => {
                let arr: PyReadonlyArrayDyn<'_, f32> = array.extract()?;
                let shape: Vec<usize> = arr.shape().to_vec();
                let slice = arr.as_slice()?;
                let buffer = device_arc.alloc_from(slice).map_err(map_kaio_err)?;
                Ok(Self {
                    device: device_arc,
                    storage: TensorStorage::F32(buffer),
                    shape,
                })
            }
            other => Err(kaio_err(format!(
                "unsupported dtype: expected float16 or float32, got {other}"
            ))),
        }
    }

    /// Copy the Tensor's contents back to a host-resident NumPy array.
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.storage {
            TensorStorage::F16(buf) => {
                let host: Vec<f16> = buf.to_host(&self.device).map_err(map_kaio_err)?;
                let arr = host.into_pyarray(py);
                let reshaped = arr.reshape(self.shape.as_slice())?;
                Ok(reshaped.into_any())
            }
            TensorStorage::F32(buf) => {
                let host: Vec<f32> = buf.to_host(&self.device).map_err(map_kaio_err)?;
                let arr = host.into_pyarray(py);
                let reshaped = arr.reshape(self.shape.as_slice())?;
                Ok(reshaped.into_any())
            }
        }
    }

    /// Shape as a list of dimension sizes.
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    /// Element dtype name (`"float16"` / `"float32"`).
    #[getter]
    fn dtype(&self) -> &'static str {
        self.storage.dtype_name()
    }

    /// Total number of elements (product of shape).
    fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// First-dim length (matches NumPy / Torch convention).
    fn __len__(&self) -> PyResult<usize> {
        self.shape
            .first()
            .copied()
            .ok_or_else(|| kaio_err("0-dim tensor has no len()"))
    }

    fn __repr__(&self) -> String {
        format!(
            "Tensor(shape={:?}, dtype={})",
            self.shape,
            self.storage.dtype_name()
        )
    }
}
