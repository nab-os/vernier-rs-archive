use std::sync::Mutex;

use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use vernier_core::buffer::BufferLayout;
use vernier_core::image::GrayImage;
use vernier_core::{Complex32, Real};
use vernier_cpu::CpuBackend;
use vernier_pose::{Calibration, absolute, periodic};
use vernier_spectral::spectrum::{Detection, analyze_two};

// ─── Backend enum ─────────────────────────────────────────────────────────────

// CpuBackend contains RefCell (not Sync), so we wrap it in a Mutex.
// CudaBackend already implements Send + Sync via its internal Arc<Mutex<...>>.
enum BackendInner {
    Cpu(Mutex<CpuBackend>),
    #[cfg(feature = "cuda")]
    Cuda(vernier_cuda::CudaBackend),
}

impl BackendInner {
    fn analyze_two(
        &self,
        data: &[Complex32],
        layout: BufferLayout,
        sigma: Real,
        min_frequency: usize,
        max_frequency: usize,
        smoothing_sigma: Real,
    ) -> vernier_core::Result<Detection> {
        match self {
            BackendInner::Cpu(m) => {
                let b = m.lock().expect("backend mutex poisoned");
                analyze_two(
                    &*b,
                    data,
                    layout,
                    sigma,
                    min_frequency,
                    max_frequency,
                    smoothing_sigma,
                )
            }
            #[cfg(feature = "cuda")]
            BackendInner::Cuda(b) => analyze_two(
                b,
                data,
                layout,
                sigma,
                min_frequency,
                max_frequency,
                smoothing_sigma,
            ),
        }
    }
}

// BackendInner is Send + Sync:
//   - Mutex<CpuBackend>: Send + Sync (Mutex provides Sync)
//   - CudaBackend: Send + Sync (unsafe impl on CudaContext)
unsafe impl Send for BackendInner {}
unsafe impl Sync for BackendInner {}

// ─── Pose ────────────────────────────────────────────────────────────────────

/// Detected pose: in-plane translation and orientation.
///
/// `x` and `y` are in the same physical units as the `period` argument.
/// `theta` is in radians.
#[pyclass(module = "vernier_py")]
pub struct Pose {
    #[pyo3(get)]
    pub x: f64,
    #[pyo3(get)]
    pub y: f64,
    #[pyo3(get)]
    pub theta: f64,
}

#[pymethods]
impl Pose {
    fn __repr__(&self) -> String {
        format!(
            "Pose(x={:.6}, y={:.6}, theta={:.6})",
            self.x, self.y, self.theta
        )
    }
}

// ─── Detector ────────────────────────────────────────────────────────────────

/// CPU- or GPU-backed pose detector.
///
/// Create with `Detector()` for the CPU backend, or `Detector.cuda()` for
/// CUDA. Reuse across frames — the internal FFT planner caches its plan,
/// so repeated calls on same-size images are cheaper than creating a new
/// Detector each time.
#[pyclass(module = "vernier_py")]
pub struct Detector {
    backend: BackendInner,
}

#[pymethods]
impl Detector {
    /// Creates a CPU-backed detector.
    #[new]
    pub fn new() -> Self {
        Self {
            backend: BackendInner::Cpu(Mutex::new(CpuBackend::new())),
        }
    }

    /// Creates a CUDA-backed detector.
    ///
    /// Raises `RuntimeError` if no CUDA device is available or if the
    /// extension was not built with CUDA support (`--features cuda`).
    #[staticmethod]
    pub fn cuda() -> PyResult<Self> {
        #[cfg(feature = "cuda")]
        {
            vernier_cuda::CudaBackend::new()
                .map(|b| Self {
                    backend: BackendInner::Cuda(b),
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(PyRuntimeError::new_err(
                "CUDA support not compiled in (rebuild with --features cuda)",
            ))
        }
    }

    /// Returns the name of the active backend (`\"cpu-rustfft\"` or `\"cuda\"`).
    pub fn backend_name(&self) -> &str {
        match &self.backend {
            BackendInner::Cpu(_) => "cpu-rustfft",
            #[cfg(feature = "cuda")]
            BackendInner::Cuda(_) => "cuda",
        }
    }

    /// Periodic (relative) detection.
    ///
    /// Recovers `x` and `y` modulo `period` and the in-image orientation
    /// `theta`. Use when the pattern has no absolute code, or when you only
    /// need the fine sub-period displacement.
    ///
    /// Args:
    ///     image:            2-D float32 numpy array, shape (H, W), values in [0, 1].
    ///     period:           Pattern spatial period in physical units.
    ///     sigma:            Bandpass filter half-width in frequency bins (default 3.0).
    ///     min_frequency:    Inner spectral annulus radius for peak search; 0 = no limit (default 0).
    ///     max_frequency:    Outer spectral annulus radius; 0 = no limit (default 0).
    ///     smoothing_sigma:  Gaussian blur sigma on magnitude spectrum before peak search (default 0.5).
    ///
    /// Returns:
    ///     Pose with `.x`, `.y`, `.theta`.
    ///
    /// Raises:
    ///     RuntimeError: if no carrier peaks are found.
    #[pyo3(signature = (image, period, sigma=3.0, min_frequency=0, max_frequency=0, smoothing_sigma=0.5))]
    pub fn detect_periodic(
        &self,
        image: PyReadonlyArray2<f32>,
        period: f64,
        sigma: f64,
        min_frequency: usize,
        max_frequency: usize,
        smoothing_sigma: f64,
    ) -> PyResult<Pose> {
        let arr = image.as_array();
        let shape = arr.shape();
        let (height, width) = (shape[0], shape[1]);
        let gray = GrayImage::from_vec(width, height, arr.iter().copied().collect())
            .ok_or_else(|| PyRuntimeError::new_err("image dimensions mismatch"))?;

        let detection = self
            .backend
            .analyze_two(
                &gray.to_complex(),
                gray.layout(),
                sigma as Real,
                min_frequency,
                max_frequency,
                smoothing_sigma as Real,
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let calib = Calibration::new(period as Real, width, height);
        let pose = periodic::estimate(&detection.dir1.plane, &detection.dir2.plane, &calib);

        Ok(Pose {
            x: pose.x,
            y: pose.y,
            theta: pose.theta,
        })
    }

    /// Megarena absolute detection.
    ///
    /// Recovers an unambiguous `(x, y, theta)` by combining the fine phase
    /// measurement with the LFSR binary code embedded in the Megarena pattern.
    ///
    /// Args:
    ///     image:            2-D float32 numpy array, shape (H, W), values in [0, 1].
    ///     physical_period:  Pattern spatial period in micrometres (9.0 for the reference pattern).
    ///     code_size:        LFSR order in bits (12 for the reference pattern).
    ///     sigma:            Bandpass filter half-width in frequency bins (default 3.0).
    ///     min_frequency:    Inner spectral annulus radius; 0 = no limit (default 20).
    ///     max_frequency:    Outer spectral annulus radius; 0 = no limit (default 500).
    ///     smoothing_sigma:  Gaussian blur sigma on magnitude spectrum (default 0.5).
    ///
    /// Returns:
    ///     Pose with `.x`, `.y`, `.theta`.
    ///
    /// Raises:
    ///     RuntimeError: if detection or LFSR decode fails.
    #[pyo3(signature = (image, physical_period, code_size, sigma=3.0, min_frequency=20, max_frequency=500, smoothing_sigma=0.5))]
    pub fn detect_megarena(
        &self,
        image: PyReadonlyArray2<f32>,
        physical_period: f64,
        code_size: u32,
        sigma: f64,
        min_frequency: usize,
        max_frequency: usize,
        smoothing_sigma: f64,
    ) -> PyResult<Pose> {
        let arr = image.as_array();
        let shape = arr.shape();
        let (height, width) = (shape[0], shape[1]);
        let gray = GrayImage::from_vec(width, height, arr.iter().copied().collect())
            .ok_or_else(|| PyRuntimeError::new_err("image dimensions mismatch"))?;

        let complex = gray.to_complex();
        let layout = gray.layout();

        let detection = self
            .backend
            .analyze_two(
                &complex,
                layout,
                sigma as Real,
                min_frequency,
                max_frequency,
                smoothing_sigma as Real,
            )
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let calib = Calibration::new(physical_period as Real, width, height);
        let pose = absolute::solve_megarena(&detection, gray.as_slice(), &calib, code_size)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(Pose {
            x: pose.x,
            y: pose.y,
            theta: pose.theta,
        })
    }
}

// ─── Module ──────────────────────────────────────────────────────────────────

#[pymodule]
fn vernier_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Pose>()?;
    m.add_class::<Detector>()?;
    Ok(())
}
