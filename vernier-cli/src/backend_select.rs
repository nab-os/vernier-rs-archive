//! Backend selection — the one place allowed to name concrete backends and
//! choose between them at runtime.
//!
//! [`ComputeBackend`](vernier_core::ComputeBackend) has an associated `Buffer2D`
//! type, so it isn't object-safe — you can't `Box<dyn ComputeBackend>` and swap
//! at runtime without erasing the buffer type. Instead the unit of work stays
//! generic: [`BackendTask::run`] is a generic method written once against
//! `B: ComputeBackend`, and [`dispatch`] calls it in each match arm with the
//! concrete backend. Adding a backend is one arm here, and every task gets it.
//!
//! The camera commands demodulate through [`LocalDemodulator`] instead, whose
//! frame type differs per backend too; [`Demodulator`] picks one at runtime
//! by delegating to it.

use vernier_camera::CpuDemodulator;
use vernier_camera::measure::Image;
use vernier_core::{
    ComputeBackend, DemodWindow, FieldDemod, LocalDemodulator, Real, VernierError, WindowDemod,
};
use vernier_cpu::CpuBackend;
#[cfg(feature = "cuda")]
use vernier_cuda::CudaBackend;
#[cfg(feature = "gpu")]
use vernier_gpu::{GpuBackend, GpuFrame};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
    #[cfg(feature = "gpu")]
    Gpu,
    #[cfg(feature = "cuda")]
    Cuda,
}

impl BackendKind {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "cpu" => Some(Self::Cpu),
            #[cfg(feature = "gpu")]
            "gpu" => Some(Self::Gpu),
            #[cfg(feature = "cuda")]
            "cuda" => Some(Self::Cuda),
            _ => None,
        }
    }

    pub fn hint() -> &'static str {
        #[cfg(all(feature = "gpu", feature = "cuda"))]
        return "cpu, gpu, cuda";
        #[cfg(all(feature = "gpu", not(feature = "cuda")))]
        return "cpu, gpu";
        #[cfg(not(feature = "gpu"))]
        return "cpu";
    }
}

pub trait BackendTask {
    type Output;
    fn run<B: ComputeBackend>(&self, backend: &B) -> Self::Output;
}

pub fn dispatch<T: BackendTask>(kind: BackendKind, task: &T) -> T::Output {
    match kind {
        BackendKind::Cpu => task.run(&CpuBackend::new()),
        #[cfg(feature = "gpu")]
        BackendKind::Gpu => task.run(&GpuBackend::new()),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => task.run(&CudaBackend::new().expect("CUDA init failed")),
    }
}

/// The [`LocalDemodulator`] the camera commands measure with, chosen at
/// runtime. Its frames are the chosen backend's, so the camera code stays
/// generic and one value can be shared between the measuring thread and the
/// page's.
pub enum Demodulator {
    Cpu(CpuDemodulator),
    #[cfg(feature = "gpu")]
    Gpu(Box<GpuBackend>),
}

/// A frame loaded on a [`Demodulator`].
pub enum DemodulatorFrame<'a> {
    Cpu(Image<'a>),
    #[cfg(feature = "gpu")]
    Gpu(GpuFrame),
}

impl Demodulator {
    pub fn new(kind: BackendKind) -> Result<Self, String> {
        match kind {
            BackendKind::Cpu => Ok(Self::Cpu(CpuDemodulator)),
            #[cfg(feature = "gpu")]
            BackendKind::Gpu => Ok(Self::Gpu(Box::new(GpuBackend::new()))),
            #[cfg(feature = "cuda")]
            BackendKind::Cuda => Err("the camera commands run on cpu or gpu, not cuda".into()),
        }
    }
}

fn other_backend() -> VernierError {
    VernierError::Backend("frame loaded on another demodulator".into())
}

impl LocalDemodulator for Demodulator {
    type Frame<'a> = DemodulatorFrame<'a>;

    fn load<'a>(
        &'a self,
        data: &'a [f32],
        width: usize,
        height: usize,
    ) -> vernier_core::Result<DemodulatorFrame<'a>> {
        match self {
            Self::Cpu(d) => d.load(data, width, height).map(DemodulatorFrame::Cpu),
            #[cfg(feature = "gpu")]
            Self::Gpu(d) => d.load(data, width, height).map(DemodulatorFrame::Gpu),
        }
    }

    fn demodulate_windows(
        &self,
        frame: &DemodulatorFrame<'_>,
        windows: &[DemodWindow],
    ) -> vernier_core::Result<Vec<WindowDemod>> {
        match (self, frame) {
            (Self::Cpu(d), DemodulatorFrame::Cpu(f)) => d.demodulate_windows(f, windows),
            #[cfg(feature = "gpu")]
            (Self::Gpu(d), DemodulatorFrame::Gpu(f)) => d.demodulate_windows(f, windows),
            #[allow(unreachable_patterns)]
            _ => Err(other_backend()),
        }
    }

    fn demodulate_field(
        &self,
        frame: &DemodulatorFrame<'_>,
        references: [&[Real]; 2],
        sigma: Real,
    ) -> vernier_core::Result<FieldDemod> {
        match (self, frame) {
            (Self::Cpu(d), DemodulatorFrame::Cpu(f)) => d.demodulate_field(f, references, sigma),
            #[cfg(feature = "gpu")]
            (Self::Gpu(d), DemodulatorFrame::Gpu(f)) => d.demodulate_field(f, references, sigma),
            #[allow(unreachable_patterns)]
            _ => Err(other_backend()),
        }
    }
}
