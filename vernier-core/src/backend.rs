//! The [`ComputeBackend`] and [`ComputeJob`] traits. `vernier-cpu` implements
//! them with `rustfft`/`ndarray`, `vernier-gpu` with Vulkano compute and
//! `vernier-cuda` with cuFFT and CUDA kernels.
//!
//! A job groups several operations. You `begin()` a job, queue work on it, then
//! `submit()`. On the CPU everything runs immediately and `submit` is a no-op;
//! on the GPU the operations are recorded into a command buffer that `submit`
//! dispatches and waits on.
//!
//! ```text
//! backend.upload(...)         → Buffer2D       (immediate)
//! let mut job = backend.begin()?
//! job.fft2d(&mut buf)?        → queues work
//! job.peak_search(...)        → queues work, returns a Buffer2D handle
//! job.submit()?               → runs everything queued
//! backend.download(&buf)?     → reads result back to host
//! ```

use crate::buffer::{Buffer2D, BufferLayout};
use crate::complex::Complex32;
use crate::error::Result;
use crate::scalar::Real;

/// A unit of queued work produced by [`ComputeBackend::begin`].
///
/// Each method records an operation; [`submit`](ComputeJob::submit) executes them.
/// On the CPU every method executes immediately and `submit` is a no-op.
pub trait ComputeJob {
    /// The buffer type this job operates on (must match the backend's `Buffer2D`).
    type Buffer2D;

    /// Uploads `data` into a new device buffer as part of this job's command
    /// stream, so the staging copy rides along with the compute dispatches.
    fn upload(&mut self, data: &[Complex32], layout: BufferLayout) -> Result<Self::Buffer2D>;

    /// Deep-copies `src` into a new, independently-writable buffer. Use it before
    /// a destructive in-place op when the same source must be processed two ways
    /// (e.g. the two carrier directions).
    fn copy_buffer(&mut self, src: &Self::Buffer2D) -> Result<Self::Buffer2D>;

    /// In-place 2D forward FFT.
    fn fft2d(&mut self, buf: &mut Self::Buffer2D) -> Result<()>;

    /// In-place 2D inverse FFT.
    fn ifft2d(&mut self, buf: &mut Self::Buffer2D) -> Result<()>;

    /// Gaussian band-pass filter in-place, centred on frequency bin `(cx, cy)`.
    fn bandpass_filter(
        &mut self,
        buf: &mut Self::Buffer2D,
        cx: usize,
        cy: usize,
        sigma: Real,
    ) -> Result<()>;

    /// Like `bandpass_filter`, but reads the carrier bin from a `peak_search`
    /// output buffer. `direction` picks the pair: 0 → (cx1, cy1), 1 → (cx2, cy2).
    /// On the GPU this avoids a host round-trip for the coordinates.
    fn bandpass_from_peaks(
        &mut self,
        buf: &mut Self::Buffer2D,
        peaks: &Self::Buffer2D,
        direction: u32,
        sigma: Real,
    ) -> Result<()>;

    /// Annulus mask: zeroes bins outside `[min_frequency, max_frequency]` from DC.
    fn filter(
        &mut self,
        buf: &mut Self::Buffer2D,
        min_frequency: usize,
        max_frequency: usize,
    ) -> Result<()>;

    /// Separable 2D Gaussian blur on the real component, in-place. Taps wrap
    /// circularly (the target is a periodic magnitude spectrum).
    fn gaussian_blur_2d(&mut self, buf: &mut Self::Buffer2D, sigma: Real) -> Result<()>;

    /// Per-pixel `atan2(im, re)` → new buffer with phase in the `.re` lane.
    fn extract_phase(&mut self, buf: &Self::Buffer2D) -> Result<Self::Buffer2D>;

    /// Finds the two carrier peaks and returns a 2×2 buffer holding
    /// `[cx1, cy1, cx2, cy2]` (ordered so direction 1 has the larger signed
    /// column frequency). Returns `None` when no valid peaks are found.
    fn peak_search(
        &mut self,
        buf: &mut Self::Buffer2D,
        min_frequency: usize,
        max_frequency: usize,
        smoothing_sigma: Real,
        sigma: Real,
    ) -> Result<Option<Self::Buffer2D>>;

    /// Computes (a, b, c) for both carrier directions from the frequency-domain
    /// spectrum using a Gaussian-weighted spectral centroid. Returns 6 elements:
    /// `[a1, b1, c1, a2, b2, c2]`. Fast but biased by coding sidebands — use for
    /// orientation only, not absolute pose (the pipeline fits planes host-side
    /// by least squares over the unwrapped phase, matching C++ RegressionPlane).
    fn spectral_plane_fit_two(
        &mut self,
        spectrum: &Self::Buffer2D,
        peaks: &Self::Buffer2D,
        sigma: Real,
    ) -> Result<Self::Buffer2D>;

    /// Runs all queued operations and waits for completion. Consumes the job;
    /// call [`ComputeBackend::begin`] again for more work.
    fn submit(self) -> Result<()>;
}

/// A compute backend: creates jobs, uploads data, and downloads results.
pub trait ComputeBackend {
    /// The backend's concrete 2D buffer type.
    type Buffer2D: Buffer2D;

    /// The type of job returned by [`begin`](ComputeBackend::begin).
    type Job<'a>: ComputeJob<Buffer2D = Self::Buffer2D>
    where
        Self: 'a;

    /// Begins a new unit of queued work.
    fn begin(&self) -> Result<Self::Job<'_>>;

    /// Uploads contiguous, row-major complex data into a device buffer.
    fn upload(&self, data: &[Complex32], layout: BufferLayout) -> Result<Self::Buffer2D>;

    /// Downloads a device buffer back to host memory, row-major.
    ///
    /// Must be called after the job that produced `buffer` has been submitted.
    fn download(&self, buffer: &Self::Buffer2D) -> Result<Vec<Complex32>>;

    /// Uploads a real (greyscale) image as complex (imaginary parts zeroed).
    fn upload_real(&self, image: &crate::image::GrayImage) -> Result<Self::Buffer2D> {
        let complex = image.to_complex();
        self.upload(&complex, image.layout())
    }

    /// A short human-readable name for the backend, e.g. `"cpu-rustfft"`.
    fn name(&self) -> &str;
}
