use vernier_core::buffer::{Buffer2D, BufferLayout};
use vernier_core::{Complex32, ComputeBackend, ComputeJob, Real, Result, VernierError};

use crate::planefit::{PhasePlane, fit_plane_to_unwrapped};
use crate::unwrap::quarters_unwrap_phase;

#[derive(Clone, Copy, Debug)]
pub struct DirectionResult {
    pub plane: PhasePlane,
    pub peak: (Real, Real),
    pub peak_bin: (usize, usize),
}

#[derive(Clone, Debug)]
pub struct Detection {
    pub dir1: DirectionResult,
    pub dir2: DirectionResult,
    /// Measured unwrapped phase map of direction 1 (row-major `width × height`),
    /// from the band-passed IFFT — the same map the plane was fitted to,
    /// mirroring C++ `PatternPhase::getUnwrappedPhase1`. Consumed by the
    /// megarena decode (cell indexing) and the 3D sign disambiguation.
    pub phase1: Vec<Real>,
    /// Measured unwrapped phase map of direction 2.
    pub phase2: Vec<Real>,
    pub width: usize,
    pub height: usize,
}

pub fn forward<B: ComputeBackend>(backend: &B, buffer: &mut B::Buffer2D) -> Result<()> {
    let mut job = backend.begin()?;
    job.fft2d(buffer)?;
    job.submit()
}

pub fn analyze_direction<B: ComputeBackend>(
    backend: &B,
    mut spectrum: B::Buffer2D,
    sigma: Real,
) -> Result<DirectionResult> {
    let layout = spectrum.layout();
    let (width, height) = (layout.width, layout.height);

    let (peaks_buffer, planes_buffer) = {
        let mut job = backend.begin()?;
        let peaks_buffer = job
            .peak_search(&mut spectrum, 0, 0, 0.5, sigma)?
            .ok_or_else(|| VernierError::Backend("no carrier peaks found".into()))?;
        let planes_buffer = job.spectral_plane_fit_two(&spectrum, &peaks_buffer, sigma)?;
        job.submit()?;
        (peaks_buffer, planes_buffer)
    };

    let peaks_data = backend.download(&peaks_buffer)?;
    let (peak_x, peak_y) = (peaks_data[0].re as usize, peaks_data[1].re as usize);

    let planes_data = backend.download(&planes_buffer)?;
    let plane = PhasePlane {
        a: planes_data[0].re as Real,
        b: planes_data[1].re as Real,
        c: planes_data[2].re as Real,
    };
    let peak = plane.peak_location(width, height);
    Ok(DirectionResult {
        plane,
        peak,
        peak_bin: (peak_x, peak_y),
    })
}

/// Crop factor of the phase-plane regression, matching C++
/// `RegressionPlane::cropFactor` (only the center half of each axis is fitted,
/// avoiding band-pass edge artefacts).
pub const REGRESSION_CROP_FACTOR: Real = 0.5;

/// Downloads a band-passed IFFT field, takes its per-pixel argument, unwraps it
/// outward from the image center (C++ `quartersUnwrapPhase`), and fits the
/// phase plane by least squares in f64 (C++ `RegressionPlane::compute`).
fn unwrap_and_fit<B: ComputeBackend>(
    backend: &B,
    spec: &B::Buffer2D,
    width: usize,
    height: usize,
) -> Result<(PhasePlane, Vec<Real>)> {
    let mut phase: Vec<Real> = backend.download(spec)?.iter().map(|c| c.arg()).collect();
    quarters_unwrap_phase(&mut phase, width, height);
    let plane = fit_plane_to_unwrapped(&phase, width, height, REGRESSION_CROP_FACTOR);
    Ok((plane, phase))
}

/// Two-direction spectral analysis: forward FFT, carrier peak search, one
/// Gaussian band-pass + inverse FFT per direction, then per-direction phase
/// unwrap and least-squares plane fit (host-side, f64). The returned
/// [`Detection`] carries the fitted planes and the measured unwrapped phase
/// maps they were fitted to.
pub fn analyze_two<B: ComputeBackend>(
    backend: &B,
    data: &[Complex32],
    layout: BufferLayout,
    sigma: Real,
    min_frequency: usize,
    max_frequency: usize,
    smoothing_sigma: Real,
) -> Result<Detection> {
    let (width, height) = (layout.width, layout.height);

    let (peaks_buffer, spec1, spec2) = {
        let mut job = backend.begin()?;
        let mut buffer = job.upload(data, layout)?;
        job.fft2d(&mut buffer)?;
        let peaks_buffer = job
            .peak_search(
                &mut buffer,
                min_frequency,
                max_frequency,
                smoothing_sigma,
                sigma,
            )?
            .ok_or_else(|| VernierError::Backend("no carrier peaks found in spectrum".into()))?;
        let mut spec1 = job.copy_buffer(&buffer)?;
        let mut spec2 = job.copy_buffer(&buffer)?;
        job.bandpass_from_peaks(&mut spec1, &peaks_buffer, 0, sigma)?;
        job.bandpass_from_peaks(&mut spec2, &peaks_buffer, 1, sigma)?;
        job.ifft2d(&mut spec1)?;
        job.ifft2d(&mut spec2)?;
        job.submit()?;
        (peaks_buffer, spec1, spec2)
    };

    let peaks_data = backend.download(&peaks_buffer)?;
    let (peak_x1, peak_y1) = (peaks_data[0].re as usize, peaks_data[1].re as usize);
    let (peak_x2, peak_y2) = (peaks_data[2].re as usize, peaks_data[3].re as usize);

    let (plane1, phase1) = unwrap_and_fit(backend, &spec1, width, height)?;
    let (plane2, phase2) = unwrap_and_fit(backend, &spec2, width, height)?;

    let dir1 = DirectionResult {
        plane: plane1,
        peak: plane1.peak_location(width, height),
        peak_bin: (peak_x1, peak_y1),
    };
    let dir2 = DirectionResult {
        plane: plane2,
        peak: plane2.peak_location(width, height),
        peak_bin: (peak_x2, peak_y2),
    };

    Ok(Detection {
        dir1,
        dir2,
        phase1,
        phase2,
        width,
        height,
    })
}
