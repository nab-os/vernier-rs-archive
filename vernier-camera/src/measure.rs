//! One view of the board, turned into pixel ↔ board correspondences.
//!
//! A tilted or fisheye view squeezes and turns the carriers across the frame,
//! so the single global band-pass `vernier-spectral` uses for a camera square
//! on to the pattern no longer holds the whole board. The phase is demodulated
//! locally instead, in a Gaussian window that follows the carriers:
//!
//! 1. Find both carriers in the spectrum of the whole frame.
//! 2. Seed where they fit best, then walk a coarse grid outwards, measuring the
//!    local frequency at each node and unwrapping from the node it was reached
//!    from.
//! 3. Fit a quadratic phase around each coarse node and demodulate a finer grid
//!    against it. A window centred on its node sees a frequency error as an odd
//!    phase ramp, which cancels; curvature is even and would not, so the
//!    reference carries it.
//! 4. Read the code with the decoder in `vernier-pose`, which places the
//!    measured lattice on the board.
//! 5. Paint the coding squares back to a plain checkerboard and measure the
//!    fine grid again, refitting the curvature on the fine nodes themselves.
//!
//! The windows go to a [`LocalDemodulator`] in batches: the seeds, each wave
//! of the walk, each pass of the fine grid. [`CpuDemodulator`] sums them in
//! `f64` on the CPU; `vernier-gpu`'s `GpuBackend` on a GPU, through
//! [`measure_view_traced_with`].

use std::collections::{HashMap, HashSet};

use nalgebra::{SMatrix, SVector};
use rayon::prelude::*;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;
use vernier_core::scalar::consts::{PI, TAU};
use vernier_core::{
    CarrierModel as Carrier, DemodWindow, FieldDemod, LocalDemodulator, Real, WindowDemod as Demod,
};
use vernier_patterns::checkerboard::Checkerboard;
use vernier_pose::checkerboard::{CheckerboardCode, CheckerboardError, extract_code_from_phases};

use crate::megarena::{self, MegarenaCode, MegarenaError};
use crate::target::{PatternKind, Printed, Target};

/// Carrier amplitude over local contrast below which a window is not on the
/// board. A clean board reads about 0.3 to 0.5, clutter well under 0.1.
const MIN_QUALITY: Real = 0.15;

/// Coarse grid step, in carrier periods.
const COARSE_STEP: Real = 2.0;

/// Gaussian window of the coarse walk and of the fine measurement, in carrier
/// periods.
const COARSE_WINDOW: Real = 1.0;
const FINE_WINDOW: Real = 1.0;

/// Spectrum at three times a peak over the peak itself, above which the peak
/// was the code's line at a third of the carrier. Measured: a true carrier
/// under 0.2 (its third harmonic plus the code's lines there), the code's
/// line above 1.
const SUBHARMONIC_LIMIT: Real = 0.5;

/// A spectral peak counts as a carrier candidate when it stands out of its
/// ring at least this much relative to the most prominent one.
const PROMINENCE: Real = 0.2;

/// Fine points whose carrier sits further than this off the window centre, in
/// window sigmas, are dropped (see [`Demod::offset`]).
const MAX_OFFSET: Real = 0.2;

/// Fine nodes this close to where the board ends are dropped, in grid steps
/// (a carrier period each). A window that reaches past the pattern sees, say,
/// the paper's white margin on one side only, and its phase leans; the
/// carrier-offset test catches the worst of it, not all.
const EDGE_MARGIN: i64 = 2;
/// The code is read first on the squares this many grid steps inside the
/// board's end (see [`decode`]).
const DECODE_MARGIN: i64 = 1;

/// Fine passes with the local phase model refitted on the fine nodes.
const REFIT_PASSES: usize = 2;
const REFIT_STAGES: [&str; REFIT_PASSES] = ["refit 1", "refit 2"];

/// Carrier pairs tried, strongest first, before giving up on a frame.
const CARRIER_CANDIDATES: usize = 4;

/// Fewer correspondences than this and the view is not worth keeping.
const MIN_POINTS: usize = 30;

/// Times the walk may fail to reach a coarse node, from different parents,
/// before it stops trying.
const WALK_TRIES: u8 = 3;

/// Missing nodes in a stretch this large or larger are taken for the end of
/// the board, or for something covering it, rather than for a hole in it.
const HOLE_LIMIT: usize = 9;

/// Lattice squares either way that a blurred square can reach.
const BLUR_REACH: i64 = 3;

/// One measured point: where it is in the frame and where it is on the board.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointMatch {
    /// In pixels.
    pub pixel: [Real; 2],
    /// In the board's own units, from [`Target::board_point`].
    pub board: [Real; 2],
}

/// Everything measured in one frame.
#[derive(Clone, Debug)]
pub struct View {
    pub width: usize,
    pub height: usize,
    pub points: Vec<PointMatch>,
    /// When the code was read, `points` are in the board's own frame. When it
    /// was not, they sit on the right lattice in an unknown quarter-turn and
    /// offset, which calibration absorbs into the view's pose but which makes
    /// the pose itself meaningless.
    ///
    /// Both axes carry the same sequence, so the board is its own mirror image
    /// about its diagonal. A mirrored picture (a webcam's selfie flip) reads
    /// as well as a straight one, with `x` and `y` swapped. The same holds
    /// for a megarena.
    pub code: Result<Code, CodeError>,
    /// Carrier period where the walk started, in pixels.
    pub period: Real,
    /// Defocus fitted while painting the code back, Gaussian sigma in pixels;
    /// the next frame's blur fit starts from it. None without a code.
    pub defocus: Option<Real>,
    /// The carrier pair the walk started from, for the next frame.
    carriers: [[Real; 2]; 2],
}

impl View {
    /// Whether the code was read, so the points are in the board's own frame.
    pub fn is_absolute(&self) -> bool {
        self.code.is_ok()
    }
}

/// Where the measured lattice sits on the board, as the target's decoder
/// found it.
#[derive(Clone, Debug, PartialEq)]
pub enum Code {
    Checkerboard(CheckerboardCode),
    Megarena(MegarenaCode),
}

impl Code {
    /// Bits that disagreed with the code along either axis.
    pub fn bit_errors(&self) -> (usize, usize) {
        match self {
            Self::Checkerboard(c) => c.bit_errors,
            Self::Megarena(c) => c.bit_errors,
        }
    }
}

/// Why the code was not read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CodeError {
    Checkerboard(CheckerboardError),
    Megarena(MegarenaError),
}

impl std::fmt::Display for CodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Checkerboard(e) => e.fmt(f),
            Self::Megarena(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for CodeError {}

/// Why a frame gave no view.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MeasureError {
    /// The target's LFSR order has no pattern (checkerboard or megarena).
    UnsupportedOrder(u32),
    /// No pair of crossed carriers in the spectrum.
    NoCarrier,
    /// The carriers were found but no part of the frame holds them cleanly.
    NoBoard,
    /// Fewer than the minimum of points survived; holds how many did.
    TooFewPoints(usize),
    /// The demodulator failed: a device lost, say.
    Backend,
    /// The pixels don't fill a `width × height` frame of at least 2×2.
    BadFrame {
        width: usize,
        height: usize,
        len: usize,
    },
}

impl std::fmt::Display for MeasureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedOrder(n) => write!(f, "unsupported LFSR order {n}"),
            Self::NoCarrier => write!(f, "no checkerboard carrier in the image"),
            Self::NoBoard => write!(f, "no region of the image holds the board cleanly"),
            Self::TooFewPoints(n) => write!(f, "only {n} points measured"),
            Self::Backend => write!(f, "the demodulator failed"),
            Self::BadFrame { width, height, len } => {
                write!(f, "{len} pixels don't make a {width}×{height} frame")
            }
        }
    }
}

impl std::error::Error for MeasureError {}

/// What one measurement went through, for debugging: the spectral search and
/// every carrier pair followed. See [`measure_view_traced`].
#[derive(Clone, Debug, Default)]
pub struct Trace {
    /// Whether the whole spectrum was searched; a video frame skips it when
    /// the previous frame's carriers still read the code.
    pub searched: bool,
    pub search_ms: f64,
    /// Peaks that stood out of their ring, strongest first.
    pub peaks: Vec<Peak>,
    /// Log magnitude of the frame's spectrum, `width × height` row-major with
    /// zero frequency at the centre. Only with images.
    pub spectrum: Option<Vec<f32>>,
    /// In the order they were followed.
    pub attempts: Vec<Attempt>,
    /// The attempt the result came from, if any got as far as points.
    pub chosen: Option<usize>,
}

/// A peak of the frame's spectrum that stood out of its ring.
#[derive(Clone, Copy, Debug)]
pub struct Peak {
    /// Radians per pixel.
    pub k: [Real; 2],
    /// Over the mean of its ring.
    pub score: Real,
    /// It was the code's line at a third of the carrier, and the carrier
    /// three times it was taken instead.
    pub lifted: bool,
}

/// A carrier pair followed through the walk, the fine grid and the code.
#[derive(Clone, Debug, Default)]
pub struct Attempt {
    /// Radians per pixel.
    pub carriers: [[Real; 2]; 2],
    /// The pair came from the previous frame, not from the spectrum.
    pub from_previous: bool,
    /// Carrier period, in pixels.
    pub period: Real,
    /// Where the walk started and its quality there.
    pub seed: Option<([Real; 2], Real)>,
    pub coarse_step: usize,
    /// Coarse nodes the walk reached, with their quality.
    pub coarse: Vec<([Real; 2], Real)>,
    /// Coarse nodes tried and refused.
    pub coarse_refused: Vec<[Real; 2]>,
    pub fine_step: usize,
    /// Fine nodes left after each stage.
    pub funnel: Vec<(&'static str, usize)>,
    /// Fine nodes dropped, and at which stage.
    pub dropped: Vec<([Real; 2], &'static str)>,
    /// Per point of the view: carrier quality and window offset.
    pub point_quality: Vec<(Real, Real)>,
    /// The pattern followed, which sets how phases map to its lattice.
    pub kind: PatternKind,
    pub code: Option<Result<Code, CodeError>>,
    pub error: Option<MeasureError>,
    /// Milliseconds per stage.
    pub timings: Vec<(&'static str, f64)>,
    /// Both phases per pixel, NaN off the board. Only with images.
    pub phase: Option<Vec<[f32; 2]>>,
    /// The frame with the coding squares painted back. Only with images and
    /// a code.
    pub restored: Option<Vec<f32>>,
    /// Defocus fitted while painting them back, Gaussian sigma in pixels.
    pub defocus: Option<Real>,
    /// Times the blur fit evaluated its misfit, a measure of its cost.
    pub defocus_misfits: usize,
}

impl Attempt {
    /// Square of the board pattern under a pixel, from the phase maps and the
    /// code; integers at square centres.
    pub fn pattern_square(&self, x: usize, y: usize, width: usize) -> Option<(Real, Real)> {
        let [p1, p2] = self.phase.as_ref()?[y * width + x];
        if !(p1.is_finite() && p2.is_finite()) {
            return None;
        }
        Some(self.to_pattern([p1 as Real, p2 as Real]))
    }

    /// Square of the board pattern at a pair of unwrapped phases.
    pub fn to_pattern(&self, phase: [Real; 2]) -> (Real, Real) {
        let code = match &self.code {
            Some(Ok(code)) => Some(code),
            _ => None,
        };
        pattern_square(self.kind, code, phase)
    }
}

/// Square (dot for a megarena) of the board pattern at a pair of unwrapped
/// phases: the measured lattice placed on the board by the code when there is
/// one, else the measured lattice as it is.
fn pattern_square(kind: PatternKind, code: Option<&Code>, phase: [Real; 2]) -> (Real, Real) {
    let measured = match kind {
        PatternKind::Checkerboard => lattice(phase),
        PatternKind::Megarena => megarena::lattice(phase),
    };
    match code {
        Some(Code::Checkerboard(code)) => code.to_pattern(measured),
        Some(Code::Megarena(code)) => code.to_pattern(measured),
        None => measured,
    }
}

/// The demodulated field of a frame: at every pixel and for each carrier, a
/// phase and how much carrier there is.
#[derive(Clone, Debug)]
pub struct Field {
    pub width: usize,
    pub height: usize,
    /// Per carrier, the phase in radians, unwrapped where the attempt
    /// measured the board.
    pub phase: [Vec<f32>; 2],
    /// Per carrier, the carrier amplitude over the local contrast, the
    /// quality the measurement thresholds at 0.15. A clean board reads
    /// about 0.3 to 0.5.
    pub amplitude: [Vec<f32>; 2],
    /// The attempt's carriers, the global frequency of each, in radians per
    /// pixel.
    pub carriers: [[Real; 2]; 2],
}

/// Demodulates every pixel of a frame in a Gaussian window of the fine grid's
/// size, against the phase the attempt measured there, or off the board
/// against its global carrier. Needs the attempt's phase maps (a trace kept
/// with images).
pub fn demodulated_field(
    intensity: &[f32],
    width: usize,
    height: usize,
    attempt: &Attempt,
) -> Option<Field> {
    demodulated_field_with(&CpuDemodulator, intensity, width, height, attempt)
}

/// [`demodulated_field`] on the given demodulator. None also when it fails.
pub fn demodulated_field_with<D: LocalDemodulator>(
    demodulator: &D,
    intensity: &[f32],
    width: usize,
    height: usize,
    attempt: &Attempt,
) -> Option<Field> {
    let maps = attempt.phase.as_ref()?;
    // The measured phase on the board, the global plane wave off it.
    let reference = |c: usize| -> Vec<Real> {
        let k = attempt.carriers[c];
        maps.par_iter()
            .enumerate()
            .map(|(i, m)| {
                let p = m[c];
                if p.is_finite() {
                    p as Real
                } else {
                    k[0] * (i % width) as Real + k[1] * (i / width) as Real
                }
            })
            .collect()
    };
    let references = [reference(0), reference(1)];
    let frame = demodulator.load(intensity, width, height).ok()?;
    let found = demodulator
        .demodulate_field(
            &frame,
            [&references[0], &references[1]],
            FINE_WINDOW * attempt.period,
        )
        .ok()?;
    let unwrapped = |c: usize| -> Vec<f32> {
        references[c]
            .par_iter()
            .zip(&found.phase[c])
            .map(|(&r, &p)| (r + p) as f32)
            .collect()
    };
    let amplitude =
        |c: usize| -> Vec<f32> { found.amplitude[c].iter().map(|&a| a as f32).collect() };
    Some(Field {
        width,
        height,
        phase: [unwrapped(0), unwrapped(1)],
        amplitude: [amplitude(0), amplitude(1)],
        carriers: attempt.carriers,
    })
}

/// Measures the time each stage takes.
struct Stopwatch(std::time::Instant);

impl Stopwatch {
    fn start() -> Self {
        Self(std::time::Instant::now())
    }

    /// Milliseconds since the last lap.
    fn lap(&mut self) -> f64 {
        let now = std::time::Instant::now();
        let ms = (now - self.0).as_secs_f64() * 1000.0;
        self.0 = now;
        ms
    }
}

/// A grayscale frame, row-major: what [`CpuDemodulator`] reads.
#[derive(Clone, Copy)]
pub struct Image<'a> {
    data: &'a [f32],
    width: usize,
    height: usize,
}

/// The reference [`LocalDemodulator`]: every window summed in `f64` on the
/// CPU, windows spread over threads.
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuDemodulator;

impl LocalDemodulator for CpuDemodulator {
    type Frame<'a> = Image<'a>;

    fn load<'a>(
        &'a self,
        data: &'a [f32],
        width: usize,
        height: usize,
    ) -> vernier_core::Result<Image<'a>> {
        Ok(Image {
            data,
            width,
            height,
        })
    }

    fn demodulate_windows(
        &self,
        image: &Image<'_>,
        windows: &[DemodWindow],
    ) -> vernier_core::Result<Vec<Demod>> {
        Ok(windows
            .par_iter()
            .map(|w| demodulate(image, w.x, w.y, &w.carriers, w.sigma))
            .collect())
    }

    fn demodulate_field(
        &self,
        image: &Image<'_>,
        references: [&[Real]; 2],
        sigma: Real,
    ) -> vernier_core::Result<FieldDemod> {
        let (width, height) = (image.width, image.height);
        let value = |i: usize| image.data[i] as Real;
        let local_mean = blur_with(width, height, sigma, value);
        let local_square = blur_with(width, height, sigma, |i| value(i) * value(i));
        let demodulate_carrier = |reference: &[Real]| {
            let blurred = |f: &(dyn Fn(usize) -> Real + Sync)| blur_with(width, height, sigma, f);
            let signal_re = blurred(&|i| value(i) * reference[i].cos());
            let signal_im = blurred(&|i| -value(i) * reference[i].sin());
            let unit_re = blurred(&|i| reference[i].cos());
            let unit_im = blurred(&|i| -reference[i].sin());
            let mut phase = vec![0.0; width * height];
            let mut amplitude = vec![0.0; width * height];
            phase
                .par_iter_mut()
                .zip(amplitude.par_iter_mut())
                .enumerate()
                .for_each(|(i, (p, a))| {
                    // Σw·(I − mean)·e^{−iψ} over Σw, as `demodulate` does.
                    let z = Complex64::new(
                        signal_re[i] - local_mean[i] * unit_re[i],
                        signal_im[i] - local_mean[i] * unit_im[i],
                    );
                    let deviation = (local_square[i] - local_mean[i] * local_mean[i])
                        .max(0.0)
                        .sqrt();
                    *p = z.arg();
                    *a = if deviation > 0.0 {
                        z.norm() / deviation
                    } else {
                        0.0
                    };
                });
            (phase, amplitude)
        };
        let (phase1, amplitude1) = demodulate_carrier(references[0]);
        let (phase2, amplitude2) = demodulate_carrier(references[1]);
        Ok(FieldDemod {
            phase: [phase1, phase2],
            amplitude: [amplitude1, amplitude2],
        })
    }
}

/// A frame loaded on a demodulator, with the pixels the host-side stages
/// (the spectrum, the code, the restoration) read.
struct Frame<'a, D: LocalDemodulator + 'a> {
    image: Image<'a>,
    demodulator: &'a D,
    loaded: D::Frame<'a>,
}

impl<'a, D: LocalDemodulator> Frame<'a, D> {
    fn load(
        demodulator: &'a D,
        data: &'a [f32],
        width: usize,
        height: usize,
    ) -> Result<Self, MeasureError> {
        // The grid and the windows index width × height pixels and step
        // between neighbours, so anything smaller or shorter would panic.
        if width < 2 || height < 2 || width.checked_mul(height) != Some(data.len()) {
            return Err(MeasureError::BadFrame {
                width,
                height,
                len: data.len(),
            });
        }
        Ok(Self {
            image: Image {
                data,
                width,
                height,
            },
            demodulator,
            loaded: demodulator
                .load(data, width, height)
                .map_err(|_| MeasureError::Backend)?,
        })
    }

    /// One [`Demod`] per window, in order.
    fn demodulate(&self, windows: &[DemodWindow]) -> Result<Vec<Demod>, MeasureError> {
        if windows.is_empty() {
            return Ok(Vec::new());
        }
        self.demodulator
            .demodulate_windows(&self.loaded, windows)
            .map_err(|_| MeasureError::Backend)
    }
}

/// A window of `sigma` pixels at a pixel, against `carriers`.
fn window(x: usize, y: usize, carriers: &[Carrier; 2], sigma: Real) -> DemodWindow {
    DemodWindow {
        x,
        y,
        sigma,
        carriers: *carriers,
    }
}

/// Plane-wave models of both carriers.
fn planes(k: [[Real; 2]; 2]) -> [Carrier; 2] {
    [Carrier::plane(k[0]), Carrier::plane(k[1])]
}

/// An angle brought into `-π..=π`.
fn wrap(angle: Real) -> Real {
    angle - TAU * (angle / TAU).round()
}

/// One carrier's window sums in [`demodulate`], with `w` the window weight,
/// `I` the intensity, `e = e^{−iψ}` the reference phasor and `(qx, qy)` the
/// offset from the window centre.
#[derive(Clone, Copy, Default)]
struct WindowSums {
    /// Σw·I·e
    signal: Complex64,
    /// Σw·e
    unit: Complex64,
    /// Σw·I·e·qx
    signal_x: Complex64,
    /// Σw·e·qx
    unit_x: Complex64,
    /// Σw·I·e·qy
    signal_y: Complex64,
    /// Σw·e·qy
    unit_y: Complex64,
}

/// Phase of both carriers at a pixel, against a reference that follows each
/// carrier's local model, under a Gaussian window of `sigma` pixels. The
/// window is cut at the frame edge.
fn demodulate(image: &Image, x: usize, y: usize, carriers: &[Carrier; 2], sigma: Real) -> Demod {
    let zero = Complex64::new(0.0, 0.0);
    let radius = (3.0 * sigma).ceil() as usize;
    let (x0, x1) = (x.saturating_sub(radius), (x + radius).min(image.width - 1));
    let (y0, y1) = (y.saturating_sub(radius), (y + radius).min(image.height - 1));
    let exponent = -0.5 / (sigma * sigma);
    let weights_x: Vec<Real> = (x0..=x1)
        .map(|px| ((px as Real - x as Real).powi(2) * exponent).exp())
        .collect();
    let first_qx = x0 as Real - x as Real;

    let mut sums = [WindowSums::default(); 2];
    let (mut weight_total, mut intensity_total, mut square_total) = (0.0, 0.0, 0.0);
    for py in y0..=y1 {
        let qy = py as Real - y as Real;
        let weight_y = (qy * qy * exponent).exp();
        let row = &image.data[py * image.width + x0..=py * image.width + x1];
        for (&v, &weight_x) in row.iter().zip(&weights_x) {
            let (w, v) = (weight_y * weight_x, v as Real);
            weight_total += w;
            intensity_total += w * v;
            square_total += w * v * v;
        }
        // Along the row the reference is ψ(qx) = a + b·qx + c·qx², so its
        // phasor is stepped by a rotation that itself rotates, with no
        // trigonometry per pixel.
        for (carrier, total) in carriers.iter().zip(&mut sums) {
            let a = carrier.k[1] * qy + 0.5 * carrier.h[2] * qy * qy;
            let b = carrier.k[0] + carrier.h[1] * qy;
            let c = 0.5 * carrier.h[0];
            let mut phasor =
                Complex64::from_polar(1.0, -(a + b * first_qx + c * first_qx * first_qx));
            let mut turn = Complex64::from_polar(1.0, -(b + c * (2.0 * first_qx + 1.0)));
            let turn_step = Complex64::from_polar(1.0, -2.0 * c);
            let (mut signal, mut unit, mut signal_x, mut unit_x) = (zero, zero, zero, zero);
            let mut qx = first_qx;
            for (&v, &weight_x) in row.iter().zip(&weights_x) {
                let weighted = phasor * (weight_y * weight_x);
                let weighted_signal = weighted * v as Real;
                signal += weighted_signal;
                unit += weighted;
                signal_x += weighted_signal * qx;
                unit_x += weighted * qx;
                phasor *= turn;
                turn *= turn_step;
                qx += 1.0;
            }
            total.signal += signal;
            total.unit += unit;
            total.signal_x += signal_x;
            total.unit_x += unit_x;
            total.signal_y += signal * qy;
            total.unit_y += unit * qy;
        }
    }

    let mean = intensity_total / weight_total;
    let deviation = (square_total / weight_total - mean * mean).max(0.0).sqrt();
    let mut out = Demod {
        phase: [0.0; 2],
        quality: [0.0; 2],
        offset: 0.0,
    };
    for (c, s) in sums.iter().enumerate() {
        // Subtracting the mean's share takes out the window's DC leaking
        // through the reference.
        let z = s.signal - s.unit * mean;
        let power = z.norm_sqr();
        out.phase[c] = z.arg();
        out.quality[c] = if deviation > 0.0 {
            z.norm() / (weight_total * deviation)
        } else {
            0.0
        };
        // The carrier's centre of weight within the window, projected on its
        // own phasor.
        if power > 0.0 {
            let shift_x = ((s.signal_x - s.unit_x * mean) * z.conj()).re / power;
            let shift_y = ((s.signal_y - s.unit_y * mean) * z.conj()).re / power;
            out.offset = out.offset.max(shift_x.hypot(shift_y) / sigma);
        } else {
            out.offset = Real::INFINITY;
        }
    }
    out
}

/// What the spectral search found.
struct Search {
    /// Candidate carrier pairs, in radians per pixel, strongest first, each
    /// with its second carrier turned so the pair has the pattern's own
    /// handedness (`k1 × k2 < 0`).
    pairs: Vec<[[Real; 2]; 2]>,
    peaks: Vec<Peak>,
    spectrum: Option<Vec<f32>>,
}

/// Searches the spectrum of the whole frame for crossed carrier pairs. With
/// `images`, also keeps the log spectrum.
fn find_carriers(image: &Image, images: bool) -> Search {
    let spectrum = Spectrum::new(image);
    let band = spectrum.band();
    let mut pairs: Vec<[[Real; 2]; 2]> = Vec::new();
    let mut found = Vec::new();
    for peak in spectrum.peaks(&band) {
        let first = spectrum.lift(peak);
        found.push(Peak {
            k: spectrum.wavevector(peak),
            score: spectrum.score(peak),
            lifted: first != peak,
        });
        if pairs.len() == CARRIER_CANDIDATES {
            continue;
        }
        let k1 = spectrum.wavevector(first);
        let Some(k2) = spectrum.partner(&band, first) else {
            continue;
        };
        // The partner's own peak finds the same pair again.
        if pairs
            .iter()
            .any(|p| p.iter().any(|&k| same_line(k, k1) || same_line(k, k2)))
        {
            continue;
        }
        pairs.push([k1, k2]);
    }
    Search {
        pairs,
        peaks: found,
        spectrum: images.then(|| spectrum.log_image()),
    }
}

/// Whether two frequencies are, within 5 %, the same or opposite: the same
/// spectral line.
fn same_line(a: [Real; 2], b: [Real; 2]) -> bool {
    let tolerance = 0.05 * norm(a);
    norm([a[0] - b[0], a[1] - b[1]]) < tolerance || norm([a[0] + b[0], a[1] + b[1]]) < tolerance
}

/// A frequency index of an `n`-point FFT, as a signed frequency.
fn signed_frequency(f: usize, n: usize) -> i64 {
    if f > n / 2 {
        f as i64 - n as i64
    } else {
        f as i64
    }
}

/// The Hann-windowed spectrum of a frame, with what the carrier search reads
/// off it. Every per-frequency table is stored by column, `[fx * height +
/// fy]`, and frequencies are signed and wrapped.
struct Spectrum {
    width: usize,
    height: usize,
    magnitude: Vec<Real>,
    /// Magnitude summed over each frequency's 3×3 neighbourhood, so a peak
    /// that falls between bins still reads whole.
    smoothed: Vec<Real>,
    /// `smoothed` over the mean of its ring (see [`Spectrum::ring_scores`]).
    scores: Vec<Real>,
}

impl Spectrum {
    fn new(image: &Image) -> Self {
        let magnitude: Vec<Real> = windowed_fft(image).par_iter().map(|c| c.norm()).collect();
        let mut spectrum = Self {
            width: image.width,
            height: image.height,
            magnitude,
            smoothed: Vec::new(),
            scores: Vec::new(),
        };
        spectrum.smoothed = spectrum.smooth();
        spectrum.scores = spectrum.ring_scores();
        spectrum
    }

    fn index(&self, fx: i64, fy: i64) -> usize {
        fx.rem_euclid(self.width as i64) as usize * self.height
            + fy.rem_euclid(self.height as i64) as usize
    }

    /// Radius in cycles per pixel.
    fn radius(&self, fx: i64, fy: i64) -> Real {
        ((fx as Real / self.width as Real).powi(2) + (fy as Real / self.height as Real).powi(2))
            .sqrt()
    }

    fn short_side(&self) -> Real {
        self.width.min(self.height) as Real
    }

    /// Rings one frequency bin of the short side wide.
    fn ring_count(&self) -> usize {
        (0.75 * self.short_side()) as usize + 2
    }

    fn ring(&self, fx: i64, fy: i64) -> usize {
        ((self.radius(fx, fy) * self.short_side()) as usize).min(self.ring_count() - 1)
    }

    /// Radians per pixel.
    fn wavevector(&self, (fx, fy): (i64, i64)) -> [Real; 2] {
        [
            TAU * fx as Real / self.width as Real,
            TAU * fy as Real / self.height as Real,
        ]
    }

    fn score(&self, (fx, fy): (i64, i64)) -> Real {
        self.scores[self.index(fx, fy)]
    }

    fn strength(&self, (fx, fy): (i64, i64)) -> Real {
        self.smoothed[self.index(fx, fy)]
    }

    fn smooth(&self) -> Vec<Real> {
        let (w, h) = (self.width, self.height);
        let mut smoothed = vec![0.0; w * h];
        smoothed
            .par_chunks_mut(h)
            .enumerate()
            .for_each(|(fx, column)| {
                let sx = signed_frequency(fx, w);
                for (fy, value) in column.iter_mut().enumerate() {
                    let sy = signed_frequency(fy, h);
                    for dx in -1..=1 {
                        for dy in -1..=1 {
                            *value += self.magnitude[self.index(sx + dx, sy + dy)];
                        }
                    }
                }
            });
        smoothed
    }

    /// `smoothed` normalized by the mean of each ring, so a sharp carrier
    /// beats the broad low-frequency clutter of a room.
    fn ring_scores(&self) -> Vec<Real> {
        let (w, h) = (self.width, self.height);
        let rings = self.ring_count();
        let mut ring_sum = vec![0.0; rings];
        let mut ring_size = vec![0usize; rings];
        for fx in 0..w {
            for fy in 0..h {
                let ring = self.ring(signed_frequency(fx, w), signed_frequency(fy, h));
                ring_sum[ring] += self.smoothed[fx * h + fy];
                ring_size[ring] += 1;
            }
        }
        // A ring quieter than the average one is held to the average: in a
        // blurred frame with little noise the high rings are all but empty, and
        // the faintest harmonic there would outscore the carrier itself.
        let ring_means: Vec<Real> = ring_sum
            .iter()
            .zip(&ring_size)
            .map(|(&s, &n)| s / n.max(1) as Real)
            .collect();
        let floor = ring_means.iter().sum::<Real>() / rings as Real;
        (0..w * h)
            .into_par_iter()
            .map(|i| {
                let ring = self.ring(signed_frequency(i / h, w), signed_frequency(i % h, h));
                self.smoothed[i] / ring_means[ring].max(floor).max(1e-300)
            })
            .collect()
    }

    /// The frequencies a carrier may have: above the lowest few bins, where
    /// the room's shading lives, and below 0.35 cycles per pixel.
    fn band(&self) -> Vec<(i64, i64)> {
        let (w, h) = (self.width, self.height);
        let (min_radius, max_radius) = (6.0 / self.short_side(), 0.35);
        (0..w)
            .into_par_iter()
            .flat_map_iter(|fx| {
                (0..h).map(move |fy| (signed_frequency(fx, w), signed_frequency(fy, h)))
            })
            .filter(|&(sx, sy)| (min_radius..=max_radius).contains(&self.radius(sx, sy)))
            .collect()
    }

    /// Local maxima of the band's upper half plane that stand out of their
    /// ring, strongest first. Other periodic things in the frame (another
    /// pattern, a blind, a keyboard) make peaks of their own, so all are kept
    /// for the caller to try in turn.
    fn peaks(&self, band: &[(i64, i64)]) -> Vec<(i64, i64)> {
        let top = band
            .iter()
            .map(|&(sx, sy)| self.score((sx, sy)))
            .fold(0.0, Real::max);
        let upper_half = |sx: i64, sy: i64| sy > 0 || (sy == 0 && sx > 0);
        let mut peaks: Vec<(i64, i64)> = band
            .par_iter()
            .copied()
            .filter(|&(sx, sy)| upper_half(sx, sy) && self.score((sx, sy)) >= PROMINENCE * top)
            .filter(|&(sx, sy)| {
                let v = self.strength((sx, sy));
                (-1..=1).all(|dx| {
                    (-1..=1).all(|dy| (dx, dy) == (0, 0) || self.strength((sx + dx, sy + dy)) < v)
                })
            })
            .collect();
        peaks.sort_by(|a, b| self.strength(*b).total_cmp(&self.strength(*a)));
        peaks
    }

    /// The strongest frequency of the band that `accept` allows, among those
    /// that stand out of their ring: on a clean board the harmonics stand out
    /// too, but they are weaker.
    fn strongest(
        &self,
        band: &[(i64, i64)],
        accept: &(dyn Fn(i64, i64) -> bool + Sync),
    ) -> Option<(i64, i64)> {
        let candidates: Vec<(i64, i64)> = band
            .par_iter()
            .copied()
            .filter(|&(sx, sy)| accept(sx, sy))
            .collect();
        let top = candidates
            .iter()
            .map(|&f| self.score(f))
            .fold(0.0, Real::max);
        candidates
            .into_iter()
            .filter(|&f| self.score(f) >= PROMINENCE * top)
            .max_by(|&a, &b| self.strength(a).total_cmp(&self.strength(b)))
    }

    /// The crossing carrier of `first`, in radians per pixel, turned to the
    /// pattern's handedness: the strongest peak of about the same radius at
    /// more than 60° from it.
    fn partner(&self, band: &[(i64, i64)], first: (i64, i64)) -> Option<[Real; 2]> {
        let k1 = self.wavevector(first);
        let r1 = self.radius(first.0, first.1);
        let second = self.strongest(band, &|fx, fy| {
            let k = self.wavevector((fx, fy));
            let r = self.radius(fx, fy);
            let cos = (k[0] * k1[0] + k[1] * k1[1]) / (norm(k) * norm(k1));
            (0.7 * r1..1.4 * r1).contains(&r) && cos.abs() < 0.5
        })?;
        let mut k2 = self.wavevector(second);
        if k1[0] * k2[1] - k1[1] * k2[0] > 0.0 {
            k2 = [-k2[0], -k2[1]];
        }
        Some(k2)
    }

    /// The carrier a peak stands for. The code puts a line at a third of each
    /// carrier, nearly as strong. If a peak is that line, the carrier is at
    /// three times it and towers over what a true carrier's third harmonic
    /// would be there; the strongest bin near the triple is taken instead.
    fn lift(&self, first: (i64, i64)) -> (i64, i64) {
        let (tx, ty) = (3 * first.0, 3 * first.1);
        if tx.unsigned_abs() >= (self.width / 2) as u64
            || ty.unsigned_abs() >= (self.height / 2) as u64
        {
            return first;
        }
        let reach = ((0.1 * ((tx * tx + ty * ty) as Real).sqrt()).round() as i64).max(2);
        let mut strongest = ((tx, ty), 0.0);
        for dx in -reach..=reach {
            for dy in -reach..=reach {
                let v = self.strength((tx + dx, ty + dy));
                if v > strongest.1 {
                    strongest = ((tx + dx, ty + dy), v);
                }
            }
        }
        let mut near = 0.0;
        for dx in -2..=2 {
            for dy in -2..=2 {
                near = Real::max(near, self.strength((tx + dx, ty + dy)));
            }
        }
        if near > SUBHARMONIC_LIMIT * self.strength(first) {
            strongest.0
        } else {
            first
        }
    }

    /// `ln(1 + |F|)`, row-major with zero frequency at the centre.
    fn log_image(&self) -> Vec<f32> {
        let (w, h) = (self.width, self.height);
        let mut out = vec![0.0f32; w * h];
        out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            let fy = y as i64 - (h / 2) as i64;
            for (x, v) in row.iter_mut().enumerate() {
                let fx = x as i64 - (w / 2) as i64;
                *v = (1.0 + self.magnitude[self.index(fx, fy)]).ln() as f32;
            }
        });
        out
    }
}

/// 2-D FFT of the frame less its mean, under a Hann window so the frame's
/// edges do not smear lines across the spectrum. Stored by column,
/// `[fx * height + fy]`.
fn windowed_fft(image: &Image) -> Vec<Complex64> {
    let (w, h) = (image.width, image.height);
    let zero = Complex64::new(0.0, 0.0);
    let mean = image.data.par_iter().map(|&v| v as Real).sum::<Real>() / (w * h) as Real;
    let hann = |i: usize, n: usize| 0.5 - 0.5 * (TAU * (i as Real + 0.5) / n as Real).cos();
    let mut planner = FftPlanner::<Real>::new();
    let (row_fft, column_fft) = (planner.plan_fft_forward(w), planner.plan_fft_forward(h));
    let mut rows = vec![zero; w * h];
    rows.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, v) in row.iter_mut().enumerate() {
            *v = Complex64::new(
                (image.data[y * w + x] as Real - mean) * hann(x, w) * hann(y, h),
                0.0,
            );
        }
        row_fft.process(row);
    });
    let mut spectrum = vec![zero; w * h];
    spectrum
        .par_chunks_mut(h)
        .enumerate()
        .for_each(|(x, column)| {
            for (y, v) in column.iter_mut().enumerate() {
                *v = rows[y * w + x];
            }
            column_fft.process(column);
        });
    spectrum
}

/// Euclidean length of a 2-vector.
fn norm(v: [Real; 2]) -> Real {
    v[0].hypot(v[1])
}

/// Carrier period of a local model, in pixels. Windows are sized by it, so a
/// foreshortened corner of the board gets windows as small as its squares.
fn period_of(carriers: &[Carrier; 2]) -> Real {
    TAU / (0.5 * (norm(carriers[0].k) + norm(carriers[1].k)))
}

/// The probes [`local_frequency`] reads at a node: windows half a period
/// either way, left, right, up and down. None when one would leave the frame.
fn frequency_probes(
    image: &Image,
    x: usize,
    y: usize,
    expected: &[Carrier; 2],
) -> Option<[DemodWindow; 4]> {
    let period = period_of(expected);
    let sigma = COARSE_WINDOW * period;
    let delta = ((0.5 * period).round() as usize).max(1);
    if x < delta || y < delta || x + delta >= image.width || y + delta >= image.height {
        return None;
    }
    let at = |px: usize, py: usize| window(px, py, expected, sigma);
    Some([
        at(x - delta, y),
        at(x + delta, y),
        at(x, y - delta),
        at(x, y + delta),
    ])
}

/// Local frequency of both carriers at a node, from the phase its
/// [`frequency_probes`] found, unwrapped against the frequency we expected.
fn local_frequency(
    expected: &[Carrier; 2],
    probes: &[DemodWindow; 4],
    found: &[Demod],
) -> [[Real; 2]; 2] {
    let [left, right, up, down] = [0, 1, 2, 3].map(|i| found[i]);
    let baseline = (probes[1].x - probes[0].x) as Real;
    let mut k = [[0.0; 2]; 2];
    for (c, carrier) in expected.iter().enumerate() {
        let e = carrier.k;
        k[c][0] = e[0] + wrap(right.phase[c] - left.phase[c] - e[0] * baseline) / baseline;
        k[c][1] = e[1] + wrap(down.phase[c] - up.phase[c] - e[1] * baseline) / baseline;
    }
    k
}

/// A node of the walk: both unwrapped phases, both local frequencies and the
/// worse carrier's quality.
#[derive(Clone, Copy, Debug)]
struct Node {
    phase: [Real; 2],
    k: [[Real; 2]; 2],
    quality: Real,
}

/// A grid node, `(gx, gy)`.
type GridNode = (usize, usize);

/// A square grid of nodes `step` pixels apart, centred on the frame.
struct Grid {
    step: usize,
    /// Pixel of node `(0, 0)`.
    offset: [usize; 2],
    /// Nodes along each axis.
    size: [usize; 2],
}

impl Grid {
    fn new(width: usize, height: usize, step: usize) -> Self {
        let size = [(width - 1) / step + 1, (height - 1) / step + 1];
        let offset = [
            (width - 1 - (size[0] - 1) * step) / 2,
            (height - 1 - (size[1] - 1) * step) / 2,
        ];
        Self { step, offset, size }
    }

    /// Pixel of a node.
    fn pixel(&self, (gx, gy): GridNode) -> (usize, usize) {
        (
            self.offset[0] + gx * self.step,
            self.offset[1] + gy * self.step,
        )
    }

    /// Pixel of a node, as reals.
    fn point(&self, node: GridNode) -> [Real; 2] {
        let (x, y) = self.pixel(node);
        [x as Real, y as Real]
    }

    /// The node nearest a pixel, if inside the grid.
    fn nearest(&self, x: Real, y: Real) -> Option<GridNode> {
        let gx = ((x - self.offset[0] as Real) / self.step as Real).round();
        let gy = ((y - self.offset[1] as Real) / self.step as Real).round();
        (gx >= 0.0 && gy >= 0.0 && (gx as usize) < self.size[0] && (gy as usize) < self.size[1])
            .then_some((gx as usize, gy as usize))
    }

    /// Every node, row by row.
    fn nodes(&self) -> impl Iterator<Item = GridNode> + '_ {
        (0..self.size[1]).flat_map(move |gy| (0..self.size[0]).map(move |gx| (gx, gy)))
    }

    /// The node `(dx, dy)` steps from `node`, if inside the grid.
    fn offset_node(&self, node: GridNode, dx: i64, dy: i64) -> Option<GridNode> {
        let (nx, ny) = (node.0 as i64 + dx, node.1 as i64 + dy);
        (nx >= 0 && ny >= 0 && (nx as usize) < self.size[0] && (ny as usize) < self.size[1])
            .then_some((nx as usize, ny as usize))
    }

    /// The four-way neighbours of a node inside the grid.
    fn neighbours(&self, node: GridNode) -> impl Iterator<Item = GridNode> + '_ {
        [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)]
            .into_iter()
            .filter_map(move |(dx, dy)| self.offset_node(node, dx, dy))
    }

    /// Whether a node is on the grid's outer ring.
    fn on_edge(&self, node: GridNode) -> bool {
        node.0 == 0 || node.1 == 0 || node.0 + 1 == self.size[0] || node.1 + 1 == self.size[1]
    }
}

/// Walks the coarse grid out from the best seed, breadth first, each wave in
/// one batch. A node joins when both carriers are clean there and its phase
/// lands within a quarter turn of what its parent predicts.
fn walk<D: LocalDemodulator>(
    frame: &Frame<D>,
    grid: &Grid,
    start: [[Real; 2]; 2],
    period: Real,
    attempt: &mut Attempt,
) -> Result<HashMap<GridNode, Node>, MeasureError> {
    let (seed, seed_quality) = best_seed(frame, grid, start, period)?;
    attempt.seed = Some((grid.point(seed), seed_quality));
    if seed_quality < MIN_QUALITY {
        return Err(MeasureError::NoBoard);
    }
    let mut nodes = HashMap::new();
    nodes.insert(seed, measure_seed(frame, grid, seed, start)?);
    let refused = grow(frame, grid, &mut nodes, seed)?;
    attempt.coarse = nodes
        .iter()
        .map(|(&n, node)| (grid.point(n), node.quality))
        .collect();
    attempt.coarse_refused = refused.keys().map(|&n| grid.point(n)).collect();
    Ok(nodes)
}

/// The coarse node where the global carriers read best, and their quality
/// there. Only nodes away from the frame edge are tried, where the window is
/// whole and the frequency probe has room.
fn best_seed<D: LocalDemodulator>(
    frame: &Frame<D>,
    grid: &Grid,
    start: [[Real; 2]; 2],
    period: Real,
) -> Result<(GridNode, Real), MeasureError> {
    let image = &frame.image;
    let sigma = COARSE_WINDOW * period;
    let start_planes = planes(start);
    let margin = (2.0 * sigma).ceil() as usize;
    let candidates: Vec<GridNode> = grid
        .nodes()
        .filter(|&n| {
            let (x, y) = grid.pixel(n);
            x >= margin && y >= margin && x + margin < image.width && y + margin < image.height
        })
        .collect();
    let windows: Vec<DemodWindow> = candidates
        .iter()
        .map(|&node| {
            let (x, y) = grid.pixel(node);
            window(x, y, &start_planes, sigma)
        })
        .collect();
    let found = frame.demodulate(&windows)?;
    candidates
        .into_iter()
        .zip(found.iter().map(Demod::quality))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .ok_or(MeasureError::NoBoard)
}

/// The seed node: its local frequency refined from the global carriers over
/// three probes, then its phase against it.
fn measure_seed<D: LocalDemodulator>(
    frame: &Frame<D>,
    grid: &Grid,
    seed: GridNode,
    start: [[Real; 2]; 2],
) -> Result<Node, MeasureError> {
    let (sx, sy) = grid.pixel(seed);
    let mut k = start;
    for _ in 0..3 {
        let expected = planes(k);
        let probes =
            frequency_probes(&frame.image, sx, sy, &expected).ok_or(MeasureError::NoBoard)?;
        k = local_frequency(&expected, &probes, &frame.demodulate(&probes)?);
    }
    let carriers = planes(k);
    let sigma = COARSE_WINDOW * period_of(&carriers);
    let found = frame.demodulate(&[window(sx, sy, &carriers, sigma)])?[0];
    Ok(Node {
        phase: found.phase,
        k,
        quality: found.quality(),
    })
}

/// Grows the walk from `seed` until no wave adds a node. Each frontier node
/// is reached from its best-quality neighbour already in the walk; a node
/// refused [`WALK_TRIES`] times is given up. Returns the refused nodes with
/// how many times each was.
fn grow<D: LocalDemodulator>(
    frame: &Frame<D>,
    grid: &Grid,
    nodes: &mut HashMap<GridNode, Node>,
    seed: GridNode,
) -> Result<HashMap<GridNode, u8>, MeasureError> {
    let mut refused: HashMap<GridNode, u8> = HashMap::new();
    let mut fresh = vec![seed];
    while !fresh.is_empty() {
        let mut frontier: Vec<GridNode> = fresh
            .iter()
            .flat_map(|&n| grid.neighbours(n))
            .filter(|n| {
                !nodes.contains_key(n) && refused.get(n).is_none_or(|&tries| tries < WALK_TRIES)
            })
            .collect();
        frontier.sort_unstable();
        frontier.dedup();

        let steps: Vec<Step> = frontier
            .iter()
            .map(|&target| {
                let (from, parent) = grid
                    .neighbours(target)
                    .filter_map(|n| nodes.get(&n).map(|node| (n, *node)))
                    .max_by(|a, b| a.1.quality.total_cmp(&b.1.quality))
                    .expect("frontier nodes touch the walk");
                Step::new(grid, from, parent, target)
            })
            .collect();
        let measured = step_all(frame, grid, &steps)?;

        fresh.clear();
        for (step, node) in steps.iter().zip(measured) {
            match node {
                Some(node) => {
                    nodes.insert(step.target, node);
                    fresh.push(step.target);
                }
                None => *refused.entry(step.target).or_insert(0) += 1,
            }
        }
    }
    Ok(refused)
}

/// A step of the walk, from `parent` at node `from` to `target`, and the
/// window that measures it against the parent's frequency.
struct Step {
    from: GridNode,
    parent: Node,
    target: GridNode,
    expected: [Carrier; 2],
    window: DemodWindow,
}

impl Step {
    fn new(grid: &Grid, from: GridNode, parent: Node, target: GridNode) -> Self {
        let (x, y) = grid.pixel(target);
        let expected = planes(parent.k);
        Self {
            from,
            parent,
            target,
            expected,
            window: window(x, y, &expected, COARSE_WINDOW * period_of(&expected)),
        }
    }
}

/// Takes every step of a wave, in two batches: the targets, then the
/// frequency probes of those whose carriers are clean. None for a step
/// refused.
fn step_all<D: LocalDemodulator>(
    frame: &Frame<D>,
    grid: &Grid,
    steps: &[Step],
) -> Result<Vec<Option<Node>>, MeasureError> {
    let windows: Vec<DemodWindow> = steps.iter().map(|s| s.window).collect();
    let found = frame.demodulate(&windows)?;
    // Each clean target's probes, if they stay in the frame.
    let probes: Vec<Option<[DemodWindow; 4]>> = steps
        .iter()
        .zip(&found)
        .map(|(step, found)| {
            if found.quality() < MIN_QUALITY {
                return None;
            }
            let DemodWindow { x, y, .. } = step.window;
            frequency_probes(&frame.image, x, y, &step.expected)
        })
        .collect();
    let probe_windows: Vec<DemodWindow> = probes.iter().flatten().flatten().copied().collect();
    let mut probed = frame.demodulate(&probe_windows)?.into_iter();
    Ok(steps
        .iter()
        .zip(&found)
        .zip(&probes)
        .map(|((step, &found), probes)| {
            if found.quality() < MIN_QUALITY {
                return None;
            }
            let k = match probes {
                Some(probes) => {
                    let probed: Vec<Demod> = probed.by_ref().take(4).collect();
                    local_frequency(&step.expected, probes, &probed)
                }
                None => step.parent.k,
            };
            step_to(grid, step, found, k)
        })
        .collect())
}

/// Measures `step.target` from its neighbour in the walk, given what its
/// window found against the parent's frequency and its own local frequency
/// `k`: unwraps its phase against the parent's, carried over with the mean
/// of both frequencies. None when the phase misses by more than a quarter
/// turn.
fn step_to(grid: &Grid, step: &Step, found: Demod, k: [[Real; 2]; 2]) -> Option<Node> {
    let parent = &step.parent;
    let (x, y) = grid.pixel(step.target);
    let (px, py) = grid.pixel(step.from);
    let (dx, dy) = (x as Real - px as Real, y as Real - py as Real);
    let mut phase = [0.0; 2];
    for c in 0..2 {
        let predicted = parent.phase[c]
            + 0.5 * ((parent.k[c][0] + k[c][0]) * dx + (parent.k[c][1] + k[c][1]) * dy);
        let miss = wrap(found.phase[c] - predicted);
        if miss.abs() > 0.5 * PI {
            return None;
        }
        phase[c] = predicted + miss;
    }
    Some(Node {
        phase,
        k,
        quality: found.quality(),
    })
}

/// Quadratic phase around a coarse node, from the nodes within two steps:
/// value, gradient and Hessian, per carrier.
#[derive(Clone, Copy, Debug)]
struct Patch {
    /// Pixel the model is centred on.
    centre: [Real; 2],
    /// Phase at the centre.
    value: [Real; 2],
    model: [Carrier; 2],
}

impl Patch {
    /// The phase the patch predicts at a pixel, and the local model there:
    /// the same curvature, with the gradient carried over.
    fn at(&self, x: Real, y: Real) -> ([Real; 2], [Carrier; 2]) {
        let (qx, qy) = (x - self.centre[0], y - self.centre[1]);
        let mut value = [0.0; 2];
        let mut model = self.model;
        for c in 0..2 {
            let carrier = self.model[c];
            let Carrier { k, h } = carrier;
            value[c] = carrier.phase_at(self.value[c], qx, qy);
            model[c].k = [k[0] + h[0] * qx + h[1] * qy, k[1] + h[1] * qx + h[2] * qy];
        }
        (value, model)
    }
}

/// Least-squares quadratic through the phases of the nodes within two steps
/// of `centre`, per carrier. With fewer than nine such nodes, or a singular
/// fit, falls back to the centre node's own phase and plane wave. None when
/// `centre` itself was not measured.
fn fit_patch(grid: &Grid, nodes: &HashMap<GridNode, Node>, centre: GridNode) -> Option<Patch> {
    let node = nodes.get(&centre)?;
    let scale = grid.step as Real;
    // Normal equations in grid steps, with unknowns (value, ∂x, ∂y, ∂xx, ∂xy,
    // ∂yy) shared by both carriers' right-hand sides.
    let mut normal = SMatrix::<Real, 6, 6>::zeros();
    let mut rhs = [SVector::<Real, 6>::zeros(); 2];
    let mut count = 0;
    for dy in -2i64..=2 {
        for dx in -2i64..=2 {
            let (gx, gy) = (centre.0 as i64 + dx, centre.1 as i64 + dy);
            if gx < 0 || gy < 0 {
                continue;
            }
            let Some(other) = nodes.get(&(gx as usize, gy as usize)) else {
                continue;
            };
            let (u, v) = (dx as Real, dy as Real);
            let row = SVector::<Real, 6>::from([1.0, u, v, 0.5 * u * u, u * v, 0.5 * v * v]);
            normal += row * row.transpose();
            for (rhs, &phase) in rhs.iter_mut().zip(&other.phase) {
                *rhs += row * phase;
            }
            count += 1;
        }
    }
    let mut patch = Patch {
        centre: grid.point(centre),
        value: node.phase,
        model: planes(node.k),
    };
    if count < 9 {
        return Some(patch);
    }
    let Some(inverse) = normal.try_inverse() else {
        return Some(patch);
    };
    for (c, rhs) in rhs.iter().enumerate() {
        let s = inverse * rhs;
        // Back from grid steps to pixels.
        patch.value[c] = s[0];
        patch.model[c] = Carrier {
            k: [s[1] / scale, s[2] / scale],
            h: [
                s[3] / (scale * scale),
                s[4] / (scale * scale),
                s[5] / (scale * scale),
            ],
        };
    }
    Some(patch)
}

/// A patch around every walked node that has one.
fn fit_patches(grid: &Grid, nodes: &HashMap<GridNode, Node>) -> HashMap<GridNode, Patch> {
    nodes
        .keys()
        .collect::<Vec<_>>()
        .into_par_iter()
        .filter_map(|&n| fit_patch(grid, nodes, n).map(|p| (n, p)))
        .collect()
}

/// A fine node with the phase it should land near and the local model to
/// demodulate it against.
type Prediction = (GridNode, [Real; 2], [Carrier; 2]);

/// A prediction for every fine node whose nearest coarse node has a patch.
fn predict_fine(coarse: &Grid, patches: &HashMap<GridNode, Patch>, fine: &Grid) -> Vec<Prediction> {
    fine.nodes()
        .filter_map(|n| {
            let (x, y) = fine.pixel(n);
            let patch = patches.get(&coarse.nearest(x as Real, y as Real)?)?;
            let (predicted, model) = patch.at(x as Real, y as Real);
            Some((n, predicted, model))
        })
        .collect()
}

/// A fine node, measured.
#[derive(Clone, Copy, Debug)]
struct Sample {
    node: GridNode,
    /// Unwrapped, per carrier.
    phase: [Real; 2],
    /// The local model it was demodulated against.
    model: [Carrier; 2],
    /// See [`Demod::offset`].
    offset: Real,
    quality: Real,
}

/// Demodulates each node against its predicted phase and local model. A node
/// is kept when both carriers are clean and land within a sixth of a turn of
/// the prediction.
fn measure_fine<D: LocalDemodulator>(
    frame: &Frame<D>,
    fine: &Grid,
    predictions: &[Prediction],
) -> Result<Vec<Sample>, MeasureError> {
    let windows: Vec<DemodWindow> = predictions
        .iter()
        .map(|&(node, _, model)| {
            let (x, y) = fine.pixel(node);
            window(x, y, &model, FINE_WINDOW * period_of(&model))
        })
        .collect();
    let found = frame.demodulate(&windows)?;
    Ok(predictions
        .iter()
        .zip(found)
        .filter_map(|(&(node, predicted, model), found)| {
            if found.quality() < MIN_QUALITY {
                return None;
            }
            let mut phase = [0.0; 2];
            for c in 0..2 {
                let miss = wrap(found.phase[c] - predicted[c]);
                if miss.abs() > PI / 3.0 {
                    return None;
                }
                phase[c] = predicted[c] + miss;
            }
            Some(Sample {
                node,
                phase,
                model,
                offset: found.offset,
                quality: found.quality(),
            })
        })
        .collect())
}

/// Measures one grayscale frame (row-major, `0.0..=1.0`).
///
/// Each candidate carrier pair is followed in turn; the first whose code reads
/// wins, and failing that the one that covered the most board.
pub fn measure_view(
    intensity: &[f32],
    width: usize,
    height: usize,
    target: &Target,
) -> Result<View, MeasureError> {
    measure(intensity, width, height, target, None, false).0
}

/// [`measure_view`] for a frame that follows `previous`, as in a video: the
/// board has barely turned since, so its carriers are tried first and the
/// search of the whole spectrum only runs if they no longer lead to the code.
pub fn measure_view_after(
    intensity: &[f32],
    width: usize,
    height: usize,
    target: &Target,
    previous: &View,
) -> Result<View, MeasureError> {
    measure(intensity, width, height, target, Some(previous), false).0
}

/// [`measure_view`], or [`measure_view_after`] given `previous`, with a
/// record of how it went. With `images`, the trace also keeps the spectrum,
/// the phase maps and the restored frame.
pub fn measure_view_traced(
    intensity: &[f32],
    width: usize,
    height: usize,
    target: &Target,
    previous: Option<&View>,
    images: bool,
) -> (Result<View, MeasureError>, Trace) {
    measure(intensity, width, height, target, previous, images)
}

/// [`measure_view_traced`] with the windows demodulated on `demodulator`,
/// a GPU say. The spectral search, the code and the restoration stay on the
/// CPU.
pub fn measure_view_traced_with<D: LocalDemodulator>(
    demodulator: &D,
    intensity: &[f32],
    width: usize,
    height: usize,
    target: &Target,
    previous: Option<&View>,
    images: bool,
) -> (Result<View, MeasureError>, Trace) {
    measure_on(
        demodulator,
        intensity,
        width,
        height,
        target,
        previous,
        images,
    )
}

/// [`measure_on`] the CPU.
fn measure(
    intensity: &[f32],
    width: usize,
    height: usize,
    target: &Target,
    previous: Option<&View>,
    images: bool,
) -> (Result<View, MeasureError>, Trace) {
    measure_on(
        &CpuDemodulator,
        intensity,
        width,
        height,
        target,
        previous,
        images,
    )
}

/// What the public `measure_view*` functions share: tries the previous
/// frame's carriers, then each pair the spectral search finds, keeping the
/// first view that reads the code, else the one with the most points.
fn measure_on<D: LocalDemodulator>(
    demodulator: &D,
    intensity: &[f32],
    width: usize,
    height: usize,
    target: &Target,
    previous: Option<&View>,
    images: bool,
) -> (Result<View, MeasureError>, Trace) {
    let mut trace = Trace::default();
    let Some(board) = target.printed() else {
        return (Err(MeasureError::UnsupportedOrder(target.order)), trace);
    };
    let frame = match Frame::load(demodulator, intensity, width, height) {
        Ok(frame) => frame,
        Err(e) => return (Err(e), trace),
    };
    // The blur changes little from one frame to the next, so the fit starts
    // from the previous one's.
    let hint = previous.and_then(|p| p.defocus);
    let follow = |carriers, from_previous, trace: &mut Trace| {
        let mut attempt = Attempt {
            from_previous,
            kind: target.kind,
            ..Attempt::default()
        };
        let view = measure_with(&frame, carriers, hint, target, &board, &mut attempt, images);
        attempt.error = view.as_ref().err().copied();
        trace.attempts.push(attempt);
        view
    };
    // A view from the previous carriers that doesn't read the code is still
    // kept, in case the search finds nothing better.
    let mut outcome: Result<View, MeasureError> = Err(MeasureError::NoCarrier);
    if let Some(previous) = previous {
        let view = follow(previous.carriers, true, &mut trace);
        if view.is_ok() {
            trace.chosen = Some(0);
            if view.as_ref().is_ok_and(View::is_absolute) {
                return (view, trace);
            }
            outcome = view;
        }
    }

    let mut clock = Stopwatch::start();
    let search = find_carriers(&frame.image, images);
    trace.searched = true;
    trace.search_ms = clock.lap();
    trace.peaks = search.peaks;
    trace.spectrum = search.spectrum;
    for (index, carriers) in search.pairs.into_iter().enumerate() {
        let view = follow(carriers, false, &mut trace);
        if matches!(&view, Ok(v) if v.is_absolute()) {
            trace.chosen = Some(trace.attempts.len() - 1);
            return (view, trace);
        }
        // Errors report the strongest candidate's, unless a view was found.
        let better = match (&outcome, &view) {
            (Err(_), Err(_)) => index == 0,
            (Ok(_), Err(_)) => false,
            (Ok(best), Ok(v)) => v.points.len() > best.points.len(),
            (Err(_), Ok(_)) => true,
        };
        if better {
            trace.chosen = view.is_ok().then(|| trace.attempts.len() - 1);
            outcome = view;
        }
    }
    (outcome, trace)
}

/// Follows one carrier pair through every stage of the module docs, recording
/// each in `attempt`. `defocus_hint` is where the blur fit starts looking.
fn measure_with<D: LocalDemodulator>(
    frame: &Frame<D>,
    carriers: [[Real; 2]; 2],
    defocus_hint: Option<Real>,
    target: &Target,
    board: &Printed,
    attempt: &mut Attempt,
    images: bool,
) -> Result<View, MeasureError> {
    let image = &frame.image;
    let (width, height) = (image.width, image.height);
    let period = period_of(&planes(carriers));
    attempt.carriers = carriers;
    attempt.period = period;
    let mut clock = Stopwatch::start();

    // The coarse walk, and a quadratic phase around each of its nodes.
    let coarse = Grid::new(
        width,
        height,
        ((COARSE_STEP * period).round() as usize).max(4),
    );
    attempt.coarse_step = coarse.step;
    let nodes = walk(frame, &coarse, carriers, period, attempt);
    attempt.timings.push(("walk", clock.lap()));
    let nodes = nodes?;
    let patches = fit_patches(&coarse, &nodes);

    // A first pass of the fine grid against the coarse patches.
    let fine = Grid::new(width, height, (period.round() as usize).max(2));
    attempt.fine_step = fine.step;
    let predictions = predict_fine(&coarse, &patches, &fine);
    attempt.funnel.push(("predicted", predictions.len()));
    attempt.timings.push(("patches", clock.lap()));
    let mut samples = measure_fine(frame, &fine, &predictions)?;
    attempt.funnel.push(("first pass", samples.len()));
    let predicted: Vec<_> = predictions.iter().map(|p| p.0).collect();
    attempt
        .dropped
        .extend(dropped(&fine, &predicted, &samples, "first pass"));
    attempt.timings.push(("fine", clock.lap()));
    if samples.len() < MIN_POINTS {
        return Err(MeasureError::TooFewPoints(samples.len()));
    }

    let maps = phase_maps(&fine, &samples, width, height);
    let code = decode(image, &fine, &maps, &samples, target);
    attempt.code = Some(code.clone());
    attempt.timings.push(("code", clock.lap()));

    // The code inverts squares here and there, which dents the carrier
    // unevenly across a window and pulls its phase. Once the code is known,
    // paint those squares back and measure the plain checkerboard. A
    // megarena's code leaves dots out, which pulls the phase the same way;
    // they are painted back too.
    let (plain, defocus) = match (code.as_ref(), board) {
        (Ok(Code::Checkerboard(code)), Printed::Checkerboard(board)) => {
            let (restored, defocus) = restore(image, &maps, code, board, period, defocus_hint);
            (Some(restored), defocus)
        }
        (Ok(Code::Megarena(code)), Printed::Megarena(_)) => {
            let restored =
                megarena::restore(image.data, width, height, &maps, code, target.order, period);
            (Some(restored), None)
        }
        _ => (None, None),
    };
    attempt.defocus = defocus.map(|d| d.0);
    attempt.defocus_misfits = defocus.map_or(0, |d| d.1);
    let defocus = attempt.defocus;
    let first_pass = nodes_of(&samples);
    let mut remeasure = |clean: &Frame<D>| {
        let again: Vec<_> = samples.iter().map(|s| (s.node, s.phase, s.model)).collect();
        let samples = measure_fine(clean, &fine, &again)?;
        attempt.funnel.push(("restored", samples.len()));
        attempt.timings.push(("restore", clock.lap()));
        refit(clean, &fine, samples, attempt)
    };
    samples = match plain.as_deref() {
        Some(data) => remeasure(&Frame::load(frame.demodulator, data, width, height)?)?,
        None => remeasure(frame)?,
    };
    attempt
        .dropped
        .extend(dropped(&fine, &first_pass, &samples, "refit"));
    attempt.timings.push(("refit", clock.lap()));
    drop_leaning(&fine, &mut samples, attempt);
    attempt.point_quality = samples.iter().map(|s| (s.quality, s.offset)).collect();
    if images {
        attempt.phase = Some(maps.iter().map(|m| [m[0] as f32, m[1] as f32]).collect());
        attempt.restored = plain;
    }
    if samples.len() < MIN_POINTS {
        return Err(MeasureError::TooFewPoints(samples.len()));
    }

    let points = board_points(&fine, &samples, code.as_ref().ok(), target);
    Ok(View {
        width,
        height,
        points,
        code,
        period,
        defocus,
        carriers,
    })
}

/// The nodes of some samples, in order.
fn nodes_of(samples: &[Sample]) -> Vec<GridNode> {
    samples.iter().map(|s| s.node).collect()
}

/// Nodes in `before` that are not in `after`, as dropped at `stage`, for the
/// trace.
fn dropped(
    fine: &Grid,
    before: &[GridNode],
    after: &[Sample],
    stage: &'static str,
) -> Vec<([Real; 2], &'static str)> {
    let left: HashSet<_> = after.iter().map(|s| s.node).collect();
    before
        .iter()
        .filter(|n| !left.contains(n))
        .map(|&n| (fine.point(n), stage))
        .collect()
}

/// Reads the code. Squares past the board's end, on the paper margin or the
/// room, read as random bits at the ends of each run and sink the decode; so
/// the code is read on the squares at least a step inside the board first,
/// and on everything only if that fails (a board that fills the frame loses
/// its outer squares to the frame's edge, which looks like the board ending).
fn decode(
    image: &Image,
    fine: &Grid,
    maps: &[[Real; 3]],
    samples: &[Sample],
    target: &Target,
) -> Result<Code, CodeError> {
    let kept: HashSet<GridNode> = samples.iter().map(|s| s.node).collect();
    let beyond = beyond_board(fine, &kept);
    let inner: Vec<Sample> = samples
        .iter()
        .filter(|s| inland(fine, &beyond, s.node, DECODE_MARGIN))
        .copied()
        .collect();
    let inner_code = (inner.len() >= MIN_POINTS).then(|| {
        let inner_maps = phase_maps(fine, &inner, image.width, image.height);
        read_code(image, &inner_maps, &inner, target)
    });
    match inner_code {
        Some(Ok(code)) => Ok(code),
        _ => read_code(image, maps, samples, target),
    }
}

/// The fine nodes as walk nodes, to fit patches on.
fn samples_as_nodes(samples: &[Sample]) -> HashMap<GridNode, Node> {
    samples
        .iter()
        .map(|s| {
            (
                s.node,
                Node {
                    phase: s.phase,
                    k: [s.model[0].k, s.model[1].k],
                    quality: 0.0,
                },
            )
        })
        .collect()
}

/// Remeasures the fine grid with the local model refitted on the fine nodes
/// themselves, [`REFIT_PASSES`] times. The curvature the windows were matched
/// to came from the coarse grid, four periods either way. Refitting it on the
/// fine nodes, two periods either way, takes most of what is left of the
/// window's curvature bias out (measured: a third of it remains after two
/// passes).
fn refit<D: LocalDemodulator>(
    frame: &Frame<D>,
    fine: &Grid,
    mut samples: Vec<Sample>,
    attempt: &mut Attempt,
) -> Result<Vec<Sample>, MeasureError> {
    for stage in REFIT_STAGES {
        let as_nodes = samples_as_nodes(&samples);
        let refitted: Vec<_> = samples
            .par_iter()
            .filter_map(|s| Some((s.node, s.phase, fit_patch(fine, &as_nodes, s.node)?.model)))
            .collect();
        samples = measure_fine(frame, fine, &refitted)?;
        attempt.funnel.push((stage, samples.len()));
    }
    Ok(samples)
}

/// Drops the samples whose phase a window past the board may have pulled:
/// those whose carrier sits off the window centre, then those near where the
/// board ends.
fn drop_leaning(fine: &Grid, samples: &mut Vec<Sample>, attempt: &mut Attempt) {
    let before_offset = nodes_of(samples);
    samples.retain(|s| s.offset <= MAX_OFFSET);
    attempt.funnel.push(("offset", samples.len()));
    attempt
        .dropped
        .extend(dropped(fine, &before_offset, samples, "offset"));

    let kept: HashSet<GridNode> = samples.iter().map(|s| s.node).collect();
    let before_edge = nodes_of(samples);
    let beyond = beyond_board(fine, &kept);
    samples.retain(|s| inland(fine, &beyond, s.node, EDGE_MARGIN));
    attempt.funnel.push(("edge", samples.len()));
    attempt
        .dropped
        .extend(dropped(fine, &before_edge, samples, "edge"));
}

/// Each sample's pixel against its point on the board. With a code, the
/// squares are shifted by whole code periods to bring the view's mean near
/// the board's origin.
fn board_points(
    fine: &Grid,
    samples: &[Sample],
    code: Option<&Code>,
    target: &Target,
) -> Vec<PointMatch> {
    let mut squares: Vec<(Real, Real)> = samples
        .iter()
        .map(|s| pattern_square(target.kind, code, s.phase))
        .collect();
    if code.is_some() {
        let count = squares.len() as Real;
        let code_period = target.period_squares() as Real;
        let (sum_i, sum_j) = squares
            .iter()
            .fold((0.0, 0.0), |(a, b), &(i, j)| (a + i, b + j));
        let shift = (
            code_period * (sum_i / count / code_period).round(),
            code_period * (sum_j / count / code_period).round(),
        );
        for s in &mut squares {
            *s = (s.0 - shift.0, s.1 - shift.1);
        }
    }
    samples
        .iter()
        .zip(&squares)
        .map(|(s, &(i, j))| PointMatch {
            pixel: fine.point(s.node),
            board: target.board_point(i, j),
        })
        .collect()
}

/// The nodes off the measured board: missing nodes whose stretch (four-way
/// connected) reaches the frame's edge or is too large for a hole. A hole is
/// a few nodes that failed inside the board, around a coding square that blur
/// has spread, say; the board goes on around it, so the windows next to it
/// are whole.
fn beyond_board(grid: &Grid, kept: &HashSet<GridNode>) -> HashSet<GridNode> {
    let mut seen = HashSet::new();
    let mut beyond = HashSet::new();
    for start in grid.nodes() {
        if kept.contains(&start) || !seen.insert(start) {
            continue;
        }
        // Flood the stretch of missing nodes that `start` belongs to.
        let mut stretch = vec![start];
        let mut open = vec![start];
        let mut at_edge = false;
        while let Some(node) = open.pop() {
            at_edge |= grid.on_edge(node);
            for next in grid.neighbours(node) {
                if !kept.contains(&next) && seen.insert(next) {
                    stretch.push(next);
                    open.push(next);
                }
            }
        }
        if at_edge || stretch.len() >= HOLE_LIMIT {
            beyond.extend(stretch);
        }
    }
    beyond
}

/// No node within `margin` steps is off the board. The frame edge only
/// truncates a window, it brings nothing foreign into it.
fn inland(grid: &Grid, beyond: &HashSet<GridNode>, node: GridNode, margin: i64) -> bool {
    (-margin..=margin).all(|dy| {
        (-margin..=margin).all(|dx| {
            grid.offset_node(node, dx, dy)
                .is_none_or(|near| !beyond.contains(&near))
        })
    })
}

/// Position in the measured square lattice, integers at square centres. The
/// carriers run along the squares' diagonals, half a turn per square.
fn lattice(phase: [Real; 2]) -> (Real, Real) {
    let (s, d) = (phase[0] / PI, phase[1] / PI);
    ((s + d) * 0.5, (s - d) * 0.5)
}

/// Per pixel: both phases, from the nearest fine node's local model, and the
/// square side in pixels there. NaN off the measured region.
fn phase_maps(fine: &Grid, samples: &[Sample], width: usize, height: usize) -> Vec<[Real; 3]> {
    let by_node = with_holes_filled(fine, samples);
    let mut maps = vec![[Real::NAN; 3]; width * height];
    maps.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        for (x, out) in row.iter_mut().enumerate() {
            let Some(node) = fine.nearest(x as Real, y as Real) else {
                continue;
            };
            let Some(sample) = by_node.get(&node) else {
                continue;
            };
            let (nx, ny) = fine.pixel(sample.node);
            let (qx, qy) = (x as Real - nx as Real, y as Real - ny as Real);
            for (c, phase) in out.iter_mut().take(2).enumerate() {
                *phase = sample.model[c].phase_at(sample.phase[c], qx, qy);
            }
            // A carrier spans a square's diagonal: |k| = π√2 / side.
            out[2] = PI * core::f64::consts::SQRT_2 / norm(sample.model[0].k);
        }
    });
    maps
}

/// The sample of each fine node, with every hole in the board (see
/// [`beyond_board`]) given the sample of its nearest measured node within
/// three steps, so the squares there can be read and painted back too.
fn with_holes_filled<'a>(fine: &Grid, samples: &'a [Sample]) -> HashMap<GridNode, &'a Sample> {
    let mut by_node: HashMap<GridNode, &Sample> = samples.iter().map(|s| (s.node, s)).collect();
    let kept: HashSet<GridNode> = by_node.keys().copied().collect();
    let beyond = beyond_board(fine, &kept);
    let holes: Vec<GridNode> = fine
        .nodes()
        .filter(|n| !kept.contains(n) && !beyond.contains(n))
        .collect();
    for hole in holes {
        let reach = 3i64;
        let nearest = (-reach..=reach)
            .flat_map(|dy| (-reach..=reach).map(move |dx| (dx, dy)))
            .filter_map(|(dx, dy)| {
                let (x, y) = (hole.0 as i64 + dx, hole.1 as i64 + dy);
                let sample = *by_node.get(&(usize::try_from(x).ok()?, usize::try_from(y).ok()?))?;
                Some((dx * dx + dy * dy, sample))
            })
            // Only measured nodes, not holes filled earlier in this loop.
            .filter(|(_, s)| kept.contains(&s.node))
            .min_by_key(|(d, _)| *d);
        if let Some((_, sample)) = nearest {
            by_node.insert(hole, sample);
        }
    }
    by_node
}

/// Reads the code off the phase maps: a checkerboard's with `vernier-pose`'s
/// decoder, anchored at the middle sample; a megarena's with
/// [`megarena::read_code`].
fn read_code(
    image: &Image,
    maps: &[[Real; 3]],
    samples: &[Sample],
    target: &Target,
) -> Result<Code, CodeError> {
    if target.is_megarena() {
        return megarena::read_code(maps, image.data, target.order)
            .map(Code::Megarena)
            .map_err(CodeError::Megarena);
    }
    let phase1: Vec<Real> = maps.iter().map(|p| p[0]).collect();
    let phase2: Vec<Real> = maps.iter().map(|p| p[1]).collect();
    let anchor = lattice(samples[samples.len() / 2].phase);
    extract_code_from_phases(
        &phase1,
        &phase2,
        image.data,
        image.width,
        image.height,
        anchor,
        target.order,
        target.layout,
        target.packing,
    )
    .map(Code::Checkerboard)
    .map_err(CodeError::Checkerboard)
}

/// The frame with every coding square painted back to its checkerboard colour.
///
/// The frame is modelled as the board blurred by a Gaussian: the defocus and
/// the local contrast are fitted (see [`fit_defocus`]) and each inverted
/// square's correction is added blurred as the square itself was. Even a
/// sharp frame has pixel-sized blur, and painting the squares back through
/// the model halved the points' error against mirroring each square about
/// the local mean (0.028 to 0.013 px on clean tilted views); out of focus the
/// mirror would also leave a halo where a square spreads past its edge. The
/// mirror stays for when the fit cannot be made. Also gives the defocus
/// found, Gaussian sigma in pixels, and the misfits its search evaluated;
/// the search starts from `hint` when there is one.
fn restore(
    image: &Image,
    maps: &[[Real; 3]],
    code: &CheckerboardCode,
    board: &Checkerboard,
    period: Real,
    hint: Option<Real>,
) -> (Vec<f32>, Option<(Real, usize)>) {
    let fit = CodeModel::new(maps, code, board)
        .and_then(|model| fit_defocus(image, maps, &model, period, hint).map(|fit| (model, fit)));
    match fit {
        Some((model, defocus)) => {
            let found = (defocus.sigma, defocus.misfits);
            (restore_blurred(image, maps, &model, &defocus), Some(found))
        }
        None => (restore_by_mirroring(image, maps, code, board, period), None),
    }
}

/// [`restore`] through the fitted blur model: each pixel gets back twice the
/// blurred inverted squares' share, scaled by the local contrast.
fn restore_blurred(
    image: &Image,
    maps: &[[Real; 3]],
    model: &CodeModel,
    defocus: &Defocus,
) -> Vec<f32> {
    let w = image.width;
    let mut out = image.data.to_vec();
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, v) in row.iter_mut().enumerate() {
            let Some((_, coded)) = model.at(maps[y * w + x], defocus.sigma) else {
                continue;
            };
            *v = (*v as Real + 2.0 * defocus.contrast(x, y) * coded) as f32;
        }
    });
    out
}

/// [`restore`] without a blur model: each inverted square mirrored about the
/// local mean, the square's edge pixels in proportion to how much of them it
/// covers.
fn restore_by_mirroring(
    image: &Image,
    maps: &[[Real; 3]],
    code: &CheckerboardCode,
    board: &Checkerboard,
    period: Real,
) -> Vec<f32> {
    let w = image.width;
    let mean = blur(image, period);
    let mut out = image.data.to_vec();
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, v) in row.iter_mut().enumerate() {
            let [p1, p2, side] = maps[y * w + x];
            if !(p1.is_finite() && p2.is_finite()) {
                continue;
            }
            let (ic, jc) = lattice([p1, p2]);
            let (i, j) = (ic.round(), jc.round());
            let (pi, pj) = code.to_pattern((i, j));
            if !board.square_inverted(pi as i64, pj as i64) {
                continue;
            }
            // Pixels from the pixel to the square's edge, positive inside.
            let inside = (0.5 - (ic - i).abs().max((jc - j).abs())) * side;
            let cover = (inside + 0.5).clamp(0.0, 1.0);
            let value = *v as Real;
            *v = (value + 2.0 * cover * (mean[y * w + x] - value)) as f32;
        }
    });
    out
}

/// The board as the measured lattice sees it: which squares are inverted,
/// signed by their checkerboard colour, over the lattice the phase maps
/// cover.
struct CodeModel {
    /// Lowest lattice square held, and how many along each axis.
    origin: (i64, i64),
    size: (usize, usize),
    /// Per square: 0 for a plain one, else ±1, the sign of its plain colour,
    /// `(−1)^(i+j)`.
    inverted: Vec<i8>,
}

impl CodeModel {
    /// Covers the lattice squares the maps reach (sampled every seventh
    /// pixel), plus a blur's reach either way. None when the maps are empty.
    fn new(maps: &[[Real; 3]], code: &CheckerboardCode, board: &Checkerboard) -> Option<Self> {
        let (mut lo, mut hi) = ((i64::MAX, i64::MAX), (i64::MIN, i64::MIN));
        for m in maps.iter().step_by(7) {
            if !(m[0].is_finite() && m[1].is_finite()) {
                continue;
            }
            let (i, j) = lattice([m[0], m[1]]);
            let (i, j) = (i.round() as i64, j.round() as i64);
            lo = (lo.0.min(i), lo.1.min(j));
            hi = (hi.0.max(i), hi.1.max(j));
        }
        if lo.0 > hi.0 {
            return None;
        }
        let reach = BLUR_REACH + 1;
        let origin = (lo.0 - reach, lo.1 - reach);
        let size = (
            (hi.0 - lo.0 + 2 * reach + 1) as usize,
            (hi.1 - lo.1 + 2 * reach + 1) as usize,
        );
        let inverted = (0..size.0 * size.1)
            .map(|index| {
                let (i, j) = (
                    origin.0 + (index % size.0) as i64,
                    origin.1 + (index / size.0) as i64,
                );
                let (pi, pj) = code.to_pattern((i as Real, j as Real));
                if board.square_inverted(pi as i64, pj as i64) {
                    if (i + j).rem_euclid(2) == 0 { 1 } else { -1 }
                } else {
                    0
                }
            })
            .collect();
        Some(Self {
            origin,
            size,
            inverted,
        })
    }

    /// See [`CodeModel::inverted`]; 0 outside the squares held.
    fn inverted(&self, i: i64, j: i64) -> i8 {
        let (a, b) = (i - self.origin.0, j - self.origin.1);
        if a < 0 || b < 0 || a as usize >= self.size.0 || b as usize >= self.size.1 {
            return 0;
        }
        self.inverted[b as usize * self.size.0 + a as usize]
    }

    /// At a pixel's phases, under a Gaussian defocus of `sigma` pixels: the
    /// plain checkerboard, `±1` on its squares, and the sum of the inverted
    /// squares' plain colours, each blurred. The frame is then about
    /// `mean + h·(plain − 2·coded)`.
    fn at(&self, map: [Real; 3], sigma: Real) -> Option<(Real, Real)> {
        let [p1, p2, side] = map;
        if !(p1.is_finite() && p2.is_finite()) {
            return None;
        }
        let (u, v) = lattice([p1, p2]);
        let (ru, rv) = (u.round() as i64, v.round() as i64);
        // The blur is separable, so each square's blurred box is the product
        // of one along `u` and one along `v`, measured in squares here.
        let sigma_squares = sigma / side;
        let reach = ((3.0 * sigma_squares).ceil() as i64 + 1).min(BLUR_REACH);
        let span = (2 * reach + 1) as usize;
        let along_u = blurred_squares(u - ru as Real, reach, sigma_squares);
        let along_v = blurred_squares(v - rv as Real, reach, sigma_squares);
        // A blurred ±1 square wave along one axis.
        let wave = |r: i64, along: &[Real; SPAN]| {
            let mut sign = if (r - reach).rem_euclid(2) == 0 {
                1.0
            } else {
                -1.0
            };
            let mut sum = 0.0;
            for &f in &along[..span] {
                sum += sign * f;
                sign = -sign;
            }
            sum
        };
        let plain = wave(ru, &along_u) * wave(rv, &along_v);
        let mut coded = 0.0;
        for (b, &fv) in along_v[..span].iter().enumerate() {
            let mut row = 0.0;
            for (a, &fu) in along_u[..span].iter().enumerate() {
                let c = self.inverted(ru + a as i64 - reach, rv + b as i64 - reach);
                if c != 0 {
                    row += c as Real * fu;
                }
            }
            coded += row * fv;
        }
        Some((plain, coded))
    }
}

/// Squares a blurred square's spread is held for: [`BLUR_REACH`] either way.
const SPAN: usize = 2 * BLUR_REACH as usize + 1;

/// One-square-wide boxes centred `−reach..=reach` squares from a pixel that
/// sits `t` squares off the nearest centre, blurred by a Gaussian of `s`
/// squares: how much of each the pixel sees. Neighbouring boxes share an
/// edge, so this takes `2·reach + 2` normal CDFs, not twice as many.
fn blurred_squares(t: Real, reach: i64, s: Real) -> [Real; SPAN] {
    let mut out = [0.0; SPAN];
    if s < 1e-3 {
        out[reach as usize] = 1.0;
        return out;
    }
    // Box `k` is centred `k − reach` squares from the nearest centre, so its
    // near edge is `t + reach − k + ½` squares behind the pixel.
    let edge = |k: i64| normal_cdf((t + (reach - k) as Real + 0.5) / s);
    let mut before = edge(0);
    for (k, f) in out[..(2 * reach + 1) as usize].iter_mut().enumerate() {
        let after = edge(k as i64 + 1);
        *f = before - after;
        before = after;
    }
    out
}

/// The standard normal cumulative distribution.
fn normal_cdf(x: Real) -> Real {
    // Past 6 deviations it is within 1e-9 of its limit, closer than `erf`
    // gets anyway, and most of a sharp board's square edges are that far.
    if x.abs() > 6.0 {
        return if x > 0.0 { 1.0 } else { 0.0 };
    }
    0.5 * (1.0 + erf(x / core::f64::consts::SQRT_2))
}

/// Abramowitz and Stegun 7.1.26, within 1.5e-7.
fn erf(x: Real) -> Real {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let y = 1.0 - poly * (-x * x).exp();
    if x < 0.0 { -y } else { y }
}

/// The defocus of a frame and the board's local contrast under it.
struct Defocus {
    /// Gaussian sigma, pixels.
    sigma: Real,
    /// Misfits the search for it evaluated.
    misfits: usize,
    /// Half the white-to-black step, per tile, for [`Defocus::contrast`].
    tiles: Vec<Real>,
    /// Tile side, in pixels.
    tile: usize,
    columns: usize,
    rows: usize,
}

impl Defocus {
    /// Contrast at a pixel, bilinear between tile centres.
    fn contrast(&self, x: usize, y: usize) -> Real {
        let fx = (x as Real / self.tile as Real - 0.5).clamp(0.0, (self.columns - 1) as Real);
        let fy = (y as Real / self.tile as Real - 0.5).clamp(0.0, (self.rows - 1) as Real);
        let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.columns - 1), (y0 + 1).min(self.rows - 1));
        let (tx, ty) = (fx - x0 as Real, fy - y0 as Real);
        let at = |c: usize, r: usize| self.tiles[r * self.columns + c];
        (at(x0, y0) * (1.0 - tx) + at(x1, y0) * tx) * (1.0 - ty)
            + (at(x0, y1) * (1.0 - tx) + at(x1, y1) * tx) * ty
    }
}

/// Running sums of one tile for the line `i ≈ a + h·m` of [`fit_defocus`]:
/// `[n, Σm, Σi, Σmm, Σmi, Σii]`.
type LineSums = [Real; 6];

/// The least-squares line through a tile's sums: its slope `h` and residual
/// sum of squares. None when the tile has too few pixels or no spread in `m`.
fn fit_line(s: &LineSums) -> Option<(Real, Real)> {
    let n = s[0];
    let mm = s[3] - s[1] * s[1] / n;
    let mi = s[4] - s[1] * s[2] / n;
    let ii = s[5] - s[2] * s[2] / n;
    (n >= 20.0 && mm > 1e-9 * n).then(|| (mi / mm, ii - mi * mi / mm))
}

/// One in this many of the sampled pixels takes part in the search for the
/// sigma; the contrast map at the sigma found uses them all.
const SEARCH_SHARE: u64 = 3;

/// A pixel the fit samples: where it is, its tile, its phase maps and its
/// intensity.
struct Site {
    pixel: usize,
    tile: usize,
    map: [Real; 3],
    value: Real,
}

/// Fits the defocus: the sigma at which `mean + h·(plain − 2·coded)`, with a
/// mean and a contrast `h` free per tile of four periods, best matches the
/// frame on a sample of its pixels. The search starts from `hint`, the
/// previous frame's sigma, when there is one.
fn fit_defocus(
    image: &Image,
    maps: &[[Real; 3]],
    model: &CodeModel,
    period: Real,
    hint: Option<Real>,
) -> Option<Defocus> {
    let w = image.width;
    let tile = ((4.0 * period).round() as usize).max(8);
    let (columns, rows) = (w.div_ceil(tile), image.height.div_ceil(tile));
    let stride = ((0.25 * period).round() as usize).max(1);
    let sites: Vec<Site> = (0..image.height)
        .step_by(stride)
        .flat_map(|y| (0..w).step_by(stride).map(move |x| (x, y)))
        .filter(|&(x, y)| maps[y * w + x][0].is_finite())
        .map(|(x, y)| Site {
            pixel: y * w + x,
            tile: (y / tile) * columns + x / tile,
            map: maps[y * w + x],
            value: image.data[y * w + x] as Real,
        })
        .collect();
    if sites.len() < 200 {
        return None;
    }
    // Picked by a hash of the position rather than every so many, which
    // could beat against the squares.
    let all: Vec<&Site> = sites.iter().collect();
    let search: Vec<&Site> = sites
        .iter()
        .filter(|s| scramble(s.pixel as u64).is_multiple_of(SEARCH_SHARE))
        .collect();
    let tally = |sites: &[&Site], sigma: Real| -> Vec<LineSums> {
        let mut sums = vec![[0.0; 6]; columns * rows];
        let values: Vec<(usize, Real, Real)> = sites
            .par_iter()
            .filter_map(|site| {
                let (plain, coded) = model.at(site.map, sigma)?;
                Some((site.tile, plain - 2.0 * coded, site.value))
            })
            .collect();
        for (tile_index, m, i) in values {
            let s = &mut sums[tile_index];
            s[0] += 1.0;
            s[1] += m;
            s[2] += i;
            s[3] += m * m;
            s[4] += m * i;
            s[5] += i * i;
        }
        sums
    };
    let misfit = |sigma: Real| -> Real {
        tally(&search, sigma)
            .iter()
            .filter_map(fit_line)
            .map(|(_, residual)| residual)
            .sum()
    };

    let (sigma, misfits) = minimize_misfit(period, hint, misfit);
    let fits: Vec<Option<Real>> = tally(&all, sigma)
        .iter()
        .map(|s| fit_line(s).map(|l| l.0))
        .collect();
    let found: Vec<Real> = fits.iter().flatten().copied().collect();
    if found.is_empty() {
        return None;
    }
    // A tile without a fit takes the mean contrast of those with one.
    let overall = found.iter().sum::<Real>() / found.len() as Real;
    Some(Defocus {
        sigma,
        misfits,
        tiles: fits.iter().map(|f| f.unwrap_or(overall)).collect(),
        tile,
        columns,
        rows,
    })
}

/// SplitMix64's finalizer: spreads a value's bits over the whole word.
fn scramble(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Even steps of the search without a hint, over 0 to half a period.
const SIGMA_STEPS: usize = 7;

/// The sigma of least `misfit` over 0 to half a period, where the carrier is
/// all but gone, and how many misfits it took. The misfit is smooth in sigma,
/// so once a minimum is bracketed Brent's method closes in on it in a handful
/// of steps. The bracket is `hint` and a step either side of it, a sixth or
/// so of the sigma, when the hint's misfit is the least of the three; else
/// the least of [`SIGMA_STEPS`] even steps over the whole range and its
/// neighbours.
fn minimize_misfit(
    period: Real,
    hint: Option<Real>,
    misfit: impl Fn(Real) -> Real,
) -> (Real, usize) {
    let top = 0.5 * period;
    let calls = std::cell::Cell::new(0);
    let f = |sigma: Real| {
        calls.set(calls.get() + 1);
        misfit(sigma)
    };
    let least = |values: &[Real]| {
        (0..values.len())
            .min_by(|&a, &b| values[a].total_cmp(&values[b]))
            .unwrap_or(0)
    };
    let around_hint = hint.filter(|h| (0.0..top).contains(h)).and_then(|h| {
        let step = 0.15 * h + 0.02 * period;
        let points = [(h - step).max(0.0), h, (h + step).min(top)];
        let values = points.map(&f);
        let best = least(&values);
        // Bracketed when the hint is the least, or when the least is at 0,
        // which the sigma cannot go below.
        (best == 1 || points[best] == 0.0).then_some((points, values, best))
    });
    let (points, values, best) = around_hint.unwrap_or_else(|| {
        let grid: Vec<Real> = (0..=SIGMA_STEPS)
            .map(|k| top * k as Real / SIGMA_STEPS as Real)
            .collect();
        let values: Vec<Real> = grid.iter().map(|&s| f(s)).collect();
        let best = least(&values);
        let (lo, hi) = (best.saturating_sub(1), (best + 1).min(SIGMA_STEPS));
        let bracket = [lo, best, hi];
        (bracket.map(|i| grid[i]), bracket.map(|i| values[i]), 1)
    });
    let sigma = brent(points, values, best, 5e-4 * period, f);
    (sigma, calls.get())
}

/// Brent's minimization of `f` over `points[0]..points[2]`, to within about
/// `tolerance`, from those three points already evaluated, `points[best]` the
/// least.
fn brent(
    points: [Real; 3],
    values: [Real; 3],
    best: usize,
    tolerance: Real,
    f: impl Fn(Real) -> Real,
) -> Real {
    /// The golden section's smaller part, `(3 − √5)/2`.
    const GOLDEN: Real = 0.381_966_011_250_105_1;
    let (mut a, mut b) = (points[0], points[2]);
    let (mut x, mut fx) = (points[best], values[best]);
    // The other two, the next best first, so that the first step can already
    // be a parabola through all three.
    let mut rest: Vec<usize> = (0..3).filter(|&i| i != best).collect();
    rest.sort_by(|&i, &j| values[i].total_cmp(&values[j]));
    let (mut w, mut fw) = (points[rest[0]], values[rest[0]]);
    let (mut v, mut fv) = (points[rest[1]], values[rest[1]]);
    // The last step, and the one before it.
    let (mut d, mut e): (Real, Real) = (0.0, b - a);
    for _ in 0..40 {
        let middle = 0.5 * (a + b);
        if (x - middle).abs() <= 2.0 * tolerance - 0.5 * (b - a) {
            break;
        }
        let mut parabolic = false;
        if e.abs() > tolerance && w != x && v != x && v != w {
            let r = (x - w) * (fx - fv);
            let q = (x - v) * (fx - fw);
            let (mut p, mut q) = ((x - v) * q - (x - w) * r, 2.0 * (q - r));
            if q > 0.0 {
                p = -p;
            } else {
                q = -q;
            }
            if p.abs() < (0.5 * q * e).abs() && p > q * (a - x) && p < q * (b - x) {
                e = d;
                d = p / q;
                let u = x + d;
                if u - a < 2.0 * tolerance || b - u < 2.0 * tolerance {
                    d = if x < middle { tolerance } else { -tolerance };
                }
                parabolic = true;
            }
        }
        if !parabolic {
            e = if x < middle { b - x } else { a - x };
            d = GOLDEN * e;
        }
        let u = if d.abs() >= tolerance {
            x + d
        } else {
            x + tolerance.copysign(d)
        };
        let fu = f(u);
        if fu <= fx {
            if u < x {
                b = x;
            } else {
                a = x;
            }
            (v, fv, w, fw, x, fx) = (w, fw, x, fx, u, fu);
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                (v, fv, w, fw) = (w, fw, u, fu);
            } else if fu <= fv || v == x || v == w {
                (v, fv) = (u, fu);
            }
        }
    }
    x
}

/// Separable Gaussian blur of `sigma` pixels, normalized at the frame edge.
fn blur(image: &Image, sigma: Real) -> Vec<Real> {
    blur_with(image.width, image.height, sigma, |i| image.data[i] as Real)
}

/// [`blur`] of whatever `at` gives per pixel, row-major.
fn blur_with(w: usize, h: usize, sigma: Real, at: impl Fn(usize) -> Real + Sync) -> Vec<Real> {
    let radius = (3.0 * sigma).ceil() as isize;
    let kernel: Vec<Real> = (-radius..=radius)
        .map(|q| (-0.5 * (q as Real / sigma).powi(2)).exp())
        .collect();
    let mut rows = vec![0.0; w * h];
    rows.par_chunks_mut(w).enumerate().for_each(|(y, out)| {
        let src: Vec<Real> = (y * w..(y + 1) * w).map(&at).collect();
        for (x, v) in out.iter_mut().enumerate() {
            let (mut sum, mut weight) = (0.0, 0.0);
            for (t, &k) in kernel.iter().enumerate() {
                let j = x as isize + t as isize - radius;
                if j < 0 || j >= w as isize {
                    continue;
                }
                sum += k * src[j as usize];
                weight += k;
            }
            *v = sum / weight;
        }
    });
    // Down the columns a whole row at a time, which keeps memory access in
    // order.
    let mut out = vec![0.0; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, out)| {
        let mut weight = 0.0;
        for (t, &k) in kernel.iter().enumerate() {
            let j = y as isize + t as isize - radius;
            if j < 0 || j >= h as isize {
                continue;
            }
            weight += k;
            let src = &rows[j as usize * w..(j as usize + 1) * w];
            for (v, &s) in out.iter_mut().zip(src) {
                *v += k * s;
            }
        }
        out.iter_mut().for_each(|v| *v /= weight);
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bad_frames_are_errors_not_panics() {
        let target = Target::new(10.0, 8);
        let bad = |data: &[f32], width, height| {
            matches!(
                measure_view(data, width, height, &target),
                Err(MeasureError::BadFrame { .. })
            )
        };
        assert!(bad(&[0.5; 15], 4, 4));
        assert!(bad(&[], 0, 0));
        assert!(bad(&[0.5; 8], 8, 1));
    }

    /// A misfit shaped like the fit's: smooth, least at `at`, flattening off
    /// towards large sigma.
    fn misfit(at: Real) -> impl Fn(Real) -> Real {
        move |s: Real| 1.0 - (-((s - at) / (0.6 + 0.4 * at)).powi(2)).exp()
    }

    #[test]
    fn search_finds_the_least_misfit() {
        let period = 11.3;
        for at in [0.0, 0.3, 1.04, 2.5, 4.9] {
            let (sigma, calls) = minimize_misfit(period, None, misfit(at));
            assert!((sigma - at).abs() < 0.02, "{at}: found {sigma}");
            assert!(calls <= 16, "{at}: {calls} misfits");
        }
    }

    #[test]
    fn hint_saves_misfits_and_survives_being_wrong() {
        let period = 11.3;
        for at in [0.3, 1.04, 2.5, 4.9] {
            let (_, cold) = minimize_misfit(period, None, misfit(at));
            let (sigma, warm) = minimize_misfit(period, Some(at * 1.05), misfit(at));
            assert!((sigma - at).abs() < 0.02, "{at}: found {sigma}");
            assert!(warm < cold, "{at}: {warm} misfits warm, {cold} cold");
            // A hint far off falls back on the whole range.
            let (sigma, _) = minimize_misfit(period, Some(at + 2.0), misfit(at));
            assert!(
                (sigma - at).abs() < 0.02,
                "{at}: found {sigma} from a wrong hint"
            );
        }
    }

    #[test]
    fn blurred_squares_tile_the_line() {
        // The boxes cover the whole line, so what a pixel sees of them sums
        // to one, and each matches its box blurred on its own.
        for (t, s) in [(0.0, 0.05), (0.3, 0.4), (-0.45, 0.9)] {
            let spread = blurred_squares(t, BLUR_REACH, s);
            assert!((spread.iter().sum::<Real>() - 1.0).abs() < 1e-3, "{t} {s}");
            for (k, &f) in spread.iter().enumerate() {
                let d = t - (k as i64 - BLUR_REACH) as Real;
                let alone = normal_cdf((d + 0.5) / s) - normal_cdf((d - 0.5) / s);
                assert!((f - alone).abs() < 1e-12);
            }
        }
    }
}
