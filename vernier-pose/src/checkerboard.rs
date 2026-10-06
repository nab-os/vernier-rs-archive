//! Absolute decode for the coded checkerboard.
//!
//! The carriers give the position inside a square; the code says which square.
//!
//! 1. Sample: map each pixel to a square through the two carrier phases.
//! 2. Binarize each square against its neighbours.
//! 3. Mark squares whose colour breaks the checkerboard parity: the coding sites.
//! 4. Try every rotation and supercell offset, read the bits, find them in the LFSR.
//! 5. Keep the best hypothesis if the evidence is strong and unambiguous.

use std::collections::BTreeMap;

use vernier_core::buffer::BufferLayout;
use vernier_core::scalar::consts::{PI, TAU};
use vernier_core::{Complex32, ComputeBackend, Pose, Real};
use vernier_patterns::checkerboard::{Checkerboard, CodeLayout, CodePacking};
use vernier_patterns::lfsr::{Lfsr, WindowIndex};
use vernier_spectral::spectrum::{Detection, analyze_two};

use crate::Calibration;
use crate::absolute::{CoarseDecoder, CoarseOrders};

/// Only pixels this close to a square centre are sampled (edges are blurry).
const SAMPLE_RADIUS: Real = 0.35;

const MIN_SAMPLES: u64 = 4;

/// Bits needed beyond `order` to check a placement.
const MIN_SPARE_BITS: usize = 3;

/// Bit errors always tolerated; long runs may tolerate more.
const MAX_BIT_ERRORS: usize = 1;

/// Max chance that a wrong placement of one code passes by luck.
const FALSE_ACCEPT_LIMIT: Real = 1e-4;

/// The four quarter-turns. Mirrors are ruled out earlier by `handedness`.
const TRANSFORMS: [[i64; 4]; 4] = [
    [1, 0, 0, 1],   // identity
    [0, -1, 1, 0],  // +90°
    [-1, 0, 0, -1], // 180°
    [0, 1, -1, 0],  // −90°
];

#[derive(Clone, Debug, PartialEq)]
pub struct CheckerboardCode {
    /// Index into `TRANSFORMS`.
    pub transform: usize,
    /// `pattern = transform(measured) + delta`.
    pub delta: (i64, i64),
    /// Square under the image centre.
    pub centre_square: (i64, i64),
    /// Same, with the sub-square position from the phases.
    pub centre: (Real, Real),
    pub k_x: i64,
    pub k_y: i64,
    pub x_window: Vec<u8>,
    pub y_window: Vec<u8>,
    pub check_bits: usize,
    pub bit_errors: (usize, usize),
    /// Chance a wrong hypothesis would match this well. Small is good.
    pub false_accept: Real,
    /// `false_accept` of the best hypothesis on a different square.
    pub runner_up_false_accept: Real,
}

impl CheckerboardCode {
    /// A point of the measured square lattice, in the pattern's lattice: the
    /// decoded quarter-turn, then the decoded shift.
    pub fn to_pattern(&self, (i, j): (Real, Real)) -> (Real, Real) {
        // Row-major 2×2 rotation matrix.
        let turn = TRANSFORMS[self.transform].map(|v| v as Real);
        (
            turn[0] * i + turn[1] * j + self.delta.0 as Real,
            turn[2] * i + turn[3] * j + self.delta.1 as Real,
        )
    }
}

pub struct CheckerboardDecoder {
    code: CheckerboardCode,
}

impl CheckerboardDecoder {
    pub fn new(code: CheckerboardCode) -> Self {
        Self { code }
    }
}

impl CoarseDecoder for CheckerboardDecoder {
    fn decode(&self) -> Option<CoarseOrders> {
        Some(CoarseOrders {
            k1: self.code.k_x,
            k2: self.code.k_y,
            k3: (self.code.transform % 4) as u8,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckerboardError {
    UnsupportedOrder(u32),
    NotEnoughSquares,
    NoConsistentHypothesis,
    /// Detection locked onto the code's sub-carrier line. Use `detect_checkerboard`.
    SubharmonicLock,
    /// The code matched no better than chance.
    WeakEvidence,
    /// Another square matched almost as well (e.g. heavy motion blur).
    AmbiguousPosition,
}

impl std::fmt::Display for CheckerboardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedOrder(n) => write!(f, "unsupported LFSR order {n}"),
            Self::NotEnoughSquares => {
                write!(
                    f,
                    "too few squares sampled to read {MIN_SPARE_BITS} spare bits"
                )
            }
            Self::NoConsistentHypothesis => {
                write!(
                    f,
                    "no orientation hypothesis reproduced the coding geometry"
                )
            }
            Self::AmbiguousPosition => {
                write!(
                    f,
                    "two different positions matched the code about equally well"
                )
            }
            Self::WeakEvidence => {
                write!(
                    f,
                    "the code was not read with enough evidence to trust a position"
                )
            }
            Self::SubharmonicLock => {
                write!(
                    f,
                    "carrier detection locked onto the code's sub-carrier line; \
                     detect with detect_checkerboard"
                )
            }
        }
    }
}

impl std::error::Error for CheckerboardError {}

pub fn extract_code(
    detection: &Detection,
    intensity: &[f32],
    order: u32,
) -> Result<CheckerboardCode, CheckerboardError> {
    extract_code_with_layout(detection, intensity, order, CodeLayout::Squares)
}

pub fn extract_code_with_layout(
    detection: &Detection,
    intensity: &[f32],
    order: u32,
    layout: CodeLayout,
) -> Result<CheckerboardCode, CheckerboardError> {
    extract_code_with_packing(detection, intensity, order, layout, CodePacking::OneBit)
}

pub fn extract_code_with_packing(
    detection: &Detection,
    intensity: &[f32],
    order: u32,
    layout: CodeLayout,
    packing: CodePacking,
) -> Result<CheckerboardCode, CheckerboardError> {
    if Lfsr::maximal(order).is_none() {
        return Err(CheckerboardError::UnsupportedOrder(order));
    }
    if subharmonic_ratios_with_packing(detection, intensity, packing)
        .iter()
        .any(|&r| r > SUBHARMONIC_LIMIT)
    {
        return Err(CheckerboardError::SubharmonicLock);
    }

    let sign2 = handedness(detection);
    // Centre phase: the plane fit for precision, the map for the 2π multiple
    // (the squares were indexed from the map). Then the measured lattice
    // point there: square `(i, j)` sits at `φ₁/π = i + j`, `φ₂/π = i - j`.
    let centre_pixel = (detection.height / 2) * detection.width + detection.width / 2;
    let snap = |fitted: Real, measured: Real| fitted + TAU * ((measured - fitted) / TAU).round();
    let phase1 = snap(detection.dir1.plane.c, detection.phase1[centre_pixel]);
    let phase2 = sign2 * snap(detection.dir2.plane.c, detection.phase2[centre_pixel]);
    let centre_measured = (
        (phase1 / PI + phase2 / PI) * 0.5,
        (phase1 / PI - phase2 / PI) * 0.5,
    );

    decode_phases(
        &detection.phase1,
        &detection.phase2,
        sign2,
        intensity,
        detection.width,
        detection.height,
        centre_measured,
        order,
        layout,
        packing,
    )
}

/// The decode on bare unwrapped phase maps, for callers that measure the
/// carriers some other way than one global detection (a perspective view,
/// where the carrier frequency drifts across the frame).
///
/// The maps must have the pattern's own handedness: with `k1`, `k2` the phase
/// gradients in pixel coordinates, `k1 × k2 < 0`, as [`handedness`] arranges
/// for a detection. Pixels with a non-finite phase are skipped, so a map can mark
/// where the pattern is not. `reference` is any point of the measured square
/// lattice, `((φ₁ + φ₂)/2π, (φ₁ - φ₂)/2π)`; the returned `centre` is that
/// point in the pattern's lattice.
#[allow(clippy::too_many_arguments)]
pub fn extract_code_from_phases(
    phase1: &[Real],
    phase2: &[Real],
    intensity: &[f32],
    width: usize,
    height: usize,
    reference: (Real, Real),
    order: u32,
    layout: CodeLayout,
    packing: CodePacking,
) -> Result<CheckerboardCode, CheckerboardError> {
    decode_phases(
        phase1, phase2, 1.0, intensity, width, height, reference, order, layout, packing,
    )
}

/// The decode shared by both entry points. `sign2` (±1) turns `phase2` to
/// the pattern's handedness; `centre_measured` is the measured-lattice point
/// reported, mapped into the pattern, as the code's `centre`.
#[allow(clippy::too_many_arguments)]
fn decode_phases(
    phase1: &[Real],
    phase2: &[Real],
    sign2: Real,
    intensity: &[f32],
    width: usize,
    height: usize,
    centre_measured: (Real, Real),
    order: u32,
    layout: CodeLayout,
    packing: CodePacking,
) -> Result<CheckerboardCode, CheckerboardError> {
    let lfsr = Lfsr::maximal(order).ok_or(CheckerboardError::UnsupportedOrder(order))?;
    let index = lfsr.window_index();

    let samples = accumulate_squares(phase1, phase2, sign2, intensity, width, height);
    // Squares per bit in two dimensions: one supercell holds `bits_per_cell`
    // of them per axis.
    let squares_per_bit = (packing.cell() * packing.cell() / packing.bits_per_cell()) as usize;
    if samples.len() < squares_per_bit * (order as usize + MIN_SPARE_BITS) {
        return Err(CheckerboardError::NotEnoughSquares);
    }
    let white = binarize(&samples);

    let mut candidates: Vec<CheckerboardCode> = TRANSFORMS
        .iter()
        .enumerate()
        .flat_map(|(transform_index, matrix)| {
            try_transform(
                &white,
                centre_measured,
                transform_index,
                matrix,
                layout,
                packing,
                &lfsr,
                &index,
                order,
            )
        })
        .collect();
    candidates.sort_by(|a, b| {
        if stronger(a, b) {
            std::cmp::Ordering::Less
        } else if stronger(b, a) {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    });

    let mut ranked = candidates.into_iter();
    let mut best = ranked
        .next()
        .ok_or(CheckerboardError::NoConsistentHypothesis)?;
    let centre = best.centre_square;
    best.runner_up_false_accept = ranked
        .find(|c| c.centre_square != centre)
        .map_or(Real::INFINITY, |c| c.false_accept);

    if best.false_accept > MAX_FALSE_ACCEPT {
        return Err(CheckerboardError::WeakEvidence);
    }
    if best.runner_up_false_accept <= MAX_FALSE_ACCEPT {
        return Err(CheckerboardError::AmbiguousPosition);
    }
    Ok(best)
}

// ─── Diagnostic API ──────────────────────────────────────────────────────────

/// One square as the decoder saw it, for anything that wants to show the
/// extraction rather than just its answer.
#[derive(Clone, Copy, Debug)]
pub struct SquareReadout {
    /// Square index in the measured frame — the frame the phases define, before
    /// any rotation hypothesis puts it back in the pattern's own.
    pub i: i64,
    pub j: i64,
    /// Mean image intensity over the centre of the square.
    pub mean: Real,
    /// Colour after the local threshold. `None` for squares [`binarize`] drops
    /// for want of samples or contrast.
    pub is_white: Option<bool>,
    /// Whether the square breaks the checkerboard parity, which is what makes
    /// it a coding site. `None` wherever `is_white` is.
    pub is_coding_site: Option<bool>,
}

/// The square-level stages of the decode, kept instead of discarded: the
/// dewarped thumbnail, the binarization, and the coding sites read out of it.
///
/// This is a view of steps 1–3 in the module header, run exactly as
/// [`extract_code_with_packing`] runs them. `code` is that function's own answer
/// on the same detection, so a caller can show the sites and what they decoded
/// to side by side.
#[derive(Clone, Debug)]
pub struct CodingReadout {
    /// Every square with at least one sample, in `(i, j)` order.
    pub squares: Vec<SquareReadout>,
    /// Inclusive bounds of the sampled lattice, as `(min, max)`.
    pub i_range: (i64, i64),
    pub j_range: (i64, i64),
    /// Square under the image centre, with the sub-square part kept.
    pub centre: (Real, Real),
    /// What the full decode made of it.
    pub code: Result<CheckerboardCode, CheckerboardError>,
}

/// Samples the squares, binarizes them and marks the coding sites, without
/// committing to an orientation.
///
/// Returns `None` when the phases yield no square at all — an empty image, or a
/// detection so far off that nothing lands inside a sampling radius.
pub fn read_squares(
    detection: &Detection,
    intensity: &[f32],
    order: u32,
    layout: CodeLayout,
) -> Option<CodingReadout> {
    read_squares_with_packing(detection, intensity, order, layout, CodePacking::OneBit)
}

/// [`read_squares`] for a denser packing. Only `code` depends on it: a site is a
/// square that breaks the parity whichever supercell grid wrote it.
pub fn read_squares_with_packing(
    detection: &Detection,
    intensity: &[f32],
    order: u32,
    layout: CodeLayout,
    packing: CodePacking,
) -> Option<CodingReadout> {
    let sign2 = handedness(detection);
    let samples = accumulate_squares(
        &detection.phase1,
        &detection.phase2,
        sign2,
        intensity,
        detection.width,
        detection.height,
    );
    if samples.is_empty() {
        return None;
    }
    let white = binarize(&samples);

    // The identity transform: the measured frame is the one the thumbnail is
    // drawn in, so the defects are marked where the squares actually are. Only
    // the parity choice matters here, and `build_defect_map` makes that by
    // majority — independent of which quarter-turn the code later turns out to
    // need.
    let map = build_defect_map(&white, &TRANSFORMS[0]);
    let defects: BTreeMap<(i64, i64), bool> = map
        .squares
        .iter()
        .map(|&((i, j), defect)| ((i - map.parity_shift, j), defect))
        .collect();

    let squares: Vec<SquareReadout> = samples
        .iter()
        .map(|(&(i, j), &(sum, count))| SquareReadout {
            i,
            j,
            mean: sum / count as Real,
            is_white: white.get(&(i, j)).copied(),
            is_coding_site: defects.get(&(i, j)).copied(),
        })
        .collect();

    let (i_min, i_max) = squares.iter().fold((i64::MAX, i64::MIN), |(lo, hi), s| {
        (lo.min(s.i), hi.max(s.i))
    });
    let (j_min, j_max) = squares.iter().fold((i64::MAX, i64::MIN), |(lo, hi), s| {
        (lo.min(s.j), hi.max(s.j))
    });

    let centre_pixel = (detection.height / 2) * detection.width + detection.width / 2;
    let snap = |fitted: Real, measured: Real| fitted + TAU * ((measured - fitted) / TAU).round();
    let phase1 = snap(detection.dir1.plane.c, detection.phase1[centre_pixel]);
    let phase2 = sign2 * snap(detection.dir2.plane.c, detection.phase2[centre_pixel]);

    Some(CodingReadout {
        squares,
        i_range: (i_min, i_max),
        j_range: (j_min, j_max),
        centre: (
            (phase1 / PI + phase2 / PI) * 0.5,
            (phase1 / PI - phase2 / PI) * 0.5,
        ),
        code: extract_code_with_packing(detection, intensity, order, layout, packing),
    })
}

fn stronger(a: &CheckerboardCode, b: &CheckerboardCode) -> bool {
    match a.false_accept.total_cmp(&b.false_accept) {
        std::cmp::Ordering::Less => true,
        std::cmp::Ordering::Greater => false,
        std::cmp::Ordering::Equal => a.check_bits > b.check_bits,
    }
}

/// Mean intensity per square, using only pixels near each centre.
fn accumulate_squares(
    phase1: &[Real],
    phase2: &[Real],
    sign2: Real,
    intensity: &[f32],
    width: usize,
    height: usize,
) -> BTreeMap<(i64, i64), (Real, u64)> {
    let mut pools: BTreeMap<(i64, i64), (Real, u64)> = BTreeMap::new();
    for pixel in 0..width * height {
        let s = phase1[pixel] / PI;
        let d = sign2 * phase2[pixel] / PI;
        if !(s.is_finite() && d.is_finite()) {
            continue;
        }
        let i = ((s + d) * 0.5).round();
        let j = ((s - d) * 0.5).round();
        if (s - (i + j)).abs() >= SAMPLE_RADIUS || (d - (i - j)).abs() >= SAMPLE_RADIUS {
            continue;
        }
        let entry = pools.entry((i as i64, j as i64)).or_insert((0.0, 0));
        entry.0 += intensity[pixel] as Real;
        entry.1 += 1;
    }
    pools
}

const THRESHOLD_RADIUS: i64 = 3;
const MIN_PER_PARITY: usize = 6;
/// Squares with less local contrast than this (relative) are dropped.
const MIN_RELATIVE_CONTRAST: Real = 0.15;

fn median(values: &mut [Real]) -> Real {
    let mid = values.len() / 2;
    let (_, m, _) = values.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    *m
}

/// Black/white per square, with a local threshold so uneven lighting is fine.
///
/// Around each square, the median of each parity class gives the local black
/// and white levels (the few code-flipped squares don't move a median). The
/// threshold is their midpoint.
fn binarize(samples: &BTreeMap<(i64, i64), (Real, u64)>) -> BTreeMap<(i64, i64), bool> {
    let mut counts: Vec<Real> = samples.values().map(|&(_, n)| n as Real).collect();
    if counts.is_empty() {
        return BTreeMap::new();
    }
    // Small tiles only get a pixel or two per square, so scale the minimum.
    let typical_count = median(&mut counts);
    let min_count = ((typical_count / 2.0).floor() as u64).clamp(1, MIN_SAMPLES);

    let means: BTreeMap<(i64, i64), Real> = samples
        .iter()
        .filter(|&(_, &(_, count))| count >= min_count)
        .map(|(&key, &(sum, count))| (key, sum / count as Real))
        .collect();
    if means.is_empty() {
        return BTreeMap::new();
    }

    let (i_min, i_max) = means
        .keys()
        .fold((i64::MAX, i64::MIN), |(lo, hi), &(i, _)| {
            (lo.min(i), hi.max(i))
        });
    let (j_min, j_max) = means
        .keys()
        .fold((i64::MAX, i64::MIN), |(lo, hi), &(_, j)| {
            (lo.min(j), hi.max(j))
        });
    let width = (i_max - i_min + 1) as usize;
    let height = (j_max - j_min + 1) as usize;
    let mut grid = vec![Real::NAN; width * height];
    for (&(i, j), &m) in &means {
        grid[(j - j_min) as usize * width + (i - i_min) as usize] = m;
    }

    let mut judged: Vec<((i64, i64), Real, Real, Real)> = Vec::with_capacity(means.len());
    let (mut even, mut odd) = (Vec::new(), Vec::new());
    for (&(i, j), &m) in &means {
        let mut radius = THRESHOLD_RADIUS;
        loop {
            even.clear();
            odd.clear();
            for nj in (j - radius).max(j_min)..=(j + radius).min(j_max) {
                for ni in (i - radius).max(i_min)..=(i + radius).min(i_max) {
                    let v = grid[(nj - j_min) as usize * width + (ni - i_min) as usize];
                    if v.is_nan() {
                        continue;
                    }
                    if (ni + nj).rem_euclid(2) == 0 {
                        even.push(v);
                    } else {
                        odd.push(v);
                    }
                }
            }
            let enough = even.len() >= MIN_PER_PARITY && odd.len() >= MIN_PER_PARITY;
            if enough || radius >= 4 * THRESHOLD_RADIUS {
                break;
            }
            radius *= 2;
        }
        if even.is_empty() || odd.is_empty() {
            continue;
        }
        let (level_even, level_odd) = (median(&mut even), median(&mut odd));
        judged.push((
            (i, j),
            0.5 * (level_even + level_odd),
            (level_even - level_odd).abs(),
            m,
        ));
    }
    if judged.is_empty() {
        return BTreeMap::new();
    }

    let mut contrasts: Vec<Real> = judged.iter().map(|&(_, _, c, _)| c).collect();
    let typical_contrast = median(&mut contrasts);
    judged
        .into_iter()
        .filter(|&(_, _, contrast, _)| contrast >= MIN_RELATIVE_CONTRAST * typical_contrast)
        .map(|(key, threshold, _, mean)| (key, mean >= threshold))
        .collect()
}

/// -1 if the peak search returned the carriers swapped (a mirrored frame).
fn handedness(detection: &Detection) -> Real {
    let (p1, p2) = (&detection.dir1.plane, &detection.dir2.plane);
    if p1.a * p2.b - p1.b * p2.a > 0.0 {
        -1.0
    } else {
        1.0
    }
}

/// True locks measure ~0.1, false locks 1.3 and up.
const SUBHARMONIC_LIMIT: Real = 0.5;

fn signed_bin(bin: usize, n: usize) -> i64 {
    if bin > n / 2 {
        bin as i64 - n as i64
    } else {
        bin as i64
    }
}

/// Hann-windowed amplitude of an image at given FFT bins, without a full FFT.
struct Demodulator {
    width: usize,
    height: usize,
    xs: Vec<Real>,
    ys: Vec<Real>,
    samples: Vec<Real>,
}

impl Demodulator {
    fn new(intensity: &[f32], width: usize, height: usize, step: usize) -> Self {
        let mean = intensity.iter().map(|&v| v as Real).sum::<Real>() / intensity.len() as Real;
        let xs: Vec<usize> = (0..width).step_by(step).collect();
        let ys: Vec<usize> = (0..height).step_by(step).collect();
        let hann = |i: usize, n: usize| 0.5 - 0.5 * (TAU * i as Real / (n - 1) as Real).cos();
        let wx: Vec<Real> = xs.iter().map(|&x| hann(x, width)).collect();
        let mut samples = Vec::with_capacity(xs.len() * ys.len());
        for &y in &ys {
            let wy = hann(y, height);
            for (k, &x) in xs.iter().enumerate() {
                samples.push((intensity[y * width + x] as Real - mean) * wx[k] * wy);
            }
        }
        Self {
            width,
            height,
            xs: xs.into_iter().map(|x| x as Real).collect(),
            ys: ys.into_iter().map(|y| y as Real).collect(),
            samples,
        }
    }

    // The phasor factors into row and column terms, so no trig per pixel.
    fn amplitude(&self, bx: i64, by: i64) -> Real {
        let fx = TAU * bx as Real / self.width as Real;
        let fy = TAU * by as Real / self.height as Real;
        let (cx, sx): (Vec<Real>, Vec<Real>) = self
            .xs
            .iter()
            .map(|&x| ((fx * x).cos(), (fx * x).sin()))
            .unzip();
        let cols = self.xs.len();
        let (mut re, mut im) = (0.0, 0.0);
        for (r, &y) in self.ys.iter().enumerate() {
            let row = &self.samples[r * cols..(r + 1) * cols];
            let (mut rc, mut rs) = (0.0, 0.0);
            for k in 0..cols {
                rc += row[k] * cx[k];
                rs += row[k] * sx[k];
            }
            let (cy, sy) = ((fy * y).cos(), (fy * y).sin());
            re += cy * rc - sy * rs;
            im -= cy * rs + sy * rc;
        }
        (re * re + im * im).sqrt()
    }
}

/// For each carrier: amplitude at 3× its frequency over amplitude at it.
///
/// The code puts a line at 1/3 of each carrier. If detection picked that line,
/// 3× lands on the real carrier and the ratio is large. On a real lock, 3× is
/// the third harmonic, about 1/9.
pub fn subharmonic_ratios(detection: &Detection, intensity: &[f32]) -> [Real; 2] {
    subharmonic_ratios_with_packing(detection, intensity, CodePacking::OneBit)
}

/// The code repeats every `cell` squares, so if the peak search locked onto its
/// line instead of the carrier, the carrier sits at `cell` times the found bin
/// and dwarfs it. Measured ~0.1 for a true lock, 1.3 and up for a false one.
pub fn subharmonic_ratios_with_packing(
    detection: &Detection,
    intensity: &[f32],
    packing: CodePacking,
) -> [Real; 2] {
    let (w, h) = (detection.width, detection.height);
    let cell = packing.cell();
    [detection.dir1.peak_bin, detection.dir2.peak_bin].map(|(px, py)| {
        let (bx, by) = (signed_bin(px, w), signed_bin(py, h));
        let (tx, ty) = (cell * bx, cell * by);
        if tx.abs() + 2 >= (w / 2) as i64 || ty.abs() + 2 >= (h / 2) as i64 {
            return 0.0;
        }
        let step = if tx.abs() + 2 < (w / 4) as i64 && ty.abs() + 2 < (h / 4) as i64 {
            2
        } else {
            1
        };
        let demod = Demodulator::new(intensity, w, h, step);
        let base = demod.amplitude(bx, by);
        let mut harmonic: Real = 0.0;
        for dy in -2..=2 {
            for dx in -2..=2 {
                harmonic = harmonic.max(demod.amplitude(tx + dx, ty + dy));
            }
        }
        harmonic / base.max(1e-12)
    })
}

/// `analyze_two`, retried above the peak if it locked onto the code's line.
pub fn detect_checkerboard<B: ComputeBackend>(
    backend: &B,
    intensity: &[f32],
    layout: BufferLayout,
    sigma: Real,
    min_frequency: usize,
    max_frequency: usize,
    smoothing_sigma: Real,
) -> vernier_core::Result<Detection> {
    detect_checkerboard_with_packing(
        backend,
        intensity,
        layout,
        sigma,
        min_frequency,
        max_frequency,
        smoothing_sigma,
        CodePacking::OneBit,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn detect_checkerboard_with_packing<B: ComputeBackend>(
    backend: &B,
    intensity: &[f32],
    layout: BufferLayout,
    sigma: Real,
    min_frequency: usize,
    max_frequency: usize,
    smoothing_sigma: Real,
    packing: CodePacking,
) -> vernier_core::Result<Detection> {
    let complex: Vec<Complex32> = intensity.iter().map(|&v| Complex32::new(v, 0.0)).collect();
    let first = analyze_two(
        backend,
        &complex,
        layout,
        sigma,
        min_frequency,
        max_frequency,
        smoothing_sigma,
    )?;
    let ratios = subharmonic_ratios_with_packing(&first, intensity, packing);
    if ratios.iter().all(|&r| r <= SUBHARMONIC_LIMIT) {
        return Ok(first);
    }

    let (w, h) = (first.width, first.height);
    let false_radius = [first.dir1.peak_bin, first.dir2.peak_bin]
        .iter()
        .zip(ratios)
        .filter(|&(_, r)| r > SUBHARMONIC_LIMIT)
        .map(|(&(px, py), _)| {
            let (bx, by) = (signed_bin(px, w), signed_bin(py, h));
            ((bx * bx + by * by) as Real).sqrt()
        })
        .fold(0.0, Real::max);
    // Search above the false line (r) but below the real carrier, which sits at
    // `cell` times it.
    let raised = min_frequency.max((2.0 * false_radius).ceil() as usize);
    let Ok(second) = analyze_two(
        backend,
        &complex,
        layout,
        sigma,
        raised,
        max_frequency,
        smoothing_sigma,
    ) else {
        return Ok(first);
    };
    let worst = |d: &Detection| {
        subharmonic_ratios_with_packing(d, intensity, packing)
            .into_iter()
            .fold(0.0, Real::max)
    };
    if worst(&second) < worst(&first) {
        Ok(second)
    } else {
        Ok(first)
    }
}

/// Coordinates the code is counted in: `(i, j)`, or `(i+j, i-j)` for diagonals.
fn code_coords(layout: CodeLayout, i: i64, j: i64) -> (i64, i64) {
    match layout {
        CodeLayout::Squares => (i, j),
        CodeLayout::Diamonds => (i + j, i - j),
    }
}

/// Every square in one rotated frame, and whether it breaks the parity.
struct DefectMap {
    squares: Vec<((i64, i64), bool)>,
    parity_shift: i64,
}

#[allow(clippy::too_many_arguments)]
fn try_transform(
    white: &BTreeMap<(i64, i64), bool>,
    centre_measured: (Real, Real),
    transform_index: usize,
    matrix: &[i64; 4],
    layout: CodeLayout,
    packing: CodePacking,
    lfsr: &Lfsr,
    index: &WindowIndex,
    order: u32,
) -> Vec<CheckerboardCode> {
    let map = build_defect_map(white, matrix);

    // Roughly one square in nine is a defect at one bit per supercell, one in
    // thirteen at two — both well inside this band.
    let rate = map.squares.iter().filter(|&&(_, defect)| defect).count() as Real
        / map.squares.len() as Real;
    if !(0.02..0.30).contains(&rate) {
        return Vec::new();
    }

    let cell = packing.cell();
    let mut candidates = Vec::new();
    for delta_i_residue in 0..cell {
        for delta_j_residue in 0..cell {
            candidates.extend(try_origin(
                &map,
                centre_measured,
                transform_index,
                matrix,
                (delta_i_residue, delta_j_residue),
                layout,
                packing,
                lfsr,
                index,
                order,
            ));
        }
    }
    candidates
}

/// Rotates the squares and marks parity defects. The parity choice that gives
/// fewer defects is the right one.
fn build_defect_map(white: &BTreeMap<(i64, i64), bool>, matrix: &[i64; 4]) -> DefectMap {
    let apply = |i: i64, j: i64| (matrix[0] * i + matrix[1] * j, matrix[2] * i + matrix[3] * j);
    let transformed: Vec<((i64, i64), bool)> = white
        .iter()
        .map(|(&(i, j), &is_white)| (apply(i, j), is_white))
        .collect();

    let disagreeing = transformed
        .iter()
        .filter(|&&((i, j), is_white)| is_white != Checkerboard::parity_is_white(i, j))
        .count();
    let parity_shift = i64::from(disagreeing * 2 > transformed.len());

    let squares = transformed
        .iter()
        .map(|&((i, j), is_white)| {
            let square = (i + parity_shift, j);
            (
                square,
                is_white != Checkerboard::parity_is_white(square.0, square.1),
            )
        })
        .collect();

    DefectMap {
        squares,
        parity_shift,
    }
}

/// Reads both codes assuming the supercells start at `delta_residue`.
#[allow(clippy::too_many_arguments)]
fn try_origin(
    map: &DefectMap,
    centre_measured: (Real, Real),
    transform_index: usize,
    matrix: &[i64; 4],
    delta_residue: (i64, i64),
    layout: CodeLayout,
    packing: CodePacking,
    lfsr: &Lfsr,
    index: &WindowIndex,
    order: u32,
) -> Option<CheckerboardCode> {
    let (cell, per_cell) = (packing.cell(), packing.bits_per_cell());
    let (x_sites, y_sites) = (packing.x_sites(), packing.y_sites());

    let (x_bits, x_first) = read_site_bits(map, layout, packing, delta_residue, x_sites, true)?;
    let (y_bits, y_first) = read_site_bits(map, layout, packing, delta_residue, y_sites, false)?;
    let (k_x, x_checks, x_errors) = localize(&x_bits, lfsr, index, order)?;
    let (k_y, y_checks, y_errors) = localize(&y_bits, lfsr, index, order)?;

    // `localize` places the run only modulo the sequence length, but a bit's
    // slot inside its supercell is congruent to its index modulo
    // `per_cell`, and we read that slot directly. The length is odd, so the
    // two facts together pin the index modulo `per_cell · len` — without this
    // lift, a two-bit packing would be a half-supercell out half the time.
    let length = lfsr.len() as i64;
    // The length is odd and `per_cell` is 1 or 2, so stepping by a length walks
    // every slot residue: `per_cell` steps always suffice.
    let lift = |k: i64, first_slot: i64| {
        let want = first_slot.rem_euclid(per_cell);
        let mut bit = k.rem_euclid(length);
        for _ in 0..per_cell {
            if bit.rem_euclid(per_cell) == want {
                break;
            }
            bit += length;
        }
        bit
    };
    // Coordinate, along the coded axis, of a bit slot in the pattern's frame.
    let coord = |slot_index: i64, sites: &[(i64, i64)], by_i: bool| {
        let site = sites[slot_index.rem_euclid(per_cell) as usize];
        cell * slot_index.div_euclid(per_cell) + if by_i { site.0 } else { site.1 }
    };

    // Offset from our frame to the pattern's, in code coordinates. `read_site_bits`
    // indexed the run in the shifted frame, so undo that shift here.
    let delta_a =
        coord(lift(k_x, x_first), x_sites, true) - coord(x_first, x_sites, true) + delta_residue.0;
    let delta_b = coord(lift(k_y, y_first), y_sites, false) - coord(y_first, y_sites, false)
        + delta_residue.1;

    // Diagonal layout: back to (i, j). The deltas are only known modulo an odd
    // period, so if their sum is odd, shift one by a period to make it even.
    let (delta_i, delta_j) = match layout {
        CodeLayout::Squares => (delta_a, delta_b),
        CodeLayout::Diamonds => {
            let period = cell * lfsr.len() as i64;
            let delta_b = if (delta_a + delta_b).rem_euclid(2) != 0 {
                delta_b + period
            } else {
                delta_b
            };
            ((delta_a + delta_b) / 2, (delta_a - delta_b) / 2)
        }
    };

    let (i, j) = centre_measured;
    let raw_i =
        matrix[0] as Real * i + matrix[1] as Real * j + (map.parity_shift + delta_i) as Real;
    let raw_j = matrix[2] as Real * i + matrix[3] as Real * j + delta_j as Real;

    // Wrap to one code period, in (i, j) for both layouts.
    let period = (cell * lfsr.len() as i64) as Real;
    let wrap = |value: Real| value - period * (value / period).floor();
    let (centre_i, centre_j) = (wrap(raw_i), wrap(raw_j));

    Some(CheckerboardCode {
        transform: transform_index,
        delta: (map.parity_shift + delta_i, delta_j),
        centre_square: (centre_i.round() as i64, centre_j.round() as i64),
        centre: (centre_i, centre_j),
        k_x,
        k_y,
        x_window: x_bits[..order as usize].to_vec(),
        y_window: y_bits[..order as usize].to_vec(),
        check_bits: x_checks + y_checks,
        bit_errors: (x_errors, y_errors),
        runner_up_false_accept: Real::INFINITY,
        false_accept: chance_by_luck(x_checks, x_errors)
            * chance_by_luck(y_checks, y_errors)
            * hypotheses(packing),
    })
}

/// One bit per coding site by majority vote, keyed by the site's index in the
/// sequence rather than by its supercell, so a packing carrying several bits
/// per supercell still yields one run of consecutive bits. Returns the longest
/// run and the index of its first bit.
fn read_site_bits(
    map: &DefectMap,
    layout: CodeLayout,
    packing: CodePacking,
    delta_residue: (i64, i64),
    sites: &[(i64, i64)],
    by_i: bool,
) -> Option<(Vec<u8>, i64)> {
    let (cell, per_cell) = (packing.cell(), packing.bits_per_cell());
    let mut votes: BTreeMap<i64, (usize, usize)> = BTreeMap::new();
    for &((i, j), defect) in &map.squares {
        let (first, second) = code_coords(layout, i, j);
        // Shift into the frame this hypothesis proposes. Indexing in the
        // pattern's own frame is what keeps a supercell's bits consecutive:
        // subtracting the residue instead reorders the slots whenever the shift
        // carries one of them across a supercell boundary, and the run shatters.
        let (shifted_first, shifted_second) = (first + delta_residue.0, second + delta_residue.1);
        let within = (
            shifted_first.rem_euclid(cell),
            shifted_second.rem_euclid(cell),
        );
        // The two axes never share a site, so a residue match names one slot.
        let Some(slot) = sites.iter().position(|&s| s == within) else {
            continue;
        };
        let (axis, site) = if by_i {
            (shifted_first, sites[slot].0)
        } else {
            (shifted_second, sites[slot].1)
        };
        let bit = per_cell * (axis - site).div_euclid(cell) + slot as i64;
        let entry = votes.entry(bit).or_insert((0, 0));
        entry.0 += usize::from(defect);
        entry.1 += 1;
    }

    // A defect is a 0 bit.
    let bits: Vec<(i64, u8)> = votes
        .into_iter()
        .map(|(slot, (defective, total))| (slot, u8::from(2 * defective <= total)))
        .collect();

    let mut best: (usize, usize) = (0, 0); // (start index, length)
    let mut run_start = 0usize;
    for position in 1..=bits.len() {
        let broken = position == bits.len() || bits[position].0 != bits[position - 1].0 + 1;
        if broken {
            if position - run_start > best.1 {
                best = (run_start, position - run_start);
            }
            run_start = position;
        }
    }
    if best.1 == 0 {
        return None;
    }
    Some((
        bits[best.0..best.0 + best.1]
            .iter()
            .map(|&(_, b)| b)
            .collect(),
        bits[best.0].0,
    ))
}

/// Bit errors tolerated for a run of `bits`: grows with length, but a wrong
/// placement must still pass by luck less than `FALSE_ACCEPT_LIMIT`.
fn allowed_bit_errors(bits: usize, order: usize) -> usize {
    let spare = bits.saturating_sub(order);
    let mut allowed = MAX_BIT_ERRORS;
    for errors in 0..=spare {
        if chance_by_luck(spare, errors) > FALSE_ACCEPT_LIMIT {
            break;
        }
        allowed = allowed.max(errors);
    }
    allowed
}

/// Chance that random check bits match with at most `errors` misses, over all
/// anchors tried.
fn chance_by_luck(spare: usize, errors: usize) -> Real {
    let (mut cumulative, mut choose) = (0.0, 1.0);
    for k in 0..=errors.min(spare) {
        if k > 0 {
            choose *= (spare - k + 1) as Real / k as Real;
        }
        cumulative += choose;
    }
    (cumulative * (0.5 as Real).powi(spare as i32) * (spare + 1) as Real).min(1.0)
}

/// Correct decodes measured ≤ 0.006, garbage ones ≥ 20.
const MAX_FALSE_ACCEPT: Real = 0.1;

/// Orientations times supercell origins — every placement the search tries,
/// which is what a match has to beat to count as evidence.
fn hypotheses(packing: CodePacking) -> Real {
    (TRANSFORMS.len() as i64 * packing.cell() * packing.cell()) as Real
}

/// Finds the bit run in the LFSR, trying every anchor and scoring the whole run.
/// Returns (position of first bit, spare bits, errors).
fn localize(
    bits: &[u8],
    lfsr: &Lfsr,
    index: &WindowIndex,
    order: u32,
) -> Option<(i64, usize, usize)> {
    let order = order as usize;
    if bits.len() < order + MIN_SPARE_BITS {
        return None;
    }
    let length = lfsr.len();
    let mut best: Option<(i64, usize)> = None; // (position of bit 0, agreements)

    for anchor in 0..=bits.len() - order {
        let Some(found) = index.locate(&bits[anchor..anchor + order]) else {
            continue;
        };
        let first = (found + length - anchor % length) % length;
        let agreements = bits
            .iter()
            .enumerate()
            .filter(|&(offset, &bit)| lfsr.bit_at(first + offset) == bit)
            .count();
        if best.is_none_or(|(_, previous)| agreements > previous) {
            best = Some((first as i64, agreements));
        }
    }

    let (first, agreements) = best?;
    let errors = bits.len() - agreements;
    if errors > allowed_bit_errors(bits.len(), order) {
        return None;
    }
    Some((first, bits.len() - order, errors))
}

/// Absolute pose. `square_size` is the side of one square.
pub fn solve_checkerboard(
    detection: &Detection,
    intensity: &[f32],
    square_size: Real,
    order: u32,
) -> Result<(Pose, CheckerboardCode), CheckerboardError> {
    solve_checkerboard_with_layout(
        detection,
        intensity,
        square_size,
        order,
        CodeLayout::Squares,
    )
}

pub fn solve_checkerboard_with_layout(
    detection: &Detection,
    intensity: &[f32],
    square_size: Real,
    order: u32,
    layout: CodeLayout,
) -> Result<(Pose, CheckerboardCode), CheckerboardError> {
    solve_checkerboard_with_packing(
        detection,
        intensity,
        square_size,
        order,
        layout,
        CodePacking::OneBit,
    )
}

pub fn solve_checkerboard_with_packing(
    detection: &Detection,
    intensity: &[f32],
    square_size: Real,
    order: u32,
    layout: CodeLayout,
    packing: CodePacking,
) -> Result<(Pose, CheckerboardCode), CheckerboardError> {
    let code = extract_code_with_packing(detection, intensity, order, layout, packing)?;

    // Centre in the lattice frame, then in the pattern frame (turned for diamonds).
    let (x, y) = layout.from_lattice(
        square_size * (code.centre.0 + 0.5),
        square_size * (code.centre.1 + 0.5),
    );

    // The carrier runs at 45° to the squares, then undo the decoded quarter-turn.
    let quadrant = (code.transform % 4) as Real;
    let raw = detection.dir1.plane.orientation()
        - PI / 4.0
        - quadrant * (PI / 2.0)
        - layout.lattice_angle();
    let theta = raw - TAU * ((raw + PI) / TAU).floor();

    let gradient = (detection.dir1.plane.a.powi(2) + detection.dir1.plane.b.powi(2)).sqrt();
    let pixel_size = square_size * core::f64::consts::SQRT_2 * gradient / TAU;

    Ok((Pose::new_2d(x, y, theta, pixel_size), code))
}

pub fn solve(
    detection: &Detection,
    intensity: &[f32],
    calib: &Calibration,
    order: u32,
) -> Result<Pose, CheckerboardError> {
    // `period` is the carrier period; the square side is that over √2.
    let square_size = calib.period / core::f64::consts::SQRT_2;
    solve_checkerboard(detection, intensity, square_size, order).map(|(pose, _)| pose)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_amplitude(
        intensity: &[f32],
        width: usize,
        height: usize,
        bx: i64,
        by: i64,
        step: usize,
    ) -> Real {
        let mean = intensity.iter().map(|&v| v as Real).sum::<Real>() / intensity.len() as Real;
        let (fx, fy) = (
            TAU * bx as Real / width as Real,
            TAU * by as Real / height as Real,
        );
        let (mut re, mut im) = (0.0, 0.0);
        for y in (0..height).step_by(step) {
            let wy = 0.5 - 0.5 * (TAU * y as Real / (height - 1) as Real).cos();
            for x in (0..width).step_by(step) {
                let wx = 0.5 - 0.5 * (TAU * x as Real / (width - 1) as Real).cos();
                let v = (intensity[y * width + x] as Real - mean) * wx * wy;
                let phase = fx * x as Real + fy * y as Real;
                re += v * phase.cos();
                im -= v * phase.sin();
            }
        }
        (re * re + im * im).sqrt()
    }

    #[test]
    fn demodulator_matches_direct_sum() {
        let (w, h) = (96, 64);
        let image: Vec<f32> = (0..w * h)
            .map(|p| {
                let (x, y) = ((p % w) as f32, (p / w) as f32);
                0.5 + 0.3 * (0.7 * x + 0.2 * y).sin() + 0.1 * (0.05 * x * y).cos()
            })
            .collect();
        for step in [1, 2] {
            let demod = Demodulator::new(&image, w, h, step);
            for (bx, by) in [(0, 0), (5, -3), (-12, 7), (20, 11), (-3, -9)] {
                let fast = demod.amplitude(bx, by);
                let slow = naive_amplitude(&image, w, h, bx, by, step);
                assert!(
                    (fast - slow).abs() <= 1e-9 * slow.max(1.0),
                    "bin ({bx},{by}) step {step}: {fast} vs {slow}"
                );
            }
        }
    }
}
