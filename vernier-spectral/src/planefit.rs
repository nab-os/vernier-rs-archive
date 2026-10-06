//! Least-squares phase-plane fitting — where the resolution comes from.
//!
//! After a spectral lobe is isolated and inverse-transformed, the argument of
//! the resulting field is the wrapped phase of one pattern direction. Rather
//! than read one value, the method fits a plane `φ(i, j) = a·i + b·j + c` across
//! the whole (unwrapped) phase map by least squares (André et al. 2021). That
//! averages the redundant phase over every pixel, which is what reaches
//! ~1/1000-pixel resolution. It yields `c`, the sub-period phase at the image
//! center, and `(a, b)`, the phase gradients that give the peak location and
//! orientation `atan2(b, a)`.
//!
//! The fit must run on *unwrapped* phase — 2π discontinuities corrupt a fit over
//! wrapped values — so this module unwraps first; see [`fit_plane`].

use vernier_core::Real;

use crate::unwrap::unwrap_1d;

/// Coefficients of a fitted phase plane `φ(i, j) = a·i + b·j + c`, with `(i, j)`
/// measured from the image center.
#[derive(Clone, Copy, Debug)]
pub struct PhasePlane {
    /// Phase gradient along the column (x) axis, radians per pixel.
    pub a: Real,
    /// Phase gradient along the row (y) axis, radians per pixel.
    pub b: Real,
    /// Phase at the image center, radians. This is the high-resolution phase
    /// `φ` used for sub-period position (modulo 2π).
    pub c: Real,
}

impl PhasePlane {
    /// Orientation from the plane gradients, `atan2(b, a)` in `(-π, π]`. This is
    /// the pattern angle before quadrant disambiguation (resolved later from the
    /// missing corner). André et al. 2021, Eq. 3.
    pub fn orientation(&self) -> Real {
        self.b.atan2(self.a)
    }

    /// Spectral peak location `(m, n)` implied by the gradients, given image
    /// dimensions. `m = w·a/2π`, `n = h·b/2π` (André et al. 2022, Eq. 7).
    pub fn peak_location(&self, width: usize, height: usize) -> (Real, Real) {
        use vernier_core::scalar::consts::TAU;
        (width as Real * self.a / TAU, height as Real * self.b / TAU)
    }
}

/// Fits `φ(i, j) = a·i + b·j + c` to a wrapped phase map by least squares:
/// unwrap into a continuous surface, then solve the normal equations in
/// image-centered coordinates so `c` is the center phase.
///
/// `wrapped` is row-major, length `width * height`. `crop_factor ∈ [0, 1)`
/// mirrors C++ `RegressionPlane::cropFactor` (default 0.5): only the center
/// `(1 − crop_factor)` fraction of each axis is used, avoiding edge artefacts
/// from the bandpass IFFT. Pass `0.0` to use the full image.
pub fn fit_plane(wrapped: &[Real], width: usize, height: usize, crop_factor: Real) -> PhasePlane {
    let phase = unwrap_2d(wrapped, width, height);
    fit_plane_to_unwrapped(&phase, width, height, crop_factor)
}

/// Separable 2D phase unwrap: unwrap each row, then reconcile rows via the first
/// column. Returns a continuous surface — what the plane fit needs, and what the
/// megarena decode needs to compute cell indices `round(φ/2π)`. Fine for the
/// near-planar phase of a periodic pattern; steep or noisy phase would want a
/// quality-guided unwrap.
///
/// C++ (`Spatial::quartersUnwrapPhase`) instead propagates outward from the
/// center through four quadrants. On well-conditioned maps the two match to
/// machine epsilon in the gradients (differing only in the offset `c`, a
/// convention); quarter-propagation is only more robust when the seed row/column
/// fall on noise. Separable is kept for simplicity.
pub fn unwrap_2d(wrapped: &[Real], width: usize, height: usize) -> Vec<Real> {
    let mut phase = wrapped.to_vec();

    // Unwrap each row in place.
    for r in 0..height {
        let start = r * width;
        unwrap_1d(&mut phase[start..start + width]);
    }
    // Unwrap down the first column, then propagate each row's offset so rows are
    // mutually consistent.
    let mut first_col: Vec<Real> = (0..height).map(|r| phase[r * width]).collect();
    let before: Vec<Real> = first_col.clone();
    unwrap_1d(&mut first_col);
    for r in 0..height {
        let row_shift = first_col[r] - before[r];
        if row_shift != 0.0 {
            let start = r * width;
            for v in &mut phase[start..start + width] {
                *v += row_shift;
            }
        }
    }
    phase
}

/// Fits the plane to an already-unwrapped phase surface. The production
/// pipeline entry point (C++ `RegressionPlane::compute`); accumulation is
/// always `f64` regardless of `Real`.
///
/// `crop_factor` trims a border of `(crop_factor/2) * dimension` pixels on each
/// side before fitting; coordinates remain centered on the FULL image so `c` is
/// still the phase at the full-image center. Mirrors C++ `RegressionPlane`.
pub fn fit_plane_to_unwrapped(
    phase: &[Real],
    width: usize,
    height: usize,
    crop_factor: Real,
) -> PhasePlane {
    // --- Least-squares plane fit, centered coordinates ---
    let col_off = ((width as f64 * crop_factor) / 2.0) as usize;
    let row_off = ((height as f64 * crop_factor) / 2.0) as usize;

    let cropped_w = width - 2 * col_off;
    let cropped_h = height - 2 * row_off;

    // C++ integer division
    let center_x = (cropped_w / 2) as f64;
    let center_y = (cropped_h / 2) as f64;

    // Accumulate normal-equation sums for [a, b, c] in f64: sii reaches ~1e9
    // for a 512² crop, far beyond f32's 24-bit mantissa.
    let (mut sii, mut sjj, mut sij) = (0.0f64, 0.0f64, 0.0f64);
    let (mut si, mut sj, mut sn) = (0.0f64, 0.0f64, 0.0f64);
    let (mut spi, mut spj, mut sp) = (0.0f64, 0.0f64, 0.0f64);

    for r in row_off..(height - row_off) {
        let j = (r - row_off) as f64 - center_y;
        for col in col_off..(width - col_off) {
            let i = (col - col_off) as f64 - center_x;
            let p = phase[r * width + col];
            sii += i * i;
            sjj += j * j;
            sij += i * j;
            si += i;
            sj += j;
            sn += 1.0;
            spi += p * i;
            spj += p * j;
            sp += p;
        }
    }

    // Solve the 3x3 symmetric system:
    // [sii sij si][a]   [spi]
    // [sij sjj sj][b] = [spj]
    // [si  sj  sn][c]   [sp ]
    let (a, b, c) = solve_3x3(
        [[sii, sij, si], [sij, sjj, sj], [si, sj, sn]],
        [spi, spj, sp],
    );

    PhasePlane {
        a: a as Real,
        b: b as Real,
        c: c as Real,
    }
}

/// Solves a 3x3 linear system by Cramer's rule. The matrix is tiny and
/// well-conditioned in centered image coordinates.
fn solve_3x3(m: [[f64; 3]; 3], v: [f64; 3]) -> (f64, f64, f64) {
    let det = det3(m);
    let mx = det3([
        [v[0], m[0][1], m[0][2]],
        [v[1], m[1][1], m[1][2]],
        [v[2], m[2][1], m[2][2]],
    ]);
    let my = det3([
        [m[0][0], v[0], m[0][2]],
        [m[1][0], v[1], m[1][2]],
        [m[2][0], v[2], m[2][2]],
    ]);
    let mz = det3([
        [m[0][0], m[0][1], v[0]],
        [m[1][0], m[1][1], v[1]],
        [m[2][0], m[2][1], v[2]],
    ]);
    (mx / det, my / det, mz / det)
}

fn det3(m: [[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use vernier_core::scalar::consts::{PI, TAU};

    /// Build a wrapped phase map for a known plane and check we recover it.
    #[test]
    fn recovers_a_known_plane() {
        let (width, height) = (32, 32);
        let (true_a, true_b, true_c) = (0.30, -0.15, 0.4);
        let center_x = width as Real / 2.0;
        let center_y = height as Real / 2.0;

        let mut wrapped = vec![0.0; width * height];
        for r in 0..height {
            for col in 0..width {
                let i = col as Real - center_x;
                let j = r as Real - center_y;
                let mut p = true_a * i + true_b * j + true_c;
                // Wrap into (-π, π].
                p = ((p + PI).rem_euclid(TAU)) - PI;
                wrapped[r * width + col] = p;
            }
        }

        let plane = fit_plane(&wrapped, width, height, 0.0);
        assert!((plane.a - true_a).abs() < 1e-3, "a={}", plane.a);
        assert!((plane.b - true_b).abs() < 1e-3, "b={}", plane.b);
        // c recovered modulo 2π.
        let dc = ((plane.c - true_c + PI).rem_euclid(TAU)) - PI;
        assert!(dc.abs() < 1e-3, "c off by {dc}");
    }
}
