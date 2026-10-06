//! Phase-only pose estimation from fitted phase planes — the fine, sub-period
//! regime (André et al. 2020/2021, 2022 IJCV).
//!
//! Given the two fitted planes (one per pattern direction), the in-plane
//! relations are:
//!
//! - `x = (φ1 / 2π)·λ + k1·λ`   (sub-period part + integer period order)
//! - `y = (φ2 / 2π)·λ + k2·λ`
//! - `α = atan2(b1, a1) + k3·(π/2)`
//!
//! where `φi = ci` is the center phase of direction `i`, `λ` the physical
//! period, `(ai, bi)` the plane gradients, and `k1, k2, k3` the integer
//! orders/quadrant. This module computes the fine part; the `k·λ` and `k3·π/2`
//! terms come from [`absolute`](crate::absolute).

use vernier_core::scalar::consts::{PI, TAU};
use vernier_core::{Pose, Real};
use vernier_spectral::PhasePlane;

use crate::Calibration;

/// Estimates the fine (sub-period) pose from the two direction phase planes,
/// porting C++ `PeriodicPatternDetector::get2DPose`. Translation is correct
/// modulo `calib.period` and orientation is the raw in-image angle; the integer
/// orders and quadrant come from the absolute path, which uses the same sign
/// convention ([`crate::absolute::solve_megarena`]).
pub fn estimate(plane1: &PhasePlane, plane2: &PhasePlane, calib: &Calibration) -> Pose {
    // Sub-period displacements from the center phases (mod one period). The
    // pattern sits at minus the phase's offset, as in C++.
    let x = -(plane1.c / TAU) * calib.period;
    let y = -(plane2.c / TAU) * calib.period;

    // Orientation from the first plane's gradients (high-resolution angle).
    let theta = plane1.orientation();
    // Physical units per pixel: the period over its length in pixels.
    let pixelic_period = TAU / plane1.a.hypot(plane1.b);

    Pose::new_2d(x, y, theta, calib.period / pixelic_period)
}

/// The four ambiguous 3D poses of a periodic pattern under orthographic
/// projection, porting C++ `PeriodicPatternDetector::getAll3DPoses` (André et
/// al. 2022 IJCV). The two gradients fix the out-of-plane tilt up to a four-fold
/// sign ambiguity in `(beta, gamma)`; all four are returned. `pixel_size` is
/// `1/sqrt(s²)`. Picking the right one needs [`compute_phase_gradients`].
pub fn all_3d_poses(plane1: &PhasePlane, plane2: &PhasePlane, calib: &Calibration) -> [Pose; 4] {
    let period = calib.period;
    let n1 = plane1.a * plane1.a + plane1.b * plane1.b;
    let n2 = plane2.a * plane2.a + plane2.b * plane2.b;

    let u1 = plane1.a * TAU / n1;
    let v1 = plane1.b * TAU / n1;
    let u2 = plane2.a * TAU / n2;
    let v2 = plane2.b * TAU / n2;

    let alpha0 = plane1.orientation();
    let (sa, ca) = alpha0.sin_cos();

    let p2 = period * period;
    let b = (u1 * u1 + v1 * v1) / p2;
    let g = (-u2 * sa + v2 * ca).powi(2) / p2;
    let d = (u2 * ca + v2 * sa).powi(2) / p2;

    // Larger root of s² (the C++ `s_2`). Guard the discriminant/acos arguments
    // against tiny negative values from noise so degenerate inputs yield real
    // (rather than NaN) angles.
    let disc = ((b + g + d).powi(2) - 4.0 * b * g).max(0.0);
    let s2 = (b + g + d + disc.sqrt()) / 2.0;
    let inv_s = 1.0 / s2.sqrt();

    let clamp = |x: Real| x.clamp(-1.0, 1.0);
    let gamma = clamp((g / s2).sqrt()).acos();
    let beta = clamp(u1 * inv_s / period / ca).acos();

    let fine = estimate(plane1, plane2, calib);
    let (x, y) = (fine.x, fine.y);
    // C++ takes the pose's in-plane angle from the second plane.
    let alpha = plane2.orientation() - PI / 2.0;
    let ps = inv_s;

    [
        Pose::new_3d(x, y, 0.0, alpha, beta, gamma, ps),
        Pose::new_3d(x, y, 0.0, alpha, beta, -gamma, ps),
        Pose::new_3d(x, y, 0.0, alpha, -beta, gamma, ps),
        Pose::new_3d(x, y, 0.0, alpha, -beta, -gamma, ps),
    ]
}

/// Recovers the signs of `(beta, gamma)` from the curvature of the two measured
/// unwrapped phase maps, porting C++ `PatternPhase::computePhaseGradients`.
/// `phase1`/`phase2` are the measured unwrapped maps carried by
/// [`Detection`](vernier_spectral::spectrum::Detection), row-major
/// `width × height`; `crop_factor` matches the plane fit (0.5). Returns
/// each sign in `{-1, 0, 1}` to pick among the [`all_3d_poses`] candidates.
pub fn compute_phase_gradients(
    phase1: &[Real],
    phase2: &[Real],
    width: usize,
    height: usize,
    plane1: &PhasePlane,
    plane2: &PhasePlane,
    crop_factor: Real,
) -> (i32, i32) {
    let side = ((width as Real * crop_factor) / 2.0) as usize;
    let beta = gradient_sign(phase1, width, height, side, plane1.a, plane1.b, true);
    let gamma = gradient_sign(phase2, width, height, side, plane2.a, plane2.b, false);
    (beta, gamma)
}

/// Builds the C++ `phaseDerived` field over the cropped phase map and returns
/// `sign(mean(Sobel(phaseDerived)))` — Sobel-X (`horizontal`) for beta,
/// Sobel-Y for gamma.
fn gradient_sign(
    phase: &[Real],
    width: usize,
    height: usize,
    side: usize,
    a: Real,
    b: Real,
    horizontal: bool,
) -> i32 {
    if width < 2 * side + 2 || height < 2 * side + 2 {
        return 0;
    }
    let cw = width - 2 * side;
    let ch = height - 2 * side;
    let denom = a * a + b * b;
    if denom == 0.0 {
        return 0;
    }
    let get = |i: usize, j: usize| -> f64 { phase[(i + side) * width + (j + side)] };

    let (dw, dh) = (cw - 1, ch - 1);
    let mut derived = vec![0.0f64; dw * dh];
    for i in 0..dh {
        for j in 0..dw {
            let dx = -(get(i + 1, j) - get(i, j)) * a / denom;
            let dy = (get(i, j + 1) - get(i, j)) * b / denom;
            derived[i * dw + j] = dx + dy;
        }
    }

    let mean = sobel_mean(&derived, dw, dh, horizontal);
    (mean > 0.0) as i32 - (mean < 0.0) as i32
}

/// Mean of a 3×3 Sobel derivative over `field` (`BORDER_REFLECT_101`, matching
/// OpenCV's default). `horizontal` selects Sobel-X, else Sobel-Y.
fn sobel_mean(field: &[f64], w: usize, h: usize, horizontal: bool) -> f64 {
    if w == 0 || h == 0 {
        return 0.0;
    }
    // Sobel-X and Sobel-Y kernels.
    let kx = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
    let ky = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];
    let k = if horizontal { kx } else { ky };

    let reflect = |idx: i32, n: usize| -> usize {
        if n == 1 {
            return 0;
        }
        let n = n as i32;
        let mut v = idx;
        loop {
            if v < 0 {
                v = -v;
            } else if v >= n {
                v = 2 * (n - 1) - v;
            } else {
                break;
            }
        }
        v as usize
    };

    let mut sum = 0.0;
    for i in 0..h {
        for j in 0..w {
            let mut acc = 0.0;
            for (di, krow) in k.iter().enumerate() {
                let ii = reflect(i as i32 + di as i32 - 1, h);
                for (dj, &kv) in krow.iter().enumerate() {
                    let jj = reflect(j as i32 + dj as i32 - 1, w);
                    acc += kv * field[ii * w + jj];
                }
            }
            sum += acc;
        }
    }
    sum / (w * h) as f64
}

/// Fine estimate from a single phase plane: recovers one translation component
/// and the orientation, leaving the orthogonal component at zero.
pub fn estimate_single(plane: &PhasePlane, calib: &Calibration) -> Pose {
    let theta = plane.orientation();
    let disp = -(plane.c / TAU) * calib.period;
    Pose::new(disp * theta.cos(), disp * theta.sin(), theta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vernier_core::Real;
    use vernier_core::scalar::consts::PI;
    use vernier_spectral::PhasePlane;

    fn plane(a: Real, b: Real, c: Real) -> PhasePlane {
        PhasePlane { a, b, c }
    }

    #[test]
    fn zero_phase_zero_position() {
        let calib = Calibration::new(10.0, 32, 32);
        // Horizontal direction (gradient along x), zero center phase.
        let p1 = plane(0.5, 0.0, 0.0);
        let p2 = plane(0.0, 0.5, 0.0);
        let pose = estimate(&p1, &p2, &calib);
        assert!(pose.x.abs() < 1e-6 && pose.y.abs() < 1e-6);
        assert!(pose.theta.abs() < 1e-6);
    }

    #[test]
    fn quarter_period_phase_gives_minus_quarter_period_shift() {
        let calib = Calibration::new(10.0, 32, 32);
        // c = π/2 is a quarter of 2π -> x = -period/4 = -2.5, as C++ get2DPose.
        let p1 = plane(0.5, 0.0, PI / 2.0);
        let p2 = plane(0.0, 0.5, 0.0);
        let pose = estimate(&p1, &p2, &calib);
        assert!((pose.x + 2.5).abs() < 1e-4, "x={}", pose.x);
        // A gradient of 0.5 rad/px is a 4π px period: 10 / 4π units per pixel.
        assert!(
            (pose.pixel_size - 10.0 / (4.0 * PI)).abs() < 1e-9,
            "{}",
            pose.pixel_size
        );
    }

    #[test]
    fn all_3d_poses_flat_pattern_is_untilted() {
        let calib = Calibration::new(10.0, 64, 64);
        let a = 0.5 as Real;
        // Two perpendicular directions, equal gradient magnitude, no tilt.
        let p1 = plane(a, 0.0, 0.0);
        let p2 = plane(0.0, a, 0.0);
        let poses = all_3d_poses(&p1, &p2, &calib);
        assert_eq!(poses.len(), 4);
        for pz in &poses {
            assert!(pz.is_3d);
            assert!(pz.beta.abs() < 1e-4, "beta={}", pz.beta);
            assert!(pz.gamma.abs() < 1e-4, "gamma={}", pz.gamma);
        }
        // pixel_size = period / pixelic_period, pixelic_period = 2π / a.
        let expected_ps = 10.0 / (TAU / a);
        assert!(
            (poses[0].pixel_size - expected_ps).abs() < 1e-3,
            "ps={}",
            poses[0].pixel_size
        );
    }

    #[test]
    fn phase_gradient_sign_follows_curvature() {
        // The phase-derivative formula for direction 1 with plane (a=0, b>0)
        // reduces to 2·b/denom·(∂phase/∂col). A curvature k·x² (x = col) then
        // makes phaseDerived trend linearly in col, so Sobel-X mean has sign k.
        let (w, h) = (64usize, 64usize);
        let b = 0.5 as Real;
        let p1 = plane(0.0, b, 0.0);
        let p2 = plane(b, 0.0, 0.0);

        let build = |k: Real| -> Vec<Real> {
            let mut v = vec![0.0 as Real; w * h];
            for row in 0..h {
                for col in 0..w {
                    let x = col as Real - w as Real / 2.0;
                    let y = row as Real - h as Real / 2.0;
                    // plane (b·y) + quadratic curvature in x scaled by k.
                    v[row * w + col] = b * y + k * x * x;
                }
            }
            v
        };

        let flat = vec![0.0 as Real; w * h];
        let (beta_pos, _) = compute_phase_gradients(&build(0.001), &flat, w, h, &p1, &p2, 0.5);
        let (beta_neg, _) = compute_phase_gradients(&build(-0.001), &flat, w, h, &p1, &p2, 0.5);
        assert_eq!(beta_pos, 1, "positive curvature -> +1");
        assert_eq!(beta_neg, -1, "negative curvature -> -1");
    }

    #[test]
    fn orientation_from_gradients() {
        let calib = Calibration::new(10.0, 32, 32);
        // Equal a and b -> 45 degrees.
        let p1 = plane(0.3, 0.3, 0.0);
        let p2 = plane(-0.3, 0.3, 0.0);
        let pose = estimate(&p1, &p2, &calib);
        assert!((pose.theta - PI / 4.0).abs() < 1e-4, "theta={}", pose.theta);
    }
}
