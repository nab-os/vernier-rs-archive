//! Quantitative accuracy regression tests.
//!
//! These pin the *measurement* quality of the pipeline, not just its
//! self-consistency: a rendered ground-truth displacement must be recovered to
//! sub-millipixel accuracy. They guard the properties the papers claim (the
//! least-squares phase-plane fit and f64 pose math) — a regression to a
//! noisier estimator or to f32 accumulation fails these bounds.

use vernier_core::buffer::BufferLayout;
use vernier_core::{Complex32, Real};
use vernier_cpu::CpuBackend;
use vernier_patterns::PatternPose;
use vernier_patterns::megarena::Megarena;
use vernier_patterns::periodic::Periodic;
use vernier_pose::{Calibration, absolute, periodic};
use vernier_spectral::spectrum::{Detection, analyze_two};

/// Deterministic xorshift + Box-Muller for reproducible noise without deps.
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f32 / (1u64 << 53) as f32
    }
    fn gauss(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-12);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

fn detect(image: &[f32], size: usize, noise: Option<(f32, u64)>) -> Detection {
    let backend = CpuBackend::new();
    let layout = BufferLayout::packed(size, size);
    let complex: Vec<Complex32> = match noise {
        None => image.iter().map(|&v| Complex32::new(v, 0.0)).collect(),
        Some((sigma, seed)) => {
            let mut rng = Rng(0x9E3779B97F4A7C15u64.wrapping_mul(seed));
            image
                .iter()
                .map(|&v| Complex32::new(v + sigma * rng.gauss(), 0.0))
                .collect()
        }
    };
    analyze_two(&backend, &complex, layout, 3.0, 20, 500, 0.5).expect("detection failed")
}

/// Renders a periodic pattern at two poses one known sub-pixel step apart and
/// checks the pipeline recovers the step. Differential measurement cancels any
/// constant frame-convention offset, so this directly tests displacement
/// sensitivity — the quantity the method's resolution claims are about.
fn periodic_step_error(size: usize, step: Real, noise: Option<(f32, u64)>) -> Real {
    let period_px: Real = 10.0;
    let pattern = Periodic::new(period_px);
    let calib = Calibration::new(period_px, size, size);

    // A generic pose: slight rotation, fractional offsets.
    let base = PatternPose::new(0.1234, -0.271, 0.05);
    let moved = PatternPose::new(base.x + step, base.y, base.theta);

    let img0 = pattern.render(size, size, &base);
    let img1 = pattern.render(size, size, &moved);

    let noise1 = noise.map(|(s, seed)| (s, seed + 1));
    let det0 = detect(img0.as_slice(), size, noise);
    let det1 = detect(img1.as_slice(), size, noise1);

    let pose0 = periodic::estimate(&det0.dir1.plane, &det0.dir2.plane, &calib);
    let pose1 = periodic::estimate(&det1.dir1.plane, &det1.dir2.plane, &calib);

    // Wrap the measured step into (-period/2, period/2] before comparing.
    let mut measured = pose1.x - pose0.x;
    measured = (measured + period_px / 2.0).rem_euclid(period_px) - period_px / 2.0;
    (measured.abs() - step.abs()).abs()
}

#[test]
fn periodic_recovers_subpixel_step_noise_free() {
    // Noise-free: error budget is edge effects + estimator bias only.
    let err = periodic_step_error(512, 0.37, None);
    assert!(err < 1e-4, "step error {err} px exceeds 1e-4 px");
}

#[test]
fn periodic_recovers_subpixel_step_under_noise() {
    // 2% full-scale Gaussian noise over five fixed seeds (deterministic). The
    // LSQ plane fit keeps the mean step error near 4e-4 px here; the pre-LSQ
    // estimator (single-pixel c) sat ~3.7× higher and fails this bound.
    let seeds = [7u64, 21, 63, 99, 123];
    let mean_err: Real = seeds
        .iter()
        .map(|&seed| periodic_step_error(512, 0.37, Some((0.02, seed))))
        .sum::<Real>()
        / seeds.len() as Real;
    assert!(
        mean_err < 8e-4,
        "mean step error {mean_err} px exceeds 8e-4 px"
    );
}

#[test]
fn periodic_orientation_recovers_small_rotation() {
    let size = 512usize;
    let period_px: Real = 10.0;
    let pattern = Periodic::new(period_px);
    let dtheta: Real = 2e-3;

    let img0 = pattern.render(size, size, &PatternPose::new(0.0, 0.0, 0.05));
    let img1 = pattern.render(size, size, &PatternPose::new(0.0, 0.0, 0.05 + dtheta));
    let det0 = detect(img0.as_slice(), size, None);
    let det1 = detect(img1.as_slice(), size, None);

    let measured = det1.dir1.plane.orientation() - det0.dir1.plane.orientation();
    let err = (measured.abs() - dtheta).abs();
    assert!(err < 1e-5, "rotation step error {err} rad exceeds 1e-5 rad");
}

#[test]
fn megarena_absolute_step_and_theta() {
    // Absolute path: a known sub-period step must appear in the absolute
    // position, and theta must come out quadrant-resolved (≈ rendered theta,
    // not offset by a multiple of π/2).
    let size = 512usize;
    let period_px: Real = 12.0;
    let order = 8u32;
    let step: Real = 0.43;

    let pattern = Megarena::new(period_px, order)
        .unwrap()
        .with_lfsr_offset(order as i64);
    let calib = Calibration::new(period_px, size, size);

    let img0 = pattern.render(size, size, &PatternPose::new(0.0, 0.0, 0.0));
    let img1 = pattern.render(size, size, &PatternPose::new(step, 0.0, 0.0));
    let det0 = detect(img0.as_slice(), size, None);
    let det1 = detect(img1.as_slice(), size, None);

    let pose0 = absolute::solve_megarena(&det0, img0.as_slice(), &calib, order)
        .expect("absolute solve failed");
    let pose1 = absolute::solve_megarena(&det1, img1.as_slice(), &calib, order)
        .expect("absolute solve failed");

    let measured = (pose1.x - pose0.x).abs();
    let err = (measured - step).abs();
    assert!(err < 5e-3, "absolute step error {err} px exceeds 5e-3 px");
    // Same absolute cell: y must not jump by a period.
    assert!(
        (pose1.y - pose0.y).abs() < 0.5,
        "y jumped: {} vs {}",
        pose0.y,
        pose1.y
    );
    // Rendered theta = 0: the solved theta must be ~0 (not a stray k·π/2).
    assert!(
        pose0.theta.abs() < 1e-3,
        "theta {} not quadrant-resolved",
        pose0.theta
    );
}
