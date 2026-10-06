//! The blur fit behind the repaint: how close the fitted defocus comes to the
//! blur in the frame, what it costs, and what following the previous frame's
//! defocus saves in tracking.
//!
//! The board is rendered square on at random poses and turns, blurred by a
//! Gaussian of growing sigma, with a little sensor noise. Every frame is
//! measured on its own, then the same frames are measured as a video, each
//! after the previous one, with the pose and the blur drifting a little from
//! frame to frame.
//!
//! ```text
//! cargo run --release -p vernier-camera --example blur_fit
//! ```

use vernier_camera::{Target, Trace, View, measure_view_traced};
use vernier_patterns::PatternPose;
use vernier_patterns::checkerboard::Checkerboard;
use vernier_patterns::render::into_pattern_frame;

const ORDER: u32 = 8;
const POSES: usize = 12;
/// Square sides in pixels: carrier periods 11.3 and 19.8 px.
const SQUARES: [f64; 2] = [8.0, 14.0];
const NOISE: f64 = 0.01;
/// Frames in each tracked sequence.
const FRAMES: usize = 24;

/// SplitMix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform().max(1e-300), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

/// `image` blurred by a Gaussian of `sigma` px cut at 3 sigma, edges clamped,
/// plus noise, clipped to `0..=1`.
fn degrade(image: &[f32], size: usize, sigma: f64, rng: &mut Rng) -> Vec<f32> {
    let mut out = image.to_vec();
    if sigma > 0.0 {
        let radius = (3.0 * sigma).ceil() as isize;
        let kernel: Vec<f64> = (-radius..=radius)
            .map(|q| (-0.5 * (q as f64 / sigma).powi(2)).exp())
            .collect();
        let total: f64 = kernel.iter().sum();
        let clamp = |v: isize| v.clamp(0, size as isize - 1) as usize;
        for horizontal in [true, false] {
            let src = out.clone();
            for (i, v) in out.iter_mut().enumerate() {
                let (x, y) = ((i % size) as isize, (i / size) as isize);
                let sum: f64 = kernel
                    .iter()
                    .enumerate()
                    .map(|(t, &k)| {
                        let q = t as isize - radius;
                        let at = if horizontal {
                            y as usize * size + clamp(x + q)
                        } else {
                            clamp(y + q) * size + x as usize
                        };
                        k * src[at] as f64
                    })
                    .sum();
                *v = (sum / total) as f32;
            }
        }
    }
    out.iter_mut()
        .for_each(|v| *v = (*v as f64 + NOISE * rng.normal()).clamp(0.0, 1.0) as f32);
    out
}

/// The blur a frame really carries: the applied kernel, sampled at whole
/// pixels and cut at 3 sigma, plus the renderer's supersampled pixel.
fn applied(sigma: f64) -> f64 {
    let pixel: f64 = 1.0 / 12.0;
    if sigma <= 0.0 {
        return pixel.sqrt();
    }
    let radius = (3.0 * sigma).ceil() as i64;
    let weights: Vec<(f64, f64)> = (-radius..=radius)
        .map(|q| (q as f64, (-0.5 * (q as f64 / sigma).powi(2)).exp()))
        .collect();
    let total: f64 = weights.iter().map(|w| w.1).sum();
    let variance: f64 = weights.iter().map(|(q, w)| q * q * w).sum::<f64>() / total;
    (variance + pixel).sqrt()
}

/// What one measurement gave.
struct Run {
    fitted: Option<f64>,
    /// Misfits the blur fit evaluated.
    misfits: usize,
    /// Milliseconds painting the code back, the blur fit included.
    restore_ms: f64,
    /// Rms of the points against the truth, px.
    rms: Option<f64>,
    view: Option<View>,
}

fn run(
    image: &[f32],
    size: usize,
    square: f64,
    pattern: &Checkerboard,
    pose: &PatternPose,
    previous: Option<&View>,
) -> Run {
    let target = Target::new(square, ORDER);
    let (view, trace): (_, Trace) =
        measure_view_traced(image, size, size, &target, previous, false);
    let attempt = trace.chosen.map(|i| &trace.attempts[i]);
    let fitted = attempt.and_then(|a| a.defocus);
    let misfits = attempt.map_or(0, |a| a.defocus_misfits);
    let restore_ms = attempt
        .and_then(|a| a.timings.iter().find(|t| t.0 == "restore"))
        .map_or(f64::NAN, |t| t.1);
    let centre = size as f64 / 2.0;
    let rms = view.as_ref().ok().filter(|v| v.is_absolute()).map(|v| {
        let sum: f64 = v
            .points
            .iter()
            .map(|p| {
                let (x, y) = into_pattern_frame(p.pixel[0], p.pixel[1], centre, centre, pose.theta);
                let (ex, ey) =
                    pattern.wrap_offset(p.board[0] - (x - pose.x), p.board[1] - (y - pose.y));
                ex * ex + ey * ey
            })
            .sum();
        (sum / v.points.len() as f64).sqrt()
    });
    Run {
        fitted,
        misfits,
        restore_ms,
        rms,
        view: view.ok(),
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    values.retain(|v| v.is_finite());
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

fn mean_misfits(runs: &[Run]) -> f64 {
    runs.iter().map(|r| r.misfits as f64).sum::<f64>() / runs.len().max(1) as f64
}

fn main() {
    let mut rng = Rng(7);
    let poses: Vec<PatternPose> = (0..POSES)
        .map(|_| {
            PatternPose::new(
                (rng.uniform() - 0.5) * 4000.0,
                (rng.uniform() - 0.5) * 4000.0,
                rng.uniform() * std::f64::consts::TAU,
            )
        })
        .collect();
    // A warm-up, so the first timings do not pay for page faults.
    {
        let pattern = Checkerboard::new(8.0, ORDER).unwrap();
        let render = pattern.render(512, 512, &poses[0]).as_slice().to_vec();
        run(&render, 512, 8.0, &pattern, &poses[0], None);
    }

    println!("## Single frames, {POSES} poses, noise {NOISE}\n");
    println!("square  sigma  applied  fitted  bias     points rms  restore ms  misfits");
    for square in SQUARES {
        let size = (64.0 * square) as usize;
        let pattern = Checkerboard::new(square, ORDER).unwrap();
        let renders: Vec<Vec<f32>> = poses
            .iter()
            .map(|p| pattern.render(size, size, p).as_slice().to_vec())
            .collect();
        let max_sigma = 0.25 * square * std::f64::consts::SQRT_2;
        let mut sigma = 0.0;
        while sigma <= max_sigma {
            let runs: Vec<Run> = poses
                .iter()
                .zip(&renders)
                .map(|(pose, render)| {
                    let image = degrade(render, size, sigma, &mut rng);
                    run(&image, size, square, &pattern, pose, None)
                })
                .collect();
            let fitted = median(runs.iter().filter_map(|r| r.fitted).collect());
            println!(
                "{square:>6}  {sigma:>5.1}  {:>7.3}  {fitted:>6.3}  {:>+7.4}  {:>10.4}  {:>10.2}  {:>7.1}",
                applied(sigma),
                fitted - applied(sigma),
                median(runs.iter().filter_map(|r| r.rms).collect()),
                median(runs.iter().map(|r| r.restore_ms).collect()),
                mean_misfits(&runs),
            );
            sigma += 1.0;
        }
    }

    println!("\n## Tracked, {FRAMES} frames per sequence, blur and pose drifting\n");
    println!(
        "square  sigma  cold ms  tracked ms  cold rms  tracked rms  max |fit diff|  cold misfits  tracked misfits"
    );
    for square in SQUARES {
        let size = (64.0 * square) as usize;
        let pattern = Checkerboard::new(square, ORDER).unwrap();
        for start in [0.5, 1.5, 3.0] {
            let (mut cold, mut tracked) = (Vec::new(), Vec::new());
            let mut previous: Option<View> = None;
            let mut diff: f64 = 0.0;
            let mut pose = poses[0];
            for frame in 0..FRAMES {
                let sigma = start * (1.0 + 0.1 * (frame as f64 * 0.4).sin());
                pose = PatternPose::new(pose.x + 1.3, pose.y - 0.7, pose.theta + 0.004);
                let render = pattern.render(size, size, &pose).as_slice().to_vec();
                let image = degrade(&render, size, sigma, &mut rng);
                let a = run(&image, size, square, &pattern, &pose, None);
                let b = run(&image, size, square, &pattern, &pose, previous.as_ref());
                if let (Some(x), Some(y)) = (a.fitted, b.fitted) {
                    diff = diff.max((x - y).abs());
                }
                previous = b.view.clone();
                cold.push(a);
                tracked.push(b);
            }
            let ms = |runs: &[Run]| median(runs.iter().skip(1).map(|r| r.restore_ms).collect());
            let rms = |runs: &[Run]| median(runs.iter().filter_map(|r| r.rms).collect());
            println!(
                "{square:>6}  {start:>5.1}  {:>7.2}  {:>10.2}  {:>8.4}  {:>11.4}  {diff:>14.4}  {:>12.1}  {:>15.1}",
                ms(&cold),
                ms(&tracked),
                rms(&cold),
                rms(&tracked),
                mean_misfits(&cold),
                mean_misfits(&tracked[1..]),
            );
        }
    }
}
