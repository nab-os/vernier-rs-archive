//! Defocus: the local demodulation of `vernier-camera` against the global
//! band-pass of `vernier-pose`, on the views the global method is made for.
//!
//! The board is rendered square on to the camera, at random positions and
//! turns, with the renderer `vernier-pose`'s degradation suites use, then
//! blurred by a Gaussian of growing sigma, with and without a little sensor
//! noise. Both methods are judged on what the global one gives: the pattern
//! position under the image centre and the turn. The local method's points are
//! brought to that by a similarity fit, and are also compared one by one with
//! the truth.
//!
//! ```text
//! cargo run --release -p vernier-camera --example defocus [-- results.csv]
//! ```

use std::io::Write;

use rayon::prelude::*;
use vernier_camera::{PointMatch, Target, measure_view_traced};
use vernier_core::buffer::BufferLayout;
use vernier_cpu::CpuBackend;
use vernier_patterns::PatternPose;
use vernier_patterns::checkerboard::{Checkerboard, CodeLayout};
use vernier_patterns::render::into_pattern_frame;
use vernier_pose::checkerboard::{detect_checkerboard, solve_checkerboard_with_layout};

const ORDER: u32 = 8;
const POSES: usize = 24;
/// Square sides in pixels: the suites' 8 (carrier period 11.3 px), and 14
/// (period 19.8 px, as in the 720p camera benchmark).
const SQUARES: [f64; 2] = [8.0, 14.0];
/// Sensor noise, standard deviation on intensities in `0..=1`.
const NOISES: [f64; 2] = [0.0, 0.01];
/// Defocus steps, px, up to half a carrier period.
const SIGMA_STEP: f64 = 0.5;

/// Frame side for a square size: the suites' 512 px for 8 px squares, and as
/// many squares across for larger ones, enough to read the code.
fn frame_size(square: f64) -> usize {
    (64.0 * square) as usize
}

/// SplitMix64: small, seeded, good enough for noise.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..1`.
    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal, by Box–Muller.
    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform().max(1e-300), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

/// A square `size`×`size` image blurred by a Gaussian of `sigma` px, as two
/// one-dimensional passes cut at 3 sigma.
fn gaussian_blur(image: &[f32], sigma: f64, size: usize) -> Vec<f32> {
    if sigma <= 0.0 {
        return image.to_vec();
    }
    let radius = (3.0 * sigma).ceil() as isize;
    let kernel: Vec<f64> = (-radius..=radius)
        .map(|offset| (-0.5 * (offset as f64 / sigma).powi(2)).exp())
        .collect();
    let kernel_sum: f64 = kernel.iter().sum();
    // Edges clamped, as the suites do.
    let clamp = |v: isize| v.clamp(0, size as isize - 1) as usize;
    let pass = |src: &[f32], horizontal: bool| -> Vec<f32> {
        (0..size * size)
            .into_par_iter()
            .map(|i| {
                let (x, y) = ((i % size) as isize, (i / size) as isize);
                let sum: f64 = kernel
                    .iter()
                    .enumerate()
                    .map(|(tap, &weight)| {
                        let offset = tap as isize - radius;
                        let at = if horizontal {
                            y as usize * size + clamp(x + offset)
                        } else {
                            clamp(y + offset) * size + x as usize
                        };
                        weight * src[at] as f64
                    })
                    .sum();
                (sum / kernel_sum) as f32
            })
            .collect()
    };
    pass(&pass(image, true), false)
}

/// The image a camera would give: `render` blurred by `sigma`, plus sensor
/// noise of deviation `noise` seeded by the pose and sigma, clipped to `0..=1`.
fn degrade(render: &[f32], size: usize, sigma: f64, noise: f64, pose_index: usize) -> Vec<f32> {
    let mut image = gaussian_blur(render, sigma, size);
    if noise > 0.0 {
        let mut rng = Rng(pose_index as u64 * 7919 + (sigma * 100.0) as u64 + 1);
        image
            .iter_mut()
            .for_each(|v| *v += (noise * rng.normal()) as f32);
    }
    image.iter_mut().for_each(|v| *v = v.clamp(0.0, 1.0));
    image
}

/// An angle brought into `-π..π`.
fn wrap_angle(a: f64) -> f64 {
    let tau = std::f64::consts::TAU;
    a - tau * ((a + std::f64::consts::PI) / tau).floor()
}

#[derive(Clone, Copy, PartialEq)]
enum Status {
    /// No carrier or board found.
    Lost,
    /// Measured, but the code did not read.
    NoCode,
    /// Read, but more than half a square from the truth.
    Wrong,
    Correct,
}

/// What one method made of one image.
struct Outcome {
    status: Status,
    /// Position under the image centre, px, and turn, degrees, when correct.
    error: Option<(f64, f64)>,
    /// Local method only: rms over its points against the truth, px.
    points_rms: Option<f64>,
    points: usize,
    /// Local method only: the defocus it fitted, px.
    defocus: Option<f64>,
}

impl Outcome {
    fn failed(status: Status) -> Self {
        Self {
            status,
            error: None,
            points_rms: None,
            points: 0,
            defocus: None,
        }
    }

    fn position_error(&self) -> Option<f64> {
        self.error.map(|(position, _)| position)
    }

    fn turn_error(&self) -> Option<f64> {
        self.error.map(|(_, turn)| turn)
    }
}

/// The global band-pass of `vernier-pose`: one pattern pose for the whole
/// image, compared with the pose it was rendered at.
fn global(
    image: &[f32],
    size: usize,
    pattern: &Checkerboard,
    square: f64,
    pose: &PatternPose,
) -> Outcome {
    let backend = CpuBackend::new();
    let Ok(detection) = detect_checkerboard(
        &backend,
        image,
        BufferLayout::packed(size, size),
        4.0,
        10,
        0,
        0.0,
    ) else {
        return Outcome::failed(Status::Lost);
    };
    let Ok((recovered, _)) =
        solve_checkerboard_with_layout(&detection, image, square, ORDER, CodeLayout::Squares)
    else {
        return Outcome::failed(Status::NoCode);
    };
    let (ex, ey) = pattern.wrap_offset(recovered.x + pose.x, recovered.y + pose.y);
    if ex.abs() >= 0.5 * square || ey.abs() >= 0.5 * square {
        return Outcome::failed(Status::Wrong);
    }
    Outcome {
        status: Status::Correct,
        error: Some((
            ex.hypot(ey),
            wrap_angle(recovered.theta - pose.theta).abs().to_degrees(),
        )),
        points_rms: None,
        points: 0,
        defocus: None,
    }
}

/// The local demodulation of `vernier-camera`: points one by one, checked
/// against the truth, then reduced by a similarity fit to the global method's
/// position under the centre and turn.
fn local(
    image: &[f32],
    size: usize,
    pattern: &Checkerboard,
    square: f64,
    pose: &PatternPose,
) -> Outcome {
    let (measured, trace) =
        measure_view_traced(image, size, size, &Target::new(square, ORDER), None, false);
    let defocus = trace.chosen.and_then(|i| trace.attempts[i].defocus);
    let Ok(view) = measured else {
        return Outcome::failed(Status::Lost);
    };
    if !view.is_absolute() {
        return Outcome::failed(Status::NoCode);
    }
    let centre = size as f64 / 2.0;
    // The board under each pixel, as the renderer placed it.
    let truth = |p: [f64; 2]| {
        let (x, y) = into_pattern_frame(p[0], p[1], centre, centre, pose.theta);
        (x - pose.x, y - pose.y)
    };
    let errors: Vec<f64> = view
        .points
        .iter()
        .map(|p| {
            let (tx, ty) = truth(p.pixel);
            let (ex, ey) = pattern.wrap_offset(p.board[0] - tx, p.board[1] - ty);
            ex.hypot(ey)
        })
        .collect();
    let points_rms = (errors.iter().map(|e| e * e).sum::<f64>() / errors.len() as f64).sqrt();

    let ((board_x, board_y), (scale_re, scale_im)) = similarity_at(&view.points, centre);
    // The truth under the centre is −pose; the renderer turns by −θ.
    let (ex, ey) = pattern.wrap_offset(board_x + pose.x, board_y + pose.y);
    if ex.abs() >= 0.5 * square || ey.abs() >= 0.5 * square {
        return Outcome::failed(Status::Wrong);
    }
    let turned = -scale_im.atan2(scale_re);
    Outcome {
        status: Status::Correct,
        error: Some((
            ex.hypot(ey),
            wrap_angle(turned - pose.theta).abs().to_degrees(),
        )),
        points_rms: Some(points_rms),
        points: view.points.len(),
        defocus,
    }
}

/// Least-squares similarity from pixel to board, in complex numbers
/// `b = a·(p − p̄) + b̄`: the board point it puts at pixel (`centre`,
/// `centre`), and the complex factor `a` (scale and turn) as (re, im).
fn similarity_at(points: &[PointMatch], centre: f64) -> ((f64, f64), (f64, f64)) {
    let n = points.len() as f64;
    let mean = |f: &dyn Fn(&PointMatch) -> f64| points.iter().map(f).sum::<f64>() / n;
    let (px, py) = (mean(&|p| p.pixel[0]), mean(&|p| p.pixel[1]));
    let (bx, by) = (mean(&|p| p.board[0]), mean(&|p| p.board[1]));
    let (mut re, mut im, mut norm) = (0.0, 0.0, 0.0);
    for p in points {
        let (u, v) = (p.pixel[0] - px, p.pixel[1] - py);
        let (s, t) = (p.board[0] - bx, p.board[1] - by);
        // (s + it)·conj(u + iv)
        re += s * u + t * v;
        im += t * u - s * v;
        norm += u * u + v * v;
    }
    let (ar, ai) = (re / norm, im / norm);
    let (dx, dy) = (centre - px, centre - py);
    let board_at_centre = (bx + ar * dx - ai * dy, by + ai * dx + ar * dy);
    (board_at_centre, (ar, ai))
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

/// Median of a quantity over the outcomes that have it.
fn median_of(outcomes: &[Outcome], quantity: impl Fn(&Outcome) -> Option<f64>) -> f64 {
    let mut values: Vec<f64> = outcomes.iter().filter_map(quantity).collect();
    median(&mut values)
}

fn count(outcomes: &[Outcome], status: Status) -> usize {
    outcomes.iter().filter(|o| o.status == status).count()
}

/// `correct/total`, then what the rest became.
fn tally(outcomes: &[Outcome]) -> String {
    let mut text = format!("{}/{}", count(outcomes, Status::Correct), outcomes.len());
    let rest: Vec<String> = [
        (Status::Lost, "lost"),
        (Status::NoCode, "no code"),
        (Status::Wrong, "wrong"),
    ]
    .iter()
    .filter(|(status, _)| count(outcomes, *status) > 0)
    .map(|(status, name)| format!("{} {name}", count(outcomes, *status)))
    .collect();
    if !rest.is_empty() {
        text += &format!(" ({})", rest.join(", "));
    }
    text
}

/// Pattern poses spread over the code's range, at any turn.
fn random_poses() -> Vec<PatternPose> {
    let mut rng = Rng(7);
    (0..POSES)
        .map(|_| {
            PatternPose::new(
                (rng.uniform() - 0.5) * 4000.0,
                (rng.uniform() - 0.5) * 4000.0,
                rng.uniform() * std::f64::consts::TAU,
            )
        })
        .collect()
}

/// One square size: the board, its pose renders, and the frame side.
struct Board {
    square: f64,
    size: usize,
    pattern: Checkerboard,
    renders: Vec<Vec<f32>>,
}

/// Both methods on every pose, degraded by `sigma` and `noise`: the global
/// outcomes, then the local ones.
fn run_both(
    board: &Board,
    poses: &[PatternPose],
    sigma: f64,
    noise: f64,
) -> (Vec<Outcome>, Vec<Outcome>) {
    let runs: Vec<(Outcome, Outcome)> = poses
        .par_iter()
        .enumerate()
        .map(|(index, pose)| {
            let image = degrade(&board.renders[index], board.size, sigma, noise, index);
            (
                global(&image, board.size, &board.pattern, board.square, pose),
                local(&image, board.size, &board.pattern, board.square, pose),
            )
        })
        .collect();
    runs.into_iter().unzip()
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        std::env::temp_dir()
            .join("vernier-defocus.csv")
            .display()
            .to_string()
    });
    let mut csv = std::fs::File::create(&path).expect("csv");
    writeln!(
        csv,
        "square_px,noise,sigma,method,correct,lost,no_code,wrong,total,median_position_px,median_turn_deg,median_points_rms_px,mean_points"
    )
    .unwrap();

    let poses = random_poses();
    for square in SQUARES {
        let size = frame_size(square);
        let pattern = Checkerboard::new(square, ORDER).expect("order");
        let renders = poses
            .iter()
            .map(|pose| pattern.render(size, size, pose).as_slice().to_vec())
            .collect();
        let board = Board {
            square,
            size,
            pattern,
            renders,
        };
        let period = square * std::f64::consts::SQRT_2;
        for noise in NOISES {
            println!(
                "\n### squares {square} px (carrier period {period:.1} px), {size}×{size}, noise σ {noise}, {POSES} poses square on\n"
            );
            println!(
                "| defocus σ px | σ / period | global: correct | local: correct | global position px | local position px | global turn ° | local turn ° | local points rms px | local points | local fitted σ px |"
            );
            println!("|---|---|---|---|---|---|---|---|---|---|---|");
            let mut sigma: f64 = 0.0;
            // The slack keeps rounding from dropping a step that lands on the limit.
            while sigma <= 0.5 * period + 1e-9 {
                let (global, local) = run_both(&board, &poses, sigma, noise);
                let points_rms = median_of(&local, |o| o.points_rms);
                let correct = count(&local, Status::Correct);
                let mean_points =
                    local.iter().map(|o| o.points).sum::<usize>() as f64 / correct.max(1) as f64;
                let fitted_defocus = median_of(&local, |o| o.defocus);
                println!(
                    "| {sigma:.1} | {:.2} | {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {points_rms:.4} | {mean_points:.0} | {fitted_defocus:.2} |",
                    sigma / period,
                    tally(&global),
                    tally(&local),
                    median_of(&global, Outcome::position_error),
                    median_of(&local, Outcome::position_error),
                    median_of(&global, Outcome::turn_error),
                    median_of(&local, Outcome::turn_error),
                );
                let methods = [
                    ("global", &global, f64::NAN, 0.0),
                    ("local", &local, points_rms, mean_points),
                ];
                for (name, outcomes, rms, points) in methods {
                    writeln!(
                        csv,
                        "{square},{noise},{sigma},{name},{},{},{},{},{},{:.5},{:.5},{rms:.5},{points:.0}",
                        count(outcomes, Status::Correct),
                        count(outcomes, Status::Lost),
                        count(outcomes, Status::NoCode),
                        count(outcomes, Status::Wrong),
                        outcomes.len(),
                        median_of(outcomes, Outcome::position_error),
                        median_of(outcomes, Outcome::turn_error),
                    )
                    .unwrap();
                }
                std::io::stdout().flush().unwrap();
                sigma += SIGMA_STEP;
            }
        }
    }
    eprintln!("\nwrote {path}");
}
