//! The local demodulation of `vernier-camera` against the flat, global
//! band-pass of `vernier-pose`, each on the GPU and on the CPU: time per frame
//! and how close each lands to the truth.
//!
//! Two sets of frames:
//!
//! - square on, rendered as `vernier-pose`'s suites render, then blurred and
//!   noised: what the global method is made for. Both are judged on the
//!   pattern position under the image centre and the turn, the local one
//!   through a similarity fit of its points.
//! - tilted through a 720p camera: what the local method is made for. Its
//!   points are compared with the true projection; the global one is only
//!   timed and checked for a reading code, one similarity being no model of a
//!   perspective view.
//!
//! ```text
//! cargo run --release -p vernier-gpu --example demod_bench [-- results.csv]
//! ```

use std::io::Write;
use std::time::Instant;

use nalgebra::Vector3;
use vernier_camera::*;
use vernier_core::LocalDemodulator;
use vernier_core::backend::ComputeBackend;
use vernier_core::buffer::BufferLayout;
use vernier_cpu::CpuBackend;
use vernier_gpu::GpuBackend;
use vernier_patterns::PatternPose;
use vernier_patterns::checkerboard::{Checkerboard, CodeLayout};
use vernier_patterns::render::into_pattern_frame;
use vernier_pose::checkerboard::{detect_checkerboard, solve_checkerboard_with_layout};

const ORDER: u32 = 8;
/// Runs per frame and method; the fastest counts.
const RUNS: usize = 5;
const POSES: usize = 8;
const NOISE: f64 = 0.01;
const SIGMAS: [f64; 3] = [0.0, 2.0, 4.0];
/// Square side and frame size of the square-on sets: the suites' 512² with 8 px
/// squares, and 720p with the camera's 14 px.
const SQUARE_ON: [(f64, usize, usize); 2] = [(8.0, 512, 512), (14.0, 1280, 720)];
/// A local point further than this from the truth, in pixels, is counted as
/// wrong rather than imprecise.
const GROSS: f64 = 1.0;

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

/// `image` blurred by a Gaussian of `sigma` px, edges clamped, plus sensor
/// noise of deviation `noise` seeded by `seed`, clipped to `0..=1`.
fn degrade(image: &[f32], width: usize, height: usize, sigma: f64, noise: f64, seed: u64) -> Vec<f32> {
    let mut out = image.to_vec();
    if sigma > 0.0 {
        let radius = (3.0 * sigma).ceil() as isize;
        let kernel: Vec<f64> = (-radius..=radius)
            .map(|offset| (-0.5 * (offset as f64 / sigma).powi(2)).exp())
            .collect();
        let total: f64 = kernel.iter().sum();
        let pass = |src: &[f32], horizontal: bool| -> Vec<f32> {
            (0..width * height)
                .map(|i| {
                    let (x, y) = ((i % width) as isize, (i / width) as isize);
                    let sum: f64 = kernel
                        .iter()
                        .enumerate()
                        .map(|(tap, &weight)| {
                            let offset = tap as isize - radius;
                            let at = if horizontal {
                                y as usize * width + (x + offset).clamp(0, width as isize - 1) as usize
                            } else {
                                (y + offset).clamp(0, height as isize - 1) as usize * width + x as usize
                            };
                            weight * src[at] as f64
                        })
                        .sum();
                    (sum / total) as f32
                })
                .collect()
        };
        out = pass(&pass(&out, true), false);
    }
    let mut rng = Rng(seed);
    for v in &mut out {
        *v = (*v + (noise * rng.normal()) as f32).clamp(0.0, 1.0);
    }
    out
}

/// An angle brought into `-π..π`.
fn wrap_angle(a: f64) -> f64 {
    let tau = std::f64::consts::TAU;
    a - tau * ((a + std::f64::consts::PI) / tau).floor()
}

/// The fastest of `RUNS` calls of `f`, ms, and what the last one gave.
fn timed<T>(mut f: impl FnMut() -> T) -> (f64, T) {
    let mut best = f64::INFINITY;
    let mut last = None;
    for _ in 0..RUNS {
        let clock = Instant::now();
        let out = f();
        best = best.min(clock.elapsed().as_secs_f64() * 1e3);
        last = Some(out);
    }
    (best, last.expect("ran"))
}

/// What one method made of one frame.
#[derive(Default)]
struct Outcome {
    ms: f64,
    /// Of `ms`: the global method's detection, the local one's spectral search.
    first_ms: f64,
    /// The code read and, square on, the position was within half a square.
    correct: bool,
    /// Square on: position under the centre, px, and turn, degrees.
    error: Option<(f64, f64)>,
    /// Local only: each point's distance from the truth, px.
    point_errors: Vec<f64>,
}

/// The global band-pass: detection, then the code read off it.
fn flat<B: ComputeBackend>(
    backend: &B,
    image: &[f32],
    width: usize,
    height: usize,
    square: f64,
) -> (f64, f64, Option<PatternPose>) {
    let layout = BufferLayout::packed(width, height);
    let (detect_ms, detection) =
        timed(|| detect_checkerboard(backend, image, layout, 4.0, 10, 0, 0.0));
    let Ok(detection) = detection else {
        return (detect_ms, detect_ms, None);
    };
    let (solve_ms, solved) = timed(|| {
        solve_checkerboard_with_layout(&detection, image, square, ORDER, CodeLayout::Squares)
    });
    let pose = solved.ok().map(|(p, _)| PatternPose::new(p.x, p.y, p.theta));
    (detect_ms + solve_ms, detect_ms, pose)
}

/// The local demodulation on `demodulator`.
fn local<D: LocalDemodulator>(
    demodulator: &D,
    image: &[f32],
    width: usize,
    height: usize,
    target: &Target,
) -> (f64, f64, Option<View>) {
    let (ms, (measured, trace)) = timed(|| {
        measure_view_traced_with(demodulator, image, width, height, target, None, false)
    });
    let view = measured.ok().filter(View::is_absolute);
    (ms, trace.search_ms, view)
}

/// Least-squares similarity from pixel to board, `b = a·(p − p̄) + b̄` in
/// complex numbers: the board point it puts at pixel `centre`, and `a` as
/// (re, im).
fn similarity_at(points: &[PointMatch], centre: [f64; 2]) -> ([f64; 2], [f64; 2]) {
    let n = points.len() as f64;
    let mean = |f: &dyn Fn(&PointMatch) -> f64| points.iter().map(f).sum::<f64>() / n;
    let (px, py) = (mean(&|p| p.pixel[0]), mean(&|p| p.pixel[1]));
    let (bx, by) = (mean(&|p| p.board[0]), mean(&|p| p.board[1]));
    let (mut re, mut im, mut norm) = (0.0, 0.0, 0.0);
    for p in points {
        let (u, v) = (p.pixel[0] - px, p.pixel[1] - py);
        let (s, t) = (p.board[0] - bx, p.board[1] - by);
        re += s * u + t * v;
        im += t * u - s * v;
        norm += u * u + v * v;
    }
    let (ar, ai) = (re / norm, im / norm);
    let (dx, dy) = (centre[0] - px, centre[1] - py);
    ([bx + ar * dx - ai * dy, by + ai * dx + ar * dy], [ar, ai])
}

/// One square-on frame's truth.
struct SquareOn<'a> {
    pattern: &'a Checkerboard,
    square: f64,
    pose: &'a PatternPose,
    width: usize,
    height: usize,
}

impl SquareOn<'_> {
    fn centre(&self) -> [f64; 2] {
        [self.width as f64 / 2.0, self.height as f64 / 2.0]
    }

    /// Error of a position under the centre and a turn, if within half a
    /// square.
    fn judge(&self, x: f64, y: f64, theta: f64) -> Option<(f64, f64)> {
        let (ex, ey) = self.pattern.wrap_offset(x + self.pose.x, y + self.pose.y);
        (ex.abs() < 0.5 * self.square && ey.abs() < 0.5 * self.square).then(|| {
            (
                ex.hypot(ey),
                wrap_angle(theta - self.pose.theta).abs().to_degrees(),
            )
        })
    }

    fn flat(&self, (ms, first_ms, pose): (f64, f64, Option<PatternPose>)) -> Outcome {
        let error = pose.and_then(|p| self.judge(p.x, p.y, p.theta));
        Outcome {
            ms,
            first_ms,
            correct: error.is_some(),
            error,
            ..Outcome::default()
        }
    }

    fn local(&self, (ms, first_ms, view): (f64, f64, Option<View>)) -> Outcome {
        let Some(view) = view else {
            return Outcome {
                ms,
                first_ms,
                ..Outcome::default()
            };
        };
        let [cx, cy] = self.centre();
        let point_errors = view
            .points
            .iter()
            .map(|p| {
                let (x, y) = into_pattern_frame(p.pixel[0], p.pixel[1], cx, cy, self.pose.theta);
                let (ex, ey) = self
                    .pattern
                    .wrap_offset(p.board[0] - (x - self.pose.x), p.board[1] - (y - self.pose.y));
                ex.hypot(ey)
            })
            .collect();
        let (board, [re, im]) = similarity_at(&view.points, self.centre());
        // The renderer turns by −θ.
        let error = self.judge(board[0], board[1], -im.atan2(re));
        Outcome {
            ms,
            first_ms,
            correct: error.is_some(),
            error,
            point_errors,
        }
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

/// One table row: correct frames, median ms (and of it the first stage),
/// median position and turn errors, then the local points' rms and gross
/// fraction.
fn row(outcomes: &[Outcome]) -> [String; 7] {
    let ms = median(outcomes.iter().map(|o| o.ms).collect());
    let first = median(outcomes.iter().map(|o| o.first_ms).collect());
    let position = median(outcomes.iter().filter_map(|o| o.error.map(|e| e.0)).collect());
    let turn = median(outcomes.iter().filter_map(|o| o.error.map(|e| e.1)).collect());
    let errors: Vec<f64> = outcomes.iter().flat_map(|o| o.point_errors.clone()).collect();
    let fine: Vec<f64> = errors.iter().copied().filter(|&e| e <= GROSS).collect();
    let (rms, gross) = if errors.is_empty() {
        ("–".into(), "–".into())
    } else {
        (
            format!("{:.4}", (fine.iter().map(|e| e * e).sum::<f64>() / fine.len() as f64).sqrt()),
            format!("{:.1}%", 100.0 * (errors.len() - fine.len()) as f64 / errors.len() as f64),
        )
    };
    let or_dash = |v: f64, digits: usize| {
        if v.is_nan() { "–".into() } else { format!("{v:.digits$}") }
    };
    [
        format!("{}/{}", outcomes.iter().filter(|o| o.correct).count(), outcomes.len()),
        format!("{ms:.1}"),
        format!("{first:.1}"),
        or_dash(position, 4),
        or_dash(turn, 4),
        rms,
        gross,
    ]
}

const METHODS: [&str; 4] = ["flat gpu", "flat cpu", "local gpu", "local cpu"];
const HEADER: &str =
    "| method | correct | ms | of which detect/search | position px | turn ° | points rms px | points > 1 px |";

fn print_rows(csv: &mut std::fs::File, set: &str, level: &str, results: &[Vec<Outcome>; 4]) {
    println!("{HEADER}");
    println!("|---|---|---|---|---|---|---|---|");
    for (method, outcomes) in METHODS.iter().zip(results) {
        let cells = row(outcomes);
        println!("| {method} | {} |", cells.join(" | "));
        writeln!(csv, "{set},{level},{method},{}", cells.join(",")).unwrap();
    }
}

fn square_on(csv: &mut std::fs::File, gpu: &GpuBackend, cpu: &CpuBackend) {
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
    for (square, width, height) in SQUARE_ON {
        let pattern = Checkerboard::new(square, ORDER).expect("order");
        let target = Target::new(square, ORDER);
        let renders: Vec<Vec<f32>> = poses
            .iter()
            .map(|pose| pattern.render(width, height, pose).as_slice().to_vec())
            .collect();
        for sigma in SIGMAS {
            let mut results: [Vec<Outcome>; 4] = Default::default();
            for (index, pose) in poses.iter().enumerate() {
                let image = degrade(&renders[index], width, height, sigma, NOISE, index as u64 + 1);
                let truth = SquareOn {
                    pattern: &pattern,
                    square,
                    pose,
                    width,
                    height,
                };
                results[0].push(truth.flat(flat(gpu, &image, width, height, square)));
                results[1].push(truth.flat(flat(cpu, &image, width, height, square)));
                results[2].push(truth.local(local(gpu, &image, width, height, &target)));
                results[3].push(truth.local(local(&CpuDemodulator, &image, width, height, &target)));
            }
            println!(
                "\n### square on, {width}×{height}, {square} px squares, defocus σ {sigma} px, noise {NOISE}, {POSES} poses\n"
            );
            print_rows(csv, &format!("square-on {width}x{height}"), &format!("sigma {sigma}"), &results);
        }
    }
}

fn tilted(csv: &mut std::fs::File, gpu: &GpuBackend, cpu: &CpuBackend) {
    const WIDTH: usize = 1280;
    const HEIGHT: usize = 720;
    let mut camera = Camera::ideal(Model::Pinhole, WIDTH, HEIGHT, 1000.0, 1002.0, 641.0, 357.0);
    camera.distortion = vec![-0.12, 0.05, 0.0004, -0.0003, 0.0];
    let target = Target::new(5.0, ORDER);
    // Square side in pixels at 350 mm, for the global method's code.
    let square_px = 5.0 * 1000.0 / 350.0;
    // Tilts from none to about 30°, as robustness.rs poses them.
    let poses: Vec<(&str, RigidPose)> = [
        ("0°", [0.0, 0.0, 0.3], [0.0, 0.0, 1.0]),
        ("~15°", [0.2, -0.15, 1.0], [5.0, -4.0, 1.0]),
        ("~30°", [0.45, 0.1, -0.4], [12.0, -8.0, 0.95]),
        ("~30°", [-0.4, 0.25, 1.2], [-15.0, 6.0, 1.05]),
        ("~30°", [0.15, -0.5, 2.0], [8.0, 10.0, 0.9]),
        ("~30°", [0.35, 0.4, 2.8], [10.0, 4.0, 1.0]),
    ]
    .iter()
    .map(|(name, r, t)| {
        (
            *name,
            RigidPose::from_vectors(
                Vector3::new(r[0], r[1], r[2]),
                Vector3::new(t[0], t[1], t[2] * 350.0),
            ),
        )
    })
    .collect();
    println!("\n### tilted through a 720p camera, 5 mm squares at ~350 mm (~14 px), noise {NOISE}\n");
    println!("Global: correct = the code read (no truth to judge one similarity against).\n");
    let mut all: [Vec<Outcome>; 4] = Default::default();
    for (index, (name, pose)) in poses.iter().enumerate() {
        let clean = Scene {
            camera: &camera,
            target: &target,
            half_size: [200.0, 160.0],
            background: 0.45,
            supersample: 3,
        }
        .render(pose);
        let image = degrade(&clean, WIDTH, HEIGHT, 0.0, NOISE, index as u64 + 100);
        let judge_flat = |(ms, first_ms, found): (f64, f64, Option<PatternPose>)| Outcome {
            ms,
            first_ms,
            correct: found.is_some(),
            ..Outcome::default()
        };
        let judge_local = |(ms, first_ms, view): (f64, f64, Option<View>)| {
            let point_errors: Vec<f64> = view
                .iter()
                .flat_map(|v| &v.points)
                .map(|p| {
                    let q = camera.project(&pose.apply(p.board)).expect("in view");
                    (q[0] - p.pixel[0]).hypot(q[1] - p.pixel[1])
                })
                .collect();
            Outcome {
                ms,
                first_ms,
                correct: view.is_some(),
                point_errors,
                ..Outcome::default()
            }
        };
        let frame = [
            judge_flat(flat(gpu, &image, WIDTH, HEIGHT, square_px)),
            judge_flat(flat(cpu, &image, WIDTH, HEIGHT, square_px)),
            judge_local(local(gpu, &image, WIDTH, HEIGHT, &target)),
            judge_local(local(&CpuDemodulator, &image, WIDTH, HEIGHT, &target)),
        ];
        print!("pose {index} ({name}):");
        for (method, o) in METHODS.iter().zip(&frame) {
            print!(" {method} {:.1} ms{};", o.ms, if o.correct { "" } else { " ✗" });
        }
        println!();
        for (list, o) in all.iter_mut().zip(frame) {
            list.push(o);
        }
    }
    println!();
    print_rows(csv, "tilted 1280x720", "all", &all);
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        std::env::temp_dir()
            .join("vernier-demod-bench.csv")
            .display()
            .to_string()
    });
    let mut csv = std::fs::File::create(&path).expect("csv");
    writeln!(
        csv,
        "set,level,method,correct,median_ms,median_first_stage_ms,median_position_px,median_turn_deg,points_rms_px,points_gross"
    )
    .unwrap();

    let gpu = GpuBackend::new();
    let cpu = CpuBackend::new();
    println!(
        "fastest of {RUNS} runs per frame, medians over frames; GPU: {}",
        ComputeBackend::name(&gpu)
    );
    square_on(&mut csv, &gpu, &cpu);
    tilted(&mut csv, &gpu, &cpu);
    println!("\nCSV: {path}");
}
