//! How the phase measurement holds up when the picture is spoiled.
//!
//! The board is rendered through a known 720p camera at a few poses, then
//! degraded one way at a time: white noise, salt and pepper, smooth additive
//! fields slower and about as fast as the carrier's window, uneven lighting,
//! vignetting, defocus, JPEG. Each frame is measured as `track` measures it,
//! and every point is compared with the true projection of its board point.
//!
//! ```text
//! cargo run --release -p vernier-camera --example robustness [-- results.csv]
//! ```
//!
//! About three minutes on 12 cores. The table goes to stdout, a CSV of it to
//! the path given or the temporary directory.

use std::io::Write;
use std::time::Instant;

use nalgebra::Vector3;
use vernier_camera::*;

const WIDTH: usize = 1280;
const HEIGHT: usize = 720;
/// Frames per pose and level, each with its own noise.
const SEEDS: u64 = 2;
/// A point further than this from the truth, in pixels, is counted as wrong
/// rather than imprecise.
const GROSS: f64 = 1.0;

/// The camera the board is rendered through and solved with.
fn camera() -> Camera {
    let mut camera = Camera::ideal(Model::Pinhole, WIDTH, HEIGHT, 1000.0, 1002.0, 641.0, 357.0);
    camera.distortion = vec![-0.12, 0.05, 0.0004, -0.0003, 0.0];
    camera
}

/// 5 mm squares, code size 8.
fn target() -> Target {
    Target::new(5.0, 8)
}

/// Poses that tilt the board up to about 30° every which way, 350 mm off:
/// squares of 13 to 16 px, a carrier period of about 20 px.
fn poses() -> Vec<RigidPose> {
    [
        ([0.0, 0.0, 0.3], [0.0, 0.0, 1.0]),
        ([0.45, 0.1, -0.4], [12.0, -8.0, 0.95]),
        ([-0.4, 0.25, 1.2], [-15.0, 6.0, 1.05]),
        ([0.15, -0.5, 2.0], [8.0, 10.0, 0.9]),
        ([-0.3, -0.35, -1.1], [-6.0, -12.0, 1.1]),
        ([0.35, 0.4, 2.8], [10.0, 4.0, 1.0]),
    ]
    .iter()
    .map(|(r, t)| {
        RigidPose::from_vectors(
            Vector3::new(r[0], r[1], r[2]),
            Vector3::new(t[0], t[1], t[2] * 350.0),
        )
    })
    .collect()
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

/// A smooth random field of unit standard deviation: plane waves in random
/// directions, wavelengths log-uniform between `shortest` and `longest`
/// pixels.
fn smooth_field(rng: &mut Rng, shortest: f64, longest: f64) -> Vec<f32> {
    const WAVES: usize = 24;
    // Each wave as its wave vector (kx, ky) and phase.
    let waves: Vec<(f64, f64, f64)> = (0..WAVES)
        .map(|_| {
            let length = shortest * (longest / shortest).powf(rng.uniform());
            let angle = std::f64::consts::TAU * rng.uniform();
            let wavenumber = std::f64::consts::TAU / length;
            (
                wavenumber * angle.cos(),
                wavenumber * angle.sin(),
                std::f64::consts::TAU * rng.uniform(),
            )
        })
        .collect();
    // A unit cosine of random phase has variance 1/2.
    let scale = (2.0 / WAVES as f64).sqrt();
    (0..WIDTH * HEIGHT)
        .map(|i| {
            let (x, y) = ((i % WIDTH) as f64, (i / WIDTH) as f64);
            let sum: f64 = waves
                .iter()
                .map(|&(kx, ky, phase)| (kx * x + ky * y + phase).cos())
                .sum();
            (scale * sum) as f32
        })
        .collect()
}

/// The frame blurred by a Gaussian of `sigma` px, as two one-dimensional passes
/// cut at 3 sigma. Near the edges the kernel is renormalised over the pixels
/// inside the frame.
fn gaussian_blur(image: &[f32], sigma: f64) -> Vec<f32> {
    let radius = (3.0 * sigma).ceil() as isize;
    let kernel: Vec<f64> = (-radius..=radius)
        .map(|offset| (-0.5 * (offset as f64 / sigma).powi(2)).exp())
        .collect();
    let pass = |src: &[f32], horizontal: bool| -> Vec<f32> {
        (0..WIDTH * HEIGHT)
            .map(|i| {
                let (x, y) = ((i % WIDTH) as isize, (i / WIDTH) as isize);
                let (mut sum, mut weight) = (0.0, 0.0);
                for (tap, &k) in kernel.iter().enumerate() {
                    let offset = tap as isize - radius;
                    let (sx, sy) = if horizontal {
                        (x + offset, y)
                    } else {
                        (x, y + offset)
                    };
                    if sx < 0 || sy < 0 || sx >= WIDTH as isize || sy >= HEIGHT as isize {
                        continue;
                    }
                    sum += k * src[sy as usize * WIDTH + sx as usize] as f64;
                    weight += k;
                }
                (sum / weight) as f32
            })
            .collect()
    };
    pass(&pass(image, true), false)
}

/// The frame after a round trip through 8-bit JPEG at `quality`.
fn jpeg(image: &[f32], quality: u8) -> Vec<f32> {
    let bytes: Vec<u8> = image
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    let mut out = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, quality)
        .encode(
            &bytes,
            WIDTH as u32,
            HEIGHT as u32,
            image::ExtendedColorType::L8,
        )
        .unwrap();
    image::load_from_memory(&out)
        .unwrap()
        .to_luma8()
        .into_raw()
        .iter()
        .map(|&v| v as f32 / 255.0)
        .collect()
}

/// One way of spoiling the picture. Each takes a level, whose meaning is
/// given variant by variant.
#[derive(Clone, Copy)]
enum Kind {
    Clean,
    /// Standard deviation, intensities in `0..=1` (black squares 0.1, white 0.9).
    WhiteNoise,
    /// Fraction of pixels set to 0 or 1.
    SaltPepper,
    /// Additive smooth field, wavelengths 8 to 30 carrier periods; level is
    /// its standard deviation.
    SlowField,
    /// Additive field with wavelengths 2 to 5 carrier periods, about the
    /// demodulation window's size.
    WindowField,
    /// Lighting: the image times `1 + level·field`, the field as slow as above.
    Shading,
    /// Gain falling to `1 − level` in the corners.
    Vignetting,
    /// Gaussian defocus, sigma in pixels.
    Defocus,
    /// JPEG quality.
    Jpeg,
}

impl Kind {
    /// Every kind, in table order.
    const ALL: [Kind; 9] = [
        Kind::Clean,
        Kind::WhiteNoise,
        Kind::SaltPepper,
        Kind::SlowField,
        Kind::WindowField,
        Kind::Shading,
        Kind::Vignetting,
        Kind::Defocus,
        Kind::Jpeg,
    ];

    /// Table label, which also names the level.
    fn name(self) -> &'static str {
        match self {
            Kind::Clean => "clean",
            Kind::WhiteNoise => "white noise σ",
            Kind::SaltPepper => "salt & pepper fraction",
            Kind::SlowField => "slow additive field σ",
            Kind::WindowField => "window-scale additive field σ",
            Kind::Shading => "uneven lighting (gain ±)",
            Kind::Vignetting => "vignetting (corner loss)",
            Kind::Defocus => "defocus σ px",
            Kind::Jpeg => "JPEG quality",
        }
    }

    /// The levels tried, mildest first.
    fn levels(self) -> &'static [f64] {
        match self {
            Kind::Clean => &[0.0],
            Kind::WhiteNoise => &[0.02, 0.05, 0.1, 0.2, 0.3],
            Kind::SaltPepper => &[0.01, 0.05, 0.1, 0.2, 0.3, 0.4],
            Kind::SlowField => &[0.05, 0.1, 0.2, 0.3, 0.5],
            Kind::WindowField => &[0.02, 0.05, 0.1, 0.2, 0.3],
            Kind::Shading => &[0.2, 0.4, 0.6, 0.8, 0.95],
            Kind::Vignetting => &[0.3, 0.6, 0.8, 0.95],
            Kind::Defocus => &[1.0, 2.0, 3.0, 4.0, 5.0],
            Kind::Jpeg => &[90.0, 70.0, 50.0, 30.0, 15.0],
        }
    }

    /// Level as printed in the table.
    fn level_text(self, level: f64) -> String {
        match self {
            Kind::Clean => "–".to_string(),
            Kind::Jpeg => format!("{level:.0}"),
            _ => format!("{level}"),
        }
    }

    /// The clean frame spoiled at `level`; `period` is the carrier's, px,
    /// which the smooth fields are scaled to.
    fn apply(self, level: f64, clean: &[f32], rng: &mut Rng, period: f64) -> Vec<f32> {
        let mut image = match self {
            Kind::Clean => clean.to_vec(),
            Kind::WhiteNoise => clean
                .iter()
                .map(|&v| v + (level * rng.normal()) as f32)
                .collect(),
            Kind::SaltPepper => clean
                .iter()
                .map(|&v| {
                    if rng.uniform() < level {
                        if rng.uniform() < 0.5 { 0.0 } else { 1.0 }
                    } else {
                        v
                    }
                })
                .collect(),
            Kind::SlowField | Kind::WindowField => {
                let (shortest, longest) = match self {
                    Kind::SlowField => (8.0 * period, 30.0 * period),
                    _ => (2.0 * period, 5.0 * period),
                };
                let field = smooth_field(rng, shortest, longest);
                clean
                    .iter()
                    .zip(&field)
                    .map(|(&v, &f)| v + level as f32 * f)
                    .collect()
            }
            Kind::Shading => {
                let field = smooth_field(rng, 8.0 * period, 30.0 * period);
                clean
                    .iter()
                    .zip(&field)
                    // The field is unit deviation; squash it into ±1.
                    .map(|(&v, &f)| v * (1.0 + level as f32 * (f / 1.5).tanh()).max(0.0))
                    .collect()
            }
            Kind::Vignetting => {
                let (cx, cy) = ((WIDTH - 1) as f64 / 2.0, (HEIGHT - 1) as f64 / 2.0);
                let corner = cx.hypot(cy);
                clean
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        let (x, y) = ((i % WIDTH) as f64, (i / WIDTH) as f64);
                        let radius = (x - cx).hypot(y - cy) / corner;
                        v * (1.0 - level * radius * radius) as f32
                    })
                    .collect()
            }
            Kind::Defocus => gaussian_blur(clean, level),
            Kind::Jpeg => jpeg(clean, level as u8),
        };
        // A sensor clips.
        image.iter_mut().for_each(|v| *v = v.clamp(0.0, 1.0));
        image
    }
}

/// What one frame gave.
#[derive(Default)]
struct Outcome {
    /// The code read, so the points are in the board's frame.
    absolute: bool,
    /// The board was found at all.
    measured: bool,
    points: usize,
    /// Distance of each point from the true projection of its board point, px;
    /// only when the code read.
    errors: Vec<f64>,
    /// PnP with the true camera against the true pose: mm and degrees.
    pose_error: Option<(f64, f64)>,
    /// Time taken to measure, ms.
    ms: f64,
}

/// Measures one frame as `track` does and compares it with the truth.
fn measure(camera: &Camera, pose: &RigidPose, image: &[f32]) -> Outcome {
    let start = Instant::now();
    let measured = measure_view(image, WIDTH, HEIGHT, &target());
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    let Ok(view) = measured else {
        return Outcome {
            ms,
            ..Outcome::default()
        };
    };
    let mut outcome = Outcome {
        measured: true,
        absolute: view.is_absolute(),
        points: view.points.len(),
        ms,
        ..Outcome::default()
    };
    if !outcome.absolute {
        return outcome;
    }
    outcome.errors = view
        .points
        .iter()
        .map(|p| {
            let q = camera.project(&pose.apply(p.board)).expect("in view");
            (q[0] - p.pixel[0]).hypot(q[1] - p.pixel[1])
        })
        .collect();
    if let Ok(fit) = solve_pnp(camera, &view) {
        outcome.pose_error = Some((
            (fit.pose.translation - pose.translation).norm(),
            fit.pose.angle_to(pose).to_degrees(),
        ));
    }
    outcome
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

/// The `q` quantile of sorted values, nearest rank.
fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

/// The figures of one table row, over all frames of one kind and level.
struct Summary {
    /// Frames whose code read.
    read: usize,
    /// Frames measured but whose code did not read.
    relative: usize,
    /// Points per frame.
    points: f64,
    /// Rms of the point errors up to `GROSS`, px.
    rms_fine: f64,
    /// 95th percentile of the point errors up to `GROSS`, px.
    p95_fine: f64,
    /// Fraction of points further than `GROSS`.
    gross: f64,
    /// Median pose errors: translation mm, rotation degrees.
    pose_mm: f64,
    pose_deg: f64,
    /// Mean measuring time, ms.
    ms: f64,
}

fn summarize(outcomes: &[Outcome]) -> Summary {
    let frames = outcomes.len() as f64;
    let mut errors: Vec<f64> = outcomes.iter().flat_map(|o| o.errors.clone()).collect();
    errors.sort_by(|a, b| a.total_cmp(b));
    // Precision of the points that are not plain wrong.
    let fine: Vec<f64> = errors.iter().copied().filter(|&e| e <= GROSS).collect();
    let mut moved: Vec<f64> = outcomes
        .iter()
        .filter_map(|o| o.pose_error.map(|(mm, _)| mm))
        .collect();
    let mut turned: Vec<f64> = outcomes
        .iter()
        .filter_map(|o| o.pose_error.map(|(_, deg)| deg))
        .collect();
    Summary {
        read: outcomes.iter().filter(|o| o.absolute).count(),
        relative: outcomes
            .iter()
            .filter(|o| o.measured && !o.absolute)
            .count(),
        points: outcomes.iter().map(|o| o.points).sum::<usize>() as f64 / frames,
        rms_fine: (fine.iter().map(|e| e * e).sum::<f64>() / fine.len() as f64).sqrt(),
        p95_fine: quantile(&fine, 0.95),
        gross: (errors.len() - fine.len()) as f64 / errors.len().max(1) as f64,
        pose_mm: median(&mut moved),
        pose_deg: median(&mut turned),
        ms: outcomes.iter().map(|o| o.ms).sum::<f64>() / frames,
    }
}

/// The board through the camera at each pose, unspoiled.
fn render_clean(camera: &Camera, target: &Target, poses: &[RigidPose]) -> Vec<Vec<f32>> {
    poses
        .iter()
        .map(|pose| {
            Scene {
                camera,
                target,
                half_size: [200.0, 160.0],
                background: 0.45,
                supersample: 3,
            }
            .render(pose)
        })
        .collect()
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        std::env::temp_dir()
            .join("vernier-robustness.csv")
            .display()
            .to_string()
    });
    let camera = camera();
    let target = target();
    let poses = poses();
    let clean = render_clean(&camera, &target, &poses);
    let period = measure_view(&clean[0], WIDTH, HEIGHT, &target)
        .expect("the clean board measures")
        .period;
    let frames = poses.len() * SEEDS as usize;
    println!(
        "{WIDTH}x{HEIGHT}, carrier period {period:.1} px, {} poses × {SEEDS} seeds = {frames} frames per level",
        poses.len()
    );
    println!("errors are against the true projection, over the points of frames whose code read\n");
    println!(
        "| degradation | level | code read | measured, no code | points | rms px | p95 px | > {GROSS} px | pose mm | pose ° | ms |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|");

    let mut csv = std::fs::File::create(&path).expect("csv");
    writeln!(
        csv,
        "degradation,level,frames,code_read,measured_no_code,points_mean,rms_px,p95_px,gross_fraction,pose_mm_median,pose_deg_median,ms_mean"
    )
    .unwrap();

    for kind in Kind::ALL {
        for &level in kind.levels() {
            let mut outcomes = Vec::new();
            for (pose_index, pose) in poses.iter().enumerate() {
                for seed in 0..SEEDS {
                    let mut rng = Rng(1000 * pose_index as u64 + seed + 1);
                    let image = kind.apply(level, &clean[pose_index], &mut rng, period);
                    outcomes.push(measure(&camera, pose, &image));
                }
            }
            let Summary {
                read,
                relative,
                points,
                rms_fine,
                p95_fine,
                gross,
                pose_mm,
                pose_deg,
                ms,
            } = summarize(&outcomes);
            println!(
                "| {} | {} | {read}/{frames} | {relative} | {points:.0} | {rms_fine:.3} | {p95_fine:.3} | {:.2}% | {pose_mm:.3} | {pose_deg:.4} | {ms:.0} |",
                kind.name(),
                kind.level_text(level),
                100.0 * gross,
            );
            writeln!(
                csv,
                "{},{level},{frames},{read},{relative},{points:.1},{rms_fine:.5},{p95_fine:.5},{gross:.6},{pose_mm:.5},{pose_deg:.6},{ms:.1}",
                kind.name(),
            )
            .unwrap();
            std::io::stdout().flush().unwrap();
        }
    }
    eprintln!("\nwrote {path}");
}
