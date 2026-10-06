//! `vernier` — the CLI front-end and benchmark harness for vernier-rs.
//!
//! This binary is the one place that names concrete backends. It parses args,
//! resolves a [`BackendKind`](backend_select::BackendKind), and
//! [`dispatch`](backend_select::dispatch)es a backend-generic
//! [`BackendTask`](backend_select::BackendTask). Swapping CPU for GPU is the
//! `--backend` flag; nothing in the library changes.

mod annotate;
mod args;
mod backend_select;
mod commands;
mod imageio;
mod pattern;
mod pgm;

use args::{Command, PatternArgs, TopLevel};
use backend_select::{BackendKind, dispatch};
use commands::benchmark::Benchmark;
use commands::calibrate;
use commands::checkerboard_figures;
use commands::detect_megarena::DetectMegarena;
use commands::render_pattern;
use commands::roundtrip_megarena::RoundtripMegarena;
use commands::{make_pattern, phone, solve_pnp, track, undistort, webcam};
use pattern::{Image, Layout, Packing, PatternFile};
use std::path::{Path, PathBuf};

/// Ends the program with the error, if any, the way the camera commands
/// report failure. They are called inside an immediately-run closure so that
/// `?` can be used while building their arguments.
fn exit_on_error(result: Result<(), String>) {
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// The backend named on the command line, for the camera commands.
fn backend(name: &str) -> Result<BackendKind, String> {
    BackendKind::parse(name)
        .ok_or_else(|| format!("unknown backend '{name}'; try {}", BackendKind::hint()))
}

fn main() {
    let top: TopLevel = argh::from_env();

    match top.command {
        Command::Bench(a) => {
            let Some(kind) = BackendKind::parse(&a.backend) else {
                eprintln!(
                    "unknown backend '{}'. try: {}",
                    a.backend,
                    BackendKind::hint()
                );
                std::process::exit(2);
            };
            let task = Benchmark {
                size: a.size,
                iterations: a.iters,
                sigma: a.sigma,
                min_frequency: a.min_frequency,
                max_frequency: a.max_frequency,
                smoothing_sigma: a.smoothing_sigma,
            };
            let r = dispatch(kind, &task);
            println!(
                "backend={} size={}x{} iters={} mean={:.2}ms best={:.2}ms",
                r.backend, r.size, r.size, r.iterations, r.mean_ms, r.best_ms
            );
        }
        Command::Calibrate(a) => exit_on_error((|| {
            calibrate::run(&calibrate::CalibrateArgs {
                images: a.images.iter().map(PathBuf::from).collect(),
                target: pattern::target(&a.pattern)?,
                model: calibrate::model(&a.model)?,
                output: PathBuf::from(&a.output),
            })
        })()),
        Command::CalibrateWebcam(a) => exit_on_error((|| {
            if !(a.interval >= 0.0 && a.interval.is_finite()) {
                return Err(format!(
                    "interval {} must be a non-negative number of seconds",
                    a.interval
                ));
            }
            webcam::run(&webcam::WebcamArgs {
                device: a.device.clone(),
                options: webcam::CaptureOptions {
                    format: a.format.clone(),
                    input_format: a.input_format.clone(),
                    video_size: a.video_size.clone(),
                    framerate: a.framerate.clone(),
                },
                views: a.views,
                interval: std::time::Duration::from_secs_f64(a.interval),
                target: pattern::target(&a.pattern)?,
                model: calibrate::model(&a.model)?,
                output: PathBuf::from(&a.output),
                save_frames: a.save_frames.as_ref().map(PathBuf::from),
                backend: backend(&a.backend)?,
            })
        })()),
        Command::Phone(a) => exit_on_error((|| {
            phone::run(&phone::PhoneArgs {
                target: pattern::target(&a.pattern)?,
                camera: PathBuf::from(&a.camera),
                views: a.views.max(2),
                model: calibrate::model(&a.model)?,
                port: a.port,
                csv: a.csv.as_ref().map(PathBuf::from),
                backend: backend(&a.backend)?,
            })
        })()),
        Command::SolvePnp(a) => exit_on_error((|| {
            solve_pnp::run(&solve_pnp::SolvePnpArgs {
                camera: PathBuf::from(&a.camera),
                images: a.images.iter().map(PathBuf::from).collect(),
                target: pattern::target(&a.pattern)?,
            })
        })()),
        Command::Track(a) => exit_on_error((|| {
            let camera = PathBuf::from(&a.camera);
            let video_size = match &a.video_size {
                Some(size) => size.clone(),
                None => {
                    let c = calibrate::load_camera(&camera)?;
                    format!("{}x{}", c.width, c.height)
                }
            };
            track::run(&track::TrackArgs {
                camera,
                target: pattern::target(&a.pattern)?,
                device: a.device.clone(),
                options: webcam::CaptureOptions {
                    format: a.format.clone(),
                    input_format: a.input_format.clone(),
                    video_size: Some(video_size),
                    framerate: a.framerate.clone(),
                },
                port: a.port,
                csv: a.csv.as_ref().map(PathBuf::from),
                backend: backend(&a.backend)?,
            })
        })()),
        Command::Undistort(a) => exit_on_error(undistort::run(&undistort::UndistortArgs {
            camera: PathBuf::from(&a.camera),
            image: PathBuf::from(&a.image),
            output: PathBuf::from(&a.output),
            zoom: a.zoom,
        })),
        Command::DetectMegarena(a) => exit_on_error((|| {
            let kind = backend(&a.backend)?;
            let (pitch, code_size, _) = pattern::megarena(&a.pattern)?;
            let task = DetectMegarena {
                image_path: PathBuf::from(&a.image),
                physical_period: pitch as f32,
                code_size,
                sigma: a.sigma,
                min_frequency: a.min_frequency,
                max_frequency: a.max_frequency,
                smoothing_sigma: a.smoothing_sigma,
                debug_image: a.debug_image.as_ref().map(PathBuf::from),
                verbose: a.verbose,
            };
            let report = dispatch(kind, &task);
            println!(
                "Estimated pose: x={:.4} µm, y={:.4} µm, θ={:.6} rad (quadrant k3={})",
                report.x, report.y, report.theta, report.k3
            );
            Ok(())
        })()),
        Command::RenderPattern(a) => exit_on_error((|| {
            render_pattern::run(&render_pattern::RenderPatternArgs {
                pattern: PatternFile::load(Path::new(&a.pattern))?,
                x: a.x,
                y: a.y,
                theta: a.theta,
                output: PathBuf::from(&a.output),
            })
        })()),
        Command::CheckerboardFigures(a) => exit_on_error((|| {
            let file = PatternFile::load(Path::new(&a.pattern))?;
            let PatternFile::Checkerboard { image, .. } = file else {
                return Err(format!(
                    "{}: this command needs a checkerboard pattern",
                    a.pattern
                ));
            };
            if image.width != image.height {
                return Err(format!(
                    "{}: the figures are square, but the image is {}x{}",
                    a.pattern, image.width, image.height
                ));
            }
            checkerboard_figures::run(&checkerboard_figures::CheckerboardFiguresArgs {
                out_dir: PathBuf::from(&a.out_dir),
                square_px: image.square,
                code_size: file.code_size(),
                size: image.width,
                poses: a.poses,
            })
        })()),
        Command::MakePattern(a) => {
            let (file, output) = match a.pattern {
                PatternArgs::Checkerboard(c) => (
                    PatternFile::Checkerboard {
                        square: c.square,
                        code_size: c.code_size,
                        layout: if c.diamonds {
                            Layout::Diamonds
                        } else {
                            Layout::Squares
                        },
                        packing: if c.two_bits {
                            Packing::TwoBits
                        } else {
                            Packing::OneBit
                        },
                        corner_radius: c.corner_radius,
                        plain: c.plain,
                        image: Image {
                            width: c.width,
                            height: c.height,
                            square: c.square_px,
                        },
                    },
                    c.output,
                ),
                PatternArgs::Megarena(m) => (
                    PatternFile::Megarena {
                        pitch: m.pitch,
                        code_size: m.code_size,
                        image: Image {
                            width: m.width,
                            height: m.height,
                            square: m.square_px,
                        },
                    },
                    m.output,
                ),
            };
            let output = output.unwrap_or_else(|| file.default_name());
            exit_on_error(make_pattern::run(&file, &PathBuf::from(output)))
        }
        Command::RoundtripMegarena(a) => exit_on_error((|| {
            let kind = backend(&a.backend)?;
            let (pitch, code_size, image) = pattern::megarena(&a.pattern)?;
            // One rendered pixel is this long on the pattern.
            let pixel_size = pitch / image.square;
            let task = RoundtripMegarena {
                width: image.width,
                height: image.height,
                true_x: a.x as f32,
                true_y: a.y as f32,
                true_theta: a.theta as f32,
                period_px: image.square as f32,
                code_size,
                sigma: a.sigma as f32,
                min_frequency: a.min_frequency,
                max_frequency: a.max_frequency,
                smoothing_sigma: a.smoothing_sigma as f32,
                render_gpu: a.render_gpu,
                pixel_size: pixel_size as f32,
            };
            let r = dispatch(kind, &task);
            let swap_label = if r.swapped { "yes" } else { "no" };
            // Convert everything out of pixels using the camera pixel size.
            let um_per_px = pixel_size;
            let nm_per_px = pixel_size * 1000.0;
            println!(
                "backend={}  renderer={}  size={}x{}  period={:.3}µm  code={}  swapped={}",
                r.backend, r.renderer, image.width, image.height, pitch, code_size, swap_label
            );
            println!(
                "true:      x={:.4}µm  y={:.4}µm  θ={:.6} rad",
                r.true_x * um_per_px,
                r.true_y * um_per_px,
                r.true_theta
            );
            println!(
                "recovered: x={:.4}µm  y={:.4}µm  θ={:.6} rad",
                r.recovered_x * um_per_px,
                r.recovered_y * um_per_px,
                r.recovered_theta
            );
            println!(
                "error abs: Δx={:.1}nm  Δy={:.1}nm  Δθ={:.2e} rad",
                r.abs_error_x * nm_per_px,
                r.abs_error_y * nm_per_px,
                r.error_theta
            );
            // Sub-period precision margin, reported in nanometres.
            println!(
                "error fine: Δx={:.1}nm  Δy={:.1}nm",
                r.fine_error_x * nm_per_px,
                r.fine_error_y * nm_per_px
            );
            Ok(())
        })()),
    }
}
