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
mod pgm;

use args::{Command, TopLevel};
use backend_select::{BackendKind, dispatch};
use commands::benchmark::Benchmark;
use commands::calibrate;
use commands::checkerboard_figures;
use commands::detect_megarena::DetectMegarena;
use commands::render_checkerboard;
use commands::render_megarena;
use commands::roundtrip_megarena::RoundtripMegarena;
use commands::{solve_pnp, undistort, webcam};
use std::path::PathBuf;

/// Ends the program with the error, if any, the way the camera commands
/// report failure. They are called inside an immediately-run closure so that
/// `?` can be used while building their arguments.
fn exit_on_error(result: Result<(), String>) {
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
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
                target: calibrate::target(a.square, a.code_size, a.diamonds)?,
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
                format: a.format.clone(),
                video_size: a.video_size.clone(),
                views: a.views,
                interval: std::time::Duration::from_secs_f64(a.interval),
                target: calibrate::target(a.square, a.code_size, a.diamonds)?,
                model: calibrate::model(&a.model)?,
                output: PathBuf::from(&a.output),
                save_frames: a.save_frames.as_ref().map(PathBuf::from),
            })
        })()),
        Command::SolvePnp(a) => exit_on_error((|| {
            solve_pnp::run(&solve_pnp::SolvePnpArgs {
                camera: PathBuf::from(&a.camera),
                images: a.images.iter().map(PathBuf::from).collect(),
                target: calibrate::target(a.square, a.code_size, a.diamonds)?,
            })
        })()),
        Command::Undistort(a) => exit_on_error(undistort::run(&undistort::UndistortArgs {
            camera: PathBuf::from(&a.camera),
            image: PathBuf::from(&a.image),
            output: PathBuf::from(&a.output),
            zoom: a.zoom,
        })),
        Command::DetectMegarena(a) => {
            let Some(kind) = BackendKind::parse(&a.backend) else {
                eprintln!("unknown backend '{}'. try: cpu", a.backend);
                std::process::exit(2);
            };
            let task = DetectMegarena {
                image_path: std::path::PathBuf::from(&a.image),
                physical_period: a.period,
                code_size: a.code_size,
                sigma: a.sigma,
                min_frequency: a.min_frequency,
                max_frequency: a.max_frequency,
                smoothing_sigma: a.smoothing_sigma,
                debug_image: a.debug_image.as_ref().map(std::path::PathBuf::from),
                verbose: a.verbose,
            };
            let report = dispatch(kind, &task);
            println!(
                "Estimated pose: x={:.4} µm, y={:.4} µm, θ={:.6} rad (quadrant k3={})",
                report.x, report.y, report.theta, report.k3
            );
        }
        Command::RenderCheckerboard(a) => {
            let args = render_checkerboard::RenderCheckerboardArgs {
                width: a.width,
                height: a.height,
                x: a.x,
                y: a.y,
                theta: a.theta,
                square_px: a.square,
                code_size: a.code_size,
                plain: a.plain,
                diamonds: a.diamonds,
                corner_radius: a.corner_radius,
                output: std::path::PathBuf::from(&a.output),
            };
            if let Err(e) = render_checkerboard::run(&args) {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
        Command::CheckerboardFigures(a) => {
            let args = checkerboard_figures::CheckerboardFiguresArgs {
                out_dir: std::path::PathBuf::from(&a.out_dir),
                square_px: a.square,
                code_size: a.code_size,
                size: a.size,
                poses: a.poses,
            };
            if let Err(e) = checkerboard_figures::run(&args) {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
        Command::RenderMegarena(a) => {
            let args = render_megarena::RenderMegarenaArgs {
                width: a.width,
                height: a.height,
                x: a.x,
                y: a.y,
                theta: a.theta,
                period_px: a.period,
                code_size: a.code_size,
                output: std::path::PathBuf::from(&a.output),
            };
            if let Err(e) = render_megarena::run(&args) {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
        Command::RoundtripMegarena(a) => {
            let Some(kind) = BackendKind::parse(&a.backend) else {
                eprintln!(
                    "unknown backend '{}'. try: {}",
                    a.backend,
                    BackendKind::hint()
                );
                std::process::exit(2);
            };
            let task = RoundtripMegarena {
                width: a.width,
                height: a.height,
                true_x: a.x as f32,
                true_y: a.y as f32,
                true_theta: a.theta as f32,
                period_px: a.period as f32,
                code_size: a.code_size,
                sigma: a.sigma as f32,
                min_frequency: a.min_frequency,
                max_frequency: a.max_frequency,
                smoothing_sigma: a.smoothing_sigma as f32,
                render_gpu: a.render_gpu,
                pixel_size: a.pixel_size as f32,
            };
            let r = dispatch(kind, &task);
            let swap_label = if r.swapped { "yes" } else { "no" };
            // Convert everything out of pixels using the camera pixel size.
            let um_per_px = a.pixel_size;
            let nm_per_px = a.pixel_size * 1000.0;
            println!(
                "backend={}  renderer={}  size={}x{}  period={:.3}µm  code={}  swapped={}",
                r.backend,
                r.renderer,
                a.width,
                a.height,
                a.period * um_per_px,
                a.code_size,
                swap_label
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
        }
    }
}
