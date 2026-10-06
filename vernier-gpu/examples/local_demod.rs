//! The local demodulation of `vernier-camera` on the CPU and on the GPU, side
//! by side on a tilted 1280×720 render: time per stage, and how far apart the
//! two put each point.
//!
//! ```text
//! cargo run --release -p vernier-gpu --example local_demod
//! ```

use std::time::Instant;

use nalgebra::Vector3;
use vernier_camera::*;
use vernier_core::LocalDemodulator;
use vernier_gpu::GpuBackend;

const WIDTH: usize = 1280;
const HEIGHT: usize = 720;
const RUNS: usize = 5;

fn main() {
    let mut camera = Camera::ideal(Model::Pinhole, WIDTH, HEIGHT, 1000.0, 1000.0, 641.3, 357.8);
    camera.distortion = vec![-0.12, 0.08, 0.0005, -0.0003, 0.0];
    let target = Target::new(5.0, 6);
    let pose = RigidPose::from_vectors(
        Vector3::new(0.35, -0.3, 0.4),
        Vector3::new(5.0, -3.0, 320.0),
    );
    let image = Scene {
        camera: &camera,
        target: &target,
        half_size: [140.0, 100.0],
        background: 0.45,
        supersample: 2,
    }
    .render(&pose);

    let gpu = GpuBackend::new();
    let cpu_view = run("cpu", &CpuDemodulator, &image, &target);
    let gpu_view = run("gpu", &gpu, &image, &target);

    // The same node on both sides lands at the same pixel; compare where on
    // the board each puts it.
    let gaps: Vec<f64> = cpu_view
        .points
        .iter()
        .filter_map(|p| {
            let q = gpu_view.points.iter().find(|q| q.pixel == p.pixel)?;
            Some((p.board[0] - q.board[0]).hypot(p.board[1] - q.board[1]))
        })
        .collect();
    let rms = (gaps.iter().map(|g| g * g).sum::<f64>() / gaps.len() as f64).sqrt();
    let worst = gaps.iter().copied().fold(0.0, f64::max);
    println!(
        "points: cpu {}, gpu {}, shared {}; board gap rms {rms:.2e} mm, worst {worst:.2e} mm",
        cpu_view.points.len(),
        gpu_view.points.len(),
        gaps.len(),
    );
}

/// Measures the frame `RUNS` times on `demodulator`, printing the stage times
/// of the fastest run and the time of the demodulated field.
fn run<D: LocalDemodulator>(name: &str, demodulator: &D, image: &[f32], target: &Target) -> View {
    let mut best: Option<(f64, Trace)> = None;
    let mut view = None;
    for _ in 0..RUNS {
        let clock = Instant::now();
        let (measured, trace) =
            measure_view_traced_with(demodulator, image, WIDTH, HEIGHT, target, None, true);
        let ms = clock.elapsed().as_secs_f64() * 1e3;
        view = Some(measured.expect("board measured"));
        if best.as_ref().is_none_or(|(b, _)| ms < *b) {
            best = Some((ms, trace));
        }
    }
    let (ms, trace) = best.expect("ran");
    let attempt = &trace.attempts[trace.chosen.expect("chosen")];
    let stages: Vec<String> = attempt
        .timings
        .iter()
        .map(|(stage, t)| format!("{stage} {t:.1}"))
        .collect();
    println!(
        "{name}: {ms:.1} ms (search {:.1}, {})",
        trace.search_ms,
        stages.join(", ")
    );
    let field_ms = (0..RUNS)
        .map(|_| {
            let clock = Instant::now();
            let field = demodulated_field_with(demodulator, image, WIDTH, HEIGHT, attempt);
            assert!(field.is_some());
            clock.elapsed().as_secs_f64() * 1e3
        })
        .fold(f64::INFINITY, f64::min);
    println!("{name}: field {field_ms:.1} ms");
    view.expect("ran")
}
