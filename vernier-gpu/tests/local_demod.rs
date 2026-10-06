//! The local demodulation on the GPU against the CPU reference in
//! `vernier-camera`: single windows, whole measurements and the field.

use nalgebra::Vector3;
use vernier_camera::*;
use vernier_core::{CarrierModel, DemodWindow, LocalDemodulator, Real, WindowDemod};
use vernier_gpu::GpuBackend;

const WIDTH: usize = 480;
const HEIGHT: usize = 360;

fn target() -> Target {
    Target::new(5.0, 6)
}

/// A tilted board through a webcam-like lens.
fn frame() -> Vec<f32> {
    let mut camera = Camera::ideal(Model::Pinhole, WIDTH, HEIGHT, 450.0, 452.0, 243.0, 177.0);
    camera.distortion = vec![-0.22, 0.07, 0.0006, -0.0004, 0.0];
    let pose = RigidPose::from_vectors(
        Vector3::new(0.45, 0.1, -0.4),
        Vector3::new(12.0, -8.0, 285.0),
    );
    Scene {
        camera: &camera,
        target: &target(),
        half_size: [200.0, 160.0],
        background: 0.45,
        supersample: 2,
    }
    .render(&pose)
}

/// Windows all over the frame, edges included, against curved carriers.
fn windows(count: usize, sigma: Real) -> Vec<DemodWindow> {
    (0..count)
        .map(|i| {
            let t = i as Real;
            let bend = 1e-4 * ((i % 7) as Real - 3.0);
            DemodWindow {
                x: (i * 37) % WIDTH,
                y: (i * 53) % HEIGHT,
                sigma,
                carriers: [
                    CarrierModel {
                        k: [0.55 + 0.01 * (t * 0.3).sin(), 0.5],
                        h: [bend, -0.5 * bend, 2.0 * bend],
                    },
                    CarrierModel {
                        k: [0.5, -0.55 + 0.01 * (t * 0.7).cos()],
                        h: [-bend, bend, 0.0],
                    },
                ],
            }
        })
        .collect()
}

fn demodulate<D: LocalDemodulator>(
    demodulator: &D,
    image: &[f32],
    windows: &[DemodWindow],
) -> Vec<WindowDemod> {
    let frame = demodulator.load(image, WIDTH, HEIGHT).unwrap();
    demodulator.demodulate_windows(&frame, windows).unwrap()
}

fn wrapped_gap(a: Real, b: Real) -> Real {
    let d = a - b;
    (d - std::f64::consts::TAU * (d / std::f64::consts::TAU).round()).abs()
}

#[test]
fn windows_match_the_cpu() {
    let image = frame();
    let windows = windows(2000, 8.0);
    let cpu = demodulate(&CpuDemodulator, &image, &windows);
    let gpu = demodulate(&GpuBackend::new(), &image, &windows);
    let mut compared = 0;
    for (c, g) in cpu.iter().zip(&gpu) {
        for k in 0..2 {
            assert!((c.quality[k] - g.quality[k]).abs() < 1e-4, "{c:?} vs {g:?}");
            // The phase only means something where there is carrier.
            if c.quality[k] > 0.05 {
                assert!(wrapped_gap(c.phase[k], g.phase[k]) < 1e-4, "{c:?} vs {g:?}");
                compared += 1;
            }
        }
        if c.offset.is_finite() && c.quality() > 0.05 {
            assert!((c.offset - g.offset).abs() < 1e-3, "{c:?} vs {g:?}");
        }
    }
    assert!(compared > 1000, "only {compared} phases compared");
}

/// More windows than a dispatch takes workgroups along one axis.
#[test]
fn a_large_batch_is_whole() {
    let image = frame();
    let windows = windows(70_000, 1.5);
    let cpu = demodulate(&CpuDemodulator, &image, &windows);
    let gpu = demodulate(&GpuBackend::new(), &image, &windows);
    assert_eq!(gpu.len(), windows.len());
    for (c, g) in cpu.iter().zip(&gpu).skip(65_000) {
        assert!((c.quality[0] - g.quality[0]).abs() < 1e-4, "{c:?} vs {g:?}");
    }
}

#[test]
fn a_view_measures_as_on_the_cpu() {
    let image = frame();
    let gpu = GpuBackend::new();
    let (cpu_view, _) = measure_view_traced_with(
        &CpuDemodulator,
        &image,
        WIDTH,
        HEIGHT,
        &target(),
        None,
        false,
    );
    let (gpu_view, _) =
        measure_view_traced_with(&gpu, &image, WIDTH, HEIGHT, &target(), None, false);
    let (cpu_view, gpu_view) = (cpu_view.unwrap(), gpu_view.unwrap());
    assert!(gpu_view.is_absolute());
    assert_eq!(cpu_view.points.len(), gpu_view.points.len());
    for (c, g) in cpu_view.points.iter().zip(&gpu_view.points) {
        assert_eq!(c.pixel, g.pixel);
        let gap = (c.board[0] - g.board[0]).hypot(c.board[1] - g.board[1]);
        assert!(gap < 1e-5, "{c:?} vs {g:?}");
    }
}

#[test]
fn the_field_matches_the_cpu() {
    let image = frame();
    let gpu = GpuBackend::new();
    let (_, trace) = measure_view_traced_with(&gpu, &image, WIDTH, HEIGHT, &target(), None, true);
    let attempt = &trace.attempts[trace.chosen.expect("a view")];
    let cpu = demodulated_field(&image, WIDTH, HEIGHT, attempt).unwrap();
    let field = demodulated_field_with(&gpu, &image, WIDTH, HEIGHT, attempt).unwrap();
    for c in 0..2 {
        let mut compared = 0;
        for i in 0..WIDTH * HEIGHT {
            let (a, b) = (cpu.amplitude[c][i], field.amplitude[c][i]);
            assert!((a - b).abs() < 1e-3, "amplitude at {i}: {a} vs {b}");
            if a > 0.15 {
                let gap = wrapped_gap(cpu.phase[c][i] as Real, field.phase[c][i] as Real);
                assert!(gap < 1e-3, "phase at {i}: {gap}");
                compared += 1;
            }
        }
        assert!(
            compared > WIDTH * HEIGHT / 4,
            "only {compared} phases compared"
        );
    }
}
