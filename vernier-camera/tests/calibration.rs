//! Synthetic views of the coded checkerboard through known cameras: the
//! measurement against the true projection, then calibration and PnP against
//! the true camera and pose.

use nalgebra::Vector3;
use vernier_camera::*;

const WIDTH: usize = 480;
const HEIGHT: usize = 360;

/// A typical webcam: barrel distortion and a slightly off-centre principal
/// point.
fn pinhole() -> Camera {
    let mut camera = Camera::ideal(Model::Pinhole, WIDTH, HEIGHT, 450.0, 452.0, 243.0, 177.0);
    camera.distortion = vec![-0.22, 0.07, 0.0006, -0.0004, 0.0];
    camera
}

/// A wide fisheye, about 160° across the width of the frame.
fn fisheye() -> Camera {
    let mut camera = Camera::ideal(Model::Fisheye, WIDTH, HEIGHT, 165.0, 166.0, 241.0, 182.0);
    camera.distortion = vec![0.03, -0.01, 0.002, -0.0005];
    camera
}

/// Squares of side 5, LFSR order 6.
fn target() -> Target {
    Target::new(5.0, 6)
}

/// The board at `pose`, large enough to fill the frame.
fn render(camera: &Camera, pose: &RigidPose) -> Vec<f32> {
    let target = target();
    Scene {
        camera,
        target: &target,
        half_size: [200.0, 160.0],
        background: 0.45,
        supersample: 2,
    }
    .render(pose)
}

/// Poses that tilt the board up to about 30° every which way, about
/// `distance` from the camera.
fn poses(distance: f64) -> Vec<RigidPose> {
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
            Vector3::new(t[0], t[1], t[2] * distance),
        )
    })
    .collect()
}

/// Per point of `view`, the distance in pixels between its measured pixel and
/// the true projection of its board point.
fn reprojection_errors(camera: &Camera, pose: &RigidPose, view: &View) -> Vec<f64> {
    view.points
        .iter()
        .map(|p| {
            let q = camera.project(&pose.apply(p.board)).expect("in view");
            (q[0] - p.pixel[0]).hypot(q[1] - p.pixel[1])
        })
        .collect()
}

fn rms(values: &[f64]) -> f64 {
    (values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64).sqrt()
}

#[test]
fn a_tilted_view_measures_to_hundredths_of_a_pixel() {
    for (camera, distance) in [(pinhole(), 200.0), (fisheye(), 120.0)] {
        let pose = poses(distance)[1];
        let view =
            measure_view(&render(&camera, &pose), WIDTH, HEIGHT, &target()).expect("board found");
        assert!(view.is_absolute(), "{:?}: {:?}", camera.model, view.code);
        assert!(
            view.points.len() > 150,
            "{:?}: {} points",
            camera.model,
            view.points.len()
        );
        let errors = reprojection_errors(&camera, &pose, &view);
        let worst = errors.iter().copied().fold(0.0, f64::max);
        assert!(
            rms(&errors) < 0.06 && worst < 0.3,
            "{:?}: rms {} worst {worst}",
            camera.model,
            rms(&errors)
        );
    }
}

#[test]
fn calibration_recovers_the_camera() {
    for (truth, distance) in [(pinhole(), 200.0), (fisheye(), 120.0)] {
        let views: Vec<View> = poses(distance)
            .iter()
            .map(|pose| {
                measure_view(&render(&truth, pose), WIDTH, HEIGHT, &target()).expect("board found")
            })
            .collect();
        let result = calibrate(&views, truth.model).expect("calibrates");
        let c = &result.camera;
        assert!(result.rms < 0.05, "{:?}: rms {}", truth.model, result.rms);
        for (found, expected) in [(c.fx, truth.fx), (c.fy, truth.fy)] {
            assert!(
                (found / expected - 1.0).abs() < 1e-3,
                "{:?}: focal {found} vs {expected}",
                truth.model
            );
        }
        for (found, expected) in [(c.cx, truth.cx), (c.cy, truth.cy)] {
            assert!(
                (found - expected).abs() < 0.3,
                "{:?}: centre {found} vs {expected}",
                truth.model
            );
        }
        assert!(
            (c.distortion[0] - truth.distortion[0]).abs() < 0.01,
            "{:?}: {:?}",
            truth.model,
            c.distortion
        );
    }
}

#[test]
fn pnp_finds_the_board_in_its_own_frame() {
    for (camera, distance) in [(pinhole(), 200.0), (fisheye(), 120.0)] {
        for pose in poses(distance).iter().skip(2).take(2) {
            let view = measure_view(&render(&camera, pose), WIDTH, HEIGHT, &target())
                .expect("board found");
            assert!(view.is_absolute());
            let fit = solve_pnp(&camera, &view).expect("solves");
            let moved = (fit.pose.translation - pose.translation).norm() / pose.translation.norm();
            let turned = fit.pose.angle_to(pose);
            assert!(
                moved < 5e-4 && turned < 5e-4,
                "{:?}: moved {moved} turned {turned}",
                camera.model
            );
        }
    }
}

/// The board is its own mirror image about its diagonal, so a mirrored
/// picture decodes too, and its pose is the mirror of the true one.
#[test]
fn a_mirrored_picture_reads_as_the_board_mirrored() {
    let mut camera = pinhole();
    camera.distortion = vec![-0.22, 0.07, 0.0, 0.0, 0.0];
    let mut mirror = camera.clone();
    mirror.cx = (WIDTH - 1) as f64 - camera.cx;
    let pose = poses(200.0)[1];
    let image = render(&camera, &pose);
    // Each row reversed: the picture a mirror would give.
    let flipped: Vec<f32> = (0..WIDTH * HEIGHT)
        .map(|p| image[(p / WIDTH) * WIDTH + WIDTH - 1 - p % WIDTH])
        .collect();

    let view = measure_view(&flipped, WIDTH, HEIGHT, &target()).expect("board found");
    assert!(view.is_absolute(), "{:?}", view.code);
    let t = solve_pnp(&mirror, &view).expect("solves").pose.translation;
    let expected = Vector3::new(-pose.translation.x, pose.translation.y, pose.translation.z);
    assert!((t - expected).norm() < 0.1, "{t:?} vs {expected:?}");
}

#[test]
fn a_blank_frame_has_no_board() {
    // Grey noise from a linear congruential generator, reproducible.
    let mut state = 1u32;
    let noise: Vec<f32> = (0..WIDTH * HEIGHT)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            0.5 + 0.1 * ((state >> 8) as f32 / (1u32 << 24) as f32 - 0.5)
        })
        .collect();
    assert!(measure_view(&noise, WIDTH, HEIGHT, &target()).is_err());
}

/// A board whose edges are in view, over a paper margin: squares sampled past
/// its end must not spoil the code, whatever its turn.
#[test]
fn a_board_with_its_edges_in_view_reads_at_any_turn() {
    let camera = pinhole();
    let target = target();
    for step in 0..12 {
        let turn = (15.0 * step as f64).to_radians();
        let pose =
            RigidPose::from_vectors(Vector3::new(0.0, 0.0, turn), Vector3::new(3.0, -2.0, 260.0));
        let image = Scene {
            camera: &camera,
            target: &target,
            half_size: [120.0, 100.0],
            background: 0.85,
            supersample: 2,
        }
        .render(&pose);
        let view = measure_view(&image, WIDTH, HEIGHT, &target).expect("board found");
        assert!(view.is_absolute(), "turned {}°: {:?}", 15 * step, view.code);
    }
}
