//! Precision on images from an independent renderer: fmac, the Vernier
//! project's marker renderer, with a thin-lens camera (depth-of-field blur
//! that changes with distance, diffraction, gamma, 8-bit output).
//!
//! The board is rendered here into a bitmap by vernier's own pattern code, so
//! fmac's marker frame is the board frame exactly, and fmac renders it at
//! known poses through a known camera. Then, as a user would: measure every
//! picture, calibrate from one set of views, and solve the pose of another,
//! comparing everything with the truth.
//!
//! ```text
//! cargo run --release -p vernier-camera --example fmac_precision -- prepare <dir>
//! <render-build>/render <dir>/camera.json <dir>/board.png <dir>/calibration.csv <dir>
//! <render-build>/render <dir>/camera.json <dir>/board.png <dir>/test.csv <dir>
//! cargo run --release -p vernier-camera --example fmac_precision -- evaluate <dir>
//! ```
//!
//! `examples/fmac/` holds the renderer and how to build it.

use std::path::Path;

use nalgebra::{Rotation3, Vector3};
use vernier_camera::*;
use vernier_patterns::PatternPose;
use vernier_patterns::checkerboard::Checkerboard;

const WIDTH: usize = 1280;
const HEIGHT: usize = 720;
/// Square side on the board, mm.
const SQUARE: f64 = 5.0;
/// LFSR code size of the board.
const ORDER: u32 = 6;
/// Board extent in squares.
const SQUARES: (usize, usize) = (56, 40);
/// Bitmap pixels per square.
const BITMAP_SQUARE: usize = 64;
/// 6 µm pixels and a 6 mm lens at f/2, focused at 300 mm.
const PIXEL_PITCH: f64 = 0.006;
const F_NUMBER: f64 = 2.0;
const FOCUS: f64 = 300.0;

/// The camera fmac renders through: the truth calibration is judged against.
fn camera() -> Camera {
    let mut camera = Camera::ideal(Model::Pinhole, WIDTH, HEIGHT, 1000.0, 1000.0, 641.3, 357.8);
    camera.distortion = vec![-0.12, 0.08, 0.0005, -0.0003, 0.0];
    camera
}

fn target() -> Target {
    Target::new(SQUARE, ORDER)
}

/// Turned `turn` about the optical axis, then tilted `tilt` about an axis in
/// the board's plane at `axis` from its x, at `distance` with its origin
/// `offset` mm off the optical axis.
fn pose(turn: f64, axis: f64, tilt: f64, offset: (f64, f64), distance: f64) -> RigidPose {
    let tilt_axis = Vector3::new(axis.to_radians().cos(), axis.to_radians().sin(), 0.0);
    let rotation = Rotation3::from_scaled_axis(tilt_axis * tilt.to_radians())
        * Rotation3::from_scaled_axis(Vector3::z() * turn.to_radians());
    RigidPose {
        rotation,
        translation: Vector3::new(offset.0, offset.1, distance),
    }
}

/// Calibration views: tilted 20 to 40° every which way, placed around the
/// frame, 260 to 380 mm off.
fn calibration_poses() -> Vec<RigidPose> {
    (0..20)
        .map(|i| {
            let f = i as f64;
            pose(
                37.0 * f,
                (i * 72 % 360) as f64 + 11.0,
                [20.0, 30.0, 40.0][i % 3],
                (45.0 * (1.3 * f).cos(), 30.0 * (0.9 * f).sin()),
                260.0 + 120.0 * ((i * 7) % 10) as f64 / 9.0,
            )
        })
        .collect()
}

/// Test poses: 220 to 450 mm, square on to 50°.
fn test_poses() -> Vec<RigidPose> {
    let mut poses = Vec::new();
    for (k, distance) in [220.0, 260.0, 300.0, 350.0, 400.0, 450.0]
        .into_iter()
        .enumerate()
    {
        for (j, tilt) in [0.0, 15.0, 30.0, 50.0].into_iter().enumerate() {
            poses.push(pose(
                23.0 * (k * 4 + j) as f64,
                (k * 4 + j) as f64 * 53.0,
                tilt,
                (8.0 * k as f64 - 20.0, 5.0 * j as f64 - 7.0),
                distance,
            ));
        }
    }
    poses
}

/// Writes poses as fmac reads them: one line per picture, its name, then the
/// rotation vector and the translation.
fn write_poses(path: &Path, prefix: &str, poses: &[RigidPose]) {
    let text: String = poses
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let r = p.rotation.scaled_axis();
            let t = p.translation;
            format!(
                "{prefix}_{i:02},{:.12},{:.12},{:.12},{:.9},{:.9},{:.9}\n",
                r.x, r.y, r.z, t.x, t.y, t.z
            )
        })
        .collect();
    std::fs::write(path, text).unwrap();
}

/// Writes into `dir` what fmac renders from: the board bitmap, the camera and
/// the two sets of poses.
fn prepare(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    // The board in the pattern's own frame: bitmap pixel (c, r) has its
    // centre at (c − n/2, r − m/2) bitmap pixels from the origin.
    let (n, m) = (SQUARES.0 * BITMAP_SQUARE, SQUARES.1 * BITMAP_SQUARE);
    let board = Checkerboard::new(BITMAP_SQUARE as f64, ORDER).unwrap();
    let image = board.render(n, m, &PatternPose::new(0.0, 0.0, 0.0));
    let bytes: Vec<u8> = image
        .as_slice()
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    image::GrayImage::from_raw(n as u32, m as u32, bytes)
        .unwrap()
        .save(dir.join("board.png"))
        .unwrap();
    // fmac puts bitmap column c over x from c·pw − origin to (c+1)·pw −
    // origin, so with the origin half a pixel past the centre, its marker
    // frame is the board frame in millimetres.
    let pixel_mm = SQUARE / BITMAP_SQUARE as f64;
    let c = camera();
    let d = &c.distortion;
    let json = format!(
        r#"{{
    "image_width": {WIDTH},
    "image_height": {HEIGHT},
    "camera_matrix": {{ "type_id": "opencv-matrix", "rows": 3, "cols": 3, "dt": "d",
        "data": [ {}, 0.0, {}, 0.0, {}, {}, 0.0, 0.0, 1.0 ] }},
    "distortion_coefficients": {{ "type_id": "opencv-matrix", "rows": 5, "cols": 1, "dt": "d",
        "data": [ {}, {}, {}, {}, {} ] }},
    "bit_depth": 8,
    "focus_distance": {FOCUS},
    "pixel_pitch": {PIXEL_PITCH},
    "f_number": {F_NUMBER},
    "f_number_max": {F_NUMBER},
    "f_number_min": {F_NUMBER},
    "marker_width": {},
    "marker_height": {},
    "marker_origin_x": {},
    "marker_origin_y": {},
    "light_wavelength": 0.00055,
    "background_intensity": 0.85,
    "unit": "mm",
    "brand": "vernier fmac_precision camera",
    "yaw_min": -3.14159, "yaw_max": 3.14159, "pitch_min": -0.785, "pitch_max": 0.785,
    "roll_min": -0.785, "roll_max": 0.785, "x_min": 0.0, "x_max": 0.0, "y_min": 0.0,
    "y_max": 0.0, "z_min": 200.0, "z_max": 500.0
}}
"#,
        c.fx,
        c.cx,
        c.fy,
        c.cy,
        d[0],
        d[1],
        d[2],
        d[3],
        d[4],
        n as f64 * pixel_mm,
        m as f64 * pixel_mm,
        (n as f64 / 2.0 + 0.5) * pixel_mm,
        (m as f64 / 2.0 + 0.5) * pixel_mm,
    );
    std::fs::write(dir.join("camera.json"), json).unwrap();
    write_poses(
        &dir.join("calibration.csv"),
        "calibration",
        &calibration_poses(),
    );
    write_poses(&dir.join("test.csv"), "test", &test_poses());
    eprintln!(
        "wrote the board, the camera and the poses to {}",
        dir.display()
    );
}

/// Loads one of fmac's pictures as intensities in `0..=1`, with Gaussian noise
/// of deviation `noise` added from a generator seeded by `seed`.
fn load(path: &Path, noise: f64, seed: u64) -> Vec<f32> {
    let image = image::open(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .to_luma8();
    assert_eq!(
        (image.width() as usize, image.height() as usize),
        (WIDTH, HEIGHT)
    );
    // Xorshift; the state must not be zero.
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    image
        .into_raw()
        .iter()
        .map(|&v| {
            let mut x = v as f64 / 255.0;
            if noise > 0.0 {
                // Box–Muller.
                let (u, w) = (uniform().max(1e-300), uniform());
                x += noise * (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * w).cos();
            }
            x.clamp(0.0, 1.0) as f32
        })
        .collect()
}

/// A view's points against the true projection, px.
fn point_errors(camera: &Camera, pose: &RigidPose, view: &View) -> Vec<f64> {
    view.points
        .iter()
        .filter_map(|p| {
            let q = camera.project(&pose.apply(p.board))?;
            Some((q[0] - p.pixel[0]).hypot(q[1] - p.pixel[1]))
        })
        .collect()
}

/// Root mean square; 0 for no values.
fn rms(values: &[f64]) -> f64 {
    (values.iter().map(|v| v * v).sum::<f64>() / values.len().max(1) as f64).sqrt()
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

/// Measures the pictures, calibrates from the calibration set and solves the
/// test set with the true and the calibrated camera, printing the comparison
/// with the truth, without and then with added noise.
fn evaluate(dir: &Path) {
    let truth = camera();
    let target = target();
    for noise in [0.0, 0.01] {
        println!("\n## noise σ {noise} (on intensities 0..1, after fmac's 8-bit output)\n");
        let views = measure_calibration_views(dir, noise, &target, &truth);
        let result = calibrate(&views, Model::Pinhole).expect("calibrates");
        report_calibration(&result, &truth);
        evaluate_test_views(dir, noise, &target, &truth, &result.camera);
    }
}

/// Measures the calibration pictures, printing how their points compare with
/// the true projection.
fn measure_calibration_views(dir: &Path, noise: f64, target: &Target, truth: &Camera) -> Vec<View> {
    let mut views = Vec::new();
    let mut errors = Vec::new();
    for (i, pose) in calibration_poses().iter().enumerate() {
        let image = load(
            &dir.join(format!("calibration_{i:02}.png")),
            noise,
            i as u64,
        );
        match measure_view(&image, WIDTH, HEIGHT, target) {
            Ok(view) => {
                if view.is_absolute() {
                    errors.extend(point_errors(truth, pose, &view));
                }
                views.push(view);
            }
            Err(e) => println!("calibration view {i}: {e}"),
        }
    }
    let read = views.iter().filter(|v| v.is_absolute()).count();
    println!(
        "calibration views: {} measured, {read} with the code, {} points; points against the truth: rms {:.4} px",
        views.len(),
        views.iter().map(|v| v.points.len()).sum::<usize>(),
        rms(&errors)
    );
    views
}

/// Prints the calibrated camera against the true one.
fn report_calibration(result: &Calibration, truth: &Camera) {
    let c = &result.camera;
    println!(
        "calibrated: reprojection rms {:.4} px | fx {:.3} ({:+.4}%), fy {:.3} ({:+.4}%), cx {:.3} ({:+.3} px), cy {:.3} ({:+.3} px)",
        result.rms,
        c.fx,
        100.0 * (c.fx / truth.fx - 1.0),
        c.fy,
        100.0 * (c.fy / truth.fy - 1.0),
        c.cx,
        c.cx - truth.cx,
        c.cy,
        c.cy - truth.cy
    );
    println!(
        "distortion: {:?} (truth {:?})",
        c.distortion
            .iter()
            .map(|v| (v * 1e5).round() / 1e5)
            .collect::<Vec<_>>(),
        truth.distortion
    );
    let stray = projection_stray(truth, c);
    println!(
        "calibrated against true projection over the frame: rms {:.4} px, worst {:.4} px",
        rms(&stray),
        stray.iter().copied().fold(0.0, f64::max)
    );
}

/// Over a grid on the frame, how far the calibrated camera's projection strays
/// from the true one, px, for points 300 mm off.
fn projection_stray(truth: &Camera, calibrated: &Camera) -> Vec<f64> {
    let mut stray = Vec::new();
    for y in (0..HEIGHT).step_by(40) {
        for x in (0..WIDTH).step_by(40) {
            if let Some(ray) = truth.unproject([x as f64, y as f64]) {
                let point = ray * (300.0 / ray.z);
                if let Some(q) = calibrated.project(&point) {
                    stray.push((q[0] - x as f64).hypot(q[1] - y as f64));
                }
            }
        }
    }
    stray
}

/// Position errors, µm, of the test views at one distance, solved with the
/// true and with the calibrated camera.
struct DistanceErrors {
    distance: f64,
    true_camera: Vec<f64>,
    calibrated_camera: Vec<f64>,
}

/// Solves the pose of every test picture with both cameras and prints a table
/// row per picture, then the median position error by distance.
fn evaluate_test_views(
    dir: &Path,
    noise: f64,
    target: &Target,
    truth: &Camera,
    calibrated: &Camera,
) {
    println!(
        "\n| distance mm | tilt ° | points | point rms px | defocus σ px | true camera: position µm | turn mdeg | calibrated camera: position µm | turn mdeg |"
    );
    println!("|---|---|---|---|---|---|---|---|---|");
    let mut by_distance: Vec<DistanceErrors> = Vec::new();
    for (i, pose) in test_poses().iter().enumerate() {
        let image = load(
            &dir.join(format!("test_{i:02}.png")),
            noise,
            1000 + i as u64,
        );
        let distance = pose.translation.z;
        // Angle between the board's normal and the optical axis.
        let tilt = (pose.rotation * Vector3::z())
            .z
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees();
        let (measured, trace) = measure_view_traced(&image, WIDTH, HEIGHT, target, None, false);
        let defocus = trace.chosen.and_then(|k| trace.attempts[k].defocus);
        let view = match measured {
            Ok(view) if view.is_absolute() => view,
            Ok(_) => {
                println!("| {distance:.0} | {tilt:.0} | code not read | | | | | | |");
                continue;
            }
            Err(e) => {
                println!("| {distance:.0} | {tilt:.0} | {e} | | | | | | |");
                continue;
            }
        };
        let point_rms = rms(&point_errors(truth, pose, &view));
        // Position error in µm and turn error in millidegrees.
        let solve = |camera: &Camera| {
            solve_pnp(camera, &view).ok().map(|fit| {
                (
                    1000.0 * (fit.pose.translation - pose.translation).norm(),
                    1000.0 * fit.pose.angle_to(pose).to_degrees(),
                )
            })
        };
        let (with_truth, with_calibrated) = (solve(truth), solve(calibrated));
        let cells = |errors: Option<(f64, f64)>| match errors {
            Some((position, turn)) => format!("{position:.1} | {turn:.2}"),
            None => "– | –".to_string(),
        };
        println!(
            "| {distance:.0} | {tilt:.0} | {} | {point_rms:.4} | {} | {} | {} |",
            view.points.len(),
            defocus.map_or("–".to_string(), |d| format!("{d:.2}")),
            cells(with_truth),
            cells(with_calibrated)
        );
        let entry = match by_distance.iter().position(|e| e.distance == distance) {
            Some(index) => &mut by_distance[index],
            None => {
                by_distance.push(DistanceErrors {
                    distance,
                    true_camera: Vec::new(),
                    calibrated_camera: Vec::new(),
                });
                by_distance.last_mut().unwrap()
            }
        };
        if let (Some(a), Some(b)) = (with_truth, with_calibrated) {
            entry.true_camera.push(a.0);
            entry.calibrated_camera.push(b.0);
        }
    }
    println!("\nmedian position error by distance, µm (true camera / calibrated camera):");
    for mut errors in by_distance {
        println!(
            "  {:.0} mm: {:.1} / {:.1}",
            errors.distance,
            median(&mut errors.true_camera),
            median(&mut errors.calibrated_camera)
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match (args.get(1).map(String::as_str), args.get(2)) {
        (Some("prepare"), Some(dir)) => prepare(Path::new(dir)),
        (Some("evaluate"), Some(dir)) => evaluate(Path::new(dir)),
        _ => eprintln!("usage: fmac_precision (prepare | evaluate) <dir>"),
    }
}
