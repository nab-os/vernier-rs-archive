//! Renders a set of views of a printed megarena through a known camera, to
//! calibrate from and compare against the truth (see
//! `make -C vernier-cabi/examples run-pnp-megarena`).
//!
//! ```text
//! cargo run --release -p vernier-camera --example megarena_views -- <out-dir>
//! ```
//!
//! Writes `view_00.pgm` … (8-bit binary PGM, 640×480) and `truth.txt`: the
//! camera, then one line per view `name rx ry rz tx ty tz`, an OpenCV rotation
//! vector and a translation in mm from board to camera. The megarena has 2 mm
//! dots and LFSR order 8, its origin at the board's centre.

use std::fmt::Write as _;
use std::path::PathBuf;

use nalgebra::Vector3;
use vernier_camera::{Camera, Model, RigidPose, Scene, Target};

const WIDTH: usize = 640;
const HEIGHT: usize = 480;

fn main() {
    let out = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| "megarena-views".into()));
    std::fs::create_dir_all(&out).expect("output directory");

    let mut camera = Camera::ideal(Model::Pinhole, WIDTH, HEIGHT, 700.0, 701.0, 322.5, 238.0);
    camera.distortion = vec![-0.18, 0.09, 0.0005, -0.0003, 0.0];
    let target = Target::megarena(2.0, 8);
    let scene = Scene {
        camera: &camera,
        target: &target,
        half_size: [150.0, 110.0],
        background: 0.8,
        supersample: 3,
    };

    // Rotation vector (radians) and translation (mm): square on, tilted
    // about either axis and the diagonals, turned about the optical axis.
    let poses = [
        ([0.05, -0.03, 0.2], [0.0, 0.0, 230.0]),
        ([0.5, 0.0, -0.3], [10.0, -5.0, 220.0]),
        ([-0.5, 0.05, 0.9], [-8.0, 6.0, 240.0]),
        ([0.0, 0.5, 1.6], [6.0, 8.0, 225.0]),
        ([0.05, -0.5, -1.2], [-6.0, -10.0, 235.0]),
        ([0.35, 0.35, 2.4], [25.0, 18.0, 230.0]),
        ([-0.35, 0.35, -2.2], [-25.0, 18.0, 240.0]),
        ([0.35, -0.35, 0.5], [22.0, -16.0, 225.0]),
        ([-0.35, -0.35, 3.0], [-22.0, -16.0, 235.0]),
        ([0.6, -0.2, 0.1], [0.0, 5.0, 250.0]),
    ];

    let mut truth = String::new();
    writeln!(
        truth,
        "camera pinhole {WIDTH} {HEIGHT} fx {} fy {} cx {} cy {} distortion {:?}",
        camera.fx, camera.fy, camera.cx, camera.cy, camera.distortion
    )
    .unwrap();
    writeln!(truth, "target megarena pitch 2 mm order 8").unwrap();
    for (n, (r, t)) in poses.iter().enumerate() {
        let pose = RigidPose::from_vectors(
            Vector3::new(r[0], r[1], r[2]),
            Vector3::new(t[0], t[1], t[2]),
        );
        let image = scene.render(&pose);
        let name = format!("view_{n:02}");
        let mut pgm = format!("P5\n{WIDTH} {HEIGHT}\n255\n").into_bytes();
        pgm.extend(image.iter().map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8));
        std::fs::write(out.join(format!("{name}.pgm")), pgm).expect("write view");
        writeln!(
            truth,
            "{name} {} {} {} {} {} {}",
            r[0], r[1], r[2], t[0], t[1], t[2]
        )
        .unwrap();
    }
    std::fs::write(out.join("truth.txt"), truth).expect("write truth");
    println!("{} views written to {}", poses.len(), out.display());
}
