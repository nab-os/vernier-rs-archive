//! Full-stack test: image -> detection (real chain) -> pose, across all four
//! crates, with the CPU backend as the concrete substitute.

use vernier_core::buffer::BufferLayout;
use vernier_core::{Complex32, ComputeBackend};
use vernier_cpu::CpuBackend;
use vernier_pose::{Calibration, periodic};
use vernier_spectral::spectrum::{analyze_direction, forward};

/// A horizontal cosine: fundamental along x, fy = 0.
fn cosine(width: usize, height: usize, frequency: usize) -> (Vec<Complex32>, BufferLayout) {
    use std::f32::consts::TAU;
    let layout = BufferLayout::packed(width, height);
    let mut data = Vec::with_capacity(layout.len());
    for _row in 0..height {
        for col in 0..width {
            let value = (TAU * frequency as f32 * col as f32 / width as f32).cos();
            data.push(Complex32::new(value, 0.0));
        }
    }
    (data, layout)
}

#[test]
fn image_to_pose_via_real_chain() {
    let backend = CpuBackend::new();
    let (data, layout) = cosine(32, 32, 3);
    let mut buf = backend.upload(&data, layout).unwrap();

    // Real chain: forward FFT, then analyze the dominant direction.
    forward(&backend, &mut buf).unwrap();
    let spectrum = buf.clone();
    let dir = analyze_direction(&backend, spectrum, 1.5).unwrap();

    // The fitted plane gradient should point essentially along x (b ~ 0),
    // so orientation ~ 0.
    let theta = dir.plane.orientation();
    assert!(
        theta.abs() < 0.1,
        "expected near-zero orientation, got {theta}"
    );

    // And a single-direction fine pose should be finite and sane.
    let calib = Calibration::new(10.0, 32, 32);
    let pose = periodic::estimate_single(&dir.plane, &calib);
    assert!(pose.x.is_finite() && pose.y.is_finite());
}
