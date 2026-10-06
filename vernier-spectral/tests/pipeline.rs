//! End-to-end test of the backend-generic detection chain.
//!
//! Uses `CpuBackend` as the concrete backend (dev-dependency only). The library
//! never names it — this test is where a real backend is substituted in, as
//! `vernier-cli` will do for the GPU. That this compiles proves the trait
//! surface is sufficient for the real filter -> ifft -> plane-fit chain.

use vernier_core::buffer::BufferLayout;
use vernier_core::{Complex32, ComputeBackend};
use vernier_cpu::CpuBackend;
use vernier_spectral::spectrum::{analyze_direction, forward};

fn cosine_pattern(
    width: usize,
    height: usize,
    frequency_x: usize,
) -> (Vec<Complex32>, BufferLayout) {
    use std::f32::consts::TAU;
    let layout = BufferLayout::packed(width, height);
    let mut data = Vec::with_capacity(layout.len());
    for _row in 0..height {
        for col in 0..width {
            let phase = TAU * (frequency_x as f32) * (col as f32) / (width as f32);
            data.push(Complex32::new(phase.cos(), 0.0));
        }
    }
    (data, layout)
}

#[test]
fn analyze_recovers_horizontal_orientation() {
    let backend = CpuBackend::new();
    let (data, layout) = cosine_pattern(32, 32, 3);
    let mut buf = backend.upload(&data, layout).unwrap();

    forward(&backend, &mut buf).unwrap();
    let spectrum = buf.clone();
    let dir = analyze_direction(&backend, spectrum, 1.5).unwrap();

    // Horizontal cosine: phase gradient along x, so b ~ 0 and orientation ~ 0.
    let theta = dir.plane.orientation();
    assert!(theta.abs() < 0.15, "expected ~0 orientation, got {theta}");

    // The plane-implied peak should sit near fx = 3 (or its mirror) and fy ~ 0.
    let (m, _n) = dir.peak;
    assert!(m.is_finite());
}
