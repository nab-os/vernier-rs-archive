//! # vernier-pose
//!
//! Turns detected phase planes ([`PhasePlane`](vernier_spectral::PhasePlane))
//! into a [`Pose`](vernier_core::Pose) — the algorithm layer for the `Pose` type
//! that lives in `vernier-core`.
//!
//! Three estimation paths, mirroring the regimes in the Vernier papers:
//!
//! - [`periodic`] — phase-only. Fine sub-period translation and orientation, but
//!   position is ambiguous modulo the period (you know where you are within a
//!   cell, not which cell).
//! - [`absolute`] — coarse code decode + fine phase. Resolves which period
//!   you're in for an unambiguous `(x, y, θ)`.
//! - [`checkerboard`] — the same coarse+fine split for the coded checkerboard,
//!   whose carriers run along the diagonals and whose code is read from squares
//!   painted against their checkerboard parity.
//!
//! All are plain host-side functions over already-detected features — no
//! backend generics, since the device work is done by this point.

pub mod absolute;
pub mod checkerboard;
pub mod periodic;

use vernier_core::Real;

/// Pattern calibration: the physical scale that converts image-domain
/// measurements (pixels, radians) into real-world units.
#[derive(Clone, Copy, Debug)]
pub struct Calibration {
    /// Spatial period of the pattern in physical units (e.g. micrometres per
    /// pattern cell).
    pub period: Real,
    /// Image width in pixels (needed to convert a frequency bin into cycles per
    /// pixel, then into the observed period).
    pub image_width: usize,
    /// Image height in pixels.
    pub image_height: usize,
}

impl Calibration {
    /// Creates a calibration.
    pub fn new(period: Real, image_width: usize, image_height: usize) -> Self {
        Self {
            period,
            image_width,
            image_height,
        }
    }
}
