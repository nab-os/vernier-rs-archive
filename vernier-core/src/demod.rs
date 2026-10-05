//! The [`LocalDemodulator`] trait: the phase of both carriers in Gaussian
//! windows that follow a local model of each, as `vernier-camera` measures a
//! tilted or distorted view. `vernier-camera` implements it on the CPU,
//! `vernier-gpu` with Vulkano compute.
//!
//! A frame is loaded once, then demodulated in batches: every window of a
//! batch is independent, so a backend may spread them as it likes.
//!
//! ```text
//! let frame = demodulator.load(&pixels, width, height)?;
//! demodulator.demodulate_windows(&frame, &windows)?   → one WindowDemod per window
//! demodulator.demodulate_field(&frame, references, σ)? → per pixel, per carrier
//! ```

use crate::error::Result;
use crate::scalar::Real;

/// Local phase model of one carrier around a point: gradient `k` and Hessian
/// `(xx, xy, yy)`, in radians per pixel.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CarrierModel {
    pub k: [Real; 2],
    pub h: [Real; 3],
}

impl CarrierModel {
    /// A plane wave: constant frequency `k`, no curvature.
    pub fn plane(k: [Real; 2]) -> Self {
        Self { k, h: [0.0; 3] }
    }

    /// The phase `(qx, qy)` pixels from where the model's phase is `base`.
    pub fn phase_at(&self, base: Real, qx: Real, qy: Real) -> Real {
        let Self { k, h } = *self;
        base + k[0] * qx
            + k[1] * qy
            + 0.5 * (h[0] * qx * qx + 2.0 * h[1] * qx * qy + h[2] * qy * qy)
    }
}

/// A Gaussian window of `sigma` pixels centred on pixel `(x, y)`, cut at
/// `radius` pixels either way and at the frame edge, and the model of each
/// carrier the reference follows there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DemodWindow {
    pub x: usize,
    pub y: usize,
    pub sigma: Real,
    pub carriers: [CarrierModel; 2],
}

impl DemodWindow {
    /// Pixels either way the window reaches: three sigmas.
    pub fn radius(&self) -> usize {
        (3.0 * self.sigma).ceil() as usize
    }
}

/// What a window found of both carriers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindowDemod {
    /// Against the reference, in radians, wrapped.
    pub phase: [Real; 2],
    /// Carrier amplitude over local contrast, per carrier.
    pub quality: [Real; 2],
    /// How far the carrier's weight sits from the window centre, in window
    /// sigmas, worst carrier. A window hanging off the board or the frame
    /// leans away from the edge, and its phase is biased.
    pub offset: Real,
}

impl WindowDemod {
    /// The worse carrier's quality.
    pub fn quality(&self) -> Real {
        self.quality[0].min(self.quality[1])
    }
}

/// Every pixel of a frame demodulated against a reference phase per carrier,
/// row-major.
#[derive(Clone, Debug)]
pub struct FieldDemod {
    /// Per carrier, the phase against the reference, in radians, wrapped.
    pub phase: [Vec<Real>; 2],
    /// Per carrier, the carrier amplitude over the local contrast.
    pub amplitude: [Vec<Real>; 2],
}

/// Demodulates a grayscale frame in Gaussian windows against local carrier
/// models.
pub trait LocalDemodulator {
    /// A frame ready to be demodulated: borrowed on the CPU, resident on a
    /// device elsewhere.
    type Frame<'a>
    where
        Self: 'a;

    /// Readies a row-major frame of `width × height` intensities.
    fn load<'a>(&'a self, data: &'a [f32], width: usize, height: usize) -> Result<Self::Frame<'a>>;

    /// Per window, `Σw·(I − mean)·e^{−iψ}` with `w` the window weight, `I` the
    /// intensity and `ψ` each carrier's model, read as a phase, an amplitude
    /// over the window's contrast and where the carrier's weight sits.
    fn demodulate_windows(
        &self,
        frame: &Self::Frame<'_>,
        windows: &[DemodWindow],
    ) -> Result<Vec<WindowDemod>>;

    /// At every pixel, the same sum under a separable Gaussian of `sigma`
    /// pixels (normalized at the frame edge) against a reference phase per
    /// pixel and carrier, `width × height` each.
    fn demodulate_field(
        &self,
        frame: &Self::Frame<'_>,
        references: [&[Real]; 2],
        sigma: Real,
    ) -> Result<FieldDemod>;
}
