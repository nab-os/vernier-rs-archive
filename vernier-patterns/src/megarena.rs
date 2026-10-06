//! Megarena absolute-pattern generation (André et al. 2020, 2021).
//!
//! A megarena is a grid of dots whose presence encodes an absolute position,
//! carrying two things at once: the regular dot periodicity is a carrier whose
//! phase gives sub-period position (measured by `vernier-spectral`), and dots
//! removed in a structured way encode which period you're in (recovered by
//! `vernier-pose::absolute`).
//!
//! Bits run along each axis, 3 periods per bit. The two outer periods are always
//! present; the central one is present for bit `1`, absent for `0`. Removing
//! only the central period keeps the carrier strong and doesn't shift its phase,
//! so the absolute code doesn't corrupt the fine measurement. Bits come from a
//! maximal [`Lfsr`](crate::lfsr::Lfsr), so every window of `order` bits is unique.
//!
//! The 2D pattern is the product of the x- and y-codes: a dot at (col, row) is
//! present iff both periods are present. One corner dot of the elementary cell
//! is dropped to break the π/2 rotation ambiguity.
//!
//! Dot `p` is centred on `p · period`, where the cosine carrier peaks, so its
//! period cell spans `[p - ½, p + ½) · period` (C++ `(int)(x / period + 0.5)`).
//! Removing a period therefore removes whole dots, never halves of two.

use vernier_core::scalar::consts::TAU;
use vernier_core::{GrayImage, Real};

use crate::PatternPose;
use crate::lfsr::Lfsr;
use crate::render::{into_pattern_frame, render_with};

/// Parameters of a megarena pattern.
#[derive(Clone, Debug)]
pub struct Megarena {
    /// Spatial period in pixels (one dot period).
    pub period_px: Real,
    /// LFSR order = bits per unique window. Determines absolute range.
    pub order: u32,
    /// The absolute code sequence (shared along both axes).
    code: Lfsr,
    /// LFSR index of the bit at triple 0 (image centre for pose=0). Shifting it
    /// away from 0 keeps the decode window off the sequence boundary, which would
    /// otherwise inflate the recovered periodshift by ~3×len periods.
    lfsr_offset: i64,
}

impl Megarena {
    /// Builds a megarena pattern, or `None` if `order` is unsupported.
    pub fn new(period_px: Real, order: u32) -> Option<Self> {
        let code = Lfsr::maximal(order)?;
        Some(Self {
            period_px,
            order,
            code,
            lfsr_offset: 0,
        })
    }

    /// Sets the LFSR index placed at triple 0. A value of `order as i64` keeps
    /// the decode window clear of the sequence boundary for small pixel offsets.
    pub fn with_lfsr_offset(mut self, offset: i64) -> Self {
        self.lfsr_offset = offset;
        self
    }

    /// Whether the period at integer index `p` along an axis is present. Periods
    /// group in threes (one bit each): the outer two are always present, the
    /// central one follows the code bit for that triple. Public so CAD export can
    /// decide which dots to emit (C++ `bitSequence(p)`).
    pub fn period_present(&self, p: i64) -> bool {
        let triple = p.div_euclid(3);
        let within = p.rem_euclid(3); // 0, 1, or 2
        if within != 1 {
            return true; // outer periods always present
        }
        // Central period: present iff this triple's code bit is 1.
        let k = (triple + self.lfsr_offset).rem_euclid(self.code.len() as i64) as usize;
        self.code.bit_at(k) == 1
    }

    /// Renders the pattern on the GPU at `pose` (requires the `vulkan` feature).
    /// Only active cells are submitted as draw instances; absent cells produce no
    /// geometry, leaving those spots dark.
    #[cfg(feature = "vulkan")]
    pub fn render_gpu(
        &self,
        renderer: &vernier_render::PatternRenderer,
        camera: &vernier_render::CameraModel,
        width: usize,
        height: usize,
        pose: &PatternPose,
    ) -> vernier_core::GrayImage {
        let period_um = self.period_px as f32 * camera.pixel_size;
        let pose_x_um = pose.x as f32 * camera.pixel_size;
        let pose_y_um = pose.y as f32 * camera.pixel_size;

        let half_diag_um =
            ((width * width + height * height) as f32).sqrt() * 0.5 * camera.pixel_size
                + period_um;

        let col_min = ((pose_x_um - half_diag_um) / period_um).floor() as i64;
        let col_max = ((pose_x_um + half_diag_um) / period_um).ceil() as i64;
        let row_min = ((pose_y_um - half_diag_um) / period_um).floor() as i64;
        let row_max = ((pose_y_um + half_diag_um) / period_um).ceil() as i64;

        let mut cell_origins = Vec::new();
        for col in col_min..=col_max {
            for row in row_min..=row_max {
                if self.period_present(col)
                    && self.period_present(row)
                    && !(col.rem_euclid(3) == 0 && row.rem_euclid(3) == 0)
                {
                    // Quads span a whole period; shift by half of one so the
                    // dot peak at `col · period` sits in the middle of its quad.
                    cell_origins.push([
                        (col as f32 - 0.5) * period_um,
                        (row as f32 - 0.5) * period_um,
                    ]);
                }
            }
        }

        renderer.render_quads(
            &cell_origins,
            &vernier_render::RenderParams {
                width,
                height,
                period_um,
                pixel_size: camera.pixel_size,
                pose_x_um,
                pose_y_um,
                alpha: pose.theta as f32,
            },
        )
    }

    /// Renders the pattern at `pose` into a `width × height` image.
    pub fn render(&self, width: usize, height: usize, pose: &PatternPose) -> GrayImage {
        let center_x = width as Real / 2.0;
        let center_y = height as Real / 2.0;

        render_with(width, height, |px, py| {
            // Into the pattern's own (axis-aligned) frame.
            let (x_pattern, y_pattern) = into_pattern_frame(px, py, center_x, center_y, pose.theta);
            self.intensity_at(x_pattern - pose.x, y_pattern - pose.y)
        })
    }

    /// Intensity at continuous pattern-frame coordinates `(x, y)` (no pose
    /// applied), matching C++ `getIntensity`. Source of truth for [`render`].
    pub fn intensity_at(&self, x: Real, y: Real) -> Real {
        let ux = x / self.period_px;
        let uy = y / self.period_px;

        // Which dot this point belongs to, on each axis: the nearest carrier
        // peak, not the period it falls in (that would split every dot in two).
        let pxi = (ux + 0.5).floor() as i64;
        let pyi = (uy + 0.5).floor() as i64;

        let x_on = self.period_present(pxi);
        let y_on = self.period_present(pyi);

        // Drop a fixed corner of each 3×3 cell to fix orientation.
        let corner_removed = pxi.rem_euclid(3) == 0 && pyi.rem_euclid(3) == 0;

        if !x_on || !y_on || corner_removed {
            return 0.0;
        }

        // Present dot: cosine carrier mapped to [0,1] so the spectral phase is
        // well defined.
        let carrier_x = 0.5 + 0.5 * (TAU * ux).cos();
        let carrier_y = 0.5 + 0.5 * (TAU * uy).cos();
        carrier_x * carrier_y
    }

    /// Carrier phase along the X axis (radians), matching C++ `getPhase1`.
    pub fn phase1_at(&self, x: Real, _y: Real) -> Real {
        TAU * x / self.period_px
    }

    /// Carrier phase along the Y axis (radians), matching C++ `getPhase2`.
    pub fn phase2_at(&self, _x: Real, y: Real) -> Real {
        TAU * y / self.period_px
    }

    /// The absolute code sequence, for the decoder to match windows against.
    pub fn code(&self) -> &Lfsr {
        &self.code
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_for_valid_order() {
        let m = Megarena::new(20.0, 8).unwrap();
        assert_eq!(m.order, 8);
        assert_eq!(m.code().len(), 255);
    }

    #[test]
    fn rejects_unsupported_order() {
        assert!(Megarena::new(20.0, 2).is_none());
    }

    #[test]
    fn outer_periods_always_present() {
        let m = Megarena::new(20.0, 6).unwrap();
        // Within every triple, indices 0 and 2 are present regardless of code.
        for triple in 0..10 {
            assert!(m.period_present(triple * 3)); // within = 0
            assert!(m.period_present(triple * 3 + 2)); // within = 2
        }
    }

    #[test]
    fn central_period_follows_code() {
        let m = Megarena::new(20.0, 6).unwrap();
        for triple in 0..20i64 {
            let central = triple * 3 + 1;
            let expected = m
                .code()
                .bit_at(triple.rem_euclid(m.code().len() as i64) as usize)
                == 1;
            assert_eq!(m.period_present(central), expected, "triple {triple}");
        }
    }

    #[test]
    fn removed_dot_is_removed_whole() {
        let period = 20.0;
        let m = Megarena::new(period, 6).unwrap();
        // Dot (0, 0) is the dropped orientation corner: dark over its whole
        // cell, while its always-present neighbours stay fully lit at their peaks.
        for &(x, y) in &[(0.0, 0.0), (-9.0, 0.0), (9.0, 9.0), (0.0, -9.0)] {
            assert_eq!(m.intensity_at(x, y), 0.0, "({x}, {y})");
        }
        for &(x, y) in &[(-period, 0.0), (2.0 * period, 0.0), (0.0, -period)] {
            assert!((m.intensity_at(x, y) - 1.0).abs() < 1e-9, "({x}, {y})");
        }
    }

    #[test]
    fn renders_expected_dimensions_and_range() {
        let m = Megarena::new(16.0, 8).unwrap();
        let img = m.render(64, 64, &PatternPose::IDENTITY);
        assert_eq!(img.width(), 64);
        assert_eq!(img.height(), 64);
        for &v in img.as_slice() {
            assert!((0.0..=1.0).contains(&v), "intensity {v} out of range");
        }
    }
}
