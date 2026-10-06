//! Coded checkerboard: a 50/50 carrier with the megarena's LFSR position code.
//!
//! The carriers sit at ±45° to the square edges, and at a square centre
//! `φ₁ = π(i+j)`, `φ₂ = π(i−j)`, so the phases index the squares.
//!
//! Squares are grouped in supercells. Some squares per supercell are coding
//! sites; a site is painted against its parity when its bit is 0. Flipping a
//! square centre only shrinks the carrier, never shifts its phase, and the
//! sites alternate colour so the fill stays 50/50.
//!
//! Two layouts: `Squares` (upright squares, code along their edges) and
//! `Diamonds` (squares turned 45°, code along their diagonals, so the code grid
//! stays upright).
//!
//! Two packings: one bit per axis in a 3×3 supercell, or two in a 5×5 one — see
//! [`CodePacking`].
//!
//! Corners can be rounded (see
//! [`with_corner_radius`](Checkerboard::with_corner_radius)) to suit a
//! fabrication process that will not hold a sharp corner anyway. Rounding
//! happens in the squares' own frame, so it follows the lattice through both
//! layouts and any pose, and it feeds the same supersampled renderer, so the
//! arcs come out antialiased.
//!
//! The code period `3·(2ⁿ − 1)` is odd, so the pattern really repeats after two
//! periods. We only claim one.

use vernier_core::scalar::consts::{PI, SQRT_2};
use vernier_core::{GrayImage, Real};

use crate::PatternPose;
use crate::lfsr::Lfsr;
use crate::render::{MAX_CORNER_RADIUS, into_pattern_frame, render_with, rounded_cell};

/// Squares per supercell edge, for [`CodePacking::OneBit`]. Use
/// [`CodePacking::cell`] to stay general.
pub const CELL: i64 = 3;

/// x-code site within a supercell, as `(i mod 3, j mod 3)`.
pub const X_SITE: (i64, i64) = (1, 0);

pub const Y_SITE: (i64, i64) = (0, 1);

/// Same sites for [`CodeLayout::Diamonds`], as `(u mod 3, v mod 3)`.
pub const U_SITE: (i64, i64) = (1, 0);

pub const V_SITE: (i64, i64) = (0, 1);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodeAxis {
    X,
    Y,
}

/// How many code bits each supercell carries per axis.
///
/// The supercell edge is always **odd**, and that is not cosmetic. A site lives
/// at a fixed `(first mod cell, second mod cell)`, so with an even edge both
/// residues have a fixed parity, `first + second` does too, and every coding
/// site in the pattern lands on the same colour. Inverting only ever repaints
/// that one colour: the fill drifts off 50/50 (measured 0.623 white for a 4×4
/// two-bit packing) and the imbalance shows up as a DC term comparable to the
/// carrier itself. An odd edge walks the sites through both parities, which is
/// what holds the fill at 0.500.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CodePacking {
    /// One bit per axis in a 3×3 supercell: three squares per bit.
    #[default]
    OneBit,
    /// Two bits per axis in a 5×5 supercell: 2.5 squares per bit, so 1.2× the
    /// code density and a decode window ~17% shorter for the same order.
    ///
    /// It also writes *fewer* defects than [`OneBit`](Self::OneBit) — four
    /// sites per 25 squares against two per nine, so 7.9% of squares inverted
    /// against 11.3% — which measures as a quieter carrier, not a noisier one:
    /// the code's strongest spurious line drops from 0.89× the carrier at a
    /// third of its frequency to 0.48× at a fifth.
    TwoBits,
}

impl CodePacking {
    /// Every packing, in the order a picker should show them.
    pub const ALL: [CodePacking; 2] = [CodePacking::OneBit, CodePacking::TwoBits];

    /// Squares per supercell edge. Always odd; see the type docs.
    pub fn cell(self) -> i64 {
        match self {
            Self::OneBit => CELL,
            Self::TwoBits => 5,
        }
    }

    /// Code bits per supercell, per axis.
    pub fn bits_per_cell(self) -> i64 {
        match self {
            Self::OneBit => 1,
            Self::TwoBits => 2,
        }
    }

    /// Squares per code bit along one axis — what sets how much pattern a
    /// decode window has to see.
    pub fn squares_per_bit(self) -> Real {
        self.cell() as Real / self.bits_per_cell() as Real
    }

    /// x-code sites in a supercell, in bit order, as
    /// `(first mod cell, second mod cell)`. The two-bit sites share a row, so
    /// the pair runs along the cell edge rather than across it: staggering them
    /// puts a strong spurious line on the carrier diagonal, keeping them
    /// collinear puts a weaker one on the axes.
    pub fn x_sites(self) -> &'static [(i64, i64)] {
        match self {
            Self::OneBit => &[X_SITE],
            Self::TwoBits => &[(1, 0), (3, 0)],
        }
    }

    /// y-code sites, the transpose of [`x_sites`](Self::x_sites).
    pub fn y_sites(self) -> &'static [(i64, i64)] {
        match self {
            Self::OneBit => &[Y_SITE],
            Self::TwoBits => &[(0, 1), (0, 3)],
        }
    }

    /// Which axis a square codes for and which bit of its supercell it carries,
    /// or `None` away from the coding sites. Coordinates are already in the
    /// layout's frame: `(i, j)` for `Squares`, `(i+j, i−j)` for `Diamonds`.
    pub fn coding_site(self, first: i64, second: i64) -> Option<(CodeAxis, i64)> {
        let cell = self.cell();
        let within = (first.rem_euclid(cell), second.rem_euclid(cell));
        if let Some(slot) = self.x_sites().iter().position(|&site| site == within) {
            return Some((CodeAxis::X, slot as i64));
        }
        self.y_sites()
            .iter()
            .position(|&site| site == within)
            .map(|slot| (CodeAxis::Y, slot as i64))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CodeLayout {
    /// Upright squares, code along their edges `(i, j)`.
    #[default]
    Squares,
    /// Squares turned 45°, code along their diagonals `(i+j, i−j)`, which run
    /// along the pattern axes. Costs √2 of range, and each code band is one colour.
    Diamonds,
}

impl CodeLayout {
    /// Angle of the square lattice in the pattern frame.
    pub fn lattice_angle(self) -> Real {
        match self {
            Self::Squares => 0.0,
            Self::Diamonds => PI / 4.0,
        }
    }

    /// Pattern frame to square-lattice frame.
    pub fn to_lattice(self, x: Real, y: Real) -> (Real, Real) {
        match self {
            Self::Squares => (x, y),
            Self::Diamonds => ((x + y) / SQRT_2, (y - x) / SQRT_2),
        }
    }

    /// Square-lattice frame to pattern frame.
    pub fn from_lattice(self, x: Real, y: Real) -> (Real, Real) {
        match self {
            Self::Squares => (x, y),
            Self::Diamonds => ((x - y) / SQRT_2, (x + y) / SQRT_2),
        }
    }
}

/// Parameters of a coded checkerboard pattern.
#[derive(Clone, Debug)]
pub struct Checkerboard {
    /// Square side, in pixels.
    pub square_px: Real,
    pub order: u32,
    /// Same sequence for both axes.
    code: Lfsr,
    /// LFSR index at supercell 0.
    lfsr_offset: i64,
    /// Sub-samples per pixel edge; hard edges alias without it.
    supersample: u32,
    layout: CodeLayout,
    packing: CodePacking,
    /// Corner radius as a fraction of a square side, `0.0` for square corners.
    corner_radius: Real,
}

impl Checkerboard {
    /// `None` if `order` is unsupported.
    pub fn new(square_px: Real, order: u32) -> Option<Self> {
        let code = Lfsr::maximal(order)?;
        Some(Self {
            square_px,
            order,
            code,
            lfsr_offset: 0,
            supersample: 4,
            layout: CodeLayout::Squares,
            packing: CodePacking::OneBit,
            corner_radius: 0.0,
        })
    }

    pub fn with_code_layout(mut self, layout: CodeLayout) -> Self {
        self.layout = layout;
        self
    }

    pub fn code_layout(&self) -> CodeLayout {
        self.layout
    }

    pub fn with_code_packing(mut self, packing: CodePacking) -> Self {
        self.packing = packing;
        self
    }

    pub fn code_packing(&self) -> CodePacking {
        self.packing
    }

    pub fn with_lfsr_offset(mut self, offset: i64) -> Self {
        self.lfsr_offset = offset;
        self
    }

    pub fn with_supersample(mut self, supersample: u32) -> Self {
        self.supersample = supersample.max(1);
        self
    }

    /// Rounds the square corners, `radius` being a fraction of a square side
    /// from `0.0` (square, the default) to `0.5` (as round as a square gets).
    /// Values outside that range are clamped.
    ///
    /// Squares of the same colour sharing an edge — which is how the code is
    /// written — merge into one shape rather than each rounding off, so the
    /// coding sites stay as legible as they are with square corners. Rounding is
    /// symmetric about each square centre, so it damps the carrier without
    /// shifting its phase; it does cost fill balance, since only the corners of
    /// white squares are carved away. See [`white_fraction`](Self::white_fraction).
    pub fn with_corner_radius(mut self, radius: Real) -> Self {
        self.corner_radius = radius.clamp(0.0, MAX_CORNER_RADIUS);
        self
    }

    pub fn corner_radius(&self) -> Real {
        self.corner_radius
    }

    /// The fraction of the *uncoded* carrier that is white, given the corner
    /// radius. A square checkerboard is 50/50; every white square there is
    /// isolated, so rounding carves `r²(4 − π)` off each without giving any
    /// back, and the white fraction drops by up to `(4 − π)/8 ≈ 0.107` at the
    /// full radius. The coded pattern lands slightly above this: its inverted
    /// sites put same-colour squares edge to edge, and a merged pair keeps the
    /// two corners on its seam and gains a filled concave corner elsewhere.
    pub fn white_fraction(&self) -> Real {
        let r = self.corner_radius;
        0.5 - r * r * (4.0 - PI) * 0.5
    }

    pub fn code(&self) -> &Lfsr {
        &self.code
    }

    /// `a·√2`. Give this to the detector as the period, not the square side.
    pub fn carrier_period_px(&self) -> Real {
        self.square_px * SQRT_2
    }

    /// Range in code steps, `cell·(2ⁿ − 1)`. Compare layouts with `range_px`.
    ///
    /// Under [`CodePacking::TwoBits`] the sequence is laid down twice over that
    /// span, but a bit's slot within its supercell has the same parity as its
    /// index, so the second pass never reads as the first: the span stays
    /// unambiguous.
    pub fn range_squares(&self) -> i64 {
        self.packing.cell() * self.code.len() as i64
    }

    /// Range in pixels. Diagonal steps are `a/√2`.
    pub fn range_px(&self) -> Real {
        let steps = self.range_squares() as Real;
        match self.layout {
            CodeLayout::Squares => steps * self.square_px,
            CodeLayout::Diamonds => steps * self.square_px / SQRT_2,
        }
    }

    /// A pattern-frame offset, reduced to the smallest equivalent one modulo the
    /// code period. The period lattice is turned with the squares.
    pub fn wrap_offset(&self, dx: Real, dy: Real) -> (Real, Real) {
        let period = self.range_squares() as Real * self.square_px;
        let wrap = |e: Real| e - period * (e / period).round();
        let (x, y) = self.layout.to_lattice(dx, dy);
        self.layout.from_lattice(wrap(x), wrap(y))
    }

    pub fn code_bit(&self, index: i64) -> u8 {
        let k = (index + self.lfsr_offset).rem_euclid(self.code.len() as i64) as usize;
        self.code.bit_at(k)
    }

    /// [`CodePacking::OneBit`] only, which is what the decoder and the figure
    /// commands assume. [`CodePacking::coding_site`] is the general form.
    pub fn coding_axis(i: i64, j: i64) -> Option<CodeAxis> {
        let within = (i.rem_euclid(CELL), j.rem_euclid(CELL));
        if within == X_SITE {
            Some(CodeAxis::X)
        } else if within == Y_SITE {
            Some(CodeAxis::Y)
        } else {
            None
        }
    }

    /// `(i+j, i−j)`. Only pairs with `u ≡ v (mod 2)` are real squares.
    pub fn diagonal_coords(i: i64, j: i64) -> (i64, i64) {
        (i + j, i - j)
    }

    pub fn diagonal_coding_axis(u: i64, v: i64) -> Option<CodeAxis> {
        let within = (u.rem_euclid(CELL), v.rem_euclid(CELL));
        if within == U_SITE {
            Some(CodeAxis::X)
        } else if within == V_SITE {
            Some(CodeAxis::Y)
        } else {
            None
        }
    }

    /// The square's coordinates in whichever frame the code is written in:
    /// `(i, j)` for [`CodeLayout::Squares`], `(i+j, i−j)` for
    /// [`CodeLayout::Diamonds`].
    fn code_coords(&self, i: i64, j: i64) -> (i64, i64) {
        match self.layout {
            CodeLayout::Squares => (i, j),
            CodeLayout::Diamonds => Self::diagonal_coords(i, j),
        }
    }

    /// True at a coding site whose bit is 0.
    pub fn square_inverted(&self, i: i64, j: i64) -> bool {
        let (first, second) = self.code_coords(i, j);
        let (cell, per_cell) = (self.packing.cell(), self.packing.bits_per_cell());
        // A supercell's bits are consecutive in the sequence, so a run of
        // supercells still reads as a run of the LFSR.
        match self.packing.coding_site(first, second) {
            Some((CodeAxis::X, slot)) => {
                self.code_bit(per_cell * first.div_euclid(cell) + slot) == 0
            }
            Some((CodeAxis::Y, slot)) => {
                self.code_bit(per_cell * second.div_euclid(cell) + slot) == 0
            }
            None => false,
        }
    }

    /// Uncoded colour: white when `i + j` is even.
    pub fn parity_is_white(i: i64, j: i64) -> bool {
        (i + j).rem_euclid(2) == 0
    }

    pub fn square_is_white(&self, i: i64, j: i64) -> bool {
        Self::parity_is_white(i, j) != self.square_inverted(i, j)
    }

    /// Continuous square-lattice coordinates, in units of one square side, so
    /// the square containing a point is the pair of floors. Corner rounding
    /// needs where the point sits *inside* its square, not just which one.
    fn lattice_units(&self, x: Real, y: Real) -> (Real, Real) {
        let (x, y) = self.layout.to_lattice(x, y);
        (x / self.square_px, y / self.square_px)
    }

    pub fn square_at(&self, x: Real, y: Real) -> (i64, i64) {
        let (u, v) = self.lattice_units(x, y);
        (u.floor() as i64, v.floor() as i64)
    }

    pub fn intensity_at(&self, x: Real, y: Real) -> Real {
        let (u, v) = self.lattice_units(x, y);
        let white = rounded_cell(u, v, self.corner_radius, |i, j| self.square_is_white(i, j));
        if white { 1.0 } else { 0.0 }
    }

    pub fn plain_intensity_at(&self, x: Real, y: Real) -> Real {
        let (u, v) = self.lattice_units(x, y);
        let white = rounded_cell(u, v, self.corner_radius, Self::parity_is_white);
        if white { 1.0 } else { 0.0 }
    }

    /// `π(i+j)` at a square centre.
    pub fn phase1_at(&self, x: Real, y: Real) -> Real {
        let (x, y) = self.layout.to_lattice(x, y);
        PI * ((x + y) / self.square_px - 1.0)
    }

    /// `π(i−j)` at a square centre.
    pub fn phase2_at(&self, x: Real, y: Real) -> Real {
        let (x, y) = self.layout.to_lattice(x, y);
        PI * (x - y) / self.square_px
    }

    pub fn square_from_phases(phase1: Real, phase2: Real) -> (i64, i64) {
        let sum = phase1 / PI;
        let difference = phase2 / PI;
        (
            ((sum + difference) * 0.5).round() as i64,
            ((sum - difference) * 0.5).round() as i64,
        )
    }

    pub fn render(&self, width: usize, height: usize, pose: &PatternPose) -> GrayImage {
        self.render_field(width, height, pose, false)
    }

    pub fn render_plain(&self, width: usize, height: usize, pose: &PatternPose) -> GrayImage {
        self.render_field(width, height, pose, true)
    }

    fn render_field(
        &self,
        width: usize,
        height: usize,
        pose: &PatternPose,
        plain: bool,
    ) -> GrayImage {
        let center_x = width as Real / 2.0;
        let center_y = height as Real / 2.0;
        let n = self.supersample as Real;
        let step = 1.0 / n;
        let first = 0.5 * step - 0.5;

        render_with(width, height, |px, py| {
            let mut sum = 0.0;
            for sub_y in 0..self.supersample {
                for sub_x in 0..self.supersample {
                    let sx = px + first + sub_x as Real * step;
                    let sy = py + first + sub_y as Real * step;
                    let (xp, yp) = into_pattern_frame(sx, sy, center_x, center_y, pose.theta);
                    let (x, y) = (xp - pose.x, yp - pose.y);
                    sum += if plain {
                        self.plain_intensity_at(x, y)
                    } else {
                        self.intensity_at(x, y)
                    };
                }
            }
            sum / (n * n)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phases_index_the_squares() {
        let c = Checkerboard::new(7.0, 6).unwrap();
        for i in -10..10i64 {
            for j in -10..10i64 {
                let (x, y) = ((i as Real + 0.5) * 7.0, (j as Real + 0.5) * 7.0);
                assert_eq!(
                    Checkerboard::square_from_phases(c.phase1_at(x, y), c.phase2_at(x, y)),
                    (i, j)
                );
            }
        }
    }

    #[test]
    fn code_does_not_shift_the_carrier_phase() {
        for layout in [CodeLayout::Squares, CodeLayout::Diamonds] {
            let c = Checkerboard::new(8.0, 8).unwrap().with_code_layout(layout);
            let n = c.range_squares();
            let mut im = 0.0;
            for i in 0..n {
                for j in 0..n {
                    let sign = if c.square_is_white(i, j) { 1.0 } else { -1.0 };
                    im += sign * (PI * (i + j) as Real).sin();
                }
            }
            assert!(im.abs() < 1e-6, "{layout:?}: {im}");
        }
    }

    #[test]
    fn half_the_image_is_white() {
        for layout in [CodeLayout::Squares, CodeLayout::Diamonds] {
            let c = Checkerboard::new(9.0, 8).unwrap().with_code_layout(layout);
            let img = c.render(512, 512, &PatternPose::new(13.7, -4.1, 0.37));
            let mean = img.as_slice().iter().map(|&v| v as Real).sum::<Real>()
                / img.as_slice().len() as Real;
            assert!((mean - 0.5).abs() < 0.02, "{layout:?}: {mean}");
        }
    }

    #[test]
    fn square_corners_are_the_default() {
        let c = Checkerboard::new(9.0, 8).unwrap();
        assert_eq!(c.corner_radius(), 0.0);
    }

    #[test]
    fn rounding_is_clamped_to_a_half() {
        assert_eq!(
            Checkerboard::new(9.0, 8)
                .unwrap()
                .with_corner_radius(2.0)
                .corner_radius(),
            0.5
        );
        assert_eq!(
            Checkerboard::new(9.0, 8)
                .unwrap()
                .with_corner_radius(-1.0)
                .corner_radius(),
            0.0
        );
    }

    /// The rounding must not disturb what the detector reads: the code is
    /// written by inverting whole squares, and that is untouched by the radius.
    #[test]
    fn rounding_leaves_the_code_alone() {
        let plain = Checkerboard::new(8.0, 8).unwrap();
        let round = plain.clone().with_corner_radius(0.5);
        for i in -20..20i64 {
            for j in -20..20i64 {
                assert_eq!(plain.square_is_white(i, j), round.square_is_white(i, j));
            }
        }
    }

    /// Rounding carves each white square symmetrically about its centre, so the
    /// carrier loses amplitude but keeps its phase — the same property that lets
    /// the code be written without shifting the carrier.
    #[test]
    fn rounding_does_not_shift_the_carrier_phase() {
        for layout in [CodeLayout::Squares, CodeLayout::Diamonds] {
            let c = Checkerboard::new(8.0, 8)
                .unwrap()
                .with_code_layout(layout)
                .with_corner_radius(0.4);
            // Sample one square finely and check its white mass is centred.
            let (n, side) = (64, c.square_px);
            let (mut mass, mut moment_x, mut moment_y) = (0.0, 0.0, 0.0);
            for a in 0..n {
                for b in 0..n {
                    let (dx, dy) = ((a as Real + 0.5) / n as Real, (b as Real + 0.5) / n as Real);
                    let (x, y) = c.layout.from_lattice(dx * side, dy * side);
                    if c.intensity_at(x, y) > 0.5 {
                        mass += 1.0;
                        moment_x += dx - 0.5;
                        moment_y += dy - 0.5;
                    }
                }
            }
            assert!(mass > 0.0, "{layout:?}: sampled square is empty");
            assert!(moment_x.abs() / mass < 1e-2, "{layout:?}: {moment_x}");
            assert!(moment_y.abs() / mass < 1e-2, "{layout:?}: {moment_y}");
        }
    }

    /// Rounding costs fill balance: the measured white fraction of the uncoded
    /// carrier should track the closed form in [`Checkerboard::white_fraction`].
    #[test]
    fn rounding_thins_the_uncoded_carrier_as_predicted() {
        for &radius in &[0.0, 0.25, 0.5] {
            let c = Checkerboard::new(9.0, 8)
                .unwrap()
                .with_corner_radius(radius);
            let img = c.render_plain(512, 512, &PatternPose::new(13.7, -4.1, 0.37));
            let mean = img.as_slice().iter().map(|&v| v as Real).sum::<Real>()
                / img.as_slice().len() as Real;
            assert!(
                (mean - c.white_fraction()).abs() < 0.02,
                "radius {radius}: measured {mean}, predicted {}",
                c.white_fraction(),
            );
        }
    }

    /// The reason every supercell edge is odd. An even edge fixes the parity of
    /// both site residues, so every coding site would share one colour and
    /// inverting them would drag the fill off 50/50.
    #[test]
    fn every_supercell_edge_is_odd() {
        for packing in CodePacking::ALL {
            assert_eq!(packing.cell() % 2, 1, "{packing:?} has an even cell");
        }
    }

    /// The direct statement of the same thing: the coding sites of each packing
    /// hit both colours, so the inversions cancel in the fill.
    #[test]
    fn coding_sites_visit_both_colours() {
        for packing in CodePacking::ALL {
            let cell = packing.cell();
            let mut seen = [false, false];
            for first in 0..cell * 2 {
                for second in 0..cell * 2 {
                    if packing.coding_site(first, second).is_some() {
                        seen[(first + second).rem_euclid(2) as usize] = true;
                    }
                }
            }
            assert_eq!(seen, [true, true], "{packing:?} codes on one colour only");
        }
    }

    /// The property the checkerboard is chosen for, across both packings.
    #[test]
    fn both_packings_keep_the_fill_balanced() {
        for packing in CodePacking::ALL {
            for layout in [CodeLayout::Squares, CodeLayout::Diamonds] {
                let c = Checkerboard::new(9.0, 8)
                    .unwrap()
                    .with_code_layout(layout)
                    .with_code_packing(packing);
                let img = c.render(512, 512, &PatternPose::new(13.7, -4.1, 0.37));
                let mean = img.as_slice().iter().map(|&v| v as Real).sum::<Real>()
                    / img.as_slice().len() as Real;
                assert!((mean - 0.5).abs() < 0.02, "{packing:?}/{layout:?}: {mean}");
            }
        }
    }

    /// Two bits per supercell buys density, and — because four sites in 25
    /// squares is sparser than two in nine — costs fewer inverted squares, not
    /// more.
    #[test]
    fn two_bits_is_denser_and_quieter() {
        let one = Checkerboard::new(8.0, 10).unwrap();
        let two = one.clone().with_code_packing(CodePacking::TwoBits);

        assert!(two.code_packing().squares_per_bit() < one.code_packing().squares_per_bit());

        let inverted = |c: &Checkerboard| {
            let n = 200;
            let count = (0..n)
                .flat_map(|i| (0..n).map(move |j| (i, j)))
                .filter(|&(i, j)| c.square_inverted(i, j))
                .count();
            count as Real / (n * n) as Real
        };
        let (one_rate, two_rate) = (inverted(&one), inverted(&two));
        assert!(
            two_rate < one_rate,
            "one bit {one_rate}, two bits {two_rate}"
        );
    }

    /// A supercell's bits must be consecutive in the sequence, or a run of
    /// supercells would not read as a run of the LFSR.
    #[test]
    fn a_supercells_bits_are_consecutive() {
        let c = Checkerboard::new(8.0, 8)
            .unwrap()
            .with_code_packing(CodePacking::TwoBits);
        let packing = c.code_packing();
        let slots: Vec<i64> = packing
            .x_sites()
            .iter()
            .map(|&(first, _)| {
                let (_, slot) = packing.coding_site(first, 0).expect("x site codes");
                packing.bits_per_cell() * first.div_euclid(packing.cell()) + slot
            })
            .collect();
        assert_eq!(slots, vec![0, 1]);
    }

    /// The two axes must never want the same square.
    #[test]
    fn the_axes_do_not_share_a_site() {
        for packing in CodePacking::ALL {
            for x in packing.x_sites() {
                assert!(
                    !packing.y_sites().contains(x),
                    "{packing:?} collides at {x:?}"
                );
            }
        }
    }

    /// One bit per supercell is what upstream rendered before the packing knob
    /// existed, so it must still be the default and still be the 3×3 layout.
    #[test]
    fn one_bit_is_the_default_and_unchanged() {
        let c = Checkerboard::new(8.0, 8).unwrap();
        assert_eq!(c.code_packing(), CodePacking::OneBit);
        assert_eq!(c.code_packing().cell(), CELL);
        assert_eq!(c.range_squares(), CELL * c.code().len() as i64);
        for i in -20..20i64 {
            for j in -20..20i64 {
                let expected = match Checkerboard::coding_axis(i, j) {
                    Some(CodeAxis::X) => c.code_bit(i.div_euclid(CELL)) == 0,
                    Some(CodeAxis::Y) => c.code_bit(j.div_euclid(CELL)) == 0,
                    None => false,
                };
                assert_eq!(c.square_inverted(i, j), expected, "at ({i}, {j})");
            }
        }
    }

    #[test]
    fn diamonds_are_turned_squares() {
        let squares = Checkerboard::new(8.0, 8).unwrap();
        let diamonds = squares.clone().with_code_layout(CodeLayout::Diamonds);
        let (s, c) = (PI / 4.0).sin_cos();
        for (x, y) in [(3.0, 5.0), (-20.5, 7.25), (41.0, -33.0)] {
            assert_eq!(
                diamonds.square_at(x, y),
                squares.square_at(c * x + s * y, c * y - s * x)
            );
        }
    }
}
