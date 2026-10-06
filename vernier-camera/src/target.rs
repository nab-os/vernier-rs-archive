//! The calibration target: a coded checkerboard of known square size, or a
//! megarena of known dot pitch.

use nalgebra::Vector3;
use rayon::prelude::*;
use vernier_core::Real;
use vernier_patterns::checkerboard::{Checkerboard, CodeLayout, CodePacking};
use vernier_patterns::megarena::Megarena;

use crate::camera::Camera;
use crate::geometry::RigidPose;

/// Which pattern a [`Target`] is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PatternKind {
    /// The coded checkerboard of `vernier render-pattern`.
    #[default]
    Checkerboard,
    /// The megarena dot grid of `vernier render-pattern`: carriers
    /// along the dot rows and columns, three dots per code bit, one corner dot
    /// of every 3×3 cell left out.
    Megarena,
}

/// What was printed: the square size and how the code is laid out.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Target {
    /// Side of one square, in whatever unit the poses should come out in. For
    /// a megarena, the dot pitch.
    pub square: Real,
    /// LFSR order the board was rendered with.
    pub order: u32,
    /// Squares upright or turned 45°. Checkerboard only.
    pub layout: CodeLayout,
    /// How many code bits per axis each supercell carries. Checkerboard only.
    pub packing: CodePacking,
    pub kind: PatternKind,
}

impl Target {
    /// Upright squares, one bit per axis in each 3×3 supercell: what
    /// `vernier render-pattern` prints by default.
    pub fn new(square: Real, order: u32) -> Self {
        Self {
            square,
            order,
            layout: CodeLayout::Squares,
            packing: CodePacking::OneBit,
            kind: PatternKind::Checkerboard,
        }
    }

    /// A megarena of dot pitch `pitch` (in the unit poses should come out in)
    /// and LFSR order `order`, as `vernier render-pattern` draws it:
    /// the code starts at the dot on the origin.
    pub fn megarena(pitch: Real, order: u32) -> Self {
        Self {
            kind: PatternKind::Megarena,
            ..Self::new(pitch, order)
        }
    }

    pub fn is_megarena(&self) -> bool {
        self.kind == PatternKind::Megarena
    }

    /// This target with another code layout.
    pub fn with_layout(mut self, layout: CodeLayout) -> Self {
        self.layout = layout;
        self
    }

    /// This target with another code packing.
    pub fn with_packing(mut self, packing: CodePacking) -> Self {
        self.packing = packing;
        self
    }

    /// The checkerboard itself, for rendering. `None` for a megarena or for
    /// an LFSR order the patterns crate does not support.
    pub fn checkerboard(&self) -> Option<Checkerboard> {
        if self.is_megarena() {
            return None;
        }
        Checkerboard::new(self.square, self.order).map(|c| {
            c.with_code_layout(self.layout)
                .with_code_packing(self.packing)
        })
    }

    /// The megarena itself, for rendering. `None` for a checkerboard or for
    /// an LFSR order the patterns crate does not support.
    pub fn megarena_pattern(&self) -> Option<Megarena> {
        if !self.is_megarena() {
            return None;
        }
        Megarena::new(self.square, self.order)
    }

    /// The printed pattern, `None` for an unsupported LFSR order.
    pub(crate) fn printed(&self) -> Option<Printed> {
        match self.kind {
            PatternKind::Checkerboard => self.checkerboard().map(Printed::Checkerboard),
            PatternKind::Megarena => self.megarena_pattern().map(Printed::Megarena),
        }
    }

    /// Code period along either lattice axis, in squares (dots for a
    /// megarena). Positions are only known modulo this.
    pub fn period_squares(&self) -> i64 {
        match self.kind {
            PatternKind::Checkerboard => self.packing.cell() * ((1i64 << self.order) - 1),
            // Three dots per bit.
            PatternKind::Megarena => 3 * ((1i64 << self.order) - 1),
        }
    }

    /// Board coordinates of a continuous lattice position, integers being
    /// square centres, or dot centres for a megarena.
    pub fn board_point(&self, i: Real, j: Real) -> [Real; 2] {
        if self.is_megarena() {
            return [self.square * i, self.square * j];
        }
        let (x, y) = self
            .layout
            .from_lattice(self.square * (i + 0.5), self.square * (j + 0.5));
        [x, y]
    }
}

/// A target's pattern, for its grey levels.
pub(crate) enum Printed {
    Checkerboard(Checkerboard),
    Megarena(Megarena),
}

impl Printed {
    /// Grey level at a board point, `0.0..=1.0`.
    pub(crate) fn intensity_at(&self, x: Real, y: Real) -> Real {
        match self {
            Self::Checkerboard(c) => c.intensity_at(x, y),
            Self::Megarena(m) => m.intensity_at(x, y),
        }
    }
}

/// A board of finite size, imaged through a camera, for tests and for
/// rehearsing a calibration without a camera.
pub struct Scene<'a> {
    pub camera: &'a Camera,
    pub target: &'a Target,
    /// Half extents of the printed board, in board units, around its origin.
    pub half_size: [Real; 2],
    /// Grey level off the board.
    pub background: Real,
    /// Sub-samples per pixel edge.
    pub supersample: usize,
}

impl Scene<'_> {
    /// Renders the board at `pose`, `0.0..=1.0` row-major. Black squares are
    /// drawn at `0.1` and white at `0.9`, a printed board's dynamic range.
    /// Panics on an LFSR order the patterns crate does not support.
    pub fn render(&self, pose: &RigidPose) -> Vec<f32> {
        let board = self.target.printed().expect("supported order");
        let (w, h) = (self.camera.width, self.camera.height);
        let n = self.supersample.max(1);
        let board_normal = pose.rotation * Vector3::z();
        let camera_to_board = pose.rotation.inverse();
        // Grey level at one image point: cast its ray onto the board plane.
        let sample = |u: Real, v: Real| -> Real {
            let Some(ray) = self.camera.unproject([u, v]) else {
                return self.background;
            };
            let along = board_normal.dot(&ray);
            if along.abs() < 1e-12 {
                return self.background;
            }
            // Ray parameter where it meets the plane through the board origin.
            let s = board_normal.dot(&pose.translation) / along;
            if s <= 0.0 {
                return self.background;
            }
            let p = camera_to_board * (ray * s - pose.translation);
            if p.x.abs() > self.half_size[0] || p.y.abs() > self.half_size[1] {
                return self.background;
            }
            0.1 + 0.8 * board.intensity_at(p.x, p.y)
        };
        (0..w * h)
            .into_par_iter()
            .map(|pixel| {
                let (x, y) = ((pixel % w) as Real, (pixel / w) as Real);
                // Mean over an `n × n` grid of sub-pixel centres.
                let mut sum = 0.0;
                for a in 0..n {
                    for b in 0..n {
                        let du = (a as Real + 0.5) / n as Real - 0.5;
                        let dv = (b as Real + 0.5) / n as Real - 0.5;
                        sum += sample(x + du, y + dv);
                    }
                }
                (sum / (n * n) as Real) as f32
            })
            .collect()
    }
}
