//! The calibration target: a coded checkerboard of known square size.

use nalgebra::Vector3;
use rayon::prelude::*;
use vernier_core::Real;
use vernier_patterns::checkerboard::{Checkerboard, CodeLayout, CodePacking};

use crate::camera::Camera;
use crate::geometry::RigidPose;

/// What was printed: the square size and how the code is laid out.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Target {
    /// Side of one square, in whatever unit the poses should come out in.
    pub square: Real,
    /// LFSR order the board was rendered with.
    pub order: u32,
    /// Squares upright or turned 45°.
    pub layout: CodeLayout,
    /// How many code bits per axis each supercell carries.
    pub packing: CodePacking,
}

impl Target {
    /// Upright squares, one bit per axis in each 3×3 supercell: what
    /// `vernier render-checkerboard` prints by default.
    pub fn new(square: Real, order: u32) -> Self {
        Self {
            square,
            order,
            layout: CodeLayout::Squares,
            packing: CodePacking::OneBit,
        }
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

    /// The pattern itself, for rendering. `None` for an LFSR order the
    /// patterns crate does not support.
    pub fn checkerboard(&self) -> Option<Checkerboard> {
        Checkerboard::new(self.square, self.order).map(|c| {
            c.with_code_layout(self.layout)
                .with_code_packing(self.packing)
        })
    }

    /// Code period in squares along either lattice axis. Positions are only
    /// known modulo this.
    pub fn period_squares(&self) -> i64 {
        self.packing.cell() * ((1i64 << self.order) - 1)
    }

    /// Board coordinates of a continuous square-lattice position, integers
    /// being square centres.
    pub fn board_point(&self, i: Real, j: Real) -> [Real; 2] {
        let (x, y) = self
            .layout
            .from_lattice(self.square * (i + 0.5), self.square * (j + 0.5));
        [x, y]
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
        let board = self.target.checkerboard().expect("supported order");
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
