//! # vernier-camera
//!
//! Camera calibration and pose (PnP) from the coded checkerboard.
//!
//! A checkerboard calibration usually finds corners. This one reads the board
//! the way the rest of vernier does, through the phase of its two carriers, so
//! every pixel of the board takes part and the correspondences come out dense
//! and sub-pixel. The code then says which square is which, so a view need not
//! show the whole board, and a pose comes out in the board's own frame.
//!
//! - [`measure`]: one image to pixel ↔ board correspondences.
//! - [`calibrate`]: intrinsics and distortion from several views, and
//!   [`solve_pnp`](calibrate::solve_pnp) for the pose of one.
//! - [`camera`]: the pinhole and fisheye models.
//! - [`geometry`]: rigid board poses, and the homographies they start from.
//! - [`target`]: the board, and a renderer for rehearsing without a camera.

pub mod calibrate;
pub mod camera;
pub mod geometry;
pub mod measure;
pub mod target;

pub use calibrate::{Calibration, CalibrationError, PnpSolution, ViewFit, calibrate, solve_pnp};
pub use camera::{Camera, Model};
pub use geometry::RigidPose;
pub use measure::{
    Attempt, CpuDemodulator, Field, MeasureError, Peak, PointMatch, Trace, View, demodulated_field,
    demodulated_field_with, measure_view, measure_view_after, measure_view_traced,
    measure_view_traced_with,
};
pub use target::{Scene, Target};
