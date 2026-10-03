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
//! - [`camera`]: the pinhole and fisheye models.
//! - [`geometry`]: rigid board poses, and the homographies they start from.
//! - [`target`]: the board, and a renderer for rehearsing without a camera.

pub mod camera;
pub mod geometry;
pub mod measure;
pub mod target;

pub use camera::{Camera, Model};
pub use geometry::RigidPose;
pub use measure::{
    Attempt, Field, MeasureError, Peak, PointMatch, Trace, View, demodulated_field, measure_view,
    measure_view_after, measure_view_traced,
};
pub use target::{Scene, Target};
