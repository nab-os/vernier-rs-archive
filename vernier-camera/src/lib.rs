//! # vernier-camera
//!
//! Camera calibration and pose (PnP) from the coded checkerboard.
//!
//! - [`camera`]: the pinhole and fisheye models.
//! - [`geometry`]: rigid board poses, and the homographies they start from.
//! - [`target`]: the board, and a renderer for rehearsing without a camera.

pub mod camera;
pub mod geometry;
pub mod target;

pub use camera::{Camera, Model};
pub use geometry::RigidPose;
pub use target::{Scene, Target};
