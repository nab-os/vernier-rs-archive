use crate::math::{Mat4, Scalar, Vec3, angle_in_pi_pi};
use crate::scalar::Real;
use std::fmt;

/// Pose of a pattern, mirroring the C++ `vernier::Pose`: three translations
/// (`x`, `y`, `z`), three intrinsic rotations (`theta`/alpha about Z, `beta`
/// about Y, `gamma` about X), and a pixel-to-meter scale.
///
/// `theta` is the C++ `alpha` field, renamed to match the rest of the Rust
/// engine. When [`is_3d`](Pose::is_3d) is `false` only `x`, `y`, `theta` matter;
/// the `z`/`beta`/`gamma` fields are ignored by the transform matrices.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Pose {
    /// Translation along X (in the same unit as the pattern period).
    pub x: Real,
    /// Translation along Y (in the same unit as the pattern period).
    pub y: Real,
    /// Translation along Z (in the same unit as the pattern period).
    pub z: Real,
    /// Rotation about the Z axis, in radians (C++ `alpha`).
    pub theta: Real,
    /// Rotation about the intrinsic Y axis, in radians (C++ `beta`).
    pub beta: Real,
    /// Rotation about the intrinsic X axis, in radians (C++ `gamma`).
    pub gamma: Real,
    /// Pixel-to-meter scale factor.
    pub pixel_size: Real,
    /// If `true` the pose is 3D (uses all six DOF); otherwise only `x`, `y`,
    /// `theta` are meaningful.
    pub is_3d: bool,
}

impl Pose {
    /// A null 2D pose at the origin.
    pub const ORIGIN: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        theta: 0.0,
        beta: 0.0,
        gamma: 0.0,
        pixel_size: 1.0,
        is_3d: false,
    };

    /// Creates a 2D pose from `x`, `y`, `theta` with unit pixel size. The
    /// primary constructor used across the engine. Unlike the C++ 2D constructor
    /// it does *not* normalize `theta` — use [`Pose::new_2d`] for that.
    #[inline]
    pub const fn new(x: Real, y: Real, theta: Real) -> Self {
        Self {
            x,
            y,
            z: 0.0,
            theta,
            beta: 0.0,
            gamma: 0.0,
            pixel_size: 1.0,
            is_3d: false,
        }
    }

    /// Creates a 2D pose from `x`, `y`, `alpha` and a pixel size, matching the
    /// C++ `Pose(x, y, alpha, pixelSize)` constructor (angle normalized to
    /// `(-pi, pi]`).
    #[inline]
    pub fn new_2d(x: Real, y: Real, alpha: Real, pixel_size: Real) -> Self {
        Self {
            x,
            y,
            z: 0.0,
            theta: angle_in_pi_pi(alpha as Scalar) as Real,
            beta: 0.0,
            gamma: 0.0,
            pixel_size,
            is_3d: false,
        }
    }

    /// Creates a 3D pose, matching the C++
    /// `Pose(x, y, z, alpha, beta, gamma, pixelSize)` constructor (all angles
    /// normalized to `(-pi, pi]`).
    #[inline]
    pub fn new_3d(
        x: Real,
        y: Real,
        z: Real,
        alpha: Real,
        beta: Real,
        gamma: Real,
        pixel_size: Real,
    ) -> Self {
        Self {
            x,
            y,
            z,
            theta: angle_in_pi_pi(alpha as Scalar) as Real,
            beta: angle_in_pi_pi(beta as Scalar) as Real,
            gamma: angle_in_pi_pi(gamma as Scalar) as Real,
            pixel_size,
            is_3d: true,
        }
    }

    /// Planar distance between the translations of two poses.
    #[inline]
    pub fn translation_distance(&self, other: &Pose) -> Real {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Signed shortest angular difference `self.theta - other.theta`, wrapped
    /// to `(-pi, pi]`.
    #[inline]
    pub fn angle_difference(&self, other: &Pose) -> Real {
        crate::math::angle_in_pi_pi(self.theta - other.theta)
    }

    /// Transformation matrix from the camera to the pattern frame (`cTp`),
    /// mapping a 3D point expressed in the pattern frame into the camera frame.
    ///
    /// For a 3D pose:
    /// `cTp = transl(0,0,z).rotz(alpha).roty(beta).rotx(gamma).transl(x,y,0)`;
    /// for a 2D pose: `cTp = rotz(alpha).transl(x,y,0)`.
    pub fn camera_to_pattern_transform(&self) -> Mat4 {
        let (x, y, z) = (self.x as Scalar, self.y as Scalar, self.z as Scalar);
        let a = self.theta as Scalar;
        if self.is_3d {
            let b = self.beta as Scalar;
            let g = self.gamma as Scalar;
            let (sa, ca) = a.sin_cos();
            let (sb, cb) = b.sin_cos();
            let (sg, cg) = g.sin_cos();
            Mat4::new(
                ca * cb,
                ca * sb * sg - cg * sa,
                sa * sg + ca * cg * sb,
                x * ca * cb - y * (cg * sa - ca * sb * sg),
                cb * sa,
                ca * cg + sa * sb * sg,
                cg * sa * sb - ca * sg,
                y * (ca * cg + sa * sb * sg) + x * cb * sa,
                -sb,
                cb * sg,
                cb * cg,
                z - x * sb + y * cb * sg,
                0.0,
                0.0,
                0.0,
                1.0,
            )
        } else {
            let (sa, ca) = a.sin_cos();
            Mat4::new(
                ca,
                -sa,
                0.0,
                x * ca - y * sa,
                sa,
                ca,
                0.0,
                x * sa + y * ca,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            )
        }
    }

    /// Transformation matrix from the pattern to the camera frame (`pTc`), the
    /// inverse of [`camera_to_pattern_transform`](Pose::camera_to_pattern_transform).
    pub fn pattern_to_camera_transform(&self) -> Mat4 {
        let (x, y, z) = (self.x as Scalar, self.y as Scalar, self.z as Scalar);
        let a = self.theta as Scalar;
        if self.is_3d {
            let b = self.beta as Scalar;
            let g = self.gamma as Scalar;
            let (sa, ca) = a.sin_cos();
            let (sb, cb) = b.sin_cos();
            let (sg, cg) = g.sin_cos();
            Mat4::new(
                ca * cb,
                cb * sa,
                -sb,
                z * sb - x,
                ca * sb * sg - cg * sa,
                ca * cg + sa * sb * sg,
                cb * sg,
                -y - z * cb * sg,
                sa * sg + ca * cg * sb,
                cg * sa * sb - ca * sg,
                cb * cg,
                -z * cb * cg,
                0.0,
                0.0,
                0.0,
                1.0,
            )
        } else {
            let (sa, ca) = a.sin_cos();
            Mat4::new(
                ca, sa, 0.0, -x, -sa, ca, 0.0, -y, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            )
        }
    }

    /// OpenCV-style representation: the `(rvec, tvec)` pair that transforms a 3D
    /// point in the object (pattern) frame into the camera frame, where `rvec`
    /// is the Rodrigues (axis-angle) rotation vector. Equivalent to C++
    /// `getOpenCVRepresentation`.
    pub fn opencv_representation(&self) -> (Vec3, Vec3) {
        let c_t_p = self.camera_to_pattern_transform();
        let rot = c_t_p.fixed_view::<3, 3>(0, 0).into_owned();
        let rvec = nalgebra::Rotation3::from_matrix(&rot).scaled_axis();
        let tvec = Vec3::new(c_t_p[(0, 3)], c_t_p[(1, 3)], c_t_p[(2, 3)]);
        (rvec, tvec)
    }
}

impl fmt::Display for Pose {
    /// Matches the C++ `Pose::toString` output.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_3d {
            write!(
                f,
                "[ x={}; y={}; z={}; alpha={}; beta={}; gamma={}; pixelSize={} ]",
                self.x, self.y, self.z, self.theta, self.beta, self.gamma, self.pixel_size
            )
        } else {
            write!(
                f,
                "[ x={}; y={}; alpha={}; pixelSize={} ]",
                self.x, self.y, self.theta, self.pixel_size
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctp_and_ptc_are_inverses_3d() {
        let p = Pose::new_3d(1.5, -2.0, 3.0, 0.4, -0.3, 0.7, 1.0);
        let prod = p.camera_to_pattern_transform() * p.pattern_to_camera_transform();
        assert!((prod - Mat4::identity()).abs().max() < 1e-12);
    }

    #[test]
    fn ctp_and_ptc_are_inverses_2d() {
        let p = Pose::new_2d(4.0, 5.0, 0.9, 1.0);
        let prod = p.camera_to_pattern_transform() * p.pattern_to_camera_transform();
        assert!((prod - Mat4::identity()).abs().max() < 1e-12);
    }

    #[test]
    fn ctp_2d_matches_closed_form() {
        let p = Pose::new_2d(2.0, 3.0, 0.5, 1.0);
        let m = p.camera_to_pattern_transform();
        let (s, c) = (0.5_f64).sin_cos();
        assert!((m[(0, 0)] - c).abs() < 1e-12);
        assert!((m[(1, 0)] - s).abs() < 1e-12);
        assert!((m[(0, 3)] - (2.0 * c - 3.0 * s)).abs() < 1e-12);
        assert!((m[(1, 3)] - (2.0 * s + 3.0 * c)).abs() < 1e-12);
    }

    #[test]
    fn opencv_translation_matches_ctp_column() {
        let p = Pose::new_3d(1.0, 2.0, 3.0, 0.2, 0.1, -0.4, 1.0);
        let (_rvec, tvec) = p.opencv_representation();
        let c_t_p = p.camera_to_pattern_transform();
        assert!((tvec.x - c_t_p[(0, 3)]).abs() < 1e-12);
        assert!((tvec.y - c_t_p[(1, 3)]).abs() < 1e-12);
        assert!((tvec.z - c_t_p[(2, 3)]).abs() < 1e-12);
    }

    #[test]
    fn display_matches_cpp_format() {
        let p2 = Pose::new_2d(1.0, 2.0, 0.0, 1.0);
        assert_eq!(p2.to_string(), "[ x=1; y=2; alpha=0; pixelSize=1 ]");
    }

    #[test]
    fn new_stays_2d_and_preserves_theta() {
        let p = Pose::new(1.0, 2.0, 10.0);
        assert!(!p.is_3d);
        assert_eq!(p.theta, 10.0);
        assert_eq!(p.pixel_size, 1.0);
    }
}
