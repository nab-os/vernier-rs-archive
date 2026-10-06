pub type Scalar = f64;
pub type Vec2 = nalgebra::Vector2<Scalar>;
pub type Vec3 = nalgebra::Vector3<Scalar>;
pub type Mat3 = nalgebra::Matrix3<Scalar>;
pub type Mat4 = nalgebra::Matrix4<Scalar>;

/// Normalizes an angle (in radians) to the half-open interval `(-pi, pi]`,
/// matching the C++ `vernier::angleInPiPi` helper used by [`crate::Pose`].
pub fn angle_in_pi_pi(angle: Scalar) -> Scalar {
    use std::f64::consts::{PI, TAU};
    if angle > -PI && angle <= PI {
        return angle;
    }
    // A loop subtracting TAU would never end on an infinite angle, nor on one
    // so large that `angle - TAU == angle`; this returns NaN for those.
    let wrapped = PI - (PI - angle).rem_euclid(TAU);
    if wrapped <= -PI { PI } else { wrapped }
}

#[cfg(test)]
mod tests {
    use super::angle_in_pi_pi;
    use std::f64::consts::PI;

    #[test]
    fn wraps_into_half_open_interval() {
        assert_eq!(angle_in_pi_pi(0.5), 0.5);
        assert_eq!(angle_in_pi_pi(PI), PI);
        assert_eq!(angle_in_pi_pi(-PI), PI);
        assert!((angle_in_pi_pi(3.0 * PI) - PI).abs() < 1e-12);
        assert!((angle_in_pi_pi(-2.5 * PI) + 0.5 * PI).abs() < 1e-12);
        assert!((angle_in_pi_pi(1e6) - (1e6f64).sin().atan2((1e6f64).cos())).abs() < 1e-9);
    }

    #[test]
    fn terminates_on_non_finite_and_huge_angles() {
        assert!(angle_in_pi_pi(f64::INFINITY).is_nan());
        assert!(angle_in_pi_pi(f64::NAN).is_nan());
        let huge = angle_in_pi_pi(1e300);
        assert!(huge > -PI && huge <= PI);
    }
}
