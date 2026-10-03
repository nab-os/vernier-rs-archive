//! Camera models: a pinhole with Brown-Conrady distortion, and the
//! Kannala-Brandt fisheye model of OpenCV's `fisheye` module.
//!
//! Pixel coordinates follow OpenCV: `x` along a row, `y` down the columns,
//! pixel centres on integers. Camera coordinates are `x` right, `y` down, `z`
//! forward. The coefficient orders match OpenCV too, so a calibration can be
//! handed to `cv::undistort` or `cv::fisheye::undistortImage` as is.

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use vernier_core::Real;

/// Which lens model a [`Camera`] follows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Model {
    /// Distortion `k1, k2, p1, p2, k3` on the normalized image plane.
    Pinhole,
    /// Distortion `k1, k2, k3, k4` on the angle from the optical axis:
    /// `θd = θ(1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸)`. Works past 90° off axis.
    Fisheye,
}

impl Model {
    /// The model named `"pinhole"` or `"fisheye"`, as [`Model::name`] writes it.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "pinhole" => Some(Self::Pinhole),
            "fisheye" => Some(Self::Fisheye),
            _ => None,
        }
    }

    /// Lower-case name, as in a saved calibration.
    pub fn name(self) -> &'static str {
        match self {
            Self::Pinhole => "pinhole",
            Self::Fisheye => "fisheye",
        }
    }

    /// Number of distortion coefficients.
    pub fn distortion_len(self) -> usize {
        match self {
            Self::Pinhole => 5,
            Self::Fisheye => 4,
        }
    }

    /// Number of camera parameters: `fx, fy, cx, cy` and the distortion.
    pub fn parameter_len(self) -> usize {
        4 + self.distortion_len()
    }
}

/// Intrinsics of a camera: image size, focal lengths and principal point in
/// pixels, and the distortion coefficients of its [`Model`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Camera {
    pub model: Model,
    /// Image size, pixels.
    pub width: usize,
    pub height: usize,
    pub fx: Real,
    pub fy: Real,
    pub cx: Real,
    pub cy: Real,
    pub distortion: Vec<Real>,
}

impl Camera {
    /// A camera with all distortion coefficients zero.
    pub fn ideal(
        model: Model,
        width: usize,
        height: usize,
        fx: Real,
        fy: Real,
        cx: Real,
        cy: Real,
    ) -> Self {
        Self {
            model,
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            distortion: vec![0.0; model.distortion_len()],
        }
    }

    /// The parameters as one flat list, `fx, fy, cx, cy` then the
    /// distortion: the layout [`project`] and the solvers work on.
    pub fn parameters(&self) -> Vec<Real> {
        let mut p = vec![self.fx, self.fy, self.cx, self.cy];
        p.extend_from_slice(&self.distortion);
        p
    }

    /// This camera with the parameters replaced by a flat list laid out as
    /// [`Camera::parameters`] returns it.
    pub fn with_parameters(&self, p: &[Real]) -> Self {
        Self {
            fx: p[0],
            fy: p[1],
            cx: p[2],
            cy: p[3],
            distortion: p[4..].to_vec(),
            ..self.clone()
        }
    }

    /// Pixel of a point in camera coordinates, `None` when it cannot be imaged
    /// (behind a pinhole, or straight behind a fisheye).
    pub fn project(&self, point: &Vector3<Real>) -> Option<[Real; 2]> {
        project(self.model, &self.parameters(), point)
    }

    /// A ray through the pixel, in camera coordinates and not normalized:
    /// `(x, y, 1)` for a pinhole, a unit vector for a fisheye. `None` where the
    /// distortion cannot be inverted.
    pub fn unproject(&self, pixel: [Real; 2]) -> Option<Vector3<Real>> {
        // The distorted point on the normalized image plane.
        let xd = (pixel[0] - self.cx) / self.fx;
        let yd = (pixel[1] - self.cy) / self.fy;
        match self.model {
            Model::Pinhole => {
                let (x, y) = undistort_pinhole(&self.distortion, xd, yd)?;
                Some(Vector3::new(x, y, 1.0))
            }
            Model::Fisheye => {
                // The distance from the centre is the distorted angle off axis;
                // the direction around the axis is kept.
                let theta_d = (xd * xd + yd * yd).sqrt();
                if theta_d < 1e-15 {
                    return Some(Vector3::new(0.0, 0.0, 1.0));
                }
                let theta = undistort_angle(&self.distortion, theta_d)?;
                let scale = theta.sin() / theta_d;
                Some(Vector3::new(xd * scale, yd * scale, theta.cos()))
            }
        }
    }
}

/// [`Camera::project`] over a bare parameter slice laid out as
/// [`Camera::parameters`], for the solvers.
pub fn project(model: Model, p: &[Real], point: &Vector3<Real>) -> Option<[Real; 2]> {
    // The distorted point on the normalized image plane, then through `K`.
    let (xd, yd) = match model {
        Model::Pinhole => {
            if point.z <= 0.0 {
                return None;
            }
            distort_pinhole(&p[4..], point.x / point.z, point.y / point.z)
        }
        Model::Fisheye => {
            // Distance from the optical axis; the image point lies the same
            // way round the axis, at the distorted angle from it.
            let r = (point.x * point.x + point.y * point.y).sqrt();
            if r < 1e-300 {
                if point.z <= 0.0 {
                    return None;
                }
                (0.0, 0.0)
            } else {
                let theta = r.atan2(point.z);
                let scale = distort_angle(&p[4..], theta) / r;
                (point.x * scale, point.y * scale)
            }
        }
    };
    Some([p[0] * xd + p[2], p[1] * yd + p[3]])
}

/// Brown-Conrady distortion of a point of the normalized image plane: radial
/// `k1, k2, k3` in `r²`, and tangential `p1, p2` for a lens not quite square
/// to the sensor.
fn distort_pinhole(d: &[Real], x: Real, y: Real) -> (Real, Real) {
    let (k1, k2, p1, p2, k3) = (d[0], d[1], d[2], d[3], d[4]);
    let r2 = x * x + y * y;
    let radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3));
    (
        x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
        y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y,
    )
}

/// Inverse of [`distort_pinhole`]: the undistorted point that distorts to
/// `(xd, yd)`, by Newton on the forward model with a forward-difference
/// Jacobian. Starts from the distorted point, which is the right branch as
/// long as the distortion does not fold the image back on itself. `None` when
/// the Jacobian is singular or Newton has not converged after 50 steps.
fn undistort_pinhole(d: &[Real], xd: Real, yd: Real) -> Option<(Real, Real)> {
    let (mut x, mut y) = (xd, yd);
    let scale = 1.0 + xd.abs() + yd.abs();
    for _ in 0..50 {
        let (dx, dy) = distort_pinhole(d, x, y);
        let (ex, ey) = (dx - xd, dy - yd);
        if ex.abs().max(ey.abs()) < 1e-15 * scale {
            return Some((x, y));
        }
        let h = 1e-7 * (1.0 + x.abs().max(y.abs()));
        let (x_stepped_x, y_stepped_x) = distort_pinhole(d, x + h, y);
        let (x_stepped_y, y_stepped_y) = distort_pinhole(d, x, y + h);
        // `jRC` is the derivative of output `R` along input `C`.
        let (j11, j21, j12, j22) = (
            (x_stepped_x - dx) / h,
            (y_stepped_x - dy) / h,
            (x_stepped_y - dx) / h,
            (y_stepped_y - dy) / h,
        );
        let det = j11 * j22 - j12 * j21;
        if det.abs() < 1e-12 {
            return None;
        }
        // Newton step: subtract J⁻¹·e, with the 2×2 inverse written out.
        x -= (j22 * ex - j12 * ey) / det;
        y -= (j11 * ey - j21 * ex) / det;
    }
    let (dx, dy) = distort_pinhole(d, x, y);
    ((dx - xd).abs().max((dy - yd).abs()) < 1e-9 * scale).then_some((x, y))
}

/// Kannala-Brandt distortion of the angle off axis,
/// `θ(1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸)`.
fn distort_angle(d: &[Real], theta: Real) -> Real {
    let t2 = theta * theta;
    theta * (1.0 + t2 * (d[0] + t2 * (d[1] + t2 * (d[2] + t2 * d[3]))))
}

/// Inverse of [`distort_angle`] on `0..=π`, by Newton from `θ = θd`. `None`
/// where the polynomial stops increasing (the model folds back) or Newton has
/// not converged.
fn undistort_angle(d: &[Real], theta_d: Real) -> Option<Real> {
    let mut theta = theta_d;
    for _ in 0..50 {
        let t2 = theta * theta;
        let error = distort_angle(d, theta) - theta_d;
        // d/dθ of `distort_angle`.
        let slope =
            1.0 + t2 * (3.0 * d[0] + t2 * (5.0 * d[1] + t2 * (7.0 * d[2] + t2 * 9.0 * d[3])));
        if slope <= 0.0 {
            return None;
        }
        let step = error / slope;
        theta = (theta - step).clamp(0.0, core::f64::consts::PI);
        if step.abs() < 1e-15 {
            break;
        }
    }
    ((distort_angle(d, theta) - theta_d).abs() < 1e-9).then_some(theta)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cameras() -> [Camera; 2] {
        let mut pinhole = Camera::ideal(Model::Pinhole, 640, 480, 520.0, 515.0, 322.0, 241.0);
        pinhole.distortion = vec![-0.28, 0.09, 0.001, -0.0007, -0.012];
        let mut fisheye = Camera::ideal(Model::Fisheye, 640, 480, 230.0, 231.0, 318.0, 243.0);
        fisheye.distortion = vec![0.05, -0.01, 0.004, -0.001];
        [pinhole, fisheye]
    }

    #[test]
    fn unproject_inverts_project() {
        for camera in cameras() {
            for (u, v) in [
                (320.0, 240.0),
                (10.0, 15.0),
                (630.0, 470.0),
                (100.5, 400.25),
            ] {
                let ray = camera.unproject([u, v]).expect("invertible");
                let back = camera.project(&(ray * 3.7)).expect("in view");
                assert!(
                    (back[0] - u).abs() < 1e-9 && (back[1] - v).abs() < 1e-9,
                    "{:?}: {back:?}",
                    camera.model
                );
            }
        }
    }

    #[test]
    fn fisheye_sees_past_ninety_degrees() {
        let camera = cameras()[1].clone();
        let pixel = camera
            .project(&Vector3::new(1.0, 0.2, -0.1))
            .expect("in view");
        let ray = camera.unproject(pixel).unwrap();
        assert!(ray.z < 0.0);
    }
}
