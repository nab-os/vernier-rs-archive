//! Rigid board poses and the planar homography they start from.

use nalgebra::{Matrix3, Rotation3, SymmetricEigen, UnitQuaternion, Vector3};
use vernier_core::{Pose, Real};

/// Board to camera: `p_camera = rotation · p_board + translation`, the same
/// pair OpenCV's `solvePnP` returns as `rvec, tvec`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigidPose {
    pub rotation: Rotation3<Real>,
    pub translation: Vector3<Real>,
}

impl RigidPose {
    /// From a Rodrigues rotation vector and a translation, OpenCV's `rvec,
    /// tvec`.
    pub fn from_vectors(rvec: Vector3<Real>, tvec: Vector3<Real>) -> Self {
        Self {
            rotation: Rotation3::from_scaled_axis(rvec),
            translation: tvec,
        }
    }

    /// Rodrigues vector. Taken through a quaternion: `Rotation3`'s own takes an
    /// unclamped arccosine, which is NaN for a rotation a rounding error past
    /// the identity, as a board seen square on can be.
    pub fn rvec(&self) -> Vector3<Real> {
        UnitQuaternion::from_rotation_matrix(&self.rotation).scaled_axis()
    }

    /// Angle of the rotation between two poses.
    pub fn angle_to(&self, other: &RigidPose) -> Real {
        UnitQuaternion::from_rotation_matrix(&self.rotation.rotation_to(&other.rotation)).angle()
    }

    /// Camera coordinates of a point of the board plane.
    pub fn apply(&self, board: [Real; 2]) -> Vector3<Real> {
        self.rotation * Vector3::new(board[0], board[1], 0.0) + self.translation
    }

    /// Camera centre in board coordinates.
    pub fn camera_centre(&self) -> Vector3<Real> {
        -(self.rotation.inverse() * self.translation)
    }

    /// Angle between the optical axis and the board normal: 0 when the camera
    /// faces the board square on.
    pub fn tilt(&self) -> Real {
        let normal = self.rotation * Vector3::z();
        normal.z.abs().clamp(0.0, 1.0).acos()
    }

    /// The same pose in the C++ library's parametrization, `cTp =
    /// transl(0,0,z)·rotz(α)·roty(β)·rotx(γ)·transl(x,y,0)`. `None` when the
    /// board is seen edge on and `x, y, z` stop being separable.
    pub fn to_vernier_pose(&self) -> Option<Pose> {
        // `rotz(α)·roty(β)·rotx(γ)` read back from the matrix entries.
        let r = self.rotation.matrix();
        let beta = (-r[(2, 0)]).clamp(-1.0, 1.0).asin();
        let alpha = r[(1, 0)].atan2(r[(0, 0)]);
        let gamma = r[(2, 1)].atan2(r[(2, 2)]);
        // The translation is `z` along the camera axis plus `(x, y)` along the
        // board's own axes, the first two columns of the rotation.
        let basis = Matrix3::from_columns(&[
            r.column(0).into_owned(),
            r.column(1).into_owned(),
            Vector3::z(),
        ]);
        let xyz = basis.try_inverse()? * self.translation;
        Some(Pose::new_3d(xyz.x, xyz.y, xyz.z, alpha, beta, gamma, 1.0))
    }
}

/// Least-squares homography taking `from` to `to`, normalized DLT. `None` on
/// fewer than four points or a degenerate set.
pub fn homography(from: &[[Real; 2]], to: &[[Real; 2]]) -> Option<Matrix3<Real>> {
    if from.len() < 4 || from.len() != to.len() {
        return None;
    }
    let (normalize_from, from_normalized) = normalize(from)?;
    let (normalize_to, to_normalized) = normalize(to)?;

    // Each correspondence `(x, y) → (u, v)` gives two linear equations
    // `A·h = 0` in the nine entries of `H`, row-major. The solution is the
    // eigenvector of `AᵀA` with the smallest eigenvalue.
    let mut ata = nalgebra::SMatrix::<Real, 9, 9>::zeros();
    for (a, b) in from_normalized.iter().zip(&to_normalized) {
        let (x, y, u, v) = (a[0], a[1], b[0], b[1]);
        let rows = [
            [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u],
            [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, -v],
        ];
        for row in rows {
            for i in 0..9 {
                for j in 0..9 {
                    ata[(i, j)] += row[i] * row[j];
                }
            }
        }
    }
    let eigen = SymmetricEigen::new(ata);
    let smallest = eigen.eigenvalues.imin();
    let h = eigen.eigenvectors.column(smallest);
    let between_normalized = Matrix3::new(h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]);
    let full = normalize_to.try_inverse()? * between_normalized * normalize_from;
    (full[(2, 2)].abs() > 1e-300).then(|| full / full[(2, 2)])
}

/// Hartley normalization: centroid to the origin, mean distance √2, which
/// keeps the DLT well conditioned. Returns the similarity that does it, and
/// the moved points; `None` when all points coincide.
fn normalize(points: &[[Real; 2]]) -> Option<(Matrix3<Real>, Vec<[Real; 2]>)> {
    let n = points.len() as Real;
    let (mx, my) = points
        .iter()
        .fold((0.0, 0.0), |(x, y), p| (x + p[0], y + p[1]));
    let (mx, my) = (mx / n, my / n);
    let spread = points
        .iter()
        .map(|p| ((p[0] - mx).powi(2) + (p[1] - my).powi(2)).sqrt())
        .sum::<Real>()
        / n;
    if spread < 1e-300 {
        return None;
    }
    let s = core::f64::consts::SQRT_2 / spread;
    let t = Matrix3::new(s, 0.0, -s * mx, 0.0, s, -s * my, 0.0, 0.0, 1.0);
    Some((
        t,
        points
            .iter()
            .map(|p| [s * (p[0] - mx), s * (p[1] - my)])
            .collect(),
    ))
}

/// Pose from a homography onto the normalized image plane (`K⁻¹` already
/// applied). The board is put in front of the camera.
///
/// Up to scale, `H = [r1 r2 t]`: the first two columns of the rotation and
/// the translation. The scale comes from `r1`, `r2` being unit vectors, and
/// the nearest true rotation to `[r1 r2 r1×r2]` from its SVD.
pub fn pose_from_homography(h: &Matrix3<Real>) -> Option<RigidPose> {
    let (h1, h2, h3) = (
        h.column(0).into_owned(),
        h.column(1).into_owned(),
        h.column(2).into_owned(),
    );
    let norm = 0.5 * (h1.norm() + h2.norm());
    if norm < 1e-300 {
        return None;
    }
    // `H` is only known up to sign; pick the one with the board at `z > 0`.
    let sign = if h3.z < 0.0 { -1.0 } else { 1.0 };
    let (r1, r2, t) = (h1 * sign / norm, h2 * sign / norm, h3 * sign / norm);
    let approx = Matrix3::from_columns(&[r1, r2, r1.cross(&r2)]);
    let svd = approx.svd(true, true);
    let (u, v_t) = (svd.u?, svd.v_t?);
    let mut rotation = u * v_t;
    // A reflection is the nearest orthogonal matrix; flip it to a rotation.
    if rotation.determinant() < 0.0 {
        let mut u = u;
        u.column_mut(2).neg_mut();
        rotation = u * v_t;
    }
    Some(RigidPose {
        rotation: Rotation3::from_matrix_unchecked(rotation),
        translation: t,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vernier_pose_reproduces_the_transform() {
        let pose = RigidPose::from_vectors(
            Vector3::new(0.3, -0.5, 1.2),
            Vector3::new(12.0, -7.0, 300.0),
        );
        let m = pose
            .to_vernier_pose()
            .unwrap()
            .camera_to_pattern_transform();
        for (x, y) in [(0.0, 0.0), (10.0, -4.0), (-33.0, 21.0)] {
            let expected = pose.apply([x, y]);
            let got = m * nalgebra::Vector4::new(x, y, 0.0, 1.0);
            assert!((got.xyz() - expected).norm() < 1e-9, "{got} vs {expected}");
        }
    }

    #[test]
    fn homography_pose_round_trip() {
        let pose =
            RigidPose::from_vectors(Vector3::new(0.4, 0.2, -0.3), Vector3::new(-5.0, 3.0, 80.0));
        let board: Vec<[Real; 2]> = (0..30)
            .map(|k| [(k % 6) as Real * 4.0 - 10.0, (k / 6) as Real * 3.0 - 6.0])
            .collect();
        let image: Vec<[Real; 2]> = board
            .iter()
            .map(|&b| {
                let p = pose.apply(b);
                [p.x / p.z, p.y / p.z]
            })
            .collect();
        let found = pose_from_homography(&homography(&board, &image).unwrap()).unwrap();
        assert!((found.translation - pose.translation).norm() < 1e-8);
        assert!(found.angle_to(&pose) < 1e-10);
    }
}
