//! Intrinsics from several views of the board, and the pose of one view.
//!
//! Both start in closed form from planar homographies and finish with
//! Levenberg-Marquardt on the reprojection error of every correspondence, the
//! way OpenCV's `calibrateCamera` and `solvePnP` do. The difference is the
//! input: a phase-measured view carries hundreds to thousands of points, so the
//! fit is well past determined even for a fisheye's four coefficients.

use nalgebra::{DMatrix, DVector, Rotation3, Vector3};
use rayon::prelude::*;
use vernier_core::Real;

use crate::camera::{self, Camera, Model};
use crate::geometry::{RigidPose, homography, pose_from_homography};
use crate::measure::{PointMatch, View};

/// A point is rejected beyond this many standard deviations of the residual.
const OUTLIER_SIGMAS: Real = 5.0;

/// Never reject a point closer than this, in pixels: on clean synthetic views
/// the spread is tiny and a few sigmas would cut into the inliers.
const MIN_OUTLIER_PX: Real = 0.05;

/// Rounds of fit, reject, refit.
const OUTLIER_ROUNDS: usize = 4;

/// A view left with fewer inliers than this cannot pin down its six pose
/// parameters.
const MIN_INLIERS_PER_VIEW: usize = 6;

/// Levenberg-Marquardt iterations per fit.
const MAX_ITERATIONS: usize = 200;

/// A fisheye starts from a pinhole fitted to the centre of the frame, within
/// this fraction of the shorter side from the middle.
const FISHEYE_CENTRE: Real = 0.3;

/// A view with fewer points than this near the centre gives no focal estimate.
const MIN_FOCAL_POINTS: usize = 20;

/// Finite-difference step on each component of the Rodrigues vector, radians.
const ROTATION_STEP: Real = 1e-7;

/// The fit of one view: its pose and how well the points agree with it.
#[derive(Clone, Debug)]
pub struct ViewFit {
    pub pose: RigidPose,
    /// Reprojection error over the points kept, in pixels.
    pub rms: Real,
    /// Points kept.
    pub used: usize,
    /// Points dropped as outliers.
    pub rejected: usize,
    /// Per point of the view, projected minus measured, in pixels; NaN where
    /// the point does not project.
    pub residuals: Vec<[Real; 2]>,
    /// Per point of the view, whether the fit kept it.
    pub inliers: Vec<bool>,
}

/// A calibrated camera, with the fit of every view that went into it.
#[derive(Clone, Debug)]
pub struct Calibration {
    pub camera: Camera,
    /// Over every point kept, in pixels.
    pub rms: Real,
    pub views: Vec<ViewFit>,
}

/// The pose of one view found by [`solve_pnp`].
pub type PnpSolution = ViewFit;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CalibrationError {
    TooFewViews(usize),
    /// Views of different image sizes.
    SizeMismatch,
    /// No usable starting pose for this view.
    Initialization(usize),
    /// The fit left too few points to trust.
    TooFewPoints,
}

impl std::fmt::Display for CalibrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooFewViews(n) => write!(
                f,
                "{n} view(s); calibration needs at least 2, and 10 or more is better"
            ),
            Self::SizeMismatch => write!(f, "the views do not all have the same image size"),
            Self::Initialization(v) => write!(f, "could not find a starting pose for view {v}"),
            Self::TooFewPoints => write!(f, "too few points survived the fit"),
        }
    }
}

impl std::error::Error for CalibrationError {}

/// The six numbers of a board pose as the solver moves them: the Rodrigues
/// rotation vector, then the translation.
type PoseParameters = [Real; 6];

/// Calibrates a camera from views of the board taken at varied angles.
pub fn calibrate(views: &[View], model: Model) -> Result<Calibration, CalibrationError> {
    if views.len() < 2 {
        return Err(CalibrationError::TooFewViews(views.len()));
    }
    let (width, height) = (views[0].width, views[0].height);
    if views.iter().any(|v| v.width != width || v.height != height) {
        return Err(CalibrationError::SizeMismatch);
    }

    // Starting camera: no distortion, principal point at the image centre.
    let (fx, fy) = initial_focal(views, model);
    let start = Camera::ideal(
        model,
        width,
        height,
        fx,
        fy,
        (width as Real - 1.0) / 2.0,
        (height as Real - 1.0) / 2.0,
    );
    let mut poses = Vec::with_capacity(views.len());
    for (index, view) in views.iter().enumerate() {
        poses.push(
            initial_pose(&start, &view.points).ok_or(CalibrationError::Initialization(index))?,
        );
    }

    let mut problem = Problem::new(model, views.iter().map(|v| v.points.as_slice()).collect());
    let mut intrinsics = start.parameters();
    // Distortion only once the linear part has settled: a fisheye starts far
    // from its pinhole guess. The first four parameters are `fx, fy, cx, cy`.
    problem.free = (0..model.parameter_len()).map(|i| i < 4).collect();
    problem.refine(&mut intrinsics, &mut poses);
    problem.free = vec![true; model.parameter_len()];
    problem.fit_with_rejection(&mut intrinsics, &mut poses)?;

    let camera = start.with_parameters(&intrinsics);
    let fits = problem.view_fits(&intrinsics, &poses);
    Ok(Calibration {
        camera,
        rms: overall_rms(&fits),
        views: fits,
    })
}

/// Pose of the board in one view, the camera being known. The pose is in the
/// board's own frame only when the view's code was read
/// ([`View::is_absolute`]).
pub fn solve_pnp(camera: &Camera, view: &View) -> Result<PnpSolution, CalibrationError> {
    let mut poses =
        vec![initial_pose(camera, &view.points).ok_or(CalibrationError::Initialization(0))?];
    let mut problem = Problem::new(camera.model, vec![&view.points]);
    problem.free = vec![false; camera.model.parameter_len()];
    let mut intrinsics = camera.parameters();
    problem.fit_with_rejection(&mut intrinsics, &mut poses)?;
    Ok(problem.view_fits(&intrinsics, &poses).remove(0))
}

/// RMS reprojection error over the points kept in every view, each view
/// weighted by its number of points.
fn overall_rms(fits: &[ViewFit]) -> Real {
    let (sum_squares, count) = fits.iter().fold((0.0, 0), |(sum, count), fit| {
        (sum + fit.rms * fit.rms * fit.used as Real, count + fit.used)
    });
    (sum_squares / count as Real).sqrt()
}

/// One linear equation `coeffs · (1/fx², 1/fy²) = rhs`, scaled so that
/// `coeffs` is a unit vector and every view weighs alike.
struct FocalEquation {
    coeffs: [Real; 2],
    rhs: Real,
}

/// Focal lengths from the homographies, the principal point held at the
/// centre: each tilted view gives two linear equations in `1/fx²` and `1/fy²`
/// (Zhang's constraints with the principal point known, as OpenCV starts).
/// Falls back to one focal for both axes, then to a guess from the image size.
fn initial_focal(views: &[View], model: Model) -> (Real, Real) {
    let (w, h) = (views[0].width as Real, views[0].height as Real);
    let centre = [(w - 1.0) / 2.0, (h - 1.0) / 2.0];
    let reach = FISHEYE_CENTRE * w.min(h);
    let equations: Vec<FocalEquation> = views
        .iter()
        .flat_map(|view| focal_equations(view, model, centre, reach))
        .collect();

    if let Some(focals) = solve_separate_focals(&equations) {
        return focals;
    }
    if let Some(f) = solve_shared_focal(&equations) {
        return (f, f);
    }
    let fallback = match model {
        Model::Pinhole => w.max(h),
        Model::Fisheye => w.min(h) / 2.0,
    };
    (fallback, fallback)
}

/// Zhang's two constraints from one view, none when the view has too few
/// points or no homography.
///
/// With `H = [h1 h2 h3]` taking the board onto pixels relative to the centre,
/// `h1` and `h2` are the board axes seen through `K = diag(fx, fy, 1)`. Being
/// orthogonal and of equal length once `K` is undone gives, with `B = K⁻ᵀK⁻¹ =
/// diag(1/fx², 1/fy², 1)`, `h1ᵀBh2 = 0` and `h1ᵀBh1 = h2ᵀBh2`. A fisheye only
/// looks like a pinhole near the centre, so only the points there are used.
fn focal_equations(
    view: &View,
    model: Model,
    centre: [Real; 2],
    reach: Real,
) -> Vec<FocalEquation> {
    let [cx, cy] = centre;
    let points: Vec<&PointMatch> = view
        .points
        .iter()
        .filter(|p| model == Model::Pinhole || (p.pixel[0] - cx).hypot(p.pixel[1] - cy) < reach)
        .collect();
    if points.len() < MIN_FOCAL_POINTS {
        return Vec::new();
    }
    let board: Vec<[Real; 2]> = points.iter().map(|p| p.board).collect();
    let centred_pixels: Vec<[Real; 2]> = points
        .iter()
        .map(|p| [p.pixel[0] - cx, p.pixel[1] - cy])
        .collect();
    let Some(board_to_pixels) = homography(&board, &centred_pixels) else {
        return Vec::new();
    };
    let (h1, h2) = (board_to_pixels.column(0), board_to_pixels.column(1));
    let orthogonal = ([h1[0] * h2[0], h1[1] * h2[1]], -h1[2] * h2[2]);
    let equal_length = (
        [h1[0] * h1[0] - h2[0] * h2[0], h1[1] * h1[1] - h2[1] * h2[1]],
        -(h1[2] * h1[2] - h2[2] * h2[2]),
    );
    [orthogonal, equal_length]
        .into_iter()
        .filter_map(|(coeffs, rhs)| {
            let norm = coeffs[0].hypot(coeffs[1]);
            (norm > 0.0).then(|| FocalEquation {
                coeffs: [coeffs[0] / norm, coeffs[1] / norm],
                rhs: rhs / norm,
            })
        })
        .collect()
}

/// Least squares for `(1/fx², 1/fy²)`. `None` when the system is singular, a
/// solution is not positive, or the two focals differ by a factor of two or
/// more (no real camera has pixels that oblong).
fn solve_separate_focals(equations: &[FocalEquation]) -> Option<(Real, Real)> {
    // The 2×2 normal equations AᵀA·x = Aᵀb.
    let (mut ata00, mut ata01, mut ata11, mut atb0, mut atb1) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for FocalEquation { coeffs: c, rhs: r } in equations {
        ata00 += c[0] * c[0];
        ata01 += c[0] * c[1];
        ata11 += c[1] * c[1];
        atb0 += c[0] * r;
        atb1 += c[1] * r;
    }
    let det = ata00 * ata11 - ata01 * ata01;
    if det.abs() <= 1e-12 * (ata00 * ata11).max(1e-300) {
        return None;
    }
    let inverse_fx2 = (atb0 * ata11 - atb1 * ata01) / det;
    let inverse_fy2 = (ata00 * atb1 - ata01 * atb0) / det;
    if !(inverse_fx2 > 0.0 && inverse_fy2 > 0.0) {
        return None;
    }
    let (fx, fy) = (1.0 / inverse_fx2.sqrt(), 1.0 / inverse_fy2.sqrt());
    (0.5..2.0).contains(&(fx / fy)).then_some((fx, fy))
}

/// One focal for both axes: the same equations with `1/fx² = 1/fy²`.
fn solve_shared_focal(equations: &[FocalEquation]) -> Option<Real> {
    let (num, den) = equations.iter().fold((0.0, 0.0), |(num, den), e| {
        let c = e.coeffs;
        (num + (c[0] + c[1]) * e.rhs, den + (c[0] + c[1]).powi(2))
    });
    (den > 0.0 && num / den > 0.0).then(|| 1.0 / (num / den).sqrt())
}

/// Board pose from the homography onto the normalized image plane, with the
/// points unprojected through `camera`. Points whose ray is more than about
/// 84° off axis are left out: they have no place on that plane.
fn initial_pose(camera: &Camera, points: &[PointMatch]) -> Option<PoseParameters> {
    let (mut board, mut plane) = (Vec::new(), Vec::new());
    for p in points {
        let Some(ray) = camera.unproject(p.pixel) else {
            continue;
        };
        if ray.z <= 0.1 * ray.norm() {
            continue;
        }
        board.push(p.board);
        plane.push([ray.x / ray.z, ray.y / ray.z]);
    }
    let pose = pose_from_homography(&homography(&board, &plane)?)?;
    let r = pose.rvec();
    let t = pose.translation;
    Some([r.x, r.y, r.z, t.x, t.y, t.z])
}

fn rigid_pose(pose: &PoseParameters) -> RigidPose {
    RigidPose::from_vectors(
        Vector3::new(pose[0], pose[1], pose[2]),
        Vector3::new(pose[3], pose[4], pose[5]),
    )
}

/// One view's share of the Gauss-Newton normal equations `JᵀJ·δ = -Jᵀr`.
///
/// A view's residuals depend on the free intrinsics, shared by every view, and
/// on its own six pose parameters only. Its Jacobian is thus two blocks, `Ji`
/// for the intrinsics and `Jp` for its pose, and its share of `JᵀJ` three.
struct NormalBlocks {
    /// `JiᵀJi`, added up over the views.
    intrinsic_intrinsic: DMatrix<Real>,
    /// `JiᵀJp`, off the diagonal.
    intrinsic_pose: DMatrix<Real>,
    /// `JpᵀJp`, this view's own diagonal block.
    pose_pose: DMatrix<Real>,
    /// `Jiᵀr`.
    intrinsic_gradient: DVector<Real>,
    /// `Jpᵀr`.
    pose_gradient: DVector<Real>,
}

impl NormalBlocks {
    fn zeros(free_len: usize) -> Self {
        Self {
            intrinsic_intrinsic: DMatrix::zeros(free_len, free_len),
            intrinsic_pose: DMatrix::zeros(free_len, 6),
            pose_pose: DMatrix::zeros(6, 6),
            intrinsic_gradient: DVector::zeros(free_len),
            pose_gradient: DVector::zeros(6),
        }
    }

    /// Adds the two rows (x and y residual) of one point.
    fn add_point(
        &mut self,
        d_intrinsics: &[[Real; 2]],
        d_pose: &[[Real; 2]; 6],
        residual: [Real; 2],
    ) {
        for k in 0..2 {
            for (a, da) in d_intrinsics.iter().enumerate() {
                self.intrinsic_gradient[a] += da[k] * residual[k];
                for (b, db) in d_intrinsics.iter().enumerate() {
                    self.intrinsic_intrinsic[(a, b)] += da[k] * db[k];
                }
                for (b, db) in d_pose.iter().enumerate() {
                    self.intrinsic_pose[(a, b)] += da[k] * db[k];
                }
            }
            for (a, da) in d_pose.iter().enumerate() {
                self.pose_gradient[a] += da[k] * residual[k];
                for (b, db) in d_pose.iter().enumerate() {
                    self.pose_pose[(a, b)] += da[k] * db[k];
                }
            }
        }
    }
}

/// Finite-difference steps for one view, and the stepped rotations, worked
/// out once and shared by every point of the view.
struct Perturbations {
    pose: RigidPose,
    /// Per axis, the rotation with that component of the Rodrigues vector
    /// stepped by `+ROTATION_STEP` and `-ROTATION_STEP`.
    turned: [[Rotation3<Real>; 2]; 3],
    translation_step: Real,
    /// Per free intrinsic.
    intrinsic_steps: Vec<Real>,
}

impl Perturbations {
    fn new(pose: &PoseParameters, free: &[usize]) -> Self {
        let rigid = rigid_pose(pose);
        let turned = std::array::from_fn(|axis| {
            let mut plus = *pose;
            let mut minus = *pose;
            plus[axis] += ROTATION_STEP;
            minus[axis] -= ROTATION_STEP;
            [rigid_pose(&plus).rotation, rigid_pose(&minus).rotation]
        });
        let translation_step = 1e-7 * (1.0 + rigid.translation.norm());
        // `fx, fy, cx, cy` are in pixels, hundreds of them; the distortion
        // coefficients are of order one or less.
        let intrinsic_steps = free
            .iter()
            .map(|&i| if i < 4 { 1e-4 } else { 1e-7 })
            .collect();
        Self {
            pose: rigid,
            turned,
            translation_step,
            intrinsic_steps,
        }
    }
}

/// Central difference of two residuals, `None` if either point did not
/// project.
fn central_difference(
    plus: Option<[Real; 2]>,
    minus: Option<[Real; 2]>,
    step: Real,
) -> Option<[Real; 2]> {
    let (a, b) = (plus?, minus?);
    Some([(a[0] - b[0]) / (2.0 * step), (a[1] - b[1]) / (2.0 * step)])
}

/// The least-squares problem over a set of views: which points count, and
/// which intrinsics may move. Poses always move.
struct Problem<'a> {
    model: Model,
    views: Vec<&'a [PointMatch]>,
    /// Per view, per point, whether it is still in the fit.
    inliers: Vec<Vec<bool>>,
    /// Per camera parameter, whether the fit may move it.
    free: Vec<bool>,
}

impl<'a> Problem<'a> {
    fn new(model: Model, views: Vec<&'a [PointMatch]>) -> Self {
        let inliers = views.iter().map(|v| vec![true; v.len()]).collect();
        Self {
            model,
            views,
            inliers,
            free: vec![true; model.parameter_len()],
        }
    }

    /// Projected minus measured pixel of one correspondence, `None` where the
    /// board point does not project.
    fn residual(
        &self,
        intrinsics: &[Real],
        rotation: &Rotation3<Real>,
        t: &Vector3<Real>,
        p: &PointMatch,
    ) -> Option<[Real; 2]> {
        let point = rotation * Vector3::new(p.board[0], p.board[1], 0.0) + t;
        let q = camera::project(self.model, intrinsics, &point)?;
        Some([q[0] - p.pixel[0], q[1] - p.pixel[1]])
    }

    /// The inlier points of view `v`.
    fn inlier_points(&self, v: usize) -> impl Iterator<Item = &PointMatch> {
        self.views[v]
            .iter()
            .zip(&self.inliers[v])
            .filter(|&(_, &inlier)| inlier)
            .map(|(p, _)| p)
    }

    /// Sum of squared residuals over the inliers. A point that cannot be
    /// projected counts as far off, so no step can win by hiding points.
    fn cost(&self, intrinsics: &[Real], poses: &[PoseParameters]) -> Real {
        (0..self.views.len())
            .into_par_iter()
            .map(|v| {
                let pose = rigid_pose(&poses[v]);
                self.inlier_points(v)
                    .map(|p| {
                        match self.residual(intrinsics, &pose.rotation, &pose.translation, p) {
                            Some(r) => r[0] * r[0] + r[1] * r[1],
                            None => 1e12,
                        }
                    })
                    .sum::<Real>()
            })
            .sum()
    }

    /// Residual of one point and its derivatives by central differences, into
    /// `d_intrinsics` (one per free intrinsic) and `d_pose` (rotation vector,
    /// then translation). `None` when the point or a stepped copy of it does
    /// not project.
    fn point_jacobian(
        &self,
        intrinsics: &[Real],
        free: &[usize],
        steps: &Perturbations,
        p: &PointMatch,
        d_intrinsics: &mut [[Real; 2]],
        d_pose: &mut [[Real; 2]; 6],
    ) -> Option<[Real; 2]> {
        let RigidPose {
            rotation,
            translation,
        } = &steps.pose;
        let residual = self.residual(intrinsics, rotation, translation, p)?;

        let mut shifted = intrinsics.to_vec();
        for (column, (&i, &step)) in free.iter().zip(&steps.intrinsic_steps).enumerate() {
            shifted[i] = intrinsics[i] + step;
            let plus = self.residual(&shifted, rotation, translation, p);
            shifted[i] = intrinsics[i] - step;
            let minus = self.residual(&shifted, rotation, translation, p);
            shifted[i] = intrinsics[i];
            d_intrinsics[column] = central_difference(plus, minus, step)?;
        }
        for axis in 0..3 {
            let [turned_plus, turned_minus] = &steps.turned[axis];
            let plus = self.residual(intrinsics, turned_plus, translation, p);
            let minus = self.residual(intrinsics, turned_minus, translation, p);
            d_pose[axis] = central_difference(plus, minus, ROTATION_STEP)?;

            let mut moved_plus = *translation;
            let mut moved_minus = *translation;
            moved_plus[axis] += steps.translation_step;
            moved_minus[axis] -= steps.translation_step;
            let plus = self.residual(intrinsics, rotation, &moved_plus, p);
            let minus = self.residual(intrinsics, rotation, &moved_minus, p);
            d_pose[3 + axis] = central_difference(plus, minus, steps.translation_step)?;
        }
        Some(residual)
    }

    /// View `v`'s share of the normal equations, over its inliers.
    fn normal_blocks(
        &self,
        v: usize,
        intrinsics: &[Real],
        pose: &PoseParameters,
        free: &[usize],
    ) -> NormalBlocks {
        let mut blocks = NormalBlocks::zeros(free.len());
        let steps = Perturbations::new(pose, free);
        let mut d_intrinsics = vec![[0.0; 2]; free.len()];
        let mut d_pose = [[0.0; 2]; 6];
        for p in self.inlier_points(v) {
            if let Some(residual) =
                self.point_jacobian(intrinsics, free, &steps, p, &mut d_intrinsics, &mut d_pose)
            {
                blocks.add_point(&d_intrinsics, &d_pose, residual);
            }
        }
        blocks
    }

    /// The full normal equations `(JᵀJ, Jᵀr)`. Unknowns are ordered free
    /// intrinsics first, then six per view; `JᵀJ` is an arrow: a dense
    /// intrinsic row and column, and one 6×6 block per view on the diagonal.
    fn normal_equations(
        &self,
        intrinsics: &[Real],
        poses: &[PoseParameters],
        free: &[usize],
    ) -> (DMatrix<Real>, DVector<Real>) {
        let n = free.len();
        let size = n + 6 * poses.len();
        let blocks: Vec<NormalBlocks> = (0..poses.len())
            .into_par_iter()
            .map(|v| self.normal_blocks(v, intrinsics, &poses[v], free))
            .collect();
        let mut jtj = DMatrix::<Real>::zeros(size, size);
        let mut jtr = DVector::<Real>::zeros(size);
        for (v, view) in blocks.iter().enumerate() {
            let o = n + 6 * v;
            let mut intrinsic_corner = jtj.view_mut((0, 0), (n, n));
            intrinsic_corner += &view.intrinsic_intrinsic;
            jtj.view_mut((0, o), (n, 6)).copy_from(&view.intrinsic_pose);
            jtj.view_mut((o, 0), (6, n))
                .copy_from(&view.intrinsic_pose.transpose());
            jtj.view_mut((o, o), (6, 6)).copy_from(&view.pose_pose);
            let mut intrinsic_rows = jtr.rows_mut(0, n);
            intrinsic_rows += &view.intrinsic_gradient;
            jtr.rows_mut(o, 6).copy_from(&view.pose_gradient);
        }
        (jtj, jtr)
    }

    /// One Levenberg-Marquardt step: solves `(JᵀJ + λ·diag(JᵀJ))·δ = -Jᵀr`,
    /// raising `λ` tenfold (towards small gradient-descent steps) until the
    /// step lowers the cost, then lowering it tenfold for the next. Scaling the
    /// damping by the diagonal keeps it fair between parameters as unlike as a
    /// focal length in pixels and a rotation in radians. Applies the step and
    /// returns the new cost, or `None` when no damping helps.
    #[allow(clippy::too_many_arguments)]
    fn damped_step(
        &self,
        jtj: &DMatrix<Real>,
        jtr: &DVector<Real>,
        free: &[usize],
        intrinsics: &mut [Real],
        poses: &mut [PoseParameters],
        cost: Real,
        lambda: &mut Real,
    ) -> Option<Real> {
        let n = free.len();
        while *lambda < 1e16 {
            let mut damped = jtj.clone();
            for i in 0..jtj.nrows() {
                damped[(i, i)] += *lambda * jtj[(i, i)].max(1e-12);
            }
            let Some(cholesky) = damped.cholesky() else {
                *lambda *= 10.0;
                continue;
            };
            let step = cholesky.solve(&(-jtr));
            let mut trial_intrinsics = intrinsics.to_vec();
            for (k, &i) in free.iter().enumerate() {
                trial_intrinsics[i] += step[k];
            }
            let trial_poses: Vec<PoseParameters> = poses
                .iter()
                .enumerate()
                .map(|(v, pose)| std::array::from_fn(|k| pose[k] + step[n + 6 * v + k]))
                .collect();
            let trial_cost = self.cost(&trial_intrinsics, &trial_poses);
            if trial_cost < cost {
                intrinsics.copy_from_slice(&trial_intrinsics);
                poses.copy_from_slice(&trial_poses);
                *lambda = (*lambda / 10.0).max(1e-12);
                return Some(trial_cost);
            }
            *lambda *= 10.0;
        }
        None
    }

    /// Levenberg-Marquardt over the free intrinsics and every pose, until a
    /// step no longer lowers the cost by a relative 1e-12. Returns the final
    /// cost.
    fn refine(&self, intrinsics: &mut [Real], poses: &mut [PoseParameters]) -> Real {
        let free: Vec<usize> = (0..intrinsics.len()).filter(|&i| self.free[i]).collect();
        let mut cost = self.cost(intrinsics, poses);
        let mut lambda = 1e-3;

        for _ in 0..MAX_ITERATIONS {
            let (jtj, jtr) = self.normal_equations(intrinsics, poses, &free);
            let Some(new_cost) =
                self.damped_step(&jtj, &jtr, &free, intrinsics, poses, cost, &mut lambda)
            else {
                break;
            };
            let gain = (cost - new_cost) / cost.max(1e-300);
            cost = new_cost;
            let improved = gain > 1e-12;
            if !improved {
                break;
            }
        }
        cost
    }

    /// Per view, per point (inlier or not), the length of the residual;
    /// infinite where the point does not project.
    fn residual_lengths(&self, intrinsics: &[Real], poses: &[PoseParameters]) -> Vec<Vec<Real>> {
        (0..self.views.len())
            .map(|v| {
                let pose = rigid_pose(&poses[v]);
                self.views[v]
                    .iter()
                    .map(|p| {
                        self.residual(intrinsics, &pose.rotation, &pose.translation, p)
                            .map_or(Real::INFINITY, |r| r[0].hypot(r[1]))
                    })
                    .collect()
            })
            .collect()
    }

    /// The residual length beyond which a point is an outlier, from the
    /// median over the current inliers. `None` when there are none.
    fn outlier_limit(&self, lengths: &[Vec<Real>]) -> Option<Real> {
        let mut kept: Vec<Real> = lengths
            .iter()
            .zip(&self.inliers)
            .flat_map(|(view_lengths, view_inliers)| {
                view_lengths
                    .iter()
                    .zip(view_inliers)
                    .filter(|&(_, &inlier)| inlier)
                    .map(|(length, _)| *length)
            })
            .collect();
        if kept.is_empty() {
            return None;
        }
        let mid = kept.len() / 2;
        let median = *kept.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1;
        // The median of a 2D Gaussian residual's length is 1.1774 σ.
        Some((OUTLIER_SIGMAS * median / 1.1774).max(MIN_OUTLIER_PX))
    }

    /// Fit, then drop points beyond a few robust sigmas and fit again, until
    /// the inlier set settles. A point dropped in one round may come back in
    /// the next.
    fn fit_with_rejection(
        &mut self,
        intrinsics: &mut [Real],
        poses: &mut [PoseParameters],
    ) -> Result<(), CalibrationError> {
        for _ in 0..OUTLIER_ROUNDS {
            self.refine(intrinsics, poses);
            let lengths = self.residual_lengths(intrinsics, poses);
            let limit = self
                .outlier_limit(&lengths)
                .ok_or(CalibrationError::TooFewPoints)?;
            let inliers: Vec<Vec<bool>> = lengths
                .iter()
                .map(|view| view.iter().map(|&length| length <= limit).collect())
                .collect();
            if inliers == self.inliers {
                return Ok(());
            }
            self.inliers = inliers;
            if self
                .inliers
                .iter()
                .any(|view| count_true(view) < MIN_INLIERS_PER_VIEW)
            {
                return Err(CalibrationError::TooFewPoints);
            }
        }
        self.refine(intrinsics, poses);
        Ok(())
    }

    /// The final report for every view.
    fn view_fits(&self, intrinsics: &[Real], poses: &[PoseParameters]) -> Vec<ViewFit> {
        self.views
            .iter()
            .zip(&self.inliers)
            .zip(poses)
            .map(|((points, inliers), pose)| self.view_fit(intrinsics, points, inliers, pose))
            .collect()
    }

    fn view_fit(
        &self,
        intrinsics: &[Real],
        points: &[PointMatch],
        inliers: &[bool],
        pose: &PoseParameters,
    ) -> ViewFit {
        let pose = rigid_pose(pose);
        let residuals: Vec<[Real; 2]> = points
            .iter()
            .map(|p| {
                self.residual(intrinsics, &pose.rotation, &pose.translation, p)
                    .unwrap_or([Real::NAN; 2])
            })
            .collect();
        let used = count_true(inliers);
        let sum_squares: Real = residuals
            .iter()
            .zip(inliers)
            .filter(|&(_, &inlier)| inlier)
            .map(|(r, _)| {
                let length = r[0].hypot(r[1]);
                if length.is_nan() {
                    Real::INFINITY
                } else {
                    length * length
                }
            })
            .sum();
        ViewFit {
            pose,
            rms: (sum_squares / used.max(1) as Real).sqrt(),
            used,
            rejected: inliers.len() - used,
            residuals,
            inliers: inliers.to_vec(),
        }
    }
}

fn count_true(flags: &[bool]) -> usize {
    flags.iter().filter(|&&flag| flag).count()
}
