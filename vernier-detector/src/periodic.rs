//! Periodic (relative) pattern detector — parity with C++
//! `vernier::PeriodicPatternDetector`.

use vernier_core::{ComputeBackend, GrayImage, Pose, Real, Result};
use vernier_pose::{Calibration, periodic};
use vernier_spectral::PhasePlane;
use vernier_spectral::spectrum::{Detection, analyze_two};

use crate::{Metadata, PatternDetector, SpectralConfig};

/// Runs the two-direction spectral analysis with the given configuration.
///
/// Shared by [`PeriodicPatternDetector`] and the megarena detector.
pub(crate) fn run_detection<B: ComputeBackend>(
    backend: &B,
    image: &GrayImage,
    config: &SpectralConfig,
) -> Result<Detection> {
    analyze_two(
        backend,
        &image.to_complex(),
        image.layout(),
        config.sigma as Real,
        config.min_frequency,
        config.max_frequency,
        config.smoothing_sigma as Real,
    )
}

/// Estimates the pose of a periodic pattern with subpixel resolution.
///
/// Recovers `x`, `y` modulo the pattern period and the in-image orientation
/// `theta`. Absolute position is ambiguous modulo the period; use
/// [`MegarenaPatternDetector`](crate::MegarenaPatternDetector) for an
/// unambiguous absolute pose.
pub struct PeriodicPatternDetector<B: ComputeBackend> {
    backend: B,
    config: SpectralConfig,
    meta: Metadata,
    pose: Option<Pose>,
    /// Fitted phase planes from the last `compute`, kept for 3D pose recovery.
    planes: Option<(PhasePlane, PhasePlane)>,
    /// Measured unwrapped phase maps + image dims, for the (beta, gamma) sign
    /// disambiguation in [`get_3d_pose`](PatternDetector::get_3d_pose).
    measured: Option<(Vec<Real>, Vec<Real>, usize, usize)>,
}

impl<B: ComputeBackend> PeriodicPatternDetector<B> {
    /// Creates a detector with default spectral parameters.
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            config: SpectralConfig::default(),
            meta: Metadata::default(),
            pose: None,
            planes: None,
            measured: None,
        }
    }

    /// Physical period × image calibration for pose scaling.
    fn calibration(&self, width: usize, height: usize) -> Calibration {
        Calibration::new(self.config.physical_period as Real, width, height)
    }

    /// Read-only access to the spectral configuration.
    pub fn config(&self) -> &SpectralConfig {
        &self.config
    }

    /// Mutable access to the spectral configuration.
    pub fn config_mut(&mut self) -> &mut SpectralConfig {
        &mut self.config
    }
}

impl<B: ComputeBackend> PatternDetector for PeriodicPatternDetector<B> {
    fn compute(&mut self, image: &GrayImage) -> Result<()> {
        // Forget the last frame first, so a failure here isn't reported as
        // that frame's pose.
        self.pose = None;
        self.planes = None;
        self.measured = None;
        let detection = run_detection(&self.backend, image, &self.config)?;
        let calib = self.calibration(image.width(), image.height());
        self.pose = Some(periodic::estimate(
            &detection.dir1.plane,
            &detection.dir2.plane,
            &calib,
        ));
        self.planes = Some((detection.dir1.plane, detection.dir2.plane));
        // The detection's phase maps are the measured unwrapped maps, kept for
        // the (beta, gamma) sign disambiguation in `get_3d_pose`.
        self.measured = Some((
            detection.phase1,
            detection.phase2,
            image.width(),
            image.height(),
        ));
        Ok(())
    }

    fn pattern_found(&self, _id: i32) -> bool {
        self.pose.is_some()
    }

    fn pattern_count(&self) -> i32 {
        i32::from(self.pose.is_some())
    }

    fn get_2d_pose(&self, _id: i32) -> Pose {
        self.pose.unwrap_or(Pose::ORIGIN)
    }

    fn get_3d_pose(&self, id: i32) -> Pose {
        // Most-likely 3D pose: candidate [0] (which carries +|beta|, +|gamma|)
        // with the (beta, gamma) signs resolved from the measured phase
        // curvature, mirroring C++ `get3DPose`.
        let mut pose = match self.get_all_3d_poses(id).into_iter().next() {
            Some(p) => p,
            None => return Pose::ORIGIN,
        };
        if let (Some((p1, p2)), Some((m1, m2, w, h))) = (&self.planes, &self.measured) {
            let (beta_sign, gamma_sign) =
                periodic::compute_phase_gradients(m1, m2, *w, *h, p1, p2, 0.5);
            pose.beta = pose.beta.abs() * beta_sign as Real;
            pose.gamma = pose.gamma.abs() * gamma_sign as Real;
        }
        pose.z = 0.0;
        pose
    }

    fn get_all_3d_poses(&self, _id: i32) -> Vec<Pose> {
        match self.planes {
            Some((p1, p2)) => {
                let calib = self.calibration(0, 0);
                periodic::all_3d_poses(&p1, &p2, &calib).to_vec()
            }
            None => Vec::new(),
        }
    }

    fn classname(&self) -> &str {
        "PeriodicPattern"
    }

    fn description(&self) -> &str {
        &self.meta.description
    }
    fn set_description(&mut self, value: String) {
        self.meta.description = value;
    }
    fn author(&self) -> &str {
        &self.meta.author
    }
    fn set_author(&mut self, value: String) {
        self.meta.author = value;
    }
    fn date(&self) -> &str {
        &self.meta.date
    }
    fn set_date(&mut self, value: String) {
        self.meta.date = value;
    }
    fn unit(&self) -> &str {
        &self.meta.unit
    }
    fn set_unit(&mut self, value: String) {
        self.meta.unit = value;
    }

    fn get_double(&self, attribute: &str) -> Option<f64> {
        self.config.get_double(attribute)
    }
    fn set_double(&mut self, attribute: &str, value: f64) -> bool {
        self.config.set_double(attribute, value)
    }

    fn get_int(&self, attribute: &str) -> Option<i64> {
        match attribute {
            "minFrequency" => Some(self.config.min_frequency as i64),
            "maxFrequency" => Some(self.config.max_frequency as i64),
            _ => None,
        }
    }
    fn set_int(&mut self, attribute: &str, value: i64) -> bool {
        match attribute {
            "minFrequency" => self.config.min_frequency = value.max(0) as usize,
            "maxFrequency" => self.config.max_frequency = value.max(0) as usize,
            _ => return false,
        }
        true
    }

    fn get_bool(&self, _attribute: &str) -> Option<bool> {
        None
    }
    fn set_bool(&mut self, _attribute: &str, _value: bool) -> bool {
        false
    }

    fn describe(&self) -> String {
        format!(
            "PeriodicPatternDetector [ physicalPeriod={}; sigma={}; found={} ]",
            self.config.physical_period,
            self.config.sigma,
            self.pose.is_some()
        )
    }
}
