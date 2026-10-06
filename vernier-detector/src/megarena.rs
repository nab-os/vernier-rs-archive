//! Megarena (absolute) pattern detector — parity with C++
//! `vernier::MegarenaPatternDetector`.

use vernier_core::{ComputeBackend, GrayImage, Pose, Real, Result};
use vernier_pose::{Calibration, absolute};

use crate::periodic::run_detection;
use crate::{Metadata, PatternDetector, SpectralConfig};

/// Default physical period of the reference megarena pattern (micrometres).
const DEFAULT_PERIOD: f32 = 9.0;
/// Default LFSR code size of the reference megarena pattern (bits).
const DEFAULT_CODE_SIZE: u32 = 12;

/// Estimates the unambiguous absolute pose of a megarena pattern.
///
/// Combines the fine phase pose with the LFSR position code embedded in the
/// pattern to resolve which period cell is in view, yielding an absolute
/// `(x, y, theta)`.
pub struct MegarenaPatternDetector<B: ComputeBackend> {
    backend: B,
    config: SpectralConfig,
    code_size: u32,
    meta: Metadata,
    pose: Option<Pose>,
}

impl<B: ComputeBackend> MegarenaPatternDetector<B> {
    /// Creates a detector with the reference-pattern defaults (9 µm period,
    /// 12-bit code).
    pub fn new(backend: B) -> Self {
        let mut config = SpectralConfig::default();
        config.physical_period = DEFAULT_PERIOD;
        Self {
            backend,
            config,
            code_size: DEFAULT_CODE_SIZE,
            meta: Metadata::default(),
            pose: None,
        }
    }

    /// LFSR code size in bits (C++ `codeSize`).
    pub fn code_size(&self) -> u32 {
        self.code_size
    }

    /// Sets the LFSR code size in bits.
    pub fn set_code_size(&mut self, code_size: u32) {
        self.code_size = code_size;
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

impl<B: ComputeBackend> PatternDetector for MegarenaPatternDetector<B> {
    fn compute(&mut self, image: &GrayImage) -> Result<()> {
        // Forget the last frame first, so a failure here isn't reported as
        // that frame's pose.
        self.pose = None;
        let detection = run_detection(&self.backend, image, &self.config)?;
        let calib = Calibration::new(self.config.physical_period as Real, image.width(), image.height());
        // A failed absolute solve means "no decodable pattern here": report it
        // through `pattern_found`, not as a hard error. Backend failures above
        // still propagate.
        self.pose = absolute::solve_megarena(&detection, image.as_slice(), &calib, self.code_size).ok();
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

    fn get_3d_pose(&self, _id: i32) -> Pose {
        self.pose.unwrap_or(Pose::ORIGIN)
    }

    fn get_all_3d_poses(&self, _id: i32) -> Vec<Pose> {
        self.pose.into_iter().collect()
    }

    fn classname(&self) -> &str {
        "MegarenaPattern"
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
            "codeSize" => Some(self.code_size as i64),
            "minFrequency" => Some(self.config.min_frequency as i64),
            "maxFrequency" => Some(self.config.max_frequency as i64),
            _ => None,
        }
    }
    fn set_int(&mut self, attribute: &str, value: i64) -> bool {
        match attribute {
            "codeSize" => self.code_size = value.max(0) as u32,
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
            "MegarenaPatternDetector [ physicalPeriod={}; codeSize={}; found={} ]",
            self.config.physical_period,
            self.code_size,
            self.pose.is_some()
        )
    }
}
