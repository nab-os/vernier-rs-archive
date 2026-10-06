//! # vernier-detector
//!
//! Object-level parity layer mirroring the C++ `vernier::PatternDetector`
//! interface and the `vernier::Detector` factory.
//!
//! The C++ library exposes detection through an abstract [`PatternDetector`]
//! base with a JSON-driven [`Detector`] factory. The Rust engine underneath is
//! functional (backend-generic free functions in `vernier-spectral` /
//! `vernier-pose`); this crate wraps that engine in the class-shaped API so
//! callers — and the C ABI / language bindings — get the same surface as C++:
//!
//! ```no_run
//! use vernier_detector::Detector;
//! use vernier_core::GrayImage;
//!
//! let mut det = Detector::new_instance("MegarenaPattern").unwrap();
//! det.set_double("physicalPeriod", 9.0);
//! det.set_int("codeSize", 12);
//! # let image = GrayImage::zeros(64, 64);
//! det.compute(&image).unwrap();
//! if det.pattern_found(-1) {
//!     println!("{}", det.get_2d_pose(-1));
//! }
//! ```
//!
//! Concrete detectors are generic over any [`ComputeBackend`]. The [`Detector`]
//! factory defaults to the CPU backend; use [`Detector::new_instance_with`] to
//! supply another (e.g. CUDA).

pub mod bitmap;
pub mod gds;
pub mod layout;
pub mod megarena;
pub mod periodic;
pub mod stub;
pub mod thumbnail;

use vernier_core::{ComputeBackend, GrayImage, Pose, Result, VernierError};
use vernier_cpu::CpuBackend;

pub use bitmap::BitmapPatternDetector;
pub use layout::{
    Layout, Margins, MegarenaPatternLayout, PatternLayout, PeriodicPatternLayout, Rectangle,
};
pub use megarena::MegarenaPatternDetector;
pub use periodic::PeriodicPatternDetector;
pub use stub::UnimplementedDetector;
pub use thumbnail::{BitmapThumbnail, MatchResult, match_template_ccoeff, rotate90_cw};

/// Object interface for pattern detectors, mirroring the C++
/// `vernier::PatternDetector` abstract class.
///
/// `id` selects a pattern when a detector can report several; single-pattern
/// detectors ignore it (pass `-1`, matching the C++ default argument).
pub trait PatternDetector {
    /// Detects and estimates the pose(s) of the pattern(s) in `image`.
    fn compute(&mut self, image: &GrayImage) -> Result<()>;

    /// Returns `true` if a pattern has been detected and localized.
    fn pattern_found(&self, id: i32) -> bool;

    /// Number of detected patterns.
    fn pattern_count(&self) -> i32;

    /// 2D pose of the pattern.
    fn get_2d_pose(&self, id: i32) -> Pose;

    /// Most likely 3D pose of the pattern.
    fn get_3d_pose(&self, id: i32) -> Pose;

    /// All possible 3D poses of the pattern (in case of ambiguities).
    fn get_all_3d_poses(&self, id: i32) -> Vec<Pose>;

    /// Class name of the detector (`"PeriodicPattern"`, `"MegarenaPattern"`, …).
    fn classname(&self) -> &str;

    /// Free-text metadata (mirrors the C++ modifiable fields).
    fn description(&self) -> &str;
    /// Sets the `description` metadata.
    fn set_description(&mut self, value: String);
    /// Author metadata.
    fn author(&self) -> &str;
    /// Sets the `author` metadata.
    fn set_author(&mut self, value: String);
    /// Date metadata.
    fn date(&self) -> &str;
    /// Sets the `date` metadata.
    fn set_date(&mut self, value: String);
    /// Unit metadata.
    fn unit(&self) -> &str;
    /// Sets the `unit` metadata.
    fn set_unit(&mut self, value: String);

    /// Reads a floating-point attribute by name, or `None` if unknown.
    fn get_double(&self, attribute: &str) -> Option<f64>;
    /// Sets a floating-point attribute; returns `false` if the name is unknown.
    fn set_double(&mut self, attribute: &str, value: f64) -> bool;
    /// Reads an integer attribute by name, or `None` if unknown.
    fn get_int(&self, attribute: &str) -> Option<i64>;
    /// Sets an integer attribute; returns `false` if the name is unknown.
    fn set_int(&mut self, attribute: &str, value: i64) -> bool;
    /// Reads a boolean attribute by name, or `None` if unknown.
    fn get_bool(&self, attribute: &str) -> Option<bool>;
    /// Sets a boolean attribute; returns `false` if the name is unknown.
    fn set_bool(&mut self, attribute: &str, value: bool) -> bool;

    /// Human-readable description of the detector (mirrors C++ `toString`).
    fn describe(&self) -> String;
}

/// Free-text metadata shared by all detectors (C++ `description/date/author/unit`).
#[derive(Clone, Debug, Default)]
pub struct Metadata {
    /// Free-text description.
    pub description: String,
    /// Creation date.
    pub date: String,
    /// Author.
    pub author: String,
    /// Physical unit of the reported translations.
    pub unit: String,
}

/// Spectral analysis parameters shared by the periodic and megarena detectors.
#[derive(Clone, Copy, Debug)]
pub struct SpectralConfig {
    /// Physical period of the pattern, in the reported unit (C++ `physicalPeriod`).
    pub physical_period: f32,
    /// Band-pass filter half-width in frequency bins (C++ `sigma`).
    pub sigma: f32,
    /// Inner annulus radius for peak search, in bins (0 = no limit).
    pub min_frequency: usize,
    /// Outer annulus radius for peak search, in bins (0 = no limit).
    pub max_frequency: usize,
    /// Gaussian blur sigma applied to the magnitude spectrum before peak search.
    pub smoothing_sigma: f32,
}

impl Default for SpectralConfig {
    fn default() -> Self {
        // Matches the CLI / C++ `PatternPhase` defaults.
        Self {
            physical_period: 1.0,
            sigma: 3.0,
            min_frequency: 20,
            max_frequency: 500,
            smoothing_sigma: 0.5,
        }
    }
}

impl SpectralConfig {
    /// Reads a spectral attribute by C++ name.
    pub(crate) fn get_double(&self, attribute: &str) -> Option<f64> {
        match attribute {
            "physicalPeriod" => Some(self.physical_period as f64),
            "sigma" => Some(self.sigma as f64),
            "minFrequency" => Some(self.min_frequency as f64),
            "maxFrequency" => Some(self.max_frequency as f64),
            "smoothingSigma" => Some(self.smoothing_sigma as f64),
            _ => None,
        }
    }

    /// Sets a spectral attribute by C++ name; returns `false` if unknown.
    pub(crate) fn set_double(&mut self, attribute: &str, value: f64) -> bool {
        match attribute {
            "physicalPeriod" => self.physical_period = value as f32,
            "sigma" => self.sigma = value as f32,
            "minFrequency" => self.min_frequency = value.max(0.0) as usize,
            "maxFrequency" => self.max_frequency = value.max(0.0) as usize,
            "smoothingSigma" => self.smoothing_sigma = value as f32,
            _ => return false,
        }
        true
    }
}

/// Class factory for pattern detectors, mirroring the C++ `vernier::Detector`.
pub struct Detector;

impl Detector {
    /// Creates a CPU-backed detector for the given class name.
    ///
    /// Valid names: `"PeriodicPattern"`, `"MegarenaPattern"`, `"BitmapPattern"`;
    /// `"StampPattern"` and `"HPCodePattern"` are recognised but not yet
    /// implemented, and their detectors fail on `compute`.
    pub fn new_instance(classname: &str) -> Result<Box<dyn PatternDetector>> {
        Self::new_instance_with(classname, CpuBackend::new())
    }

    /// Creates a detector for the given class name backed by `backend`.
    pub fn new_instance_with<B: ComputeBackend + 'static>(
        classname: &str,
        backend: B,
    ) -> Result<Box<dyn PatternDetector>> {
        match classname {
            "PeriodicPattern" => Ok(Box::new(PeriodicPatternDetector::new(backend))),
            "MegarenaPattern" => Ok(Box::new(MegarenaPatternDetector::new(backend))),
            // Created without a reference bitmap; supply one via
            // `BitmapPatternDetector::with_bitmap`/`set_bitmap` before compute.
            "BitmapPattern" => Ok(Box::new(BitmapPatternDetector::new(backend))),
            // Recognized for factory-surface parity, but detection is not yet
            // ported (needs CV primitives absent from the workspace); `compute`
            // returns a "not implemented" error.
            "StampPattern" => Ok(Box::new(stub::UnimplementedDetector::new("StampPattern"))),
            "HPCodePattern" => Ok(Box::new(stub::UnimplementedDetector::new("HPCodePattern"))),
            other => Err(VernierError::Message(format!(
                "{other} is not a valid class name for a pattern detector."
            ))),
        }
    }

    /// Loads a detector from a JSON document with the C++ layout:
    /// `{ "<ClassName>": { "physicalPeriod": .., "sigma": .., ... } }`.
    pub fn load_from_json(path: &str) -> Result<Box<dyn PatternDetector>> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| VernierError::Message(format!("{path}: {e}")))?;
        Self::from_json_str(&text)
    }

    /// Parses a detector from an in-memory JSON string (see [`load_from_json`]).
    ///
    /// [`load_from_json`]: Detector::load_from_json
    pub fn from_json_str(text: &str) -> Result<Box<dyn PatternDetector>> {
        let value: serde_json::Value = serde_json::from_str(text)
            .map_err(|e| VernierError::Message(format!("invalid JSON: {e}")))?;
        let obj = value
            .as_object()
            .ok_or_else(|| VernierError::Message("JSON root is not an object.".into()))?;
        let (classname, attrs) = obj
            .iter()
            .next()
            .ok_or_else(|| VernierError::Message("JSON object is empty.".into()))?;

        let mut det = Self::new_instance(classname)?;
        if let Some(map) = attrs.as_object() {
            for (key, val) in map {
                match val {
                    serde_json::Value::Number(n) => {
                        if let Some(i) = n.as_i64() {
                            // Prefer int setter, fall back to double.
                            if !det.set_int(key, i) {
                                det.set_double(key, i as f64);
                            }
                        } else if let Some(f) = n.as_f64() {
                            det.set_double(key, f);
                        }
                    }
                    serde_json::Value::Bool(b) => {
                        det.set_bool(key, *b);
                    } // b: &bool
                    serde_json::Value::String(s) => match key.as_str() {
                        "description" => det.set_description(s.clone()),
                        "date" => det.set_date(s.clone()),
                        "author" => det.set_author(s.clone()),
                        "unit" => det.set_unit(s.clone()),
                        _ => {}
                    },
                    _ => {}
                }
            }
        }
        Ok(det)
    }
}
