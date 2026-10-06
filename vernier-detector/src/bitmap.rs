//! Bitmap (absolute) pattern detector — parity with C++
//! `vernier::BitmapPatternDetector`.
//!
//! A bitmap pattern is a periodic carrier whose dots are present/absent to spell
//! out a fixed reference bitmap (a "code image"). Detection runs the periodic
//! phase analysis, reduces the image to a [`BitmapThumbnail`], then matches that
//! thumbnail against the reference bitmap in its four 90° orientations to
//! recover which orientation is in view and the integer period shifts — giving
//! an absolute pose.

use vernier_core::scalar::consts::TAU;
use vernier_core::{ComputeBackend, GrayImage, Pose, Real, Result};
use vernier_spectral::PhasePlane;

use crate::periodic::run_detection;
use crate::thumbnail::{BitmapThumbnail, match_template_ccoeff, rotate90_cw};
use crate::{Metadata, PatternDetector, SpectralConfig};

/// A reference bitmap plus its dimensions (row-major grayscale).
struct RefBitmap {
    pixels: Vec<u8>,
    width: usize,
    height: usize,
}

/// Estimates the absolute pose of a bitmap-encoded periodic pattern.
pub struct BitmapPatternDetector<B: ComputeBackend> {
    backend: B,
    config: SpectralConfig,
    meta: Metadata,
    /// Reference bitmap in its four 90° rotations (empty until a bitmap is set).
    bitmaps: Vec<RefBitmap>,
    pose: Option<Pose>,
    /// Thumbnail computed by the last `compute` (C++ `getThumbnail`).
    thumbnail: Option<BitmapThumbnail>,
    /// Orientation-adjusted planes from the last successful `compute`.
    planes: Option<(PhasePlane, PhasePlane)>,
    period_shift1: i64,
    period_shift2: i64,
    bitmap_index: i32,
    max_angle: i32,
}

impl<B: ComputeBackend> BitmapPatternDetector<B> {
    /// Creates an empty bitmap detector (no reference bitmap set; matches the
    /// C++ default constructor). Use [`set_bitmap`](Self::set_bitmap) to supply
    /// the code image before calling `compute`.
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            config: SpectralConfig::default(),
            meta: Metadata::default(),
            bitmaps: Vec::new(),
            pose: None,
            thumbnail: None,
            planes: None,
            period_shift1: 0,
            period_shift2: 0,
            bitmap_index: -1,
            max_angle: 0,
        }
    }

    /// Creates a detector for the given reference bitmap (grayscale code image).
    pub fn with_bitmap(backend: B, bitmap: &GrayImage) -> Self {
        let mut det = Self::new(backend);
        det.set_bitmap(bitmap);
        det
    }

    /// Sets the reference bitmap, precomputing its four 90° rotations (C++
    /// stores `bitmap[0..4]`).
    pub fn set_bitmap(&mut self, bitmap: &GrayImage) {
        let base: Vec<u8> = bitmap
            .as_slice()
            .iter()
            .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect();
        let (mut pixels, mut w, mut h) = (base, bitmap.width(), bitmap.height());
        let mut rots = Vec::with_capacity(4);
        for _ in 0..4 {
            rots.push(RefBitmap {
                pixels: pixels.clone(),
                width: w,
                height: h,
            });
            let (rp, rw, rh) = rotate90_cw(&pixels, w, h);
            pixels = rp;
            w = rw;
            h = rh;
        }
        self.bitmaps = rots;
    }

    /// Index of the best-matching reference orientation (0..4), or -1 if none.
    pub fn bitmap_index(&self) -> i32 {
        self.bitmap_index
    }

    /// Recovered orientation of the matched bitmap, in degrees (0/90/180/270).
    pub fn max_angle(&self) -> i32 {
        self.max_angle
    }

    /// The thumbnail computed by the last [`compute`](PatternDetector::compute),
    /// if any (C++ `getThumbnail`).
    pub fn thumbnail(&self) -> Option<&BitmapThumbnail> {
        self.thumbnail.as_ref()
    }

    fn calibration(&self) -> vernier_pose::Calibration {
        vernier_pose::Calibration::new(self.config.physical_period as Real, 0, 0)
    }

    /// Runs the template match + orientation resolution (C++ `computeAbsolutePose`).
    fn compute_absolute_pose(
        &mut self,
        thumb: &BitmapThumbnail,
        mut p1: PhasePlane,
        mut p2: PhasePlane,
    ) {
        let n = thumb.size;
        let mut best = f64::NEG_INFINITY;
        self.bitmap_index = -1;

        for (k, bm) in self.bitmaps.iter().enumerate() {
            let Some(m) =
                match_template_ccoeff(&thumb.thumbnail, n, n, &bm.pixels, bm.width, bm.height)
            else {
                continue;
            };
            if m.max_val > best {
                best = m.max_val;
                self.bitmap_index = k as i32;
                self.max_angle = (k % 4) as i32 * 90;
                self.period_shift1 = -(m.max_x as i64 - (m.width / 2) as i64) / 2;
                self.period_shift2 = -(m.max_y as i64 - (m.height / 2) as i64) / 2;
            }
        }

        if self.bitmap_index < 0 {
            self.pose = None;
            return;
        }

        // Fold the matched orientation back into the planes / shifts.
        match self.max_angle {
            90 => {
                std::mem::swap(&mut p1, &mut p2);
                p2.flip();
                std::mem::swap(&mut self.period_shift1, &mut self.period_shift2);
                self.period_shift2 = -self.period_shift2;
            }
            180 => {
                p1.flip();
                p2.flip();
                self.period_shift1 = -self.period_shift1;
                self.period_shift2 = -self.period_shift2;
            }
            270 => {
                std::mem::swap(&mut p1, &mut p2);
                p1.flip();
                std::mem::swap(&mut self.period_shift1, &mut self.period_shift2);
                self.period_shift1 = -self.period_shift1;
            }
            _ => {}
        }

        self.pose = Some(self.assemble_pose(&p1, &p2));
        self.planes = Some((p1, p2));
    }

    /// Builds the absolute 2D pose from the (adjusted) planes and period shifts,
    /// mirroring `PeriodicPatternDetector::get2DPose` with the shifts applied.
    fn assemble_pose(&self, p1: &PhasePlane, p2: &PhasePlane) -> Pose {
        let period = self.config.physical_period as f64;
        let x = -(period * (p1.c / TAU + self.period_shift1 as f64));
        let y = -(period * (p2.c / TAU + self.period_shift2 as f64));
        let alpha = (p1.b).atan2(p1.a);
        let pixelic_period = TAU / (p1.a * p1.a + p1.b * p1.b).sqrt();
        let pixel_size = period as Real / pixelic_period;
        Pose::new_2d(x as Real, y as Real, alpha, pixel_size)
    }
}

// PhasePlane has no `flip` in vernier-spectral; provide it locally.
trait Flip {
    fn flip(&mut self);
}
impl Flip for PhasePlane {
    fn flip(&mut self) {
        self.a = -self.a;
        self.b = -self.b;
        self.c = -self.c;
    }
}

impl<B: ComputeBackend> PatternDetector for BitmapPatternDetector<B> {
    fn compute(&mut self, image: &GrayImage) -> Result<()> {
        self.pose = None;
        self.planes = None;
        self.thumbnail = None;

        let detection = run_detection(&self.backend, image, &self.config)?;
        let p1 = detection.dir1.plane;
        let p2 = detection.dir2.plane;

        // Thumbnail size from the pixelic period (C++ `computeImage`).
        let pix1 = TAU / (p1.a * p1.a + p1.b * p1.b).sqrt();
        let pix2 = TAU / (p2.a * p2.a + p2.b * p2.b).sqrt();
        let approx = ((pix1 + pix2) / 2.0) as f64;
        let mut len1 = (2.82 * image.height() as f64 / approx) as i64;
        let mut len2 = (2.82 * image.width() as f64 / approx) as i64;
        if len1 % 2 == 0 {
            len1 += 1;
        }
        if len2 % 2 == 0 {
            len2 += 1;
        }
        let size = len1.max(len2).max(1) as usize;

        let mut thumb = BitmapThumbnail::new(size);
        thumb.compute(image.as_slice(), image.width(), image.height(), &p1, &p2);

        // Match against the reference bitmap(s), if any (C++ empty `bitmap`
        // vector simply yields no match / no pose).
        if !self.bitmaps.is_empty() {
            self.compute_absolute_pose(&thumb, p1, p2);
        }
        self.thumbnail = Some(thumb);
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
        self.get_all_3d_poses(id)
            .into_iter()
            .next()
            .unwrap_or(Pose::ORIGIN)
    }

    fn get_all_3d_poses(&self, _id: i32) -> Vec<Pose> {
        match self.planes {
            Some((p1, p2)) => {
                let calib = self.calibration();
                vernier_pose::periodic::all_3d_poses(&p1, &p2, &calib).to_vec()
            }
            None => Vec::new(),
        }
    }

    fn classname(&self) -> &str {
        "BitmapPattern"
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
            "periodShift1" => Some(self.period_shift1),
            "periodShift2" => Some(self.period_shift2),
            "bitmapIndex" => Some(self.bitmap_index as i64),
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
            "BitmapPatternDetector [ physicalPeriod={}; bitmaps={}; found={} ]",
            self.config.physical_period,
            self.bitmaps.len() / 4,
            self.pose.is_some()
        )
    }
}
