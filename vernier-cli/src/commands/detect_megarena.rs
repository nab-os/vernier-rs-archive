//! Port of the C++ `detectingMegarenaPattern.cpp` example.
//!
//! The original constructs `MegarenaPatternDetector(physicalPeriod, codeSize)`,
//! runs `compute(image)`, and if a pattern is found prints the estimated 2D
//! pose. This is the full absolute path on a real photo: two-direction
//! detection → binary-code extraction → LFSR localization → assembled absolute
//! `(x, y, θ)`.
//!
//! `physical_period` (µm) and `code_size` (LFSR order in bits) mirror the C++
//! constructor arguments — for the paper's target, 9 µm and 12 bits.

use std::path::PathBuf;

use vernier_core::image::GrayImage;
use vernier_core::{ComputeBackend, Real};
use vernier_pose::absolute::{
    self, CoarseDecoder, MegarenaDecoder, decode_bit_maps, detect_orientation, extract_code,
};
use vernier_spectral::spectrum::analyze_two;

use super::debug_render;
use vernier_pose::{Calibration, periodic};

use crate::backend_select::BackendTask;
use crate::imageio::load_grayscale;

/// Megarena detection parameters.
pub struct DetectMegarena {
    /// Path to the image file.
    pub image_path: PathBuf,
    /// Physical period of the pattern in micrometres (the paper uses 9 µm).
    pub physical_period: f32,
    /// LFSR code size in bits (the paper uses 12).
    pub code_size: u32,
    /// Band-pass filter width in bins (C++ default 3.0).
    pub sigma: f32,
    /// Inner annulus radius for peak search in bins; 0 = no lower limit (C++ default 20).
    pub min_frequency: usize,
    /// Outer annulus radius for peak search in bins; 0 = no upper limit (C++ default 500).
    pub max_frequency: usize,
    /// Gaussian blur sigma applied to magnitude before peak search (C++ default 0.5).
    pub smoothing_sigma: f32,
    /// Optional path prefix for debug overlay images (spectrum + decoded cells).
    pub debug_image: Option<std::path::PathBuf>,
    /// Print intermediate detection details (carriers, planes, orientation).
    pub verbose: bool,
}

/// What a DetectMegarena run reports.
pub struct DetectMegarenaReport {
    /// Pose x
    pub x: Real,
    /// Pose y
    pub y: Real,
    /// Theta
    pub theta: Real,
    /// K3
    pub k3: u8,
}

impl BackendTask for DetectMegarena {
    type Output = DetectMegarenaReport;

    fn run<B: ComputeBackend>(&self, backend: &B) -> DetectMegarenaReport {
        let img = load_grayscale(&self.image_path).expect("failed to load image");
        let (width, height) = (img.width, img.height);
        let gray = GrayImage::from_vec(width, height, img.data).expect("image dimensions mismatch");
        let data = gray.to_complex();
        let layout = gray.layout();

        // Two-direction detection -> phase planes + phase maps.
        let detection = analyze_two(
            backend,
            &data,
            layout,
            self.sigma as Real,
            self.min_frequency,
            self.max_frequency,
            self.smoothing_sigma as Real,
        )
        .expect("detection failed");

        let calib = Calibration::new(self.physical_period as Real, width, height);
        let fine = periodic::estimate(&detection.dir1.plane, &detection.dir2.plane, &calib);

        if self.verbose {
            let (b1x, b1y) = detection.dir1.peak_bin;
            let (b2x, b2y) = detection.dir2.peak_bin;
            eprintln!(
                "carrier 1: bin=({b1x},{b1y})  peak=({:.3},{:.3})",
                detection.dir1.peak.0, detection.dir1.peak.1
            );
            eprintln!(
                "carrier 2: bin=({b2x},{b2y})  peak=({:.3},{:.3})",
                detection.dir2.peak.0, detection.dir2.peak.1
            );
            eprintln!(
                "plane 1:  a={:.6}  b={:.6}  c={:.6}",
                detection.dir1.plane.a, detection.dir1.plane.b, detection.dir1.plane.c
            );
            eprintln!(
                "plane 2:  a={:.6}  b={:.6}  c={:.6}",
                detection.dir2.plane.a, detection.dir2.plane.b, detection.dir2.plane.c
            );
            eprintln!("fine theta={:.6} rad", fine.theta);
        }

        let intensity: Vec<f32> = gray.as_slice().to_vec();

        if self.verbose {
            match detect_orientation(&detection, &intensity) {
                None => eprintln!("orientation: FAILED (global-cell bin empty)"),
                Some((grid, orient)) => {
                    eprintln!(
                        "orientation: coding=({},{}) missing=({},{}) k3={}",
                        orient.coding1,
                        orient.coding2,
                        orient.missing1,
                        orient.missing2,
                        orient.quadrant
                    );
                    eprintln!("global cell (3x3 mean intensities):");
                    for row in &grid {
                        eprintln!("  [{:.3} {:.3} {:.3}]", row[0], row[1], row[2]);
                    }
                }
            }
        }

        let code =
            extract_code(&detection, &intensity, self.code_size).expect("code extraction failed");

        let decoder = MegarenaDecoder::new(
            self.code_size,
            code.x_window.clone(),
            code.y_window.clone(),
            code.k3,
        )
        .unwrap_or_else(|| panic!("unsupported code size {}", self.code_size));

        let orders = decoder.decode().expect("LFSR decode failed");

        // Absolute pose assembly shared with the C ABI / bindings (also applies
        // the C++ plane swap/flip so theta is the absolute orientation).
        let pose = absolute::solve_megarena(&detection, &intensity, &calib, self.code_size)
            .expect("absolute solve failed");

        if self.verbose {
            eprintln!(
                "msb1={}  msb2={}  k3={}  swap={}",
                code.msb1,
                code.msb2,
                code.k3,
                code.msb1 != code.msb2
            );
            eprintln!(
                "x_periodshift={}  y_periodshift={}",
                code.x_periodshift, code.y_periodshift
            );
            eprintln!("LFSR k1={}  k2={}", orders.k1, orders.k2);
            eprintln!("abs: x={:.4} µm  y={:.4} µm", pose.x, pose.y);
        }

        if let Some(prefix) = &self.debug_image {
            let gray_f64: Vec<f64> = gray.as_slice().iter().map(|&v| v as f64).collect();
            if let Err(e) = debug_render::render_spectrum_debug(
                &gray_f64,
                width,
                height,
                &detection,
                self.min_frequency,
                &with_suffix(prefix, "_spectrum.png"),
            ) {
                eprintln!("warning: spectrum debug image: {e}");
            }
            let (x_bits, y_bits) = decode_bit_maps(&detection, &intensity);
            if let Err(e) = debug_render::render_decode_debug(
                &gray_f64,
                width,
                height,
                &detection,
                &x_bits,
                &y_bits,
                &with_suffix(prefix, "_decoded.png"),
            ) {
                eprintln!("warning: decode debug image: {e}");
            }
        }

        DetectMegarenaReport {
            x: pose.x,
            y: pose.y,
            theta: pose.theta,
            k3: orders.k3,
        }
    }
}

/// Builds a sibling path by appending a suffix to the prefix's file stem.
/// e.g. prefix "out/dbg" + "_spectrum.png" -> "out/dbg_spectrum.png".
fn with_suffix(prefix: &std::path::Path, suffix: &str) -> std::path::PathBuf {
    let mut s = prefix.as_os_str().to_os_string();
    s.push(suffix);
    std::path::PathBuf::from(s)
}
