use vernier_core::buffer::BufferLayout;
use vernier_core::scalar::consts::TAU;
use vernier_core::{Complex32, ComputeBackend, Real};
use vernier_patterns::PatternPose;
use vernier_patterns::megarena::Megarena;
use vernier_pose::absolute::{CoarseDecoder, MegarenaDecoder, extract_code};
use vernier_pose::{Calibration, periodic};
use vernier_spectral::spectrum::analyze_two;

use crate::backend_select::BackendTask;

pub struct RoundtripMegarena {
    pub width: usize,
    pub height: usize,
    /// Full absolute position (can be large to exercise the LFSR).
    pub true_x: f32,
    pub true_y: f32,
    pub true_theta: f32,
    pub period_px: f32,
    pub code_size: u32,
    pub sigma: f32,
    pub min_frequency: usize,
    pub max_frequency: usize,
    pub smoothing_sigma: f32,
    /// Use the Vulkan GPU renderer instead of the CPU path.
    pub render_gpu: bool,
    /// Camera pixel size in µm/pixel (only used when render_gpu is true).
    #[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
    pub pixel_size: f32,
}

pub struct RoundtripMegarenaReport {
    pub backend: String,
    pub renderer: String,
    /// Full absolute position that was rendered.
    pub true_x: f64,
    pub true_y: f64,
    pub true_theta: f64,
    /// Recovered absolute position (LFSR coarse + spectral fine).
    pub recovered_x: f64,
    pub recovered_y: f64,
    pub recovered_theta: f64,
    /// Total absolute error — large (≥ period) means LFSR decoded the wrong cell.
    pub abs_error_x: f64,
    pub abs_error_y: f64,
    /// Sub-period spectral error in the formula (carrier) frame.
    pub fine_error_x: f64,
    pub fine_error_y: f64,
    pub error_theta: f64,
    pub swapped: bool,
}

impl BackendTask for RoundtripMegarena {
    type Output = RoundtripMegarenaReport;

    fn run<B: ComputeBackend>(&self, backend: &B) -> RoundtripMegarenaReport {
        // Offset the LFSR so that the decode window at the image centre (triple 0
        // for pose=0) reads bits [lfsr_offset .. lfsr_offset+code_size), well away
        // from the sequence boundary.  Without this, the window straddles position 0
        // and `widx.locate` returns ~len, inflating the recovered periodshift by
        // ~3×len periods.
        let lfsr_offset = self.code_size as i64;
        let pattern = Megarena::new(self.period_px as Real, self.code_size)
            .unwrap_or_else(|| panic!("unsupported code size {}", self.code_size))
            .with_lfsr_offset(lfsr_offset);

        let pose = PatternPose::new(
            self.true_x as Real,
            self.true_y as Real,
            self.true_theta as Real,
        );
        let (image, renderer_name) = if self.render_gpu {
            #[cfg(feature = "vulkan")]
            {
                use vernier_patterns::{CameraModel, PatternRenderer};
                let vk_renderer = PatternRenderer::new();
                let camera = CameraModel {
                    pixel_size: self.pixel_size,
                };
                let img = pattern.render_gpu(&vk_renderer, &camera, self.width, self.height, &pose);
                (img, "gpu-vulkan-raster")
            }
            #[cfg(not(feature = "vulkan"))]
            {
                eprintln!(
                    "--render-gpu requires the 'vulkan' feature (build with --features vulkan)"
                );
                std::process::exit(1);
            }
        } else {
            (
                pattern.render(self.width, self.height, &pose),
                "cpu-analytic",
            )
        };

        let layout = BufferLayout::packed(self.width, self.height);
        let complex: Vec<Complex32> = image
            .as_slice()
            .iter()
            .map(|&v| Complex32::new(v, 0.0))
            .collect();

        let detection = analyze_two(
            backend,
            &complex,
            layout,
            self.sigma as Real,
            self.min_frequency,
            self.max_frequency,
            self.smoothing_sigma as Real,
        )
        .expect("detection failed");

        let calib = Calibration::new(self.period_px as Real, self.width, self.height);
        let fine = periodic::estimate(&detection.dir1.plane, &detection.dir2.plane, &calib);

        let intensity: Vec<f32> = image.as_slice().to_vec();
        let code = extract_code(&detection, &intensity, self.code_size)
            .expect("code extraction failed — try a larger image or smaller period");

        let decoder = MegarenaDecoder::new(
            self.code_size,
            code.x_window.clone(),
            code.y_window.clone(),
            code.k3,
        )
        .unwrap_or_else(|| panic!("unsupported code size {}", self.code_size));
        decoder.decode().expect("LFSR decode failed");

        let period = self.period_px as Real;
        let swap = code.msb1 != code.msb2;

        // Select which direction's plane/periodshift/msb maps to each formula output.
        let (x_c, x_ps, x_msb) = if swap {
            (detection.dir2.plane.c, code.y_periodshift, code.msb2)
        } else {
            (detection.dir1.plane.c, code.x_periodshift, code.msb1)
        };
        let (y_c, y_ps, y_msb) = if swap {
            (detection.dir1.plane.c, code.x_periodshift, code.msb1)
        } else {
            (detection.dir2.plane.c, code.y_periodshift, code.msb2)
        };

        let flip_c = |c: Real, msb: bool| -> Real { if msb { c } else { -c } };
        let recovered_x = -(period * (flip_c(x_c, x_msb) / TAU + x_ps as Real));
        let recovered_y = -(period * (flip_c(y_c, y_msb) / TAU + y_ps as Real));

        // Ground truth for the absolute position (LFSR frame).  With lfsr_offset=O
        // and coding=1 (theta=0), k_t0=O, so periodshift = 3*O + 3*(n-1).
        // recovered_x should equal true_x − period × periodshift.
        let ps_baseline = 3 * lfsr_offset + 3 * (self.code_size as i64 - 1);
        let expected_abs_x = self.true_x as Real - period * ps_baseline as Real;
        let expected_abs_y = self.true_y as Real - period * ps_baseline as Real;

        let abs_error_x = (recovered_x - expected_abs_x).abs() as f64;
        let abs_error_y = (recovered_y - expected_abs_y).abs() as f64;

        // Fine error in the carrier (formula) frame: compares the detected carrier
        // phase against the expected phase from the true position. Uses fract() so
        // it's valid for any absolute position, not only small sub-period offsets.
        // wrap_half handles the one-period ambiguity when phases differ by ≈ period.
        let detected_fine_x_dir = -(period * flip_c(x_c, x_msb) / TAU);
        let detected_fine_y_dir = -(period * flip_c(y_c, y_msb) / TAU);

        let (pose_for_x, pose_for_y) = if swap {
            (self.true_y as Real, self.true_x as Real)
        } else {
            (self.true_x as Real, self.true_y as Real)
        };
        let true_fine_x_dir = -(period * (-pose_for_x / period).fract());
        let true_fine_y_dir = -(period * (-pose_for_y / period).fract());
        let expected_fine_x = if x_msb {
            true_fine_x_dir
        } else {
            -true_fine_x_dir
        };
        let expected_fine_y = if y_msb {
            true_fine_y_dir
        } else {
            -true_fine_y_dir
        };

        let wrap_half = |d: Real| -> Real {
            let d = d.rem_euclid(period);
            if d > period / 2.0 { d - period } else { d }
        };
        let fine_error_x = wrap_half(detected_fine_x_dir - expected_fine_x).abs() as f64;
        let fine_error_y = wrap_half(detected_fine_y_dir - expected_fine_y).abs() as f64;

        let recovered_theta = fine.theta as f64;
        let true_theta = self.true_theta as f64;
        let mut error_theta = recovered_theta - true_theta;
        let pi = std::f64::consts::PI;
        while error_theta > pi {
            error_theta -= 2.0 * pi;
        }
        while error_theta < -pi {
            error_theta += 2.0 * pi;
        }

        RoundtripMegarenaReport {
            backend: backend.name().to_string(),
            renderer: renderer_name.to_string(),
            true_x: expected_abs_x as f64,
            true_y: expected_abs_y as f64,
            true_theta,
            recovered_x: recovered_x as f64,
            recovered_y: recovered_y as f64,
            recovered_theta,
            abs_error_x,
            abs_error_y,
            fine_error_x,
            fine_error_y,
            error_theta: error_theta.abs(),
            swapped: swap,
        }
    }
}
