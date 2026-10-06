//! Generates the explainer figures and measurements for the coded checkerboard
//! (`vernier-patterns::checkerboard`), so the write-up is reproducible rather
//! than hand-drawn.
//!
//! Beyond the pictures it measures the one thing the design has to get right:
//! how far the absolute code moves the measured carrier phase. That is done by
//! evaluating a single exact DFT bin — the image side is a whole number of
//! carrier periods, so the carrier itself leaks nothing and any phase shift
//! between the coded and uncoded renders is the code's doing.

use std::f64::consts::TAU;
use std::path::{Path, PathBuf};

use vernier_core::{ComputeBackend, ComputeJob};
use vernier_cpu::CpuBackend;
use vernier_patterns::checkerboard::{CELL, Checkerboard, CodeAxis};
use vernier_patterns::megarena::Megarena;
use vernier_patterns::render::{into_pattern_frame, render_with};
use vernier_patterns::{PatternPose, PatternPose as Pose};

use crate::imageio;

pub struct CheckerboardFiguresArgs {
    /// Directory the PNGs are written to.
    pub out_dir: PathBuf,
    /// Checkerboard square side in pixels.
    pub square_px: f64,
    /// LFSR order.
    pub code_size: u32,
    /// Side of the square figures, in pixels.
    pub size: usize,
    /// Number of random poses used for the phase-bias measurement.
    pub poses: usize,
}

/// Tint for x-code sites (blue) and y-code sites (amber). Chosen to stay
/// distinguishable in greyscale print and for the common colour deficiencies.
const X_TINT: [f64; 3] = [0.23, 0.51, 0.96];
const Y_TINT: [f64; 3] = [0.96, 0.62, 0.07];

pub fn run(args: &CheckerboardFiguresArgs) -> Result<(), String> {
    std::fs::create_dir_all(&args.out_dir)
        .map_err(|e| format!("cannot create {}: {e}", args.out_dir.display()))?;

    let pattern = Checkerboard::new(args.square_px, args.code_size)
        .ok_or_else(|| format!("unsupported code size {}", args.code_size))?;

    // Pattern coordinates equal pixel coordinates: squares land on pixel
    // boundaries, so the annotated figures are crisp.
    let aligned = Pose::new(-(args.size as f64) / 2.0, -(args.size as f64) / 2.0, 0.0);

    let coded = pattern.render(args.size, args.size, &aligned);
    let plain = pattern.render_plain(args.size, args.size, &aligned);
    save_gray(
        &args.out_dir.join("coded.png"),
        args.size,
        args.size,
        coded.as_slice(),
    )?;
    save_gray(
        &args.out_dir.join("plain.png"),
        args.size,
        args.size,
        plain.as_slice(),
    )?;

    // The same field at half the square count, with the coding sites called
    // out. At the full density the markers are larger than the squares and the
    // picture stops being readable.
    let marked_pattern =
        Checkerboard::new(args.square_px * 2.0, args.code_size).ok_or("unsupported code size")?;
    let marked_pose = Pose::new(-(args.size as f64) / 2.0, -(args.size as f64) / 2.0, 0.0);
    let marked = marked_pattern.render(args.size, args.size, &marked_pose);
    let annotated = annotate_sites(&marked_pattern, args.size, args.size, marked.as_slice());
    save_rgb(
        &args.out_dir.join("annotated.png"),
        args.size,
        args.size,
        &annotated,
    )?;

    // A close-up: 4×4 supercells, big enough to count squares by eye.
    let zoom_square = 24.0;
    let zoom_cells = 4;
    let zoom_size = (zoom_square * (CELL * zoom_cells) as f64) as usize;
    let zoom_pattern = Checkerboard::new(zoom_square, args.code_size)
        .ok_or("unsupported code size")?
        .with_supersample(8);
    let zoom_pose = Pose::new(-(zoom_size as f64) / 2.0, -(zoom_size as f64) / 2.0, 0.0);
    let zoom = zoom_pattern.render(zoom_size, zoom_size, &zoom_pose);
    let zoom_rgb = annotate_sites(&zoom_pattern, zoom_size, zoom_size, zoom.as_slice());
    save_rgb(
        &args.out_dir.join("zoom.png"),
        zoom_size,
        zoom_size,
        &zoom_rgb,
    )?;

    // Megarena at a matched carrier period, for the side-by-side and the
    // spectrum comparison: a megarena fringe every `square_px·√2` pixels.
    let megarena_period = pattern.carrier_period_px();
    let megarena = Megarena::new(megarena_period, args.code_size)
        .ok_or("unsupported code size for megarena")?;
    let megarena_image = megarena.render(args.size, args.size, &PatternPose::IDENTITY);
    save_gray(
        &args.out_dir.join("megarena.png"),
        args.size,
        args.size,
        megarena_image.as_slice(),
    )?;

    let backend = CpuBackend::new();
    save_spectrum(
        &backend,
        &args.out_dir.join("spectrum-checkerboard.png"),
        &coded,
    )?;
    save_spectrum(
        &backend,
        &args.out_dir.join("spectrum-megarena.png"),
        &megarena_image,
    )?;

    report_metrics(&pattern, &megarena, args);
    println!("figures written to {}", args.out_dir.display());
    Ok(())
}

/// Marks the coding sites on the rendered pattern. Every site gets a coloured
/// ring — blue for the x code, amber for the y code — and a site that is
/// *inverted* (a `0` bit) additionally gets a filled centre dot. A ring reads
/// the same on a black square and a white one, which a tint does not.
fn annotate_sites(pattern: &Checkerboard, width: usize, height: usize, gray: &[f32]) -> Vec<u8> {
    let square = pattern.square_px;
    let ring = (0.12 * square).max(1.0) / square; // ring width, in square units
    let mut rgb = vec![0u8; width * height * 3];

    for row in 0..height {
        for col in 0..width {
            let value = gray[row * width + col] as f64;
            let (x, y) = (col as f64, row as f64);
            let (i, j) = pattern.square_at(x, y);
            let axis = Checkerboard::coding_axis(i, j);

            // Position within the square, centred on 0, in square units.
            let (u, v) = (x / square - i as f64 - 0.5, y / square - j as f64 - 0.5);
            let edge = 0.5 - u.abs().max(v.abs()); // distance to the square's border
            let radius = (u * u + v * v).sqrt();

            let (tint, strength) = match axis {
                Some(code_axis) => {
                    let colour = if code_axis == CodeAxis::X {
                        X_TINT
                    } else {
                        Y_TINT
                    };
                    let inverted = pattern.square_inverted(i, j);
                    if edge < ring || (inverted && radius < 0.22) {
                        (colour, 1.0)
                    } else {
                        (colour, 0.14)
                    }
                }
                None => ([0.0; 3], 0.0),
            };

            for channel in 0..3 {
                let blended = value * (1.0 - strength) + tint[channel] * strength;
                rgb[(row * width + col) * 3 + channel] = to_u8(blended);
            }
        }
    }
    rgb
}

/// Half-width, in frequency bins, of the spectrum crop written out. Wide enough
/// to hold the fundamentals and their first harmonics.
const SPECTRUM_CROP: usize = 48;

/// Writes a log-magnitude, DC-centred spectrum crop. Two diagonal peaks for the
/// checkerboard, two axis-aligned ones for the megarena; the code shows up as
/// the pedestal around them, never as a shift of the peak itself.
///
/// The image is Hann-windowed first. Without it the rectangular aperture throws
/// a bright cross through DC that swamps everything — an artefact of the crop,
/// not of the pattern. The display range is set from the brightest bin *outside*
/// DC for the same reason.
fn save_spectrum(
    backend: &CpuBackend,
    path: &Path,
    image: &vernier_core::GrayImage,
) -> Result<(), String> {
    let (width, height) = (image.width(), image.height());

    let mut windowed = vernier_core::GrayImage::zeros(width, height);
    {
        let source = image.as_slice();
        let target = windowed.as_mut_slice();
        for row in 0..height {
            let wy = hann(row, height);
            for col in 0..width {
                target[row * width + col] =
                    (source[row * width + col] as f64 * wy * hann(col, width)) as f32;
            }
        }
    }

    let mut buffer = backend
        .upload_real(&windowed)
        .map_err(|e| format!("upload failed: {e}"))?;
    {
        let mut job = backend.begin().map_err(|e| format!("begin failed: {e}"))?;
        job.fft2d(&mut buffer)
            .map_err(|e| format!("fft failed: {e}"))?;
        job.submit().map_err(|e| format!("submit failed: {e}"))?;
    }
    let data = backend
        .download(&buffer)
        .map_err(|e| format!("download failed: {e}"))?;

    let side = 2 * SPECTRUM_CROP;
    let mut crop = vec![0f64; side * side];
    let mut peak = f64::MIN;
    for row in 0..side {
        for col in 0..side {
            // Bin offsets from DC, wrapped into the FFT's own indexing.
            let (dy, dx) = (
                row as isize - SPECTRUM_CROP as isize,
                col as isize - SPECTRUM_CROP as isize,
            );
            let source_row = (dy.rem_euclid(height as isize)) as usize;
            let source_col = (dx.rem_euclid(width as isize)) as usize;
            let magnitude = data[source_row * width + source_col].norm().max(1e-12).ln();
            crop[row * side + col] = magnitude;
            // The window spreads DC over its immediate neighbourhood; scaling to
            // that would crush the carriers to black, so the display range comes
            // from outside it.
            if dx * dx + dy * dy > DC_EXCLUSION * DC_EXCLUSION {
                peak = peak.max(magnitude);
            }
        }
    }

    let floor = peak - 6.0;
    // Nearest-neighbour ×2 so single-bin peaks survive being looked at.
    let scaled = side * 2;
    let mut normalized = vec![0f32; scaled * scaled];
    for row in 0..scaled {
        for col in 0..scaled {
            let value = crop[(row / 2) * side + col / 2];
            normalized[row * scaled + col] =
                (((value - floor) / (peak - floor)).clamp(0.0, 1.0)) as f32;
        }
    }
    save_gray(path, scaled, scaled, &normalized)
}

/// Bins within this radius of DC are left out of the display scaling.
const DC_EXCLUSION: isize = 8;

fn hann(index: usize, length: usize) -> f64 {
    0.5 - 0.5 * (TAU * index as f64 / length as f64).cos()
}

/// Measures, for both pattern families, the fill ratio, the carrier amplitude,
/// and — the point of the exercise — how much the absolute code displaces the
/// measured carrier phase.
fn report_metrics(pattern: &Checkerboard, megarena: &Megarena, args: &CheckerboardFiguresArgs) {
    let size = args.size;
    let square = pattern.square_px;

    // The checkerboard carrier sits at (N/2a, N/2a) cycles over the window; for
    // that to be an exact DFT bin the window must hold a whole number of carrier
    // periods. Warn rather than lie if the caller picked sizes that don't.
    let bin = size as f64 / (2.0 * square);
    if (bin - bin.round()).abs() > 1e-9 {
        println!(
            "note: size/{:.3} is not an integer number of carrier periods; \
             phase-bias figures include window leakage",
            2.0 * square
        );
    }
    let bin = bin.round();

    let mut checker_error = Vec::with_capacity(args.poses);
    let mut megarena_error = Vec::with_capacity(args.poses);
    let mut checker_amplitude = 0.0;
    let mut plain_amplitude = 0.0;
    let mut amplitude_range = (f64::MAX, 0.0f64);

    // Spread the poses over a whole code period. How much carrier the code costs
    // depends on how many 0 bits are currently in view, and that varies a lot
    // locally — a window parked on the sequence's run of ones loses almost
    // nothing. Only an average over the whole board is a fair number.
    let span = (CELL * pattern.code().len() as i64) as f64 * square;
    for index in 0..args.poses {
        let step = index as f64 / args.poses as f64;
        let pose = Pose::new(step * span + 0.21, step * 0.61 * span - 0.47, 0.0);

        let coded = pattern.render(size, size, &pose);
        let plain = pattern.render_plain(size, size, &pose);
        let coded_bin = dft_bin(coded.as_slice(), size, size, bin, bin);
        let plain_bin = dft_bin(plain.as_slice(), size, size, bin, bin);
        checker_error.push(wrap_pi(coded_bin.1 - plain_bin.1));
        checker_amplitude += coded_bin.0;
        plain_amplitude += plain_bin.0;
        let kept = coded_bin.0 / plain_bin.0;
        amplitude_range = (amplitude_range.0.min(kept), amplitude_range.1.max(kept));

        // Megarena, matched carrier period, carrier on the x axis.
        let period = megarena.period_px;
        let coded_m = megarena.render(size, size, &pose);
        let plain_m = render_plain_megarena(size, size, &pose, period);
        let coded_m_bin = dft_bin(coded_m.as_slice(), size, size, size as f64 / period, 0.0);
        let plain_m_bin = dft_bin(plain_m.as_slice(), size, size, size as f64 / period, 0.0);
        megarena_error.push(wrap_pi(coded_m_bin.1 - plain_m_bin.1));
    }

    let poses = args.poses as f64;
    checker_amplitude /= poses;
    plain_amplitude /= poses;

    println!(
        "— coded checkerboard (square {square} px, order {}) —",
        args.code_size
    );
    println!(
        "  carrier period      : {:.3} px",
        pattern.carrier_period_px()
    );
    println!(
        "  absolute range      : {} squares",
        pattern.range_squares()
    );
    println!("  fill (white share)  : {:.4}", fill_ratio(pattern));
    println!(
        "  carrier amplitude   : {plain_amplitude:.4} uncoded → {checker_amplitude:.4} coded  \
         ({:.1}% kept on average, {:.1}–{:.1}% across the board)",
        100.0 * checker_amplitude / plain_amplitude,
        100.0 * amplitude_range.0,
        100.0 * amplitude_range.1,
    );
    report_phase(
        "  code-induced phase  ",
        &checker_error,
        pattern.carrier_period_px(),
    );
    println!("— megarena at the same carrier period —");
    report_phase(
        "  code-induced phase  ",
        &megarena_error,
        megarena.period_px,
    );
}

fn report_phase(label: &str, errors: &[f64], period_px: f64) {
    let n = errors.len() as f64;
    let rms = (errors.iter().map(|e| e * e).sum::<f64>() / n).sqrt();
    let worst = errors.iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()));
    // A phase error ε maps to a position error ε·period/2π along the carrier.
    let to_px = period_px / TAU;
    println!(
        "{label}: rms {:.3} mrad ({:.4} px), max {:.3} mrad ({:.4} px)",
        rms * 1e3,
        rms * to_px,
        worst * 1e3,
        worst * to_px
    );
}

/// Share of white area over one full code period — the 50/50 claim.
fn fill_ratio(pattern: &Checkerboard) -> f64 {
    let n = pattern.range_squares().min(3 * 255);
    let mut white = 0i64;
    for i in 0..n {
        for j in 0..n {
            if pattern.square_is_white(i, j) {
                white += 1;
            }
        }
    }
    white as f64 / (n * n) as f64
}

/// The uncoded megarena carrier: the same dot grid with every period present.
fn render_plain_megarena(
    width: usize,
    height: usize,
    pose: &Pose,
    period: f64,
) -> vernier_core::GrayImage {
    let center_x = width as f64 / 2.0;
    let center_y = height as f64 / 2.0;
    render_with(width, height, |px, py| {
        let (xp, yp) = into_pattern_frame(px, py, center_x, center_y, pose.theta);
        let (x, y) = (xp - pose.x, yp - pose.y);
        let cx = 0.5 + 0.5 * (TAU * x / period).cos();
        let cy = 0.5 + 0.5 * (TAU * y / period).cos();
        cx * cy
    })
}

/// One exact DFT bin at `(fx, fy)` cycles per image width/height. Returns
/// `(amplitude, phase)`, the amplitude normalized by the pixel count.
fn dft_bin(image: &[f32], width: usize, height: usize, fx: f64, fy: f64) -> (f64, f64) {
    let (mut re, mut im) = (0.0, 0.0);
    for row in 0..height {
        for col in 0..width {
            let value = image[row * width + col] as f64;
            let angle = -TAU * (fx * col as f64 / width as f64 + fy * row as f64 / height as f64);
            re += value * angle.cos();
            im += value * angle.sin();
        }
    }
    let count = (width * height) as f64;
    ((re * re + im * im).sqrt() * 2.0 / count, im.atan2(re))
}

fn wrap_pi(angle: f64) -> f64 {
    (angle + std::f64::consts::PI).rem_euclid(TAU) - std::f64::consts::PI
}

fn to_u8(value: f64) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn save_gray(path: &Path, width: usize, height: usize, data: &[f32]) -> Result<(), String> {
    imageio::save_grayscale_png(path, width, height, data)
}

fn save_rgb(path: &Path, width: usize, height: usize, rgb: &[u8]) -> Result<(), String> {
    image::RgbImage::from_raw(width as u32, height as u32, rgb.to_vec())
        .ok_or_else(|| "bad RGB buffer size".to_string())?
        .save(path)
        .map_err(|e| format!("failed to write {}: {e}", path.display()))
}
