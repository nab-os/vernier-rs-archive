//! Debug overlay rendering: turn a `Detection` + decode result into annotated
//! images you can look at — the spectrum with the detected carriers marked, and
//! the input image with the decoded cell grid and per-cell bits drawn on top.
//!
//! This is the visual counterpart to the printed pose: instead of trusting the
//! numbers, you see where the detector locked on and which cells it read as
//! present/absent. It is what catches a wrong-peak lock or a misread bit at a
//! glance.

use std::path::Path;

use vernier_core::buffer::BufferLayout;
use vernier_core::{Complex32, ComputeBackend, ComputeJob};
use vernier_cpu::CpuBackend;
use vernier_spectral::spectrum::Detection;

use crate::annotate::{Canvas, color};
use crate::pgm;

/// Signed frequency of a bin (bins past N/2 are negative).
fn signed(f: usize, n: usize) -> isize {
    let (f, n) = (f as isize, n as isize);
    if f > n / 2 { f - n } else { f }
}

/// Renders the FFT-magnitude spectrum (log, fftshifted) with the two detected
/// carrier peaks and the low-frequency exclusion disk marked.
pub fn render_spectrum_debug(
    image_gray: &[f64],
    width: usize,
    height: usize,
    detection: &Detection,
    min_radius: usize,
    out: &Path,
) -> Result<(), String> {
    let backend = CpuBackend::new();
    let layout = BufferLayout::packed(width, height);
    let complex: Vec<Complex32> = image_gray
        .iter()
        .map(|&v| Complex32::new(v as f32, 0.0))
        .collect();
    let mut buf = backend
        .upload(&complex, layout)
        .map_err(|e| format!("{e:?}"))?;
    {
        let mut job = backend.begin().map_err(|e| format!("{e:?}"))?;
        job.fft2d(&mut buf).map_err(|e| format!("{e:?}"))?;
        job.submit().map_err(|e| format!("{e:?}"))?;
    }
    let spec = backend.download(&buf).map_err(|e| format!("{e:?}"))?;

    // Log magnitude, fftshifted, normalized to 0..1 for the background.
    let mag: Vec<f64> = spec
        .iter()
        .map(|c| (1.0 + c.norm_sqr().sqrt()).ln())
        .collect();
    let shifted = pgm::fftshift(width, height, &mag);
    let (lo, hi) = shifted
        .iter()
        .fold((f64::MAX, f64::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    let span = if hi > lo { hi - lo } else { 1.0 };
    let norm: Vec<f64> = shifted.iter().map(|&v| (v - lo) / span).collect();

    let mut canvas = Canvas::from_gray(width, height, &norm);
    let (center_x, center_y) = ((width / 2) as isize, (height / 2) as isize);

    // Low-frequency exclusion disk (cyan): the region peak search ignores.
    canvas.circle(center_x, center_y, min_radius as isize, color::CYAN);

    // The two detected carriers, in fftshifted coordinates (DC at center).
    for (dir, draw_color) in [
        (&detection.dir1, color::RED),
        (&detection.dir2, color::YELLOW),
    ] {
        let (bx, by) = dir.peak_bin;
        let sx = signed(bx, width);
        let sy = signed(by, height);
        // shifted position = center + signed freq; also mark the conjugate.
        canvas.circle(center_x + sx, center_y + sy, 6, draw_color);
        canvas.cross(center_x + sx, center_y + sy, 4, draw_color);
        canvas.circle(center_x - sx, center_y - sy, 4, draw_color); // conjugate twin (dimmer mark)
    }

    canvas.save_png(out)
}

/// Renders the input image with the decoded cell grid and per-cell bits drawn
/// on top: each coding cell marked green (present / bit 1) or red (absent / bit
/// 0), and the cell lattice faintly overlaid.
///
/// The cell position for each pixel comes from the phase maps; we draw a marker
/// at each coding cell center. Because the bit decision is per coding column/row
/// (not per individual cell), we color a coding cell by its axis bit.
pub fn render_decode_debug(
    image_gray: &[f64],
    width: usize,
    height: usize,
    detection: &Detection,
    x_bits: &std::collections::BTreeMap<i64, u8>,
    y_bits: &std::collections::BTreeMap<i64, u8>,
    out: &Path,
) -> Result<(), String> {
    use vernier_core::scalar::consts::TAU;

    let mut canvas = Canvas::from_gray(width, height, image_gray);

    // Walk the image; for each pixel compute its 2D cell. Mark a small dot at
    // pixels that sit near a cell center (so the lattice of dot centers shows),
    // colored by whether that cell's coding axis read present/absent.
    let phase_x = &detection.phase1;
    let phase_y = &detection.phase2;

    // To avoid overdrawing, only stamp once per cell: track stamped cells.
    let mut stamped: std::collections::BTreeSet<(i64, i64)> = std::collections::BTreeSet::new();

    for row in 0..height {
        for col in 0..width {
            let flat_index = row * width + col;
            let fx = phase_x[flat_index] / TAU;
            let fy = phase_y[flat_index] / TAU;
            let cell_x = fx.round();
            let cell_y = fy.round();
            let within_cell_radius = (fx - cell_x).abs().max((fy - cell_y).abs());
            // Only stamp at cell centers (small within-cell radius).
            if within_cell_radius > 0.12 {
                continue;
            }
            let cell = (cell_x as i64, cell_y as i64);
            if !stamped.insert(cell) {
                continue;
            }
            // Determine if this is a coding cell and its bit. A cell is coding
            // along x if its x index %3==1; color by the x bit. Likewise y.
            let tx = cell.0.div_euclid(3);
            let ty = cell.1.div_euclid(3);
            let x_coding = cell.0.rem_euclid(3) == 1;
            let y_coding = cell.1.rem_euclid(3) == 1;

            let draw_color = if x_coding && y_coding {
                // both coding: blue, the absolute-code corner cells
                color::BLUE
            } else if x_coding {
                match x_bits.get(&tx) {
                    Some(1) => color::GREEN,
                    Some(0) => color::RED,
                    _ => continue, // no bit here, or an out-of-range value
                }
            } else if y_coding {
                match y_bits.get(&ty) {
                    Some(1) => color::GREEN,
                    Some(0) => color::RED,
                    _ => continue,
                }
            } else {
                // non-coding carrier cell: faint marker
                color::CYAN
            };
            canvas.fill_square(col as isize, row as isize, 2, draw_color, 0.55);
        }
    }

    canvas.save_png(out)
}
