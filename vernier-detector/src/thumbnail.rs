//! Bitmap thumbnail extraction + template matching — the model-reduction step
//! behind the bitmap detector.
//!
//! Ports C++ `vernier::BitmapThumbnail` and the `cv::matchTemplate(TM_CCOEFF)` /
//! `cv::minMaxLoc` calls used by `BitmapPatternDetector`, in pure Rust (no
//! OpenCV).

use vernier_core::scalar::consts::PI;
use vernier_spectral::PhasePlane;

/// A reduced, dewarped model of the observed pattern: each pattern lattice node
/// is condensed to one thumbnail pixel by averaging the image samples that fall
/// near it (in phase). Mirrors C++ `BitmapThumbnail`.
pub struct BitmapThumbnail {
    /// Side length in pixels (odd).
    pub size: usize,
    /// Grayscale thumbnail, row-major `size × size` (row = phase-2 iteration,
    /// col = phase-1 iteration, matching C++ `thumbnail(row, col)`).
    pub thumbnail: Vec<u8>,
    /// Binary (thresholded) thumbnail, same layout.
    pub binary: Vec<u8>,
    /// Half-width (radians) of the phase window that counts a sample as sitting
    /// on a lattice node. C++ `DELTA_PHASE = π/4`.
    pub delta_phase: f64,
}

impl BitmapThumbnail {
    /// Creates a thumbnail buffer of the given (odd) side length.
    pub fn new(size: usize) -> Self {
        assert!(size % 2 == 1, "the size of BitmapThumbnail must be odd");
        Self {
            size,
            thumbnail: vec![0; size * size],
            binary: vec![0; size * size],
            delta_phase: (PI / 4.0),
        }
    }

    /// Computes the thumbnail from the spatial image `pixels` (row-major,
    /// `width × height`) and the two fitted phase planes.
    ///
    /// Faithful port of `BitmapThumbnail::compute`.
    pub fn compute(
        &mut self,
        pixels: &[f32],
        width: usize,
        height: usize,
        plane1: &PhasePlane,
        plane2: &PhasePlane,
    ) {
        let n = self.size;
        let mut number = vec![0.0f64; n * n];
        let mut cumul = vec![0.0f64; n * n];

        let (a1, b1, c1) = (plane1.a, plane1.b, plane1.c);
        let (a2, b2, c2) = (plane2.a, plane2.b, plane2.c);
        let half_w = (width / 2) as i64;
        let half_h = (height / 2) as i64;
        let pi = PI;

        for row in 0..height {
            let y = (row as i64 - half_h) as f64;
            for col in 0..width {
                let x = (col as i64 - half_w) as f64;
                let phase_col = a1 * x + b1 * y + c1;
                let phase_row = a2 * x + b2 * y + c2;

                let it1 = (phase_col / pi).round() as i64 + (n / 2) as i64;
                let it2 = (phase_row / pi).round() as i64 + (n / 2) as i64;

                if it1 >= 0 && it2 >= 0 && (it1 as usize) < n && (it2 as usize) < n {
                    // Truncated remainder matches C std::fmod.
                    let mc = (phase_col % pi).abs();
                    let mr = (phase_row % pi).abs();
                    let on_node_c = mc <= self.delta_phase || mc >= pi - self.delta_phase;
                    let on_node_r = mr <= self.delta_phase || mr >= pi - self.delta_phase;
                    if on_node_c && on_node_r {
                        let acc = it1 as usize * n + it2 as usize;
                        number[acc] += 1.0;
                        cumul[acc] += pixels[row * width + col] as f64;
                    }
                }
            }
        }

        // Fill thumbnail(row, col) = 255 * cumul(col, row) / number(col, row).
        let (mut mean_fg, mut cnt_fg) = (0.0f64, 0usize);
        let (mut mean_bg, mut cnt_bg) = (0.0f64, 0usize);
        for row in 0..n {
            for col in 0..n {
                let acc = col * n + row;
                let v = if number[acc] > 0.0 {
                    (255.0 * cumul[acc] / number[acc]).clamp(0.0, 255.0)
                } else {
                    0.0
                };
                let v = v as u8;
                self.thumbnail[row * n + col] = v;
                if row % 2 == 0 && col % 2 == 0 {
                    mean_fg += v as f64;
                    cnt_fg += 1;
                } else {
                    mean_bg += v as f64;
                    cnt_bg += 1;
                }
            }
        }
        if cnt_fg > 0 {
            mean_fg /= cnt_fg as f64;
        }
        if cnt_bg > 0 {
            mean_bg /= cnt_bg as f64;
        }
        let threshold = ((mean_bg + mean_fg) / 2.0) as u8;
        for i in 0..n * n {
            self.binary[i] = if self.thumbnail[i] > threshold {
                255
            } else {
                0
            };
        }
    }
}

/// Rotates a row-major grayscale image 90° clockwise. Returns
/// `(rotated_pixels, new_width, new_height)` where `new_width = height` and
/// `new_height = width`.
pub fn rotate90_cw(pixels: &[u8], width: usize, height: usize) -> (Vec<u8>, usize, usize) {
    let (nw, nh) = (height, width);
    let mut out = vec![0u8; nw * nh];
    for row in 0..height {
        for col in 0..width {
            // (row, col) -> (col, height - 1 - row) in the rotated image.
            let nr = col;
            let nc = height - 1 - row;
            out[nr * nw + nc] = pixels[row * width + col];
        }
    }
    (out, nw, nh)
}

/// Result of a template match: the correlation surface plus the location of its
/// maximum.
pub struct MatchResult {
    /// Correlation surface, row-major `result_width × result_height`.
    pub surface: Vec<f64>,
    /// Surface width `image_width - template_width + 1`.
    pub width: usize,
    /// Surface height `image_height - template_height + 1`.
    pub height: usize,
    /// Column of the maximum (`cv::minMaxLoc` `maxLoc.x`).
    pub max_x: usize,
    /// Row of the maximum (`maxLoc.y`).
    pub max_y: usize,
    /// Maximum correlation value.
    pub max_val: f64,
}

/// Correlation-coefficient template match (`cv::TM_CCOEFF`) of `template` over
/// `image`, followed by `minMaxLoc`. Both are row-major grayscale.
///
/// Returns `None` if the template does not fit inside the image.
pub fn match_template_ccoeff(
    image: &[u8],
    iw: usize,
    ih: usize,
    template: &[u8],
    tw: usize,
    th: usize,
) -> Option<MatchResult> {
    if tw == 0 || th == 0 || tw > iw || th > ih {
        return None;
    }
    let rw = iw - tw + 1;
    let rh = ih - th + 1;

    let area = (tw * th) as f64;
    let t_mean = template.iter().map(|&t| t as f64).sum::<f64>() / area;
    let t_centered: Vec<f64> = template.iter().map(|&t| t as f64 - t_mean).collect();

    let mut surface = vec![0.0f64; rw * rh];
    let (mut max_val, mut max_x, mut max_y) = (f64::NEG_INFINITY, 0usize, 0usize);

    for ry in 0..rh {
        for rx in 0..rw {
            // Local image mean under the template window.
            let mut i_mean = 0.0;
            for j in 0..th {
                let base = (ry + j) * iw + rx;
                for i in 0..tw {
                    i_mean += image[base + i] as f64;
                }
            }
            i_mean /= area;

            let mut s = 0.0;
            for j in 0..th {
                let base = (ry + j) * iw + rx;
                let trow = j * tw;
                for i in 0..tw {
                    s += t_centered[trow + i] * (image[base + i] as f64 - i_mean);
                }
            }
            surface[ry * rw + rx] = s;
            if s > max_val {
                max_val = s;
                max_x = rx;
                max_y = ry;
            }
        }
    }

    Some(MatchResult {
        surface,
        width: rw,
        height: rh,
        max_x,
        max_y,
        max_val,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotate90_cw_known() {
        // 2x3 image (w=3,h=2):
        // 1 2 3
        // 4 5 6
        // rotate CW -> 3x2 (w=2,h=3):
        // 4 1
        // 5 2
        // 6 3
        let (out, w, h) = rotate90_cw(&[1, 2, 3, 4, 5, 6], 3, 2);
        assert_eq!((w, h), (2, 3));
        assert_eq!(out, vec![4, 1, 5, 2, 6, 3]);
    }

    #[test]
    fn match_template_identity_centers() {
        // Template equal to image -> 1x1 surface, max at (0,0).
        let img = vec![10u8, 20, 30, 40];
        let m = match_template_ccoeff(&img, 2, 2, &img, 2, 2).unwrap();
        assert_eq!((m.width, m.height), (1, 1));
        assert_eq!((m.max_x, m.max_y), (0, 0));
    }

    #[test]
    fn match_template_finds_offset_peak() {
        // 3x3 image with a distinctive diagonal 2x2 block at bottom-right;
        // template = that block. (TM_CCOEFF needs a non-constant template — a
        // constant one has zero mean-subtracted correlation everywhere.)
        #[rustfmt::skip]
        let img = vec![
            0u8,   0,   0,
            0,   255,   0,
            0,     0, 255,
        ];
        let tmpl = vec![255u8, 0, 0, 255];
        let m = match_template_ccoeff(&img, 3, 3, &tmpl, 2, 2).unwrap();
        assert_eq!((m.width, m.height), (2, 2));
        // Best alignment is the bottom-right corner (1,1).
        assert_eq!((m.max_x, m.max_y), (1, 1));
    }
}
