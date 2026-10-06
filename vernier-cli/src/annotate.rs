//! Minimal RGB drawing for debug overlays (peaks on the spectrum, decoded cells
//! and bits on the image). Hand-rolled primitives on an `Rgb` buffer — no
//! drawing-library dependency; the `image` crate is used only to encode the PNG.

use std::path::Path;

/// A mutable RGB canvas, row-major, 3 bytes per pixel.
pub struct Canvas {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
}

impl Canvas {
    /// Builds a canvas from a grayscale `0.0..=1.0` buffer (the background image).
    pub fn from_gray(width: usize, height: usize, gray: &[f64]) -> Self {
        let mut pixels = Vec::with_capacity(width * height * 3);
        for &v in gray {
            let g = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
            pixels.push(g);
            pixels.push(g);
            pixels.push(g);
        }
        Self {
            width,
            height,
            pixels,
        }
    }

    #[inline]
    fn put(&mut self, x: isize, y: isize, rgb: [u8; 3]) {
        if x < 0 || y < 0 || x as usize >= self.width || y as usize >= self.height {
            return;
        }
        let pixel_offset = (y as usize * self.width + x as usize) * 3;
        self.pixels[pixel_offset] = rgb[0];
        self.pixels[pixel_offset + 1] = rgb[1];
        self.pixels[pixel_offset + 2] = rgb[2];
    }

    /// Blends a color over a pixel with alpha `alpha` in 0..=1 (for translucent fills).
    #[inline]
    fn blend(&mut self, x: isize, y: isize, rgb: [u8; 3], alpha: f64) {
        if x < 0 || y < 0 || x as usize >= self.width || y as usize >= self.height {
            return;
        }
        let pixel_offset = (y as usize * self.width + x as usize) * 3;
        for (channel, &foreground) in rgb.iter().enumerate() {
            let background = self.pixels[pixel_offset + channel] as f64;
            let foreground = foreground as f64;
            self.pixels[pixel_offset + channel] = (background * (1.0 - alpha) + foreground * alpha)
                .round()
                .clamp(0.0, 255.0) as u8;
        }
    }

    /// Draws a hollow circle of the given radius (Bresenham-ish, thick by 1px).
    pub fn circle(&mut self, center_x: isize, center_y: isize, radius: isize, rgb: [u8; 3]) {
        let mut x = radius;
        let mut y = 0isize;
        let mut err = 0isize;
        while x >= y {
            for (delta_x, delta_y) in [
                (x, y),
                (y, x),
                (-x, y),
                (-y, x),
                (x, -y),
                (y, -x),
                (-x, -y),
                (-y, -x),
            ] {
                self.put(center_x + delta_x, center_y + delta_y, rgb);
            }
            y += 1;
            if err <= 0 {
                err += 2 * y + 1;
            }
            if err > 0 {
                x -= 1;
                err -= 2 * x + 1;
            }
        }
    }

    /// Draws a crosshair centered at (center_x, center_y) with arm length `length`.
    pub fn cross(&mut self, center_x: isize, center_y: isize, length: isize, rgb: [u8; 3]) {
        for d in -length..=length {
            self.put(center_x + d, center_y, rgb);
            self.put(center_x, center_y + d, rgb);
        }
    }

    /// Fills a small square centered at (center_x, center_y), half-size `half_size`, with alpha blend.
    pub fn fill_square(
        &mut self,
        center_x: isize,
        center_y: isize,
        half_size: isize,
        rgb: [u8; 3],
        alpha: f64,
    ) {
        for delta_y in -half_size..=half_size {
            for delta_x in -half_size..=half_size {
                self.blend(center_x + delta_x, center_y + delta_y, rgb, alpha);
            }
        }
    }

    /// Saves the canvas as an RGB PNG.
    pub fn save_png(&self, path: &Path) -> Result<(), String> {
        let buf =
            image::RgbImage::from_raw(self.width as u32, self.height as u32, self.pixels.clone())
                .ok_or_else(|| "canvas size mismatch".to_string())?;
        buf.save(path)
            .map_err(|e| format!("failed to save {}: {e}", path.display()))
    }
}

/// A few named colors for overlays.
pub mod color {
    pub const RED: [u8; 3] = [255, 40, 40];
    pub const GREEN: [u8; 3] = [40, 220, 40];
    pub const BLUE: [u8; 3] = [60, 120, 255];
    pub const YELLOW: [u8; 3] = [255, 220, 40];
    pub const CYAN: [u8; 3] = [40, 220, 220];
}
