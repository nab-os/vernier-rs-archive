//! Real image I/O for the CLI: load photos (JPEG/PNG/BMP/TIFF) into the
//! grayscale `f32` buffers the pipeline consumes, and save grayscale buffers
//! back out. The `image` crate dependency lives only here — the library crates
//! stay codec-free.

use std::path::Path;

use image::GenericImageView;

/// A loaded grayscale image: row-major `f32` intensities in `0.0..=1.0`.
pub struct LoadedImage {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Row-major intensities, normalized to `0.0..=1.0`.
    pub data: Vec<f32>,
}

/// Loads any supported image file and converts it to normalized grayscale.
///
/// Color images are converted to luminance. 8-bit and 16-bit inputs are both
/// normalized to `0.0..=1.0`, so the dynamic range is consistent regardless of
/// bit depth — important because the Vernier targets are often imaged at 12-bit
/// depth (stored in 16-bit containers).
pub fn load_grayscale(path: &Path) -> Result<LoadedImage, String> {
    let img = image::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
    let (img_width, img_height) = img.dimensions();
    let (width, height) = (img_width as usize, img_height as usize);

    // Convert to 16-bit luma to preserve depth, then normalize.
    let luma16 = img.to_luma16();
    let data: Vec<f32> = luma16
        .pixels()
        .map(|p| p.0[0] as f32 / u16::MAX as f32)
        .collect();

    Ok(LoadedImage {
        width,
        height,
        data,
    })
}

/// Saves a row-major grayscale `f32` buffer (`0.0..=1.0`) as an 8-bit PNG.
///
/// Used to write annotated and result images.
pub fn save_grayscale_png(
    path: &Path,
    width: usize,
    height: usize,
    data: &[f32],
) -> Result<(), String> {
    let bytes: Vec<u8> = data
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    let buf = image::GrayImage::from_raw(width as u32, height as u32, bytes)
        .ok_or_else(|| "buffer size does not match dimensions".to_string())?;
    buf.save(path)
        .map_err(|e| format!("failed to save {}: {e}", path.display()))
}
