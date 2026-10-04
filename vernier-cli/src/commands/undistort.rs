//! `undistort`: resample a picture as an ideal pinhole camera would have taken
//! it, to check a calibration by eye (straight lines should come out straight).

use std::path::PathBuf;

use nalgebra::Vector3;

use super::calibrate::load_camera;
use crate::imageio;

/// What `undistort` needs, already parsed from the command line.
pub struct UndistortArgs {
    pub camera: PathBuf,
    pub image: PathBuf,
    pub output: PathBuf,
    /// Focal length of the output over the calibrated one. Below 1 keeps more of
    /// a wide field in frame.
    pub zoom: f64,
}

/// Writes the undistorted picture. Each output pixel is turned into the ray an
/// ideal pinhole camera would see it along, and that ray is projected through
/// the calibrated camera to find where to read the input.
pub fn run(args: &UndistortArgs) -> Result<(), String> {
    let camera = load_camera(&args.camera)?;
    let image = imageio::load_grayscale(&args.image)?;
    if (image.width, image.height) != (camera.width, camera.height) {
        return Err(format!(
            "image is {}x{} but the camera was calibrated at {}x{}",
            image.width, image.height, camera.width, camera.height
        ));
    }
    if args.zoom <= 0.0 {
        return Err("zoom must be positive".into());
    }
    let (width, height) = (image.width, image.height);
    // The ideal camera shares the calibrated principal point.
    let (ideal_fx, ideal_fy) = (camera.fx * args.zoom, camera.fy * args.zoom);
    let output: Vec<f32> = (0..width * height)
        .map(|index| {
            let (u, v) = ((index % width) as f64, (index / width) as f64);
            let ray = Vector3::new((u - camera.cx) / ideal_fx, (v - camera.cy) / ideal_fy, 1.0);
            camera.project(&ray).map_or(0.0, |source| {
                bilinear(&image.data, width, height, source[0], source[1])
            })
        })
        .collect();
    imageio::save_grayscale_png(&args.output, width, height, &output)?;
    println!("wrote {}", args.output.display());
    Ok(())
}

/// The image at a fractional position, interpolated between the four nearest
/// pixels; black outside the image.
fn bilinear(data: &[f32], width: usize, height: usize, x: f64, y: f64) -> f32 {
    if x < 0.0 || y < 0.0 || x > (width - 1) as f64 || y > (height - 1) as f64 {
        return 0.0;
    }
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(width - 1), (y0 + 1).min(height - 1));
    let (fraction_x, fraction_y) = ((x - x0 as f64) as f32, (y - y0 as f64) as f32);
    let at = |x: usize, y: usize| data[y * width + x];
    let top = at(x0, y0) * (1.0 - fraction_x) + at(x1, y0) * fraction_x;
    let bottom = at(x0, y1) * (1.0 - fraction_x) + at(x1, y1) * fraction_x;
    top * (1.0 - fraction_y) + bottom * fraction_y
}
