//! `calibrate`: intrinsics from photos of the coded checkerboard, plus the
//! camera file and the report shared with `calibrate-webcam`.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use vernier_camera::{Calibration, Camera, Model, Target, View, calibrate, measure_view};

use crate::imageio;

/// What `calibrate` needs, already parsed from the command line.
pub struct CalibrateArgs {
    pub images: Vec<PathBuf>,
    pub target: Target,
    pub model: Model,
    pub output: PathBuf,
}

/// What `calibrate` writes and `solve-pnp` reads. The camera fields sit at the
/// top level, named as OpenCV names them.
#[derive(Serialize, Deserialize)]
pub struct CameraFile {
    #[serde(flatten)]
    pub camera: Camera,
    /// Reprojection error of the calibration, px.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rms: Option<f64>,
    /// How many views the calibration used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub views: Option<usize>,
}

/// The lens model named on the command line.
pub fn model(name: &str) -> Result<Model, String> {
    Model::parse(name)
        .ok_or_else(|| format!("unknown camera model '{name}'; try pinhole or fisheye"))
}

/// Loads a photo and finds the board's points in it.
pub fn measure_file(path: &Path, target: &Target) -> Result<View, String> {
    let image = imageio::load_grayscale(path)?;
    measure_view(&image.data, image.width, image.height, target).map_err(|e| e.to_string())
}

/// Reads a camera file, checking it has as many distortion coefficients as its
/// model takes.
pub fn load_camera(path: &Path) -> Result<Camera, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    let file: CameraFile =
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let camera = file.camera;
    let expected = camera.model.distortion_len();
    if camera.distortion.len() != expected {
        return Err(format!(
            "{}: a {} camera has {expected} distortion coefficients",
            path.display(),
            camera.model.name(),
        ));
    }
    Ok(camera)
}

/// Measures every photo, skipping those where the board is not found,
/// calibrates from the rest and writes the camera file.
pub fn run(args: &CalibrateArgs) -> Result<(), String> {
    let mut views = Vec::new();
    let mut names = Vec::new();
    for path in &args.images {
        match measure_file(path, &args.target) {
            Ok(view) => {
                eprintln!(
                    "{}: {} points, {}",
                    path.display(),
                    view.points.len(),
                    code_status(&view)
                );
                views.push(view);
                names.push(path.display().to_string());
            }
            Err(e) => eprintln!("{}: skipped, {e}", path.display()),
        }
    }
    let result = calibrate(&views, args.model).map_err(|e| e.to_string())?;
    report(&result, &views, &names);
    save(&args.output, &result)
}

/// Whether the board's code was read in a view, for the progress lines.
pub fn code_status(view: &View) -> String {
    match &view.code {
        Ok(_) => "code read".to_string(),
        Err(e) => format!("code not read ({e})"),
    }
}

/// Prints the calibrated camera, then the fit overall and view by view.
/// `views` and `names` are in the order the views were calibrated from.
pub fn report(result: &Calibration, views: &[View], names: &[String]) {
    let camera = &result.camera;
    println!(
        "{} camera, {}x{}",
        camera.model.name(),
        camera.width,
        camera.height
    );
    println!(
        "  fx {:.4}  fy {:.4}  cx {:.4}  cy {:.4}",
        camera.fx, camera.fy, camera.cx, camera.cy
    );
    // Coefficient names in the order OpenCV stores them.
    let labels: &[&str] = match camera.model {
        Model::Pinhole => &["k1", "k2", "p1", "p2", "k3"],
        Model::Fisheye => &["k1", "k2", "k3", "k4"],
    };
    let terms: Vec<String> = labels
        .iter()
        .zip(&camera.distortion)
        .map(|(label, value)| format!("{label} {value:.6}"))
        .collect();
    println!("  {}", terms.join("  "));

    let used: usize = result.views.iter().map(|v| v.used).sum();
    let rejected: usize = result.views.iter().map(|v| v.rejected).sum();
    println!(
        "reprojection rms {:.4} px over {} views ({used} points, {rejected} rejected)",
        result.rms,
        result.views.len()
    );
    for ((fit, view), name) in result.views.iter().zip(views).zip(names) {
        let code_note = if view.is_absolute() {
            ""
        } else {
            ", code not read"
        };
        println!(
            "  {name}: rms {:.4} px, {} points, tilt {:.1} deg{code_note}",
            fit.rms,
            fit.used,
            fit.pose.tilt().to_degrees(),
        );
    }
}

/// Writes the calibration as a camera file at `path`.
pub fn save(path: &Path, result: &Calibration) -> Result<(), String> {
    let file = CameraFile {
        camera: result.camera.clone(),
        rms: Some(result.rms),
        views: Some(result.views.len()),
    };
    let text = serde_json::to_string_pretty(&file).map_err(|e| e.to_string())?;
    std::fs::write(path, text + "\n")
        .map_err(|e| format!("failed to write {}: {e}", path.display()))?;
    println!("wrote {}", path.display());
    Ok(())
}
