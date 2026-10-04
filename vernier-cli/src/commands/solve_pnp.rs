//! `solve-pnp`: where the board is, seen through a calibrated camera.

use std::path::{Path, PathBuf};

use vernier_camera::{Camera, Target, solve_pnp};

use super::calibrate::{load_camera, measure_file};

/// What `solve-pnp` needs, already parsed from the command line.
pub struct SolvePnpArgs {
    pub camera: PathBuf,
    pub images: Vec<PathBuf>,
    pub target: Target,
}

/// Solves and prints the board's pose in every photo. A photo that gives no
/// pose is reported and the rest still go through; the command fails at the
/// end if any did.
pub fn run(args: &SolvePnpArgs) -> Result<(), String> {
    let camera = load_camera(&args.camera)?;
    let mut failed = 0;
    for path in &args.images {
        if let Err(e) = solve_image(&camera, path, &args.target) {
            eprintln!("{}: {e}", path.display());
            failed += 1;
        }
    }
    if failed > 0 {
        return Err(format!(
            "{failed} of {} image(s) gave no pose",
            args.images.len()
        ));
    }
    Ok(())
}

/// Measures one photo, solves the board's pose and prints it.
fn solve_image(camera: &Camera, path: &Path, target: &Target) -> Result<(), String> {
    let view = measure_file(path, target)?;
    if (view.width, view.height) != (camera.width, camera.height) {
        return Err(format!(
            "image is {}x{} but the camera was calibrated at {}x{}",
            view.width, view.height, camera.width, camera.height
        ));
    }
    // Without the code the points sit on the right lattice but in an unknown
    // frame, so a pose would mean nothing.
    if let Err(e) = &view.code {
        return Err(format!(
            "code not read ({e}), so the board frame is unknown"
        ));
    }
    let fit = solve_pnp(camera, &view).map_err(|e| e.to_string())?;

    let rotation = fit.pose.rvec();
    let translation = fit.pose.translation;
    let centre = fit.pose.camera_centre();
    println!(
        "{}: {} points, rms {:.4} px, {} rejected",
        path.display(),
        fit.used,
        fit.rms,
        fit.rejected
    );
    println!(
        "  rvec [{:.6}, {:.6}, {:.6}] rad",
        rotation.x, rotation.y, rotation.z
    );
    println!(
        "  tvec [{:.4}, {:.4}, {:.4}]",
        translation.x, translation.y, translation.z
    );
    println!(
        "  camera at [{:.4}, {:.4}, {:.4}] in the board frame, {:.4} from its origin, tilt {:.2} deg",
        centre.x,
        centre.y,
        centre.z,
        translation.norm(),
        fit.pose.tilt().to_degrees()
    );
    if let Some(pose) = fit.pose.to_vernier_pose() {
        println!("  vernier pose {pose}");
    }
    Ok(())
}
