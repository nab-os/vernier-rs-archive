use std::path::Path;

use vernier_patterns::{
    PatternPose,
    checkerboard::{Checkerboard, CodeLayout},
};

use crate::imageio;

pub struct RenderCheckerboardArgs {
    pub width: usize,
    pub height: usize,
    pub x: f64,
    pub y: f64,
    pub theta: f64,
    pub square_px: f64,
    pub code_size: u32,
    /// Render the uncoded carrier instead of the coded pattern.
    pub plain: bool,
    /// Diamond layout: code along the diagonals.
    pub diamonds: bool,
    /// Corner radius as a fraction of a square side, 0.0 (square) to 0.5 (round).
    pub corner_radius: f64,
    pub output: std::path::PathBuf,
}

pub fn run(args: &RenderCheckerboardArgs) -> Result<(), String> {
    let layout = if args.diamonds {
        CodeLayout::Diamonds
    } else {
        CodeLayout::Squares
    };
    // The builder clamps, but a value typed on the command line is more likely a
    // mistake than a request to clamp.
    if !(0.0..=0.5).contains(&args.corner_radius) {
        return Err(format!(
            "corner radius {} is out of range; must be 0.0..=0.5",
            args.corner_radius
        ));
    }
    let pattern = Checkerboard::new(args.square_px, args.code_size)
        .ok_or_else(|| format!("unsupported code size {}; must be 4..=12", args.code_size))?
        .with_code_layout(layout)
        .with_corner_radius(args.corner_radius);

    let pose = PatternPose::new(args.x, args.y, args.theta);
    let image = if args.plain {
        pattern.render_plain(args.width, args.height, &pose)
    } else {
        pattern.render(args.width, args.height, &pose)
    };

    imageio::save_grayscale_png(
        Path::new(&args.output),
        args.width,
        args.height,
        image.as_slice(),
    )
}
