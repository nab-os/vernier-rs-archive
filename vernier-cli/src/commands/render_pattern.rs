//! `render-pattern`: draws the board of a pattern file at a given pose and
//! saves it as PNG, at the image size the file gives.

use std::path::PathBuf;

use vernier_camera::Target;
use vernier_patterns::PatternPose;

use crate::imageio;
use crate::pattern::PatternFile;

pub struct RenderPatternArgs {
    pub pattern: PatternFile,
    pub x: f64,
    pub y: f64,
    pub theta: f64,
    pub output: PathBuf,
}

pub fn run(args: &RenderPatternArgs) -> Result<(), String> {
    let image = args.pattern.image();
    // The file's board, measured in rendered pixels.
    let in_pixels = Target {
        square: image.square,
        ..args.pattern.target()?
    };
    let pose = PatternPose::new(args.x, args.y, args.theta);
    let (width, height) = (image.width, image.height);
    let pixels = match args.pattern {
        PatternFile::Checkerboard {
            corner_radius,
            plain,
            ..
        } => {
            let pattern = in_pixels
                .checkerboard()
                .ok_or("unsupported checkerboard")?
                .with_corner_radius(corner_radius);
            if plain {
                pattern.render_plain(width, height, &pose)
            } else {
                pattern.render(width, height, &pose)
            }
        }
        PatternFile::Megarena { .. } => in_pixels
            .megarena_pattern()
            .ok_or("unsupported megarena")?
            .render(width, height, &pose),
    };

    imageio::save_grayscale_png(&args.output, width, height, pixels.as_slice())
}
