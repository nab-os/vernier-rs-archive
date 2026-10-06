//! `render-pattern`: draws a coded checkerboard or a megarena at a given pose
//! and saves it as PNG.

use std::path::PathBuf;

use vernier_patterns::{
    PatternPose,
    checkerboard::{Checkerboard, CodeLayout},
    megarena::Megarena,
};

use crate::imageio;

pub struct RenderPatternArgs {
    pub width: usize,
    pub height: usize,
    pub x: f64,
    pub y: f64,
    pub theta: f64,
    pub pattern: Pattern,
    pub output: PathBuf,
}

/// The pattern to draw, with its sizes in pixels.
pub enum Pattern {
    Checkerboard {
        square_px: f64,
        code_size: u32,
        /// Render the uncoded carrier instead of the coded pattern.
        plain: bool,
        /// Diamond layout: code along the diagonals.
        diamonds: bool,
        /// Corner radius as a fraction of a square side, 0.0 (square) to 0.5 (round).
        corner_radius: f64,
    },
    Megarena {
        period_px: f64,
        code_size: u32,
    },
}

pub fn run(args: &RenderPatternArgs) -> Result<(), String> {
    let pose = PatternPose::new(args.x, args.y, args.theta);
    let image = match args.pattern {
        Pattern::Checkerboard {
            square_px,
            code_size,
            plain,
            diamonds,
            corner_radius,
        } => {
            let layout = if diamonds {
                CodeLayout::Diamonds
            } else {
                CodeLayout::Squares
            };
            // The builder clamps, but a value typed on the command line is more
            // likely a mistake than a request to clamp.
            if !(0.0..=0.5).contains(&corner_radius) {
                return Err(format!(
                    "corner radius {corner_radius} is out of range; must be 0.0..=0.5"
                ));
            }
            let pattern = Checkerboard::new(square_px, code_size)
                .ok_or_else(|| format!("unsupported code size {code_size}; must be 4..=12"))?
                .with_code_layout(layout)
                .with_corner_radius(corner_radius);
            if plain {
                pattern.render_plain(args.width, args.height, &pose)
            } else {
                pattern.render(args.width, args.height, &pose)
            }
        }
        Pattern::Megarena {
            period_px,
            code_size,
        } => Megarena::new(period_px, code_size)
            .ok_or_else(|| format!("unsupported code size {code_size}; must be 4..=12"))?
            .render(args.width, args.height, &pose),
    };

    imageio::save_grayscale_png(&args.output, args.width, args.height, image.as_slice())
}
