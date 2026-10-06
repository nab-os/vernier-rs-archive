//! Pattern files: the board a command works with. `make-pattern` writes them
//! and every other command reads its pattern from one, given with `--pattern`.
//!
//! ```json
//! { "pattern": "checkerboard", "square": 5.0, "code_size": 6, "layout": "diamonds",
//!   "image": { "width": 2480, "height": 3508, "square": 59.0 } }
//! { "pattern": "megarena", "pitch": 1.772, "code_size": 6,
//!   "image": { "width": 1080, "height": 2400, "square": 30.0 } }
//! ```
//!
//! `square` and `pitch` are the printed sizes, in the unit poses come out in;
//! `image` is how the pattern is rendered, in pixels.

use std::path::Path;

use serde::{Deserialize, Serialize};
use vernier_camera::Target;
use vernier_patterns::checkerboard::{CodeLayout, CodePacking};

/// Code size boards are rendered with when none is given.
pub const DEFAULT_CODE_SIZE: u32 = 8;

/// Megarena dot pitch when none is given: 2 mm, a megarena printed on paper.
pub const DEFAULT_PITCH: f64 = 2.0;

/// Rendered image side when none is given, in pixels.
pub const DEFAULT_IMAGE_SIDE: usize = 512;

/// Rendered checkerboard square side when none is given, in pixels.
pub const DEFAULT_SQUARE_PX: f64 = 12.0;

/// Rendered megarena dot pitch when none is given, in pixels.
pub const DEFAULT_PITCH_PX: f64 = 20.0;

/// A printed board and its parameters, as a pattern file holds them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "pattern", rename_all = "kebab-case", deny_unknown_fields)]
pub enum PatternFile {
    /// The coded checkerboard.
    Checkerboard {
        /// Side of one printed square, in the unit poses come out in.
        square: f64,
        #[serde(default = "default_code_size")]
        code_size: u32,
        #[serde(default)]
        layout: Layout,
        #[serde(default)]
        packing: Packing,
        /// Corner radius as a fraction of a square side, from 0.0 for square
        /// corners to 0.5 for as round as a square gets.
        #[serde(default)]
        corner_radius: f64,
        /// The uncoded checkerboard: the carrier alone, which gives no
        /// absolute position.
        #[serde(default)]
        plain: bool,
        #[serde(default = "Image::checkerboard")]
        image: Image,
    },
    /// The megarena dot grid.
    Megarena {
        /// Distance between neighbouring dots, in the unit poses come out in.
        #[serde(default = "default_pitch")]
        pitch: f64,
        #[serde(default = "default_code_size")]
        code_size: u32,
        #[serde(default = "Image::megarena")]
        image: Image,
    },
}

/// How a pattern is rendered: the image size and the square side (dot pitch
/// for a megarena), in pixels.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Image {
    #[serde(default = "default_image_side")]
    pub width: usize,
    #[serde(default = "default_image_side")]
    pub height: usize,
    /// Side of one square, or the dot pitch for a megarena, in pixels.
    pub square: f64,
}

impl Image {
    /// The image a checkerboard is rendered to when none is given.
    pub fn checkerboard() -> Self {
        Self {
            width: DEFAULT_IMAGE_SIDE,
            height: DEFAULT_IMAGE_SIDE,
            square: DEFAULT_SQUARE_PX,
        }
    }

    /// The image a megarena is rendered to when none is given.
    pub fn megarena() -> Self {
        Self {
            square: DEFAULT_PITCH_PX,
            ..Self::checkerboard()
        }
    }
}

/// [`CodeLayout`], named for JSON.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Layout {
    #[default]
    Squares,
    Diamonds,
}

/// [`CodePacking`], named for JSON.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Packing {
    #[default]
    OneBit,
    TwoBits,
}

fn default_code_size() -> u32 {
    DEFAULT_CODE_SIZE
}

fn default_pitch() -> f64 {
    DEFAULT_PITCH
}

fn default_image_side() -> usize {
    DEFAULT_IMAGE_SIDE
}

impl PatternFile {
    /// The target this file describes. Fails on a size no board can be
    /// rendered with.
    pub fn target(&self) -> Result<Target, String> {
        match *self {
            PatternFile::Checkerboard {
                square,
                code_size,
                layout,
                packing,
                ..
            } => {
                if !(square > 0.0 && square.is_finite()) {
                    return Err(format!("square size {square} must be positive"));
                }
                let layout = match layout {
                    Layout::Squares => CodeLayout::Squares,
                    Layout::Diamonds => CodeLayout::Diamonds,
                };
                let packing = match packing {
                    Packing::OneBit => CodePacking::OneBit,
                    Packing::TwoBits => CodePacking::TwoBits,
                };
                let target = Target::new(square, code_size)
                    .with_layout(layout)
                    .with_packing(packing);
                // Building the checkerboard is what checks the code size.
                target.checkerboard().ok_or_else(|| {
                    format!("unsupported checkerboard code size {code_size}; must be 4..=12")
                })?;
                Ok(target)
            }
            PatternFile::Megarena {
                pitch, code_size, ..
            } => {
                if !(pitch > 0.0 && pitch.is_finite()) {
                    return Err(format!("dot pitch {pitch} must be positive"));
                }
                let target = Target::megarena(pitch, code_size);
                target.megarena_pattern().ok_or_else(|| {
                    format!("unsupported megarena code size {code_size}; must be 4..=12")
                })?;
                Ok(target)
            }
        }
    }

    /// How the pattern is rendered.
    pub fn image(&self) -> Image {
        match *self {
            PatternFile::Checkerboard { image, .. } | PatternFile::Megarena { image, .. } => image,
        }
    }

    /// LFSR order of the code.
    pub fn code_size(&self) -> u32 {
        match *self {
            PatternFile::Checkerboard { code_size, .. }
            | PatternFile::Megarena { code_size, .. } => code_size,
        }
    }

    /// The file name `make-pattern` writes to when given none: the pattern,
    /// then its parameters, those left at their default omitted, e.g.
    /// `megarena-pitch1.772-code6-1080x2400-30px.json`.
    pub fn default_name(&self) -> String {
        let mut parts = Vec::new();
        match *self {
            PatternFile::Checkerboard {
                square,
                code_size,
                layout,
                packing,
                corner_radius,
                plain,
                ..
            } => {
                parts.push("checkerboard".to_string());
                parts.push(format!("square{square}"));
                parts.push(format!("code{code_size}"));
                if layout == Layout::Diamonds {
                    parts.push("diamonds".to_string());
                }
                if packing == Packing::TwoBits {
                    parts.push("two-bits".to_string());
                }
                if corner_radius != 0.0 {
                    parts.push(format!("corner{corner_radius}"));
                }
                if plain {
                    parts.push("plain".to_string());
                }
            }
            PatternFile::Megarena {
                pitch, code_size, ..
            } => {
                parts.push("megarena".to_string());
                parts.push(format!("pitch{pitch}"));
                parts.push(format!("code{code_size}"));
            }
        }
        let image = self.image();
        parts.push(format!("{}x{}", image.width, image.height));
        parts.push(format!("{}px", image.square));
        parts.join("-") + ".json"
    }

    /// The target, then the rest of the file, checked.
    fn check(&self) -> Result<(), String> {
        self.target()?;
        let image = self.image();
        if image.width == 0 || image.height == 0 {
            return Err(format!(
                "image size {}x{} must not be empty",
                image.width, image.height
            ));
        }
        if !(image.square > 0.0 && image.square.is_finite()) {
            return Err(format!(
                "image square size {} must be positive",
                image.square
            ));
        }
        if let PatternFile::Checkerboard { corner_radius, .. } = *self {
            // The renderer clamps, but a value written in the file is more
            // likely a mistake than a request to clamp.
            if !(0.0..=0.5).contains(&corner_radius) {
                return Err(format!(
                    "corner radius {corner_radius} is out of range; must be 0.0..=0.5"
                ));
            }
        }
        Ok(())
    }

    /// Reads and checks a pattern file.
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
        let file: Self =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        file.check()
            .map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(file)
    }

    /// Checks the pattern and writes it at `path`.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        self.check()?;
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, text + "\n")
            .map_err(|e| format!("failed to write {}: {e}", path.display()))
    }
}

/// The board of the pattern file at `path`.
pub fn target(path: &str) -> Result<Target, String> {
    PatternFile::load(Path::new(path))?.target()
}

/// The pitch, code size and image of the megarena pattern file at `path`, for
/// the commands that only work with a megarena.
pub fn megarena(path: &str) -> Result<(f64, u32, Image), String> {
    match PatternFile::load(Path::new(path))? {
        PatternFile::Megarena {
            pitch,
            code_size,
            image,
        } => Ok((pitch, code_size, image)),
        PatternFile::Checkerboard { .. } => {
            Err(format!("{path}: this command needs a megarena pattern"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vernier_camera::PatternKind;

    #[test]
    fn reads_both_patterns_with_defaults() {
        let file: PatternFile =
            serde_json::from_str(r#"{ "pattern": "checkerboard", "square": 5.0 }"#).unwrap();
        assert_eq!(file.target().unwrap(), Target::new(5.0, DEFAULT_CODE_SIZE));
        assert_eq!(file.image(), Image::checkerboard());

        let file: PatternFile = serde_json::from_str(
            r#"{ "pattern": "megarena", "pitch": 0.009, "code_size": 12,
                 "image": { "width": 1080, "height": 2400, "square": 30 } }"#,
        )
        .unwrap();
        let target = file.target().unwrap();
        assert_eq!(target.kind, PatternKind::Megarena);
        assert_eq!((target.square, target.order), (0.009, 12));
        assert_eq!(
            file.image(),
            Image {
                width: 1080,
                height: 2400,
                square: 30.0
            }
        );

        let file: PatternFile = serde_json::from_str(r#"{ "pattern": "megarena" }"#).unwrap();
        let target = file.target().unwrap();
        assert_eq!(target, Target::megarena(DEFAULT_PITCH, DEFAULT_CODE_SIZE));
        assert_eq!(file.image(), Image::megarena());
    }

    #[test]
    fn round_trips() {
        let file = PatternFile::Checkerboard {
            square: 2.5,
            code_size: 6,
            layout: Layout::Diamonds,
            packing: Packing::TwoBits,
            corner_radius: 0.25,
            plain: false,
            image: Image {
                width: 640,
                height: 480,
                square: 9.5,
            },
        };
        let text = serde_json::to_string(&file).unwrap();
        assert!(text.contains(r#""layout":"diamonds""#), "{text}");
        assert!(text.contains(r#""packing":"two-bits""#), "{text}");
        assert_eq!(serde_json::from_str::<PatternFile>(&text).unwrap(), file);
    }

    #[test]
    fn names_files_after_their_parameters() {
        let file = PatternFile::Megarena {
            pitch: 1.772,
            code_size: 6,
            image: Image {
                width: 1080,
                height: 2400,
                square: 30.0,
            },
        };
        assert_eq!(
            file.default_name(),
            "megarena-pitch1.772-code6-1080x2400-30px.json"
        );
        let file: PatternFile = serde_json::from_str(
            r#"{ "pattern": "checkerboard", "square": 5.0, "code_size": 6, "layout": "diamonds" }"#,
        )
        .unwrap();
        assert_eq!(
            file.default_name(),
            "checkerboard-square5-code6-diamonds-512x512-12px.json"
        );
    }

    #[test]
    fn rejects_bad_files() {
        // A typo is caught rather than silently defaulted, in the image too.
        assert!(
            serde_json::from_str::<PatternFile>(
                r#"{ "pattern": "checkerboard", "square": 5.0, "codesize": 6 }"#
            )
            .is_err()
        );
        assert!(
            serde_json::from_str::<PatternFile>(
                r#"{ "pattern": "megarena", "image": { "square": 20, "heigth": 9 } }"#
            )
            .is_err()
        );
        assert!(
            serde_json::from_str::<PatternFile>(r#"{ "pattern": "qr", "square": 5.0 }"#).is_err()
        );
        let file = PatternFile::Megarena {
            pitch: 1.0,
            code_size: 40,
            image: Image::megarena(),
        };
        assert!(file.check().is_err());
        let file = PatternFile::Megarena {
            pitch: 1.0,
            code_size: 8,
            image: Image {
                square: 0.0,
                ..Image::megarena()
            },
        };
        assert!(file.check().is_err());
        let file: PatternFile = serde_json::from_str(
            r#"{ "pattern": "checkerboard", "square": 5.0, "corner_radius": 0.7 }"#,
        )
        .unwrap();
        assert!(file.check().is_err());
    }
}
