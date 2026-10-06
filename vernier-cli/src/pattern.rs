//! Pattern files: the board a command works with, described in JSON instead of
//! on the command line. `make-pattern` writes them; every command taking
//! `--square` also takes `--pattern` to read one.
//!
//! ```json
//! { "pattern": "checkerboard", "square": 5.0, "code_size": 6, "layout": "diamonds" }
//! { "pattern": "megarena", "pitch": 0.009, "code_size": 12 }
//! ```

use std::path::Path;

use serde::{Deserialize, Serialize};
use vernier_camera::Target;
use vernier_patterns::checkerboard::{CodeLayout, CodePacking};

/// Code size boards are rendered with when none is given.
pub const DEFAULT_CODE_SIZE: u32 = 8;

/// A printed board and its parameters, as a pattern file holds them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "pattern", rename_all = "kebab-case", deny_unknown_fields)]
pub enum PatternFile {
    /// The coded checkerboard of `render-pattern checkerboard`.
    Checkerboard {
        /// Side of one printed square, in the unit poses come out in.
        square: f64,
        #[serde(default = "default_code_size")]
        code_size: u32,
        #[serde(default)]
        layout: Layout,
        #[serde(default)]
        packing: Packing,
    },
    /// The megarena dot grid of `render-pattern megarena`.
    Megarena {
        /// Distance between neighbouring dots, in the unit poses come out in.
        pitch: f64,
        #[serde(default = "default_code_size")]
        code_size: u32,
    },
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
            PatternFile::Megarena { pitch, code_size } => {
                if !(pitch > 0.0 && pitch.is_finite()) {
                    return Err(format!("dot pitch {pitch} must be positive"));
                }
                let target = Target::megarena(pitch, code_size);
                target.megarena_pattern().ok_or_else(|| {
                    format!("unsupported megarena code size {code_size}; must be 3..=16")
                })?;
                Ok(target)
            }
        }
    }

    /// Reads and checks a pattern file.
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
        let file: Self =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        file.target()
            .map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(file)
    }

    /// Writes the pattern file at `path`.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, text + "\n")
            .map_err(|e| format!("failed to write {}: {e}", path.display()))
    }
}

/// The board named on a command line: read from `--pattern` if given, else
/// built from `--square`, `--code-size` and `--diamonds`, which `--pattern`
/// replaces.
pub fn target(
    pattern: Option<&str>,
    square: Option<f64>,
    code_size: Option<u32>,
    diamonds: bool,
) -> Result<Target, String> {
    match pattern {
        Some(path) => {
            if square.is_some() || code_size.is_some() || diamonds {
                return Err(
                    "--pattern already gives the board; drop --square, --code-size and --diamonds"
                        .to_string(),
                );
            }
            PatternFile::load(Path::new(path))?.target()
        }
        None => {
            let square = square.ok_or("give the board with --square or --pattern")?;
            PatternFile::Checkerboard {
                square,
                code_size: code_size.unwrap_or(DEFAULT_CODE_SIZE),
                layout: if diamonds {
                    Layout::Diamonds
                } else {
                    Layout::Squares
                },
                packing: Packing::OneBit,
            }
            .target()
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
        let target = file.target().unwrap();
        assert_eq!(target, Target::new(5.0, DEFAULT_CODE_SIZE));

        let file: PatternFile =
            serde_json::from_str(r#"{ "pattern": "megarena", "pitch": 0.009, "code_size": 12 }"#)
                .unwrap();
        let target = file.target().unwrap();
        assert_eq!(target.kind, PatternKind::Megarena);
        assert_eq!((target.square, target.order), (0.009, 12));
    }

    #[test]
    fn round_trips() {
        let file = PatternFile::Checkerboard {
            square: 2.5,
            code_size: 6,
            layout: Layout::Diamonds,
            packing: Packing::TwoBits,
        };
        let text = serde_json::to_string(&file).unwrap();
        assert!(text.contains(r#""layout":"diamonds""#), "{text}");
        assert!(text.contains(r#""packing":"two-bits""#), "{text}");
        assert_eq!(serde_json::from_str::<PatternFile>(&text).unwrap(), file);
    }

    #[test]
    fn rejects_bad_files() {
        // A typo is caught rather than silently defaulted.
        assert!(
            serde_json::from_str::<PatternFile>(
                r#"{ "pattern": "checkerboard", "square": 5.0, "codesize": 6 }"#
            )
            .is_err()
        );
        assert!(
            serde_json::from_str::<PatternFile>(r#"{ "pattern": "qr", "square": 5.0 }"#).is_err()
        );
        let file = PatternFile::Megarena {
            pitch: 1.0,
            code_size: 40,
        };
        assert!(file.target().is_err());
    }

    #[test]
    fn pattern_replaces_the_board_flags() {
        assert!(target(Some("pattern.json"), Some(5.0), None, false).is_err());
        assert!(target(None, None, None, false).is_err());
        assert_eq!(
            target(None, Some(5.0), Some(6), true).unwrap(),
            Target::new(5.0, 6).with_layout(CodeLayout::Diamonds)
        );
    }
}
