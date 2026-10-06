//! `make-pattern`: writes the pattern file that `--pattern` reads.

use std::path::PathBuf;

use crate::pattern::{Layout, Packing, PatternFile};

/// What `make-pattern` needs, as given on the command line.
pub struct MakePatternArgs {
    pub square: Option<f64>,
    pub pitch: Option<f64>,
    pub code_size: u32,
    pub diamonds: bool,
    pub two_bits: bool,
    pub output: PathBuf,
}

/// Builds the pattern from the command line, checks it describes a board that
/// can be rendered, and writes it.
pub fn run(args: &MakePatternArgs) -> Result<(), String> {
    let file = match (args.square, args.pitch) {
        (Some(square), None) => PatternFile::Checkerboard {
            square,
            code_size: args.code_size,
            layout: if args.diamonds {
                Layout::Diamonds
            } else {
                Layout::Squares
            },
            packing: if args.two_bits {
                Packing::TwoBits
            } else {
                Packing::OneBit
            },
        },
        (None, Some(pitch)) => {
            if args.diamonds || args.two_bits {
                return Err("--diamonds and --two-bits are for a checkerboard".to_string());
            }
            PatternFile::Megarena {
                pitch,
                code_size: args.code_size,
            }
        }
        _ => {
            return Err(
                "give --square for a checkerboard or --pitch for a megarena, not both".to_string(),
            );
        }
    };
    file.target()?;
    file.save(&args.output)?;
    println!("wrote {}", args.output.display());
    Ok(())
}
