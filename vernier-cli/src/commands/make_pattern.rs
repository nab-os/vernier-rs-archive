//! `make-pattern`: writes the pattern file that `--pattern` reads.

use std::path::Path;

use crate::pattern::PatternFile;

/// Checks the pattern describes a board that can be rendered, and writes it.
pub fn run(file: &PatternFile, output: &Path) -> Result<(), String> {
    file.target()?;
    file.save(output)?;
    println!("wrote {}", output.display());
    Ok(())
}
