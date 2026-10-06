//! Pattern layout parity layer mirroring the C++ `vernier::PatternLayout`
//! abstract class and `vernier::Layout` factory.
//!
//! Where [`PatternDetector`](crate::PatternDetector) *measures* a pattern, a
//! [`PatternLayout`] *defines and generates* one: it exposes the continuous
//! intensity / phase fields, rasterizes the pattern at a pose, and emits the
//! vector geometry used for fabrication (SVG / CSV).
//!
//! ```
//! use vernier_detector::Layout;
//! use vernier_patterns::PatternPose;
//!
//! let mut layout = Layout::new_instance("PeriodicPattern").unwrap();
//! layout.set_double("period", 12.0);
//! layout.set_int("nCols", 5);
//! layout.set_int("nRows", 5);
//! let svg = layout.to_svg();
//! assert!(svg.starts_with("<?xml"));
//! ```
//!
//! SVG and CSV export are both driven by
//! [`to_rectangles`](PatternLayout::to_rectangles). GDS export is supported;
//! OASIS and `saveToLayoutEditorMacro` from the C++ side are not.

use std::fmt::Write as _;

use vernier_core::{GrayImage, Real, Result, VernierError};
use vernier_patterns::PatternPose;
use vernier_patterns::megarena::Megarena;
use vernier_patterns::periodic::Periodic;

use crate::Metadata;

/// An axis-aligned rectangle in pattern-frame units — one drawn dot in the
/// layout (C++ `vernier::Rectangle`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rectangle {
    /// Left edge.
    pub x: f64,
    /// Top edge.
    pub y: f64,
    /// Width.
    pub width: f64,
    /// Height.
    pub height: f64,
}

impl Rectangle {
    /// Creates a rectangle.
    pub fn new(x: f64, y: f64, width: f64, height: f64) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }
}

/// Page margins around the pattern, mirroring the C++ `PatternLayout` fields.
#[derive(Clone, Copy, Debug, Default)]
pub struct Margins {
    /// Left margin.
    pub left: f64,
    /// Right margin.
    pub right: f64,
    /// Top margin.
    pub top: f64,
    /// Bottom margin.
    pub bottom: f64,
}

/// Object interface for pattern layouts, mirroring the C++
/// `vernier::PatternLayout` abstract class.
pub trait PatternLayout {
    /// Class name (`"PeriodicPattern"`, `"MegarenaPattern"`, …).
    fn classname(&self) -> &str;

    /// Intensity of the pattern at continuous pattern-frame coordinates.
    fn get_intensity(&self, x: f64, y: f64) -> f64;

    /// Carrier phase along the first (X) pattern axis, in radians.
    fn get_phase1(&self, x: f64, y: f64) -> f64;

    /// Carrier phase along the second (Y) pattern axis, in radians.
    fn get_phase2(&self, x: f64, y: f64) -> f64;

    /// Rasterizes the pattern at `pose` into a `width × height` image.
    fn render(&self, width: usize, height: usize, pose: &PatternPose) -> GrayImage;

    /// The drawn dots of the layout as a list of rectangles (C++
    /// `toRectangleVector`). SVG / CSV export is built on this.
    fn to_rectangles(&self) -> Vec<Rectangle>;

    /// Overall pattern width in pattern-frame units (excluding margins).
    fn layout_width(&self) -> f64;

    /// Overall pattern height in pattern-frame units (excluding margins).
    fn layout_height(&self) -> f64;

    /// Page margins around the pattern.
    fn margins(&self) -> Margins {
        Margins::default()
    }

    /// Free-text metadata.
    fn description(&self) -> &str;
    /// Sets the `description` metadata.
    fn set_description(&mut self, value: String);
    /// Author metadata.
    fn author(&self) -> &str;
    /// Sets the `author` metadata.
    fn set_author(&mut self, value: String);
    /// Date metadata.
    fn date(&self) -> &str;
    /// Sets the `date` metadata.
    fn set_date(&mut self, value: String);
    /// Unit metadata.
    fn unit(&self) -> &str;
    /// Sets the `unit` metadata.
    fn set_unit(&mut self, value: String);

    /// Reads a floating-point attribute by name, or `None` if unknown.
    fn get_double(&self, attribute: &str) -> Option<f64>;
    /// Sets a floating-point attribute; returns `false` if the name is unknown.
    fn set_double(&mut self, attribute: &str, value: f64) -> bool;
    /// Reads an integer attribute by name, or `None` if unknown.
    fn get_int(&self, attribute: &str) -> Option<i64>;
    /// Sets an integer attribute; returns `false` if the name is unknown.
    fn set_int(&mut self, attribute: &str, value: i64) -> bool;

    /// Human-readable description of the layout (mirrors C++ `toString`).
    fn describe(&self) -> String;

    /// Serializes the layout to an SVG document (C++ `saveToSVG`), as a string.
    fn to_svg(&self) -> String {
        let m = self.margins();
        let w = self.layout_width() + m.left + m.right;
        let h = self.layout_height() + m.top + m.bottom;
        let mut s = String::new();
        let _ = writeln!(s, "<?xml version=\"1.0\" encoding=\"utf-8\"?>");
        let _ = writeln!(s, "<!-- Created with the Vernier library -->");
        let _ = writeln!(
            s,
            "<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" x=\"0\" y=\"0\" width=\"{w}\" height=\"{h}\" viewBox=\"0 0 {w} {h}\">"
        );
        let _ = writeln!(s, "<title>{}</title>", self.description());
        let _ = writeln!(s, "<desc>");
        let _ = writeln!(s, "    class: {}", self.classname());
        let _ = writeln!(s, "    description: {}", self.description());
        let _ = writeln!(s, "    date: {}", self.date());
        let _ = writeln!(s, "    author: {}", self.author());
        let _ = writeln!(s, "    unit: {}", self.unit());
        let _ = writeln!(s, "    width: {}", self.layout_width());
        let _ = writeln!(s, "    height: {}", self.layout_height());
        let _ = writeln!(s, "</desc>");
        for r in self.to_rectangles() {
            let _ = writeln!(
                s,
                "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"black\" />",
                r.x + m.left,
                r.y + m.top,
                r.width,
                r.height
            );
        }
        let _ = writeln!(s, "</svg>");
        s
    }

    /// Serializes the layout to a `;`-separated CSV (C++ `saveToCSV`).
    fn to_csv(&self) -> String {
        let mut s = String::from("x;y;width;height:intensity\n");
        for r in self.to_rectangles() {
            let _ = writeln!(s, "{};{};{};{};1", r.x, r.y, r.width, r.height);
        }
        s
    }

    /// Writes the SVG document to `path`.
    fn save_to_svg(&self, path: &str) -> Result<()> {
        std::fs::write(path, self.to_svg())
            .map_err(|e| VernierError::Message(format!("{path}: {e}")))
    }

    /// Writes the CSV document to `path`.
    fn save_to_csv(&self, path: &str) -> Result<()> {
        std::fs::write(path, self.to_csv())
            .map_err(|e| VernierError::Message(format!("{path}: {e}")))
    }

    /// Serializes the layout to a GDSII stream (C++ `saveToGDS`), as bytes.
    ///
    /// Emits a single cell (named after the class) of the layout rectangles as
    /// `BOUNDARY` elements on layer 0, with 1 µm user units / 1 nm database
    /// units (matching the C++ `gdstk` defaults).
    fn to_gds(&self) -> Vec<u8> {
        crate::gds::write_gds(self.classname(), &self.to_rectangles())
    }

    /// Writes the GDSII stream to `path`.
    fn save_to_gds(&self, path: &str) -> Result<()> {
        std::fs::write(path, self.to_gds())
            .map_err(|e| VernierError::Message(format!("{path}: {e}")))
    }
}

/// Periodic pattern layout — parity with C++ `vernier::PeriodicPatternLayout`.
pub struct PeriodicPatternLayout {
    pattern: Periodic,
    n_rows: i64,
    n_cols: i64,
    dot_size: f64,
    margins: Margins,
    meta: Metadata,
}

impl PeriodicPatternLayout {
    /// Creates a periodic layout with unit period and a 1×1 grid (C++ default).
    pub fn new() -> Self {
        let period = 1.0;
        Self {
            pattern: Periodic::new(period as Real),
            n_rows: 1,
            n_cols: 1,
            dot_size: 0.5 * period,
            margins: Margins::default(),
            meta: Metadata::default(),
        }
    }

    fn period(&self) -> f64 {
        self.pattern.period_px as f64
    }
}

impl Default for PeriodicPatternLayout {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternLayout for PeriodicPatternLayout {
    fn classname(&self) -> &str {
        "PeriodicPattern"
    }
    fn get_intensity(&self, x: f64, y: f64) -> f64 {
        self.pattern.intensity_at(x as Real, y as Real) as f64
    }
    fn get_phase1(&self, x: f64, y: f64) -> f64 {
        self.pattern.phase1_at(x as Real, y as Real) as f64
    }
    fn get_phase2(&self, x: f64, y: f64) -> f64 {
        self.pattern.phase2_at(x as Real, y as Real) as f64
    }
    fn render(&self, width: usize, height: usize, pose: &PatternPose) -> GrayImage {
        self.pattern.render(width, height, pose)
    }

    fn to_rectangles(&self) -> Vec<Rectangle> {
        let period = self.period();
        let offset = (period / 2.0 - self.dot_size) / 2.0;
        let mut out = Vec::new();
        for col in 0..self.n_cols {
            let x = col as f64 * period + offset;
            for row in 0..self.n_rows {
                let y = row as f64 * period + offset;
                if row != 0 || col != 0 {
                    out.push(Rectangle::new(x, y, self.dot_size, self.dot_size));
                }
            }
        }
        out
    }

    fn layout_width(&self) -> f64 {
        self.period() * (self.n_cols as f64 - 0.5)
    }
    fn layout_height(&self) -> f64 {
        self.period() * (self.n_rows as f64 - 0.5)
    }
    fn margins(&self) -> Margins {
        self.margins
    }

    fn description(&self) -> &str {
        &self.meta.description
    }
    fn set_description(&mut self, value: String) {
        self.meta.description = value;
    }
    fn author(&self) -> &str {
        &self.meta.author
    }
    fn set_author(&mut self, value: String) {
        self.meta.author = value;
    }
    fn date(&self) -> &str {
        &self.meta.date
    }
    fn set_date(&mut self, value: String) {
        self.meta.date = value;
    }
    fn unit(&self) -> &str {
        &self.meta.unit
    }
    fn set_unit(&mut self, value: String) {
        self.meta.unit = value;
    }

    fn get_double(&self, attribute: &str) -> Option<f64> {
        match attribute {
            "period" => Some(self.period()),
            "dotSize" => Some(self.dot_size),
            _ => None,
        }
    }
    fn set_double(&mut self, attribute: &str, value: f64) -> bool {
        match attribute {
            "period" => {
                self.pattern.period_px = value as Real;
                // Mirror C++ `resize`: dot size tracks the period.
                self.dot_size = 0.5 * value;
            }
            "dotSize" => self.dot_size = value,
            _ => return false,
        }
        true
    }
    fn get_int(&self, attribute: &str) -> Option<i64> {
        match attribute {
            "nRows" => Some(self.n_rows),
            "nCols" => Some(self.n_cols),
            _ => None,
        }
    }
    fn set_int(&mut self, attribute: &str, value: i64) -> bool {
        match attribute {
            "nRows" => self.n_rows = value.max(0),
            "nCols" => self.n_cols = value.max(0),
            _ => return false,
        }
        true
    }

    fn describe(&self) -> String {
        format!(
            "PeriodicPatternLayout [ period={}; nRows={}; nCols={} ]",
            self.period(),
            self.n_rows,
            self.n_cols
        )
    }
}

/// Megarena pattern layout — parity with C++ `vernier::MegarenaPatternLayout`.
pub struct MegarenaPatternLayout {
    pattern: Megarena,
    /// Extent in periods along each axis. Defaults to a few code windows; set
    /// `nRows`/`nCols` to export a larger region.
    n_rows: i64,
    n_cols: i64,
    dot_size: f64,
    margins: Margins,
    meta: Metadata,
}

impl MegarenaPatternLayout {
    /// Default period (pixels) for a freshly-constructed megarena layout.
    const DEFAULT_PERIOD: Real = 9.0;
    /// Default LFSR order.
    const DEFAULT_ORDER: u32 = 12;

    /// Creates a megarena layout with the reference defaults (9 px period,
    /// 12-bit code) and an extent spanning a few code windows.
    pub fn new() -> Self {
        let pattern = Megarena::new(Self::DEFAULT_PERIOD, Self::DEFAULT_ORDER).unwrap();
        let extent = default_extent(Self::DEFAULT_ORDER);
        Self {
            dot_size: 0.5 * Self::DEFAULT_PERIOD as f64,
            n_rows: extent,
            n_cols: extent,
            margins: Margins::default(),
            meta: Metadata::default(),
            pattern,
        }
    }

    fn period(&self) -> f64 {
        self.pattern.period_px as f64
    }

    fn rebuild(&mut self, period: Real, order: u32) -> bool {
        match Megarena::new(period, order) {
            Some(p) => {
                self.pattern = p;
                true
            }
            None => false,
        }
    }
}

/// Default export extent (in periods): enough to hold a full decode window
/// (`3·order` periods) without exploding the rectangle count for large orders.
fn default_extent(order: u32) -> i64 {
    (3 * order as i64).max(3)
}

impl Default for MegarenaPatternLayout {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternLayout for MegarenaPatternLayout {
    fn classname(&self) -> &str {
        "MegarenaPattern"
    }
    fn get_intensity(&self, x: f64, y: f64) -> f64 {
        self.pattern.intensity_at(x as Real, y as Real) as f64
    }
    fn get_phase1(&self, x: f64, y: f64) -> f64 {
        self.pattern.phase1_at(x as Real, y as Real) as f64
    }
    fn get_phase2(&self, x: f64, y: f64) -> f64 {
        self.pattern.phase2_at(x as Real, y as Real) as f64
    }
    fn render(&self, width: usize, height: usize, pose: &PatternPose) -> GrayImage {
        self.pattern.render(width, height, pose)
    }

    fn to_rectangles(&self) -> Vec<Rectangle> {
        let period = self.period();
        let offset = (period / 2.0 - self.dot_size) / 2.0;
        let mut out = Vec::new();
        for col in 0..self.n_cols {
            let x = col as f64 * period + offset;
            for row in 0..self.n_rows {
                let y = row as f64 * period + offset;
                // Dot present iff both axes' periods are present and this is not
                // the removed π/2-breaking corner (C++ `bitSequence` + corner).
                if self.pattern.period_present(col)
                    && self.pattern.period_present(row)
                    && (col % 3 != 0 || row % 3 != 0)
                {
                    out.push(Rectangle::new(x, y, self.dot_size, self.dot_size));
                }
            }
        }
        out
    }

    fn layout_width(&self) -> f64 {
        self.period() * (self.n_cols as f64 - 0.5)
    }
    fn layout_height(&self) -> f64 {
        self.period() * (self.n_rows as f64 - 0.5)
    }
    fn margins(&self) -> Margins {
        self.margins
    }

    fn description(&self) -> &str {
        &self.meta.description
    }
    fn set_description(&mut self, value: String) {
        self.meta.description = value;
    }
    fn author(&self) -> &str {
        &self.meta.author
    }
    fn set_author(&mut self, value: String) {
        self.meta.author = value;
    }
    fn date(&self) -> &str {
        &self.meta.date
    }
    fn set_date(&mut self, value: String) {
        self.meta.date = value;
    }
    fn unit(&self) -> &str {
        &self.meta.unit
    }
    fn set_unit(&mut self, value: String) {
        self.meta.unit = value;
    }

    fn get_double(&self, attribute: &str) -> Option<f64> {
        match attribute {
            "period" => Some(self.period()),
            "dotSize" => Some(self.dot_size),
            _ => None,
        }
    }
    fn set_double(&mut self, attribute: &str, value: f64) -> bool {
        match attribute {
            "period" => {
                let order = self.pattern.order;
                self.dot_size = 0.5 * value;
                self.rebuild(value as Real, order)
            }
            "dotSize" => {
                self.dot_size = value;
                true
            }
            _ => false,
        }
    }
    fn get_int(&self, attribute: &str) -> Option<i64> {
        match attribute {
            "codeSize" => Some(self.pattern.order as i64),
            "nRows" => Some(self.n_rows),
            "nCols" => Some(self.n_cols),
            _ => None,
        }
    }
    fn set_int(&mut self, attribute: &str, value: i64) -> bool {
        match attribute {
            "codeSize" => {
                let period = self.pattern.period_px;
                self.rebuild(period, value.max(0) as u32)
            }
            "nRows" => {
                self.n_rows = value.max(0);
                true
            }
            "nCols" => {
                self.n_cols = value.max(0);
                true
            }
            _ => false,
        }
    }

    fn describe(&self) -> String {
        format!(
            "MegarenaPatternLayout [ period={}; codeSize={}; nRows={}; nCols={} ]",
            self.period(),
            self.pattern.order,
            self.n_rows,
            self.n_cols
        )
    }
}

/// Class factory for pattern layouts, mirroring the C++ `vernier::Layout`.
pub struct Layout;

impl Layout {
    /// Creates a layout for the given class name.
    ///
    /// Valid names: `"PeriodicPattern"`, `"MegarenaPattern"`.
    pub fn new_instance(classname: &str) -> Result<Box<dyn PatternLayout>> {
        match classname {
            "PeriodicPattern" => Ok(Box::new(PeriodicPatternLayout::new())),
            "MegarenaPattern" => Ok(Box::new(MegarenaPatternLayout::new())),
            other => Err(VernierError::Message(format!(
                "{other} is not a valid class name for a pattern layout."
            ))),
        }
    }

    /// Loads a layout from a JSON document with the C++ layout:
    /// `{ "<ClassName>": { "period": .., "codeSize": .., ... } }`.
    pub fn load_from_json(path: &str) -> Result<Box<dyn PatternLayout>> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| VernierError::Message(format!("{path}: {e}")))?;
        Self::from_json_str(&text)
    }

    /// Parses a layout from an in-memory JSON string (see [`load_from_json`]).
    ///
    /// [`load_from_json`]: Layout::load_from_json
    pub fn from_json_str(text: &str) -> Result<Box<dyn PatternLayout>> {
        let value: serde_json::Value = serde_json::from_str(text)
            .map_err(|e| VernierError::Message(format!("invalid JSON: {e}")))?;
        let obj = value
            .as_object()
            .ok_or_else(|| VernierError::Message("JSON root is not an object.".into()))?;
        let (classname, attrs) = obj
            .iter()
            .next()
            .ok_or_else(|| VernierError::Message("JSON object is empty.".into()))?;

        let mut layout = Self::new_instance(classname)?;
        if let Some(map) = attrs.as_object() {
            for (key, val) in map {
                match val {
                    serde_json::Value::Number(n) => {
                        if let Some(i) = n.as_i64() {
                            if !layout.set_int(key, i) {
                                layout.set_double(key, i as f64);
                            }
                        } else if let Some(f) = n.as_f64() {
                            layout.set_double(key, f);
                        }
                    }
                    serde_json::Value::String(s) => match key.as_str() {
                        "description" => layout.set_description(s.clone()),
                        "date" => layout.set_date(s.clone()),
                        "author" => layout.set_author(s.clone()),
                        "unit" => layout.set_unit(s.clone()),
                        _ => {}
                    },
                    _ => {}
                }
            }
        }
        Ok(layout)
    }
}
