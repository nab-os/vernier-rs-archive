//! The parameter model the UI edits, and its translation into `vernier-patterns`
//! calls.
//!
//! Everything the generators accept is represented here as one flat, cloneable
//! [`PatternSettings`] — the widgets edit fields, [`PatternSettings::render`]
//! turns the whole thing into a [`GrayImage`]. Keeping the mapping in one place
//! means a new generator parameter needs a field and one match arm, not a tour
//! of the component tree.

use vernier_core::scalar::consts::SQRT_2;
use vernier_core::{GrayImage, Real};
use vernier_patterns::checkerboard::{Checkerboard, CodeLayout, CodePacking};
use vernier_patterns::megarena::Megarena;
use vernier_patterns::periodic::Periodic;
use vernier_patterns::qrcode::QrLike;
use vernier_patterns::stamp::Stamp;
use vernier_patterns::PatternPose;

/// Brightness of a pattern at a point of its own plane, with no pose applied.
pub type Sampler = Box<dyn Fn(Real, Real) -> Real>;

/// LFSR orders `vernier_patterns::lfsr::Lfsr::maximal` accepts.
pub const ORDER_RANGE: std::ops::RangeInclusive<u32> = 4..=12;

/// Largest image side the UI will render. The generators are per-pixel CPU
/// loops, so this is a responsiveness limit, not a correctness one.
pub const MAX_SIDE: usize = 2048;
/// Smallest image side worth rendering.
pub const MIN_SIDE: usize = 16;

/// Which generator to run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PatternKind {
    /// Sinusoidal carrier — `vernier_patterns::periodic::Periodic`.
    Periodic,
    /// Absolute LFSR-coded dot grid — `vernier_patterns::megarena::Megarena`.
    Megarena,
    /// 50/50 absolute checkerboard — `vernier_patterns::checkerboard::Checkerboard`.
    Checkerboard,
    /// `vernier_patterns::stamp::Stamp` (layout not implemented upstream).
    Stamp,
    /// `vernier_patterns::qrcode::QrLike` (encoding not implemented upstream).
    QrLike,
}

impl PatternKind {
    /// Every kind, in the order the type selector shows them.
    pub const ALL: [PatternKind; 5] = [
        PatternKind::Periodic,
        PatternKind::Megarena,
        PatternKind::Checkerboard,
        PatternKind::Stamp,
        PatternKind::QrLike,
    ];

    /// Short name for the type selector.
    pub fn label(self) -> &'static str {
        match self {
            PatternKind::Periodic => "Periodic",
            PatternKind::Megarena => "Megarena",
            PatternKind::Checkerboard => "Checkerboard",
            PatternKind::Stamp => "Stamp",
            PatternKind::QrLike => "QR-like",
        }
    }

    /// One-line description of what the generator produces.
    pub fn blurb(self) -> &'static str {
        match self {
            PatternKind::Periodic => {
                "Sinusoidal stripe carrier of fixed spatial period. The relative-pose workhorse: \
                 its fundamental peak sits at the pattern orientation in the frequency domain."
            }
            PatternKind::Megarena => {
                "Absolute dot grid. Three periods per bit, the central one gated by a maximal LFSR, \
                 with one corner of each 3×3 cell dropped to break the π/2 rotation ambiguity."
            }
            PatternKind::Checkerboard => {
                "The same LFSR position code on a 50/50 black-and-white carrier. Bits are written by \
                 inverting coding squares against their parity, which shrinks the carrier peak without \
                 rotating it — so the code cannot corrupt the fine pose. Its two carriers run along the \
                 diagonals, at ±45° to the square edges. The packing sets how many bits a supercell holds."
            }
            PatternKind::Stamp => {
                "Stamp tile layout. The interface is fixed upstream but the rasterizer is a stub, \
                 so it renders a blank field."
            }
            PatternKind::QrLike => {
                "Module grid with finder patterns. The interface is fixed upstream but the cell \
                 encoding is a stub, so it renders a blank field."
            }
        }
    }

    /// Whether the explorer can sample this pattern at an arbitrary point. The
    /// stubs cannot: they have no layout and ignore the pose.
    pub fn has_point_sampler(self) -> bool {
        matches!(self, Self::Periodic | Self::Megarena | Self::Checkerboard)
    }

    /// Whether `vernier-patterns` still returns a blank image for this kind.
    /// The UI says so out loud rather than showing an unexplained black square.
    pub fn is_stub(self) -> bool {
        matches!(self, PatternKind::Stamp | PatternKind::QrLike)
    }

    /// Upstream module implementing this kind, shown next to the stub warning.
    pub fn source_path(self) -> &'static str {
        match self {
            PatternKind::Periodic => "vernier-patterns/src/periodic.rs",
            PatternKind::Megarena => "vernier-patterns/src/megarena.rs",
            PatternKind::Checkerboard => "vernier-patterns/src/checkerboard.rs",
            PatternKind::Stamp => "vernier-patterns/src/stamp.rs",
            PatternKind::QrLike => "vernier-patterns/src/qrcode.rs",
        }
    }
}

/// Every knob the generators expose, plus the display-only ones.
#[derive(Clone, PartialEq, Debug)]
pub struct PatternSettings {
    /// Which generator to run.
    pub kind: PatternKind,

    /// Output image width in pixels.
    pub width: usize,
    /// Output image height in pixels.
    pub height: usize,

    /// Pattern X translation in pixels.
    pub pose_x: Real,
    /// Pattern Y translation in pixels.
    pub pose_y: Real,
    /// Pattern orientation. Degrees here, radians at the generator boundary —
    /// a slider in radians is unreadable.
    pub theta_deg: Real,

    /// Spatial period in pixels. Shared by [`PatternKind::Periodic`] and
    /// [`PatternKind::Megarena`], which both build their carrier from it.
    pub period_px: Real,

    /// LFSR order — bits per unique window, so it sets the absolute range.
    pub order: u32,
    /// LFSR index placed at triple 0. Upstream suggests `order` to keep the
    /// decode window off the sequence boundary.
    pub lfsr_offset: i64,

    /// Checkerboard square side in pixels. Separate from [`period_px`] because
    /// it is not a carrier period: the carriers run diagonally, one fringe every
    /// `square_px · √2`.
    ///
    /// [`period_px`]: PatternSettings::period_px
    pub square_px: Real,
    /// Sub-samples per pixel edge when rasterizing the checkerboard. The pattern
    /// is binary with hard edges, so point sampling aliases and shifts the
    /// measured carrier phase.
    pub supersample: u32,
    /// Render the checkerboard with every coding site left at its parity colour
    /// — the uncoded reference, for seeing what the code costs.
    pub plain_checkerboard: bool,
    /// Whether the checkerboard's corners are rounded. Off is upstream's
    /// default, and renders square corners.
    pub rounded_corners: bool,
    /// Corner radius as a fraction of a square side, applied only while
    /// [`rounded_corners`] is on. Held across a toggle off and back on, so
    /// unticking the box does not throw away the radius that was dialled in.
    ///
    /// [`rounded_corners`]: PatternSettings::rounded_corners
    pub corner_radius: Real,
    /// Which way the coded lattice sits. `Squares` writes the code along the
    /// square edges, leaving the carriers on the diagonals; `Diamonds` turns
    /// the lattice 45°, which puts the carriers on the pattern axes instead and
    /// costs √2 of absolute range.
    pub code_layout: CodeLayout,
    /// How many code bits each supercell carries per axis. `OneBit` is the 3×3
    /// supercell upstream has always rendered; `TwoBits` packs two bits into a
    /// 5×5 one, which is denser *and* inverts fewer squares.
    pub code_packing: CodePacking,

    /// Stamp tile side in pixels.
    pub tile_px: usize,

    /// Modules per axis in the QR-like grid.
    pub modules: usize,
    /// Pixels per module.
    pub module_px: usize,

    /// Display only: swap black and white before painting.
    pub invert: bool,
}

impl Default for PatternSettings {
    fn default() -> Self {
        Self {
            kind: PatternKind::Megarena,
            width: 512,
            height: 512,
            pose_x: 0.0,
            pose_y: 0.0,
            theta_deg: 0.0,
            period_px: 20.0,
            order: 8,
            lfsr_offset: 8,
            // Matches `vernier-cli render-pattern checkerboard --square`.
            square_px: 12.0,
            // Upstream's own default; see the supersample field.
            supersample: 4,
            plain_checkerboard: false,
            // Off by default, so the app opens on what upstream renders.
            rounded_corners: false,
            // Half of upstream's maximum: clearly rounded at a glance without
            // collapsing the squares to circles the moment the box is ticked.
            corner_radius: 0.25,
            code_layout: CodeLayout::Squares,
            // Upstream's packing, so the app still opens on what it renders.
            code_packing: CodePacking::OneBit,
            tile_px: 32,
            modules: 21,
            module_px: 8,
            invert: false,
        }
    }
}

impl PatternSettings {
    /// Brightness at an arbitrary point of the pattern plane, with no pose
    /// applied — what the explorer's camera samples through its homography.
    ///
    /// Returns `Err` for the stub generators, which have no layout to sample.
    pub fn sampler(&self) -> Result<Sampler, String> {
        match self.kind {
            // `Periodic::intensity_at` is a one-dimensional stripe carrier,
            // independent of y, so it puts a single lobe pair in the spectrum
            // and a two-direction peak search has nothing to find in the
            // second direction. Building the grid from the pattern's own two
            // phase functions gives the (1+cos)(1+cos)/4 model that this
            // crate's GPU renderer and the C++ reference both use.
            PatternKind::Periodic => {
                let pattern = Periodic::new(self.period_px);
                Ok(Box::new(move |x, y| {
                    let c1 = pattern.phase1_at(x, y).cos();
                    let c2 = pattern.phase2_at(x, y).cos();
                    (1.0 + c1) * (1.0 + c2) / 4.0
                }))
            }
            PatternKind::Megarena => Megarena::new(self.period_px, self.order)
                .map(|pattern| {
                    let pattern = pattern.with_lfsr_offset(self.lfsr_offset);
                    Box::new(move |x, y| pattern.intensity_at(x, y)) as Sampler
                })
                .ok_or_else(|| self.order_error()),
            PatternKind::Checkerboard => Checkerboard::new(self.square_px, self.order)
                .map(|pattern| {
                    let pattern = pattern
                        .with_code_layout(self.code_layout)
                        .with_code_packing(self.code_packing)
                        .with_lfsr_offset(self.lfsr_offset)
                        .with_corner_radius(self.effective_corner_radius());
                    let plain = self.plain_checkerboard;
                    Box::new(move |x, y| {
                        if plain {
                            pattern.plain_intensity_at(x, y)
                        } else {
                            pattern.intensity_at(x, y)
                        }
                    }) as Sampler
                })
                .ok_or_else(|| self.order_error()),
            PatternKind::Stamp | PatternKind::QrLike => Err(format!(
                "{} has no layout upstream ({}), so there is nothing to sample at a pose. \
                 Pick Periodic, Megarena or Checkerboard.",
                self.kind.label(),
                self.kind.source_path()
            )),
        }
    }

    /// The carrier period the explorer scales the camera against: the distance
    /// between fringes, which for a checkerboard is the diagonal `a·√2` and not
    /// the square side.
    pub fn explorer_period_px(&self) -> Real {
        match self.kind {
            PatternKind::Checkerboard => self.carrier_period_px(),
            _ => self.period_px,
        }
    }

    /// The pose the generators take, with the UI's degrees converted to radians.
    pub fn pose(&self) -> PatternPose {
        PatternPose::new(self.pose_x, self.pose_y, self.theta_deg.to_radians())
    }

    /// The corner radius to hand the generator: the dialled-in value while the
    /// toggle is on, and `0.0` — square corners — while it is off. The two
    /// fields exist so the radius survives the toggle; this is what renders.
    pub fn effective_corner_radius(&self) -> Real {
        if self.rounded_corners {
            self.corner_radius
        } else {
            0.0
        }
    }

    /// Distance between successive checkerboard carrier fringes, `a·√2`. The
    /// carriers run along the diagonals, so this — not the square side — is the
    /// "period" a detector is configured with.
    pub fn carrier_period_px(&self) -> Real {
        self.square_px * SQRT_2
    }

    /// Message for an LFSR order upstream won't build a maximal sequence for.
    /// Shared by the two coded patterns, which take the order the same way.
    fn order_error(&self) -> String {
        format!(
            "LFSR order {} is unsupported — vernier-patterns generates maximal \
             sequences of order {}..={} only.",
            self.order,
            ORDER_RANGE.start(),
            ORDER_RANGE.end()
        )
    }

    /// Runs the selected generator. The error case is a parameter combination
    /// upstream rejects — currently only an unsupported LFSR order.
    pub fn render(&self) -> Result<GrayImage, String> {
        let pose = self.pose();
        let (w, h) = (self.width, self.height);

        match self.kind {
            PatternKind::Periodic => Ok(Periodic::new(self.period_px).render(w, h, &pose)),
            PatternKind::Megarena => Megarena::new(self.period_px, self.order)
                .map(|pattern| pattern.with_lfsr_offset(self.lfsr_offset).render(w, h, &pose))
                .ok_or_else(|| self.order_error()),
            PatternKind::Checkerboard => Checkerboard::new(self.square_px, self.order)
                .map(|pattern| {
                    let pattern = pattern
                        .with_code_layout(self.code_layout)
                        .with_code_packing(self.code_packing)
                        .with_lfsr_offset(self.lfsr_offset)
                        .with_supersample(self.supersample)
                        .with_corner_radius(self.effective_corner_radius());
                    if self.plain_checkerboard {
                        pattern.render_plain(w, h, &pose)
                    } else {
                        pattern.render(w, h, &pose)
                    }
                })
                .ok_or_else(|| self.order_error()),
            PatternKind::Stamp => Ok(Stamp::new(self.tile_px).render(w, h, &pose)),
            PatternKind::QrLike => {
                Ok(QrLike::new(self.modules, self.module_px).render(w, h, &pose))
            }
        }
    }

    /// Quantities derived from the current parameters that are worth reading off
    /// directly — what the code length and absolute range actually work out to.
    pub fn derived(&self) -> Vec<(String, String)> {
        match self.kind {
            PatternKind::Periodic => vec![
                ("Periods across width".into(), format!("{:.2}", self.width as Real / self.period_px)),
                ("Periods across height".into(), format!("{:.2}", self.height as Real / self.period_px)),
            ],
            PatternKind::Megarena => {
                let code_len = (1u64 << self.order) - 1;
                // Three periods per bit: the whole sequence spans this many pixels
                // before the absolute code repeats.
                let range_px = code_len as Real * 3.0 * self.period_px;
                vec![
                    ("Code length".into(), format!("{code_len} bits (2^{} − 1)", self.order)),
                    ("Absolute range".into(), format!("{range_px:.0} px")),
                    ("Bits across width".into(), format!("{:.2}", self.width as Real / (3.0 * self.period_px))),
                ]
            }
            PatternKind::Checkerboard => {
                let code_len = (1u64 << self.order) - 1;
                let packing = self.code_packing;
                // One supercell edge per bit under OneBit, half of one under
                // TwoBits — that ratio is the whole point of the packing.
                let range_squares = packing.cell() as u64 * code_len;
                let mut rows = vec![
                    ("Code length".into(), format!("{code_len} bits (2^{} − 1)", self.order)),
                    // The carriers run diagonally, so a fringe is a√2 apart —
                    // this, not the square side, is what a detector is told.
                    ("Carrier period".into(), format!("{:.2} px", self.carrier_period_px())),
                    (
                        "Squares per bit".into(),
                        format!(
                            "{:.1} ({cell}×{cell} supercell, {bits} per axis)",
                            packing.squares_per_bit(),
                            cell = packing.cell(),
                            bits = if packing.bits_per_cell() == 1 { "1 bit" } else { "2 bits" },
                        ),
                    ),
                    // What a decoder actually has to see: the window is `order`
                    // bits wide, so this is the field of view the packing buys.
                    (
                        "Decode window".into(),
                        format!(
                            "{:.0} sq ({:.0} px) for {} bits",
                            self.order as Real * packing.squares_per_bit(),
                            self.order as Real * packing.squares_per_bit() * self.square_px,
                            self.order,
                        ),
                    ),
                    ("Absolute range".into(), format!("{range_squares} sq ({:.0} px)", range_squares as Real * self.square_px)),
                    ("Squares across width".into(), format!("{:.2}", self.width as Real / self.square_px)),
                ];
                // Rounding carves only the white squares, so it costs the 50/50
                // fill the checkerboard is chosen for. Worth reading off rather
                // than discovering in a detector.
                if self.rounded_corners {
                    if let Some(pattern) = Checkerboard::new(self.square_px, self.order) {
                        let white = pattern.with_corner_radius(self.corner_radius).white_fraction();
                        rows.push((
                            "White fraction".into(),
                            format!("{white:.3} (0.500 with square corners)"),
                        ));
                    }
                }
                rows
            }
            PatternKind::Stamp => vec![
                ("Tiles across width".into(), format!("{:.2}", self.width as Real / self.tile_px.max(1) as Real)),
            ],
            PatternKind::QrLike => vec![
                ("Grid extent".into(), format!("{} px", self.modules * self.module_px)),
            ],
        }
    }

    /// The `vernier-patterns` call this UI state corresponds to, so a parameter
    /// set found here can be carried straight into Rust code or a test.
    pub fn equivalent_rust(&self) -> String {
        let pose = format!(
            "PatternPose::new({:.3}, {:.3}, {:.5}_f64.to_radians())",
            self.pose_x, self.pose_y, self.theta_deg
        );
        let constructor = match self.kind {
            PatternKind::Periodic => format!("Periodic::new({:.3})", self.period_px),
            PatternKind::Megarena => format!(
                "Megarena::new({:.3}, {})\n    .unwrap()\n    .with_lfsr_offset({})",
                self.period_px, self.order, self.lfsr_offset
            ),
            PatternKind::Checkerboard => format!(
                "Checkerboard::new({:.3}, {})\n    .unwrap()\n    .with_code_layout(CodeLayout::{:?}){}\n    .with_lfsr_offset({})\n    .with_supersample({}){}",
                self.square_px,
                self.order,
                self.code_layout,
                // One bit per supercell is the default, so the call is only
                // worth showing when it would actually change the render.
                if self.code_packing == CodePacking::OneBit {
                    String::new()
                } else {
                    format!("\n    .with_code_packing(CodePacking::{:?})", self.code_packing)
                },
                self.lfsr_offset,
                self.supersample,
                // Square corners are the default, so the call is only worth
                // showing when it would actually change the render.
                if self.rounded_corners {
                    format!("\n    .with_corner_radius({:.3})", self.corner_radius)
                } else {
                    String::new()
                },
            ),
            PatternKind::Stamp => format!("Stamp::new({})", self.tile_px),
            PatternKind::QrLike => format!("QrLike::new({}, {})", self.modules, self.module_px),
        };
        // The uncoded reference is a different method, not a different builder.
        let method = if self.kind == PatternKind::Checkerboard && self.plain_checkerboard {
            "render_plain"
        } else {
            "render"
        };
        format!(
            "let pose = {pose};\nlet image = {constructor}\n    .{method}({}, {}, &pose);",
            self.width, self.height
        )
    }

    /// Filename for the PNG download, tagged with the parameters that produced it.
    pub fn file_stem(&self) -> String {
        let base = match self.kind {
            PatternKind::Periodic => format!("periodic_p{:.0}", self.period_px),
            PatternKind::Megarena => format!("megarena_p{:.0}_n{}", self.period_px, self.order),
            PatternKind::Checkerboard => format!(
                "checkerboard{}{}{}_a{:.0}_n{}",
                if self.plain_checkerboard { "_plain" } else { "" },
                if self.code_packing == CodePacking::OneBit { "" } else { "_2b" },
                if self.rounded_corners {
                    format!("_r{:.0}", self.corner_radius * 100.0)
                } else {
                    String::new()
                },
                self.square_px,
                self.order
            ),
            PatternKind::Stamp => format!("stamp_t{}", self.tile_px),
            PatternKind::QrLike => format!("qrlike_{}x{}", self.modules, self.module_px),
        };
        format!("{base}_{}x{}", self.width, self.height)
    }
}

/// Expands a `[0, 1]` intensity field into the RGBA bytes a canvas `ImageData`
/// wants. Intensities are clamped rather than rescaled: the generators promise
/// `[0, 1]`, and rescaling would hide a generator that broke that promise.
pub fn to_rgba(image: &GrayImage, invert: bool) -> Vec<u8> {
    let mut rgba = vec![0u8; image.width() * image.height() * 4];
    for (pixel, &intensity) in rgba.chunks_exact_mut(4).zip(image.as_slice()) {
        let level = intensity.clamp(0.0, 1.0);
        let level = if invert { 1.0 - level } else { level };
        let byte = (level * 255.0).round() as u8;
        pixel[0] = byte;
        pixel[1] = byte;
        pixel[2] = byte;
        pixel[3] = 255;
    }
    rgba
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_every_kind() {
        for kind in PatternKind::ALL {
            let settings = PatternSettings { kind, width: 48, height: 32, ..Default::default() };
            let image = settings.render().unwrap();
            assert_eq!((image.width(), image.height()), (48, 32), "{kind:?}");
        }
    }

    #[test]
    fn rejects_unsupported_order() {
        for kind in [PatternKind::Megarena, PatternKind::Checkerboard] {
            let settings = PatternSettings { kind, order: 3, ..Default::default() };
            assert!(settings.render().unwrap_err().contains("4..=12"));
        }
    }

    #[test]
    fn rgba() {
        let image = GrayImage::from_vec(3, 1, vec![0.0, 1.0, 1.5]).unwrap();
        assert_eq!(to_rgba(&image, false), vec![0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255]);
        assert_eq!(&to_rgba(&image, true)[..8], &[255, 255, 255, 255, 0, 0, 0, 255]);
    }

    /// The radius is remembered while the toggle is off, but must not reach the
    /// generator — otherwise unticking the box would not restore square corners.
    #[test]
    fn the_toggle_gates_the_radius_without_losing_it() {
        let off = PatternSettings {
            kind: PatternKind::Checkerboard,
            rounded_corners: false,
            corner_radius: 0.4,
            ..Default::default()
        };
        assert_eq!(off.effective_corner_radius(), 0.0);
        assert_eq!(off.corner_radius, 0.4, "the dialled-in radius is kept");

        let on = PatternSettings { rounded_corners: true, ..off };
        assert_eq!(on.effective_corner_radius(), 0.4);
    }

    #[test]
    fn the_toggle_changes_the_render() {
        let square = PatternSettings {
            kind: PatternKind::Checkerboard,
            width: 96,
            height: 96,
            rounded_corners: false,
            corner_radius: 0.5,
            ..Default::default()
        };
        let rounded = PatternSettings { rounded_corners: true, ..square.clone() };

        let (a, b) = (square.render().unwrap(), rounded.render().unwrap());
        assert_ne!(a.as_slice(), b.as_slice(), "rounding should change the image");

        // Ticking the box only ever removes white: corners are carved away.
        let white = |image: &GrayImage| image.as_slice().iter().map(|&v| v as Real).sum::<Real>();
        assert!(white(&b) < white(&a), "rounding should not add white");
    }

    #[test]
    fn the_snippet_shows_the_radius_only_when_rounded() {
        let mut settings = PatternSettings {
            kind: PatternKind::Checkerboard,
            corner_radius: 0.3,
            rounded_corners: false,
            ..Default::default()
        };
        assert!(!settings.equivalent_rust().contains("with_corner_radius"));

        settings.rounded_corners = true;
        assert!(settings.equivalent_rust().contains(".with_corner_radius(0.300)"));
    }

    #[test]
    fn snippet_matches_settings() {
        let settings = PatternSettings {
            kind: PatternKind::Megarena,
            order: 10,
            period_px: 12.5,
            width: 640,
            height: 480,
            ..Default::default()
        };
        let snippet = settings.equivalent_rust();
        assert!(snippet.contains("Megarena::new(12.500, 10)"), "{snippet}");
        assert!(snippet.contains(".render(640, 480, &pose)"), "{snippet}");
    }
}
