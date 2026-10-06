//! A megarena's code read off the local phase maps, and its missing dots
//! painted back.
//!
//! The megarena's carriers run along its dot rows and columns, one period per
//! dot, bright at the dot centres. Each dot owns the cell of one period around
//! its centre; the pattern leaves a dot out when the period on either axis is
//! off (the middle period of a code bit that is 0) or when the dot is the
//! corner of its 3×3 group. So each dot reads as present or absent from the
//! light at its centre against the dark at its cell's edge, and the dots then
//! give, in turn, the quarter-turn and the 3×3 groups from the always-on and
//! always-off dots, and the position along each axis from the code bits.

use std::collections::HashMap;

use vernier_core::Real;
use vernier_core::scalar::consts::TAU;
use vernier_patterns::lfsr::Lfsr;
use vernier_patterns::megarena::Megarena;

/// The four quarter-turns, row-major. Mirrors are ruled out by the carriers'
/// handedness (see [`lattice`]).
const TURNS: [[i64; 4]; 4] = [
    [1, 0, 0, 1],   // identity
    [0, -1, 1, 0],  // +90°
    [-1, 0, 0, -1], // 180°
    [0, 1, -1, 0],  // −90°
];

/// Dot weight above which a pixel counts as on its dot's centre, and below which
/// as between them.
const ON_DOT: Real = 0.4;
const OFF_DOT: Real = 0.05;

/// Pixels needed on and off the dot to read a cell.
const MIN_CELL_PIXELS: usize = 3;

/// Cells either way whose contrasts set a cell's reference.
const REFERENCE_REACH: i64 = 4;

/// A cell is present above this fraction of its neighbourhood's bright
/// contrast.
const PRESENT_FRACTION: Real = 0.5;

/// Cells needed to place the 3×3 groups.
const MIN_CELLS: usize = 36;

/// Agreement needed with the always-on and always-off cells, and the lead
/// over the next placement.
const MIN_AGREEMENT: Real = 0.8;
const MIN_LEAD: Real = 0.1;

/// Bits needed beyond the order to check a position along an axis.
const SPARE_BITS: usize = 3;

/// A position along an axis must beat the next best by this many bits.
const MIN_BIT_LEAD: usize = 3;

/// Where the measured lattice sits on the megarena.
#[derive(Clone, Debug, PartialEq)]
pub struct MegarenaCode {
    /// Index of the quarter-turn.
    pub transform: usize,
    /// `pattern = turn(measured) + delta`, in dots.
    pub delta: (i64, i64),
    /// Code bits read along the pattern's x and y.
    pub bits: (usize, usize),
    /// Of those, how many disagree with the code at the position found.
    pub bit_errors: (usize, usize),
    /// Fraction of the always-on and always-off cells that read as such.
    pub agreement: Real,
}

impl MegarenaCode {
    /// A point of the measured lattice (see [`lattice`]), in the pattern's
    /// lattice: dot centres at integers.
    pub fn to_pattern(&self, (u, v): (Real, Real)) -> (Real, Real) {
        let t = TURNS[self.transform].map(|x| x as Real);
        (
            t[0] * u + t[1] * v + self.delta.0 as Real,
            t[2] * u + t[3] * v + self.delta.1 as Real,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MegarenaError {
    UnsupportedOrder(u32),
    /// Too few cells measured to place the 3×3 groups.
    NotEnoughCells,
    /// No quarter-turn and placement reproduced the always-on and always-off
    /// cells clearly.
    NoOrientation,
    /// Too few code bits seen along an axis to fix the position.
    NotEnoughBits,
    /// Two positions along an axis matched the bits about equally well.
    AmbiguousPosition,
}

impl std::fmt::Display for MegarenaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedOrder(n) => write!(f, "unsupported LFSR order {n}"),
            Self::NotEnoughCells => write!(f, "too few megarena cells measured"),
            Self::NoOrientation => {
                write!(f, "no orientation reproduced the megarena's cell groups")
            }
            Self::NotEnoughBits => {
                write!(f, "too few code bits seen to place the megarena")
            }
            Self::AmbiguousPosition => {
                write!(f, "two positions matched the megarena's code about equally well")
            }
        }
    }
}

impl std::error::Error for MegarenaError {}

/// Position in the measured dot lattice, integers at dot centres. The phases
/// are swapped so that, the carriers' cross product being negative (see
/// `find_carriers`), the lattice turns the same way as the image and the
/// pattern is reached by a quarter-turn, never a mirror.
pub(crate) fn lattice(phase: [Real; 2]) -> (Real, Real) {
    (phase[1] / TAU, phase[0] / TAU)
}

/// The carrier's profile across one axis, 1 on a dot centre, 0 halfway.
fn dot_weight(u: Real) -> Real {
    0.5 + 0.5 * (TAU * u).cos()
}

/// Turns a measured dot into the pattern's dots, up to the shift. Dot centres
/// sit on integers, which a quarter-turn keeps on integers.
fn turn_cell(transform: usize, (a, b): (i64, i64)) -> (i64, i64) {
    let t = TURNS[transform];
    (t[0] * a + t[1] * b, t[2] * a + t[3] * b)
}

/// What a cell must read as, from its place in its 3×3 group: the corner is
/// always off, and so are the cells whose both periods are outer ones always
/// on. `None` for a cell that a code bit decides.
fn fixed_cell(c: i64, r: i64) -> Option<bool> {
    match (c.rem_euclid(3), r.rem_euclid(3)) {
        (0, 0) => Some(false),
        (0 | 2, 0 | 2) => Some(true),
        _ => None,
    }
}

/// Reads which cells are present, from the phase maps (`[φ1, φ2, _]` per
/// pixel, NaN off the board) and the frame.
fn read_cells(maps: &[[Real; 3]], intensity: &[f32]) -> HashMap<(i64, i64), bool> {
    // Per dot: sum and count on its centre, then at its cell's edge.
    let mut pools: HashMap<(i64, i64), [Real; 4]> = HashMap::new();
    for (m, &value) in maps.iter().zip(intensity) {
        if !(m[0].is_finite() && m[1].is_finite()) {
            continue;
        }
        let (u, v) = lattice([m[0], m[1]]);
        let weight = dot_weight(u) * dot_weight(v);
        let pool = pools
            .entry((u.round() as i64, v.round() as i64))
            .or_default();
        if weight > ON_DOT {
            pool[0] += value as Real;
            pool[1] += 1.0;
        } else if weight < OFF_DOT {
            pool[2] += value as Real;
            pool[3] += 1.0;
        }
    }
    let min = MIN_CELL_PIXELS as Real;
    let contrast: HashMap<(i64, i64), Real> = pools
        .into_iter()
        .filter(|(_, p)| p[1] >= min && p[3] >= min)
        .map(|(cell, p)| (cell, p[0] / p[1] - p[2] / p[3]))
        .collect();

    // Against the bright cells around it: lighting and contrast vary over a
    // tilted board. About three cells in five are present, so the upper
    // quartile around a cell is a present one.
    contrast
        .iter()
        .map(|(&(a, b), &c)| {
            let mut near: Vec<Real> = (-REFERENCE_REACH..=REFERENCE_REACH)
                .flat_map(|db| (-REFERENCE_REACH..=REFERENCE_REACH).map(move |da| (da, db)))
                .filter_map(|(da, db)| contrast.get(&(a + da, b + db)).copied())
                .collect();
            near.sort_by(Real::total_cmp);
            let reference = near[(near.len() * 3) / 4];
            ((a, b), reference > 0.0 && c > PRESENT_FRACTION * reference)
        })
        .collect()
}

/// The quarter-turn and the shift modulo 3 that best reproduce the fixed
/// cells, with the fraction of them that agree.
fn orientation(cells: &HashMap<(i64, i64), bool>) -> Result<(usize, (i64, i64), Real), MegarenaError> {
    let mut scores = Vec::with_capacity(36);
    for transform in 0..TURNS.len() {
        for ra in 0..3 {
            for rb in 0..3 {
                let (mut agree, mut total) = (0usize, 0usize);
                for (&cell, &present) in cells {
                    let (c, r) = turn_cell(transform, cell);
                    if let Some(expected) = fixed_cell(c + ra, r + rb) {
                        total += 1;
                        agree += (expected == present) as usize;
                    }
                }
                let fraction = agree as Real / total.max(1) as Real;
                scores.push((fraction, transform, (ra, rb)));
            }
        }
    }
    scores.sort_by(|a, b| b.0.total_cmp(&a.0));
    let (best, transform, shift) = scores[0];
    if best < MIN_AGREEMENT || best - scores[1].0 < MIN_LEAD {
        return Err(MegarenaError::NoOrientation);
    }
    Ok((transform, shift, best))
}

/// Where the bits seen along one axis (`triple → bit`) sit in the code: the
/// shift in triples, and how many bits disagree there.
fn locate(code: &Lfsr, order: u32, bits: &HashMap<i64, bool>) -> Result<(i64, usize), MegarenaError> {
    if bits.len() < order as usize + SPARE_BITS {
        return Err(MegarenaError::NotEnoughBits);
    }
    let n = code.len() as i64;
    let mut errors: Vec<(usize, i64)> = (0..n)
        .map(|shift| {
            let wrong = bits
                .iter()
                .filter(|&(&t, &bit)| (code.bit_at((t + shift).rem_euclid(n) as usize) == 1) != bit)
                .count();
            (wrong, shift)
        })
        .collect();
    errors.sort_unstable();
    let (best, shift) = errors[0];
    if errors[1].0 < best + MIN_BIT_LEAD {
        return Err(MegarenaError::AmbiguousPosition);
    }
    Ok((shift, best))
}

/// Reads the code: which cells are present, then the quarter-turn and cell
/// groups, then the position along each axis.
pub(crate) fn read_code(
    maps: &[[Real; 3]],
    intensity: &[f32],
    order: u32,
) -> Result<MegarenaCode, MegarenaError> {
    let pattern = Megarena::new(1.0, order).ok_or(MegarenaError::UnsupportedOrder(order))?;
    let cells = read_cells(maps, intensity);
    if cells.len() < MIN_CELLS {
        return Err(MegarenaError::NotEnoughCells);
    }
    let (transform, (ra, rb), agreement) = orientation(&cells)?;

    // A code column's cells on always-on rows show its bit, and likewise for
    // rows. Votes per triple, `+1` present, `-1` absent.
    let mut votes: [HashMap<i64, i64>; 2] = [HashMap::new(), HashMap::new()];
    for (&cell, &present) in &cells {
        let (c, r) = turn_cell(transform, cell);
        let (c, r) = (c + ra, r + rb);
        let vote = if present { 1 } else { -1 };
        if c.rem_euclid(3) == 1 && r.rem_euclid(3) != 1 {
            *votes[0].entry(c.div_euclid(3)).or_default() += vote;
        }
        if r.rem_euclid(3) == 1 && c.rem_euclid(3) != 1 {
            *votes[1].entry(r.div_euclid(3)).or_default() += vote;
        }
    }
    let bits = votes.map(|v| {
        v.into_iter()
            .filter(|&(_, n)| n != 0)
            .map(|(t, n)| (t, n > 0))
            .collect::<HashMap<i64, bool>>()
    });
    let (shift_x, errors_x) = locate(pattern.code(), order, &bits[0])?;
    let (shift_y, errors_y) = locate(pattern.code(), order, &bits[1])?;
    Ok(MegarenaCode {
        transform,
        delta: (ra + 3 * shift_x, rb + 3 * shift_y),
        bits: (bits[0].len(), bits[1].len()),
        bit_errors: (errors_x, errors_y),
        agreement,
    })
}

/// Tile side for the local contrast fit, in carrier periods.
const RESTORE_TILE: Real = 4.0;

/// The frame with the megarena's missing dots painted back, so that
/// every window sees the plain dot grid and its phase is not pulled by the
/// code. The frame is taken as `A + B · pattern` locally, `A` and `B` fitted
/// per tile on the pixels under the measured lattice.
pub(crate) fn restore(
    intensity: &[f32],
    width: usize,
    height: usize,
    maps: &[[Real; 3]],
    code: &MegarenaCode,
    order: u32,
    period: Real,
) -> Vec<f32> {
    let pattern = Megarena::new(1.0, order).expect("order checked by the decode");
    // Per pixel: the pattern's grey level and the full dot grid's, where measured.
    let levels: Vec<Option<(Real, Real)>> = maps
        .iter()
        .map(|m| {
            if !(m[0].is_finite() && m[1].is_finite()) {
                return None;
            }
            let (x, y) = code.to_pattern(lattice([m[0], m[1]]));
            Some((pattern.intensity_at(x, y), dot_weight(x) * dot_weight(y)))
        })
        .collect();

    let tile = ((RESTORE_TILE * period).round() as usize).max(8);
    let mut restored = intensity.to_vec();
    for ty in (0..height).step_by(tile) {
        for tx in (0..width).step_by(tile) {
            let pixels = || {
                (ty..(ty + tile).min(height))
                    .flat_map(move |y| (tx..(tx + tile).min(width)).map(move |x| y * width + x))
            };
            // Least squares of the frame on the printed pattern.
            let (mut n, mut sp, mut si, mut spp, mut spi) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for i in pixels() {
                if let Some((printed, _)) = levels[i] {
                    let value = intensity[i] as Real;
                    n += 1.0;
                    sp += printed;
                    si += value;
                    spp += printed * printed;
                    spi += printed * value;
                }
            }
            let det = n * spp - sp * sp;
            if n < 16.0 || det.abs() < 1e-9 {
                continue;
            }
            let gain = (n * spi - sp * si) / det;
            for i in pixels() {
                if let Some((printed, full)) = levels[i] {
                    restored[i] += (gain * (full - printed)) as f32;
                }
            }
        }
    }
    restored
}
