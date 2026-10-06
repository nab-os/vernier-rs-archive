//! Shared rasterization helpers for pattern generators. The per-pixel closures
//! are independent, so they'd map cleanly onto a GPU kernel later; for now
//! they're plain CPU loops.

use vernier_core::{GrayImage, Real};

/// Fills a new [`GrayImage`] by evaluating `f(x, y)` at each pixel, where `f`
/// takes pixel coordinates and returns an intensity.
pub fn render_with<F>(width: usize, height: usize, f: F) -> GrayImage
where
    F: Fn(Real, Real) -> Real,
{
    let mut image = GrayImage::zeros(width, height);
    let pixels = image.as_mut_slice();
    for row in 0..height {
        for col in 0..width {
            pixels[row * width + col] = f(col as Real, row as Real) as f32;
        }
    }
    image
}

/// The largest useful corner radius: at half a cell the rounded corners of the
/// four quadrants meet and an isolated cell is a full circle.
pub const MAX_CORNER_RADIUS: Real = 0.5;

/// Whether the point `(x, y)` is filled once the cells' corners are rounded.
///
/// Coordinates are in cell units, so cell `(i, j)` spans `i..i+1` × `j..j+1`,
/// and `radius` is a fraction of a cell side, from `0.0` (square corners) to
/// [`MAX_CORNER_RADIUS`]. `is_set` gives the unrounded lattice.
///
/// Only the `radius × radius` square at each corner of a cell can change:
///
/// - a filled cell loses its corner when both edge neighbours sharing it are
///   empty (a convex corner), and
/// - an empty cell gains its corner when both edge neighbours *and* the diagonal
///   one are filled (a concave corner).
///
/// So cells sharing an edge merge into one smooth shape, while cells touching
/// only at a corner stay apart. Rounding is symmetric about each cell centre, so
/// it shrinks the carrier amplitude without shifting its phase.
pub fn rounded_cell<F>(x: Real, y: Real, radius: Real, is_set: F) -> bool
where
    F: Fn(i64, i64) -> bool,
{
    let (cell_x, cell_y) = (x.floor(), y.floor());
    let (fraction_x, fraction_y) = (x - cell_x, y - cell_y);
    let (i, j) = (cell_x as i64, cell_y as i64);
    let set = is_set(i, j);

    if radius <= 0.0 {
        return set;
    }

    // The nearest corner along each axis, and how far the point sits from it.
    let (side_x, distance_x) = nearest_corner(fraction_x);
    let (side_y, distance_y) = nearest_corner(fraction_y);
    if distance_x >= radius || distance_y >= radius {
        return set;
    }

    let horizontal = is_set(i + side_x, j);
    let vertical = is_set(i, j + side_y);
    let diagonal = is_set(i + side_x, j + side_y);

    // The rounding arc is centred inside the cell, `radius` from both edges.
    let (dx, dy) = (radius - distance_x, radius - distance_y);
    let inside_arc = dx * dx + dy * dy <= radius * radius;

    if set && !horizontal && !vertical {
        inside_arc
    } else if !set && horizontal && vertical && diagonal {
        !inside_arc
    } else {
        set
    }
}

/// Locates a position inside a cell relative to the nearest corner on that axis:
/// which side the corner is on (`-1` before, `+1` after) and the distance to it.
#[inline]
fn nearest_corner(fraction: Real) -> (i64, Real) {
    if fraction < 0.5 {
        (-1, fraction)
    } else {
        (1, 1.0 - fraction)
    }
}

/// Rotates pixel coordinates `(x, y)` by `-theta` about the image center, into
/// the pattern's own axis-aligned frame. Evaluating a pattern in this frame is
/// how a single 1D definition produces an arbitrarily-oriented pattern.
#[inline]
pub fn into_pattern_frame(
    x: Real,
    y: Real,
    center_x: Real,
    center_y: Real,
    theta: Real,
) -> (Real, Real) {
    let (delta_x, delta_y) = (x - center_x, y - center_y);
    let (sin_theta, cos_theta) = (-theta).sin_cos();
    (
        cos_theta * delta_x - sin_theta * delta_y,
        sin_theta * delta_x + cos_theta * delta_y,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use vernier_core::scalar::consts::SQRT_2;

    /// A single filled cell at the origin: all four of its corners are convex.
    fn lone_cell(i: i64, j: i64) -> bool {
        (i, j) == (0, 0)
    }

    #[test]
    fn zero_radius_leaves_the_lattice_alone() {
        for &(x, y) in &[
            (0.01, 0.01),
            (0.5, 0.5),
            (0.99, 0.99),
            (-0.01, 0.5),
            (1.5, 0.2),
        ] {
            assert_eq!(
                rounded_cell(x, y, 0.0, lone_cell),
                lone_cell(x.floor() as i64, y.floor() as i64),
                "at ({x}, {y})",
            );
        }
    }

    #[test]
    fn convex_corners_are_carved_away() {
        let radius = 0.4;
        // The very corner of the cell lies outside the arc, so it is carved off.
        assert!(!rounded_cell(0.001, 0.001, radius, lone_cell));
        assert!(!rounded_cell(0.999, 0.999, radius, lone_cell));
        // The centre, and the middle of each edge, are untouched.
        assert!(rounded_cell(0.5, 0.5, radius, lone_cell));
        assert!(rounded_cell(0.5, 0.01, radius, lone_cell));
        assert!(rounded_cell(0.01, 0.5, radius, lone_cell));
        // Just inside the arc, on the corner's diagonal: the arc centre sits at
        // `(radius, radius)`, so the boundary is `radius − radius/√2` out.
        let inside = radius - radius / SQRT_2 + 0.01;
        assert!(rounded_cell(inside, inside, radius, lone_cell));
    }

    #[test]
    fn edge_adjacent_cells_merge() {
        // Two cells sharing the edge x = 1.
        let pair = |i: i64, j: i64| (i, j) == (0, 0) || (i, j) == (1, 0);
        let radius = 0.4;
        // The corners on the shared edge are not convex, so the seam stays solid.
        assert!(rounded_cell(0.999, 0.001, radius, pair));
        assert!(rounded_cell(1.001, 0.001, radius, pair));
        // The outer corners are still carved.
        assert!(!rounded_cell(0.001, 0.001, radius, pair));
        assert!(!rounded_cell(1.999, 0.999, radius, pair));
    }

    #[test]
    fn diagonally_touching_cells_stay_apart() {
        // Two cells meeting only at the vertex (1, 1).
        let diagonal = |i: i64, j: i64| (i, j) == (0, 0) || (i, j) == (1, 1);
        let radius = 0.4;
        // Neither cell's corner survives, so they do not join at the vertex.
        assert!(!rounded_cell(0.999, 0.999, radius, diagonal));
        assert!(!rounded_cell(1.001, 1.001, radius, diagonal));
        // The empty cells at (1, 0) and (0, 1) are not filled in either: a
        // concave corner needs the diagonal neighbour too, and it is empty.
        assert!(!rounded_cell(1.001, 0.999, radius, diagonal));
        assert!(!rounded_cell(0.999, 1.001, radius, diagonal));
    }

    #[test]
    fn concave_corners_are_filled_in() {
        // An L of three cells leaves (1, 1) with two filled edge neighbours and
        // a filled diagonal, so its inner corner is filled.
        let corner = |i: i64, j: i64| matches!((i, j), (0, 0) | (1, 0) | (0, 1));
        let radius = 0.4;
        // The corner of the empty cell nearest the filled diagonal.
        assert!(rounded_cell(1.001, 1.001, radius, corner));
        // Far from that corner it stays empty.
        assert!(!rounded_cell(1.9, 1.9, radius, corner));
    }

    #[test]
    fn full_radius_makes_a_lone_cell_round() {
        let radius = MAX_CORNER_RADIUS;
        // At half a cell the shape is the inscribed circle: the edge midpoints
        // are on it, the corners are well outside.
        assert!(rounded_cell(0.5, 0.5, radius, lone_cell));
        assert!(!rounded_cell(0.05, 0.05, radius, lone_cell));
        assert!(!rounded_cell(0.95, 0.05, radius, lone_cell));
        // A point just inside the circle near the edge midpoint survives.
        assert!(rounded_cell(0.5, 0.02, radius, lone_cell));
    }
}
