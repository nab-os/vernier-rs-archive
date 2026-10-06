use std::collections::BTreeMap;

use vernier_core::Complex32;
use vernier_core::buffer::BufferLayout;
use vernier_cpu::CpuBackend;
use vernier_patterns::PatternPose;
use vernier_patterns::checkerboard::{Checkerboard, CodeLayout, CodePacking};
use vernier_pose::checkerboard::{
    CheckerboardError, detect_checkerboard, detect_checkerboard_with_packing, extract_code,
    extract_code_with_packing, read_squares, read_squares_with_packing,
    solve_checkerboard_with_packing,
};
use vernier_spectral::spectrum::analyze_two;

const SIZE: usize = 512;
const SQUARE: f64 = 8.0;
const ORDER: u32 = 8;
const LAYOUTS: [CodeLayout; 2] = [CodeLayout::Squares, CodeLayout::Diamonds];

fn random_poses(seed: u64, n: usize) -> Vec<PatternPose> {
    let mut state = seed;
    let mut unit = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 32) as f64 + 0.5) / (u32::MAX as f64 + 1.0)
    };
    (0..n)
        .map(|_| {
            let (x, y) = ((unit() - 0.5) * 4000.0, (unit() - 0.5) * 4000.0);
            PatternPose::new(x, y, (unit() - 0.5) * 1.26)
        })
        .collect()
}

fn solve(pattern: &Checkerboard, pose: &PatternPose) -> vernier_core::Pose {
    let image = pattern.render(SIZE, SIZE, pose);
    let buffer = BufferLayout::packed(SIZE, SIZE);
    let packing = pattern.code_packing();
    let detection = detect_checkerboard_with_packing(
        &CpuBackend::new(),
        image.as_slice(),
        buffer,
        4.0,
        10,
        0,
        0.0,
        packing,
    )
    .unwrap();
    solve_checkerboard_with_packing(
        &detection,
        image.as_slice(),
        SQUARE,
        ORDER,
        pattern.code_layout(),
        packing,
    )
    .unwrap_or_else(|e| panic!("{:?}/{packing:?} at {pose:?}: {e}", pattern.code_layout()))
    .0
}

fn position_error(pattern: &Checkerboard, pose: &PatternPose) -> f64 {
    let found = solve(pattern, pose);
    let (dx, dy) = pattern.wrap_offset(found.x + pose.x, found.y + pose.y);
    dx.abs().max(dy.abs())
}

#[test]
fn decodes_random_poses() {
    for layout in LAYOUTS {
        let pattern = Checkerboard::new(SQUARE, ORDER)
            .unwrap()
            .with_code_layout(layout);
        for pose in random_poses(0x2545_f491_4f6c_dd1d, 24) {
            let error = position_error(&pattern, &pose);
            assert!(error < 0.25, "{layout:?} at {pose:?}: {error:.3} px");
        }
    }
}

#[test]
fn reports_the_orientation() {
    for layout in LAYOUTS {
        let pattern = Checkerboard::new(SQUARE, ORDER)
            .unwrap()
            .with_code_layout(layout);
        for degrees in [-170.0f64, -80.0, 0.0, 45.0, 100.0, 150.0] {
            let theta = degrees.to_radians();
            let found = solve(&pattern, &PatternPose::new(123.4, -567.8, theta));
            let error = (found.theta - theta + std::f64::consts::PI)
                .rem_euclid(std::f64::consts::TAU)
                - std::f64::consts::PI;
            assert!(
                error.to_degrees().abs() < 0.05,
                "{layout:?} at {degrees}: {}",
                found.theta.to_degrees()
            );
        }
    }
}

// These poses make the plain peak search lock onto the code's 1/3 line.
const LOCKING_POSES: [usize; 5] = [1, 27, 38, 52, 97];

#[test]
fn recovers_from_a_subharmonic_lock() {
    let poses = random_poses(0x9e37_79b9_7f4a_7c15, 100);
    for layout in LAYOUTS {
        let pattern = Checkerboard::new(SQUARE, ORDER)
            .unwrap()
            .with_code_layout(layout);
        for index in LOCKING_POSES {
            assert!(
                position_error(&pattern, &poses[index]) < 0.5 * SQUARE,
                "{layout:?} pose {index}"
            );
        }
    }
}

#[test]
fn refuses_a_subharmonic_lock() {
    let poses = random_poses(0x9e37_79b9_7f4a_7c15, 100);
    let pattern = Checkerboard::new(SQUARE, ORDER).unwrap();
    for index in [1, 97] {
        let image = pattern.render(SIZE, SIZE, &poses[index]);
        let complex: Vec<Complex32> = image
            .as_slice()
            .iter()
            .map(|&v| Complex32::new(v, 0.0))
            .collect();
        let buffer = BufferLayout::packed(SIZE, SIZE);
        let detection = analyze_two(&CpuBackend::new(), &complex, buffer, 4.0, 10, 0, 0.0).unwrap();
        assert!(matches!(
            extract_code(&detection, image.as_slice(), ORDER),
            Err(CheckerboardError::SubharmonicLock)
        ));
    }
}

/// The same round trip on the denser packing: two bits per supercell means the
/// decoder has to lift each run's position by its slot parity, not just by the
/// LFSR index, or it lands half a supercell out.
#[test]
fn decodes_random_poses_with_two_bit_packing() {
    for layout in LAYOUTS {
        let pattern = Checkerboard::new(SQUARE, ORDER)
            .unwrap()
            .with_code_layout(layout)
            .with_code_packing(CodePacking::TwoBits);
        for pose in random_poses(0x2545_f491_4f6c_dd1d, 24) {
            let error = position_error(&pattern, &pose);
            assert!(error < 0.25, "{layout:?} at {pose:?}: {error:.3} px");
        }
    }
}

/// The denser packing must reach further for the same order: its supercell is
/// bigger, so one sequence spans more squares.
#[test]
fn two_bit_packing_reaches_further() {
    let one = Checkerboard::new(SQUARE, ORDER).unwrap();
    let two = one.clone().with_code_packing(CodePacking::TwoBits);
    assert!(two.range_squares() > one.range_squares());
}

/// The diagnostic readout has to describe the same extraction the decode runs
/// on, not an approximation of it.
///
/// The code inverts one square per axis in each 3x3 supercell, so a correct
/// readout marks coding sites in exactly two residue classes and nowhere else,
/// and within a class every supercell is all-or-nothing: one bit governs a
/// whole band of sites, so a band either breaks parity throughout or not at
/// all. A rate check would pass on a readout that were merely plausible; this
/// only passes on one that is right.
#[test]
fn reports_the_squares_the_decode_reads() {
    let pattern = Checkerboard::new(SQUARE, ORDER).unwrap();
    let pose = PatternPose::new(137.0, -62.0, 0.21);
    let image = pattern.render(SIZE, SIZE, &pose);
    let buffer = BufferLayout::packed(SIZE, SIZE);
    let detection = detect_checkerboard(
        &CpuBackend::new(),
        image.as_slice(),
        buffer,
        4.0,
        10,
        0,
        0.0,
    )
    .unwrap();

    let readout = read_squares(&detection, image.as_slice(), ORDER, CodeLayout::Squares)
        .expect("the frame is full of squares");

    // 512 px of 8 px squares is a 64-square lattice, and every node is a square.
    assert!(
        readout.squares.len() > 4000,
        "only {} squares sampled",
        readout.squares.len()
    );
    for square in &readout.squares {
        assert!(square.i >= readout.i_range.0 && square.i <= readout.i_range.1);
        assert!(square.j >= readout.j_range.0 && square.j <= readout.j_range.1);
        // A square is only called a coding site once it has a colour.
        assert_eq!(square.is_white.is_some(), square.is_coding_site.is_some());
    }

    // Sites per residue class, and per supercell band within a class.
    let mut per_class: BTreeMap<(i64, i64), (usize, usize)> = BTreeMap::new();
    for square in &readout.squares {
        let Some(is_site) = square.is_coding_site else {
            continue;
        };
        let entry = per_class
            .entry((square.i.rem_euclid(3), square.j.rem_euclid(3)))
            .or_default();
        entry.0 += usize::from(is_site);
        entry.1 += 1;
    }
    let coding: Vec<(i64, i64)> = per_class
        .iter()
        .filter(|&(_, &(sites, _))| sites > 0)
        .map(|(&class, _)| class)
        .collect();
    assert_eq!(
        coding.len(),
        2,
        "the code writes two sites per supercell, but sites landed in {per_class:?}"
    );

    // The x code varies with i and the y code with j, so one class must be
    // banded along i and the other along j.
    let banded = |class: (i64, i64), along_i: bool| {
        let mut bands: BTreeMap<i64, (usize, usize)> = BTreeMap::new();
        for square in &readout.squares {
            let Some(is_site) = square.is_coding_site else {
                continue;
            };
            if (square.i.rem_euclid(3), square.j.rem_euclid(3)) != class {
                continue;
            }
            let band = if along_i { square.i } else { square.j }.div_euclid(3);
            let entry = bands.entry(band).or_default();
            entry.0 += usize::from(is_site);
            entry.1 += 1;
        }
        let mixed = bands.values().filter(|&&(s, n)| s != 0 && s != n).count();
        (bands.len(), mixed)
    };
    for &class in &coding {
        let (i_bands, i_mixed) = banded(class, true);
        let (j_bands, j_mixed) = banded(class, false);
        assert!(
            i_bands > 10 && j_bands > 10,
            "too few bands to judge {class:?}"
        );
        assert!(
            i_mixed == 0 || j_mixed == 0,
            "class {class:?} is banded along neither axis: \
             {i_mixed} mixed of {i_bands} along i, {j_mixed} of {j_bands} along j"
        );
    }

    // And the diagnostic's own decode agrees with the decode proper.
    let direct = extract_code(&detection, image.as_slice(), ORDER).unwrap();
    let via_readout = readout.code.expect("the same code, read the same way");
    assert_eq!(direct.centre_square, via_readout.centre_square);
    assert_eq!(direct.x_window, via_readout.x_window);
    assert_eq!(direct.y_window, via_readout.y_window);
}

/// The readout must follow the decode onto the denser packing, not read the
/// pattern as if it were the 3x3 one.
#[test]
fn reports_the_squares_the_two_bit_decode_reads() {
    let pattern = Checkerboard::new(SQUARE, ORDER)
        .unwrap()
        .with_code_packing(CodePacking::TwoBits);
    let pose = PatternPose::new(137.0, -62.0, 0.21);
    let image = pattern.render(SIZE, SIZE, &pose);
    let buffer = BufferLayout::packed(SIZE, SIZE);
    let detection = detect_checkerboard_with_packing(
        &CpuBackend::new(),
        image.as_slice(),
        buffer,
        4.0,
        10,
        0,
        0.0,
        CodePacking::TwoBits,
    )
    .unwrap();

    let readout = read_squares_with_packing(
        &detection,
        image.as_slice(),
        ORDER,
        CodeLayout::Squares,
        CodePacking::TwoBits,
    )
    .expect("the frame is full of squares");

    // Two bits per axis, so sites fall in four residue classes mod 5. Site
    // marking ignores the packing; the decode below is what exercises it.
    let cell = CodePacking::TwoBits.cell();
    let expected = CodePacking::TwoBits.x_sites().len() + CodePacking::TwoBits.y_sites().len();
    let mut per_class: BTreeMap<(i64, i64), usize> = BTreeMap::new();
    for square in &readout.squares {
        if square.is_coding_site == Some(true) {
            *per_class
                .entry((square.i.rem_euclid(cell), square.j.rem_euclid(cell)))
                .or_default() += 1;
        }
    }
    assert_eq!(
        per_class.len(),
        expected,
        "the denser code writes {expected} sites per {cell}x{cell} supercell, \
         but sites landed in {per_class:?}"
    );

    let direct = extract_code_with_packing(
        &detection,
        image.as_slice(),
        ORDER,
        CodeLayout::Squares,
        CodePacking::TwoBits,
    )
    .unwrap();
    let via_readout = readout.code.expect("the same code, read the same way");
    assert_eq!(direct.centre_square, via_readout.centre_square);
    assert_eq!(direct.x_window, via_readout.x_window);
    assert_eq!(direct.y_window, via_readout.y_window);
}
