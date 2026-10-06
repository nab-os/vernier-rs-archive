//! Why do clean, undegraded images fail to decode?
//!
//! The 100-pose sweep showed the diagonal layout decoding only ~35% of clean
//! renders (and the lattice layout ~96%), where the old 8-pose set scored 8/8.
//! This classifies every failure and correlates it with the pose.
//!
//! Run with `cargo run --release --example decode_failures -p vernier-pose`.

use vernier_core::buffer::BufferLayout;
use vernier_cpu::CpuBackend;
use vernier_patterns::PatternPose;
use vernier_patterns::checkerboard::{Checkerboard, CodeLayout};
use vernier_pose::checkerboard::{detect_checkerboard, extract_code_with_layout};

const SIZE: usize = 512;
const ORDER: u32 = 8;
const POSES: usize = 100;

struct Lcg(u64);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // The top 32 bits. `>> 33` kept only 31, which pinned `unit()` to
        // [0, 0.5): every pose landed at negative x, y and theta, and the
        // Box-Muller noise was biased.
        (self.0 >> 32) as u32
    }
    fn unit(&mut self) -> f64 {
        (self.next_u32() as f64 + 0.5) / (u32::MAX as f64 + 1.0)
    }
}

/// The exact pose set the sweeps use.
fn poses() -> Vec<PatternPose> {
    let mut rng = Lcg(0x9e37_79b9_7f4a_7c15);
    (0..POSES)
        .map(|_| {
            let x = (rng.unit() - 0.5) * 4000.0;
            let y = (rng.unit() - 0.5) * 4000.0;
            let theta = (rng.unit() - 0.5) * 2.0 * 0.63;
            PatternPose::new(x, y, theta)
        })
        .collect()
}

#[derive(Debug)]
enum Outcome {
    Correct,
    DetectFailed,
    DecodeError(String),
    /// Decoded, but to the wrong square: offset mod period, transform, check bits.
    Wrong((i64, i64), usize),
}

fn run(pattern: &Checkerboard, square: f64, pose: &PatternPose) -> Outcome {
    let layout = pattern.code_layout();
    let period = 3 * pattern.code().len() as i64;
    let image = pattern.render(SIZE, SIZE, pose);
    let backend = CpuBackend::new();
    let Ok(detection) = detect_checkerboard(
        &backend,
        image.as_slice(),
        BufferLayout::packed(SIZE, SIZE),
        4.0,
        10,
        0,
        0.0,
    ) else {
        return Outcome::DetectFailed;
    };
    match extract_code_with_layout(&detection, image.as_slice(), ORDER, layout) {
        Err(e) => Outcome::DecodeError(e.to_string()),
        Ok(code) => {
            let (lx, ly) = layout.to_lattice(-pose.x, -pose.y);
            let want = (
                (lx / square - 0.5).round() as i64,
                (ly / square - 0.5).round() as i64,
            );
            let off = (
                (code.centre_square.0 - want.0).rem_euclid(period),
                (code.centre_square.1 - want.1).rem_euclid(period),
            );
            if off == (0, 0) {
                Outcome::Correct
            } else {
                Outcome::Wrong(off, code.transform)
            }
        }
    }
}

fn report(label: &str, pattern: &Checkerboard, square: f64) {
    let all = poses();
    let outcomes: Vec<Outcome> = std::thread::scope(|scope| {
        let handles: Vec<_> = all
            .iter()
            .map(|pose| scope.spawn(move || run(pattern, square, pose)))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut correct = 0;
    let mut detect = 0;
    let mut errors: std::collections::BTreeMap<String, usize> = Default::default();
    let mut offsets: std::collections::BTreeMap<(i64, i64), usize> = Default::default();
    let mut transforms: std::collections::BTreeMap<usize, usize> = Default::default();
    // Failure rate by orientation, in 6 bins over +/-0.63 rad.
    let mut theta_bins = [(0usize, 0usize); 6];
    // ...and by the sign of each pose variable on its own.
    let mut by_sign = [[(0usize, 0usize); 2]; 3]; // [x, y, theta] x [>=0, <0]

    for (pose, outcome) in all.iter().zip(&outcomes) {
        let bin = (((pose.theta + 0.63) / 1.26 * 6.0) as usize).min(5);
        let failed = !matches!(outcome, Outcome::Correct);
        theta_bins[bin].0 += usize::from(failed);
        theta_bins[bin].1 += 1;
        for (k, v) in [pose.x, pose.y, pose.theta].into_iter().enumerate() {
            let slot = &mut by_sign[k][usize::from(v < 0.0)];
            slot.0 += usize::from(failed);
            slot.1 += 1;
        }
        match outcome {
            Outcome::Correct => correct += 1,
            Outcome::DetectFailed => detect += 1,
            Outcome::DecodeError(e) => *errors.entry(e.clone()).or_default() += 1,
            Outcome::Wrong(off, transform) => {
                *offsets.entry(*off).or_default() += 1;
                *transforms.entry(*transform).or_default() += 1;
            }
        }
    }

    println!("\n=== {label} (square {square:.2} px) ===");
    println!("  correct {correct}/{POSES}, detection failed {detect}");
    for (e, n) in &errors {
        println!("  decode error x{n}: {e}");
    }
    let wrong: usize = offsets.values().sum();
    println!("  wrong square x{wrong}");
    let mut top: Vec<_> = offsets.iter().collect();
    top.sort_by(|a, b| b.1.cmp(a.1));
    for (off, n) in top.iter().take(6) {
        println!("    offset {off:?} mod period  x{n}");
    }
    println!("  winning transform on wrong decodes: {transforms:?}");
    print!("  failure rate by theta bin (-0.63..0.63):");
    for (f, n) in theta_bins {
        print!(" {f}/{n}");
    }
    println!();
    for (k, name) in ["x", "y", "theta"].into_iter().enumerate() {
        println!(
            "  failures with {name} >= 0: {}/{}   {name} < 0: {}/{}",
            by_sign[k][0].0, by_sign[k][0].1, by_sign[k][1].0, by_sign[k][1].1
        );
    }

    // A few concrete failures to reproduce by hand.
    for (pose, outcome) in all
        .iter()
        .zip(&outcomes)
        .filter(|(_, o)| !matches!(o, Outcome::Correct))
        .take(4)
    {
        println!(
            "    e.g. x={:8.1} y={:8.1} theta={:+.3}  -> {outcome:?}",
            pose.x, pose.y, pose.theta
        );
    }
}

fn main() {
    let squares = Checkerboard::new(8.0, ORDER).unwrap();
    let diamonds = squares.clone().with_code_layout(CodeLayout::Diamonds);
    let diamonds_matched = Checkerboard::new(8.0 * std::f64::consts::SQRT_2, ORDER)
        .unwrap()
        .with_code_layout(CodeLayout::Diamonds);
    report("squares", &squares, 8.0);
    report("diamonds", &diamonds, 8.0);
    report(
        "diamonds, matched range",
        &diamonds_matched,
        8.0 * std::f64::consts::SQRT_2,
    );
}
