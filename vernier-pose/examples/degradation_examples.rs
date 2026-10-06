//! Example images for every test in `diamond_vs_square`.
//!
//! Renders pose #0 of the +/-5 deg run for both designs and applies each
//! degradation at five levels across its tested range, using the suite's own
//! degradation code and seeds -- so each image is exactly one the decoder was
//! given. Also renders the tile-size sweep. Each image is decoded, and the
//! outcome recorded, so a picture can be tied to its result.
//!
//! Writes 8-bit PGM files plus manifest.csv into the directory given as the
//! first argument.
//!
//! Run with
//!   cargo run --release --example degradation_examples -p vernier-pose -- <out-dir>

use std::io::Write;

use vernier_core::buffer::BufferLayout;
use vernier_cpu::CpuBackend;
use vernier_patterns::PatternPose;
use vernier_patterns::checkerboard::{Checkerboard, CodeLayout};
use vernier_pose::checkerboard::{detect_checkerboard, solve_checkerboard_with_layout};

const SIZE: usize = 512;
const ORDER: u32 = 8;

// ------------------------------------------------------------------ randomness

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        // Mix the seed so neighbouring seeds do not start correlated.
        let mut z = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        Lcg(z ^ (z >> 31))
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn unit(&mut self) -> f64 {
        (self.next_u32() as f64 + 0.5) / (u32::MAX as f64 + 1.0)
    }
    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.unit(), self.unit());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
    /// Poisson sample: Knuth for small means, normal approximation above 30.
    fn poisson(&mut self, mean: f64) -> f64 {
        if mean <= 0.0 {
            return 0.0;
        }
        if mean > 30.0 {
            return (mean + mean.sqrt() * self.normal()).round().max(0.0);
        }
        let limit = (-mean).exp();
        let (mut k, mut p) = (0.0, 1.0);
        loop {
            p *= self.unit();
            if p <= limit {
                return k;
            }
            k += 1.0;
        }
    }
}

// ------------------------------------------------------------------ degradations

#[derive(Clone, Copy)]
enum Variant {
    Ramp,
    Vignette,
    Shading,
    Defocus,
    Motion0,
    Motion45,
    Gaussian,
    Shot,
    SaltPepper,
}

impl Variant {
    fn family(self) -> &'static str {
        match self {
            Variant::Ramp | Variant::Vignette | Variant::Shading => "illumination",
            Variant::Defocus | Variant::Motion0 | Variant::Motion45 => "blur",
            Variant::Gaussian | Variant::Shot | Variant::SaltPepper => "noise",
        }
    }
    fn name(self) -> &'static str {
        match self {
            Variant::Ramp => "linear ramp",
            Variant::Vignette => "vignetting",
            Variant::Shading => "low-frequency shading",
            Variant::Defocus => "gaussian defocus",
            Variant::Motion0 => "motion blur 0deg",
            Variant::Motion45 => "motion blur 45deg",
            Variant::Gaussian => "gaussian noise",
            Variant::Shot => "shot noise",
            Variant::SaltPepper => "salt and pepper",
        }
    }
    fn levels(self) -> Vec<f64> {
        let grid = |hi: f64, step: f64| {
            let n = (hi / step).round() as usize;
            (0..=n).map(|k| k as f64 * step).collect::<Vec<_>>()
        };
        match self {
            Variant::Ramp => grid(1.2, 0.1),
            Variant::Vignette | Variant::Shading => grid(1.0, 0.1),
            Variant::Defocus => grid(4.0, 0.25),
            Variant::Motion0 | Variant::Motion45 => grid(24.0, 2.0),
            Variant::Gaussian => grid(2.0, 0.1),
            // Photons collected by a fully white pixel: fewer is worse.
            Variant::Shot => vec![1000.0, 300.0, 100.0, 30.0, 10.0, 5.0, 3.0, 2.0, 1.0],
            Variant::SaltPepper => grid(0.5, 0.05),
        }
    }
    fn id(self) -> u64 {
        self as u64
    }
}

const ALL: [Variant; 9] = [
    Variant::Ramp,
    Variant::Vignette,
    Variant::Shading,
    Variant::Defocus,
    Variant::Motion0,
    Variant::Motion45,
    Variant::Gaussian,
    Variant::Shot,
    Variant::SaltPepper,
];

fn sample_bilinear(image: &[f32], x: f64, y: f64) -> f64 {
    let max = (SIZE - 1) as f64;
    let (x, y) = (x.clamp(0.0, max), y.clamp(0.0, max));
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(SIZE - 1), (y0 + 1).min(SIZE - 1));
    let (fx, fy) = (x - x0 as f64, y - y0 as f64);
    let at = |c: usize, r: usize| image[r * SIZE + c] as f64;
    (at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx) * (1.0 - fy)
        + (at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx) * fy
}

fn gaussian_blur(image: &mut [f32], sigma: f64) {
    if sigma <= 0.0 {
        return;
    }
    let radius = (3.0 * sigma).ceil() as isize;
    let kernel: Vec<f64> = (-radius..=radius)
        .map(|d| (-(d * d) as f64 / (2.0 * sigma * sigma)).exp())
        .collect();
    let norm: f64 = kernel.iter().sum();
    let clamp = |v: isize| v.clamp(0, SIZE as isize - 1) as usize;
    let mut pass = vec![0.0f32; image.len()];
    for r in 0..SIZE {
        for c in 0..SIZE {
            let acc: f64 = kernel
                .iter()
                .enumerate()
                .map(|(k, w)| w * image[r * SIZE + clamp(c as isize + k as isize - radius)] as f64)
                .sum();
            pass[r * SIZE + c] = (acc / norm) as f32;
        }
    }
    for c in 0..SIZE {
        for r in 0..SIZE {
            let acc: f64 = kernel
                .iter()
                .enumerate()
                .map(|(k, w)| w * pass[clamp(r as isize + k as isize - radius) * SIZE + c] as f64)
                .sum();
            image[r * SIZE + c] = (acc / norm) as f32;
        }
    }
}

fn motion_blur(image: &mut [f32], length: f64, degrees: f64) {
    if length <= 0.0 {
        return;
    }
    let (dy, dx) = degrees.to_radians().sin_cos();
    let taps = length.round() as i64 + 1;
    let source = image.to_vec();
    for r in 0..SIZE {
        for c in 0..SIZE {
            let mut acc = 0.0;
            for k in 0..taps {
                let t = k as f64 - (taps - 1) as f64 / 2.0;
                acc += sample_bilinear(&source, c as f64 + t * dx, r as f64 + t * dy);
            }
            image[r * SIZE + c] = (acc / taps as f64) as f32;
        }
    }
}

fn degrade(image: &mut [f32], variant: Variant, level: f64, seed: u64) {
    let half = SIZE as f64 / 2.0;
    let gain_map = |image: &mut [f32], gain: &dyn Fn(f64, f64) -> f64| {
        for r in 0..SIZE {
            for c in 0..SIZE {
                let p = &mut image[r * SIZE + c];
                *p = (*p as f64 * gain(c as f64 - half, r as f64 - half)) as f32;
            }
        }
    };
    match variant {
        Variant::Ramp => gain_map(image, &|x, y| {
            // Along the image diagonal, spanning [1 - a, 1 + a].
            1.0 + level * (x + y) / (2.0 * half)
        }),
        Variant::Vignette => {
            let rmax2 = 2.0 * half * half;
            gain_map(image, &|x, y| {
                (1.0 - level * (x * x + y * y) / rmax2).max(0.0)
            })
        }
        Variant::Shading => {
            // Blotches about 256 px across, a phase offset so the centre is not
            // special.
            let k = std::f64::consts::TAU / 256.0;
            gain_map(image, &|x, y| {
                1.0 + level * (k * x + 0.7).sin() * (k * y + 1.9).sin()
            })
        }
        Variant::Defocus => gaussian_blur(image, level),
        Variant::Motion0 => motion_blur(image, level, 0.0),
        Variant::Motion45 => motion_blur(image, level, 45.0),
        Variant::Gaussian => {
            let mut rng = Lcg::new(seed);
            for p in image.iter_mut() {
                *p = (*p as f64 + level * rng.normal()) as f32;
            }
        }
        Variant::Shot => {
            let mut rng = Lcg::new(seed);
            for p in image.iter_mut() {
                *p = (rng.poisson(level * (*p as f64).max(0.0)) / level) as f32;
            }
        }
        Variant::SaltPepper => {
            let mut rng = Lcg::new(seed);
            for p in image.iter_mut() {
                if rng.unit() < level {
                    *p = if rng.unit() < 0.5 { 0.0 } else { 1.0 };
                }
            }
        }
    }
    // A sensor clips; that is part of every one of these.
    for p in image.iter_mut() {
        *p = p.clamp(0.0, 1.0);
    }
}

// ------------------------------------------------------------------ examples

/// Pose #0 of `diamond_vs_square`'s offsets at +/-5 deg jitter.
fn pose_offset() -> (f64, f64, f64) {
    let mut rng = Lcg(0x9e37_79b9_7f4a_7c15);
    let x = (rng.unit() - 0.5) * 4000.0;
    let y = (rng.unit() - 0.5) * 4000.0;
    let d = (rng.unit() - 0.5) * 2.0 * 5f64.to_radians();
    (x, y, d)
}

fn write_pgm(path: &std::path::Path, image: &[f32]) {
    let mut file = std::fs::File::create(path).expect("create pgm");
    write!(file, "P5\n{SIZE} {SIZE}\n255\n").unwrap();
    let bytes: Vec<u8> = image
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    file.write_all(&bytes).unwrap();
}

fn decodes(image: &[f32], pattern: &Checkerboard, square: f64, pose: &PatternPose) -> bool {
    let backend = CpuBackend::new();
    let Ok(detection) = detect_checkerboard(
        &backend,
        image,
        BufferLayout::packed(SIZE, SIZE),
        4.0,
        10,
        0,
        0.0,
    ) else {
        return false;
    };
    let Ok((recovered, _)) =
        solve_checkerboard_with_layout(&detection, image, square, ORDER, pattern.code_layout())
    else {
        return false;
    };
    // Within half a square of the truth, modulo one code period.
    let (ex, ey) = pattern.wrap_offset(recovered.x + pose.x, recovered.y + pose.y);
    ex.abs() < 0.5 * square && ey.abs() < 0.5 * square
}

fn slug(s: &str) -> String {
    s.replace(' ', "-")
}

fn main() {
    let out = std::path::PathBuf::from(std::env::args().nth(1).expect("output directory"));
    std::fs::create_dir_all(&out).unwrap();
    let mut manifest = std::fs::File::create(out.join("manifest.csv")).unwrap();
    writeln!(manifest, "family,variant,level,design,file,decoded").unwrap();

    let (x, y, d) = pose_offset();
    let designs = [
        ("square", CodeLayout::Squares, 0.0),
        ("diamond", CodeLayout::Diamonds, 0.0),
    ];

    for (name, layout, nominal) in designs {
        let pose = PatternPose::new(x, y, nominal + d);
        let pattern = Checkerboard::new(8.0, ORDER)
            .unwrap()
            .with_code_layout(layout);
        let clean = pattern.render(SIZE, SIZE, &pose).as_slice().to_vec();

        for variant in ALL {
            let levels = variant.levels();
            let n = levels.len() - 1;
            let mut picks = vec![0, n / 4, n / 2, 3 * n / 4, n];
            picks.dedup();
            for i in picks {
                let level = levels[i];
                let mut image = clean.clone();
                // Same seed as the suite for pose index 0.
                let seed = (variant.id() << 56) ^ (level.to_bits() >> 8);
                degrade(&mut image, variant, level, seed);
                let file = format!("{name}_{}_{level}.pgm", slug(variant.name()));
                write_pgm(&out.join(&file), &image);
                let ok = decodes(&image, &pattern, 8.0, &pose);
                writeln!(
                    manifest,
                    "{},{},{level},{name},{file},{ok}",
                    variant.family(),
                    variant.name()
                )
                .unwrap();
            }
        }

        for tile in [8.0, 4.0, 3.0, 2.5, 2.0, 1.5] {
            let small = Checkerboard::new(tile, ORDER)
                .unwrap()
                .with_code_layout(layout);
            let image = small.render(SIZE, SIZE, &pose).as_slice().to_vec();
            let file = format!("{name}_tile-size_{tile}.pgm");
            write_pgm(&out.join(&file), &image);
            let ok = decodes(&image, &small, tile, &pose);
            writeln!(manifest, "resolution,tile size,{tile},{name},{file},{ok}").unwrap();
        }
        eprintln!("{name} done");
    }
}
