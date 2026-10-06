//! Evidence probe: for every decode, how strong was the evidence, and was it
//! right? Used to choose the decoder's false-accept limit.
//!
//! CSV on stdout: design,variant,level,tile,outcome,false_accept,errors_x,errors_y,check_bits,offset_i,offset_j,err_px,runner_up_fa
#![allow(dead_code)]

use vernier_core::buffer::BufferLayout;
use vernier_cpu::CpuBackend;
use vernier_patterns::PatternPose;
use vernier_patterns::checkerboard::{Checkerboard, CodeLayout};
use vernier_pose::checkerboard::{detect_checkerboard, solve_checkerboard_with_layout};

const SIZE: usize = 512;
const ORDER: u32 = 8;
const POSES: usize = 100;

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

// ------------------------------------------------------------------ probe

fn offsets() -> Vec<(f64, f64, f64)> {
    let mut rng = Lcg(0x9e37_79b9_7f4a_7c15);
    (0..POSES)
        .map(|_| {
            let x = (rng.unit() - 0.5) * 4000.0;
            let y = (rng.unit() - 0.5) * 4000.0;
            let d = (rng.unit() - 0.5) * 2.0 * 5f64.to_radians();
            (x, y, d)
        })
        .collect()
}

fn main() {
    // (variant, level, tile). Tile sweeps use a clean image (level ignored).
    let mut conditions: Vec<(Option<Variant>, f64, f64)> = vec![(None, 0.0, 8.0)];
    for (v, levels) in [
        (Variant::Ramp, vec![0.6, 1.0, 1.2]),
        (Variant::Vignette, vec![0.8, 1.0]),
        (Variant::Defocus, vec![1.5, 2.0, 2.5, 3.0, 3.5, 4.0]),
        (Variant::Motion0, vec![6.0, 8.0, 10.0, 12.0, 16.0, 22.0]),
        (Variant::Motion45, vec![8.0, 10.0, 12.0, 16.0, 22.0]),
        (Variant::Gaussian, vec![1.0, 1.6, 2.0]),
        (Variant::Shot, vec![1.0]),
        (Variant::SaltPepper, vec![0.5]),
    ] {
        for l in levels {
            conditions.push((Some(v), l, 8.0));
        }
    }
    for tile in [4.0, 3.0, 2.5, 2.0] {
        conditions.push((None, 0.0, tile));
    }

    let offs = offsets();
    println!(
        "design,variant,level,tile,outcome,false_accept,errors_x,errors_y,check_bits,offset_i,offset_j,err_px,runner_up_fa"
    );
    for (variant, level, tile) in conditions {
        for (name, layout, nominal) in [
            ("square", CodeLayout::Squares, 0.0),
            ("diamond", CodeLayout::Diamonds, 0.0),
        ] {
            let pattern = Checkerboard::new(tile, ORDER)
                .unwrap()
                .with_code_layout(layout);
            let period = 3 * pattern.code().len() as i64;
            let rows: Vec<String> = std::thread::scope(|scope| {
                let handles: Vec<_> = offs
                    .iter()
                    .enumerate()
                    .map(|(index, &(x, y, d))| {
                        let pattern = &pattern;
                        scope.spawn(move || {
                            let pose = PatternPose::new(x, y, nominal + d);
                            let mut image = pattern.render(SIZE, SIZE, &pose).as_slice().to_vec();
                            if let Some(v) = variant {
                                let seed = (v.id() << 56) ^ (level.to_bits() >> 8) ^ index as u64;
                                degrade(&mut image, v, level, seed);
                            }
                            let label = variant.map(|v| v.name()).unwrap_or("clean");
                            let backend = CpuBackend::new();
                            let prefix = format!("{name},{label},{level},{tile}");
                            let Ok(det) = detect_checkerboard(
                                &backend,
                                &image,
                                BufferLayout::packed(SIZE, SIZE),
                                4.0,
                                10,
                                0,
                                0.0,
                            ) else {
                                return format!("{prefix},nodetect,,,,,,,,");
                            };
                            let Ok((rec, code)) =
                                solve_checkerboard_with_layout(&det, &image, tile, ORDER, layout)
                            else {
                                return format!("{prefix},nodecode,,,,,,,,");
                            };
                            let (lx, ly) = layout.to_lattice(-pose.x, -pose.y);
                            let want = (
                                (lx / tile - 0.5).round() as i64,
                                (ly / tile - 0.5).round() as i64,
                            );
                            let signed = |v: i64| {
                                let v = v.rem_euclid(period);
                                if v > period / 2 { v - period } else { v }
                            };
                            let (oi, oj) = (
                                signed(code.centre_square.0 - want.0),
                                signed(code.centre_square.1 - want.1),
                            );
                            let (ex, ey) = pattern.wrap_offset(rec.x + pose.x, rec.y + pose.y);
                            let outcome = if oi == 0 && oj == 0 {
                                "correct"
                            } else {
                                "wrong"
                            };
                            format!(
                                "{prefix},{outcome},{:e},{},{},{},{oi},{oj},{:.4},{:e}",
                                code.false_accept,
                                code.bit_errors.0,
                                code.bit_errors.1,
                                code.check_bits,
                                (ex * ex + ey * ey).sqrt(),
                                code.runner_up_false_accept
                            )
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            for r in rows {
                println!("{r}");
            }
        }
        eprintln!("done {:?} {level} tile {tile}", variant.map(|v| v.name()));
    }
}
