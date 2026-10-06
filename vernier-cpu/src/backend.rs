//! [`CpuBackend`] and [`CpuJob`]: the `rustfft`/`ndarray` implementations of
//! [`ComputeBackend`](vernier_core::ComputeBackend) and [`ComputeJob`](vernier_core::ComputeJob).

use std::cell::RefCell;

use vernier_core::buffer::{Buffer2D, BufferLayout};
use vernier_core::scalar::consts::{PI, TAU};
use vernier_core::{Complex32, ComputeBackend, ComputeJob, Real, Result, VernierError};

use crate::buffer::CpuBuffer;
use crate::fft::Fft2dPlanner;

/// CPU reference backend.
pub struct CpuBackend {
    planner: RefCell<Fft2dPlanner>,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self {
            planner: RefCell::new(Fft2dPlanner::new()),
        }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBackend for CpuBackend {
    type Buffer2D = CpuBuffer;
    type Job<'a> = CpuJob<'a>;

    fn begin(&self) -> Result<CpuJob<'_>> {
        Ok(CpuJob { backend: self })
    }

    fn upload(&self, data: &[Complex32], layout: BufferLayout) -> Result<Self::Buffer2D> {
        layout.check_upload(data.len())?;
        CpuBuffer::from_slice(data, layout).ok_or(VernierError::ShapeMismatch {
            lhs: layout,
            rhs: BufferLayout::packed(layout.width, layout.height),
        })
    }

    fn download(&self, buffer: &CpuBuffer) -> Result<Vec<Complex32>> {
        Ok(buffer.as_slice().to_vec())
    }

    fn name(&self) -> &str {
        "cpu-rustfft"
    }
}

// ---------------------------------------------------------------------------
// CpuJob — synchronous; submit() is a no-op
// ---------------------------------------------------------------------------

pub struct CpuJob<'a> {
    backend: &'a CpuBackend,
}

impl CpuJob<'_> {
    fn annulus_mask(spectrum: &mut CpuBuffer, min_frequency: usize, max_frequency: usize) {
        let layout = spectrum.layout();
        let (width, height) = (layout.width, layout.height);
        let min_r2 = (min_frequency * min_frequency) as Real;
        let max_r2 = if max_frequency > 0 {
            (max_frequency * max_frequency) as Real
        } else {
            Real::INFINITY
        };
        let signed = |f: usize, n: usize| -> isize {
            let f = f as isize;
            let n = n as isize;
            if f > n / 2 { f - n } else { f }
        };
        for fy in 0..height {
            let sfy = signed(fy, height) as Real;
            for fx in 0..width {
                let sfx = signed(fx, width) as Real;
                let r2 = sfx * sfx + sfy * sfy;
                if r2 < min_r2 || r2 > max_r2 {
                    let v = &mut spectrum.as_mut_slice()[fy * width + fx];
                    v.re = 0.0;
                    v.im = 0.0;
                }
            }
        }
    }

    fn halfplane_argmax(buffer: &CpuBuffer, width: usize, height: usize) -> Option<(usize, usize)> {
        let signed = |f: usize, n: usize| -> isize {
            let f = f as isize;
            let n = n as isize;
            if f > n / 2 { f - n } else { f }
        };
        let mut best = Real::NEG_INFINITY;
        let mut result = None;
        for fy in 0..height {
            let sfy = signed(fy, height);
            if sfy < 0 {
                continue;
            }
            for fx in 0..width {
                let sfx = signed(fx, width);
                // Proper halfplane: sfy > 0, or (sfy == 0 and sfx > 0).
                // Excludes DC and the conjugate-mirror boundary row.
                if sfy == 0 && sfx <= 0 {
                    continue;
                }
                let m = buffer.as_slice()[fy * width + fx].re as Real;
                if m > best {
                    best = m;
                    result = Some((fx, fy));
                }
            }
        }
        result
    }

    fn halfplane_argmax_angular_excl(
        buffer: &CpuBuffer,
        width: usize,
        height: usize,
        center_angle: Real,
        half_width: Real,
    ) -> Option<(usize, usize)> {
        let signed = |f: usize, n: usize| -> isize {
            let f = f as isize;
            let n = n as isize;
            if f > n / 2 { f - n } else { f }
        };
        let mut best = Real::NEG_INFINITY;
        let mut result = None;
        for fy in 0..height {
            let sfy_i = signed(fy, height);
            if sfy_i < 0 {
                continue;
            }
            let sfy = sfy_i as Real;
            for fx in 0..width {
                let sfx_i = signed(fx, width);
                // Proper halfplane: sfy > 0, or (sfy == 0 and sfx > 0).
                if sfy_i == 0 && sfx_i <= 0 {
                    continue;
                }
                let sfx = sfx_i as Real;
                let angle = sfy.atan2(sfx);
                let diff = ((angle - center_angle + PI).rem_euclid(TAU)) - PI;
                if diff.abs() < half_width {
                    continue;
                }
                let m = buffer.as_slice()[fy * width + fx].re as Real;
                if m > best {
                    best = m;
                    result = Some((fx, fy));
                }
            }
        }
        result
    }
}

impl ComputeJob for CpuJob<'_> {
    type Buffer2D = CpuBuffer;

    fn upload(&mut self, data: &[Complex32], layout: BufferLayout) -> Result<CpuBuffer> {
        self.backend.upload(data, layout)
    }

    fn copy_buffer(&mut self, src: &CpuBuffer) -> Result<CpuBuffer> {
        Ok(src.clone())
    }

    fn fft2d(&mut self, buf: &mut CpuBuffer) -> Result<()> {
        self.backend.planner.borrow_mut().forward(buf);
        Ok(())
    }

    fn ifft2d(&mut self, buf: &mut CpuBuffer) -> Result<()> {
        self.backend.planner.borrow_mut().inverse(buf);
        Ok(())
    }

    fn extract_phase(&mut self, buf: &CpuBuffer) -> Result<CpuBuffer> {
        let phases: Vec<Complex32> = buf
            .as_slice()
            .iter()
            .map(|c| Complex32::new(c.arg() as f32, 0.0))
            .collect();
        CpuBuffer::from_slice(&phases, buf.layout()).ok_or(VernierError::ShapeMismatch {
            lhs: buf.layout(),
            rhs: buf.layout(),
        })
    }

    fn filter(
        &mut self,
        buf: &mut CpuBuffer,
        min_frequency: usize,
        max_frequency: usize,
    ) -> Result<()> {
        Self::annulus_mask(buf, min_frequency, max_frequency);
        Ok(())
    }

    fn gaussian_blur_2d(&mut self, buf: &mut CpuBuffer, sigma: Real) -> Result<()> {
        let width = buf.layout().width;
        let height = buf.layout().height;
        let radius = (3.0 * sigma).ceil() as usize;
        let n = 2 * radius + 1;
        let kernel: Vec<f64> = (0..n)
            .map(|i| {
                let x = i as f64 - radius as f64;
                (-x * x / (2.0 * sigma * sigma)).exp()
            })
            .collect();
        let ksum: f64 = kernel.iter().sum();
        let kernel: Vec<f64> = kernel.iter().map(|&k| k / ksum).collect();

        // The blurred buffer is a magnitude spectrum, which is periodic in both
        // axes — wrap the taps circularly so carriers near the frequency-wrap
        // boundary keep their full neighborhood.
        let mut tmp = vec![0.0_f64; width * height];

        for r in 0..height {
            for c in 0..width {
                let mut value = 0.0_f64;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let sc = (c as isize + ki as isize - radius as isize).rem_euclid(width as isize)
                        as usize;
                    value += buf.as_slice()[r * width + sc].re as f64 * kv;
                }
                tmp[r * width + c] = value;
            }
        }

        for r in 0..height {
            for c in 0..width {
                let mut value = 0.0_f64;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let sr = (r as isize + ki as isize - radius as isize)
                        .rem_euclid(height as isize) as usize;
                    value += tmp[sr * width + c] * kv;
                }
                buf.as_mut_slice()[r * width + c].re = value as f32;
            }
        }

        Ok(())
    }

    fn bandpass_from_peaks(
        &mut self,
        buf: &mut CpuBuffer,
        peaks: &CpuBuffer,
        direction: u32,
        sigma: Real,
    ) -> Result<()> {
        let base = direction as usize * 2;
        let cx = peaks.as_slice()[base].re as usize;
        let cy = peaks.as_slice()[base + 1].re as usize;
        self.bandpass_filter(buf, cx, cy, sigma)
    }

    fn bandpass_filter(
        &mut self,
        buf: &mut CpuBuffer,
        cx: usize,
        cy: usize,
        sigma: Real,
    ) -> Result<()> {
        let layout = buf.layout();
        let (width, height) = (layout.width, layout.height);
        let two_sigma_sq = 2.0 * sigma * sigma;

        let circular_delta = |a: usize, center: usize, n: usize| -> Real {
            let d = a as isize - center as isize;
            let n = n as isize;
            let d = ((d % n) + n) % n;
            let d = if d > n / 2 { d - n } else { d };
            d as Real
        };

        let inv_denom = -1.0 / two_sigma_sq;
        let gain_x: Vec<f32> = (0..width)
            .map(|fx| {
                let dx = circular_delta(fx, cx, width);
                (dx * dx * inv_denom).exp() as f32
            })
            .collect();
        let gain_y: Vec<f32> = (0..height)
            .map(|fy| {
                let dy = circular_delta(fy, cy, height);
                (dy * dy * inv_denom).exp() as f32
            })
            .collect();

        let data = buf.as_mut_slice();
        for fy in 0..height {
            let gy = gain_y[fy];
            let row = &mut data[fy * width..(fy + 1) * width];
            for (fx, v) in row.iter_mut().enumerate() {
                let gain = gy * gain_x[fx];
                v.re *= gain;
                v.im *= gain;
            }
        }
        Ok(())
    }

    fn peak_search(
        &mut self,
        buffer: &mut CpuBuffer,
        min_frequency: usize,
        max_frequency: usize,
        smoothing_sigma: Real,
        sigma: Real,
    ) -> Result<Option<CpuBuffer>> {
        let mut spectrum = buffer.clone();
        let layout = spectrum.layout();
        let (width, height) = (layout.width, layout.height);

        for v in spectrum.as_mut_slice().iter_mut() {
            v.re = (v.re * v.re + v.im * v.im).sqrt();
            v.im = 0.0;
        }

        Self::annulus_mask(&mut spectrum, min_frequency, max_frequency);

        let signed = |f: usize, n: usize| -> isize {
            let f = f as isize;
            let n = n as isize;
            if f > n / 2 { f - n } else { f }
        };

        if smoothing_sigma > 0.0 {
            self.gaussian_blur_2d(&mut spectrum, smoothing_sigma)
                .unwrap();
        }

        let (cx1, cy1) = {
            if let Some(a) = Self::halfplane_argmax(&spectrum, width, height) {
                a
            } else {
                return Ok(None);
            }
        };
        let sfx1 = signed(cx1, width) as Real;
        let sfy1 = signed(cy1, height) as Real;

        let distance = (sfx1 * sfx1 + sfy1 * sfy1).sqrt();
        let center_angle = sfy1.atan2(sfx1);
        let half_width = (3.0 * sigma).atan2(distance);

        let r_min_sq = (distance * 0.5) * (distance * 0.5);
        let r_max_sq = (distance * 2.0) * (distance * 2.0);
        for fy in 0..height {
            let sfy_i = signed(fy, height) as Real;
            for fx in 0..width {
                let sfx_i = signed(fx, width) as Real;
                let r_sq = sfx_i * sfx_i + sfy_i * sfy_i;
                if r_sq < r_min_sq || r_sq > r_max_sq {
                    spectrum.as_mut_slice()[fy * width + fx].re = 0.0;
                }
            }
        }

        let (cx2, cy2) = {
            if let Some(a) = Self::halfplane_argmax_angular_excl(
                &spectrum,
                width,
                height,
                center_angle,
                half_width,
            ) {
                a
            } else {
                return Ok(None);
            }
        };

        let sfx2 = signed(cx2, width) as Real;
        let (d1x, d1y, d2x, d2y) = if sfx1 >= sfx2 {
            (cx1, cy1, cx2, cy2)
        } else {
            (cx2, cy2, cx1, cy1)
        };

        Ok(CpuBuffer::from_slice(
            &[
                Complex32 {
                    re: d1x as f32,
                    im: 0.0,
                },
                Complex32 {
                    re: d1y as f32,
                    im: 0.0,
                },
                Complex32 {
                    re: d2x as f32,
                    im: 0.0,
                },
                Complex32 {
                    re: d2y as f32,
                    im: 0.0,
                },
            ],
            BufferLayout {
                width: 2,
                height: 2,
                row_stride: 2,
            },
        ))
    }

    fn spectral_plane_fit_two(
        &mut self,
        spectrum: &CpuBuffer,
        peaks: &CpuBuffer,
        sigma: Real,
    ) -> Result<CpuBuffer> {
        let layout = spectrum.layout();
        let (width, height) = (layout.width, layout.height);
        let data = spectrum.as_slice();
        let peaks_data = peaks.as_slice();

        let signed = |f: usize, n: usize| -> isize {
            let f = f as isize;
            let n = n as isize;
            if f > n / 2 { f - n } else { f }
        };

        let sfx1 = signed(peaks_data[0].re as usize, width) as f64;
        let sfy1 = signed(peaks_data[1].re as usize, height) as f64;
        let sfx2 = signed(peaks_data[2].re as usize, width) as f64;
        let sfy2 = signed(peaks_data[3].re as usize, height) as f64;

        let neg_inv_two_sigma_sq = -1.0_f64 / (2.0 * sigma.powi(2));

        let mut acc = [[0.0_f64; 5]; 2]; // [c_re, c_im, sfx_numerator, sfy_numerator, denominator] per direction

        for fy in 0..height {
            let sfy = signed(fy, height) as f64;
            for fx in 0..width {
                let sfx = signed(fx, width) as f64;
                let s = data[fy * width + fx];
                let sign = if (fx + fy) % 2 == 0 {
                    1.0_f64
                } else {
                    -1.0_f64
                };
                let s_re = s.re as f64 * sign;
                let s_im = s.im as f64 * sign;
                let magnitude_sq = (s.re as f64).powi(2) + (s.im as f64).powi(2);

                let dx1 = sfx - sfx1;
                let dy1 = sfy - sfy1;
                let weight1 = (neg_inv_two_sigma_sq * (dx1 * dx1 + dy1 * dy1)).exp();
                let weighted_magnitude1 = weight1 * magnitude_sq;
                acc[0][0] += weight1 * s_re;
                acc[0][1] += weight1 * s_im;
                acc[0][2] += weighted_magnitude1 * sfx;
                acc[0][3] += weighted_magnitude1 * sfy;
                acc[0][4] += weighted_magnitude1;

                let dx2 = sfx - sfx2;
                let dy2 = sfy - sfy2;
                let weight2 = (neg_inv_two_sigma_sq * (dx2 * dx2 + dy2 * dy2)).exp();
                let weighted_magnitude2 = weight2 * magnitude_sq;
                acc[1][0] += weight2 * s_re;
                acc[1][1] += weight2 * s_im;
                acc[1][2] += weighted_magnitude2 * sfx;
                acc[1][3] += weighted_magnitude2 * sfy;
                acc[1][4] += weighted_magnitude2;
            }
        }

        let tau = std::f64::consts::TAU;
        let mut result = Vec::with_capacity(6);
        for [c_re, c_im, sfx_numerator, sfy_numerator, denominator] in acc {
            let a = if denominator != 0.0 {
                (tau * sfx_numerator / denominator / width as f64) as f32
            } else {
                0.0
            };
            let b = if denominator != 0.0 {
                (tau * sfy_numerator / denominator / height as f64) as f32
            } else {
                0.0
            };
            let c = c_im.atan2(c_re) as f32;
            result.push(Complex32::new(a, 0.0));
            result.push(Complex32::new(b, 0.0));
            result.push(Complex32::new(c, 0.0));
        }

        Ok(CpuBuffer::from_slice(
            &result,
            vernier_core::buffer::BufferLayout {
                width: 6,
                height: 1,
                row_stride: 6,
            },
        )
        .unwrap())
    }

    fn submit(self) -> Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Keep argmax_magnitude as a standalone method on CpuBackend for tests
// ---------------------------------------------------------------------------
impl CpuBackend {
    pub fn argmax_magnitude(&self, buffer: &CpuBuffer) -> Result<(usize, Real)> {
        let mut best_index = 0usize;
        let mut best = Real::NEG_INFINITY;
        for (index, element) in buffer.as_slice().iter().enumerate() {
            let magnitude_sq = element.norm_sqr();
            if magnitude_sq > best {
                best = magnitude_sq;
                best_index = index;
            }
        }
        Ok((best_index, best.sqrt()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use vernier_core::buffer::BufferLayout;

    fn checkerboard(width: usize, height: usize) -> (Vec<Complex32>, BufferLayout) {
        let layout = BufferLayout::packed(width, height);
        let mut data = Vec::with_capacity(layout.len());
        for r in 0..height {
            for c in 0..width {
                let value = if (r + c) % 2 == 0 { 1.0 } else { -1.0 };
                data.push(Complex32::new(value, 0.0));
            }
        }
        (data, layout)
    }

    #[test]
    fn fft_then_ifft_is_identity() {
        let backend = CpuBackend::new();
        let (data, layout) = checkerboard(8, 8);
        let mut buf = backend.upload(&data, layout).unwrap();
        let mut job = backend.begin().unwrap();
        job.fft2d(&mut buf).unwrap();
        job.ifft2d(&mut buf).unwrap();
        job.submit().unwrap();
        let out = backend.download(&buf).unwrap();
        for (orig, got) in data.iter().zip(out.iter()) {
            assert_abs_diff_eq!(orig.re, got.re, epsilon = 1e-4);
            assert_abs_diff_eq!(orig.im, got.im, epsilon = 1e-4);
        }
    }

    #[test]
    fn argmax_finds_the_dc_spike() {
        let backend = CpuBackend::new();
        let layout = BufferLayout::packed(8, 8);
        let data = vec![Complex32::new(1.0, 0.0); layout.len()];
        let mut buf = backend.upload(&data, layout).unwrap();
        let mut job = backend.begin().unwrap();
        job.fft2d(&mut buf).unwrap();
        job.submit().unwrap();
        let (idx, _mag) = backend.argmax_magnitude(&buf).unwrap();
        assert_eq!(idx, 0, "all energy should sit at the DC bin");
    }

    #[test]
    fn non_square_round_trips() {
        let backend = CpuBackend::new();
        let (data, layout) = checkerboard(16, 4);
        let mut buf = backend.upload(&data, layout).unwrap();
        let mut job = backend.begin().unwrap();
        job.fft2d(&mut buf).unwrap();
        job.ifft2d(&mut buf).unwrap();
        job.submit().unwrap();
        let out = backend.download(&buf).unwrap();
        for (orig, got) in data.iter().zip(out.iter()) {
            assert_abs_diff_eq!(orig.re, got.re, epsilon = 1e-4);
        }
    }
}
