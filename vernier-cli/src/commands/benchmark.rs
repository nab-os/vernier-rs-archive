//! The benchmark command: time the full detection pipeline on a synthetic image.
//!
//! Times upload → FFT → peak search → two-direction bandpass/IFFT/phase,
//! so host↔device transfer cost is included in the honest GPU comparison.

use std::time::Instant;

use vernier_core::buffer::BufferLayout;
use vernier_core::{Complex32, ComputeBackend, Real};
use vernier_patterns::PatternPose;
use vernier_patterns::periodic::Periodic;
use vernier_spectral::spectrum::analyze_two;

use crate::backend_select::BackendTask;

/// Parameters for a benchmark run.
pub struct Benchmark {
    /// Square image side length.
    pub size: usize,
    /// Number of timed iterations.
    pub iterations: usize,
    /// Band-pass filter width in bins.
    pub sigma: f32,
    /// Inner annulus radius for peak search (bins); 0 = no lower limit.
    pub min_frequency: usize,
    /// Outer annulus radius for peak search (bins); 0 = no upper limit.
    pub max_frequency: usize,
    /// Gaussian blur sigma applied to magnitude before peak search.
    pub smoothing_sigma: f32,
}

/// What a benchmark run reports.
pub struct BenchReport {
    /// Backend name.
    pub backend: String,
    /// Image side length used.
    pub size: usize,
    /// Iterations timed.
    pub iterations: usize,
    /// Mean wall-clock time per full pipeline iteration, in milliseconds.
    pub mean_ms: f64,
    /// Fastest single iteration, in milliseconds.
    pub best_ms: f64,
}

impl Benchmark {
    fn synthetic_image(size: usize) -> Vec<f32> {
        let period = (size as f32) / 16.0;
        let pattern = Periodic::new(period as Real);
        let pose = PatternPose::new(0.0, 0.0, 0.1);
        let img = pattern.render(size, size, &pose);
        img.as_slice().to_vec()
    }
}

impl BackendTask for Benchmark {
    type Output = BenchReport;

    fn run<B: ComputeBackend>(&self, backend: &B) -> BenchReport {
        let pixels = Self::synthetic_image(self.size);
        let layout = BufferLayout::packed(self.size, self.size);
        let complex: Vec<Complex32> = pixels
            .iter()
            .map(|&v| Complex32::new(v as f32, 0.0))
            .collect();
        let sigma = self.sigma as Real;
        let smoothing = self.smoothing_sigma as Real;

        // Warm-up: prime any FFT planning or GPU pipeline caches.
        let _ = analyze_two(
            backend,
            &complex,
            layout,
            sigma,
            self.min_frequency,
            self.max_frequency,
            smoothing,
        );

        let mut best = f64::INFINITY;
        let mut total = 0.0;
        for _ in 0..self.iterations {
            let start = Instant::now();
            analyze_two(
                backend,
                &complex,
                layout,
                sigma,
                self.min_frequency,
                self.max_frequency,
                smoothing,
            )
            .expect("detection failed during benchmark");
            let ms = start.elapsed().as_secs_f64() * 1e3;
            total += ms;
            best = best.min(ms);
        }

        BenchReport {
            backend: backend.name().to_string(),
            size: self.size,
            iterations: self.iterations,
            mean_ms: total / self.iterations as f64,
            best_ms: best,
        }
    }
}
