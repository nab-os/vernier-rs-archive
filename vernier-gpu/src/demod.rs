//! [`LocalDemodulator`] on the GPU.
//!
//! A batch of windows is one dispatch, a workgroup per window, each summing
//! its window in `f32`; the host reads the sums back and turns them into a
//! phase, a quality and an offset in `f64`. The field is three dispatches:
//! the per-pixel terms, a blur along the rows, a blur down the columns
//! combined into `z` and the local deviation, whose phase and amplitude the
//! host takes.
//!
//! The sums only hold the phase against the reference, which stays within a
//! turn or so; the unwrapped phase is added back on the host, in `f64`.

use std::sync::Arc;

use rayon::prelude::*;
use vernier_core::scalar::consts::TAU;
use vernier_core::{
    DemodWindow, FieldDemod, LocalDemodulator, Real, Result, VernierError, WindowDemod,
};
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::device::Device;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};

use crate::backend::{GpuBackend, build_pipeline};

/// Floats per window in the window buffer and in the sums buffer; see
/// `demod_windows.glsl`.
const WINDOW_FLOATS: usize = 16;
const SUM_FLOATS: usize = 28;

/// Most workgroups along one dispatch axis that every device takes.
const MAX_GROUPS: usize = 65535;

/// The pipelines of the local demodulation.
pub(crate) struct DemodPipelines {
    windows: Arc<ComputePipeline>,
    field_terms: Arc<ComputePipeline>,
    field_rows: Arc<ComputePipeline>,
    field_columns: Arc<ComputePipeline>,
    field_combine: Arc<ComputePipeline>,
}

impl DemodPipelines {
    pub(crate) fn new(device: Arc<Device>) -> Self {
        let build = |module| build_pipeline(device.clone(), module);
        Self {
            windows: build(windows_shader::load(device.clone()).unwrap()),
            field_terms: build(field_terms_shader::load(device.clone()).unwrap()),
            field_rows: build(field_rows_shader::load(device.clone()).unwrap()),
            field_columns: build(field_columns_shader::load(device.clone()).unwrap()),
            field_combine: build(field_combine_shader::load(device.clone()).unwrap()),
        }
    }
}

/// Term planes of the demodulated field; see `field_terms.glsl`.
const FIELD_TERMS: usize = 10;

/// A frame resident on the GPU, what [`GpuBackend`] demodulates.
#[derive(Clone, Debug)]
pub struct GpuFrame {
    data: Subbuffer<[f32]>,
    width: usize,
    height: usize,
    /// The frame's mean intensity, which the field takes off every pixel so
    /// its variance does not cancel in `f32`.
    level: f32,
}

fn backend_error(e: impl std::fmt::Display) -> VernierError {
    VernierError::Backend(e.to_string())
}

impl GpuBackend {
    /// A buffer the host fills once and the device reads.
    fn input_buffer(&self, data: &[f32]) -> Result<Subbuffer<[f32]>> {
        Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            data.iter().copied(),
        )
        .map_err(backend_error)
    }

    /// A buffer of `n` floats only the device touches, or the host reads
    /// back with `readable`.
    fn work_buffer(&self, n: usize, readable: bool) -> Result<Subbuffer<[f32]>> {
        let memory_type_filter = if readable {
            MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS
        } else {
            MemoryTypeFilter::PREFER_DEVICE
        };
        Buffer::new_slice::<f32>(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter,
                ..Default::default()
            },
            n as u64,
        )
        .map_err(backend_error)
    }

    /// Records one dispatch of `pipeline` over `groups` workgroups, with
    /// `buffers` bound in order from binding 0.
    fn record<P: BufferContents>(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        pipeline: &Arc<ComputePipeline>,
        buffers: &[&Subbuffer<[f32]>],
        push: P,
        groups: [u32; 3],
    ) -> Result<()> {
        let set = self.make_descriptor_set(
            pipeline,
            buffers.iter().enumerate().map(|(binding, buffer)| {
                WriteDescriptorSet::buffer(binding as u32, (*buffer).clone())
            }),
        );
        builder
            .bind_pipeline_compute(pipeline.clone())
            .map_err(backend_error)?
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline.layout().clone(),
                0,
                set,
            )
            .map_err(backend_error)?
            .push_constants(pipeline.layout().clone(), 0, push)
            .map_err(backend_error)?;
        unsafe { builder.dispatch(groups) }.map_err(backend_error)?;
        Ok(())
    }
}

/// A window as `demod_windows.glsl` reads it.
fn pack(window: &DemodWindow) -> [f32; WINDOW_FLOATS] {
    let [c1, c2] = window.carriers;
    [
        window.x as f32,
        window.y as f32,
        window.sigma as f32,
        window.radius() as f32,
        c1.k[0] as f32,
        c1.k[1] as f32,
        c1.h[0] as f32,
        c1.h[1] as f32,
        c1.h[2] as f32,
        c2.k[0] as f32,
        c2.k[1] as f32,
        c2.h[0] as f32,
        c2.h[1] as f32,
        c2.h[2] as f32,
        0.0,
        0.0,
    ]
}

/// A window's phase, quality and offset from its sums, as the CPU reference
/// takes them.
fn finish(sums: &[f32], sigma: Real) -> WindowDemod {
    let s = |i: usize| sums[i] as Real;
    let weight = s(0);
    let mean = s(1) / weight;
    let deviation = (s(2) / weight - mean * mean).max(0.0).sqrt();
    let mut out = WindowDemod {
        phase: [0.0; 2],
        quality: [0.0; 2],
        offset: 0.0,
    };
    for c in 0..2 {
        let o = 3 + 12 * c;
        // Σw·(I − mean)·e, and the same weighted by qx and by qy.
        let less_mean = |at: usize| {
            (
                s(o + at) - s(o + at + 2) * mean,
                s(o + at + 1) - s(o + at + 3) * mean,
            )
        };
        let z = less_mean(0);
        let power = z.0 * z.0 + z.1 * z.1;
        out.phase[c] = z.1.atan2(z.0);
        out.quality[c] = if deviation > 0.0 {
            power.sqrt() / (weight * deviation)
        } else {
            0.0
        };
        if power > 0.0 {
            // The carrier's centre of weight, projected on its own phasor.
            let along = |m: (Real, Real)| (m.0 * z.0 + m.1 * z.1) / power;
            let (shift_x, shift_y) = (along(less_mean(4)), along(less_mean(8)));
            out.offset = out.offset.max(shift_x.hypot(shift_y) / sigma);
        } else {
            out.offset = Real::INFINITY;
        }
    }
    out
}

impl LocalDemodulator for GpuBackend {
    type Frame<'a> = GpuFrame;

    fn load<'a>(&'a self, data: &'a [f32], width: usize, height: usize) -> Result<GpuFrame> {
        if data.len() != width * height || data.is_empty() {
            return Err(VernierError::UnsupportedSize(width, height));
        }
        Ok(GpuFrame {
            data: self.input_buffer(data)?,
            width,
            height,
            level: (data.par_iter().map(|&v| v as Real).sum::<Real>() / data.len() as Real) as f32,
        })
    }

    fn demodulate_windows(
        &self,
        frame: &GpuFrame,
        windows: &[DemodWindow],
    ) -> Result<Vec<WindowDemod>> {
        let count = windows.len();
        if count == 0 {
            return Ok(Vec::new());
        }
        let packed: Vec<f32> = windows.iter().flat_map(pack).collect();
        let packed = self.input_buffer(&packed)?;
        let sums = self.work_buffer(count * SUM_FLOATS, true)?;
        let groups_x = count.min(MAX_GROUPS);
        let mut builder = self.new_builder()?;
        self.record(
            &mut builder,
            &self.demod.windows,
            &[&frame.data, &packed, &sums],
            windows_shader::PushConstantData {
                width: frame.width as u32,
                height: frame.height as u32,
                count: count as u32,
                groups_x: groups_x as u32,
            },
            [groups_x as u32, count.div_ceil(groups_x) as u32, 1],
        )?;
        self.submit_one_shot(builder)?;
        let sums = sums.read().map_err(backend_error)?;
        Ok(windows
            .iter()
            .zip(sums.as_chunks::<SUM_FLOATS>().0)
            .map(|(window, sums)| finish(sums, window.sigma))
            .collect())
    }

    fn demodulate_field(
        &self,
        frame: &GpuFrame,
        references: [&[Real]; 2],
        sigma: Real,
    ) -> Result<FieldDemod> {
        let (width, height) = (frame.width, frame.height);
        let n = width * height;
        if references.iter().any(|r| r.len() != n) {
            return Err(VernierError::UnsupportedSize(width, height));
        }
        // Wrapped in f64, so the f32 phase keeps its precision.
        let wrapped: Vec<f32> = references
            .par_iter()
            .flat_map_iter(|r| r.iter().map(|&p| (p - TAU * (p / TAU).round()) as f32))
            .collect();
        let references = self.input_buffer(&wrapped)?;
        let terms = self.work_buffer(FIELD_TERMS * n, false)?;
        let rows = self.work_buffer(FIELD_TERMS * n, false)?;
        let out = self.work_buffer(5 * n, true)?;
        let (w, h, planes) = (width as u32, height as u32, FIELD_TERMS as u32);
        let blurs = [
            (
                &self.demod.field_rows,
                &terms,
                &rows,
                [w.div_ceil(256), h, planes],
            ),
            // Back into `terms`, which the rows no longer need.
            (
                &self.demod.field_columns,
                &rows,
                &terms,
                [h.div_ceil(32), w.div_ceil(32), planes],
            ),
        ];

        let mut builder = self.new_builder()?;
        self.record(
            &mut builder,
            &self.demod.field_terms,
            &[&frame.data, &references, &terms],
            field_terms_shader::PushConstantData {
                width: w,
                height: h,
                level: frame.level,
            },
            [w.div_ceil(8), h.div_ceil(8), 1],
        )?;
        for (pipeline, source, target, groups) in blurs {
            // Both blurs take the same push constants.
            self.record(
                &mut builder,
                pipeline,
                &[source, target],
                field_rows_shader::PushConstantData {
                    width: w,
                    height: h,
                    sigma: sigma as f32,
                    radius: (3.0 * sigma).ceil() as i32,
                },
                groups,
            )?;
        }
        self.record(
            &mut builder,
            &self.demod.field_combine,
            &[&terms, &out],
            field_combine_shader::PushConstantData {
                width: w,
                height: h,
            },
            [w.div_ceil(64), h, 1],
        )?;
        self.submit_one_shot(builder)?;

        let out = out.read().map_err(backend_error)?;
        let deviation = &out[4 * n..5 * n];
        let carrier = |c: usize| {
            let re = &out[2 * c * n..(2 * c + 1) * n];
            let im = &out[(2 * c + 1) * n..(2 * c + 2) * n];
            let phase = re
                .par_iter()
                .zip(im)
                .map(|(&re, &im)| (im as Real).atan2(re as Real))
                .collect();
            let amplitude = re
                .par_iter()
                .zip(im)
                .zip(deviation)
                .map(|((&re, &im), &d)| {
                    if d > 0.0 {
                        (re as Real).hypot(im as Real) / d as Real
                    } else {
                        0.0
                    }
                })
                .collect();
            (phase, amplitude)
        };
        let (phase1, amplitude1) = carrier(0);
        let (phase2, amplitude2) = carrier(1);
        Ok(FieldDemod {
            phase: [phase1, phase2],
            amplitude: [amplitude1, amplitude2],
        })
    }
}

mod windows_shader {
    vulkano_shaders::shader! { ty: "compute", path: "src/shaders/demod_windows.glsl", include: ["."], spirv_version: "1.3" }
}
mod field_terms_shader {
    vulkano_shaders::shader! { ty: "compute", path: "src/shaders/field_terms.glsl", include: ["."], spirv_version: "1.3" }
}
mod field_rows_shader {
    vulkano_shaders::shader! { ty: "compute", path: "src/shaders/field_rows.glsl", include: ["."], spirv_version: "1.3" }
}
mod field_columns_shader {
    vulkano_shaders::shader! { ty: "compute", path: "src/shaders/field_columns.glsl", include: ["."], spirv_version: "1.3" }
}
mod field_combine_shader {
    vulkano_shaders::shader! { ty: "compute", path: "src/shaders/field_combine.glsl", include: ["."], spirv_version: "1.3" }
}
