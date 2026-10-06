//! [`GpuBackend`] and [`GpuJob`]: the Vulkano/GLSL implementations of
//! [`ComputeBackend`](vernier_core::ComputeBackend) and [`ComputeJob`](vernier_core::ComputeJob).

use std::sync::Arc;

use vernier_core::buffer::BufferLayout;
use vernier_core::{Complex32, ComputeBackend, ComputeJob, Real, Result, VernierError};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryAutoCommandBuffer,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::sync::GpuFuture;
use vulkano::{VulkanLibrary, sync};

fn is_pow2(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

/// Next power-of-two ≥ 2*n−1 for Bluestein's algorithm (n must be ≥ 1).
fn bluestein_m(n: usize) -> usize {
    let min = 2 * n - 1;
    let mut m = 1usize;
    while m < min {
        m <<= 1;
    }
    m
}

use crate::buffer::GpuBuffer;
use crate::demod::DemodPipelines;

// ---------------------------------------------------------------------------
// Shared compute pipelines
// ---------------------------------------------------------------------------

struct ComputeContext {
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    fft_pipeline: Arc<ComputePipeline>,
    ifft_pipeline: Arc<ComputePipeline>,
    blur_pipeline: Arc<ComputePipeline>,
    extract_phase_pipeline: Arc<ComputePipeline>,
    filter_pipeline: Arc<ComputePipeline>,
    bandpass_pipeline: Arc<ComputePipeline>,
    magnitude_pipeline: Arc<ComputePipeline>,
    argmax_local_pipeline: Arc<ComputePipeline>,
    argmax_global_pipeline: Arc<ComputePipeline>,
    band_angular_filter_pipeline: Arc<ComputePipeline>,
    order_peaks_pipeline: Arc<ComputePipeline>,
    bandpass_peaks_pipeline: Arc<ComputePipeline>,
    spectral_plane_fit_partial_pipeline: Arc<ComputePipeline>,
    spectral_plane_fit_global_pipeline: Arc<ComputePipeline>,
    bluestein_pre_pipeline: Arc<ComputePipeline>,
    bluestein_pointwise_pipeline: Arc<ComputePipeline>,
    bluestein_post_pipeline: Arc<ComputePipeline>,
}

pub(crate) fn build_pipeline(
    device: Arc<Device>,
    shader_module: Arc<vulkano::shader::ShaderModule>,
) -> Arc<ComputePipeline> {
    let cs = shader_module.entry_point("main").unwrap();
    let stage = PipelineShaderStageCreateInfo::new(cs);
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();
    ComputePipeline::new(
        device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .unwrap()
}

// ---------------------------------------------------------------------------
// GpuBackend
// ---------------------------------------------------------------------------

/// GPU compute backend (Vulkano).
pub struct GpuBackend {
    device: Arc<Device>,
    queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub(crate) memory_allocator: Arc<StandardMemoryAllocator>,
    ccx: ComputeContext,
    pub(crate) demod: DemodPipelines,
}

impl GpuBackend {
    pub fn new() -> Self {
        let library = VulkanLibrary::new().unwrap();

        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                ..Default::default()
            },
        )
        .unwrap();

        let device_extensions = DeviceExtensions {
            khr_storage_buffer_storage_class: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .position(|q| q.queue_flags.intersects(QueueFlags::COMPUTE))
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .expect("no suitable physical device found");

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .unwrap();

        let queue = queues.next().unwrap();
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let fft_pipeline =
            build_pipeline(device.clone(), fft_shader::load(device.clone()).unwrap());
        let ifft_pipeline =
            build_pipeline(device.clone(), ifft_shader::load(device.clone()).unwrap());
        let blur_pipeline =
            build_pipeline(device.clone(), blur_shader::load(device.clone()).unwrap());
        let extract_phase_pipeline = build_pipeline(
            device.clone(),
            extract_phase_shader::load(device.clone()).unwrap(),
        );
        let filter_pipeline =
            build_pipeline(device.clone(), filter_shader::load(device.clone()).unwrap());
        let bandpass_pipeline = build_pipeline(
            device.clone(),
            bandpass_shader::load(device.clone()).unwrap(),
        );
        let magnitude_pipeline = build_pipeline(
            device.clone(),
            magnitude_shader::load(device.clone()).unwrap(),
        );
        let argmax_local_pipeline = build_pipeline(
            device.clone(),
            argmax_local_shader::load(device.clone()).unwrap(),
        );
        let argmax_global_pipeline = build_pipeline(
            device.clone(),
            argmax_global_shader::load(device.clone()).unwrap(),
        );
        let band_angular_filter_pipeline = build_pipeline(
            device.clone(),
            band_angular_filter_shader::load(device.clone()).unwrap(),
        );
        let order_peaks_pipeline = build_pipeline(
            device.clone(),
            peak_search_shader::load(device.clone()).unwrap(),
        );
        let bandpass_peaks_pipeline = build_pipeline(
            device.clone(),
            bandpass_peaks_shader::load(device.clone()).unwrap(),
        );
        let spectral_plane_fit_partial_pipeline = build_pipeline(
            device.clone(),
            spectral_plane_fit_partial_shader::load(device.clone()).unwrap(),
        );
        let spectral_plane_fit_global_pipeline = build_pipeline(
            device.clone(),
            spectral_plane_fit_global_shader::load(device.clone()).unwrap(),
        );
        let bluestein_pre_pipeline = build_pipeline(
            device.clone(),
            bluestein_pre_shader::load(device.clone()).unwrap(),
        );
        let bluestein_pointwise_pipeline = build_pipeline(
            device.clone(),
            bluestein_pointwise_shader::load(device.clone()).unwrap(),
        );
        let bluestein_post_pipeline = build_pipeline(
            device.clone(),
            bluestein_post_shader::load(device.clone()).unwrap(),
        );
        let demod = DemodPipelines::new(device.clone());

        Self {
            device,
            queue,
            memory_allocator,
            command_buffer_allocator,
            demod,
            ccx: ComputeContext {
                descriptor_set_allocator,
                fft_pipeline,
                ifft_pipeline,
                blur_pipeline,
                extract_phase_pipeline,
                filter_pipeline,
                bandpass_pipeline,
                magnitude_pipeline,
                argmax_local_pipeline,
                argmax_global_pipeline,
                band_angular_filter_pipeline,
                order_peaks_pipeline,
                bandpass_peaks_pipeline,
                spectral_plane_fit_partial_pipeline,
                spectral_plane_fit_global_pipeline,
                bluestein_pre_pipeline,
                bluestein_pointwise_pipeline,
                bluestein_post_pipeline,
            },
        }
    }

    /// Allocates a DEVICE-local storage buffer of `n` complex elements.
    fn alloc_device_buffer(&self, n: usize) -> Subbuffer<[Complex32]> {
        Buffer::new_slice::<Complex32>(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            n as u64,
        )
        .unwrap()
    }

    pub(crate) fn make_descriptor_set(
        &self,
        pipeline: &Arc<ComputePipeline>,
        writes: impl IntoIterator<Item = WriteDescriptorSet>,
    ) -> Arc<DescriptorSet> {
        let layout = pipeline.layout().set_layouts()[0].clone();
        DescriptorSet::new(
            self.ccx.descriptor_set_allocator.clone(),
            layout,
            writes,
            [],
        )
        .unwrap()
    }

    /// Submits a one-shot command buffer and blocks until complete.
    pub(crate) fn submit_one_shot(
        &self,
        builder: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) -> Result<()> {
        let cb = builder
            .build()
            .map_err(|e| VernierError::Backend(e.to_string()))?;
        sync::now(self.device.clone())
            .then_execute(self.queue.clone(), cb)
            .map_err(|e| VernierError::Backend(e.to_string()))?
            .then_signal_fence_and_flush()
            .map_err(|e| VernierError::Backend(e.to_string()))?
            .wait(None)
            .map_err(|e| VernierError::Backend(e.to_string()))
    }

    pub(crate) fn new_builder(&self) -> Result<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>> {
        AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(|e| VernierError::Backend(e.to_string()))
    }
}

impl Default for GpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBackend for GpuBackend {
    type Buffer2D = GpuBuffer;
    type Job<'a> = GpuJob<'a>;

    fn begin(&self) -> Result<GpuJob<'_>> {
        let builder = self.new_builder()?;
        Ok(GpuJob {
            backend: self,
            builder,
            staging_buffers: Vec::new(),
        })
    }

    fn upload(&self, data: &[Complex32], layout: BufferLayout) -> Result<GpuBuffer> {
        layout.check_upload(data.len())?;

        let host_buffer = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            data.iter().copied(),
        )
        .map_err(|e| VernierError::Backend(e.to_string()))?;

        let n = layout.width * layout.height;
        let device_buffer = self.alloc_device_buffer((n as u64).next_power_of_two() as usize);

        let mut builder = self.new_builder()?;
        builder
            .copy_buffer(CopyBufferInfo::buffers(host_buffer, device_buffer.clone()))
            .map_err(|e| VernierError::Backend(e.to_string()))?;
        self.submit_one_shot(builder)?;

        Ok(GpuBuffer {
            buffer: device_buffer,
            width: layout.width,
            height: layout.height,
        })
    }

    fn download(&self, buffer: &GpuBuffer) -> Result<Vec<Complex32>> {
        let host_buffer = Buffer::new_slice::<Complex32>(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            buffer.size() as u64,
        )
        .map_err(|e| VernierError::Backend(e.to_string()))?;

        let mut builder = self.new_builder()?;
        builder
            .copy_buffer(CopyBufferInfo::buffers(
                buffer.buffer.clone(),
                host_buffer.clone(),
            ))
            .map_err(|e| VernierError::Backend(e.to_string()))?;
        self.submit_one_shot(builder)?;

        Ok(host_buffer
            .read()
            .map_err(|e| VernierError::Backend(e.to_string()))?
            .to_vec())
    }

    fn name(&self) -> &str {
        "gpu-vulkan-fft"
    }
}

// ---------------------------------------------------------------------------
// GpuJob — records commands into an AutoCommandBufferBuilder
// ---------------------------------------------------------------------------

pub struct GpuJob<'a> {
    backend: &'a GpuBackend,
    builder: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    staging_buffers: Vec<Subbuffer<[Complex32]>>,
}

impl GpuJob<'_> {
    fn descriptor_set(
        &self,
        pipeline: &Arc<ComputePipeline>,
        writes: impl IntoIterator<Item = WriteDescriptorSet>,
    ) -> Arc<DescriptorSet> {
        self.backend.make_descriptor_set(pipeline, writes)
    }

    fn alloc_buffer(&self, n: usize) -> Subbuffer<[Complex32]> {
        self.backend.alloc_device_buffer(n)
    }
}

impl GpuJob<'_> {
    /// Bluestein chirp-Z 1D FFT/IFFT of one dimension of `data_buf`.
    ///
    /// AutoCommandBufferBuilder automatically inserts the required memory barriers between
    /// dispatches that share storage buffers, so no explicit barrier calls are needed here.
    ///
    /// `pass=0` transforms along the width (row pass),
    /// `pass=1` transforms along the height (column pass).
    /// `is_inverse=0` → forward DFT, `is_inverse=1` → normalized inverse DFT.
    fn bluestein_pass(
        &mut self,
        data_buf: Subbuffer<[Complex32]>,
        width: usize,
        height: usize,
        pass: u32,
        is_inverse: u32,
    ) -> Result<()> {
        let (n, n_transforms) = if pass == 0 {
            (width, height)
        } else {
            (height, width)
        };
        let m = bluestein_m(n);

        // work_a: n_transforms × M (row pass) or M × n_transforms (col pass)
        let work_a_size = n_transforms * m;
        let work_a = self.backend.alloc_device_buffer(work_a_size);
        // work_b: M elements (the chirp sequence, 1D)
        let work_b = self.backend.alloc_device_buffer(m);

        // --- Step 1: pre-weight data → work_a, compute b_circ → work_b ------
        let pre_total = work_a_size;
        let ds_pre = self.backend.make_descriptor_set(
            &self.backend.ccx.bluestein_pre_pipeline,
            [
                WriteDescriptorSet::buffer(0, data_buf.clone()),
                WriteDescriptorSet::buffer(1, work_a.clone()),
                WriteDescriptorSet::buffer(2, work_b.clone()),
            ],
        );
        self.builder
            .bind_pipeline_compute(self.backend.ccx.bluestein_pre_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.backend.ccx.bluestein_pre_pipeline.layout().clone(),
                0,
                ds_pre,
            )
            .unwrap()
            .push_constants(
                self.backend.ccx.bluestein_pre_pipeline.layout().clone(),
                0,
                bluestein_pre_shader::PushConstantData {
                    N: n as u32,
                    M: m as u32,
                    width: width as u32,
                    height: height as u32,
                    pass,
                    is_inverse,
                },
            )
            .unwrap();
        let pre_groups = pre_total.div_ceil(256);
        unsafe { self.builder.dispatch([pre_groups as u32, 1, 1]) }.unwrap();

        // --- Step 2a: FFT work_a (M-length PoT transforms) -------------------
        // Row pass:  work_a is (height × M) → FFT each row (pass=0, width=M, height=height)
        // Col pass:  work_a is (M × width)  → FFT each column (pass=1, width=width, height=M)
        let (fft_w, fft_h, fft_pass, fft_dispatch) = if pass == 0 {
            (m as u32, height as u32, 0u32, [1u32, height as u32, 1u32])
        } else {
            (width as u32, m as u32, 1u32, [width as u32, 1u32, 1u32])
        };
        let ds_fft_a = self.backend.make_descriptor_set(
            &self.backend.ccx.fft_pipeline,
            [WriteDescriptorSet::buffer(0, work_a.clone())],
        );
        self.builder
            .bind_pipeline_compute(self.backend.ccx.fft_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.backend.ccx.fft_pipeline.layout().clone(),
                0,
                ds_fft_a,
            )
            .unwrap()
            .push_constants(
                self.backend.ccx.fft_pipeline.layout().clone(),
                0,
                fft_shader::PushConstantData {
                    width: fft_w,
                    height: fft_h,
                    pass: fft_pass,
                },
            )
            .unwrap();
        unsafe { self.builder.dispatch(fft_dispatch) }.unwrap();

        // --- Step 2b: FFT work_b (single M-length row) -----------------------
        let ds_fft_b = self.backend.make_descriptor_set(
            &self.backend.ccx.fft_pipeline,
            [WriteDescriptorSet::buffer(0, work_b.clone())],
        );
        self.builder
            .bind_pipeline_compute(self.backend.ccx.fft_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.backend.ccx.fft_pipeline.layout().clone(),
                0,
                ds_fft_b,
            )
            .unwrap()
            .push_constants(
                self.backend.ccx.fft_pipeline.layout().clone(),
                0,
                fft_shader::PushConstantData {
                    width: m as u32,
                    height: 1,
                    pass: 0,
                },
            )
            .unwrap();
        unsafe { self.builder.dispatch([1, 1, 1]) }.unwrap();

        // --- Step 3: pointwise multiply work_a *= work_b ---------------------
        // pass=0: work_b_idx = i % M    (period = M)
        // pass=1: work_b_idx = i / width (period = width)
        let pw_period = if pass == 0 { m as u32 } else { width as u32 };
        let ds_pw = self.backend.make_descriptor_set(
            &self.backend.ccx.bluestein_pointwise_pipeline,
            [
                WriteDescriptorSet::buffer(0, work_a.clone()),
                WriteDescriptorSet::buffer(1, work_b.clone()),
            ],
        );
        self.builder
            .bind_pipeline_compute(self.backend.ccx.bluestein_pointwise_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.backend
                    .ccx
                    .bluestein_pointwise_pipeline
                    .layout()
                    .clone(),
                0,
                ds_pw,
            )
            .unwrap()
            .push_constants(
                self.backend
                    .ccx
                    .bluestein_pointwise_pipeline
                    .layout()
                    .clone(),
                0,
                bluestein_pointwise_shader::PushConstantData {
                    n_elements: work_a_size as u32,
                    period: pw_period,
                    pass,
                },
            )
            .unwrap();
        let pw_groups = work_a_size.div_ceil(256);
        unsafe { self.builder.dispatch([pw_groups as u32, 1, 1]) }.unwrap();

        // --- Step 4: IFFT work_a (M-length PoT) ------------------------------
        let (ifft_w, ifft_h, ifft_pass, ifft_dispatch) = (fft_w, fft_h, fft_pass, fft_dispatch);
        let ds_ifft_a = self.backend.make_descriptor_set(
            &self.backend.ccx.ifft_pipeline,
            [WriteDescriptorSet::buffer(0, work_a.clone())],
        );
        self.builder
            .bind_pipeline_compute(self.backend.ccx.ifft_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.backend.ccx.ifft_pipeline.layout().clone(),
                0,
                ds_ifft_a,
            )
            .unwrap()
            .push_constants(
                self.backend.ccx.ifft_pipeline.layout().clone(),
                0,
                ifft_shader::PushConstantData {
                    width: ifft_w,
                    height: ifft_h,
                    pass: ifft_pass,
                },
            )
            .unwrap();
        unsafe { self.builder.dispatch(ifft_dispatch) }.unwrap();

        // --- Step 5: post-weight and extract N elements back into data_buf ---
        let ds_post = self.backend.make_descriptor_set(
            &self.backend.ccx.bluestein_post_pipeline,
            [
                WriteDescriptorSet::buffer(0, data_buf.clone()),
                WriteDescriptorSet::buffer(1, work_a.clone()),
            ],
        );
        self.builder
            .bind_pipeline_compute(self.backend.ccx.bluestein_post_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.backend.ccx.bluestein_post_pipeline.layout().clone(),
                0,
                ds_post,
            )
            .unwrap()
            .push_constants(
                self.backend.ccx.bluestein_post_pipeline.layout().clone(),
                0,
                bluestein_post_shader::PushConstantData {
                    N: n as u32,
                    M: m as u32,
                    width: width as u32,
                    height: height as u32,
                    pass,
                    is_inverse,
                },
            )
            .unwrap();
        let post_groups = (width * height).div_ceil(256);
        unsafe { self.builder.dispatch([post_groups as u32, 1, 1]) }.unwrap();

        Ok(())
    }
}

impl ComputeJob for GpuJob<'_> {
    type Buffer2D = GpuBuffer;

    fn copy_buffer(&mut self, src: &GpuBuffer) -> Result<GpuBuffer> {
        let destination = self.alloc_buffer(src.size());
        self.builder
            .copy_buffer(CopyBufferInfo::buffers(
                src.buffer.clone(),
                destination.clone(),
            ))
            .map_err(|e| VernierError::Backend(e.to_string()))?;
        Ok(GpuBuffer {
            buffer: destination,
            width: src.width,
            height: src.height,
        })
    }

    fn fft2d(&mut self, buf: &mut GpuBuffer) -> Result<()> {
        let (width, height) = (buf.width, buf.height);
        if width > 2048 || height > 2048 {
            return Err(VernierError::UnsupportedSize(width, height));
        }

        // Row pass
        if is_pow2(width) {
            let ds = self.descriptor_set(
                &self.backend.ccx.fft_pipeline,
                [WriteDescriptorSet::buffer(0, buf.buffer.clone())],
            );
            self.builder
                .bind_pipeline_compute(self.backend.ccx.fft_pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.backend.ccx.fft_pipeline.layout().clone(),
                    0,
                    ds,
                )
                .unwrap()
                .push_constants(
                    self.backend.ccx.fft_pipeline.layout().clone(),
                    0,
                    fft_shader::PushConstantData {
                        width: width as u32,
                        height: height as u32,
                        pass: 0,
                    },
                )
                .unwrap();
            unsafe { self.builder.dispatch([1, height as u32, 1]) }.unwrap();
        } else {
            self.bluestein_pass(buf.buffer.clone(), width, height, 0, 0)?;
        }

        // Column pass
        if is_pow2(height) {
            let ds = self.descriptor_set(
                &self.backend.ccx.fft_pipeline,
                [WriteDescriptorSet::buffer(0, buf.buffer.clone())],
            );
            self.builder
                .bind_pipeline_compute(self.backend.ccx.fft_pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.backend.ccx.fft_pipeline.layout().clone(),
                    0,
                    ds,
                )
                .unwrap()
                .push_constants(
                    self.backend.ccx.fft_pipeline.layout().clone(),
                    0,
                    fft_shader::PushConstantData {
                        width: width as u32,
                        height: height as u32,
                        pass: 1,
                    },
                )
                .unwrap();
            unsafe { self.builder.dispatch([width as u32, 1, 1]) }.unwrap();
        } else {
            self.bluestein_pass(buf.buffer.clone(), width, height, 1, 0)?;
        }

        Ok(())
    }

    fn ifft2d(&mut self, buf: &mut GpuBuffer) -> Result<()> {
        let (width, height) = (buf.width, buf.height);
        if width > 2048 || height > 2048 {
            return Err(VernierError::UnsupportedSize(width, height));
        }

        // Row pass
        if is_pow2(width) {
            let ds = self.descriptor_set(
                &self.backend.ccx.ifft_pipeline,
                [WriteDescriptorSet::buffer(0, buf.buffer.clone())],
            );
            self.builder
                .bind_pipeline_compute(self.backend.ccx.ifft_pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.backend.ccx.ifft_pipeline.layout().clone(),
                    0,
                    ds,
                )
                .unwrap()
                .push_constants(
                    self.backend.ccx.ifft_pipeline.layout().clone(),
                    0,
                    ifft_shader::PushConstantData {
                        width: width as u32,
                        height: height as u32,
                        pass: 0,
                    },
                )
                .unwrap();
            unsafe { self.builder.dispatch([1, height as u32, 1]) }.unwrap();
        } else {
            self.bluestein_pass(buf.buffer.clone(), width, height, 0, 1)?;
        }

        // Column pass
        if is_pow2(height) {
            let ds = self.descriptor_set(
                &self.backend.ccx.ifft_pipeline,
                [WriteDescriptorSet::buffer(0, buf.buffer.clone())],
            );
            self.builder
                .bind_pipeline_compute(self.backend.ccx.ifft_pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.backend.ccx.ifft_pipeline.layout().clone(),
                    0,
                    ds,
                )
                .unwrap()
                .push_constants(
                    self.backend.ccx.ifft_pipeline.layout().clone(),
                    0,
                    ifft_shader::PushConstantData {
                        width: width as u32,
                        height: height as u32,
                        pass: 1,
                    },
                )
                .unwrap();
            unsafe { self.builder.dispatch([width as u32, 1, 1]) }.unwrap();
        } else {
            self.bluestein_pass(buf.buffer.clone(), width, height, 1, 1)?;
        }

        Ok(())
    }

    fn extract_phase(&mut self, buf: &GpuBuffer) -> Result<GpuBuffer> {
        let output = self.alloc_buffer(buf.size());
        let (width, height) = (buf.width as u32, buf.height as u32);
        let descriptor_set = self.descriptor_set(
            &self.backend.ccx.extract_phase_pipeline,
            [
                WriteDescriptorSet::buffer(0, buf.buffer.clone()),
                WriteDescriptorSet::buffer(1, output.clone()),
            ],
        );
        self.builder
            .bind_pipeline_compute(self.backend.ccx.extract_phase_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.backend.ccx.extract_phase_pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .unwrap()
            .push_constants(
                self.backend.ccx.extract_phase_pipeline.layout().clone(),
                0,
                extract_phase_shader::PushConstantData { width, height },
            )
            .unwrap();
        unsafe {
            self.builder
                .dispatch([(width + 7) / 8, (height + 7) / 8, 1])
        }
        .unwrap();
        Ok(GpuBuffer {
            buffer: output,
            width: buf.width,
            height: buf.height,
        })
    }

    fn filter(
        &mut self,
        buf: &mut GpuBuffer,
        min_frequency: usize,
        max_frequency: usize,
    ) -> Result<()> {
        let (width, height) = (buf.width as u32, buf.height as u32);
        let descriptor_set = self.descriptor_set(
            &self.backend.ccx.filter_pipeline,
            [WriteDescriptorSet::buffer(0, buf.buffer.clone())],
        );
        self.builder
            .bind_pipeline_compute(self.backend.ccx.filter_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.backend.ccx.filter_pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .unwrap()
            .push_constants(
                self.backend.ccx.filter_pipeline.layout().clone(),
                0,
                filter_shader::PushConstantData {
                    width,
                    height,
                    min_frequency: min_frequency as u32,
                    max_frequency: max_frequency as u32,
                },
            )
            .unwrap();
        unsafe {
            self.builder
                .dispatch([(width + 7) / 8, (height + 7) / 8, 1])
        }
        .unwrap();
        Ok(())
    }

    fn gaussian_blur_2d(&mut self, buf: &mut GpuBuffer, sigma: Real) -> Result<()> {
        let temp = self.alloc_buffer(buf.size());
        let (width, height) = (buf.width as u32, buf.height as u32);

        let descriptor_set_horizontal = self.descriptor_set(
            &self.backend.ccx.blur_pipeline,
            [
                WriteDescriptorSet::buffer(0, buf.buffer.clone()),
                WriteDescriptorSet::buffer(1, temp.clone()),
            ],
        );
        self.builder
            .bind_pipeline_compute(self.backend.ccx.blur_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.backend.ccx.blur_pipeline.layout().clone(),
                0,
                descriptor_set_horizontal,
            )
            .unwrap()
            .push_constants(
                self.backend.ccx.blur_pipeline.layout().clone(),
                0,
                blur_shader::PushConstantData {
                    width,
                    height,
                    sigma: sigma as f32,
                    pass: 0,
                },
            )
            .unwrap();
        unsafe {
            self.builder
                .dispatch([(width + 7) / 8, (height + 7) / 8, 1])
        }
        .unwrap();

        let descriptor_set_vertical = self.descriptor_set(
            &self.backend.ccx.blur_pipeline,
            [
                WriteDescriptorSet::buffer(0, temp.clone()),
                WriteDescriptorSet::buffer(1, buf.buffer.clone()),
            ],
        );
        self.builder
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.backend.ccx.blur_pipeline.layout().clone(),
                0,
                descriptor_set_vertical,
            )
            .unwrap()
            .push_constants(
                self.backend.ccx.blur_pipeline.layout().clone(),
                0,
                blur_shader::PushConstantData {
                    width,
                    height,
                    sigma: sigma as f32,
                    pass: 1,
                },
            )
            .unwrap();
        unsafe {
            self.builder
                .dispatch([(width + 7) / 8, (height + 7) / 8, 1])
        }
        .unwrap();
        Ok(())
    }

    fn bandpass_filter(
        &mut self,
        buf: &mut GpuBuffer,
        cx: usize,
        cy: usize,
        sigma: Real,
    ) -> Result<()> {
        let (width, height) = (buf.width as u32, buf.height as u32);
        let descriptor_set = self.descriptor_set(
            &self.backend.ccx.bandpass_pipeline,
            [WriteDescriptorSet::buffer(0, buf.buffer.clone())],
        );
        self.builder
            .bind_pipeline_compute(self.backend.ccx.bandpass_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.backend.ccx.bandpass_pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .unwrap()
            .push_constants(
                self.backend.ccx.bandpass_pipeline.layout().clone(),
                0,
                bandpass_shader::PushConstantData {
                    width,
                    height,
                    cx: cx as u32,
                    cy: cy as u32,
                    sigma: sigma as f32,
                },
            )
            .unwrap();
        unsafe {
            self.builder
                .dispatch([(width + 7) / 8, (height + 7) / 8, 1])
        }
        .unwrap();
        Ok(())
    }

    fn bandpass_from_peaks(
        &mut self,
        buf: &mut GpuBuffer,
        peaks: &GpuBuffer,
        direction: u32,
        sigma: Real,
    ) -> Result<()> {
        let (width, height) = (buf.width as u32, buf.height as u32);
        let descriptor_set = self.descriptor_set(
            &self.backend.ccx.bandpass_peaks_pipeline,
            [
                WriteDescriptorSet::buffer(0, buf.buffer.clone()),
                WriteDescriptorSet::buffer(1, peaks.buffer.clone()),
            ],
        );
        self.builder
            .bind_pipeline_compute(self.backend.ccx.bandpass_peaks_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.backend.ccx.bandpass_peaks_pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .unwrap()
            .push_constants(
                self.backend.ccx.bandpass_peaks_pipeline.layout().clone(),
                0,
                bandpass_peaks_shader::PushConstantData {
                    width,
                    height,
                    direction,
                    sigma: sigma as f32,
                },
            )
            .unwrap();
        unsafe {
            self.builder
                .dispatch([(width + 7) / 8, (height + 7) / 8, 1])
        }
        .unwrap();
        Ok(())
    }

    fn peak_search(
        &mut self,
        buffer: &mut GpuBuffer,
        min_frequency: usize,
        max_frequency: usize,
        smoothing_sigma: Real,
        sigma: Real,
    ) -> Result<Option<GpuBuffer>> {
        let (width, height) = (buffer.width, buffer.height);
        let n = width * height;
        let n_groups = (n + 255) / 256;

        // 1. Deep copy input → working magnitude buffer
        let magnitude_raw = self.alloc_buffer(n);
        self.builder
            .copy_buffer(CopyBufferInfo::buffers(
                buffer.buffer.clone(),
                magnitude_raw.clone(),
            ))
            .map_err(|e| VernierError::Backend(e.to_string()))?;
        let mut magnitude = GpuBuffer {
            buffer: magnitude_raw,
            width,
            height,
        };

        // 2. magnitude
        let descriptor_set_magnitude = self.descriptor_set(
            &self.backend.ccx.magnitude_pipeline,
            [WriteDescriptorSet::buffer(0, magnitude.buffer.clone())],
        );
        self.builder
            .bind_pipeline_compute(self.backend.ccx.magnitude_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.backend.ccx.magnitude_pipeline.layout().clone(),
                0,
                descriptor_set_magnitude,
            )
            .unwrap()
            .push_constants(
                self.backend.ccx.magnitude_pipeline.layout().clone(),
                0,
                magnitude_shader::PushConstantData { n: n as u32 },
            )
            .unwrap();
        unsafe { self.builder.dispatch([(n as u32 + 255) / 256, 1, 1]) }.unwrap();

        // 3. annulus mask
        self.filter(&mut magnitude, min_frequency, max_frequency)?;

        // 4. blur
        if smoothing_sigma > 0.0 {
            self.gaussian_blur_2d(&mut magnitude, smoothing_sigma)?;
        }

        // 5. argmax local → partials1 → argmax global → peak1_buffer
        let partials1 = self.alloc_buffer(n_groups);
        let descriptor_set_argmax_local1 = self.descriptor_set(
            &self.backend.ccx.argmax_local_pipeline,
            [
                WriteDescriptorSet::buffer(0, magnitude.buffer.clone()),
                WriteDescriptorSet::buffer(1, partials1.clone()),
            ],
        );
        self.builder
            .bind_pipeline_compute(self.backend.ccx.argmax_local_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.backend.ccx.argmax_local_pipeline.layout().clone(),
                0,
                descriptor_set_argmax_local1,
            )
            .unwrap()
            .push_constants(
                self.backend.ccx.argmax_local_pipeline.layout().clone(),
                0,
                argmax_local_shader::PushConstantData {
                    width: width as u32,
                    height: height as u32,
                    n: n as u32,
                },
            )
            .unwrap();
        unsafe { self.builder.dispatch([n_groups as u32, 1, 1]) }.unwrap();

        let peak1_buffer = self.alloc_buffer(1);
        let descriptor_set_argmax_global1 = self.descriptor_set(
            &self.backend.ccx.argmax_global_pipeline,
            [
                WriteDescriptorSet::buffer(0, partials1.clone()),
                WriteDescriptorSet::buffer(1, peak1_buffer.clone()),
            ],
        );
        self.builder
            .bind_pipeline_compute(self.backend.ccx.argmax_global_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.backend.ccx.argmax_global_pipeline.layout().clone(),
                0,
                descriptor_set_argmax_global1,
            )
            .unwrap()
            .push_constants(
                self.backend.ccx.argmax_global_pipeline.layout().clone(),
                0,
                argmax_global_shader::PushConstantData {
                    n_groups: n_groups as u32,
                    width: width as u32,
                },
            )
            .unwrap();
        unsafe { self.builder.dispatch([1, 1, 1]) }.unwrap();

        // 6. band + angular filter
        let descriptor_set_band_angular = self.descriptor_set(
            &self.backend.ccx.band_angular_filter_pipeline,
            [
                WriteDescriptorSet::buffer(0, magnitude.buffer.clone()),
                WriteDescriptorSet::buffer(1, peak1_buffer.clone()),
            ],
        );
        self.builder
            .bind_pipeline_compute(self.backend.ccx.band_angular_filter_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.backend
                    .ccx
                    .band_angular_filter_pipeline
                    .layout()
                    .clone(),
                0,
                descriptor_set_band_angular,
            )
            .unwrap()
            .push_constants(
                self.backend
                    .ccx
                    .band_angular_filter_pipeline
                    .layout()
                    .clone(),
                0,
                band_angular_filter_shader::PushConstantData {
                    width: width as u32,
                    height: height as u32,
                    sigma: sigma as f32,
                },
            )
            .unwrap();
        unsafe {
            self.builder
                .dispatch([(width as u32 + 7) / 8, (height as u32 + 7) / 8, 1])
        }
        .unwrap();

        // 7. second argmax → peak2_buffer
        let partials2 = self.alloc_buffer(n_groups);
        let descriptor_set_argmax_local2 = self.descriptor_set(
            &self.backend.ccx.argmax_local_pipeline,
            [
                WriteDescriptorSet::buffer(0, magnitude.buffer.clone()),
                WriteDescriptorSet::buffer(1, partials2.clone()),
            ],
        );
        self.builder
            .bind_pipeline_compute(self.backend.ccx.argmax_local_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.backend.ccx.argmax_local_pipeline.layout().clone(),
                0,
                descriptor_set_argmax_local2,
            )
            .unwrap()
            .push_constants(
                self.backend.ccx.argmax_local_pipeline.layout().clone(),
                0,
                argmax_local_shader::PushConstantData {
                    width: width as u32,
                    height: height as u32,
                    n: n as u32,
                },
            )
            .unwrap();
        unsafe { self.builder.dispatch([n_groups as u32, 1, 1]) }.unwrap();

        let peak2_buffer = self.alloc_buffer(1);
        let descriptor_set_argmax_global2 = self.descriptor_set(
            &self.backend.ccx.argmax_global_pipeline,
            [
                WriteDescriptorSet::buffer(0, partials2.clone()),
                WriteDescriptorSet::buffer(1, peak2_buffer.clone()),
            ],
        );
        self.builder
            .bind_pipeline_compute(self.backend.ccx.argmax_global_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.backend.ccx.argmax_global_pipeline.layout().clone(),
                0,
                descriptor_set_argmax_global2,
            )
            .unwrap()
            .push_constants(
                self.backend.ccx.argmax_global_pipeline.layout().clone(),
                0,
                argmax_global_shader::PushConstantData {
                    n_groups: n_groups as u32,
                    width: width as u32,
                },
            )
            .unwrap();
        unsafe { self.builder.dispatch([1, 1, 1]) }.unwrap();

        // 8. order peaks → 4-element result buffer
        let result_buffer = self.alloc_buffer(4);
        let descriptor_set_order = self.descriptor_set(
            &self.backend.ccx.order_peaks_pipeline,
            [
                WriteDescriptorSet::buffer(0, peak1_buffer.clone()),
                WriteDescriptorSet::buffer(1, peak2_buffer.clone()),
                WriteDescriptorSet::buffer(2, result_buffer.clone()),
            ],
        );
        self.builder
            .bind_pipeline_compute(self.backend.ccx.order_peaks_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.backend.ccx.order_peaks_pipeline.layout().clone(),
                0,
                descriptor_set_order,
            )
            .unwrap()
            .push_constants(
                self.backend.ccx.order_peaks_pipeline.layout().clone(),
                0,
                peak_search_shader::PushConstantData {
                    width: width as u32,
                    height: height as u32,
                },
            )
            .unwrap();
        unsafe { self.builder.dispatch([1, 1, 1]) }.unwrap();

        Ok(Some(GpuBuffer {
            buffer: result_buffer,
            width: 2,
            height: 2,
        }))
    }

    fn upload(&mut self, data: &[Complex32], layout: BufferLayout) -> Result<GpuBuffer> {
        layout.check_upload(data.len())?;
        let n = layout.width * layout.height;
        let staging = Buffer::from_iter(
            self.backend.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            data.iter().copied(),
        )
        .map_err(|e| VernierError::Backend(e.to_string()))?;
        let device_buffer = self.alloc_buffer(n.next_power_of_two());
        self.builder
            .copy_buffer(CopyBufferInfo::buffers(
                staging.clone(),
                device_buffer.clone(),
            ))
            .map_err(|e| VernierError::Backend(e.to_string()))?;
        self.staging_buffers.push(staging);
        Ok(GpuBuffer {
            buffer: device_buffer,
            width: layout.width,
            height: layout.height,
        })
    }

    fn spectral_plane_fit_two(
        &mut self,
        spectrum: &GpuBuffer,
        peaks: &GpuBuffer,
        sigma: Real,
    ) -> Result<GpuBuffer> {
        let (width, height) = (spectrum.width, spectrum.height);
        let n = width * height;
        let n_groups = (n + 63) / 64;

        let partials = self.alloc_buffer(n_groups * 5); // 5 Complex32 = 10 floats per workgroup
        let descriptor_set_partial = self.descriptor_set(
            &self.backend.ccx.spectral_plane_fit_partial_pipeline,
            [
                WriteDescriptorSet::buffer(0, spectrum.buffer.clone()),
                WriteDescriptorSet::buffer(1, peaks.buffer.clone()),
                WriteDescriptorSet::buffer(2, partials.clone()),
            ],
        );
        self.builder
            .bind_pipeline_compute(self.backend.ccx.spectral_plane_fit_partial_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.backend
                    .ccx
                    .spectral_plane_fit_partial_pipeline
                    .layout()
                    .clone(),
                0,
                descriptor_set_partial,
            )
            .unwrap()
            .push_constants(
                self.backend
                    .ccx
                    .spectral_plane_fit_partial_pipeline
                    .layout()
                    .clone(),
                0,
                spectral_plane_fit_partial_shader::PushConstantData {
                    W: width as u32,
                    H: height as u32,
                    N: n as u32,
                    sigma: sigma as f32,
                },
            )
            .unwrap();
        unsafe { self.builder.dispatch([n_groups as u32, 1, 1]) }.unwrap();

        let result = self.alloc_buffer(6);
        let descriptor_set_global = self.descriptor_set(
            &self.backend.ccx.spectral_plane_fit_global_pipeline,
            [
                WriteDescriptorSet::buffer(0, partials.clone()),
                WriteDescriptorSet::buffer(1, result.clone()),
            ],
        );
        self.builder
            .bind_pipeline_compute(self.backend.ccx.spectral_plane_fit_global_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.backend
                    .ccx
                    .spectral_plane_fit_global_pipeline
                    .layout()
                    .clone(),
                0,
                descriptor_set_global,
            )
            .unwrap()
            .push_constants(
                self.backend
                    .ccx
                    .spectral_plane_fit_global_pipeline
                    .layout()
                    .clone(),
                0,
                spectral_plane_fit_global_shader::PushConstantData {
                    n_groups: n_groups as u32,
                    W: width as f32,
                    H: height as f32,
                },
            )
            .unwrap();
        unsafe { self.builder.dispatch([1, 1, 1]) }.unwrap();

        Ok(GpuBuffer {
            buffer: result,
            width: 6,
            height: 1,
        })
    }

    fn submit(self) -> Result<()> {
        self.backend.submit_one_shot(self.builder)
    }
}

// ---------------------------------------------------------------------------
// Shader modules
// ---------------------------------------------------------------------------

mod fft_shader {
    vulkano_shaders::shader! { ty: "compute", path: "src/shaders/fft.glsl", include: ["."], spirv_version: "1.3" }
}
mod ifft_shader {
    vulkano_shaders::shader! { ty: "compute", path: "src/shaders/ifft.glsl", include: ["."], spirv_version: "1.3" }
}
mod blur_shader {
    vulkano_shaders::shader! { ty: "compute", path: "src/shaders/blur.glsl", include: ["."], spirv_version: "1.3" }
}
mod extract_phase_shader {
    vulkano_shaders::shader! { ty: "compute", path: "src/shaders/extract_phase.glsl", include: ["."], spirv_version: "1.3" }
}
mod filter_shader {
    vulkano_shaders::shader! { ty: "compute", path: "src/shaders/filter.glsl", include: ["."], spirv_version: "1.3" }
}
mod peak_search_shader {
    vulkano_shaders::shader! { ty: "compute", path: "src/shaders/peak_search.glsl", include: ["."], spirv_version: "1.3" }
}
mod bandpass_shader {
    vulkano_shaders::shader! { ty: "compute", path: "src/shaders/bandpass.glsl", include: ["."], spirv_version: "1.3" }
}
mod magnitude_shader {
    vulkano_shaders::shader! { ty: "compute", path: "src/shaders/magnitude.glsl", include: ["."], spirv_version: "1.3" }
}
mod argmax_local_shader {
    vulkano_shaders::shader! { ty: "compute", path: "src/shaders/argmax_local.glsl", include: ["."], spirv_version: "1.3" }
}
mod argmax_global_shader {
    vulkano_shaders::shader! { ty: "compute", path: "src/shaders/argmax_global.glsl", include: ["."], spirv_version: "1.3" }
}
mod band_angular_filter_shader {
    vulkano_shaders::shader! { ty: "compute", path: "src/shaders/band_angular_filter.glsl", include: ["."], spirv_version: "1.3" }
}
mod bandpass_peaks_shader {
    vulkano_shaders::shader! { ty: "compute", path: "src/shaders/bandpass_peaks.glsl", include: ["."], spirv_version: "1.3" }
}
mod spectral_plane_fit_partial_shader {
    vulkano_shaders::shader! { ty: "compute", path: "src/shaders/spectral_plane_fit_partial.glsl", include: ["."], spirv_version: "1.3" }
}
mod spectral_plane_fit_global_shader {
    vulkano_shaders::shader! { ty: "compute", path: "src/shaders/spectral_plane_fit_global.glsl", include: ["."], spirv_version: "1.3" }
}
mod bluestein_pre_shader {
    vulkano_shaders::shader! { ty: "compute", path: "src/shaders/bluestein_pre.glsl", include: ["."], spirv_version: "1.3" }
}
mod bluestein_pointwise_shader {
    vulkano_shaders::shader! { ty: "compute", path: "src/shaders/bluestein_pointwise.glsl", include: ["."], spirv_version: "1.3" }
}
mod bluestein_post_shader {
    vulkano_shaders::shader! { ty: "compute", path: "src/shaders/bluestein_post.glsl", include: ["."], spirv_version: "1.3" }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use vernier_core::buffer::BufferLayout;

    fn checkerboard(w: usize, h: usize) -> (Vec<Complex32>, BufferLayout) {
        let layout = BufferLayout::packed(w, h);
        let mut v = Vec::with_capacity(layout.len());
        for r in 0..h {
            for c in 0..w {
                let val = if (r + c) % 2 == 0 { 1.0 } else { -1.0 };
                v.push(Complex32::new(val, 0.0));
            }
        }
        (v, layout)
    }

    #[test]
    fn fft_then_ifft_is_identity() {
        let backend = GpuBackend::new();
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
    fn non_square_round_trips() {
        let backend = GpuBackend::new();
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

    #[test]
    fn non_power_of_two_height_round_trips() {
        // width=8 is PoT (uses FFT), height=6 is not (uses direct DFT).
        let backend = GpuBackend::new();
        let (data, layout) = checkerboard(8, 6);
        let mut buf = backend.upload(&data, layout).unwrap();
        let mut job = backend.begin().unwrap();
        job.fft2d(&mut buf).unwrap();
        job.ifft2d(&mut buf).unwrap();
        job.submit().unwrap();
        let out = backend.download(&buf).unwrap();
        for (orig, got) in data.iter().zip(out.iter()) {
            assert_abs_diff_eq!(orig.re, got.re, epsilon = 1e-3);
            assert_abs_diff_eq!(orig.im, got.im, epsilon = 1e-3);
        }
    }

    #[test]
    fn bluestein_matches_cpu_at_video_height() {
        // 720 rows go through Bluestein, whose chirp angle πn²/N reaches
        // ~2000 rad: it must be reduced before f32 sin/cos see it.
        let (w, h) = (4, 720);
        let layout = BufferLayout::packed(w, h);
        let data: Vec<Complex32> = (0..w * h)
            .map(|i| Complex32::new(((i * 7919) % 1000) as f32 / 1000.0 - 0.5, 0.0))
            .collect();

        let gpu = GpuBackend::new();
        let mut buf = gpu.upload(&data, layout).unwrap();
        let mut job = gpu.begin().unwrap();
        job.fft2d(&mut buf).unwrap();
        job.submit().unwrap();
        let got = gpu.download(&buf).unwrap();

        let cpu = vernier_cpu::CpuBackend::new();
        let mut cpu_buf = cpu.upload(&data, layout).unwrap();
        let mut cpu_job = cpu.begin().unwrap();
        cpu_job.fft2d(&mut cpu_buf).unwrap();
        cpu_job.submit().unwrap();
        let want = cpu.download(&cpu_buf).unwrap();

        let scale = want.iter().map(|z| z.re.hypot(z.im)).fold(0.0f32, f32::max);
        let worst = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g.re - w.re).hypot(g.im - w.im))
            .fold(0.0f32, f32::max);
        assert!(
            worst / scale < 1e-5,
            "worst error {worst} against a peak of {scale}"
        );
    }
}
