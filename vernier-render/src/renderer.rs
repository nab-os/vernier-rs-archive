//! Vulkan rasterisation renderer — one quad per period cell, cosine carrier
//! evaluated in the fragment shader. See the crate docs for the projection.

use std::sync::Arc;

use smallvec::smallvec;
use vulkano::VulkanLibrary;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyImageToBufferInfo, RenderPassBeginInfo,
    SubpassBeginInfo, SubpassContents, SubpassEndInfo,
};
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::format::{ClearValue, Format};
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::{
    VertexInputAttributeDescription, VertexInputBindingDescription, VertexInputRate,
    VertexInputState,
};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::sync::{self, GpuFuture};

use vernier_core::GrayImage;

use crate::RenderParams;

// ---------------------------------------------------------------------------
// Vertex / instance data
// ---------------------------------------------------------------------------

/// One corner of the unit quad.  Four of these form each dot quad via the
/// shared index buffer.
#[derive(Clone, Copy, Debug, Default, BufferContents)]
#[repr(C)]
pub(crate) struct VertexData {
    pub local_corner: [f32; 2],
}

/// One entry per dot: the bottom-left corner of its period cell in µm.
#[derive(Clone, Copy, Debug, Default, BufferContents)]
#[repr(C)]
pub(crate) struct InstanceData {
    pub cell_origin: [f32; 2],
}

// ---------------------------------------------------------------------------
// Shader modules  (compiled from GLSL at build time by vulkano-shaders)
// ---------------------------------------------------------------------------

mod vert_shader {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/vert.glsl",
        spirv_version: "1.3"
    }
}

mod frag_shader {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/frag.glsl",
        spirv_version: "1.3"
    }
}

// ---------------------------------------------------------------------------
// PatternRenderer
// ---------------------------------------------------------------------------

/// A GPU rasterisation context for calibrated patterns.
///
/// Holds the Vulkan device, a shared graphics pipeline, and the static
/// vertex/index buffers for a unit quad.  Create once, then call
/// [`render_quads`](PatternRenderer::render_quads) for each frame.
pub struct PatternRenderer {
    device: Arc<Device>,
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    render_pass: Arc<RenderPass>,
    pipeline: Arc<GraphicsPipeline>,
    vertex_buffer: Subbuffer<[VertexData]>,
    index_buffer: Subbuffer<[u32]>,
}

impl PatternRenderer {
    /// Initialises Vulkan and creates the graphics pipeline.
    ///
    /// Selects the best available physical device (discrete > integrated > …)
    /// that supports the required features.
    #[allow(clippy::new_without_default)] // Creating a Vulkan device isn't a default.
    pub fn new() -> Self {
        let library = VulkanLibrary::new().expect("no Vulkan library");

        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                ..Default::default()
            },
        )
        .expect("failed to create Vulkan instance");

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
                    .position(|q| {
                        q.queue_flags
                            .intersects(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                _ => 4,
            })
            .expect("no suitable physical device found");

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
        .expect("failed to create device");

        let queue = queues.next().unwrap();
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        // Render pass: single R32_SFLOAT colour attachment, cleared to 0 on
        // load and transitioned to TRANSFER_SRC_OPTIMAL for readback.
        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: Format::R32_SFLOAT,
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        )
        .unwrap();

        // Static unit-quad: 4 corners covering [0,1]×[0,1].
        let vertices = [
            VertexData {
                local_corner: [0.0, 0.0],
            },
            VertexData {
                local_corner: [1.0, 0.0],
            },
            VertexData {
                local_corner: [0.0, 1.0],
            },
            VertexData {
                local_corner: [1.0, 1.0],
            },
        ];
        let vertex_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices,
        )
        .unwrap();

        // Index buffer: two CCW triangles from the 4 corners.
        let indices: [u32; 6] = [0, 1, 2, 1, 3, 2];
        let index_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            indices,
        )
        .unwrap();

        // Graphics pipeline.
        let pipeline = {
            let vert_ep = vert_shader::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let frag_ep = frag_shader::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();

            let stages = [
                PipelineShaderStageCreateInfo::new(vert_ep.clone()),
                PipelineShaderStageCreateInfo::new(frag_ep),
            ];

            let vertex_input_state = VertexInputState {
                bindings: [
                    (
                        0,
                        VertexInputBindingDescription {
                            stride: std::mem::size_of::<VertexData>() as u32,
                            input_rate: VertexInputRate::Vertex,
                            ..Default::default()
                        },
                    ),
                    (
                        1,
                        VertexInputBindingDescription {
                            stride: std::mem::size_of::<InstanceData>() as u32,
                            input_rate: VertexInputRate::Instance { divisor: 1 },
                            ..Default::default()
                        },
                    ),
                ]
                .into_iter()
                .collect(),
                attributes: [
                    (
                        0,
                        VertexInputAttributeDescription {
                            binding: 0,
                            format: Format::R32G32_SFLOAT,
                            offset: 0,
                            ..Default::default()
                        },
                    ),
                    (
                        1,
                        VertexInputAttributeDescription {
                            binding: 1,
                            format: Format::R32G32_SFLOAT,
                            offset: 0,
                            ..Default::default()
                        },
                    ),
                ]
                .into_iter()
                .collect(),
                ..Default::default()
            };

            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();

            let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

            GraphicsPipeline::new(
                device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState::default()),
                    rasterization_state: Some(RasterizationState {
                        cull_mode: CullMode::None,
                        ..Default::default()
                    }),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        1,
                        ColorBlendAttachmentState::default(),
                    )),
                    dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap()
        };

        Self {
            device,
            queue,
            memory_allocator,
            command_buffer_allocator,
            render_pass,
            pipeline,
            vertex_buffer,
            index_buffer,
        }
    }

    /// Renders `cell_origins` (one entry per period-cell in µm) into a
    /// `width × height` grayscale image and returns it as a [`GrayImage`].
    ///
    /// If `cell_origins` is empty the result is all-zero (no light).
    pub fn render_quads(&self, cell_origins: &[[f32; 2]], params: &RenderParams) -> GrayImage {
        let width = params.width;
        let height = params.height;

        if cell_origins.is_empty() {
            return GrayImage::zeros(width, height);
        }

        // Instance buffer — one entry per dot quad.
        let instance_buffer = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            cell_origins
                .iter()
                .map(|&o| InstanceData { cell_origin: o }),
        )
        .unwrap();

        // Offscreen R32_SFLOAT render target.
        let render_image = Image::new(
            self.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R32_SFLOAT,
                extent: [width as u32, height as u32, 1],
                usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .unwrap();
        let render_image_view = ImageView::new_default(render_image.clone()).unwrap();

        // Readback buffer: one f32 per pixel.
        let readback_buffer = Buffer::new_slice::<f32>(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            (width * height) as u64,
        )
        .unwrap();

        let framebuffer = Framebuffer::new(
            self.render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![render_image_view],
                ..Default::default()
            },
        )
        .unwrap();

        // Push constants.
        let alpha = params.alpha;
        let push_data = vert_shader::PushConstantData {
            cos_alpha: alpha.cos(),
            sin_alpha: alpha.sin(),
            pose_x: params.pose_x_um,
            pose_y: params.pose_y_um,
            pixel_size: params.pixel_size,
            period: params.period_um,
            img_width: width as f32,
            img_height: height as f32,
        };

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: [width as f32, height as f32],
            depth_range: 0.0..=1.0,
        };

        // Record and submit.
        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some(ClearValue::Float([0.0, 0.0, 0.0, 0.0]))],
                    ..RenderPassBeginInfo::framebuffer(framebuffer)
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .unwrap()
            .set_viewport(0, smallvec![viewport])
            .unwrap()
            .bind_pipeline_graphics(self.pipeline.clone())
            .unwrap()
            .bind_vertex_buffers(0, (self.vertex_buffer.clone(), instance_buffer.clone()))
            .unwrap()
            .bind_index_buffer(self.index_buffer.clone())
            .unwrap()
            .push_constants(self.pipeline.layout().clone(), 0, push_data)
            .unwrap();

        unsafe {
            builder
                .draw_indexed(6, cell_origins.len() as u32, 0, 0, 0)
                .unwrap()
        };

        builder
            .end_render_pass(SubpassEndInfo::default())
            .unwrap()
            .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                render_image.clone(),
                readback_buffer.clone(),
            ))
            .unwrap();

        let cb = builder.build().unwrap();

        sync::now(self.device.clone())
            .then_execute(self.queue.clone(), cb)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        // Assemble into GrayImage.
        let pixel_data = readback_buffer.read().unwrap();
        let mut image = GrayImage::zeros(width, height);
        let out = image.as_mut_slice();
        for (i, &v) in pixel_data.iter().enumerate() {
            out[i] = v;
        }
        image
    }
}
