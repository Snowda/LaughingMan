//! The reflection-driven graphics pipeline: descriptor-set layout + pipeline layout from
//! `spirv-reader`, and a dynamic-rendering pipeline whose color attachment format matches the
//! swapchain. Adapted from Aspire vk-run's `graphics.rs`, targeting the swapchain format.
#![allow(unsafe_code)]
#![allow(clippy::as_conversions)]

use std::ffi::CString;

use anyhow::{Context as _, anyhow};
use ash::vk;
use spirv_reader::Stage;

/// The pipeline, its layout, and the set-0 descriptor-set layout it binds against. Destroys all
/// three (pipeline, shader module, layouts) on drop.
pub struct GraphicsPipeline {
    device: ash::Device,
    pipeline: vk::Pipeline,
    shader_module: vk::ShaderModule,
    pipeline_layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
}

impl GraphicsPipeline {
    /// Builds the pipeline for the vertex+fragment module in `spirv`, rendering into a single color
    /// attachment of `color_format` (the swapchain format).
    pub fn new(
        device: &ash::Device,
        spirv: &[u8],
        color_format: vk::Format,
    ) -> anyhow::Result<Self> {
        let module = spirv_reader::read(spirv).map_err(|e| anyhow!("reading SPIR-V: {e}"))?;
        let vertex = entry_name(&module, Stage::Vertex).ok_or_else(|| anyhow!("no vertex entry"))?;
        let fragment =
            entry_name(&module, Stage::Fragment).ok_or_else(|| anyhow!("no fragment entry"))?;

        // Descriptor-set layout for set 0 straight from reflection (texture + sampler bindings).
        let bindings = module.vk_descriptor_set_layout_bindings(0);
        let set_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: `set_info` references live locals; the layout is freed in `Drop`.
        let set_layout = unsafe { device.create_descriptor_set_layout(&set_info, None) }
            .context("descriptor set layout")?;

        let set_layouts = [set_layout];
        // Push-constant ranges from reflection (the compositor's letterbox `View`).
        let push_ranges = module.vk_push_constant_ranges();
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_ranges);
        // SAFETY: live locals; freed in `Drop`.
        let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
            .context("pipeline layout")?;

        let words = spirv_words(spirv);
        let module_info = vk::ShaderModuleCreateInfo::default().code(&words);
        // SAFETY: live locals; freed in `Drop`.
        let shader_module = unsafe { device.create_shader_module(&module_info, None) }
            .context("shader module")?;

        let pipeline = match build(device, &vertex, &fragment, shader_module, pipeline_layout, color_format) {
            Ok(pipeline) => pipeline,
            Err(e) => {
                // SAFETY: cleanup on the failure path; every handle is live and unused.
                unsafe {
                    device.destroy_shader_module(shader_module, None);
                    device.destroy_pipeline_layout(pipeline_layout, None);
                    device.destroy_descriptor_set_layout(set_layout, None);
                }
                return Err(e);
            }
        };

        Ok(Self {
            device: device.clone(),
            pipeline,
            shader_module,
            pipeline_layout,
            set_layout,
        })
    }

    pub fn raw(&self) -> vk::Pipeline {
        self.pipeline
    }

    pub fn layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }

    pub fn set_layout(&self) -> vk::DescriptorSetLayout {
        self.set_layout
    }
}

impl Drop for GraphicsPipeline {
    fn drop(&mut self) {
        // SAFETY: owner drains the device before dropping; all handles live.
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_shader_module(self.shader_module, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

fn entry_name(module: &spirv_reader::Module, stage: Stage) -> Option<String> {
    module
        .entry_points()
        .iter()
        .find(|ep| ep.stage == stage)
        .map(|ep| ep.name.clone())
}

fn build(
    device: &ash::Device,
    vertex: &str,
    fragment: &str,
    shader_module: vk::ShaderModule,
    layout: vk::PipelineLayout,
    color_format: vk::Format,
) -> anyhow::Result<vk::Pipeline> {
    let vertex_name = CString::new(vertex).context("vertex entry name")?;
    let fragment_name = CString::new(fragment).context("fragment entry name")?;
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(shader_module)
            .name(&vertex_name),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(shader_module)
            .name(&fragment_name),
    ];

    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let rgba = vk::ColorComponentFlags::R
        | vk::ColorComponentFlags::G
        | vk::ColorComponentFlags::B
        | vk::ColorComponentFlags::A;
    let blend_attachments =
        [vk::PipelineColorBlendAttachmentState::default().color_write_mask(rgba)];
    let color_blend =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let color_formats = [color_format];
    let mut rendering =
        vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);

    let create_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .push_next(&mut rendering);
    let infos = [create_info];

    // SAFETY: `infos` references live locals; the pipeline is freed by `GraphicsPipeline::Drop`.
    let pipelines = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &infos, None)
    }
    .map_err(|(_, code)| anyhow!("vkCreateGraphicsPipelines: {code}"))?;
    pipelines
        .first()
        .copied()
        .filter(|p| *p != vk::Pipeline::null())
        .ok_or_else(|| anyhow!("null graphics pipeline"))
}

/// Decodes little-endian SPIR-V bytes into words for `VkShaderModuleCreateInfo`.
fn spirv_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
