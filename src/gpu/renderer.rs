//! The per-frame compositor renderer: owns the compositor pipeline, the video texture, the two MSDF
//! atlases (uploaded once), the per-frame faces storage buffer, the descriptor set binding all five,
//! and per-frame sync. `draw` uploads the RGB frame and the packed face instances, then records the
//! compositor pass into the acquired swapchain image.
#![allow(unsafe_code)]

use anyhow::Context as _;
use ash::vk;

use crate::gpu::context::VkContext;
use crate::gpu::pipeline::GraphicsPipeline;
use crate::gpu::resources::{Buffer, Image, Sampler, Transition, transition_image};
use crate::gpu::swapchain::{Acquired, Presented, Swapchain};
use crate::num::Cast as _;
use crate::overlay::{FACE_FLOATS, MAX_FACES};
use crate::shaders::COMPOSITOR;

const FULLSCREEN_TRIANGLE_VERTICES: u32 = 3;
const RGBA_BYTES: u64 = 4;

/// What `draw` reports back to the event loop.
pub enum DrawOutcome {
    Presented,
    NeedRecreate,
}

/// The baked logo the renderer uploads once: the static + text MTSDF atlases (RGBA8), their shared
/// square side length, and the ring pivot in atlas UV.
pub struct LogoAtlases<'a> {
    pub static_rgba: &'a [u8],
    pub text_rgba: &'a [u8],
    pub size: u32,
    pub ring_pivot: (f32, f32),
}

/// The compositor renderer for one swapchain color format, capture resolution, and MSDF atlas size.
pub struct Renderer {
    device: ash::Device,
    command_buffer: vk::CommandBuffer,
    pipeline: GraphicsPipeline,
    _sampler: Sampler,
    video: Image,
    video_staging: Buffer,
    _static_msdf: Image,
    _text_msdf: Image,
    faces: Buffer,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    image_available: vk::Semaphore,
    render_finished: vk::Semaphore,
    // Reused RGB→RGBA and face-bytes scratch.
    rgba: Vec<u8>,
    face_bytes: Vec<u8>,
    cap_width: u32,
    cap_height: u32,
    // Atlas-UV centre the text ring spins around (the logo's circle centre, offset by the cap brim).
    ring_pivot: (f32, f32),
}

impl Renderer {
    /// Builds the renderer: the compositor pipeline, the video texture + staging, the two
    /// `msdf_size`² MSDF atlases (uploaded once from `static_rgba`/`text_rgba`), the faces storage
    /// buffer, the five-binding descriptor set, and per-frame sync.
    pub fn new(
        ctx: &VkContext,
        color_format: vk::Format,
        cap_width: u32,
        cap_height: u32,
        atlases: &LogoAtlases,
    ) -> anyhow::Result<Self> {
        let msdf_size = atlases.size;
        let ring_pivot = atlases.ring_pivot;
        let device = ctx.device.clone();
        let pipeline = GraphicsPipeline::new(&device, COMPOSITOR.spv, color_format)?;
        let sampler = Sampler::linear_clamp(&device)?;

        let sampled = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST;
        let video = Image::new(&device, &ctx.memory_properties, cap_width, cap_height, vk::Format::R8G8B8A8_UNORM, sampled)?;
        let video_staging = Buffer::new(
            &device,
            &ctx.memory_properties,
            u64::from(cap_width) * u64::from(cap_height) * RGBA_BYTES,
            vk::BufferUsageFlags::TRANSFER_SRC,
        )?;
        let static_msdf = Image::new(&device, &ctx.memory_properties, msdf_size, msdf_size, vk::Format::R8G8B8A8_UNORM, sampled)?;
        let text_msdf = Image::new(&device, &ctx.memory_properties, msdf_size, msdf_size, vk::Format::R8G8B8A8_UNORM, sampled)?;

        let faces = Buffer::new(
            &device,
            &ctx.memory_properties,
            (MAX_FACES * FACE_FLOATS * 4).to_u64(),
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;

        // One-time upload of the two MSDF atlases via a transient command buffer.
        upload_msdf_atlases(ctx, &device, &static_msdf, atlases.static_rgba, &text_msdf, atlases.text_rgba, msdf_size)?;

        let (descriptor_pool, descriptor_set) = build_descriptor_set(&device, &pipeline)?;
        write_descriptor_set(&device, descriptor_set, &video, &static_msdf, &text_msdf, &sampler, &faces);

        let cmd_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(ctx.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: `cmd_alloc` references the live command pool.
        let command_buffer = unsafe { device.allocate_command_buffers(&cmd_alloc) }.context("command buffer")?[0];

        let sem_info = vk::SemaphoreCreateInfo::default();
        // SAFETY: live device; semaphores freed in `Drop`.
        let image_available = unsafe { device.create_semaphore(&sem_info, None) }.context("semaphore")?;
        // SAFETY: as above.
        let render_finished = unsafe { device.create_semaphore(&sem_info, None) }.context("semaphore")?;

        Ok(Self {
            device,
            command_buffer,
            pipeline,
            _sampler: sampler,
            video,
            video_staging,
            _static_msdf: static_msdf,
            _text_msdf: text_msdf,
            faces,
            descriptor_pool,
            descriptor_set,
            image_available,
            render_finished,
            rgba: Vec::with_capacity((cap_width * cap_height * 4).to_usize()),
            face_bytes: Vec::with_capacity(MAX_FACES * FACE_FLOATS * 4),
            cap_width,
            cap_height,
            ring_pivot,
        })
    }

    /// Uploads the `width`×`height` RGB frame and the packed face instances (`MAX_FACES *
    /// FACE_FLOATS` floats), then draws the composite into the swapchain. `NeedRecreate` on a stale
    /// swapchain. Frames whose size differs from the texture are skipped.
    pub fn draw(
        &mut self,
        ctx: &VkContext,
        swapchain: &Swapchain,
        rgb: &[u8],
        width: u32,
        height: u32,
        face_floats: &[f32],
    ) -> anyhow::Result<DrawOutcome> {
        if width != self.cap_width || height != self.cap_height {
            return Ok(DrawOutcome::Presented);
        }
        // SAFETY: fully drain the previous frame so the command buffer / semaphores / host-visible
        // buffers are free to reuse.
        unsafe { self.device.device_wait_idle() }.context("device_wait_idle")?;

        self.expand_to_rgba(rgb);
        self.video_staging.write(&self.rgba)?;
        self.face_bytes.clear();
        for f in face_floats {
            self.face_bytes.extend_from_slice(&f.to_le_bytes());
        }
        self.faces.write(&self.face_bytes)?;

        let image_index = match swapchain.acquire(self.image_available)? {
            Acquired::Image(index) => index,
            Acquired::OutOfDate => return Ok(DrawOutcome::NeedRecreate),
        };
        let swap_image = swapchain.image(image_index);
        let swap_view = swapchain.view(image_index);
        let extent = swapchain.extent;
        let cmd = self.command_buffer;

        // SAFETY: the device is idle; `cmd` is owned and not in flight.
        unsafe {
            self.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).context("reset")?;
            let begin = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device.begin_command_buffer(cmd, &begin).context("begin")?;
            self.record_video_upload(cmd);
            self.record_pass(cmd, swap_image, swap_view, extent);
            self.device.end_command_buffer(cmd).context("end")?;
        }

        let wait = [self.image_available];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let cmds = [cmd];
        let signal = [self.render_finished];
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&wait)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&cmds)
            .signal_semaphores(&signal);
        // SAFETY: `submit` references live locals; `cmd` was just recorded.
        unsafe {
            self.device.queue_submit(ctx.queue, &[submit], vk::Fence::null()).context("submit")?;
        }

        match swapchain.present(ctx.queue, image_index, self.render_finished)? {
            Presented::Ok => Ok(DrawOutcome::Presented),
            Presented::Stale => Ok(DrawOutcome::NeedRecreate),
        }
    }

    // Records the video-texture upload: transition to TRANSFER_DST, copy staging, → SHADER_READ.
    unsafe fn record_video_upload(&self, cmd: vk::CommandBuffer) {
        transition_image(&self.device, cmd, self.video.raw(), &Transition {
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            src_stage: vk::PipelineStageFlags::TOP_OF_PIPE,
            dst_stage: vk::PipelineStageFlags::TRANSFER,
            src_access: vk::AccessFlags::empty(),
            dst_access: vk::AccessFlags::TRANSFER_WRITE,
        });
        let region = copy_region(self.cap_width, self.cap_height);
        // SAFETY: `cmd` is recording; buffer/image live; the region outlives the call.
        unsafe {
            self.device.cmd_copy_buffer_to_image(cmd, self.video_staging.raw(), self.video.raw(), vk::ImageLayout::TRANSFER_DST_OPTIMAL, std::slice::from_ref(&region));
        }
        transition_image(&self.device, cmd, self.video.raw(), &Transition {
            old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            src_stage: vk::PipelineStageFlags::TRANSFER,
            dst_stage: vk::PipelineStageFlags::FRAGMENT_SHADER,
            src_access: vk::AccessFlags::TRANSFER_WRITE,
            dst_access: vk::AccessFlags::SHADER_READ,
        });
    }

    // Records the compositor pass into the swapchain image.
    unsafe fn record_pass(&self, cmd: vk::CommandBuffer, swap_image: vk::Image, swap_view: vk::ImageView, extent: vk::Extent2D) {
        transition_image(&self.device, cmd, swap_image, &Transition {
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            src_stage: vk::PipelineStageFlags::TOP_OF_PIPE,
            dst_stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            src_access: vk::AccessFlags::empty(),
            dst_access: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        });

        let clear = vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } };
        let attachment = vk::RenderingAttachmentInfo::default()
            .image_view(swap_view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(clear);
        let color_attachments = [attachment];
        let rendering = vk::RenderingInfo::default()
            .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent })
            .layer_count(1)
            .color_attachments(&color_attachments);

        let viewport = vk::Viewport { x: 0.0, y: 0.0, width: extent.width.to_f32(), height: extent.height.to_f32(), min_depth: 0.0, max_depth: 1.0 };
        let scissor = vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent };

        // Aspect-preserving letterbox: window fragment → video pixel. `View` = [inv_fit, off_x,
        // off_y, vid_w, vid_h].
        let (sw, sh) = (extent.width.to_f32(), extent.height.to_f32());
        let (vw, vh) = (self.cap_width.to_f32(), self.cap_height.to_f32());
        let fit = (sw / vw).min(sh / vh);
        let (px, py) = self.ring_pivot;
        let view = [1.0 / fit, (sw - vw * fit) * 0.5, (sh - vh * fit) * 0.5, vw, vh, px, py];
        let mut push = [0u8; 28];
        for (i, f) in view.iter().enumerate() {
            push[i * 4..i * 4 + 4].copy_from_slice(&f.to_le_bytes());
        }

        // SAFETY: `cmd` is recording; all referenced objects are live for the calls.
        unsafe {
            self.device.cmd_begin_rendering(cmd, &rendering);
            self.device.cmd_set_viewport(cmd, 0, &[viewport]);
            self.device.cmd_set_scissor(cmd, 0, &[scissor]);
            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline.raw());
            self.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline.layout(), 0, &[self.descriptor_set], &[]);
            self.device.cmd_push_constants(cmd, self.pipeline.layout(), vk::ShaderStageFlags::FRAGMENT, 0, &push);
            self.device.cmd_draw(cmd, FULLSCREEN_TRIANGLE_VERTICES, 1, 0, 0);
            self.device.cmd_end_rendering(cmd);
        }

        transition_image(&self.device, cmd, swap_image, &Transition {
            old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            src_stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_stage: vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            src_access: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dst_access: vk::AccessFlags::empty(),
        });
    }

    fn expand_to_rgba(&mut self, rgb: &[u8]) {
        self.rgba.clear();
        for px in rgb.chunks_exact(3) {
            self.rgba.extend_from_slice(&[px[0], px[1], px[2], 255]);
        }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        // SAFETY: drain the device before destroying sync/descriptor objects; the RAII resource
        // fields drop afterwards.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_semaphore(self.image_available, None);
            self.device.destroy_semaphore(self.render_finished, None);
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}

fn copy_region(width: u32, height: u32) -> vk::BufferImageCopy {
    let subresource = vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1);
    vk::BufferImageCopy::default()
        .image_subresource(subresource)
        .image_extent(vk::Extent3D { width, height, depth: 1 })
}

// Uploads both MSDF atlases into their images via a transient command buffer submitted once.
fn upload_msdf_atlases(
    ctx: &VkContext,
    device: &ash::Device,
    static_msdf: &Image,
    static_rgba: &[u8],
    text_msdf: &Image,
    text_rgba: &[u8],
    size: u32,
) -> anyhow::Result<()> {
    let staging_size = u64::from(size) * u64::from(size) * RGBA_BYTES;
    let static_staging = Buffer::new(device, &ctx.memory_properties, staging_size, vk::BufferUsageFlags::TRANSFER_SRC)?;
    let text_staging = Buffer::new(device, &ctx.memory_properties, staging_size, vk::BufferUsageFlags::TRANSFER_SRC)?;
    static_staging.write(static_rgba)?;
    text_staging.write(text_rgba)?;

    let alloc = vk::CommandBufferAllocateInfo::default()
        .command_pool(ctx.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    // SAFETY: valid command pool.
    let cmd = unsafe { device.allocate_command_buffers(&alloc) }.context("upload cmd")?[0];
    let begin = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    // SAFETY: `cmd` is freshly allocated and not in flight.
    unsafe {
        device.begin_command_buffer(cmd, &begin).context("begin upload")?;
        for (image, staging) in [(static_msdf, &static_staging), (text_msdf, &text_staging)] {
            transition_image(device, cmd, image.raw(), &Transition {
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                src_stage: vk::PipelineStageFlags::TOP_OF_PIPE,
                dst_stage: vk::PipelineStageFlags::TRANSFER,
                src_access: vk::AccessFlags::empty(),
                dst_access: vk::AccessFlags::TRANSFER_WRITE,
            });
            let region = copy_region(size, size);
            device.cmd_copy_buffer_to_image(cmd, staging.raw(), image.raw(), vk::ImageLayout::TRANSFER_DST_OPTIMAL, std::slice::from_ref(&region));
            transition_image(device, cmd, image.raw(), &Transition {
                old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                src_stage: vk::PipelineStageFlags::TRANSFER,
                dst_stage: vk::PipelineStageFlags::FRAGMENT_SHADER,
                src_access: vk::AccessFlags::TRANSFER_WRITE,
                dst_access: vk::AccessFlags::SHADER_READ,
            });
        }
        device.end_command_buffer(cmd).context("end upload")?;
        let cmds = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cmds);
        device.queue_submit(ctx.queue, &[submit], vk::Fence::null()).context("submit upload")?;
        device.device_wait_idle().context("await upload")?;
        device.free_command_buffers(ctx.command_pool, &[cmd]);
    }
    Ok(())
}

// Creates the descriptor pool + set for the compositor's bindings (3 sampled images, 1 sampler,
// 1 storage buffer).
fn build_descriptor_set(device: &ash::Device, pipeline: &GraphicsPipeline) -> anyhow::Result<(vk::DescriptorPool, vk::DescriptorSet)> {
    let sizes = [
        vk::DescriptorPoolSize::default().ty(vk::DescriptorType::SAMPLED_IMAGE).descriptor_count(3),
        vk::DescriptorPoolSize::default().ty(vk::DescriptorType::SAMPLER).descriptor_count(1),
        vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1),
    ];
    let pool_info = vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes);
    // SAFETY: live locals; freed in `Renderer::Drop`.
    let pool = unsafe { device.create_descriptor_pool(&pool_info, None) }.context("descriptor pool")?;
    let set_layouts = [pipeline.set_layout()];
    let alloc = vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&set_layouts);
    // SAFETY: `alloc` references the just-created pool/layout.
    let set = unsafe { device.allocate_descriptor_sets(&alloc) }.context("allocate set")?[0];
    Ok((pool, set))
}

// Writes the five compositor bindings into `set`. Image handles are stable, so this runs once.
fn write_descriptor_set(
    device: &ash::Device,
    set: vk::DescriptorSet,
    video: &Image,
    static_msdf: &Image,
    text_msdf: &Image,
    sampler: &Sampler,
    faces: &Buffer,
) {
    let read_only = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
    let video_info = [vk::DescriptorImageInfo::default().image_view(video.view()).image_layout(read_only)];
    let sampler_info = [vk::DescriptorImageInfo::default().sampler(sampler.raw())];
    let static_info = [vk::DescriptorImageInfo::default().image_view(static_msdf.view()).image_layout(read_only)];
    let text_info = [vk::DescriptorImageInfo::default().image_view(text_msdf.view()).image_layout(read_only)];
    let faces_info = [vk::DescriptorBufferInfo::default().buffer(faces.raw()).offset(0).range(vk::WHOLE_SIZE)];

    let writes = [
        vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).image_info(&video_info),
        vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::SAMPLER).image_info(&sampler_info),
        vk::WriteDescriptorSet::default().dst_set(set).dst_binding(2).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).image_info(&static_info),
        vk::WriteDescriptorSet::default().dst_set(set).dst_binding(3).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).image_info(&text_info),
        vk::WriteDescriptorSet::default().dst_set(set).dst_binding(4).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&faces_info),
    ];
    // SAFETY: `writes` references live locals; the set and all handles are valid.
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}
