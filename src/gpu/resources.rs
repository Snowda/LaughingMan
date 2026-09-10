//! Minimal Vulkan presenter resources (staging buffer, sampled image, sampler, layout transition), simplified from Aspire vk-run's `memory.rs`.
#![allow(unsafe_code)]

use anyhow::{Context as _, anyhow};
use ash::vk;

use crate::num::Cast as _;

/// A host-visible, host-coherent buffer. Frees buffer then memory on drop.
pub struct Buffer {
    device: ash::Device,
    raw: vk::Buffer,
    memory: vk::DeviceMemory,
}

impl Buffer {
    /// Allocates a `size`-byte HOST_VISIBLE|HOST_COHERENT buffer with `usage`.
    pub fn new(
        device: &ash::Device,
        mem_props: &vk::PhysicalDeviceMemoryProperties,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> anyhow::Result<Self> {
        let info = vk::BufferCreateInfo::default()
            .size(size.max(4))
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: `info` references live locals; the buffer is freed in `Drop`.
        let raw = unsafe { device.create_buffer(&info, None) }.context("vkCreateBuffer")?;
        // SAFETY: `raw` was just created on `device`.
        let reqs = unsafe { device.get_buffer_memory_requirements(raw) };
        let flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let type_index = find_memory_type(mem_props, reqs.memory_type_bits, flags)
            .ok_or_else(|| anyhow!("no host-visible memory type"))?;
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(reqs.size)
            .memory_type_index(type_index);
        // SAFETY: valid device + alloc info.
        let memory = unsafe { device.allocate_memory(&alloc, None) }.context("vkAllocateMemory")?;
        // SAFETY: `raw`/`memory` are live and unbound.
        unsafe { device.bind_buffer_memory(raw, memory, 0) }.context("vkBindBufferMemory")?;
        Ok(Self {
            device: device.clone(),
            raw,
            memory,
        })
    }

    /// Copies `bytes` into the buffer (coherent memory, no flush needed).
    pub fn write(&self, bytes: &[u8]) -> anyhow::Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        let size = bytes.len() as vk::DeviceSize;
        // SAFETY: mapping the whole allocation; `memory` is host-visible and live.
        let ptr = unsafe {
            self.device
                .map_memory(self.memory, 0, size, vk::MemoryMapFlags::empty())
        }
        .context("vkMapMemory")?;
        // SAFETY: `ptr` maps at least `size` bytes; `bytes` is exactly `size` long.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast::<u8>(), bytes.len());
            self.device.unmap_memory(self.memory);
        }
        Ok(())
    }

    pub fn raw(&self) -> vk::Buffer {
        self.raw
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: owner has drained the device; handles are live.
        unsafe {
            self.device.destroy_buffer(self.raw, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

/// A device-local 2D image plus its view. Frees view, image, then memory on drop.
pub struct Image {
    device: ash::Device,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
}

impl Image {
    /// Allocates a `width`×`height` 2D color image of `format` with `usage` and a matching view.
    pub fn new(
        device: &ash::Device,
        mem_props: &vk::PhysicalDeviceMemoryProperties,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> anyhow::Result<Self> {
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: width.max(1),
                height: height.max(1),
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // SAFETY: `info` references live locals; the image is freed in `Drop`.
        let image = unsafe { device.create_image(&info, None) }.context("vkCreateImage")?;
        // SAFETY: `image` was just created on `device`.
        let reqs = unsafe { device.get_image_memory_requirements(image) };
        let type_index =
            find_memory_type(mem_props, reqs.memory_type_bits, vk::MemoryPropertyFlags::empty())
                .ok_or_else(|| anyhow!("no memory type for image"))?;
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(reqs.size)
            .memory_type_index(type_index);
        // SAFETY: valid device + alloc info.
        let memory = unsafe { device.allocate_memory(&alloc, None) }.context("vkAllocateMemory")?;
        // SAFETY: `image`/`memory` are live and unbound.
        unsafe { device.bind_image_memory(image, memory, 0) }.context("vkBindImageMemory")?;

        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(range);
        // SAFETY: valid device + info; the view is freed in `Drop`.
        let view = unsafe { device.create_image_view(&view_info, None) }
            .context("vkCreateImageView")?;

        Ok(Self {
            device: device.clone(),
            image,
            memory,
            view,
        })
    }

    pub fn raw(&self) -> vk::Image {
        self.image
    }

    pub fn view(&self) -> vk::ImageView {
        self.view
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        // SAFETY: owner has drained the device; handles are live.
        unsafe {
            self.device.destroy_image_view(self.view, None);
            self.device.destroy_image(self.image, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

/// A linear-filter, clamp-to-edge sampler. Frees on drop.
pub struct Sampler {
    device: ash::Device,
    raw: vk::Sampler,
}

impl Sampler {
    pub fn linear_clamp(device: &ash::Device) -> anyhow::Result<Self> {
        let info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
        // SAFETY: `info` references live locals; the sampler is freed in `Drop`.
        let raw = unsafe { device.create_sampler(&info, None) }.context("vkCreateSampler")?;
        Ok(Self {
            device: device.clone(),
            raw,
        })
    }

    pub fn raw(&self) -> vk::Sampler {
        self.raw
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        // SAFETY: owner has drained the device; handle is live.
        unsafe { self.device.destroy_sampler(self.raw, None) };
    }
}

/// A color-image layout transition: old/new layouts plus src/dst pipeline-stage and access scopes.
pub struct Transition {
    pub old_layout: vk::ImageLayout,
    pub new_layout: vk::ImageLayout,
    pub src_stage: vk::PipelineStageFlags,
    pub dst_stage: vk::PipelineStageFlags,
    pub src_access: vk::AccessFlags,
    pub dst_access: vk::AccessFlags,
}

/// Records a color-image layout transition into `cmd` via a classic pipeline barrier.
pub fn transition_image(device: &ash::Device, cmd: vk::CommandBuffer, image: vk::Image, t: &Transition) {
    let range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1);
    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(t.old_layout)
        .new_layout(t.new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(range)
        .src_access_mask(t.src_access)
        .dst_access_mask(t.dst_access);
    // SAFETY: `cmd` is recording; `image` is live; the barrier array outlives the call.
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            t.src_stage,
            t.dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            std::slice::from_ref(&barrier),
        );
    }
}

/// The first memory-type index allowed by `type_bits` that carries all of `flags`.
fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..props.memory_type_count).find(|&i| {
        let allowed = type_bits & (1 << i) != 0;
        allowed && props.memory_types[i.to_usize()].property_flags.contains(flags)
    })
}

#[cfg(test)]
mod tests {
    use super::find_memory_type;
    use ash::vk;

    #[test]
    fn finds_first_allowed_type_with_flags() {
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE;
        // ash structs have private padding, so they can't be built with a literal — default + set.
        let mut props = vk::PhysicalDeviceMemoryProperties::default();
        props.memory_types[0].property_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        props.memory_types[1].property_flags = host;
        props.memory_types[2].property_flags = host;
        props.memory_type_count = 3;
        assert_eq!(find_memory_type(&props, 0b111, host), Some(1));
        assert_eq!(find_memory_type(&props, 0b001, host), None);
        assert_eq!(find_memory_type(&props, 0b100, host), Some(2));
    }
}
