//! The swapchain: presentable images the renderer draws into. Wraps creation/recreation (resize /
//! out-of-date), image views, and acquire/present as safe outcome enums so the frame loop stays
//! `unsafe`-free at its call sites. Adapted from HardLight's `window/swapchain.rs`.
#![allow(unsafe_code)]

use anyhow::{Context as _, anyhow};
use ash::vk;

use crate::gpu::context::VkContext;
use crate::num::Cast as _;

const PREFERRED_FORMAT: vk::Format = vk::Format::B8G8R8A8_SRGB;
const PREFERRED_COLOR_SPACE: vk::ColorSpaceKHR = vk::ColorSpaceKHR::SRGB_NONLINEAR;
// One image over the driver minimum, so the CPU can build the next frame while one is presenting.
const EXTRA_IMAGES: u32 = 1;
const NO_IMAGE_MAX: u32 = 0;
const ACQUIRE_TIMEOUT: u64 = u64::MAX;

/// A swapchain image index (distinct from any frame index).
#[derive(Clone, Copy)]
pub struct ImageIndex(pub u32);

/// Outcome of acquiring: a usable image, or a stale swapchain to recreate.
pub enum Acquired {
    Image(ImageIndex),
    OutOfDate,
}

/// Outcome of presenting: presented, or stale/suboptimal (recreate before the next frame).
pub enum Presented {
    Ok,
    Stale,
}

struct Built {
    handle: vk::SwapchainKHR,
    format: vk::Format,
    extent: vk::Extent2D,
    images: Vec<vk::Image>,
    views: Vec<vk::ImageView>,
}

/// The swapchain plus its per-image color views. Destroyed on drop (views then swapchain).
pub struct Swapchain {
    device: ash::Device,
    loader: ash::khr::swapchain::Device,
    handle: vk::SwapchainKHR,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    images: Vec<vk::Image>,
    views: Vec<vk::ImageView>,
}

impl Swapchain {
    /// Creates a swapchain for `ctx`'s surface, sized to `desired` (clamped to the allowed extent).
    pub fn new(ctx: &VkContext, desired: vk::Extent2D) -> anyhow::Result<Self> {
        let loader = ash::khr::swapchain::Device::new(&ctx.instance, &ctx.device);
        let built = build(ctx, &loader, desired, vk::SwapchainKHR::null())?;
        Ok(Self {
            device: ctx.device.clone(),
            loader,
            handle: built.handle,
            format: built.format,
            extent: built.extent,
            images: built.images,
            views: built.views,
        })
    }

    /// Rebuilds at `desired` after a resize / out-of-date, reusing the old swapchain for handoff.
    pub fn recreate(&mut self, ctx: &VkContext, desired: vk::Extent2D) -> anyhow::Result<()> {
        // SAFETY: best-effort idle so the images about to be destroyed are no longer in use.
        unsafe {
            let _ = ctx.device.device_wait_idle();
        }
        let built = build(ctx, &self.loader, desired, self.handle)?;
        self.destroy_views();
        // SAFETY: the device is idle and the new swapchain supersedes the old handle.
        unsafe { self.loader.destroy_swapchain(self.handle, None) };
        self.handle = built.handle;
        self.format = built.format;
        self.extent = built.extent;
        self.images = built.images;
        self.views = built.views;
        Ok(())
    }

    /// Acquires the next image, signalling `signal` when it is ready to render into.
    pub fn acquire(&self, signal: vk::Semaphore) -> anyhow::Result<Acquired> {
        // SAFETY: `handle`/`signal` are live; no fence (the semaphore gates rendering).
        let result = unsafe {
            self.loader
                .acquire_next_image(self.handle, ACQUIRE_TIMEOUT, signal, vk::Fence::null())
        };
        match result {
            Ok((index, _suboptimal)) => Ok(Acquired::Image(ImageIndex(index))),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(Acquired::OutOfDate),
            Err(e) => Err(anyhow!("vkAcquireNextImageKHR: {e}")),
        }
    }

    /// Presents `image` on `queue` after `wait` (the render-finished semaphore) is signalled.
    pub fn present(
        &self,
        queue: vk::Queue,
        image: ImageIndex,
        wait: vk::Semaphore,
    ) -> anyhow::Result<Presented> {
        let wait_semaphores = [wait];
        let swapchains = [self.handle];
        let image_indices = [image.0];
        let info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);
        // SAFETY: all referenced arrays are live for the call; `queue` supports present.
        let result = unsafe { self.loader.queue_present(queue, &info) };
        match result {
            Ok(false) => Ok(Presented::Ok),
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(Presented::Stale),
            Err(e) => Err(anyhow!("vkQueuePresentKHR: {e}")),
        }
    }

    pub fn image(&self, index: ImageIndex) -> vk::Image {
        self.images[index.0.to_usize()]
    }

    pub fn view(&self, index: ImageIndex) -> vk::ImageView {
        self.views[index.0.to_usize()]
    }

    fn destroy_views(&mut self) {
        for view in self.views.drain(..) {
            // SAFETY: the device is idle; each view is live.
            unsafe { self.device.destroy_image_view(view, None) };
        }
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        self.destroy_views();
        // SAFETY: the owner drains the device before dropping; the handle is live.
        unsafe { self.loader.destroy_swapchain(self.handle, None) };
    }
}

fn build(
    ctx: &VkContext,
    loader: &ash::khr::swapchain::Device,
    desired: vk::Extent2D,
    old: vk::SwapchainKHR,
) -> anyhow::Result<Built> {
    // SAFETY: `physical_device`/`surface` belong to this instance and are live.
    let capabilities = unsafe {
        ctx.surface_loader
            .get_physical_device_surface_capabilities(ctx.physical_device, ctx.surface)
    }
    .context("surface capabilities")?;
    // SAFETY: same validity as above.
    let formats = unsafe {
        ctx.surface_loader
            .get_physical_device_surface_formats(ctx.physical_device, ctx.surface)
    }
    .context("surface formats")?;

    let surface_format = choose_format(&formats).ok_or_else(|| anyhow!("surface has no formats"))?;
    let extent = choose_extent(&capabilities, desired);
    if extent.width == 0 || extent.height == 0 {
        return Err(anyhow!("surface extent is zero (minimized)"));
    }
    let image_count = choose_image_count(&capabilities);

    let info = vk::SwapchainCreateInfoKHR::default()
        .surface(ctx.surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(capabilities.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        // FIFO: the only present mode guaranteed available, and it vsyncs (no tearing).
        .present_mode(vk::PresentModeKHR::FIFO)
        .clipped(true)
        .old_swapchain(old);

    // SAFETY: `info` references live locals; the swapchain is destroyed via `Swapchain`.
    let handle = unsafe { loader.create_swapchain(&info, None) }.context("vkCreateSwapchainKHR")?;
    // SAFETY: `handle` was just created by this loader.
    let images = match unsafe { loader.get_swapchain_images(handle) } {
        Ok(images) => images,
        Err(e) => {
            // SAFETY: the swapchain is unused; destroy it before returning.
            unsafe { loader.destroy_swapchain(handle, None) };
            return Err(anyhow!("vkGetSwapchainImagesKHR: {e}"));
        }
    };

    let mut views: Vec<vk::ImageView> = Vec::with_capacity(images.len());
    for &image in &images {
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(surface_format.format)
            .subresource_range(range);
        // SAFETY: `image` belongs to the swapchain just created; `view_info` is live.
        match unsafe { ctx.device.create_image_view(&view_info, None) } {
            Ok(view) => views.push(view),
            Err(e) => {
                for view in views {
                    // SAFETY: each created view is live and unused.
                    unsafe { ctx.device.destroy_image_view(view, None) };
                }
                // SAFETY: the swapchain is otherwise unused.
                unsafe { loader.destroy_swapchain(handle, None) };
                return Err(anyhow!("swapchain image view: {e}"));
            }
        }
    }

    Ok(Built {
        handle,
        format: surface_format.format,
        extent,
        images,
        views,
    })
}

fn choose_format(formats: &[vk::SurfaceFormatKHR]) -> Option<vk::SurfaceFormatKHR> {
    if formats.is_empty() {
        return None;
    }
    let preferred = formats
        .iter()
        .find(|f| f.format == PREFERRED_FORMAT && f.color_space == PREFERRED_COLOR_SPACE);
    Some(*preferred.unwrap_or(&formats[0]))
}

fn choose_extent(capabilities: &vk::SurfaceCapabilitiesKHR, desired: vk::Extent2D) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        return capabilities.current_extent;
    }
    vk::Extent2D {
        width: desired.width.clamp(
            capabilities.min_image_extent.width,
            capabilities.max_image_extent.width,
        ),
        height: desired.height.clamp(
            capabilities.min_image_extent.height,
            capabilities.max_image_extent.height,
        ),
    }
}

fn choose_image_count(capabilities: &vk::SurfaceCapabilitiesKHR) -> u32 {
    let wanted = capabilities.min_image_count + EXTRA_IMAGES;
    if capabilities.max_image_count == NO_IMAGE_MAX {
        return wanted;
    }
    wanted.min(capabilities.max_image_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn format(format: vk::Format, color_space: vk::ColorSpaceKHR) -> vk::SurfaceFormatKHR {
        vk::SurfaceFormatKHR::default()
            .format(format)
            .color_space(color_space)
    }

    #[test]
    fn choose_format_prefers_bgra_srgb_else_first() {
        let other = format(vk::Format::R8G8B8A8_UNORM, PREFERRED_COLOR_SPACE);
        let preferred = format(PREFERRED_FORMAT, PREFERRED_COLOR_SPACE);
        assert_eq!(
            choose_format(&[other, preferred]).map(|f| f.format),
            Some(PREFERRED_FORMAT)
        );
        assert_eq!(
            choose_format(&[other]).map(|f| f.format),
            Some(vk::Format::R8G8B8A8_UNORM)
        );
        assert!(choose_format(&[]).is_none());
    }

    #[test]
    fn choose_extent_uses_fixed_size_or_clamps() {
        let fixed = vk::SurfaceCapabilitiesKHR::default().current_extent(vk::Extent2D {
            width: 800,
            height: 600,
        });
        assert_eq!(
            choose_extent(&fixed, vk::Extent2D { width: 1280, height: 720 }),
            vk::Extent2D { width: 800, height: 600 }
        );

        let flexible = vk::SurfaceCapabilitiesKHR::default()
            .current_extent(vk::Extent2D { width: u32::MAX, height: u32::MAX })
            .min_image_extent(vk::Extent2D { width: 640, height: 480 })
            .max_image_extent(vk::Extent2D { width: 1920, height: 1080 });
        assert_eq!(
            choose_extent(&flexible, vk::Extent2D { width: 320, height: 240 }),
            vk::Extent2D { width: 640, height: 480 }
        );
        assert_eq!(
            choose_extent(&flexible, vk::Extent2D { width: 4000, height: 3000 }),
            vk::Extent2D { width: 1920, height: 1080 }
        );
    }

    #[test]
    fn choose_image_count_adds_one_and_clamps() {
        let no_max = vk::SurfaceCapabilitiesKHR::default()
            .min_image_count(2)
            .max_image_count(NO_IMAGE_MAX);
        assert_eq!(choose_image_count(&no_max), 3);
        let capped = vk::SurfaceCapabilitiesKHR::default()
            .min_image_count(3)
            .max_image_count(3);
        assert_eq!(choose_image_count(&capped), 3);
    }
}
