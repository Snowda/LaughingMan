//! Vulkan context: instance, surface, graphics+present device (dynamic rendering + swapchain), queue, command pool.
#![allow(unsafe_code)]

use std::ffi::{CStr, c_void};

use anyhow::{Context as _, anyhow};
use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;

use crate::num::Cast as _;

const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

pub struct VkContext {
    _entry: ash::Entry,
    pub instance: ash::Instance,
    debug: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    pub surface_loader: ash::khr::surface::Instance,
    pub surface: vk::SurfaceKHR,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub command_pool: vk::CommandPool,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
}

impl VkContext {
    pub fn new(window: &Window, app_name: &CStr) -> anyhow::Result<Self> {
        // SAFETY: `Entry::load` dynamically links the system Vulkan loader; `Err` if absent.
        let entry = unsafe { ash::Entry::load() }.context("loading the Vulkan loader")?;

        let display_handle = window.display_handle()?.as_raw();
        let window_handle = window.window_handle()?.as_raw();
        let mut extensions = ash_window::enumerate_required_extensions(display_handle)?.to_vec();
        let mut layers: Vec<*const std::ffi::c_char> = Vec::new();
        let use_validation = cfg!(debug_assertions) && validation_layer_present(&entry);
        if use_validation {
            layers.push(VALIDATION_LAYER.as_ptr());
            extensions.push(ash::ext::debug_utils::NAME.as_ptr());
        }

        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .api_version(vk::API_VERSION_1_3);
        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions);
        // SAFETY: `instance_info` references live locals; the instance is destroyed in `Drop`.
        let instance =
            unsafe { entry.create_instance(&instance_info, None) }.context("vkCreateInstance")?;

        let debug = if use_validation {
            setup_debug_messenger(&entry, &instance)
        } else {
            None
        };

        // SAFETY: handles come from `window`, valid for the surface's lifetime (window outlives context).
        let surface = unsafe {
            ash_window::create_surface(&entry, &instance, display_handle, window_handle, None)
        }
        .context("creating the window surface")?;
        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);

        let (physical_device, queue_family) = select_device(&instance, &surface_loader, surface)?;

        let priorities = [1.0f32];
        let queue_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        let device_exts = [ash::khr::swapchain::NAME.as_ptr()];
        let mut features13 = vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true);
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_exts)
            .push_next(&mut features13);
        // SAFETY: `device_info` references live locals; the device is destroyed in `Drop`.
        let device = unsafe { instance.create_device(physical_device, &device_info, None) }
            .context("vkCreateDevice")?;
        // SAFETY: queue 0 of the family just requested exists on `device`.
        let queue = unsafe { device.get_device_queue(queue_family, 0) };

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: valid device + info; the pool is destroyed in `Drop`.
        let command_pool =
            unsafe { device.create_command_pool(&pool_info, None) }.context("vkCreateCommandPool")?;

        // SAFETY: `physical_device` belongs to `instance`.
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        Ok(Self {
            _entry: entry,
            instance,
            debug,
            surface_loader,
            surface,
            physical_device,
            device,
            queue,
            command_pool,
            memory_properties,
        })
    }
}

fn validation_layer_present(entry: &ash::Entry) -> bool {
    // SAFETY: `entry` is a live loader.
    let Ok(layers) = (unsafe { entry.enumerate_instance_layer_properties() }) else {
        return false;
    };
    layers.iter().any(|layer| {
        // SAFETY: `layer_name` is a NUL-terminated fixed array from the loader.
        let name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
        name == VALIDATION_LAYER
    })
}

/// Creates a debug messenger printing validation warnings/errors to stderr; `None` if creation fails.
fn setup_debug_messenger(
    entry: &ash::Entry,
    instance: &ash::Instance,
) -> Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)> {
    let loader = ash::ext::debug_utils::Instance::new(entry, instance);
    let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(debug_callback));
    // SAFETY: `info` references live locals; the messenger is destroyed in `Drop`.
    let messenger = unsafe { loader.create_debug_utils_messenger(&info, None) }.ok()?;
    Some((loader, messenger))
}

/// Validation-layer callback: prints each message to stderr; always returns `FALSE`.
unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut c_void,
) -> vk::Bool32 {
    // SAFETY: the loader guarantees `data` and its `p_message` are valid for the call.
    let message = unsafe { CStr::from_ptr((*data).p_message) };
    eprintln!("[vulkan {severity:?}] {}", message.to_string_lossy());
    vk::FALSE
}

/// Picks a physical device + queue-family with graphics+present on `surface`; prefers a discrete GPU.
fn select_device(
    instance: &ash::Instance,
    surface_loader: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> anyhow::Result<(vk::PhysicalDevice, u32)> {
    // SAFETY: `instance` is valid for the call.
    let devices = unsafe { instance.enumerate_physical_devices() }
        .context("enumerating physical devices")?;

    let mut fallback: Option<(vk::PhysicalDevice, u32)> = None;
    for physical_device in devices {
        // SAFETY: `physical_device` came from this instance's enumeration.
        let families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        for (index, family) in families.iter().enumerate() {
            let family_index = index.to_u32();
            if !family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                continue;
            }
            // SAFETY: `physical_device`/`surface` belong to this instance and are live.
            let presents = unsafe {
                surface_loader.get_physical_device_surface_support(
                    physical_device,
                    family_index,
                    surface,
                )
            }
            .unwrap_or(false);
            if !presents {
                continue;
            }
            // SAFETY: `physical_device` is valid.
            let props = unsafe { instance.get_physical_device_properties(physical_device) };
            if props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                return Ok((physical_device, family_index));
            }
            fallback.get_or_insert((physical_device, family_index));
        }
    }
    fallback.ok_or_else(|| anyhow!("no Vulkan device with a graphics+present queue found"))
}

impl Drop for VkContext {
    fn drop(&mut self) {
        // SAFETY: the presenter drains the device before dropping; every handle is live and owned.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            if let Some((loader, messenger)) = self.debug.take() {
                loader.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}
