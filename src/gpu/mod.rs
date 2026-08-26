//! The Vulkan presenter (Phase 1): a winit window whose swapchain is drawn by the Aspire
//! `passthrough` pipeline sampling the uploaded video frame. Split into a context (instance /
//! surface / device / queue), resource helpers (buffer / image / sampler / barriers), the
//! swapchain, the reflection-driven pipeline, the per-frame renderer, and the winit event loop.
//!
//! Everything here is `unsafe`-heavy ash interop; each module re-permits `unsafe` locally with
//! `// SAFETY:` notes rather than the crate-wide deny. The core render path is proven headless in
//! `shaders::gpu_tests` (offscreen render + readback); this module adds surface/swapchain/present,
//! which need a real window and are verified interactively.

mod context;
mod pipeline;
mod present;
mod renderer;
mod resources;
mod swapchain;

pub use present::{DEFAULT_RING_OMEGA, present_window};
