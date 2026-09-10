//! The winit event loop. Owns the frame source and face provider; on `resumed` it builds the
//! window, Vulkan context, swapchain, and compositor renderer (with the MSDF atlases). Each
//! `RedrawRequested`: grab a frame, detect faces, track them (coast/smooth), turn each track into a
//! per-face mask instance with a time-advancing ring phase, and composite. Esc/Space or closing the
//! window exits.

use std::ffi::CString;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::Context as _;
use ash::vk;
use triomphe::ThinArc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::capture::{Frame, FrameSource};
use crate::detect::track::{Tracker, TrackerParams};
use crate::detect::{Detection, FaceProvider};
use crate::gpu::context::VkContext;
use crate::gpu::renderer::{DrawOutcome, Renderer};
use crate::gpu::swapchain::Swapchain;
use crate::logo;
use crate::num::Cast as _;
use crate::overlay::{
    DEFAULT_COVER_SCALE, FaceInstance, composite_pixel, face_instance, pack_faces_padded,
};

/// One frame handed to the background detector.
struct DetectionInput {
    rgb: ThinArc<(), u8>,
    width: u32,
    height: u32,
    elapsed: f32,
}

/// Runs `provider` on a background thread, decoupling detection cadence from the render rate: it
/// takes the latest frame off `in_rx` (capacity 1, so stale frames are simply skipped) and posts
/// detections to `out_tx`. Returns the submit/receive ends for the render loop.
fn spawn_detector(
    mut provider: Box<dyn FaceProvider>,
) -> (SyncSender<DetectionInput>, Receiver<Vec<Detection>>, JoinHandle<()>) {
    let (in_tx, in_rx) = sync_channel::<DetectionInput>(1);
    let (out_tx, out_rx) = sync_channel::<Vec<Detection>>(1);
    let handle = std::thread::spawn(move || {
        while let Ok(input) = in_rx.recv() {
            match provider.detect_frame(&input.rgb.slice, input.width, input.height, input.elapsed) {
                Ok(dets) => {
                    let _ = out_tx.try_send(dets);
                }
                Err(e) => eprintln!("detection failed: {e:#}"),
            }
        }
    });
    (in_tx, out_rx, handle)
}

const FPS_REPORT_INTERVAL: Duration = Duration::from_millis(500);
// Procedural MSDF atlas size and distance range (the synthetic logo stand-in).
const MSDF_SIZE: u32 = 256;
const MSDF_RANGE: f32 = 16.0;
const RING_TEETH: u32 = 12;
/// Ring spin rate (radians/second). Negative = counter-clockwise; magnitude ≈ 0.1 rev/s (a slow spin).
pub const DEFAULT_RING_OMEGA: f32 = -0.6;

/// The live rendering state, built once the window exists. Field order is the drop order: renderer
/// → swapchain → context → window, so the surface is destroyed before the window it references.
struct State {
    renderer: Renderer,
    swapchain: Swapchain,
    ctx: VkContext,
    window: Arc<Window>,
    needs_recreate: bool,
    frames: u32,
    last_report: Instant,
    presented: u32,
    start: Instant,
    last_frame: Instant,
    // The MSDF atlas side length (loaded or procedural), for the per-face AA range.
    msdf_size: u32,
    // CPU copies of the composited atlases (static with its 3-level alpha, text) + the ring pivot,
    // kept so the `S` key can composite a screenshot on the CPU via `overlay::composite_pixel`.
    static_atlas: Vec<u8>,
    text_atlas: Vec<u8>,
    ring_pivot: (f32, f32),
    screenshot: bool,
}

/// The winit application: frame source, background detector channels, tracker, and render state.
struct App {
    source: Box<dyn FrameSource>,
    detect_in: SyncSender<DetectionInput>,
    detect_out: Receiver<Vec<Detection>>,
    _detector: JoinHandle<()>,
    tracker: Tracker,
    ring_omega: f32,
    title_index: u32,
    app_name: CString,
    max_frames: Option<u32>,
    // Baked MTSDF atlas PNGs; when both are set the real logo is composited, else a placeholder.
    logo_static: Option<PathBuf>,
    logo_text: Option<PathBuf>,
    logo_front: Option<PathBuf>,
    state: Option<State>,
}

/// Opens a window and runs the capture → detect → track → composite pipeline until the user quits
/// (or `max_frames` frames are presented). `faces` supplies detections (real detector or demo).
pub fn present_window(
    source: Box<dyn FrameSource>,
    faces: Box<dyn FaceProvider>,
    title_index: u32,
    ring_omega: f32,
    logo_static: Option<PathBuf>,
    logo_text: Option<PathBuf>,
    logo_front: Option<PathBuf>,
    max_frames: Option<u32>,
) -> anyhow::Result<()> {
    let event_loop = EventLoop::new().context("creating the winit event loop")?;
    // Poll, not Wait: a live video feed must redraw every frame, not only on input events.
    event_loop.set_control_flow(ControlFlow::Poll);
    let (detect_in, detect_out, detector) = spawn_detector(faces);
    let mut app = App {
        source,
        detect_in,
        detect_out,
        _detector: detector,
        tracker: Tracker::new(TrackerParams::default()),
        ring_omega,
        title_index,
        app_name: CString::new("LaughingMan").context("app name")?,
        max_frames,
        logo_static,
        logo_text,
        logo_front,
        state: None,
    };
    event_loop.run_app(&mut app).context("running the event loop")?;
    Ok(())
}

// Resolves the two MSDF atlases: the baked PNGs when both paths are given (the real logo), else the
// procedural placeholder. Returns `(static_rgba, text_rgba, size)`.
fn resolve_atlases(
    logo_static: &Option<PathBuf>,
    logo_text: &Option<PathBuf>,
) -> anyhow::Result<(Vec<u8>, Vec<u8>, u32)> {
    match (logo_static, logo_text) {
        (Some(s), Some(t)) => {
            let (static_rgba, ss) = logo::load_atlas(s)?;
            let (text_rgba, ts) = logo::load_atlas(t)?;
            if ss != ts {
                anyhow::bail!("static atlas is {ss}px but text atlas is {ts}px; bake both at one --size");
            }
            Ok((static_rgba, text_rgba, ss))
        }
        // A single combined logo atlas (e.g. the whole laugh.svg): static-only, blank rotating layer.
        (Some(s), None) => {
            let (static_rgba, ss) = logo::load_atlas(s)?;
            Ok((static_rgba, logo::blank_atlas(ss), ss))
        }
        _ => {
            eprintln!(
                "no --logo-static supplied: using a PROCEDURAL placeholder, not the real Laughing \
                 Man art. Bake the logo SVG with `laughing-bake` and pass it via --logo-static."
            );
            Ok((
                logo::synthetic_static(MSDF_SIZE, MSDF_RANGE),
                logo::synthetic_text(MSDF_SIZE, MSDF_RANGE, RING_TEETH),
                MSDF_SIZE,
            ))
        }
    }
}

// The logo's blue (#23498c) and white, matching the compositor's ink.
const SCREENSHOT_WHITE: [f32; 3] = [1.0, 1.0, 1.0];
const SCREENSHOT_BLUE: [f32; 3] = [0.137, 0.286, 0.549];

// Composites the mask over `frame` on the CPU — reusing `overlay::composite_pixel`, the verified
// mirror of the compositor shader — into an RGBA image. Captures the full-resolution video frame
// *with* the overlay (unlike a window grab, no letterbox bars); nearest atlas sampling makes the MSDF
// edges marginally harder than the GPU's, but the composite matches.
fn composite_frame(
    frame: &Frame,
    faces: &[FaceInstance],
    static_atlas: &[u8],
    text_atlas: &[u8],
    size: u32,
    pivot: (f32, f32),
) -> image::RgbaImage {
    let s = size.to_f32();
    let texel = |u: f32, v: f32| -> usize {
        let x = (u.clamp(0.0, 1.0) * (s - 1.0)).to_u32();
        let y = (v.clamp(0.0, 1.0) * (s - 1.0)).to_u32();
        ((y * size + x) * 4).to_usize()
    };
    let sample_static = |u: f32, v: f32| -> [f32; 4] {
        let i = texel(u, v);
        [
            f32::from(static_atlas[i]) / 255.0,
            f32::from(static_atlas[i + 1]) / 255.0,
            f32::from(static_atlas[i + 2]) / 255.0,
            f32::from(static_atlas[i + 3]) / 255.0,
        ]
    };
    let sample_text = |u: f32, v: f32| -> [f32; 3] {
        let i = texel(u, v);
        [f32::from(text_atlas[i]) / 255.0, f32::from(text_atlas[i + 1]) / 255.0, f32::from(text_atlas[i + 2]) / 255.0]
    };

    let (w, h) = (frame.width, frame.height);
    let mut img = image::RgbaImage::new(w, h);
    for py in 0..h {
        for px in 0..w {
            let vi = ((py * w + px) * 3).to_usize();
            let video = [
                f32::from(frame.rgb[vi]) / 255.0,
                f32::from(frame.rgb[vi + 1]) / 255.0,
                f32::from(frame.rgb[vi + 2]) / 255.0,
            ];
            let out = composite_pixel(video, faces, (px.to_f32(), py.to_f32()), SCREENSHOT_WHITE, SCREENSHOT_BLUE, pivot, &sample_static, &sample_text);
            let q = |c: f32| (c.clamp(0.0, 1.0) * 255.0).to_u8();
            img.put_pixel(px, py, image::Rgba([q(out[0]), q(out[1]), q(out[2]), 255]));
        }
    }
    img
}

// Composites the frame with the overlay and saves it as a `laugh_<timestamp>.png` under `dir`.
fn save_screenshot(
    frame: &Frame,
    faces: &[FaceInstance],
    static_atlas: &[u8],
    text_atlas: &[u8],
    size: u32,
    pivot: (f32, f32),
    dir: &std::path::Path,
) -> anyhow::Result<std::path::PathBuf> {
    let img = composite_frame(frame, faces, static_atlas, text_atlas, size, pivot);
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    let path = dir.join(format!("laugh_{stamp}.png"));
    img.save(&path).with_context(|| format!("saving {}", path.display()))?;
    Ok(path)
}

#[cfg(test)]
mod screenshot_tests {
    use super::{composite_frame, save_screenshot};
    use crate::capture::Frame;
    use crate::detect::{Bbox, Detection};
    use crate::num::Cast as _;
    use crate::overlay::{FaceInstance, face_instance};

    const SZ: u32 = 40;
    const ATLAS: u32 = 64;
    const GRAY: u8 = 100;

    // A 40x40 solid-gray frame + the silhouette-stamped synthetic atlases + one face filling the
    // frame (box 4..36 → centre 20,20), so the face centre samples the white mask.
    fn scene() -> (Frame, Vec<FaceInstance>, Vec<u8>, Vec<u8>) {
        let frame = Frame { width: SZ, height: SZ, rgb: vec![GRAY; (SZ * SZ * 3).to_usize()] };
        let mut static_atlas = crate::logo::synthetic_static(ATLAS, 8.0);
        crate::logo::stamp_silhouette(&mut static_atlas, ATLAS); // alpha := band-level face disc
        let text_atlas = crate::logo::blank_atlas(ATLAS);
        let det = Detection {
            bbox: Bbox { x1: 4.0, y1: 4.0, x2: 36.0, y2: 36.0 },
            score: 1.0,
            landmarks: [(14.0, 16.0), (26.0, 16.0), (20.0, 22.0), (16.0, 28.0), (24.0, 28.0)],
        };
        let faces = vec![face_instance(&det, 1.0, ATLAS.to_f32(), 8.0, 0.0, 1.0)];
        (frame, faces, static_atlas, text_atlas)
    }

    #[test]
    fn composite_frame_draws_the_overlay_over_the_video() {
        let (frame, faces, static_atlas, text_atlas) = scene();
        let img = composite_frame(&frame, &faces, &static_atlas, &text_atlas, ATLAS, (0.5, 0.5));
        let center = img.get_pixel(SZ / 2, SZ / 2);
        let corner = img.get_pixel(0, 0);
        assert!(center[0] > 200 && center[1] > 200, "face centre shows the white mask: {center:?}");
        assert_eq!([corner[0], corner[1], corner[2]], [GRAY, GRAY, GRAY], "corner stays untouched video");
    }

    #[test]
    fn save_screenshot_writes_a_loadable_png_with_the_overlay() {
        let (frame, faces, static_atlas, text_atlas) = scene();
        let dir = std::env::temp_dir().join("laughingman_screenshot_test");
        let path = save_screenshot(&frame, &faces, &static_atlas, &text_atlas, ATLAS, (0.5, 0.5), &dir)
            .expect("save screenshot");
        assert!(path.exists(), "the PNG was written");
        // Round-trip: it's a valid PNG at full frame resolution, with the overlay composited in.
        let loaded = image::open(&path).expect("open png").to_rgba8();
        assert_eq!(loaded.dimensions(), (SZ, SZ), "full frame resolution");
        let center = loaded.get_pixel(SZ / 2, SZ / 2);
        assert!(center[0] > 200 && center[1] > 200, "saved png has the mask: {center:?}");
        std::fs::remove_dir_all(&dir).ok();
    }
}

impl App {
    // Builds the window + Vulkan state + compositor renderer (with the procedural MSDF atlases).
    fn build_state(&mut self, event_loop: &ActiveEventLoop) -> anyhow::Result<State> {
        let attributes = Window::default_attributes().with_title("LaughingMan");
        let window = Arc::new(event_loop.create_window(attributes).context("creating the window")?);

        let (cap_width, cap_height) = self.source.dimensions();
        let ctx = VkContext::new(&window, &self.app_name)?;
        let size = window.inner_size();
        let extent = vk::Extent2D { width: size.width, height: size.height };
        let swapchain = Swapchain::new(&ctx, extent)?;

        let (mut static_rgba, text_rgba, msdf_size) =
            resolve_atlases(&self.logo_static, &self.logo_text)?;
        // Derive the white face silhouette into the static atlas's alpha; the returned circle centre
        // is the pivot the text ring spins around (offset from the atlas centre by the cap brim).
        let ring_pivot = logo::stamp_silhouette(&mut static_rgba, msdf_size);
        // If a front layer is supplied, mark where it occludes the ring (downgrades the face alpha to
        // the occluder level there). Loaded only for its alpha; not uploaded to the GPU.
        if let Some(front_path) = &self.logo_front {
            let (front_rgba, front_size) = logo::load_atlas(front_path)?;
            if front_size != msdf_size {
                anyhow::bail!("front atlas is {front_size}px but static/text are {msdf_size}px");
            }
            logo::apply_occluder(&mut static_rgba, &front_rgba, msdf_size);
        }
        let renderer = Renderer::new(
            &ctx,
            swapchain.format,
            cap_width,
            cap_height,
            &crate::gpu::renderer::LogoAtlases {
                static_rgba: &static_rgba,
                text_rgba: &text_rgba,
                size: msdf_size,
                ring_pivot,
            },
        )?;

        let now = Instant::now();
        Ok(State {
            renderer,
            swapchain,
            ctx,
            window,
            needs_recreate: false,
            frames: 0,
            last_report: now,
            presented: 0,
            start: now,
            last_frame: now,
            msdf_size,
            static_atlas: static_rgba,
            text_atlas: text_rgba,
            ring_pivot,
            screenshot: false,
        })
    }

    // One frame: capture → (async) detect → track → composite. The render rate is decoupled from
    // detection: each frame submits the latest frame to the detector and applies whatever detections
    // are ready, while the tracker predicts (coasts) every frame. Returns `true` at `max_frames`.
    fn render(
        source: &mut dyn FrameSource,
        detect_in: &SyncSender<DetectionInput>,
        detect_out: &Receiver<Vec<Detection>>,
        tracker: &mut Tracker,
        ring_omega: f32,
        state: &mut State,
        title_index: u32,
        max_frames: Option<u32>,
    ) -> bool {
        if state.needs_recreate {
            let size = state.window.inner_size();
            if size.width == 0 || size.height == 0 {
                return false;
            }
            let extent = vk::Extent2D { width: size.width, height: size.height };
            if let Err(e) = state.swapchain.recreate(&state.ctx, extent) {
                eprintln!("swapchain recreate failed: {e:#}");
                return false;
            }
            state.needs_recreate = false;
        }

        let frame = match source.next_frame() {
            Ok(frame) => frame,
            Err(e) => {
                eprintln!("frame capture failed: {e:#}");
                return false;
            }
        };

        // Timing: elapsed drives the ring/demo; dt drives the tracker.
        let now = Instant::now();
        let elapsed = state.start.elapsed().as_secs_f32();
        let dt = (now - state.last_frame).as_secs_f32().max(1.0e-4);
        state.last_frame = now;

        // Hand the latest frame to the background detector (skipped if it's still busy), and apply
        // any completed detection. On frames without a new result, `tracker.update(&[])` coasts.
        let _ = detect_in.try_send(DetectionInput {
            rgb: ThinArc::from_header_and_slice((), frame.rgb.as_slice()),
            width: frame.width,
            height: frame.height,
            elapsed,
        });
        let detections = detect_out.try_recv().unwrap_or_default();
        let tracks = tracker.update(&detections, dt);

        let ring_phase = ring_omega * elapsed;
        let atlas = state.msdf_size.to_f32();
        let instances: Vec<_> = tracks
            .iter()
            .map(|t| {
                let det = Detection { bbox: t.bbox, score: 1.0, landmarks: t.landmarks };
                face_instance(&det, DEFAULT_COVER_SCALE, atlas, MSDF_RANGE, ring_phase, t.fade)
            })
            .collect();
        let packed = pack_faces_padded(&instances);

        match state.renderer.draw(&state.ctx, &state.swapchain, &frame.rgb, frame.width, frame.height, &packed) {
            Ok(DrawOutcome::Presented) => state.presented += 1,
            Ok(DrawOutcome::NeedRecreate) => state.needs_recreate = true,
            Err(e) => eprintln!("draw failed: {e:#}"),
        }

        if state.screenshot {
            state.screenshot = false;
            let dir = std::path::Path::new("screenshots");
            match save_screenshot(&frame, &instances, &state.static_atlas, &state.text_atlas, state.msdf_size, state.ring_pivot, dir) {
                Ok(path) => println!("saved screenshot {}", path.display()),
                Err(e) => eprintln!("screenshot failed: {e:#}"),
            }
        }

        state.frames += 1;
        let report_elapsed = state.last_report.elapsed();
        if report_elapsed >= FPS_REPORT_INTERVAL {
            let fps = f64::from(state.frames) / report_elapsed.as_secs_f64();
            state.window.set_title(&format!(
                "LaughingMan (cam {title_index}) — {}x{} @ {fps:.0} fps, {} face(s)",
                frame.width,
                frame.height,
                tracks.len()
            ));
            state.frames = 0;
            state.last_report = Instant::now();
        }

        let done = max_frames.is_some_and(|max| state.presented >= max);
        if done {
            println!("presented {} frames; exiting", state.presented);
        }
        done
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        match self.build_state(event_loop) {
            Ok(state) => {
                state.window.request_redraw();
                self.state = Some(state);
            }
            Err(e) => {
                eprintln!("presenter init failed: {e:#}");
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::Escape | KeyCode::Space),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::KeyS),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                if let Some(state) = self.state.as_mut() {
                    state.screenshot = true; // captured on the next rendered frame
                }
            }
            WindowEvent::Resized(_) => {
                if let Some(state) = self.state.as_mut() {
                    state.needs_recreate = true;
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(state) = self.state.as_mut() {
                    let done = Self::render(
                        self.source.as_mut(),
                        &self.detect_in,
                        &self.detect_out,
                        &mut self.tracker,
                        self.ring_omega,
                        state,
                        self.title_index,
                        self.max_frames,
                    );
                    if done {
                        event_loop.exit();
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Drive the next frame ourselves: under ControlFlow::Poll nothing else requests redraws.
        if let Some(state) = self.state.as_ref() {
            state.window.request_redraw();
        }
    }
}
