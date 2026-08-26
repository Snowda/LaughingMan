//! GPU shaders authored in Aspire's `dsl` (Rust-syntax lowered to SPIR-V at macro-expansion time).
//! The `#[spirv_shader] mod passthrough` is replaced by the macro with `pub const PASSTHROUGH:
//! aspire::ShaderModule` at this module's scope; `.spv` is the lowered bytes the presenter feeds to
//! `VkShaderModule`, and `spirv-reader` reflects the same bytes to build the descriptor layout.
//!
//! Phase 1 has one module: a fullscreen triangle whose fragment samples the uploaded video frame.
//! The overlay compositor (video + MSDF logo, per-face uniforms) replaces `fs_main` in Phase 4.

use dsl::spirv_shader;

/// Fullscreen-triangle passthrough: the fragment samples the bound frame texture at the UV varying.
/// Set 0 — binding 0: the frame texture; binding 1: a linear sampler.
#[spirv_shader]
mod passthrough {
    #[vertex]
    fn vs_main(
        #[vertex_index] i: u32,
        #[position] out_pos: &mut F32Vec4,
        #[location(0)] out_uv: &mut F32Vec2,
    ) {
        // Fullscreen triangle: UVs (0,0),(2,0),(0,2) → clip (-1,-1),(3,-1),(-1,3). The visible
        // [-1,1] square interpolates UV across [0,1]. Vulkan clip-space y points down, so UV
        // origin (0,0) lands at the framebuffer top-left, matching row 0 of the uploaded frame.
        let u = ((i << 1) & 2) as f32;
        let v = (i & 2) as f32;
        *out_uv = F32Vec2::new(u, v);
        *out_pos = F32Vec4::new(u * 2.0 - 1.0, v * 2.0 - 1.0, 0.0, 1.0);
    }

    #[fragment]
    fn fs_main(
        #[location(0)] in_uv: F32Vec2,
        #[texture(set = 0, binding = 0)] frame: Texture2d,
        #[sampler(set = 0, binding = 1)] samp: Sampler,
        #[location(0)] out_color: &mut F32Vec4,
    ) {
        *out_color = frame.sample(samp, in_uv);
    }
}

/// The compositor: samples the video frame and, for each detected face, overlays the Laughing Man
/// MSDF logo (a static layer + a ring-phase-rotated text layer). Set 0 — binding 0: video texture,
/// 1: linear sampler, 2: static MSDF, 3: text MSDF, 4: a storage buffer of `Face` instances (see
/// `overlay::FaceInstance`; all-scalar so the std430 stride is 36 bytes). The per-pixel math mirrors
/// `overlay::composite_pixel`, which is verified analytically on the CPU.
#[spirv_shader]
mod compositor {
    struct Face {
        center_x: f32,
        center_y: f32,
        half_size: f32,
        roll_cos: f32,
        roll_sin: f32,
        phase_cos: f32,
        phase_sin: f32,
        screen_px_range: f32,
        fade: f32,
    }

    // The letterbox transform (window fragment → video pixel) so the overlay tracks the video
    // regardless of window size: `video_px = (frag - offset) * inv_fit`.
    struct View {
        inv_fit: f32,
        off_x: f32,
        off_y: f32,
        vid_w: f32,
        vid_h: f32,
        pivot_x: f32,
        pivot_y: f32,
    }

    fn median3(a: f32, b: f32, c: f32) -> f32 {
        return a.max(b).min(a.min(b).max(c));
    }

    #[vertex]
    fn vs_main(#[vertex_index] i: u32, #[position] out_pos: &mut F32Vec4) {
        let u = ((i << 1) & 2) as f32;
        let v = (i & 2) as f32;
        *out_pos = F32Vec4::new(u * 2.0 - 1.0, v * 2.0 - 1.0, 0.0, 1.0);
    }

    #[fragment]
    fn fs_main(
        #[frag_coord] frag: F32Vec4,
        #[push_constant] view: View,
        #[texture(set = 0, binding = 0)] video: Texture2d,
        #[sampler(set = 0, binding = 1)] samp: Sampler,
        #[texture(set = 0, binding = 2)] static_tex: Texture2d,
        #[texture(set = 0, binding = 3)] text_tex: Texture2d,
        #[storage(set = 0, binding = 4)] faces: &[Face],
        #[location(0)] out_color: &mut F32Vec4,
    ) {
        // Window fragment → video pixel → video UV (undoing the aspect-preserving letterbox).
        let vpx_x = (frag.x - view.off_x) * view.inv_fit;
        let vpx_y = (frag.y - view.off_y) * view.inv_fit;
        let uv_x = vpx_x / view.vid_w;
        let uv_y = vpx_y / view.vid_h;

        // Black outside the video rect (letterbox bars), else the sampled frame.
        let mut color = F32Vec3::new(0.0, 0.0, 0.0);
        if uv_x >= 0.0 && uv_x <= 1.0 && uv_y >= 0.0 && uv_y <= 1.0 {
            let base = video.sample(samp, F32Vec2::new(uv_x, uv_y));
            color = F32Vec3::new(base.x, base.y, base.z);
        }
        // The logo is blue linework on a white face; paint the white disc first, then the blue.
        let white = F32Vec3::new(1.0, 1.0, 1.0);
        let blue = F32Vec3::new(0.137, 0.286, 0.549);

        let n = faces.len();
        let mut i = 0u32;
        while i < n {
            // Overlay math is in video pixels, so it is isotropic (rotation-correct) and aligned to
            // the video wherever the letterbox places it.
            let dx = vpx_x - faces[i].center_x;
            let dy = vpx_y - faces[i].center_y;
            // R(-roll) · (dx, dy) into mask-local pixels, then map to [0, 1] mask UV.
            let lx = faces[i].roll_cos * dx + faces[i].roll_sin * dy;
            let ly = (0.0 - faces[i].roll_sin) * dx + faces[i].roll_cos * dy;
            let inv = 1.0 / (2.0 * faces[i].half_size);
            let mu = lx * inv + 0.5;
            let mv = ly * inv + 0.5;
            if mu >= 0.0 && mu <= 1.0 && mv >= 0.0 && mv <= 1.0 {
                let s = static_tex.sample(samp, F32Vec2::new(mu, mv));
                // White face background. The static alpha is 3-level: ~1.0 = band (ring shows), ~0.5 =
                // occluder (front layer, ring hidden), 0 = outside. Both non-zero levels paint white.
                let a = s.w;
                let white_op = ((a - 0.25) * 4.0).clamp(0.0, 1.0) * faces[i].fade;
                color = color.mix(white, white_op);

                // Rotating text ring, phase-rotated about the logo's circle centre (the pivot, not the
                // atlas centre which the cap brim shifts). Gated to the band only, so the front layer's
                // white (the occluder level) hides it — the hat reads as in front of the ring.
                let tx = mu - view.pivot_x;
                let ty = mv - view.pivot_y;
                let tu = faces[i].phase_cos * tx - faces[i].phase_sin * ty + view.pivot_x;
                let tv = faces[i].phase_sin * tx + faces[i].phase_cos * ty + view.pivot_y;
                let t = text_tex.sample(samp, F32Vec2::new(tu, tv));
                let text_gate = ((a - 0.75) * 4.0).clamp(0.0, 1.0);
                let text_cov = (faces[i].screen_px_range * (median3(t.x, t.y, t.z) - 0.5) + 0.5).clamp(0.0, 1.0);
                color = color.mix(blue, text_cov * text_gate * faces[i].fade);

                // Static blue linework (rings, features, cap) on top of everything.
                let static_cov = (faces[i].screen_px_range * (median3(s.x, s.y, s.z) - 0.5) + 0.5).clamp(0.0, 1.0);
                color = color.mix(blue, static_cov * faces[i].fade);
            }
            i = i + 1u32;
        }

        *out_color = F32Vec4::new(color.x, color.y, color.z, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::{COMPOSITOR, PASSTHROUGH};
    use spirv_reader::{ComponentType, DescriptorKind, Stage};
    use std::error::Error;

    #[test]
    fn passthrough_emits_spirv() {
        // The lowered module is non-empty and starts with the little-endian SPIR-V magic word.
        assert!(!PASSTHROUGH.spv.is_empty());
        assert_eq!(&PASSTHROUGH.spv[0..4], &[0x03, 0x02, 0x23, 0x07]);
    }

    #[test]
    fn passthrough_has_vertex_and_fragment_entries() -> Result<(), Box<dyn Error>> {
        let module = spirv_reader::read(PASSTHROUGH.spv)?;
        let stages: Vec<Stage> = module.entry_points().iter().map(|e| e.stage).collect();
        assert!(stages.contains(&Stage::Vertex), "vertex entry present");
        assert!(stages.contains(&Stage::Fragment), "fragment entry present");
        Ok(())
    }

    #[test]
    fn passthrough_binds_texture_then_sampler_in_set0() -> Result<(), Box<dyn Error>> {
        let module = spirv_reader::read(PASSTHROUGH.spv)?;
        let bindings = module.descriptor_bindings();
        let tex = bindings
            .iter()
            .find(|b| b.set == 0 && b.binding == 0)
            .ok_or("no texture at set 0 binding 0")?;
        let samp = bindings
            .iter()
            .find(|b| b.set == 0 && b.binding == 1)
            .ok_or("no sampler at set 0 binding 1")?;
        assert_eq!(tex.kind, DescriptorKind::SampledImage);
        assert_eq!(samp.kind, DescriptorKind::Sampler);
        Ok(())
    }

    #[test]
    fn passthrough_writes_one_float_color_output() -> Result<(), Box<dyn Error>> {
        let module = spirv_reader::read(PASSTHROUGH.spv)?;
        let outputs = module.fragment_outputs();
        assert_eq!(outputs.len(), 1, "single color attachment");
        assert_eq!(outputs[0].component, ComponentType::Float);
        Ok(())
    }

    #[test]
    fn compositor_emits_spirv() {
        assert!(!COMPOSITOR.spv.is_empty());
        assert_eq!(&COMPOSITOR.spv[0..4], &[0x03, 0x02, 0x23, 0x07]);
    }

    #[test]
    fn compositor_has_vertex_and_fragment_entries() -> Result<(), Box<dyn Error>> {
        let module = spirv_reader::read(COMPOSITOR.spv)?;
        let stages: Vec<Stage> = module.entry_points().iter().map(|e| e.stage).collect();
        assert!(stages.contains(&Stage::Vertex));
        assert!(stages.contains(&Stage::Fragment));
        Ok(())
    }

    #[test]
    fn compositor_binds_video_sampler_two_msdfs_and_faces_storage() -> Result<(), Box<dyn Error>> {
        let module = spirv_reader::read(COMPOSITOR.spv)?;
        let bindings = module.descriptor_bindings();
        let kind_at = |binding: u32| {
            bindings
                .iter()
                .find(|b| b.set == 0 && b.binding == binding)
                .map(|b| b.kind)
        };
        // Set 0: video(0), sampler(1), static MSDF(2), text MSDF(3), faces storage(4).
        assert_eq!(kind_at(0), Some(DescriptorKind::SampledImage), "video texture");
        assert_eq!(kind_at(1), Some(DescriptorKind::Sampler));
        assert_eq!(kind_at(2), Some(DescriptorKind::SampledImage), "static MSDF");
        assert_eq!(kind_at(3), Some(DescriptorKind::SampledImage), "text MSDF");
        assert_eq!(kind_at(4), Some(DescriptorKind::StorageBuffer), "faces storage");
        Ok(())
    }

    #[test]
    fn compositor_writes_one_float_color_output() -> Result<(), Box<dyn Error>> {
        let module = spirv_reader::read(COMPOSITOR.spv)?;
        let outputs = module.fragment_outputs();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].component, ComponentType::Float);
        Ok(())
    }

    #[test]
    fn compositor_declares_the_letterbox_push_constant() -> Result<(), Box<dyn Error>> {
        let module = spirv_reader::read(COMPOSITOR.spv)?;
        assert!(module.push_constant_count() > 0, "the View push constant is present");
        Ok(())
    }
}

// GPU render verification: builds a real Vulkan pipeline from the passthrough shader (via
// spirv-reader reflection), renders it over a known texture, and reads the pixels back — proving
// the whole Aspire → spirv-reader → ash graphics path produces correct output on hardware. Behind
// the `integration_tests` feature; skips cleanly when no Vulkan device is present.
#[cfg(all(test, feature = "integration_tests"))]
mod gpu_tests {
    use super::PASSTHROUGH;
    use std::error::Error;
    use vk_run::{Gpu, ImageFormat, ImageInput, render_texture};

    const EPS: f32 = 0.02;
    const DIM: u32 = 8;
    const FULLSCREEN_TRIANGLE: u32 = 3;
    const LANES: usize = 4; // RGBA32F output lanes per pixel.

    // An 8x8 Rgba8 texture split left→right: columns 0..4 red, 4..8 green.
    fn split_texture() -> Vec<u8> {
        let mut texels = Vec::with_capacity((DIM * DIM) as usize * 4);
        for _y in 0..DIM {
            for x in 0..DIM {
                if x < DIM / 2 {
                    texels.extend_from_slice(&[255, 0, 0, 255]);
                } else {
                    texels.extend_from_slice(&[0, 255, 0, 255]);
                }
            }
        }
        texels
    }

    // The RGBA32F float lanes of output pixel (x, y).
    fn pixel_at(pixels: &[f32], x: u32, y: u32) -> [f32; 4] {
        let base = ((y * DIM + x) as usize) * LANES;
        [pixels[base], pixels[base + 1], pixels[base + 2], pixels[base + 3]]
    }

    fn close(a: [f32; 4], b: [f32; 4]) -> bool {
        (0..4).all(|i| (a[i] - b[i]).abs() < EPS)
    }

    #[test]
    fn passthrough_samples_frame_spatially_on_gpu() -> Result<(), Box<dyn Error>> {
        let gpu = match Gpu::new()? {
            Some(gpu) => gpu,
            None => return Ok(()), // No Vulkan device — skip cleanly.
        };
        let texels = split_texture();
        let out = render_texture(
            &gpu,
            PASSTHROUGH.spv,
            ImageInput { format: ImageFormat::Rgba8, width: DIM, height: DIM, bytes: &texels },
            DIM,
            DIM,
            FULLSCREEN_TRIANGLE,
        )?;

        let pixels: Vec<f32> = out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(pixels.len(), (DIM * DIM) as usize * LANES);

        // Corners sample deep inside each color region (clamp addressing keeps them pure), so the
        // left corner must be red and the right corner green: the fragment samples the bound frame
        // and passes it through spatially. A constant-output shader would fail the differ check.
        let top_left = pixel_at(&pixels, 0, 0);
        let top_right = pixel_at(&pixels, DIM - 1, 0);
        assert!(close(top_left, [1.0, 0.0, 0.0, 1.0]), "left corner is red: {top_left:?}");
        assert!(close(top_right, [0.0, 1.0, 0.0, 1.0]), "right corner is green: {top_right:?}");
        assert!(!close(top_left, top_right), "output is spatially data-dependent");
        Ok(())
    }
}
