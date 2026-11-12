//! GPU shaders written in Aspire's `dsl`. The macro lowers the Rust syntax to SPIR-V at compile time.
//! `#[spirv_shader] mod passthrough` expands to `pub const PASSTHROUGH: aspire::ShaderModule`.
//! `.spv` holds the lowered bytes, and `spirv-reader` reflects them to build the descriptor layout.
//!
//! `passthrough` is the original fullscreen-triangle frame sampler, and only tests use it now.
//! `compositor` is the live shader that the renderer builds its pipeline from. It draws the video
//! and the per-face MSDF logo overlay.

use macros::spirv_shader;

/// Fullscreen-triangle passthrough. The fragment samples the bound frame texture at the interpolated UV.
/// In set 0, slot 0 is the frame texture and slot 1 is a linear sampler.
#[spirv_shader]
mod passthrough {
    #[vertex]
    fn vs_main(
        #[vertex_index] i: u32,
        #[position] out_pos: &mut F32Vec4,
        #[location(0)] out_uv: &mut F32Vec2,
    ) {
        // Fullscreen triangle: UVs (0,0),(2,0),(0,2) → clip (-1,-1),(3,-1),(-1,3). Y points down, so the UV origin is top-left.
        let u: f32 = ((i << 1) & 2) as f32;
        let v: f32 = (i & 2) as f32;
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

/// The compositor: samples the video frame and, per detected face, overlays the Laughing Man MSDF
/// logo (static layer + phase-rotated text). Set 0: video(0), sampler(1), static(2), text(3),
/// `Face` storage(4) — std430 stride 36. Per-pixel math mirrors `overlay::composite_pixel`.
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

    // The letterbox transform (window fragment → video pixel, `video_px = (frag - offset) * inv_fit`),
    // the ring pivot, and the logo palette (`overlay::LOGO_INK` / `LOGO_FACE`), packed by `renderer::view_push`.
    struct View {
        inv_fit: f32,
        off_x: f32,
        off_y: f32,
        vid_w: f32,
        vid_h: f32,
        pivot_x: f32,
        pivot_y: f32,
        ink_r: f32,
        ink_g: f32,
        ink_b: f32,
        face_r: f32,
        face_g: f32,
        face_b: f32,
    }

    fn median3(a: f32, b: f32, c: f32) -> f32 {
        return a.max(b).min(a.min(b).max(c));
    }

    #[vertex]
    fn vs_main(#[vertex_index] i: u32, #[position] out_pos: &mut F32Vec4) {
        let u: f32 = ((i << 1) & 2) as f32;
        let v: f32 = (i & 2) as f32;
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
        // Window fragment → video pixel → video UV. This undoes the letterbox that keeps the aspect.
        let vpx_x: f32 = (frag.x - view.off_x) * view.inv_fit;
        let vpx_y: f32 = (frag.y - view.off_y) * view.inv_fit;
        let uv_x: f32 = vpx_x / view.vid_w;
        let uv_y: f32 = vpx_y / view.vid_h;

        let mut color: F32Vec3 = F32Vec3::new(0.0, 0.0, 0.0);
        if uv_x >= 0.0 && uv_x <= 1.0 && uv_y >= 0.0 && uv_y <= 1.0 {
            let base: F32Vec4 = video.sample(samp, F32Vec2::new(uv_x, uv_y));
            color = F32Vec3::new(base.x, base.y, base.z);
        }
        // The logo is blue linework on a white face. Paint the white disc first, then the blue.
        let white: F32Vec3 = F32Vec3::new(view.face_r, view.face_g, view.face_b);
        let blue: F32Vec3 = F32Vec3::new(view.ink_r, view.ink_g, view.ink_b);

        let n: u32 = faces.len();
        let mut i: u32 = 0u32;
        while i < n {
            // Overlay math in video pixels: isotropic (rotation-correct) and aligned to the letterboxed video.
            let dx: f32 = vpx_x - faces[i].center_x;
            let dy: f32 = vpx_y - faces[i].center_y;
            // R(-roll) · (dx, dy) into mask-local pixels, then map to [0, 1] mask UV.
            let lx: f32 = faces[i].roll_cos * dx + faces[i].roll_sin * dy;
            let ly: f32 = (0.0 - faces[i].roll_sin) * dx + faces[i].roll_cos * dy;
            let inv: f32 = 1.0 / (2.0 * faces[i].half_size);
            let mu: f32 = lx * inv + 0.5;
            let mv: f32 = ly * inv + 0.5;
            if mu >= 0.0 && mu <= 1.0 && mv >= 0.0 && mv <= 1.0 {
                let s: F32Vec4 = static_tex.sample(samp, F32Vec2::new(mu, mv));
                // Alpha ~1.0 is the band (ring shows) and ~0.5 the occluder (ring hidden). Both paint white.
                let a: f32 = s.w;
                let white_op: f32 = ((a - 0.25) * 4.0).clamp(0.0, 1.0) * faces[i].fade;
                color = color.mix(white, white_op);

                // Turn the text ring about the logo pivot, not the atlas centre.
                // Show it only in the band, so the front layer hides it.
                let tx: f32 = mu - view.pivot_x;
                let ty: f32 = mv - view.pivot_y;
                let tu: f32 = faces[i].phase_cos * tx - faces[i].phase_sin * ty + view.pivot_x;
                let tv: f32 = faces[i].phase_sin * tx + faces[i].phase_cos * ty + view.pivot_y;
                let t: F32Vec4 = text_tex.sample(samp, F32Vec2::new(tu, tv));
                let text_gate: f32 = ((a - 0.75) * 4.0).clamp(0.0, 1.0);
                let text_cov: f32 = (faces[i].screen_px_range * (median3(t.x, t.y, t.z) - 0.5) + 0.5).clamp(0.0, 1.0);
                color = color.mix(blue, text_cov * text_gate * faces[i].fade);

                let static_cov: f32 = (faces[i].screen_px_range * (median3(s.x, s.y, s.z) - 0.5) + 0.5).clamp(0.0, 1.0);
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
    use spirv_reader::{ComponentType, DescriptorKind, ReadError, Stage};

    #[test]
    fn passthrough_emits_spirv() {
        // The lowered module is non-empty and starts with the little-endian SPIR-V magic word.
        assert!(!PASSTHROUGH.spv.is_empty());
        assert_eq!(&PASSTHROUGH.spv[0..4], &[0x03, 0x02, 0x23, 0x07]);
    }

    #[test]
    fn passthrough_has_vertex_and_fragment_entries() -> Result<(), ReadError> {
        let module: spirv_reader::Module = spirv_reader::read(PASSTHROUGH.spv)?;
        let has = |stage: Stage| module.entry_points().any(|e| e.stage == stage);
        assert!(has(Stage::Vertex), "vertex entry present");
        assert!(has(Stage::Fragment), "fragment entry present");
        Ok(())
    }

    #[test]
    fn passthrough_binds_texture_then_sampler_in_set0() -> Result<(), ReadError> {
        let module: spirv_reader::Module = spirv_reader::read(PASSTHROUGH.spv)?;
        let bindings: Vec<spirv_reader::DescriptorBinding> = module.descriptor_bindings();
        let kind_at = |binding: u32| bindings.iter().find(|b| b.set == 0 && b.binding == binding).map(|b| b.kind);
        assert_eq!(kind_at(0), Some(DescriptorKind::SampledImage), "texture at set 0 binding 0");
        assert_eq!(kind_at(1), Some(DescriptorKind::Sampler), "sampler at set 0 binding 1");
        Ok(())
    }

    #[test]
    fn passthrough_writes_one_float_color_output() -> Result<(), ReadError> {
        let module: spirv_reader::Module = spirv_reader::read(PASSTHROUGH.spv)?;
        let outputs: Vec<spirv_reader::FragmentOutput> = module.fragment_outputs();
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
    fn compositor_has_vertex_and_fragment_entries() -> Result<(), ReadError> {
        let module: spirv_reader::Module = spirv_reader::read(COMPOSITOR.spv)?;
        let has = |stage: Stage| module.entry_points().any(|e| e.stage == stage);
        assert!(has(Stage::Vertex), "vertex entry present");
        assert!(has(Stage::Fragment), "fragment entry present");
        Ok(())
    }

    #[test]
    fn compositor_binds_video_sampler_two_msdfs_and_faces_storage() -> Result<(), ReadError> {
        let module: spirv_reader::Module = spirv_reader::read(COMPOSITOR.spv)?;
        let bindings: Vec<spirv_reader::DescriptorBinding> = module.descriptor_bindings();
        let kind_at = |binding: u32| {
            bindings
                .iter()
                .find(|b| b.set == 0 && b.binding == binding)
                .map(|b| b.kind)
        };
        assert_eq!(kind_at(0), Some(DescriptorKind::SampledImage), "video texture");
        assert_eq!(kind_at(1), Some(DescriptorKind::Sampler));
        assert_eq!(kind_at(2), Some(DescriptorKind::SampledImage), "static MSDF");
        assert_eq!(kind_at(3), Some(DescriptorKind::SampledImage), "text MSDF");
        assert_eq!(kind_at(4), Some(DescriptorKind::StorageBuffer), "faces storage");
        Ok(())
    }

    #[test]
    fn compositor_writes_one_float_color_output() -> Result<(), ReadError> {
        let module: spirv_reader::Module = spirv_reader::read(COMPOSITOR.spv)?;
        let outputs: Vec<spirv_reader::FragmentOutput> = module.fragment_outputs();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].component, ComponentType::Float);
        Ok(())
    }

    #[test]
    fn compositor_declares_the_letterbox_push_constant() -> Result<(), ReadError> {
        let module: spirv_reader::Module = spirv_reader::read(COMPOSITOR.spv)?;
        assert!(module.push_constant_count() > 0, "the View push constant is present");
        Ok(())
    }
}

// GPU render check. It builds a real Vulkan pipeline from the passthrough shader and reads pixels back.
// This proves the Aspire → spirv-reader → ash path. It needs `integration_tests` and skips with no device.
#[cfg(all(test, feature = "integration_tests"))]
mod gpu_tests {
    use super::PASSTHROUGH;
    use crate::num::u32_to_usize;
    use runtime::{Gpu, ImageFormat, ImageInput, RunError, render_texture};

    const EPS: f32 = 0.02;
    const DIM: u32 = 8;
    const TEXELS: usize = 64; // DIM * DIM
    const FULLSCREEN_TRIANGLE: u32 = 3;
    const LANES: usize = 4; // RGBA32F output lanes per pixel.

    // An 8x8 Rgba8 texture split left→right: columns 0..4 red, 4..8 green.
    fn split_texture() -> [[u8; 4]; TEXELS] {
        std::array::from_fn(|i| if i % 8 < 4 { [255, 0, 0, 255] } else { [0, 255, 0, 255] })
    }

    // The RGBA32F float lanes of output pixel (x, y).
    fn pixel_at(pixels: &[f32], x: u32, y: u32) -> [f32; 4] {
        let base: usize = u32_to_usize(y * DIM + x) * LANES;
        [pixels[base], pixels[base + 1], pixels[base + 2], pixels[base + 3]]
    }

    fn close(a: [f32; 4], b: [f32; 4]) -> bool {
        (0..4).all(|i| (a[i] - b[i]).abs() < EPS)
    }

    #[test]
    fn passthrough_samples_frame_spatially_on_gpu() -> Result<(), RunError> {
        assert_eq!(u32_to_usize(DIM * DIM), TEXELS, "TEXELS tracks DIM");
        let gpu: Gpu = match Gpu::new()? {
            Some(gpu) => gpu,
            None => return Ok(()), // No Vulkan device — skip cleanly.
        };
        let texels: [[u8; 4]; 64] = split_texture();
        let out: Vec<u8> = render_texture(
            &gpu,
            PASSTHROUGH.spv,
            ImageInput { format: ImageFormat::Rgba8, width: DIM, height: DIM, bytes: texels.as_flattened() },
            DIM,
            DIM,
            FULLSCREEN_TRIANGLE,
        )?;

        let words: Option<&[[u8; 4]; TEXELS * LANES]> = out.as_chunks::<4>().0.try_into().ok();
        assert!(words.is_some(), "readback is DIM*DIM RGBA32F pixels, got {} bytes", out.len());
        let pixels: [f32; TEXELS * LANES] = words.map_or([0.0; TEXELS * LANES], |w| w.map(f32::from_le_bytes));

        // Corners sample deep inside each region (clamp address mode), so left is red and right green.
        // This shows that the fragment samples the frame by position. A shader with a constant output fails the test.
        let top_left: [f32; 4] = pixel_at(&pixels, 0, 0);
        let top_right: [f32; 4] = pixel_at(&pixels, DIM - 1, 0);
        assert!(close(top_left, [1.0, 0.0, 0.0, 1.0]), "left corner is red: {top_left:?}");
        assert!(close(top_right, [0.0, 1.0, 0.0, 1.0]), "right corner is green: {top_right:?}");
        assert!(!close(top_left, top_right), "output is spatially data-dependent");
        Ok(())
    }
}
