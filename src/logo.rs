//! A procedural stand-in for the Laughing Man logo MSDF atlases, so the compositor runs (and is
//! demoable) without the copyrighted art. The static layer is a stylized face as **linework** — a
//! head outline, two eyes, and a smile — NOT a filled disk: a filled disk would paint a solid blob
//! and (under the shader's `max` compositing) hide the rotating ring entirely. The text layer is a
//! toothed ring whose rotation is visible through the open linework. Each atlas is RGBA8 with the
//! signed distance in every channel, so the shader's `median(r, g, b)` recovers it (0.5 = edge).
//! Real Phase 2 bakes drop in via [`load_atlas`].
#![allow(clippy::as_conversions, clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use std::path::Path;

use anyhow::Context as _;

/// An all-outside (fully transparent) MSDF atlas — every channel encodes "far outside" (0), so it
/// contributes nothing to the composite. Used for the unused text layer when only a single combined
/// logo atlas is supplied.
#[must_use]
pub fn blank_atlas(size: u32) -> Vec<u8> {
    vec![0u8; (size * size * 4) as usize]
}

/// Loads a baked MTSDF atlas PNG as tightly-packed RGBA8 bytes plus its (square) side length. This
/// is the real-logo path: run `laughing-bake` on the Laughing Man SVG, then point the app at the
/// resulting PNGs. Errors if the image is not square.
pub fn load_atlas(path: &Path) -> anyhow::Result<(Vec<u8>, u32)> {
    let img = image::open(path)
        .with_context(|| format!("opening atlas {}", path.display()))?
        .to_rgba8();
    let (w, h) = img.dimensions();
    if w != h {
        anyhow::bail!("atlas {} must be square, got {w}x{h}", path.display());
    }
    Ok((img.into_raw(), w))
}

/// Encodes a signed distance `d` (texels, positive inside) as a `[0, 255]` MSDF channel: 128 at the
/// edge, spread over `range` texels.
fn encode(d: f32, range: f32) -> u8 {
    ((0.5 + d / range).clamp(0.0, 1.0) * 255.0).round() as u8
}

// Fills `size`×`size` RGBA8 by evaluating a positive-inside signed distance per texel.
fn field(size: u32, range: f32, sdf: impl Fn(f32, f32) -> f32) -> Vec<u8> {
    let mut out = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let v = encode(sdf(x as f32 + 0.5, y as f32 + 0.5), range);
            out.extend_from_slice(&[v, v, v, v]);
        }
    }
    out
}

// A stroked-ring signed distance: positive within `half_thickness` of `radius`.
fn ring_band(dist: f32, radius: f32, half_thickness: f32) -> f32 {
    half_thickness - (dist - radius).abs()
}

// A filled-disc signed distance centered at (cx, cy).
fn disc(x: f32, y: f32, cx: f32, cy: f32, r: f32) -> f32 {
    r - (x - cx).hypot(y - cy)
}

/// A stylized face as linework: head outline + two eyes + a smile — the static layer.
#[must_use]
pub fn synthetic_static(size: u32, range: f32) -> Vec<u8> {
    let s = size as f32;
    let c = s * 0.5;
    let t = s * 0.025; // linework half-thickness
    let eye_r = s * 0.05;
    field(size, range, |x, y| {
        let dist = (x - c).hypot(y - c);
        let head = ring_band(dist, s * 0.46, t);
        let left_eye = disc(x, y, c - s * 0.16, c - s * 0.07, eye_r);
        let right_eye = disc(x, y, c + s * 0.16, c - s * 0.07, eye_r);
        // Smile: the lower arc of a circle centered above the middle, kept only below center.
        let smile = if y > c + s * 0.03 {
            ring_band((x - c).hypot(y - (c - s * 0.10)), s * 0.24, t)
        } else {
            -range
        };
        head.max(left_eye).max(right_eye).max(smile)
    })
}

/// A toothed-ring distance field: an annulus present only in `teeth` angular sectors, so rotating it
/// reads as motion — the text layer.
#[must_use]
pub fn synthetic_text(size: u32, range: f32, teeth: u32) -> Vec<u8> {
    let s = size as f32;
    let c = s * 0.5;
    let r_outer = s * 0.40;
    let r_inner = s * 0.30;
    field(size, range, |x, y| {
        let (dx, dy) = (x - c, y - c);
        let dist = (dx * dx + dy * dy).sqrt();
        let ring = (r_outer - dist).min(dist - r_inner);
        if (dy.atan2(dx) * teeth as f32).sin() > 0.0 {
            ring
        } else {
            -range
        }
    })
}

#[cfg(test)]
mod tests {
    use super::{synthetic_static, synthetic_text};

    const SIZE: u32 = 64;
    const RANGE: f32 = 8.0;

    // The MSDF value at texel (x, y) as a fraction (all channels equal here, so channel 0 is it).
    fn value_at(buf: &[u8], x: u32, y: u32) -> f32 {
        f32::from(buf[((y * SIZE + x) * 4) as usize]) / 255.0
    }

    #[test]
    fn static_face_is_open_at_the_center_and_solid_on_the_linework() {
        let buf = synthetic_static(SIZE, RANGE);
        assert_eq!(buf.len(), (SIZE * SIZE * 4) as usize);
        // The center is background (open), NOT a filled blob — this is the whole point.
        assert!(value_at(&buf, SIZE / 2, SIZE / 2) < 0.5, "center is open");
        // A point on the head outline ring (radius 0.46·64 ≈ 29 from center, at x = 61) is inside.
        assert!(value_at(&buf, 61, SIZE / 2) > 0.5, "head outline is drawn");
        // An eye (center ≈ (22, 28)) is inside.
        assert!(value_at(&buf, 22, 28) > 0.5, "eye is drawn");
    }

    #[test]
    fn text_ring_hole_is_outside_band_can_be_inside() {
        let buf = synthetic_text(SIZE, RANGE, 8);
        assert!(value_at(&buf, SIZE / 2, SIZE / 2) < 0.1, "hole is outside");
        let ring_has_inside = (0..SIZE)
            .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
            .any(|(x, y)| value_at(&buf, x, y) > 0.7);
        assert!(ring_has_inside, "at least one tooth reads clearly inside (past the 0.5 edge)");
    }

    #[test]
    fn blank_atlas_is_all_outside() {
        let buf = super::blank_atlas(8);
        assert_eq!(buf.len(), 8 * 8 * 4);
        assert!(buf.iter().all(|&b| b == 0), "every texel is far-outside");
    }

    #[test]
    fn load_atlas_reads_square_and_rejects_nonsquare() -> anyhow::Result<()> {
        let dir = std::env::temp_dir();
        let square = dir.join("laughing_test_atlas_square.png");
        image::RgbaImage::from_pixel(8, 8, image::Rgba([1, 2, 3, 4])).save(&square)?;
        let (rgba, size) = super::load_atlas(&square)?;
        assert_eq!(size, 8);
        assert_eq!(rgba.len(), 8 * 8 * 4);

        let wide = dir.join("laughing_test_atlas_wide.png");
        image::RgbaImage::from_pixel(8, 4, image::Rgba([0, 0, 0, 0])).save(&wide)?;
        assert!(super::load_atlas(&wide).is_err(), "non-square is rejected");
        Ok(())
    }
}
