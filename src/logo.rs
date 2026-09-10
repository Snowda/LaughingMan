//! A procedural stand-in for the Laughing Man logo MSDF atlases, so the compositor runs without the
//! copyrighted art: a **linework** face (outline + eyes + smile, NOT a filled disk that would hide the
//! ring) as the static layer, a toothed ring as the text layer. RGBA8, distance per channel (median, 0.5 = edge).

use std::path::Path;

use anyhow::Context as _;

use crate::num::Cast as _;

/// An all-outside (fully transparent) MSDF atlas — contributes nothing; used for the unused text layer.
#[must_use]
pub fn blank_atlas(size: u32) -> Vec<u8> {
    vec![0u8; (size * size * 4).to_usize()]
}

/// Loads a baked MTSDF atlas PNG as RGBA8 bytes + its square side length. Errors if not square.
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

// Median of three bytes — the MSDF coverage reconstruction (matches the shader's `median(r,g,b)`).
fn median_u8(r: u8, g: u8, b: u8) -> u8 {
    r.max(g).min(r.min(g).max(b))
}

// The face disc is shrunk below the outer radius so its white never peeks past the blue outer ring.
const FACE_DISC_SHRINK: f64 = 0.97;

// Static alpha levels: 255 = band (ring shows through), this = white that OCCLUDES the ring (front layer),
// 0 = outside. The compositor paints white for both non-zero levels but draws text only at the band.
const OCCLUDER_ALPHA: u8 = 128;
// Front coverage is closed by radius `size / this` so the cap's hollow brim fills solid (else the ring shows).
const OCCLUDER_CLOSE_DIV: usize = 20;

/// The filled silhouette of `coverage`: texels not reachable by an exterior flood-fill (barrier-dilated to seal gaps).
fn silhouette_mask(coverage: &[bool], w: usize) -> Vec<bool> {
    let n = w * w;
    let barrier: Vec<bool> = (0..n)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            coverage[i]
                || (x > 0 && coverage[i - 1])
                || (x + 1 < w && coverage[i + 1])
                || (y > 0 && coverage[i - w])
                || (y + 1 < w && coverage[i + w])
        })
        .collect();
    let mut exterior = vec![false; n];
    let mut stack: Vec<usize> = (0..n)
        .filter(|&i| {
            let (x, y) = (i % w, i / w);
            (x == 0 || y == 0 || x + 1 == w || y + 1 == w) && !barrier[i]
        })
        .collect();
    for &i in &stack {
        exterior[i] = true;
    }
    while let Some(i) = stack.pop() {
        let (x, y) = (i % w, i / w);
        let neighbors = [
            (x > 0).then(|| i - 1),
            (x + 1 < w).then(|| i + 1),
            (y > 0).then(|| i - w),
            (y + 1 < w).then(|| i + w),
        ];
        for j in neighbors.into_iter().flatten() {
            if !exterior[j] && !barrier[j] {
                exterior[j] = true;
                stack.push(j);
            }
        }
    }
    (0..n).map(|i| !exterior[i]).collect()
}

/// The logo's face circle in texel space: centre = deepest interior point (BFS), radius = median ray to
/// the boundary. The median discards the cap-brim outlier, so the circle tracks the face, not the hat.
fn face_circle(mask: &[bool], w: usize) -> (f64, f64, f64) {
    let n = w * w;
    let mut dist = vec![u32::MAX; n];
    let mut q = std::collections::VecDeque::new();
    for i in 0..n {
        if !mask[i] {
            dist[i] = 0;
            q.push_back(i);
        }
    }
    while let Some(i) = q.pop_front() {
        let (x, y) = ((i % w).to_i32(), (i / w).to_i32());
        for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)] {
            let (nx, ny) = (x + dx, y + dy);
            if nx >= 0 && ny >= 0 && nx < w.to_i32() && ny < w.to_i32() {
                let j = ny.to_usize() * w + nx.to_usize();
                if dist[j] == u32::MAX {
                    dist[j] = dist[i] + 1;
                    q.push_back(j);
                }
            }
        }
    }
    let center = (0..n).filter(|&i| mask[i]).max_by_key(|&i| dist[i]).unwrap_or(0);
    let (cx, cy) = ((center % w).to_f64(), (center / w).to_f64());
    let mut radii = Vec::with_capacity(360);
    for k in 0..360 {
        let a = f64::from(k) * std::f64::consts::PI / 180.0;
        let (dx, dy) = (a.cos(), a.sin());
        let mut r = 0.0;
        loop {
            let (x, y) = ((cx + dx * r).round().to_i32(), (cy + dy * r).round().to_i32());
            if x < 0 || y < 0 || x >= w.to_i32() || y >= w.to_i32() || !mask[y.to_usize() * w + x.to_usize()] {
                break;
            }
            r += 1.0;
        }
        radii.push(r);
    }
    radii.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    (cx, cy, radii[radii.len() / 2])
}

/// Fills the logo's white face into the atlas alpha (255 inside, 0 outside): the silhouette sealed
/// against a shrunk face disc, so the cap brim fills solid and a filled face is guaranteed. Returns the
/// face-circle centre in atlas UV — the pivot the compositor spins the ring around (off-centre due to the brim).
pub fn stamp_silhouette(rgba: &mut [u8], size: u32) -> (f32, f32) {
    let w = size.to_usize();
    let n = w * w;
    let coverage: Vec<bool> = (0..n)
        .map(|i| median_u8(rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2]) > 127)
        .collect();
    // First pass locates the face circle from the raw silhouette; the brim direction is an outlier.
    let (cx, cy, r) = face_circle(&silhouette_mask(&coverage, w), w);
    let rs = r * FACE_DISC_SHRINK;
    // Seal: union the coverage with the shrunk face disc, then re-fill.
    let mut sealed = coverage;
    for (i, cell) in sealed.iter_mut().enumerate() {
        let (x, y) = ((i % w).to_f64(), (i / w).to_f64());
        if (x - cx).hypot(y - cy) <= rs {
            *cell = true;
        }
    }
    let face = silhouette_mask(&sealed, w);
    for i in 0..n {
        rgba[i * 4 + 3] = if face[i] { 255 } else { 0 };
    }
    (cx.to_f32() / size.to_f32(), cy.to_f32() / size.to_f32())
}

// Separable square dilation of a boolean mask by radius `k` (out-of-bounds treated as unset).
fn dilate(m: &[bool], w: usize, k: usize) -> Vec<bool> {
    let horiz: Vec<bool> = (0..m.len())
        .map(|i| {
            let (x, y) = (i % w, i / w);
            (x.saturating_sub(k)..=(x + k).min(w - 1)).any(|xx| m[y * w + xx])
        })
        .collect();
    (0..m.len())
        .map(|i| {
            let (x, y) = (i % w, i / w);
            (y.saturating_sub(k)..=(y + k).min(w - 1)).any(|yy| horiz[yy * w + x])
        })
        .collect()
}

// Separable square erosion by radius `k` (out-of-bounds treated as set, so borders survive).
fn erode(m: &[bool], w: usize, k: usize) -> Vec<bool> {
    let horiz: Vec<bool> = (0..m.len())
        .map(|i| {
            let (x, y) = (i % w, i / w);
            (x.saturating_sub(k)..=(x + k).min(w - 1)).all(|xx| m[y * w + xx])
        })
        .collect();
    (0..m.len())
        .map(|i| {
            let (x, y) = (i % w, i / w);
            (y.saturating_sub(k)..=(y + k).min(w - 1)).all(|yy| horiz[yy * w + x])
        })
        .collect()
}

/// Downgrades the static face alpha from band (255) to [`OCCLUDER_ALPHA`] wherever the front layer covers
/// it, so the compositor keeps white but stops the text there (front reads in front of the ring). Front
/// coverage is morphologically closed. Call after [`stamp_silhouette`]; both atlases share frame + size.
pub fn apply_occluder(static_rgba: &mut [u8], front_rgba: &[u8], size: u32) {
    let w = size.to_usize();
    let n = w * w;
    let coverage: Vec<bool> = (0..n)
        .map(|i| median_u8(front_rgba[i * 4], front_rgba[i * 4 + 1], front_rgba[i * 4 + 2]) > 127)
        .collect();
    let k = (w / OCCLUDER_CLOSE_DIV).max(4);
    let occluder = erode(&dilate(&coverage, w, k), w, k); // close: fills the cap's hollow interior
    for i in 0..n {
        if occluder[i] && static_rgba[i * 4 + 3] > 127 {
            static_rgba[i * 4 + 3] = OCCLUDER_ALPHA;
        }
    }
}

/// Encodes a signed distance `d` (texels, positive inside) as a `[0,255]` MSDF channel (128 = edge, over `range`).
fn encode(d: f32, range: f32) -> u8 {
    ((0.5 + d / range).clamp(0.0, 1.0) * 255.0).round().to_u8()
}

// Fills `size`×`size` RGBA8 by evaluating a positive-inside signed distance per texel.
fn field(size: u32, range: f32, sdf: impl Fn(f32, f32) -> f32) -> Vec<u8> {
    let mut out = Vec::with_capacity((size * size * 4).to_usize());
    for y in 0..size {
        for x in 0..size {
            let v = encode(sdf(x.to_f32() + 0.5, y.to_f32() + 0.5), range);
            out.extend_from_slice(&[v, v, v, v]);
        }
    }
    out
}

fn ring_band(dist: f32, radius: f32, half_thickness: f32) -> f32 {
    half_thickness - (dist - radius).abs()
}

fn disc(x: f32, y: f32, cx: f32, cy: f32, r: f32) -> f32 {
    r - (x - cx).hypot(y - cy)
}

/// A stylized face as linework: head outline + two eyes + a smile — the static layer.
#[must_use]
pub fn synthetic_static(size: u32, range: f32) -> Vec<u8> {
    let s = size.to_f32();
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
    let s = size.to_f32();
    let c = s * 0.5;
    let r_outer = s * 0.40;
    let r_inner = s * 0.30;
    field(size, range, |x, y| {
        let (dx, dy) = (x - c, y - c);
        let dist = (dx * dx + dy * dy).sqrt();
        let ring = (r_outer - dist).min(dist - r_inner);
        if (dy.atan2(dx) * teeth.to_f32()).sin() > 0.0 {
            ring
        } else {
            -range
        }
    })
}

#[cfg(test)]
mod tests {
    use super::{synthetic_static, synthetic_text};
    use crate::num::Cast as _;

    const SIZE: u32 = 64;
    const RANGE: f32 = 8.0;

    #[test]
    fn apply_occluder_closes_the_hollow_cap_and_downgrades_the_alpha() {
        let n = (SIZE * SIZE).to_usize();
        let mut static_rgba = vec![0u8; n * 4];
        for i in 0..n {
            static_rgba[i * 4 + 3] = 255;
        }
        // Front: a hollow "bar" — two horizontal blue edges 8px apart, no fill between (like the cap).
        let mut front = vec![0u8; n * 4];
        for x in 16..48u32 {
            for y in [28u32, 36] {
                let i = ((y * SIZE + x) * 4).to_usize();
                front[i] = 255;
                front[i + 1] = 255;
                front[i + 2] = 255;
            }
        }
        super::apply_occluder(&mut static_rgba, &front, SIZE);
        let alpha = |x: u32, y: u32| static_rgba[((y * SIZE + x) * 4 + 3).to_usize()];
        // The close bridges the 8px gap → the hollow interior occludes the ring; well outside stays band.
        assert_eq!(alpha(32, 32), super::OCCLUDER_ALPHA, "hollow interior is occluded");
        assert_eq!(alpha(4, 4), 255, "far from the front stays band");
    }

    // The MSDF value at texel (x, y) as a fraction (all channels equal here, so channel 0 is it).
    fn value_at(buf: &[u8], x: u32, y: u32) -> f32 {
        f32::from(buf[((y * SIZE + x) * 4).to_usize()]) / 255.0
    }

    #[test]
    fn static_face_is_open_at_the_center_and_solid_on_the_linework() {
        let buf = synthetic_static(SIZE, RANGE);
        assert_eq!(buf.len(), (SIZE * SIZE * 4).to_usize());
        // Center is open (NOT a filled blob), the head outline and an eye are drawn.
        assert!(value_at(&buf, SIZE / 2, SIZE / 2) < 0.5, "center is open");
        assert!(value_at(&buf, 61, SIZE / 2) > 0.5, "head outline is drawn");
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
    fn silhouette_fills_the_interior_enclosed_by_linework() {
        // The head is a closed ring: flood-fill can't reach the open center, so the silhouette fills it (the "white face").
        let mut buf = super::synthetic_static(SIZE, RANGE);
        super::stamp_silhouette(&mut buf, SIZE);
        let alpha = |x: u32, y: u32| buf[((y * SIZE + x) * 4 + 3).to_usize()];
        assert_eq!(alpha(SIZE / 2, SIZE / 2), 255, "enclosed center is filled silhouette");
        assert_eq!(alpha(1, 1), 0, "far corner is exterior");
    }

    #[test]
    #[ignore = "needs LAUGH_ATLAS=<baked png>; validates the silhouette on the real logo art"]
    fn silhouette_on_the_real_atlas_is_sane() {
        let path = std::env::var("LAUGH_ATLAS").expect("set LAUGH_ATLAS to a baked logo PNG");
        let img = image::open(&path).expect("open atlas").to_rgba8();
        let (w, _h) = img.dimensions();
        let mut buf = img.into_raw();
        let (px, py) = super::stamp_silhouette(&mut buf, w);
        let n = (w * w).to_usize();
        let filled = (0..n).filter(|&i| buf[i * 4 + 3] > 127).count();
        let frac = filled.to_f64() / n.to_f64();
        let center = buf[(((w / 2) * w + w / 2) * 4 + 3).to_usize()];
        eprintln!("silhouette: {:.1}% filled, center alpha={center}, ring pivot=({px:.3}, {py:.3})", frac * 100.0);
        // A sane silhouette fills a meaningful region — neither nothing (no enclosure) nor the whole atlas (a leak).
        assert!((0.2..0.9).contains(&frac), "silhouette fraction {frac}: no-fill or a leak");
        assert_eq!(center, 255, "the logo center is enclosed → filled");
        // The cap brim shifts the fit, so the circle centre (the ring pivot) is off the atlas centre.
        assert!((px - 0.5).abs() > 0.01, "ring pivot x should be off-centre (brim shift): {px}");
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
