//! Writes the static atlas alpha: fills the white face and marks where the front layer hides the ring.

use crate::num::{f64_to_f32, u32_to_f32, u32_to_usize, usize_to_f64};
use bitvec::slice::BitSlice;
use bitvec::vec::BitVec;
use num_traits::ToPrimitive as _;

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
fn silhouette_mask(coverage: &BitSlice, w: usize) -> BitVec {
    let n: usize = w * w;
    let barrier: BitVec = (0..n)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            coverage[i]
                || (x > 0 && coverage[i - 1])
                || (x + 1 < w && coverage[i + 1])
                || (y > 0 && coverage[i - w])
                || (y + 1 < w && coverage[i + w])
        })
        .collect();
    let mut exterior: BitVec = BitVec::repeat(false, n);
    let mut stack: Vec<usize> = (0..n)
        .filter(|&i| {
            let (x, y) = (i % w, i / w);
            (x == 0 || y == 0 || x + 1 == w || y + 1 == w) && !barrier[i]
        })
        .collect();
    for &i in &stack {
        exterior.set(i, true);
    }
    while let Some(i) = stack.pop() {
        let (x, y) = (i % w, i / w);
        let neighbors: [Option<usize>; 4] = [
            (x > 0).then(|| i - 1),
            (x + 1 < w).then(|| i + 1),
            (y > 0).then(|| i - w),
            (y + 1 < w).then(|| i + w),
        ];
        for j in neighbors.into_iter().flatten() {
            if !exterior[j] && !barrier[j] {
                exterior.set(j, true);
                stack.push(j);
            }
        }
    }
    !exterior
}

/// The logo's face circle in texel space: centre = deepest interior point (BFS), radius = median ray to
/// the boundary. The median discards the cap-brim outlier, so the circle tracks the face, not the hat.
fn face_circle(mask: &BitSlice, w: usize) -> (f64, f64, f64) {
    let n: usize = w * w;
    let mut dist: Vec<u32> = vec![u32::MAX; n];
    let mut q: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    for i in 0..n {
        if !mask[i] {
            dist[i] = 0;
            q.push_back(i);
        }
    }
    while let Some(i) = q.pop_front() {
        let (x, y) = (i % w, i / w);
        for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)] {
            let (Some(nx), Some(ny)) = (x.checked_add_signed(dx), y.checked_add_signed(dy)) else { continue };
            if nx < w && ny < w {
                let j: usize = ny * w + nx;
                if dist[j] == u32::MAX {
                    dist[j] = dist[i] + 1;
                    q.push_back(j);
                }
            }
        }
    }
    let center: usize = (0..n).filter(|&i| mask[i]).max_by_key(|&i| dist[i]).unwrap_or(0);
    let (cx, cy) = (usize_to_f64(center % w), usize_to_f64(center / w));
    let mut radii: [f64; 360] = std::array::from_fn(|k| ray_length(mask, w, (cx, cy), k));
    radii.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    (cx, cy, radii[radii.len() / 2])
}

// Whole texels a ray from `(cx, cy)` at `degrees` stays inside `mask` before it leaves.
fn ray_length(mask: &BitSlice, w: usize, (cx, cy): (f64, f64), degrees: usize) -> f64 {
    let a: f64 = usize_to_f64(degrees) * std::f64::consts::PI / 180.0;
    let (dx, dy) = (a.cos(), a.sin());
    let mut r: f64 = 0.0;
    // `to_usize` is None below zero (and on NaN): the ray left the atlas.
    while let (Some(x), Some(y)) = ((cx + dx * r).round().to_usize(), (cy + dy * r).round().to_usize()) {
        if x >= w || y >= w || !mask[y * w + x] {
            break;
        }
        r += 1.0;
    }
    r
}

/// Fills the face (sealed by a shrunk face disc) into alpha as 255/0. Returns the ring pivot in atlas UV.
pub fn stamp_silhouette(rgba: &mut [u8], size: u32) -> (f32, f32) {
    let w: usize = u32_to_usize(size);
    let n: usize = w * w;
    let coverage: BitVec = (0..n)
        .map(|i| median_u8(rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2]) > 127)
        .collect();
    // The first pass finds the face circle from the raw silhouette. The brim direction is an outlier.
    let (cx, cy, r) = face_circle(&silhouette_mask(&coverage, w), w);
    let rs: f64 = r * FACE_DISC_SHRINK;
    // Seal: union the coverage with the shrunk face disc, then re-fill.
    let mut sealed: BitVec = coverage;
    for i in 0..n {
        let (x, y) = (usize_to_f64(i % w), usize_to_f64(i / w));
        if (x - cx).hypot(y - cy) <= rs {
            sealed.set(i, true);
        }
    }
    let face: BitVec = silhouette_mask(&sealed, w);
    for i in 0..n {
        rgba[i * 4 + 3] = if face[i] { 255 } else { 0 };
    }
    (f64_to_f32(cx) / u32_to_f32(size), f64_to_f32(cy) / u32_to_f32(size))
}

// Separable square dilation of a boolean mask by radius `k` (out-of-bounds treated as unset).
fn dilate(m: &BitSlice, w: usize, k: usize) -> BitVec {
    let horiz: BitVec = (0..m.len())
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

// Square erosion by radius `k`, the dual of `dilate` over the same clamped window (so borders survive).
fn erode(m: &BitSlice, w: usize, k: usize) -> BitVec {
    !dilate(&!m.to_bitvec(), w, k)
}

/// Sets face alpha to [`OCCLUDER_ALPHA`] under the closed front coverage. Call after [`stamp_silhouette`].
pub fn apply_occluder(static_rgba: &mut [u8], front_rgba: &[u8], size: u32) {
    let w: usize = u32_to_usize(size);
    let n: usize = w * w;
    let coverage: BitVec = (0..n)
        .map(|i| median_u8(front_rgba[i * 4], front_rgba[i * 4 + 1], front_rgba[i * 4 + 2]) > 127)
        .collect();
    let k: usize = (w / OCCLUDER_CLOSE_DIV).max(4);
    let occluder: BitVec = erode(&dilate(&coverage, w, k), w, k); // close: fills the cap's hollow interior
    for i in 0..n {
        if occluder[i] && static_rgba[i * 4 + 3] > 127 {
            static_rgba[i * 4 + 3] = OCCLUDER_ALPHA;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        apply_occluder, dilate, erode, face_circle, median_u8, silhouette_mask, stamp_silhouette, OCCLUDER_ALPHA,
    };
    use crate::logo::synthetic_static;
    use crate::num::{u32_to_usize, usize_to_f64};
    use bitvec::slice::BitSlice;
    use bitvec::vec::BitVec;

    const SIZE: u32 = 64;
    const RANGE: f32 = 8.0;

    fn mask_from(w: usize, pred: impl Fn(usize, usize) -> bool) -> BitVec {
        (0..w * w).map(|i| pred(i % w, i / w)).collect()
    }

    /// The filled square `lo..=hi` on both axes.
    fn square(w: usize, lo: usize, hi: usize) -> BitVec {
        let span: std::ops::RangeInclusive<usize> = lo..=hi;
        mask_from(w, |x, y| span.contains(&x) && span.contains(&y))
    }

    /// The perimeter of [`square`], minus the texels in `gaps`.
    fn square_ring(w: usize, lo: usize, hi: usize, gaps: &[(usize, usize)]) -> BitVec {
        let inner: BitVec = square(w, lo + 1, hi - 1);
        square(w, lo, hi)
            .into_iter()
            .zip(inner)
            .enumerate()
            .map(|(i, (outer, inside))| outer && !inside && !gaps.contains(&(i % w, i / w)))
            .collect()
    }

    fn dist(x: usize, y: usize, c: f64) -> f64 {
        (usize_to_f64(x) - c).hypot(usize_to_f64(y) - c)
    }

    fn count(mask: &BitSlice) -> usize {
        mask.count_ones()
    }

    fn alpha_at(rgba: &[u8], w: usize, x: usize, y: usize) -> u8 {
        rgba[(y * w + x) * 4 + 3]
    }

    /// An RGBA atlas whose median channel is white exactly where `mask` is set, alpha 255 everywhere.
    fn rgba_from(mask: &BitSlice) -> Vec<u8> {
        mask.iter().by_vals().flat_map(|m| if m { [255; 4] } else { [0, 0, 0, 255] }).collect()
    }

    #[test]
    fn median_u8_matches_a_sort_oracle() {
        let levels: [u8; 5] = [0u8, 1, 2, 128, 255];
        for r in levels {
            for g in levels {
                for b in levels {
                    let mut sorted: [u8; 3] = [r, g, b];
                    sorted.sort_unstable();
                    assert_eq!(median_u8(r, g, b), sorted[1], "median of ({r}, {g}, {b})");
                }
            }
        }
    }

    #[test]
    fn silhouette_mask_fills_a_closed_ring() {
        let w: usize = 16;
        let face: BitVec = silhouette_mask(&square_ring(w, 4, 11, &[]), w);
        // Texels 4-adjacent to the ring are barrier, so the fill also holds a 1-texel halo (no corners).
        let halo: BitVec = mask_from(w, |x, y| (4..=11).contains(&x) || (4..=11).contains(&y));
        let expect: BitVec = square(w, 3, 12).into_iter().zip(halo).map(|(s, h)| s && h).collect();
        assert_eq!(face, expect, "ring, enclosed interior and barrier halo; nothing else");
    }

    #[test]
    fn silhouette_mask_seals_a_two_texel_gap() {
        let w: usize = 16;
        let face: BitVec = silhouette_mask(&square_ring(w, 4, 11, &[(7, 4), (8, 4)]), w);
        assert!(face[8 * w + 8], "barrier dilation bridges a 2-texel gap, so the interior stays filled");
    }

    #[test]
    fn silhouette_mask_leaks_through_a_three_texel_gap() {
        let w: usize = 16;
        let face: BitVec = silhouette_mask(&square_ring(w, 4, 11, &[(7, 4), (8, 4), (9, 4)]), w);
        assert!(!face[8 * w + 8], "the middle gap texel has no covered neighbour, so the flood fill leaks in");
    }

    #[test]
    fn silhouette_mask_of_empty_or_full_coverage_is_unchanged() {
        for fill in [false, true] {
            let uniform: BitVec = BitVec::<usize>::repeat(fill, 64);
            assert_eq!(silhouette_mask(&uniform, 8), uniform, "uniform coverage {fill} maps to itself");
        }
    }

    #[test]
    fn face_circle_finds_a_disc_centre_and_radius() {
        let (cx, cy, r) = face_circle(&mask_from(48, |x, y| dist(x, y, 20.0) <= 10.0), 48);
        assert!((cx - 20.0).abs() <= 1.0 && (cy - 20.0).abs() <= 1.0, "centre ({cx}, {cy}) is the disc centre");
        assert!((r - 10.0).abs() <= 1.5, "radius {r} matches the disc radius");
    }

    #[test]
    fn face_circle_radius_ignores_a_brim_outlier() {
        // A disc (r=10 at 20,20) plus a wide brim that goes right to the atlas edge.
        let hat: BitVec = mask_from(48, |x, y| dist(x, y, 20.0) <= 10.0 || (x >= 28 && (14..=26).contains(&y)));
        let (cx, cy, r) = face_circle(&hat, 48);
        assert!((cx - 20.0).hypot(cy - 20.0) <= 2.0, "centre ({cx}, {cy}) stays in the disc, not the brim");
        assert!(r <= 12.5, "median radius {r} tracks the disc; a max or mean would be pulled toward the brim");
    }

    #[test]
    fn face_circle_of_empty_mask_is_the_origin() {
        let (cx, cy, r) = face_circle(&BitVec::<usize>::repeat(false, 64), 8);
        assert!(cx.abs() + cy.abs() + r.abs() < 1e-9, "empty mask falls back to (0, 0, 0): ({cx}, {cy}, {r})");
    }

    #[test]
    fn dilate_grows_a_point_into_a_square() {
        assert_eq!(dilate(&square(12, 5, 5), 12, 2), square(12, 3, 7), "radius-2 square dilation is a 5x5 block");
    }

    #[test]
    fn dilate_clips_at_the_border() {
        assert_eq!(count(&dilate(&square(12, 0, 0), 12, 2)), 9, "only the in-bounds 3x3 quarter is set");
    }

    #[test]
    fn dilate_by_zero_is_identity() {
        let ring: BitVec = square_ring(10, 2, 7, &[(4, 2)]);
        assert_eq!(dilate(&ring, 10, 0), ring, "k = 0 leaves the mask unchanged");
    }

    #[test]
    fn erode_shrinks_a_square_by_k() {
        assert_eq!(erode(&square(12, 3, 8), 12, 1), square(12, 4, 7), "radius-1 erosion trims one texel from each side");
    }

    #[test]
    fn erode_keeps_a_full_mask_full() {
        assert_eq!(count(&erode(&BitVec::<usize>::repeat(true, 36), 6, 2)), 36, "the clamped window means borders survive erosion");
    }

    #[test]
    fn erode_removes_an_isolated_point() {
        assert_eq!(count(&erode(&square(8, 4, 4), 8, 1)), 0, "a lone texel cannot survive a radius-1 erosion");
    }

    /// Front coverage of two horizontal 1-texel bars at rows `top` and `bottom`, over `x0..x1`.
    fn bars(w: usize, top: usize, bottom: usize, x0: usize, x1: usize) -> Vec<u8> {
        rgba_from(&mask_from(w, |x, y| (x0..x1).contains(&x) && (y == top || y == bottom)))
    }

    /// Runs [`apply_occluder`] over a fully opaque static atlas and returns it.
    fn occlude_opaque(w: usize, front: &[u8]) -> Vec<u8> {
        let mut static_rgba: Vec<u8> = vec![255u8; w * w * 4];
        apply_occluder(&mut static_rgba, front, u32::try_from(w).unwrap_or(0));
        static_rgba
    }

    #[test]
    fn apply_occluder_does_not_bridge_a_gap_wider_than_twice_the_radius() {
        // Size 64 → k = max(64/20, 4) = 4, so the close bridges at most 8 empty rows. Here there are 10.
        let out: Vec<u8> = occlude_opaque(64, &bars(64, 28, 39, 16, 48));
        assert_eq!(alpha_at(&out, 64, 32, 28), OCCLUDER_ALPHA, "the bar itself occludes");
        assert_eq!(alpha_at(&out, 64, 32, 33), 255, "the wide gap stays band");
    }

    #[test]
    fn apply_occluder_radius_scales_with_atlas_size() {
        // size 200 → k = 200/20 = 10, wide enough to bridge 15 empty rows (k = 4 would not).
        let out: Vec<u8> = occlude_opaque(200, &bars(200, 90, 106, 50, 150));
        assert_eq!(alpha_at(&out, 200, 100, 98), OCCLUDER_ALPHA, "the close radius grows with size and fills the gap");
    }

    #[test]
    fn apply_occluder_leaves_exterior_and_colour_untouched() {
        // Exterior (alpha 0) everywhere, with a distinct colour so a stray write would show.
        let before: [[u8; 4]; 4096] = [[10u8, 20, 30, 0]; 64 * 64];
        let mut static_rgba: [[u8; 4]; 4096] = before;
        apply_occluder(static_rgba.as_flattened_mut(), &bars(64, 28, 36, 16, 48), SIZE);
        assert!(static_rgba == before, "only face texels (alpha > 127) may become occluders");
    }

    #[test]
    fn stamp_silhouette_pivot_is_the_centre_of_a_symmetric_head() {
        let mut buf: Vec<u8> = synthetic_static(SIZE, RANGE);
        let (px, py) = stamp_silhouette(&mut buf, SIZE);
        assert!((px - 0.5).hypot(py - 0.5) < 0.04, "a round head pivots at the atlas centre: ({px}, {py})");
    }

    #[test]
    fn stamp_silhouette_writes_only_alpha() {
        fn rgb(buf: &[u8]) -> impl Iterator<Item = [u8; 3]> + '_ {
            buf.as_chunks::<4>().0.iter().map(|&[r, g, b, _]| [r, g, b])
        }
        let before: Vec<u8> = synthetic_static(SIZE, RANGE);
        let mut buf: Vec<u8> = synthetic_static(SIZE, RANGE);
        stamp_silhouette(&mut buf, SIZE);
        assert!(rgb(&buf).eq(rgb(&before)), "the MSDF colour channels are preserved");
        assert!(buf.chunks_exact(4).all(|p| p[3] % 255 == 0), "alpha is binary 0/255");
    }

    #[test]
    fn stamp_silhouette_seals_a_leaky_compartment_inside_the_face_disc() {
        // A head outline with a chord at y=44. A 6-texel gap below the chord makes that part leak.
        // The face disc must fill the leaked texels that it covers.
        let head: BitVec = mask_from(64, |x, y| {
            let d: f64 = dist(x, y, 32.0);
            let outline: bool = (25.0..=27.0).contains(&d) && !((29..=34).contains(&x) && y > 50);
            outline || (y == 44 && d < 26.0)
        });
        // Two rows below the chord, clear of its barrier dilation.
        assert!(!silhouette_mask(&head, 64)[47 * 64 + 32], "precondition: below the chord leaks in the raw silhouette");
        let mut buf: Vec<u8> = rgba_from(&head);
        stamp_silhouette(&mut buf, SIZE);
        let alpha = |y| alpha_at(&buf, 64, 32, y);
        assert_eq!(alpha(47), 255, "the face disc seals the texel just below the chord");
        assert_eq!(alpha(32), 255, "the closed compartment stays filled");
        assert_eq!(alpha(62), 0, "the gap's outside stays exterior");
    }

    #[test]
    fn apply_occluder_closes_the_hollow_cap_and_downgrades_the_alpha() {
        let n: usize = u32_to_usize(SIZE * SIZE);
        let mut static_rgba: Vec<u8> = vec![0u8; n * 4];
        for i in 0..n {
            static_rgba[i * 4 + 3] = 255;
        }
        // Front: a hollow "bar" — two horizontal blue edges 8px apart, no fill between (like the cap).
        let mut front: Vec<u8> = vec![0u8; n * 4];
        for x in 16..48u32 {
            for y in [28u32, 36] {
                let i: usize = u32_to_usize((y * SIZE + x) * 4);
                front[i] = 255;
                front[i + 1] = 255;
                front[i + 2] = 255;
            }
        }
        apply_occluder(&mut static_rgba, &front, SIZE);
        let alpha = |x: u32, y: u32| static_rgba[u32_to_usize((y * SIZE + x) * 4 + 3)];
        // The close bridges the 8px gap, so the hollow interior hides the ring. Far outside stays band.
        assert_eq!(alpha(32, 32), OCCLUDER_ALPHA, "hollow interior is occluded");
        assert_eq!(alpha(4, 4), 255, "far from the front stays band");
    }

    #[test]
    fn silhouette_fills_the_interior_enclosed_by_linework() {
        // The head is a closed ring: flood-fill can't reach the open center, so the silhouette fills it (the "white face").
        let mut buf: Vec<u8> = synthetic_static(SIZE, RANGE);
        stamp_silhouette(&mut buf, SIZE);
        let alpha = |x: u32, y: u32| buf[u32_to_usize((y * SIZE + x) * 4 + 3)];
        assert_eq!(alpha(SIZE / 2, SIZE / 2), 255, "enclosed center is filled silhouette");
        assert_eq!(alpha(1, 1), 0, "far corner is exterior");
    }

    #[test]
    #[ignore = "needs LAUGH_ATLAS=<baked png>; validates the silhouette on the real logo art"]
    fn silhouette_on_the_real_atlas_is_sane() -> crate::error::Result<()> {
        use crate::error::Context as _;
        let path: String = std::env::var("LAUGH_ATLAS").context("set LAUGH_ATLAS to a baked logo PNG")?;
        let img: image::RgbaImage = image::open(&path).context("open atlas")?.to_rgba8();
        let (w, _h) = img.dimensions();
        let mut buf: Vec<u8> = img.into_raw();
        let (px, py) = stamp_silhouette(&mut buf, w);
        let n: usize = u32_to_usize(w * w);
        let filled: usize = (0..n).filter(|&i| buf[i * 4 + 3] > 127).count();
        let frac: f64 = usize_to_f64(filled) / usize_to_f64(n);
        let center: u8 = buf[u32_to_usize(((w / 2) * w + w / 2) * 4 + 3)];
        eprintln!("silhouette: {:.1}% filled, center alpha={center}, ring pivot=({px:.3}, {py:.3})", frac * 100.0);
        // A sane silhouette fills a meaningful region — neither nothing (no enclosure) nor the whole atlas (a leak).
        assert!((0.2..0.9).contains(&frac), "silhouette fraction {frac}: no-fill or a leak");
        assert_eq!(center, 255, "the logo center is enclosed → filled");
        // The cap brim shifts the fit, so the circle centre (the ring pivot) is off the atlas centre.
        assert!((px - 0.5).abs() > 0.01, "ring pivot x should be off-centre (brim shift): {px}");
        Ok(())
    }
}
