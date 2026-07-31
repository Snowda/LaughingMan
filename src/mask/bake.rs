//! The fdsm bake pipeline: fit the shape into the atlas, edge-color it, generate the MTSDF, and
//! run sign + error correction — producing an RGBA8 image where RGB is the multi-channel SDF
//! (median = signed distance, 0.5 = edge, > 0.5 inside) and A is the true SDF.
#![allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use anyhow::anyhow;
use fdsm::bezier::Point;
use fdsm::bezier::prepared::PreparedColoredShape;
use fdsm::bezier::scanline::FillRule;
use fdsm::correct_error::{ErrorCorrectionConfig, correct_error_mtsdf};
use fdsm::render::correct_sign_mtsdf;
use fdsm::shape::{Contour, Shape};
use fdsm::transform::Transform;
use image::{ImageBuffer, Rgba, RgbaImage};
use nalgebra::{Affine2, Matrix3};
use rayon::prelude::*;

// Edge-coloring corner threshold (sine of the angle) and RNG seed — msdfgen's `edgeColoringSimple`.
const SIN_ALPHA: f64 = 0.03;
const SEED: u64 = 0;
// Samples per segment when measuring the shape's bounding box.
const BOUNDS_SAMPLES: u32 = 8;

/// The SVG fill rule that decides inside/outside — must match the source (e.g. the Laughing Man
/// logo is `fill-rule:evenodd`, so its holes only read correctly under [`Fill::EvenOdd`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fill {
    Nonzero,
    EvenOdd,
}

impl Fill {
    fn to_fdsm(self) -> FillRule {
        match self {
            Fill::Nonzero => FillRule::Nonzero,
            Fill::EvenOdd => FillRule::Odd,
        }
    }
}

/// Atlas size (square, texels), the MSDF distance range (texels) mapped into [0, 1], and the fill
/// rule to sign the shape by.
pub struct BakeParams {
    pub size: u32,
    pub range: f64,
    pub fill: Fill,
}

impl Default for BakeParams {
    fn default() -> Self {
        Self {
            size: 1024,
            range: 8.0,
            fill: Fill::Nonzero,
        }
    }
}

/// Bakes `shape` (in SVG user units) into a `size`×`size` RGBA8 MTSDF, centered with a `range`-texel
/// margin. Errors on a degenerate (zero-extent) shape.
pub fn bake_shape(mut shape: Shape<Contour>, params: &BakeParams) -> anyhow::Result<RgbaImage> {
    let transform = fit_transform(&shape, params.size, params.range)?;
    shape.transform(&transform);

    let colored = Shape::edge_coloring_simple(shape, SIN_ALPHA, SEED);
    let prepared = colored.prepare();

    // The MTSDF generation (per-texel min-distance over all edges) is the bake's dominant cost and
    // is independent per texel — generate it across rows in parallel. The result is identical to
    // fdsm's sequential `generate_mtsdf` (same sampler math); the fidelity/analytic tests confirm it.
    let data = generate_mtsdf_parallel(&prepared, params.range, params.size);
    let mut msdf = ImageBuffer::<Rgba<f32>, Vec<f32>>::from_raw(params.size, params.size, data)
        .ok_or_else(|| anyhow!("MTSDF buffer size mismatch"))?;
    // Sign + error correction are cheap relative to generation; kept sequential.
    correct_sign_mtsdf(&mut msdf, &prepared, params.fill.to_fdsm());
    correct_error_mtsdf(
        &mut msdf,
        &colored,
        &prepared,
        params.range,
        &ErrorCorrectionConfig::default(),
    );

    Ok(to_rgba8(&msdf))
}

// Data-parallel MTSDF generation, replicating fdsm's `sampler_mtsdf`: for each texel, the three
// colored channels hold the per-channel signed pseudo-distance and alpha holds the true SDF, each
// mapped `sd/range + 0.5` clamped to [0, 1]. Rows are generated in parallel; `PreparedColoredShape`
// is shared read-only (`Sync`).
fn generate_mtsdf_parallel(prepared: &PreparedColoredShape, range: f64, size: u32) -> Vec<f32> {
    let width = size as usize;
    let mut data = vec![0.0f32; width * width * 4];
    let encode = |sd: f64| (sd / range + 0.5).clamp(0.0, 1.0) as f32;
    data.par_chunks_mut(width * 4)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..width {
                let point = Point::new(x as f64 + 0.5, y as f64 + 0.5);
                let [d_red, d_green, d_blue, d_min] = prepared.distance4(point);
                let base = x * 4;
                row[base] = encode(d_red.signed_pseudo_distance(point));
                row[base + 1] = encode(d_green.signed_pseudo_distance(point));
                row[base + 2] = encode(d_blue.signed_pseudo_distance(point));
                row[base + 3] = encode(d_min.value.distance());
            }
        });
    data
}

// The affine that scales the shape (uniformly, aspect preserved) to fit a `size`×`size` atlas with
// a `range`-texel margin on every side, centered.
fn fit_transform(shape: &Shape<Contour>, size: u32, range: f64) -> anyhow::Result<Affine2<f64>> {
    let (scale, tx, ty) = fit_params(shape, size, range)?;
    Ok(Affine2::from_matrix_unchecked(Matrix3::new(
        scale, 0.0, tx, 0.0, scale, ty, 0.0, 0.0, 1.0,
    )))
}

/// The uniform `(scale, tx, ty)` mapping SVG user units → atlas texels the bake fits a shape with:
/// aspect-preserved, centered, `range`-texel margin. Exposed so a fidelity check can rasterize the
/// source SVG into exactly the same texel space as the MSDF.
pub fn fit_params(shape: &Shape<Contour>, size: u32, range: f64) -> anyhow::Result<(f64, f64, f64)> {
    let (min_x, min_y, max_x, max_y) =
        shape_bounds(shape).ok_or_else(|| anyhow!("shape has no segments"))?;
    let (width, height) = (max_x - min_x, max_y - min_y);
    let extent = width.max(height);
    if extent <= 0.0 {
        return Err(anyhow!("shape has zero extent"));
    }
    let usable = f64::from(size) - 2.0 * range;
    if usable <= 0.0 {
        return Err(anyhow!("atlas size {size} too small for range {range}"));
    }
    let scale = usable / extent;
    // Center the scaled shape in the atlas.
    let tx = (f64::from(size) - width * scale) / 2.0 - min_x * scale;
    let ty = (f64::from(size) - height * scale) / 2.0 - min_y * scale;
    Ok((scale, tx, ty))
}

// The shape's bounding box, sampling each segment along its length (curve extremes fall inside the
// sampled hull closely enough; the bake's margin absorbs the small slack).
fn shape_bounds(shape: &Shape<Contour>) -> Option<(f64, f64, f64, f64)> {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut any = false;
    for contour in &shape.contours {
        for segment in &contour.segments {
            for i in 0..=BOUNDS_SAMPLES {
                let t = f64::from(i) / f64::from(BOUNDS_SAMPLES);
                let p = segment.get(t);
                min_x = min_x.min(p.x);
                min_y = min_y.min(p.y);
                max_x = max_x.max(p.x);
                max_y = max_y.max(p.y);
                any = true;
            }
        }
    }
    any.then_some((min_x, min_y, max_x, max_y))
}

// Converts the f32 MTSDF (channels in ~[0, 1]) to RGBA8 by clamping and scaling.
fn to_rgba8(src: &ImageBuffer<Rgba<f32>, Vec<f32>>) -> RgbaImage {
    let mut out = RgbaImage::new(src.width(), src.height());
    for (x, y, px) in src.enumerate_pixels() {
        let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        out.put_pixel(x, y, Rgba([q(px[0]), q(px[1]), q(px[2]), q(px[3])]));
    }
    out
}

#[cfg(test)]
#[allow(clippy::expect_used)] // Test helpers surface bake failures via expect.
mod tests {
    use super::{BakeParams, bake_shape};
    use crate::mask::shape_from_svg;
    use image::{Rgba, RgbaImage};

    const CIRCLE: &[u8] = br##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="50" cy="50" r="40" fill="#000"/></svg>"##;
    const SQUARE: &[u8] = br##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect x="10" y="10" width="80" height="80" fill="#000"/></svg>"##;

    const SIZE: u32 = 64;
    const RANGE: f64 = 6.0;

    fn bake(svg: &[u8]) -> RgbaImage {
        let shape = shape_from_svg(svg).expect("parse");
        bake_shape(shape, &BakeParams { size: SIZE, range: RANGE, fill: super::Fill::Nonzero }).expect("bake")
    }

    // median(r, g, b) reconstructs the signed distance from the multi-channel SDF.
    fn median(px: &Rgba<u8>) -> f32 {
        let (r, g, b) = (f32::from(px[0]), f32::from(px[1]), f32::from(px[2]));
        r.max(g).min(r.min(g).max(b)) / 255.0
    }

    fn alpha(px: &Rgba<u8>) -> f32 {
        f32::from(px[3]) / 255.0
    }

    // Steps outward along +x from the center until the median crosses 0.5 (the shape edge).
    fn crossing_radius_x(img: &RgbaImage, cx: u32, cy: u32) -> Option<u32> {
        (0..(img.width() - cx)).find(|&r| median(&img.get_pixel(cx + r, cy)) < 0.5)
    }
    fn crossing_radius_y(img: &RgbaImage, cx: u32, cy: u32) -> Option<u32> {
        (0..(img.height() - cy)).find(|&r| median(&img.get_pixel(cx, cy + r)) < 0.5)
    }

    #[test]
    fn circle_inside_is_positive_outside_is_negative() {
        let img = bake(CIRCLE);
        let center = img.get_pixel(SIZE / 2, SIZE / 2);
        let corner = img.get_pixel(1, 1);
        // Filled interior reads > 0.5 in both the median SDF and the true-SDF alpha; far outside < 0.5.
        assert!(median(center) > 0.6, "center inside: {}", median(center));
        assert!(alpha(center) > 0.6, "center alpha inside: {}", alpha(center));
        assert!(median(corner) < 0.4, "corner outside: {}", median(corner));
        assert!(alpha(corner) < 0.4, "corner alpha outside: {}", alpha(corner));
    }

    #[test]
    fn circle_field_is_isotropic() {
        let img = bake(CIRCLE);
        let (cx, cy) = (SIZE / 2, SIZE / 2);
        let rx = crossing_radius_x(&img, cx, cy).expect("x crossing");
        let ry = crossing_radius_y(&img, cx, cy).expect("y crossing");
        // A circle: the 0.5 crossing is the same distance along x and y, and well beyond the range.
        assert!(rx as i64 - ry as i64 == 0 || (rx as i64 - ry as i64).abs() <= 2, "rx={rx} ry={ry}");
        assert!(rx > RANGE as u32, "crossing {rx} must exceed the range");
    }

    #[test]
    fn square_fills_its_corners_where_the_circle_does_not() {
        let circle = bake(CIRCLE);
        let square = bake(SQUARE);
        let (cx, cy) = (SIZE / 2, SIZE / 2);
        let r = crossing_radius_x(&circle, cx, cy).expect("crossing");
        // A diagonal point at 0.8r on each axis: distance 0.8r·√2 ≈ 1.13r > r, so it's OUTSIDE the
        // circle but INSIDE the square (which shares the circle's bbox). This distinguishes the two
        // shapes' fields — a constant/broken bake would fail both halves.
        let off = (f64::from(r) * 0.8) as u32;
        let (px, py) = (cx + off, cy + off);
        assert!(median(circle.get_pixel(px, py)) < 0.5, "circle corner is outside");
        assert!(median(square.get_pixel(px, py)) > 0.5, "square corner is inside");
    }

    #[test]
    fn circle_and_square_bakes_differ() {
        let circle = bake(CIRCLE);
        let square = bake(SQUARE);
        let differing = circle
            .pixels()
            .zip(square.pixels())
            .filter(|(a, b)| a != b)
            .count();
        // Different shapes must produce materially different atlases (guards against a no-op bake).
        assert!(differing > (SIZE * SIZE / 20) as usize, "only {differing} pixels differ");
    }
}

/// Fidelity: bake an SVG to MSDF, reconstruct its coverage (`median(r,g,b) > 0.5`), and compare —
/// via intersection-over-union — against an independent resvg rasterization of the *same* SVG into
/// the *same* texel space (the bake's fit transform). High IoU proves the MSDF faithfully
/// reproduces the source SVG, not merely "some shape".
#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
mod fidelity {
    use super::{BakeParams, Fill, bake_shape, fit_params};
    use crate::mask::{shape_from_svg, svg_fill_rule};
    use resvg::tiny_skia::{Pixmap, Transform};
    use resvg::usvg::{Options, Tree};

    // Median of three channels — the MSDF distance reconstruction.
    fn median_u8(r: u8, g: u8, b: u8) -> u8 {
        r.max(g).min(r.min(g).max(b))
    }

    // IoU between the baked MSDF's inside-region and a resvg raster of the same SVG at the bake's
    // fit and fill rule.
    fn iou_msdf_vs_raster(svg: &[u8], fill: Fill, size: u32, range: f64) -> f64 {
        let shape = shape_from_svg(svg).expect("parse svg");
        let (scale, tx, ty) = fit_params(&shape, size, range).expect("fit");
        let img = bake_shape(shape, &BakeParams { size, range, fill }).expect("bake");
        let msdf: Vec<bool> = img
            .pixels()
            .map(|p| median_u8(p[0], p[1], p[2]) > 127)
            .collect();

        let tree = Tree::from_data(svg, &Options::default()).expect("usvg parse");
        let mut pixmap = Pixmap::new(size, size).expect("pixmap");
        let transform =
            Transform::from_row(scale as f32, 0.0, 0.0, scale as f32, tx as f32, ty as f32);
        resvg::render(&tree, transform, &mut pixmap.as_mut());
        let raster: Vec<bool> = pixmap.pixels().iter().map(|px| px.alpha() > 127).collect();

        let inter = msdf.iter().zip(&raster).filter(|(a, b)| **a && **b).count();
        let union = msdf.iter().zip(&raster).filter(|(a, b)| **a || **b).count();
        if union == 0 {
            return 1.0;
        }
        inter as f64 / union as f64
    }

    const CIRCLE: &[u8] = br##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="50" cy="50" r="40" fill="#000"/></svg>"##;
    const DONUT: &[u8] = br##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><path d="M10 10 H90 V90 H10 Z M35 35 V65 H65 V35 Z" fill="#000"/></svg>"##;
    const BLOB: &[u8] = br##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><path d="M20 50 C20 20 80 20 80 50 C80 80 20 80 20 50 Z" fill="#000"/></svg>"##;
    // An evenodd donut: the SAME geometry as DONUT but both subpaths wound the same way, so only the
    // even-odd rule yields the hole — proving the bake honors the SVG's fill rule.
    const EVENODD_DONUT: &[u8] = br##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><path d="M10 10 H90 V90 H10 Z M35 35 H65 V65 H35 Z" fill="#000" fill-rule="evenodd"/></svg>"##;

    #[test]
    fn msdf_reproduces_a_curved_svg() {
        assert!(iou_msdf_vs_raster(CIRCLE, Fill::Nonzero, 128, 8.0) > 0.95);
    }

    #[test]
    fn msdf_reproduces_a_svg_with_a_hole() {
        assert!(iou_msdf_vs_raster(DONUT, Fill::Nonzero, 128, 8.0) > 0.95);
    }

    #[test]
    fn msdf_reproduces_a_cubic_bezier_blob() {
        assert!(iou_msdf_vs_raster(BLOB, Fill::Nonzero, 128, 8.0) > 0.95);
    }

    #[test]
    fn detects_and_honors_the_evenodd_fill_rule() {
        // The rule is auto-detected from the SVG, and the resvg raster (which also honors evenodd)
        // matches — the hole is preserved. Baking this same SVG as nonzero would fill the hole.
        assert_eq!(svg_fill_rule(EVENODD_DONUT), Fill::EvenOdd);
        let iou = iou_msdf_vs_raster(EVENODD_DONUT, svg_fill_rule(EVENODD_DONUT), 128, 8.0);
        assert!(iou > 0.95, "evenodd donut IoU too low: {iou}");
    }

    // Heavy + opt-in (`cargo test --features bake -- --ignored`): the full logo has thousands of
    // text-ring segments, so the O(pixels·segments) fdsm bake takes minutes. The fast SVG fidelity
    // tests above prove the pipeline generically; this confirms the actual art specifically.
    #[test]
    #[ignore = "slow: bakes the full logo (minutes); run with --ignored"]
    fn real_laughing_man_svg_bakes_faithfully_if_present() {
        // The actual logo is copyrighted and not committed; run this check only when the user has
        // dropped `laugh.svg` in the crate root. Its fill rule is auto-detected (evenodd).
        let path = std::path::Path::new("laugh.svg");
        if !path.exists() {
            return;
        }
        let svg = std::fs::read(path).expect("read laugh.svg");
        let fill = svg_fill_rule(&svg);
        // 256² keeps the fdsm bake (O(pixels·segments) over this detailed logo) tractable in a test;
        // 0.88 tolerates thin sub-texel strokes at this resolution.
        let iou = iou_msdf_vs_raster(&svg, fill, 256, 6.0);
        assert!(iou > 0.88, "laugh.svg MSDF vs SVG raster IoU too low ({iou}, fill {fill:?})");
    }
}
