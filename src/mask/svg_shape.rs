//! SVG → fdsm `Shape<Contour>`. usvg lowers strokes/primitives to filled paths; we stream every
//! filled path's segments (with its absolute transform applied) into fdsm contours. fdsm takes
//! cubics natively, so no flattening; SVG coordinates are kept y-down to match the bake's image
//! space, and the SVG winding is preserved so the bake's `FillRule::Nonzero` resolves inside/outside.
//!
//! Live `<text>` is dropped (usvg needs fonts to shape it); convert text to paths in the editor —
//! which is exactly how the logo's text ring is prepared for the rotating-layer bake.

use anyhow::{Context as _, anyhow};
use fdsm::bezier::{Point, Segment};
use fdsm::shape::{Contour, Shape};
use usvg::tiny_skia_path::PathSegment;
use usvg::{Group, Node, Options, Tree};

use crate::mask::bake::Fill;

// Endpoints closer than this (in SVG units) are treated as coincident when auto-closing a subpath.
const CLOSE_EPS: f64 = 1e-6;

/// Parses `svg` into an fdsm shape in SVG user units. Errors if the SVG can't be parsed or has no
/// filled path.
pub fn shape_from_svg(svg: &[u8]) -> anyhow::Result<Shape<Contour>> {
    let tree = Tree::from_data(svg, &Options::default()).context("parsing SVG")?;
    let mut contours: Vec<Contour> = Vec::new();
    collect_group(tree.root(), &mut contours);
    contours.retain(|contour| !contour.segments.is_empty());
    if contours.is_empty() {
        return Err(anyhow!("SVG has no filled path (convert text/strokes to filled paths?)"));
    }
    Ok(Shape { contours })
}

/// Parses `svg` into one fdsm shape per filled `<path>`, each paired with its own fill rule. Baking
/// these as separate layers and unioning the results reproduces SVG fill semantics — overlapping
/// paths combine instead of XOR-ing into holes — while each path keeps its own interior holes.
pub fn shapes_from_svg(svg: &[u8]) -> anyhow::Result<Vec<(Shape<Contour>, Fill)>> {
    let tree = Tree::from_data(svg, &Options::default()).context("parsing SVG")?;
    let mut layers: Vec<(Shape<Contour>, Fill)> = Vec::new();
    collect_layers(tree.root(), &mut layers);
    layers.retain(|(shape, _)| !shape.contours.is_empty());
    if layers.is_empty() {
        return Err(anyhow!("SVG has no filled path (convert text/strokes to filled paths?)"));
    }
    Ok(layers)
}

// Walks a group, emitting one (shape, fill) per filled path (absolute transform applied).
fn collect_layers(group: &Group, out: &mut Vec<(Shape<Contour>, Fill)>) {
    for node in group.children() {
        match node {
            Node::Group(child) => collect_layers(child, out),
            Node::Path(path) => {
                if let Some(fill) = path.fill() {
                    let mut contours = Vec::new();
                    append_path(path, &mut contours);
                    let rule = match fill.rule() {
                        usvg::FillRule::EvenOdd => Fill::EvenOdd,
                        usvg::FillRule::NonZero => Fill::Nonzero,
                    };
                    out.push((Shape { contours }, rule));
                }
            }
            _ => {}
        }
    }
}

/// The fill rule of the first filled path in `svg` — the rule the bake must sign by so holes read
/// correctly (the Laughing Man logo is `fill-rule:evenodd`). Defaults to nonzero when unparseable or
/// unfilled.
#[must_use]
pub fn svg_fill_rule(svg: &[u8]) -> Fill {
    match Tree::from_data(svg, &Options::default()) {
        Ok(tree) => first_fill_rule(tree.root()).unwrap_or(Fill::Nonzero),
        Err(_) => Fill::Nonzero,
    }
}

// The fill rule of the first filled path found walking `group`.
fn first_fill_rule(group: &Group) -> Option<Fill> {
    for node in group.children() {
        match node {
            Node::Group(child) => {
                if let Some(rule) = first_fill_rule(child) {
                    return Some(rule);
                }
            }
            Node::Path(path) => {
                if let Some(fill) = path.fill() {
                    return Some(match fill.rule() {
                        usvg::FillRule::EvenOdd => Fill::EvenOdd,
                        usvg::FillRule::NonZero => Fill::Nonzero,
                    });
                }
            }
            _ => {}
        }
    }
    None
}

// Walks a group, streaming every filled path's subpaths (absolute transform applied) into contours.
fn collect_group(group: &Group, out: &mut Vec<Contour>) {
    for node in group.children() {
        match node {
            Node::Group(child) => collect_group(child, out),
            Node::Path(path) if path.fill().is_some() => append_path(path, out),
            _ => {}
        }
    }
}

// Streams one path's segments into contours, one per subpath. Each subpath is closed (an explicit
// `Close`, or an added closing line when the pen didn't return to the start).
fn append_path(path: &usvg::Path, out: &mut Vec<Contour>) {
    let t = path.abs_transform();
    let map = |x: f32, y: f32| -> Point {
        Point::new(
            f64::from(t.sx * x + t.kx * y + t.tx),
            f64::from(t.ky * x + t.sy * y + t.ty),
        )
    };

    let mut segments: Vec<Segment> = Vec::new();
    let mut start: Option<Point> = None;
    let mut cur: Option<Point> = None;
    for segment in path.data().segments() {
        match segment {
            PathSegment::MoveTo(p) => {
                end_subpath(&mut segments, start, cur, out);
                let m = map(p.x, p.y);
                start = Some(m);
                cur = Some(m);
            }
            PathSegment::LineTo(p) => {
                let e = map(p.x, p.y);
                if let Some(s) = cur {
                    segments.push(Segment::line(s, e));
                }
                cur = Some(e);
            }
            PathSegment::QuadTo(c, p) => {
                let (control, e) = (map(c.x, c.y), map(p.x, p.y));
                if let Some(s) = cur {
                    segments.push(Segment::quad(s, control, e));
                }
                cur = Some(e);
            }
            PathSegment::CubicTo(c1, c2, p) => {
                let (a, b, e) = (map(c1.x, c1.y), map(c2.x, c2.y), map(p.x, p.y));
                if let Some(s) = cur {
                    segments.push(Segment::cubic(s, a, b, e));
                }
                cur = Some(e);
            }
            PathSegment::Close => {
                end_subpath(&mut segments, start, cur, out);
                cur = start; // pen returns to the subpath start
            }
        }
    }
    end_subpath(&mut segments, start, cur, out);
}

// Closes the accumulated subpath (adding a closing line if the pen isn't back at the start) and
// pushes it as a contour. No-op when empty.
fn end_subpath(
    segments: &mut Vec<Segment>,
    start: Option<Point>,
    cur: Option<Point>,
    out: &mut Vec<Contour>,
) {
    if segments.is_empty() {
        return;
    }
    if let (Some(s), Some(c)) = (start, cur)
        && (s - c).norm() > CLOSE_EPS
    {
        segments.push(Segment::line(c, s));
    }
    out.push(Contour {
        segments: std::mem::take(segments),
    });
}

#[cfg(test)]
mod tests {
    use super::shape_from_svg;

    const SQUARE: &[u8] = br##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><rect width="10" height="10" fill="#000"/></svg>"##;
    const DONUT: &[u8] = br##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><path d="M0 0 H10 V10 H0 Z M3 3 V7 H7 V3 Z" fill="#000"/></svg>"##;
    const UNFILLED: &[u8] = br##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><rect width="10" height="10" fill="none" stroke="black"/></svg>"##;

    #[test]
    fn square_becomes_one_closed_contour() -> anyhow::Result<()> {
        let shape = shape_from_svg(SQUARE)?;
        assert_eq!(shape.contours.len(), 1);
        // Rect lowers to 4 sides; the contour is closed (start == end).
        let contour = &shape.contours[0];
        assert!(contour.segments.len() >= 4);
        let first = contour.segments[0].start();
        let last = contour.segments[contour.segments.len() - 1].end();
        assert!((first - last).norm() < 1e-3, "contour must be closed");
        Ok(())
    }

    #[test]
    fn donut_yields_two_contours() -> anyhow::Result<()> {
        let shape = shape_from_svg(DONUT)?;
        assert_eq!(shape.contours.len(), 2, "outer loop + hole");
        Ok(())
    }

    #[test]
    fn unfilled_or_garbage_errors() {
        assert!(shape_from_svg(UNFILLED).is_err());
        assert!(shape_from_svg(b"not an svg").is_err());
    }
}
