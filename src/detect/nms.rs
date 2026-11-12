//! Intersection-over-union and greedy non-maximum suppression, the core of the detector post-process.
//! Pure, with tests against fixtures computed by hand.

use std::cmp::Ordering;

use crate::detect::{Bbox, Detection};

/// Intersection-over-union of two boxes. 0 when they don't overlap or either is degenerate.
#[must_use]
pub fn iou(a: &Bbox, b: &Bbox) -> f32 {
    let ix1: f32 = a.x1.max(b.x1);
    let iy1: f32 = a.y1.max(b.y1);
    let ix2: f32 = a.x2.min(b.x2);
    let iy2: f32 = a.y2.min(b.y2);
    let inter: f32 = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
    let union: f32 = a.area() + b.area() - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

/// Greedy NMS in place, highest score first. A box that overlaps no kept box by more than
/// `iou_threshold` moves to the front. Returns the kept count `n`. `dets[..n]` holds them.
#[must_use]
pub fn nms(dets: &mut [Detection], iou_threshold: f32) -> usize {
    dets.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    let mut kept: usize = 0;
    for i in 0..dets.len() {
        let det: Detection = dets[i];
        if dets[..kept].iter().all(|k| iou(&k.bbox, &det.bbox) <= iou_threshold) {
            dets[kept] = det;
            kept += 1;
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::{iou, nms};
    use crate::detect::{Bbox, Detection, Landmarks};

    const NO_LM: Landmarks = [(0.0, 0.0); 5];

    fn det(x1: f32, y1: f32, x2: f32, y2: f32, score: f32) -> Detection {
        Detection {
            bbox: Bbox { x1, y1, x2, y2 },
            score,
            landmarks: NO_LM,
        }
    }

    #[test]
    fn iou_of_identical_boxes_is_one() {
        let b: Bbox = Bbox { x1: 0.0, y1: 0.0, x2: 10.0, y2: 10.0 };
        assert!((iou(&b, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn iou_of_disjoint_boxes_is_zero() {
        let a: Bbox = Bbox { x1: 0.0, y1: 0.0, x2: 10.0, y2: 10.0 };
        let b: Bbox = Bbox { x1: 20.0, y1: 20.0, x2: 30.0, y2: 30.0 };
        assert_eq!(iou(&a, &b).to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn iou_of_half_overlap_is_one_third() {
        // Two 10x10 boxes with a 5x10 overlap: inter=50, union=150 → 1/3.
        let a: Bbox = Bbox { x1: 0.0, y1: 0.0, x2: 10.0, y2: 10.0 };
        let b: Bbox = Bbox { x1: 5.0, y1: 0.0, x2: 15.0, y2: 10.0 };
        assert!((iou(&a, &b) - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn nms_keeps_highest_score_and_drops_overlap() {
        // Two boxes that overlap a lot, and one apart. NMS keeps the stronger overlap box and the one apart.
        let strong: Detection = det(0.0, 0.0, 10.0, 10.0, 0.9);
        let weak: Detection = det(1.0, 1.0, 11.0, 11.0, 0.6); // IoU with `strong` ≈ 0.68 > 0.4
        let apart: Detection = det(50.0, 50.0, 60.0, 60.0, 0.7);
        let mut dets: [Detection; 3] = [weak, apart, strong];
        let n: usize = nms(&mut dets, 0.4);
        let kept: &[Detection] = &dets[..n];
        assert_eq!(kept.len(), 2);
        assert!((kept[0].score - 0.9).abs() < 1e-6, "highest score kept first");
        assert!(kept.iter().any(|d| (d.score - 0.7).abs() < 1e-6), "disjoint box kept");
        assert!(!kept.iter().any(|d| (d.score - 0.6).abs() < 1e-6), "overlapping weak box dropped");
    }

    #[test]
    fn nms_below_threshold_keeps_both() {
        // Small overlap under the threshold: both survive.
        let a: Detection = det(0.0, 0.0, 10.0, 10.0, 0.9);
        let b: Detection = det(8.0, 0.0, 18.0, 10.0, 0.8); // IoU = 2/18 ≈ 0.11
        assert_eq!(nms(&mut [a, b], 0.4), 2);
    }

    #[test]
    fn nms_on_empty_is_empty() {
        assert_eq!(nms(&mut [], 0.4), 0);
    }
}
