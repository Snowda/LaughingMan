//! SCRFD post-processing (pure): letterbox preprocess, per-stride anchor decode (`distance2bbox`/
//! `distance2kps`), map back to image space. Dependency-free and unit-tested; the ONNX session is in [`super::session`].

use crate::detect::nms::nms;
use crate::detect::{Bbox, Detection, Landmarks};
use crate::num::Cast as _;

pub const INPUT_SIZE: usize = 640;
// The three feature-map strides SCRFD emits (coarse→fine reversed, matching output ordering).
pub const STRIDES: [usize; 3] = [8, 16, 32];
pub const NUM_ANCHORS: usize = 2; // Anchors per feature-map location.
pub const SCORE_THRESHOLD: f32 = 0.5; // Default NMS IoU threshold.
pub const NMS_THRESHOLD: f32 = 0.4; // Default NMS IoU threshold.

// Pixel normalization: (v - MEAN) / STD, matching InsightFace's blob (mean 127.5, scale 1/128).
const MEAN: f32 = 127.5;
const STD: f32 = 128.0;

/// The uniform resize factor mapping the source into a `target`-square letterbox (top-left, aspect preserved).
#[must_use]
pub fn letterbox_scale(width: u32, height: u32, target: usize) -> f32 {
    let t = target.to_f32();
    (t / width.to_f32()).min(t / height.to_f32())
}

/// Nearest-neighbor letterbox of an RGB8 frame into a normalized planar-RGB (NCHW) `3*target*target` tensor; pad = normalized zero.
#[must_use]
pub fn preprocess(rgb: &[u8], width: u32, height: u32, target: usize) -> Vec<f32> {
    let scale = letterbox_scale(width, height, target);
    let new_w = (width.to_f32() * scale).round().to_usize();
    let new_h = (height.to_f32() * scale).round().to_usize();
    let plane = target * target;
    let pad = (0.0 - MEAN) / STD;
    let mut out = vec![pad; 3 * plane];
    let (w, h) = (width.to_usize(), height.to_usize());
    for oy in 0..new_h.min(target) {
        let sy = (oy.to_f32() / scale).to_usize().min(h - 1);
        for ox in 0..new_w.min(target) {
            let sx = (ox.to_f32() / scale).to_usize().min(w - 1);
            let src = (sy * w + sx) * 3;
            let dst = oy * target + ox;
            out[dst] = (f32::from(rgb[src]) - MEAN) / STD;
            out[plane + dst] = (f32::from(rgb[src + 1]) - MEAN) / STD;
            out[2 * plane + dst] = (f32::from(rgb[src + 2]) - MEAN) / STD;
        }
    }
    out
}

/// A box from an anchor center and four stride-scaled edge distances `[left, top, right, bottom]`.
#[must_use]
pub fn distance2bbox(cx: f32, cy: f32, d: [f32; 4]) -> Bbox {
    Bbox {
        x1: cx - d[0],
        y1: cy - d[1],
        x2: cx + d[2],
        y2: cy + d[3],
    }
}

/// Five landmarks from an anchor center and ten (already stride-scaled) offsets `[x0,y0,...,x4,y4]`.
#[must_use]
pub fn distance2kps(cx: f32, cy: f32, d: &[f32; 10]) -> Landmarks {
    let mut lm: Landmarks = [(0.0, 0.0); 5];
    for (i, point) in lm.iter_mut().enumerate() {
        *point = (cx + d[2 * i], cy + d[2 * i + 1]);
    }
    lm
}

/// Decodes one stride's raw outputs (letterboxed input space): anchor centers `(col*stride, row*stride)`
/// row-major, `num_anchors` innermost, preds ×stride. Keeps anchors scoring ≥ `score_threshold`.
#[must_use]
pub fn decode_stride(
    raw: StrideOutputs,
    stride: usize,
    num_anchors: usize,
    score_threshold: f32,
) -> Vec<Detection> {
    let (scores, bbox, kps, feat_w, feat_h) = raw;
    let stride_f = stride.to_f32();
    let mut out = Vec::new();
    let mut idx = 0;
    for row in 0..feat_h {
        for col in 0..feat_w {
            for _ in 0..num_anchors {
                let score = scores[idx];
                if score >= score_threshold {
                    let (cx, cy) = (col.to_f32() * stride_f, row.to_f32() * stride_f);
                    let d = [
                        bbox[idx * 4] * stride_f,
                        bbox[idx * 4 + 1] * stride_f,
                        bbox[idx * 4 + 2] * stride_f,
                        bbox[idx * 4 + 3] * stride_f,
                    ];
                    let mut kd = [0.0f32; 10];
                    for (k, slot) in kd.iter_mut().enumerate() {
                        *slot = kps[idx * 10 + k] * stride_f;
                    }
                    out.push(Detection {
                        bbox: distance2bbox(cx, cy, d),
                        score,
                        landmarks: distance2kps(cx, cy, &kd),
                    });
                }
                idx += 1;
            }
        }
    }
    out
}

/// Maps a detection from letterboxed input space back to original pixels (divide by scale; no offset).
#[must_use]
pub fn rescale(mut det: Detection, scale: f32) -> Detection {
    let inv = 1.0 / scale;
    det.bbox = Bbox {
        x1: det.bbox.x1 * inv,
        y1: det.bbox.y1 * inv,
        x2: det.bbox.x2 * inv,
        y2: det.bbox.y2 * inv,
    };
    for point in &mut det.landmarks {
        *point = (point.0 * inv, point.1 * inv);
    }
    det
}

/// One stride's raw network outputs: `(scores, bbox_preds, kps_preds, feat_w, feat_h)`.
pub type StrideOutputs<'a> = (&'a [f32], &'a [f32], &'a [f32], usize, usize);

/// Assembles original-image-space detections from every stride: decode each, NMS across all, rescale. `per_stride[i]` ↔ `STRIDES[i]`.
#[must_use]
pub fn assemble(
    per_stride: &[StrideOutputs],
    scale: f32,
    score_threshold: f32,
    nms_threshold: f32,
) -> Vec<Detection> {
    let mut all = Vec::new();
    for (i, &raw) in per_stride.iter().enumerate() {
        all.extend(decode_stride(raw, STRIDES[i], NUM_ANCHORS, score_threshold));
    }
    nms(all, nms_threshold)
        .into_iter()
        .map(|det| rescale(det, scale))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        INPUT_SIZE, assemble, decode_stride, distance2bbox, distance2kps, letterbox_scale,
        preprocess, rescale,
    };
    use crate::detect::{Bbox, Detection};

    #[test]
    fn distance2bbox_offsets_from_center() {
        let b = distance2bbox(100.0, 100.0, [10.0, 20.0, 30.0, 40.0]);
        assert_eq!(b, Bbox { x1: 90.0, y1: 80.0, x2: 130.0, y2: 140.0 });
    }

    #[test]
    fn distance2kps_adds_offsets_to_center() {
        let d = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let lm = distance2kps(100.0, 200.0, &d);
        assert_eq!(lm[0], (101.0, 202.0));
        assert_eq!(lm[2], (105.0, 206.0));
        assert_eq!(lm[4], (109.0, 210.0));
    }

    #[test]
    fn decode_stride_single_anchor_hand_computed() {
        // 1x1 map, stride 8, one anchor at center (0,0). bbox preds [1,1,1,1]*8 = 8 each →
        // box (-8,-8,8,8). Score above threshold → kept.
        let scores = [0.9];
        let bbox = [1.0, 1.0, 1.0, 1.0];
        let kps = [0.0; 10];
        let dets = decode_stride((&scores, &bbox, &kps, 1, 1), 8, 1, 0.5);
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].bbox, Bbox { x1: -8.0, y1: -8.0, x2: 8.0, y2: 8.0 });
    }

    #[test]
    fn decode_stride_drops_below_threshold() {
        let dets = decode_stride((&[0.3], &[1.0; 4], &[0.0; 10], 1, 1), 8, 1, 0.5);
        assert!(dets.is_empty());
    }

    #[test]
    fn decode_stride_anchor_centers_follow_col_row() {
        // 2x2 map, stride 10, one anchor. All scores pass; boxes have zero distance so each box is a
        // point at the anchor center (col*10, row*10).
        let scores = [1.0; 4];
        let bbox = [0.0; 16];
        let kps = [0.0; 40];
        let dets = decode_stride((&scores, &bbox, &kps, 2, 2), 10, 1, 0.5);
        let centers: Vec<(f32, f32)> = dets.iter().map(|d| (d.bbox.x1, d.bbox.y1)).collect();
        // Row-major: (0,0),(10,0),(0,10),(10,10).
        assert_eq!(centers, vec![(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0)]);
    }

    #[test]
    fn decode_stride_two_anchors_share_a_center() {
        // 1x1 map, 2 anchors: both at center (0,0), distinct scores, both kept.
        let scores = [0.9, 0.8];
        let bbox = [0.0; 8];
        let kps = [0.0; 20];
        let dets = decode_stride((&scores, &bbox, &kps, 1, 1), 8, 2, 0.5);
        assert_eq!(dets.len(), 2);
        assert!(dets.iter().all(|d| d.bbox.x1 == 0.0 && d.bbox.y1 == 0.0));
    }

    #[test]
    fn letterbox_scale_uses_the_limiting_dimension() {
        // 1280x720 into 640: width limits (640/1280=0.5 < 640/720≈0.889).
        assert!((letterbox_scale(1280, 720, INPUT_SIZE) - 0.5).abs() < 1e-6);
        // Square image scales to fill.
        assert!((letterbox_scale(640, 640, INPUT_SIZE) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rescale_divides_out_the_letterbox_scale() {
        let det = Detection {
            bbox: Bbox { x1: 50.0, y1: 50.0, x2: 100.0, y2: 100.0 },
            score: 0.9,
            landmarks: [(60.0, 60.0); 5],
        };
        let back = rescale(det, 0.5);
        assert_eq!(back.bbox, Bbox { x1: 100.0, y1: 100.0, x2: 200.0, y2: 200.0 });
        assert_eq!(back.landmarks[0], (120.0, 120.0));
    }

    #[test]
    fn preprocess_produces_normalized_nchw() {
        // 2x1 RGB image: a white and a black pixel. Target 4 for a small deterministic buffer.
        let rgb = [255, 255, 255, 0, 0, 0];
        let out = preprocess(&rgb, 2, 1, 4);
        assert_eq!(out.len(), 3 * 4 * 4);
        // Top-left maps to the source's first (white) pixel: (255-127.5)/128 ≈ 0.996.
        assert!((out[0] - (255.0 - 127.5) / 128.0).abs() < 1e-6);
        // The pad region carries the normalized zero pixel.
        let last = out.len() - 1;
        assert!((out[last] - (0.0 - 127.5) / 128.0).abs() < 1e-6);
    }

    #[test]
    fn assemble_merges_strides_nms_and_rescales() {
        // Strides 8 and 16, each 1x1 with 2 anchors: anchor 0 decodes to the same box on both, anchor 1
        // is below threshold. NMS collapses the duplicates to 0.9, then rescale(0.5) doubles to ±16.
        let s8 = ([0.9f32, 0.1], [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0f32; 20]);
        let s16 = ([0.85f32, 0.1], [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0], [0.0f32; 20]);
        let per_stride: Vec<super::StrideOutputs> = vec![
            (&s8.0, &s8.1, &s8.2, 1, 1),
            (&s16.0, &s16.1, &s16.2, 1, 1),
        ];
        let dets = assemble(&per_stride, 0.5, 0.5, 0.4);
        assert_eq!(dets.len(), 1, "NMS collapses the duplicate box");
        assert!((dets[0].score - 0.9).abs() < 1e-6);
        assert_eq!(dets[0].bbox, Bbox { x1: -16.0, y1: -16.0, x2: 16.0, y2: 16.0 });
    }
}
