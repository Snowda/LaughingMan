//! Pure SCRFD pre- and post-process: letterbox, anchor decode per stride, and the map back to the image.
//! The ONNX session is in [`super::session`].

use crate::detect::nms::nms;
use crate::detect::{Bbox, Detection, Landmarks};
use crate::num::{float_to_int, u32_to_f32, u32_to_usize, usize_to_f32};

pub const INPUT_SIZE: usize = 640;
// The three feature-map strides of SCRFD, in the order of its outputs.
pub const STRIDES: [usize; 3] = [8, 16, 32];
pub const NUM_ANCHORS: usize = 2; // Anchors per feature-map location.
pub const SCORE_THRESHOLD: f32 = 0.5; // The presenter lowers this to the low band of the tracker.
pub const NMS_THRESHOLD: f32 = 0.4;

// Pixel normalization (v - MEAN) / STD, the same as the InsightFace blob.
const MEAN: f32 = 127.5;
const STD: f32 = 128.0;

/// The uniform scale that fits the source into a `target` square letterbox at the top left.
#[must_use]
pub fn letterbox_scale(width: u32, height: u32, target: usize) -> f32 {
    let t: f32 = usize_to_f32(target);
    (t / u32_to_f32(width)).min(t / u32_to_f32(height))
}

/// Nearest-neighbor letterbox of an RGB8 frame into a normalized NCHW tensor of `3*target*target`.
/// The pad is the normalized zero.
#[must_use]
pub fn preprocess(rgb: &[u8], width: u32, height: u32, target: usize) -> Vec<f32> {
    let scale: f32 = letterbox_scale(width, height, target);
    let new_w: usize = float_to_int((u32_to_f32(width) * scale).round());
    let new_h: usize = float_to_int((u32_to_f32(height) * scale).round());
    let plane: usize = target * target;
    let pad: f32 = (0.0 - MEAN) / STD;
    let mut out: Vec<f32> = vec![pad; 3 * plane];
    let (w, h) = (u32_to_usize(width), u32_to_usize(height));
    for oy in 0..new_h.min(target) {
        let sy: usize = float_to_int::<usize>(usize_to_f32(oy) / scale).min(h - 1);
        for ox in 0..new_w.min(target) {
            let sx: usize = float_to_int::<usize>(usize_to_f32(ox) / scale).min(w - 1);
            let src: usize = (sy * w + sx) * 3;
            let dst: usize = oy * target + ox;
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

/// Decodes the raw outputs of one stride in letterbox space. Anchor centers are `(col*stride, row*stride)`.
/// The order is row-major with `num_anchors` innermost. Keeps anchors with a score ≥ `score_threshold`.
pub fn decode_stride<'a>(
    raw: StrideOutputs<'a>,
    stride: usize,
    num_anchors: usize,
    score_threshold: f32,
) -> impl Iterator<Item = Detection> + 'a {
    let (scores, bbox, kps, feat_w, feat_h) = raw;
    let stride_f: f32 = usize_to_f32(stride);
    scores
        .iter()
        .take(feat_w * feat_h * num_anchors)
        .enumerate()
        .filter(move |&(_, &score)| score >= score_threshold)
        .map(move |(idx, &score)| {
            let cell: usize = idx / num_anchors;
            let (cx, cy) = (usize_to_f32(cell % feat_w) * stride_f, usize_to_f32(cell / feat_w) * stride_f);
            let d: [f32; 4] = std::array::from_fn(|k| bbox[idx * 4 + k] * stride_f);
            let kd: [f32; 10] = std::array::from_fn(|k| kps[idx * 10 + k] * stride_f);
            Detection { bbox: distance2bbox(cx, cy, d), score, landmarks: distance2kps(cx, cy, &kd) }
        })
}

/// Maps a detection from letterbox space back to source pixels. It divides by the scale, with no offset.
#[must_use]
pub fn rescale(mut det: Detection, scale: f32) -> Detection {
    let inv: f32 = 1.0 / scale;
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

/// Decodes all strides, applies NMS, and rescales to source pixels. `per_stride[i]` ↔ `STRIDES[i]`.
#[must_use]
pub fn assemble(
    per_stride: &[StrideOutputs; STRIDES.len()],
    scale: f32,
    score_threshold: f32,
    nms_threshold: f32,
) -> Vec<Detection> {
    let mut all: Vec<Detection> = per_stride
        .iter()
        .zip(STRIDES)
        .flat_map(|(&raw, stride)| decode_stride(raw, stride, NUM_ANCHORS, score_threshold))
        .collect();
    let kept: usize = nms(&mut all, nms_threshold);
    all.truncate(kept);
    all.iter_mut().for_each(|det| *det = rescale(*det, scale));
    all
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
        let b: Bbox = distance2bbox(100.0, 100.0, [10.0, 20.0, 30.0, 40.0]);
        assert_eq!(b, Bbox { x1: 90.0, y1: 80.0, x2: 130.0, y2: 140.0 });
    }

    #[test]
    fn distance2kps_adds_offsets_to_center() {
        let d: [f32; 10] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let lm: [(f32, f32); 5] = distance2kps(100.0, 200.0, &d);
        assert_eq!(lm[0], (101.0, 202.0));
        assert_eq!(lm[2], (105.0, 206.0));
        assert_eq!(lm[4], (109.0, 210.0));
    }

    #[test]
    fn decode_stride_single_anchor_hand_computed() {
        // 1x1 map, stride 8, one anchor at (0,0). Preds [1,1,1,1]*8 give box (-8,-8,8,8). The score passes.
        let scores: [f32; 1] = [0.9];
        let bbox: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
        let kps: [f32; 10] = [0.0; 10];
        let mut dets = decode_stride((&scores, &bbox, &kps, 1, 1), 8, 1, 0.5);
        assert_eq!(dets.next().map(|d| d.bbox), Some(Bbox { x1: -8.0, y1: -8.0, x2: 8.0, y2: 8.0 }));
        assert!(dets.next().is_none(), "exactly one anchor decoded");
    }

    #[test]
    fn decode_stride_drops_below_threshold() {
        assert!(
            decode_stride((&[0.3], &[1.0; 4], &[0.0; 10], 1, 1), 8, 1, 0.5).next().is_none(),
            "below-threshold anchor dropped"
        );
    }

    #[test]
    fn decode_stride_anchor_centers_follow_col_row() {
        // 2x2 map, stride 10, one anchor. All scores pass. Each box is a point at (col*10, row*10).
        let scores: [f32; 4] = [1.0; 4];
        let bbox: [f32; 16] = [0.0; 16];
        let kps: [f32; 40] = [0.0; 40];
        assert!(
            decode_stride((&scores, &bbox, &kps, 2, 2), 10, 1, 0.5)
                .map(|d| (d.bbox.x1, d.bbox.y1))
                .eq([(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0)]),
            "anchor centers are row-major (col*10, row*10)"
        );
    }

    #[test]
    fn decode_stride_rows_follow_feat_w_on_a_wide_map() {
        let mut scores: [f32; 12] = [0.0; 12];
        scores[9] = 0.9;
        assert!(
            decode_stride((&scores, &[0.0; 48], &[0.0; 120], 3, 2), 10, 2, 0.5)
                .map(|d| (d.bbox.x1, d.bbox.y1))
                .eq([(10.0, 10.0)]),
            "3x2 map, 2 anchors: anchor 9 is cell 4 = (col 1, row 1)"
        );
    }

    #[test]
    fn decode_stride_two_anchors_share_a_center() {
        // 1x1 map, 2 anchors: both at center (0,0), distinct scores, both kept.
        let scores: [f32; 2] = [0.9, 0.8];
        let bbox: [f32; 8] = [0.0; 8];
        let kps: [f32; 20] = [0.0; 20];
        assert!(
            decode_stride((&scores, &bbox, &kps, 1, 1), 8, 2, 0.5)
                .map(|d| d.bbox.x1 == 0.0 && d.bbox.y1 == 0.0)
                .eq([true, true]),
            "both anchors kept, both at the cell center"
        );
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
        let det: Detection = Detection {
            bbox: Bbox { x1: 50.0, y1: 50.0, x2: 100.0, y2: 100.0 },
            score: 0.9,
            landmarks: [(60.0, 60.0); 5],
        };
        let back: Detection = rescale(det, 0.5);
        assert_eq!(back.bbox, Bbox { x1: 100.0, y1: 100.0, x2: 200.0, y2: 200.0 });
        assert_eq!(back.landmarks[0], (120.0, 120.0));
    }

    #[test]
    fn preprocess_produces_normalized_nchw() {
        // 2x1 RGB image: a white and a black pixel. Target 4 for a small deterministic buffer.
        let rgb: [u8; 6] = [255, 255, 255, 0, 0, 0];
        let out: Vec<f32> = preprocess(&rgb, 2, 1, 4);
        assert_eq!(out.len(), 3 * 4 * 4);
        // Top-left maps to the source's first (white) pixel: (255-127.5)/128 ≈ 0.996.
        assert!((out[0] - (255.0 - 127.5) / 128.0).abs() < 1e-6);
        // The pad region carries the normalized zero pixel.
        let last: usize = out.len() - 1;
        assert!((out[last] - (0.0 - 127.5) / 128.0).abs() < 1e-6);
    }

    #[test]
    fn assemble_merges_strides_nms_and_rescales() {
        // Strides 8 and 16, each 1x1 with 2 anchors. Anchor 0 gives the same box on both. Anchor 1 fails.
        // NMS keeps the 0.9 box. Then rescale(0.5) doubles it to ±16.
        let s8: ([f32; 2], [f32; 8], [f32; 20]) = ([0.9f32, 0.1], [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0f32; 20]);
        let s16: ([f32; 2], [f32; 8], [f32; 20]) = ([0.85f32, 0.1], [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0], [0.0f32; 20]);
        let per_stride: [super::StrideOutputs; 3] = [
            (&s8.0, &s8.1, &s8.2, 1, 1),
            (&s16.0, &s16.1, &s16.2, 1, 1),
            (&[], &[], &[], 0, 0),
        ];
        let dets: Vec<Detection> = assemble(&per_stride, 0.5, 0.5, 0.4);
        assert_eq!(dets.len(), 1, "NMS collapses the duplicate box");
        assert!((dets[0].score - 0.9).abs() < 1e-6);
        assert_eq!(dets[0].bbox, Bbox { x1: -16.0, y1: -16.0, x2: 16.0, y2: 16.0 });
    }
}
