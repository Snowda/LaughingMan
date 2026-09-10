//! Face detection (SCRFD): pure decode (anchors, distance→box/landmarks, letterbox) and NMS
//! live here, unit-tested; the ONNX session is behind the `detect` feature. Tracking (`track`)
//! and smoothing (`smooth`) live alongside it.

pub mod nms;
pub mod scrfd;
pub mod smooth;
pub mod track;

#[cfg(feature = "detect")]
pub mod session;

use crate::num::Cast as _;

/// An axis-aligned face box in image pixels (top-left origin, y-down).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bbox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl Bbox {
    #[must_use]
    pub fn width(&self) -> f32 {
        (self.x2 - self.x1).max(0.0)
    }

    #[must_use]
    pub fn height(&self) -> f32 {
        (self.y2 - self.y1).max(0.0)
    }

    #[must_use]
    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }
}

/// The five SCRFD landmarks: left eye, right eye, nose, left/right mouth corner — `(x, y)` pixels.
pub type Landmarks = [(f32, f32); 5];

/// One detected face: its box, confidence, and landmarks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Detection {
    pub bbox: Bbox,
    pub score: f32,
    pub landmarks: Landmarks,
}

/// A source of face detections for one RGB frame.
pub trait Detector {
    /// Detects faces in a tightly-packed RGB8 frame (`width * height * 3` bytes).
    fn detect(&mut self, rgb: &[u8], width: u32, height: u32) -> anyhow::Result<Vec<Detection>>;
}

/// A per-frame face source for the presenter — real detector or synthetic demo. `Send` for the detect thread.
pub trait FaceProvider: Send {
    fn detect_frame(
        &mut self,
        rgb: &[u8],
        width: u32,
        height: u32,
        elapsed_secs: f32,
    ) -> anyhow::Result<Vec<Detection>>;
}

/// A synthetic moving, tilting face — ignores the frame and drives one face from `elapsed_secs`, so
/// the whole detect→track→composite path (and the ring animation) is demoable with no hardware.
pub struct DemoFaceProvider;

impl FaceProvider for DemoFaceProvider {
    fn detect_frame(
        &mut self,
        _rgb: &[u8],
        width: u32,
        height: u32,
        elapsed_secs: f32,
    ) -> anyhow::Result<Vec<Detection>> {
        Ok(vec![demo_detection(width, height, elapsed_secs)])
    }
}

fn demo_detection(width: u32, height: u32, t: f32) -> Detection {
    let (fw, fh) = (width.to_f32(), height.to_f32());
    // Drift the face around the frame center and oscillate the head tilt (roll).
    let cx = fw * 0.5 + fw * 0.18 * (t * 0.6).cos();
    let cy = fh * 0.5 + fh * 0.18 * (t * 0.6).sin();
    let size = width.min(height).to_f32() * 0.28;
    let tilt = (t * 0.4).sin() * 0.6;
    let (ec, es) = (tilt.cos(), tilt.sin());
    let e = size * 0.22;
    let left_eye = (cx - e * ec, cy - e * es);
    let right_eye = (cx + e * ec, cy + e * es);
    Detection {
        bbox: Bbox {
            x1: cx - size * 0.5,
            y1: cy - size * 0.5,
            x2: cx + size * 0.5,
            y2: cy + size * 0.5,
        },
        score: 0.9,
        landmarks: [
            left_eye,
            right_eye,
            (cx, cy + e * 0.5),
            (cx - e * 0.7, cy + e),
            (cx + e * 0.7, cy + e),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::{DemoFaceProvider, FaceProvider};

    #[test]
    fn demo_face_is_one_detection_with_ordered_eyes() -> anyhow::Result<()> {
        let mut provider = DemoFaceProvider;
        let dets = provider.detect_frame(&[], 640, 480, 0.0)?;
        assert_eq!(dets.len(), 1);
        let d = dets[0];
        assert!(d.score > 0.5);
        assert!(d.bbox.width() > 0.0 && d.bbox.height() > 0.0);
        // At t=0 the head is level: the left eye is left of the right eye at the same height.
        assert!(d.landmarks[0].0 < d.landmarks[1].0, "left eye left of right");
        assert!((d.landmarks[0].1 - d.landmarks[1].1).abs() < 1e-3, "level at t=0");
        Ok(())
    }

    #[test]
    fn demo_face_moves_and_tilts_over_time() -> anyhow::Result<()> {
        let mut provider = DemoFaceProvider;
        let a = provider.detect_frame(&[], 640, 480, 0.0)?[0];
        let b = provider.detect_frame(&[], 640, 480, 1.5)?[0];
        let ca = ((a.bbox.x1 + a.bbox.x2) * 0.5, (a.bbox.y1 + a.bbox.y2) * 0.5);
        let cb = ((b.bbox.x1 + b.bbox.x2) * 0.5, (b.bbox.y1 + b.bbox.y2) * 0.5);
        assert!((ca.0 - cb.0).abs() + (ca.1 - cb.1).abs() > 1.0, "the demo face drifts");
        // The eye-line tilt (roll) differs, so the overlay's rotation animates.
        let tilt_a = a.landmarks[1].1 - a.landmarks[0].1;
        let tilt_b = b.landmarks[1].1 - b.landmarks[0].1;
        assert!((tilt_a - tilt_b).abs() > 0.01, "roll animates over time");
        Ok(())
    }
}
