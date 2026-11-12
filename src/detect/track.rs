//! Face tracks: the layer that covers detector gaps. A constant-velocity Kalman filter per face
//! predicts through missed detections ("coast"), so the mask does not drop when the detector blinks.
//! Greedy IoU association ties detections to tracks in the ByteTrack style.
//! High-confidence detections start tracks, and low-confidence ones keep them alive.
//! A fade ramps the mask out over the last part of the coast window, so it does not pop.
//! A One Euro filter smooths the landmarks of each track.
//!
//! The 8-state constant-velocity model `[cx, cy, w, h, ẋ, ẏ, ẇ, ḣ]` splits into four independent
//! 2-state (position, velocity) filters, one per box parameter. [`Kf1d`] is that filter.

use std::cmp::Ordering;

use crate::detect::nms::iou;
use crate::detect::smooth::OneEuroPoint;
use crate::detect::{Bbox, Detection, Landmarks};

/// A scalar constant-velocity Kalman filter (state = position + velocity).
#[derive(Clone, Copy)]
pub struct Kf1d {
    p: f32,
    v: f32,
    // 2x2 covariance, row-major.
    c00: f32,
    c01: f32,
    c10: f32,
    c11: f32,
    q: f32,
    r: f32,
}

impl Kf1d {
    fn new(p0: f32, q: f32, r: f32) -> Self {
        Self {
            p: p0,
            v: 0.0,
            c00: r,
            c01: 0.0,
            c10: 0.0,
            c11: 1000.0, // unknown initial velocity
            q,
            r,
        }
    }

    fn predict(&mut self, dt: f32, q_scale: f32) {
        self.p += self.v * dt;
        // P = F P Fᵀ, F = [[1, dt], [0, 1]].
        let c00: f32 = self.c00 + dt * self.c10 + dt * (self.c01 + dt * self.c11);
        let c01: f32 = self.c01 + dt * self.c11;
        let c10: f32 = self.c10 + dt * self.c11;
        let c11: f32 = self.c11;
        // + process noise Q (white-noise acceleration), multiplied by q_scale during a coast.
        let q: f32 = self.q * q_scale;
        self.c00 = c00 + q * dt * dt * dt / 3.0;
        self.c01 = c01 + q * dt * dt / 2.0;
        self.c10 = c10 + q * dt * dt / 2.0;
        self.c11 = c11 + q * dt;
    }

    fn update(&mut self, z: f32) {
        // Innovation with H = [1, 0].
        let y: f32 = z - self.p;
        let s: f32 = self.c00 + self.r;
        let k0: f32 = self.c00 / s;
        let k1: f32 = self.c10 / s;
        self.p += k0 * y;
        self.v += k1 * y;
        // P = (I - K H) P, K H = [[k0, 0], [k1, 0]].
        let c00: f32 = (1.0 - k0) * self.c00;
        let c01: f32 = (1.0 - k0) * self.c01;
        let c10: f32 = self.c10 - k1 * self.c00;
        let c11: f32 = self.c11 - k1 * self.c01;
        self.c00 = c00;
        self.c01 = c01;
        self.c10 = c10;
        self.c11 = c11;
    }
}

/// A constant-velocity Kalman filter over a box's `(cx, cy, w, h)`.
#[derive(Clone, Copy)]
struct BoxKalman {
    cx: Kf1d,
    cy: Kf1d,
    w: Kf1d,
    h: Kf1d,
}

impl BoxKalman {
    fn from_box(b: &Bbox, q: f32, r: f32) -> Self {
        let (cx, cy) = ((b.x1 + b.x2) * 0.5, (b.y1 + b.y2) * 0.5);
        Self {
            cx: Kf1d::new(cx, q, r),
            cy: Kf1d::new(cy, q, r),
            w: Kf1d::new(b.width(), q, r),
            h: Kf1d::new(b.height(), q, r),
        }
    }

    fn predict(&mut self, dt: f32, q_scale: f32) {
        self.cx.predict(dt, q_scale);
        self.cy.predict(dt, q_scale);
        self.w.predict(dt, q_scale);
        self.h.predict(dt, q_scale);
    }

    fn update(&mut self, b: &Bbox) {
        self.cx.update((b.x1 + b.x2) * 0.5);
        self.cy.update((b.y1 + b.y2) * 0.5);
        self.w.update(b.width());
        self.h.update(b.height());
    }

    fn bbox(&self) -> Bbox {
        let (hw, hh) = (self.w.p.max(0.0) * 0.5, self.h.p.max(0.0) * 0.5);
        Bbox {
            x1: self.cx.p - hw,
            y1: self.cy.p - hh,
            x2: self.cx.p + hw,
            y2: self.cy.p + hh,
        }
    }
}

/// Tunable tracker parameters. Defaults follow the plan (ByteTrack thresholds, ~0.4 s coast, ~0.15 s
/// fade, One Euro min_cutoff/beta).
#[derive(Clone, Copy)]
pub struct TrackerParams {
    pub iou_threshold: f32,
    pub min_hits: u32,
    pub max_coast: f32,
    pub fade_time: f32,
    pub high_score: f32,
    pub low_score: f32,
    pub q: f32,
    pub r: f32,
    pub coast_q_scale: f32,
    pub min_cutoff: f32,
    pub beta: f32,
}

impl Default for TrackerParams {
    fn default() -> Self {
        Self {
            iou_threshold: 0.3,
            min_hits: 3,
            max_coast: 0.4,
            fade_time: 0.15,
            high_score: 0.5,
            low_score: 0.2,
            q: 40.0,
            r: 4.0,
            coast_q_scale: 4.0,
            min_cutoff: 1.0,
            beta: 0.5,
        }
    }
}

/// A tracked face emitted to the compositor: the smoothed box, smoothed landmarks, a stable id, and
/// the fade (1 = solid, ramps to 0 as a lost track ages out).
#[derive(Clone, Copy, Debug)]
pub struct TrackOutput {
    pub id: u32,
    pub bbox: Bbox,
    pub landmarks: Landmarks,
    pub fade: f32,
}

struct Track {
    id: u32,
    kf: BoxKalman,
    landmark_filters: [OneEuroPoint; 5],
    landmarks: Landmarks,
    hits: u32,
    coast_time: f32,
    confirmed: bool,
}

impl Track {
    fn fade(&self, params: &TrackerParams) -> f32 {
        ((params.max_coast - self.coast_time) / params.fade_time).clamp(0.0, 1.0)
    }
}

/// Multi-face tracker: constant-velocity Kalman tracks, IoU association, a One Euro filter on the
/// landmarks, and the coast/fade lifecycle.
pub struct Tracker {
    tracks: Vec<Track>,
    next_id: u32,
    params: TrackerParams,
}

impl Tracker {
    #[must_use]
    pub fn new(params: TrackerParams) -> Self {
        Self {
            tracks: Vec::new(),
            next_id: 0,
            params,
        }
    }

    /// Advances all tracks by `dt` seconds against this frame's `detections`. Returns the confirmed
    /// tracks for the compositor, also the tracks that coast inside the fade window.
    pub fn update(&mut self, detections: &[Detection], dt: f32) -> Vec<TrackOutput> {
        // 1. Predict every track forward. Increase the process noise during a coast.
        for track in &mut self.tracks {
            let q_scale: f32 = if track.coast_time > 0.0 {
                self.params.coast_q_scale
            } else {
                1.0
            };
            track.kf.predict(dt, q_scale);
            track.coast_time += dt;
        }

        // 2. Split detections by confidence (ByteTrack).
        let params: TrackerParams = self.params;
        let high = |d: &Detection| d.score >= params.high_score;
        let low = |d: &Detection| d.score >= params.low_score && d.score < params.high_score;

        let mut track_matched = vec![false; self.tracks.len()];
        let mut det_matched = vec![false; detections.len()];

        // 3. Round 1: all tracks vs high-confidence detections.
        self.associate_round(detections, high, &mut track_matched, &mut det_matched, dt);
        // 4. Round 2: still-unmatched tracks vs low-confidence detections (keep-alive).
        self.associate_round(detections, low, &mut track_matched, &mut det_matched, dt);

        // 5. Birth a track for each unmatched high-confidence detection.
        for (det, _) in detections.iter().zip(&det_matched).filter(|&(d, &matched)| high(d) && !matched) {
            self.birth(det);
        }

        // 6. Retire tracks past the coast window.
        let max_coast: f32 = self.params.max_coast;
        self.tracks.retain(|t| t.coast_time < max_coast);

        // 7. Emit confirmed tracks.
        let params: TrackerParams = self.params;
        self.tracks
            .iter()
            .filter(|t| t.confirmed)
            .map(|t| TrackOutput {
                id: t.id,
                bbox: t.kf.bbox(),
                landmarks: t.landmarks,
                fade: t.fade(&params),
            })
            .collect()
    }

    // Matches free tracks to the detections that `in_subset` accepts, greedy by IoU, and applies each match.
    fn associate_round(
        &mut self,
        detections: &[Detection],
        in_subset: impl Fn(&Detection) -> bool,
        track_matched: &mut [bool],
        det_matched: &mut [bool],
        dt: f32,
    ) {
        // Candidate (iou, track_idx, det_idx) pairs above threshold, best first.
        let mut pairs: Vec<(f32, usize, usize)> = Vec::new();
        for (ti, track) in self.tracks.iter().enumerate() {
            if track_matched[ti] {
                continue;
            }
            let tb: Bbox = track.kf.bbox();
            for (di, det) in detections.iter().enumerate().filter(|&(_, d)| in_subset(d)) {
                if det_matched[di] {
                    continue;
                }
                let score: f32 = iou(&tb, &det.bbox);
                if score >= self.params.iou_threshold {
                    pairs.push((score, ti, di));
                }
            }
        }
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        for (_, ti, di) in pairs {
            if track_matched[ti] || det_matched[di] {
                continue;
            }
            track_matched[ti] = true;
            det_matched[di] = true;
            self.apply_match(ti, &detections[di], dt);
        }
    }

    fn apply_match(&mut self, ti: usize, det: &Detection, dt: f32) {
        let track: &mut Track = &mut self.tracks[ti];
        track.kf.update(&det.bbox);
        for k in 0..5 {
            track.landmarks[k] = track.landmark_filters[k].filter(det.landmarks[k], dt);
        }
        track.hits += 1;
        track.coast_time = 0.0;
        if track.hits >= self.params.min_hits {
            track.confirmed = true;
        }
    }

    fn birth(&mut self, det: &Detection) {
        let id: u32 = self.next_id;
        self.next_id += 1;
        let mut filters: [OneEuroPoint; 5] =
            [(); 5].map(|()| OneEuroPoint::new(self.params.min_cutoff, self.params.beta));
        let mut landmarks: [(f32, f32); 5] = det.landmarks;
        for k in 0..5 {
            landmarks[k] = filters[k].filter(det.landmarks[k], 1.0 / 30.0);
        }
        self.tracks.push(Track {
            id,
            kf: BoxKalman::from_box(&det.bbox, self.params.q, self.params.r),
            landmark_filters: filters,
            landmarks,
            hits: 1,
            coast_time: 0.0,
            confirmed: self.params.min_hits <= 1,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::{Kf1d, Tracker, TrackerParams};
    use crate::detect::{Bbox, Detection, Landmarks};

    const DT: f32 = 1.0 / 30.0;
    const LM: Landmarks = [(0.0, 0.0); 5];

    // A detection centered at (cx, cy) with size w×h and confidence `score`.
    fn det(cx: f32, cy: f32, w: f32, h: f32, score: f32) -> Detection {
        Detection {
            bbox: Bbox {
                x1: cx - w * 0.5,
                y1: cy - h * 0.5,
                x2: cx + w * 0.5,
                y2: cy + h * 0.5,
            },
            score,
            landmarks: LM,
        }
    }

    fn center(b: &Bbox) -> (f32, f32) {
        ((b.x1 + b.x2) * 0.5, (b.y1 + b.y2) * 0.5)
    }

    #[test]
    fn kf1d_predict_advances_by_velocity_and_update_pulls_to_measurement() {
        let mut kf: Kf1d = Kf1d::new(0.0, 40.0, 4.0);
        // Two measurements 1.0 apart at dt → the filter learns a positive velocity, so a predict
        // moves the position forward.
        kf.update(0.0);
        kf.predict(DT, 1.0);
        kf.update(1.0);
        let before: f32 = kf.p;
        kf.predict(DT, 1.0);
        assert!(kf.p > before, "predict advances along the learned velocity");
        assert!(kf.v > 0.0, "velocity estimated positive");
    }

    // Drives a confirmed, constant-velocity track and returns the tracker mid-motion.
    fn confirmed_moving_tracker(v: f32) -> (Tracker, f32, f32) {
        let mut tracker: Tracker = Tracker::new(TrackerParams::default());
        let (mut cx, cy) = (100.0, 100.0);
        for _ in 0..6 {
            tracker.update(&[det(cx, cy, 40.0, 40.0, 0.9)], DT);
            cx += v * DT;
        }
        (tracker, cx, cy)
    }

    #[test]
    fn coasts_through_missed_detections_without_a_gap() {
        // Track at 60 px/s, detector drops 0.2 s (< 0.4 s coast). Expect output every frame, near the truth.
        let v: f32 = 60.0;
        let (mut tracker, mut cx, cy) = confirmed_moving_tracker(v);
        for i in 0..6 {
            let out: Vec<super::TrackOutput> = tracker.update(&[], DT);
            cx += v * DT;
            assert!(!out.is_empty(), "no gap frame during the coast (frame {i})");
            let (ox, _) = center(&out[0].bbox);
            assert!((ox - cx).abs() < 12.0, "coast tracks ground truth: {ox} vs {cx}");
            assert!(out[0].fade > 0.9, "still solid early in the coast");
        }
        let _ = cy;
    }

    #[test]
    fn lost_track_fades_to_zero_then_is_removed() {
        // After detections stop, the fade never goes up, it drops near zero, and the track goes away.
        let (mut tracker, _, _) = confirmed_moving_tracker(0.0);
        // Fine timestep so the ~4-frame fade window resolves smoothly through the near-zero region.
        let fine_dt: f32 = 1.0 / 120.0;
        let mut prev_fade: f32 = 1.0f32;
        let mut removed: bool = false;
        let mut min_fade: f32 = 1.0f32;
        for _ in 0..80 {
            let out: Vec<super::TrackOutput> = tracker.update(&[], fine_dt);
            match out.first() {
                Some(o) => {
                    assert!(o.fade <= prev_fade + 1e-6, "fade must not increase");
                    prev_fade = o.fade;
                    min_fade = min_fade.min(o.fade);
                }
                None => {
                    removed = true;
                    break;
                }
            }
        }
        assert!(removed, "the track is retired past the coast window");
        assert!(min_fade < 0.1, "fade reaches near zero before removal: {min_fade}");
    }

    #[test]
    fn reacquisition_after_a_gap_does_not_jump() {
        // Coast a few frames, then the detection reappears at ground truth: the pose stays continuous.
        let v: f32 = 60.0;
        let (mut tracker, mut cx, cy) = confirmed_moving_tracker(v);
        let mut last_coast: f32 = cx;
        for _ in 0..3 {
            let out: Vec<super::TrackOutput> = tracker.update(&[], DT);
            cx += v * DT;
            last_coast = center(&out[0].bbox).0;
        }
        // Reappear at ground truth.
        let out: Vec<super::TrackOutput> = tracker.update(&[det(cx, cy, 40.0, 40.0, 0.9)], DT);
        let reacquired: f32 = center(&out[0].bbox).0;
        assert!((reacquired - last_coast).abs() < 8.0, "no jump on reacquisition: {reacquired} vs {last_coast}");
    }

    #[test]
    fn low_confidence_alone_never_births_a_track() {
        // ByteTrack: low-confidence detections keep tracks alive but cannot start one.
        let mut tracker: Tracker = Tracker::new(TrackerParams::default());
        for _ in 0..10 {
            let out: Vec<super::TrackOutput> = tracker.update(&[det(100.0, 100.0, 40.0, 40.0, 0.3)], DT);
            assert!(out.is_empty(), "a 0.3-score detection never confirms a track");
        }
    }

    #[test]
    fn confirmation_needs_min_hits() {
        let mut tracker: Tracker = Tracker::new(TrackerParams::default());
        // First two high-confidence frames are tentative (min_hits = 3), then it confirms.
        assert!(tracker.update(&[det(100.0, 100.0, 40.0, 40.0, 0.9)], DT).is_empty());
        assert!(tracker.update(&[det(100.0, 100.0, 40.0, 40.0, 0.9)], DT).is_empty());
        assert_eq!(tracker.update(&[det(100.0, 100.0, 40.0, 40.0, 0.9)], DT).len(), 1);
    }
}
