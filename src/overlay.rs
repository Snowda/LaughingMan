//! Landmarks → per-face mask placement, and a CPU mirror of the compositor's per-pixel math.
//! [`face_instance`] turns a [`Detection`] into the GPU `FaceInstance` the compositor loops over
//! (center + half-size, roll/ring-phase as cos/sin, AA range, fade; overlay kept level with the
//! camera). [`composite_pixel`] is the exact per-pixel shader math, kept here for CPU verification.

use crate::detect::Detection;

pub const MAX_FACES: usize = 8;
/// The mask art overhangs the face box; scale the box up to cover the whole head.
pub const DEFAULT_COVER_SCALE: f32 = 1.6;
/// f32 per `FaceInstance` in the std430 buffer; all-scalar, so the array stride is 9·4 = 36 bytes.
pub const FACE_FLOATS: usize = 9;

/// One face's GPU instance data. `#[repr(C)]` std430-compatible: matches the shader's `Face` struct.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceInstance {
    pub center: [f32; 2],
    pub half_size: f32,
    pub roll_cos: f32,
    pub roll_sin: f32,
    pub phase_cos: f32,
    pub phase_sin: f32,
    pub screen_px_range: f32,
    pub fade: f32,
}

impl FaceInstance {
    /// The instance as the flat f32 storage-buffer words, in struct order.
    #[must_use]
    pub fn to_floats(self) -> [f32; FACE_FLOATS] {
        [
            self.center[0],
            self.center[1],
            self.half_size,
            self.roll_cos,
            self.roll_sin,
            self.phase_cos,
            self.phase_sin,
            self.screen_px_range,
            self.fade,
        ]
    }
}

/// Builds the mask instance for `det`: center = box center; half-size = larger box dim × `cover_scale`;
/// overlay kept level (identity roll); ring phase spins the text; `screen_px_range` is the MSDF AA range, floored at 1.
#[must_use]
pub fn face_instance(
    det: &Detection,
    cover_scale: f32,
    atlas_size: f32,
    px_range: f32,
    ring_phase: f32,
    fade: f32,
) -> FaceInstance {
    let cx = (det.bbox.x1 + det.bbox.x2) * 0.5;
    let cy = (det.bbox.y1 + det.bbox.y2) * 0.5;
    let box_size = det.bbox.width().max(det.bbox.height());
    let half_size = box_size * 0.5 * cover_scale;
    let screen_px_range = (2.0 * half_size / atlas_size * px_range).max(1.0);
    FaceInstance {
        center: [cx, cy],
        half_size,
        // The overlay stays level with the camera, not the head — no roll from the eye line.
        roll_cos: 1.0,
        roll_sin: 0.0,
        phase_cos: ring_phase.cos(),
        phase_sin: ring_phase.sin(),
        screen_px_range,
        fade,
    }
}

/// Packs up to [`MAX_FACES`] instances into the storage buffer's flat f32 layout.
#[must_use]
pub fn pack_faces(faces: &[FaceInstance]) -> Vec<f32> {
    faces
        .iter()
        .take(MAX_FACES)
        .flat_map(|face| face.to_floats())
        .collect()
}

/// Like [`pack_faces`] but always emits [`MAX_FACES`] instances, padding with invisible (`fade = 0`) faces.
#[must_use]
pub fn pack_faces_padded(faces: &[FaceInstance]) -> Vec<f32> {
    let dummy = FaceInstance {
        center: [0.0, 0.0],
        half_size: 1.0,
        roll_cos: 1.0,
        roll_sin: 0.0,
        phase_cos: 1.0,
        phase_sin: 0.0,
        screen_px_range: 1.0,
        fade: 0.0,
    };
    let real = faces.len().min(MAX_FACES);
    let mut out = pack_faces(faces);
    for _ in real..MAX_FACES {
        out.extend(dummy.to_floats());
    }
    out
}

/// Median of three channels — the MSDF signed-distance reconstruction.
fn median3(c: [f32; 3]) -> f32 {
    c[0].max(c[1]).min(c[0].min(c[1]).max(c[2]))
}

fn mix3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] * (1.0 - t) + b[0] * t,
        a[1] * (1.0 - t) + b[1] * t,
        a[2] * (1.0 - t) + b[2] * t,
    ]
}

/// The compositor fragment's per-pixel math, on the CPU. Per face: rotate the pixel by `-roll` into
/// mask-local space; if inside, paint `white` (both alpha levels), the ring-phase text `blue` gated to
/// the band, then static `blue` linework. Static alpha: ~1.0 band, ~0.5 occluder, 0 outside; text about `pivot`.
pub fn composite_pixel(
    video: [f32; 3],
    faces: &[FaceInstance],
    px: (f32, f32),
    white: [f32; 3],
    blue: [f32; 3],
    pivot: (f32, f32),
    sample_static: impl Fn(f32, f32) -> [f32; 4],
    sample_text: impl Fn(f32, f32) -> [f32; 3],
) -> [f32; 3] {
    let mut color = video;
    for face in faces {
        let dx = px.0 - face.center[0];
        let dy = px.1 - face.center[1];
        // R(-roll) · (dx, dy), with roll_cos/roll_sin = cos/sin(roll).
        let lx = face.roll_cos * dx + face.roll_sin * dy;
        let ly = -face.roll_sin * dx + face.roll_cos * dy;
        let mu = lx / (2.0 * face.half_size) + 0.5;
        let mv = ly / (2.0 * face.half_size) + 0.5;
        if (0.0..=1.0).contains(&mu) && (0.0..=1.0).contains(&mv) {
            let s = sample_static(mu, mv);
            // Static alpha is 3-level: ~1.0 band (ring shows), ~0.5 occluder (ring hidden), 0 outside; both paint white.
            let a = s[3];
            let white_op = ((a - 0.25) * 4.0).clamp(0.0, 1.0) * face.fade;
            color = mix3(color, white, white_op);

            // Text ring, rotated about the pivot, gated to the band so the occluder level hides it.
            let (tx, ty) = (mu - pivot.0, mv - pivot.1);
            let tu = face.phase_cos * tx - face.phase_sin * ty + pivot.0;
            let tv = face.phase_sin * tx + face.phase_cos * ty + pivot.1;
            let t = sample_text(tu, tv);
            let text_gate = ((a - 0.75) * 4.0).clamp(0.0, 1.0);
            let text_cov = (face.screen_px_range * (median3(t) - 0.5) + 0.5).clamp(0.0, 1.0);
            color = mix3(color, blue, text_cov * text_gate * face.fade);

            // Static blue linework (rings, features, cap) on top.
            let static_cov = (face.screen_px_range * (median3([s[0], s[1], s[2]]) - 0.5) + 0.5).clamp(0.0, 1.0);
            color = mix3(color, blue, static_cov * face.fade);
        }
    }
    color
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_COVER_SCALE, FaceInstance, MAX_FACES, composite_pixel, face_instance, pack_faces,
    };
    use crate::detect::{Bbox, Detection};
    use crate::num::Cast as _;

    const ATLAS: f32 = 1024.0;
    const RANGE: f32 = 8.0;

    // A detection with a box and eyes at `(le, re)`; other landmarks unused here.
    fn detection(box_: Bbox, le: (f32, f32), re: (f32, f32)) -> Detection {
        Detection {
            bbox: box_,
            score: 0.99,
            landmarks: [le, re, (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
        }
    }

    #[test]
    fn center_and_size_from_the_box() {
        let det = detection(Bbox { x1: 100.0, y1: 100.0, x2: 200.0, y2: 200.0 }, (130.0, 140.0), (170.0, 140.0));
        let fi = face_instance(&det, 1.0, ATLAS, RANGE, 0.0, 1.0);
        assert_eq!(fi.center, [150.0, 150.0]);
        assert!((fi.half_size - 50.0).abs() < 1e-4, "half of the 100px box at scale 1");
    }

    #[test]
    fn overlay_stays_level_with_the_camera() {
        // The mask no longer rolls with the head: identity roll for horizontal AND tilted eyes.
        for (le, re) in [((30.0, 40.0), (70.0, 40.0)), ((30.0, 30.0), (70.0, 70.0))] {
            let det = detection(Bbox { x1: 0.0, y1: 0.0, x2: 100.0, y2: 100.0 }, le, re);
            let fi = face_instance(&det, DEFAULT_COVER_SCALE, ATLAS, RANGE, 0.0, 1.0);
            assert!((fi.roll_cos - 1.0).abs() < 1e-5, "level: cos = 1");
            assert!(fi.roll_sin.abs() < 1e-5, "level: sin = 0");
        }
    }

    #[test]
    fn screen_px_range_scales_and_floors_at_one() {
        // Large face: 2*half_size/atlas*range = 2*160/1024*8 = 2.5.
        let big = detection(Bbox { x1: 0.0, y1: 0.0, x2: 200.0, y2: 200.0 }, (60.0, 60.0), (140.0, 60.0));
        let fi = face_instance(&big, DEFAULT_COVER_SCALE, ATLAS, RANGE, 0.0, 1.0);
        assert!((fi.screen_px_range - 2.5).abs() < 1e-4);
        // Tiny face floors the range at 1 so AA never collapses.
        let tiny = detection(Bbox { x1: 0.0, y1: 0.0, x2: 4.0, y2: 4.0 }, (1.0, 1.0), (3.0, 1.0));
        let fi = face_instance(&tiny, 1.0, ATLAS, RANGE, 0.0, 1.0);
        assert!((fi.screen_px_range - 1.0).abs() < 1e-6);
    }

    #[test]
    fn pack_padded_is_always_max_faces_long() {
        use super::pack_faces_padded;
        let one = face_instance(
            &detection(Bbox { x1: 0.0, y1: 0.0, x2: 10.0, y2: 10.0 }, (3.0, 4.0), (7.0, 4.0)),
            1.0,
            ATLAS,
            RANGE,
            0.0,
            1.0,
        );
        let expected = MAX_FACES * super::FACE_FLOATS;
        assert_eq!(pack_faces_padded(&[]).len(), expected);
        assert_eq!(pack_faces_padded(&[one]).len(), expected);
        assert_eq!(pack_faces_padded(&vec![one; MAX_FACES + 5]).len(), expected);
        // The padding faces are invisible (fade 0): the last face's fade word is 0.
        let padded = pack_faces_padded(&[one]);
        let last_fade = padded[expected - super::FACE_FLOATS + 8];
        assert_eq!(last_fade, 0.0);
    }

    #[test]
    fn pack_flattens_and_caps_at_max_faces() {
        let one = face_instance(
            &detection(Bbox { x1: 0.0, y1: 0.0, x2: 10.0, y2: 10.0 }, (3.0, 4.0), (7.0, 4.0)),
            1.0,
            ATLAS,
            RANGE,
            0.0,
            1.0,
        );
        assert_eq!(pack_faces(&[one]).len(), super::FACE_FLOATS);
        let many = vec![one; MAX_FACES + 3];
        assert_eq!(pack_faces(&many).len(), MAX_FACES * super::FACE_FLOATS, "capped at MAX_FACES");
    }

    // A face centered at (50,50), half_size 50, no rotation, fade 1, AA range 4.
    fn centered_face(px_range: f32) -> FaceInstance {
        FaceInstance {
            center: [50.0, 50.0],
            half_size: 50.0,
            roll_cos: 1.0,
            roll_sin: 0.0,
            phase_cos: 1.0,
            phase_sin: 0.0,
            screen_px_range: px_range,
            fade: 1.0,
        }
    }

    const WHITE: [f32; 3] = [1.0, 1.0, 1.0];
    const BLUE: [f32; 3] = [0.137, 0.286, 0.549];
    const VIDEO: [f32; 3] = [0.2, 0.3, 0.4];
    const CENTER: (f32, f32) = (0.5, 0.5); // ring pivot = mask centre for these synthetic cases

    fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
        (0..3).map(|c| (a[c] - b[c]).abs()).sum()
    }

    #[test]
    fn no_faces_is_passthrough() {
        let out = composite_pixel(VIDEO, &[], (50.0, 50.0), WHITE, BLUE, CENTER,|_, _| [1.0; 4], |_, _| [1.0; 3]);
        assert_eq!(out, VIDEO);
    }

    #[test]
    fn silhouette_alpha_paints_the_white_face() {
        // Inside the silhouette (alpha 1) but no MSDF detail: the white face shows (the "white component" fix).
        let out = composite_pixel(
            VIDEO,
            &[centered_face(4.0)],
            (50.0, 50.0),
            WHITE,
            BLUE,
            CENTER,
            |_, _| [0.0, 0.0, 0.0, 1.0],
            |_, _| [0.0; 3],
        );
        assert!(dist(out, WHITE) < dist(out, VIDEO), "center is the white face, not video: {out:?}");
    }

    #[test]
    fn outside_the_silhouette_is_passthrough() {
        let out = composite_pixel(
            VIDEO,
            &[centered_face(4.0)],
            (50.0, 50.0),
            WHITE,
            BLUE,
            CENTER,
            |_, _| [0.0; 4],
            |_, _| [0.0; 3],
        );
        assert_eq!(out, VIDEO, "no silhouette, no detail → video: {out:?}");
    }

    #[test]
    fn full_detail_paints_blue_over_the_white_face() {
        let out = composite_pixel(
            VIDEO,
            &[centered_face(4.0)],
            (50.0, 50.0),
            WHITE,
            BLUE,
            CENTER,
            |_, _| [1.0; 4],
            |_, _| [1.0; 3],
        );
        for c in 0..3 {
            assert!((out[c] - BLUE[c]).abs() < 1e-4, "channel {c} is blue detail");
        }
    }

    #[test]
    fn stamped_logo_is_blue_detail_on_a_white_face() {
        // All-blue synthetic logo, silhouette-stamped: the open middle shows WHITE (flood-fill interior), not video.
        const N: u32 = 64;
        let mut static_buf = crate::logo::synthetic_static(N, 8.0);
        crate::logo::stamp_silhouette(&mut static_buf, N);
        let text_buf = crate::logo::synthetic_text(N, 8.0, 8);
        let sample4 = |buf: &[u8], u: f32, v: f32| -> [f32; 4] {
            let x = (u.clamp(0.0, 1.0) * (N.to_f32() - 1.0)).to_u32();
            let y = (v.clamp(0.0, 1.0) * (N.to_f32() - 1.0)).to_u32();
            let i = ((y * N + x) * 4).to_usize();
            [
                f32::from(buf[i]) / 255.0,
                f32::from(buf[i + 1]) / 255.0,
                f32::from(buf[i + 2]) / 255.0,
                f32::from(buf[i + 3]) / 255.0,
            ]
        };
        let sample3 = |buf: &[u8], u: f32, v: f32| -> [f32; 3] {
            let x = (u.clamp(0.0, 1.0) * (N.to_f32() - 1.0)).to_u32();
            let y = (v.clamp(0.0, 1.0) * (N.to_f32() - 1.0)).to_u32();
            let i = ((y * N + x) * 4).to_usize();
            let c = f32::from(buf[i]) / 255.0;
            [c, c, c]
        };
        let out = composite_pixel(VIDEO, &[centered_face(8.0)], (50.0, 50.0), WHITE, BLUE, CENTER,|u, v| sample4(&static_buf, u, v), |u, v| sample3(&text_buf, u, v));
        assert!(dist(out, WHITE) < dist(out, VIDEO), "face center is white, not video: {out:?}");
    }

    #[test]
    fn roll_rotates_which_mask_texel_a_pixel_samples() {
        // Static "inside" only on its bottom half (v>0.5), alpha 0. Pixel offset (+40,+10) from center:
        // roll 0 maps to the bottom half → blue paints; roll 90° maps to the top → no paint. Same pixel, rotated.
        let bottom_inside = |_u: f32, v: f32| if v > 0.5 { [1.0, 1.0, 1.0, 0.0] } else { [0.0; 4] };
        let px = (90.0, 60.0); // (+40, +10) from center (50,50)

        let no_roll = composite_pixel(VIDEO, &[centered_face(8.0)], px, WHITE, BLUE, CENTER,bottom_inside, |_, _| [0.0; 3]);
        assert!(no_roll != VIDEO, "roll 0: maps to the mask's bottom half → paints");

        let mut rolled = centered_face(8.0);
        rolled.roll_cos = 0.0;
        rolled.roll_sin = 1.0; // roll = 90°
        let rolled_out =
            composite_pixel(VIDEO, &[rolled], px, WHITE, BLUE, CENTER,bottom_inside, |_, _| [0.0; 3]);
        assert_eq!(rolled_out, VIDEO, "roll 90°: the same pixel maps to the top half → no paint");
    }

    #[test]
    fn text_ring_rotates_about_the_pivot_not_the_mask_center() {
        // 180° ring phase, pixel at mask-uv (0.7,0.5), marker at (0.5,0.5). About pivot (0.6,0.5) it
        // rotates to (0.5,0.5) → hits; about the mask centre → (0.3,0.5) misses. Proves the pivot is used.
        let mut face = centered_face(8.0);
        face.phase_cos = -1.0;
        face.phase_sin = 0.0;
        let px = (70.0, 50.0); // mask-uv (0.7, 0.5)
        // Band level (alpha 1.0), no static blue → the text gate is open, white shows.
        let band_static = |_: f32, _: f32| [0.0, 0.0, 0.0, 1.0];
        let marker = |u: f32, v: f32| if (u - 0.5).abs() < 0.05 && (v - 0.5).abs() < 0.05 { [1.0; 3] } else { [0.0; 3] };

        let about_pivot = composite_pixel(VIDEO, &[face], px, WHITE, BLUE, (0.6, 0.5), band_static, marker);
        // About the pivot the rotated sample hits the marker → blue text over the white face.
        assert!(dist(about_pivot, BLUE) < dist(about_pivot, WHITE), "pivot rotation shows the text: {about_pivot:?}");
        let about_center = composite_pixel(VIDEO, &[face], px, WHITE, BLUE, CENTER, band_static, marker);
        // About the mask centre it misses the marker → just the white face, no text.
        assert!(dist(about_center, WHITE) < dist(about_center, BLUE), "centre rotation misses the marker: {about_center:?}");
    }

    #[test]
    fn occluder_alpha_hides_the_text() {
        // Occluder alpha (0.5) with a text marker: text gated off (front layer occludes), only white shows.
        let face = centered_face(8.0);
        let marker = |_: f32, _: f32| [1.0; 3]; // text everywhere
        let occluder = |_: f32, _: f32| [0.0, 0.0, 0.0, 0.5];
        let band = |_: f32, _: f32| [0.0, 0.0, 0.0, 1.0];
        let hidden = composite_pixel(VIDEO, &[face], (50.0, 50.0), WHITE, BLUE, CENTER, occluder, marker);
        assert!(dist(hidden, WHITE) < dist(hidden, BLUE), "occluder level hides the text: {hidden:?}");
        let shown = composite_pixel(VIDEO, &[face], (50.0, 50.0), WHITE, BLUE, CENTER, band, marker);
        assert!(dist(shown, BLUE) < dist(shown, WHITE), "band level shows the text: {shown:?}");
    }
}
