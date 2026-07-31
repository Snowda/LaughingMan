//! Landmarks → per-face mask placement, and a CPU mirror of the compositor's per-pixel math.
//!
//! [`face_instance`] turns a [`Detection`] into the GPU `FaceInstance` the compositor shader loops
//! over: the face center + half-size (frame pixels), the roll and ring-phase rotations precomputed
//! as cos/sin (so the shader needs no trig), the screen-space AA range, and a track fade. Rotations
//! come from the eye line (landmarks 0/1). [`composite_pixel`] is the exact per-pixel math the
//! fragment shader performs, kept here so the compositing logic is verified analytically on the CPU
//! (the shader is a faithful DSL translation, reflection-checked in `shaders`).
#![allow(clippy::as_conversions)]

use crate::detect::Detection;

/// Maximum faces composited per frame (the compositor storage buffer / uniform bound).
pub const MAX_FACES: usize = 8;
/// The mask art overhangs the face box; scale the box up to cover the whole head.
pub const DEFAULT_COVER_SCALE: f32 = 1.6;
/// f32 per `FaceInstance` in the std430 storage buffer. All-scalar (no vec2), so struct align is 4
/// and the array stride is exactly 9·4 = 36 bytes.
pub const FACE_FLOATS: usize = 9;

/// One face's GPU instance data. `#[repr(C)]` and std430-compatible: it matches the shader's `Face`
/// storage struct field-for-field (`center_x`, `center_y`, then the seven placement scalars).
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

/// Builds the mask instance for `det`: center = box center; half-size = the larger box dimension
/// scaled by `cover_scale`; roll = the eye-line angle (right eye − left eye); the ring phase spins
/// the text layer; `screen_px_range` is the MSDF AA range at this on-screen size (`px_range` texels
/// spread over the mask's screen extent), floored at 1.
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
    let (lex, ley) = det.landmarks[0];
    let (rex, rey) = det.landmarks[1];
    let roll = (rey - ley).atan2(rex - lex);
    let box_size = det.bbox.width().max(det.bbox.height());
    let half_size = box_size * 0.5 * cover_scale;
    let screen_px_range = (2.0 * half_size / atlas_size * px_range).max(1.0);
    FaceInstance {
        center: [cx, cy],
        half_size,
        roll_cos: roll.cos(),
        roll_sin: roll.sin(),
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

/// Like [`pack_faces`] but always emits exactly [`MAX_FACES`] instances, padding with invisible
/// (`fade = 0`) faces so the compositor's storage buffer has a fixed length and the shader can loop
/// a constant count.
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

/// Mask-local radius of the white face disc (the "white background" the all-blue logo SVG can't
/// supply). Must match the constant in the compositor shader.
pub const FACE_DISC_RADIUS: f32 = 0.46;

/// The compositor fragment's per-pixel math, on the CPU. For each face: rotate the pixel offset by
/// `-roll` into mask-local space, and if it lands within the mask, paint the `white` face disc, then
/// sample the static + (ring-phase-rotated) text MSDF layers, take the max of their medians as the
/// coverage, and blend `blue` (the logo detail) on top. `sample_static`/`sample_text` return the
/// MSDF texel `(r,g,b)` at a mask-local `(u, v)`.
pub fn composite_pixel(
    video: [f32; 3],
    faces: &[FaceInstance],
    px: (f32, f32),
    white: [f32; 3],
    blue: [f32; 3],
    sample_static: impl Fn(f32, f32) -> [f32; 3],
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
            // White face background disc, AA'd by the signed pixel distance to its edge.
            let (hx, hy) = (mu - 0.5, mv - 0.5);
            let radius = (hx * hx + hy * hy).sqrt();
            let disc = ((FACE_DISC_RADIUS - radius) * 2.0 * face.half_size + 0.5).clamp(0.0, 1.0)
                * face.fade;
            color = mix3(color, white, disc);

            // Blue logo detail on top.
            let s = sample_static(mu, mv);
            let (tx, ty) = (mu - 0.5, mv - 0.5);
            let tu = face.phase_cos * tx - face.phase_sin * ty + 0.5;
            let tv = face.phase_sin * tx + face.phase_cos * ty + 0.5;
            let t = sample_text(tu, tv);
            let sd = median3(s).max(median3(t));
            let opacity = (face.screen_px_range * (sd - 0.5) + 0.5).clamp(0.0, 1.0) * face.fade;
            color = mix3(color, blue, opacity);
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
    fn horizontal_eyes_give_zero_roll() {
        let det = detection(Bbox { x1: 0.0, y1: 0.0, x2: 100.0, y2: 100.0 }, (30.0, 40.0), (70.0, 40.0));
        let fi = face_instance(&det, DEFAULT_COVER_SCALE, ATLAS, RANGE, 0.0, 1.0);
        assert!((fi.roll_cos - 1.0).abs() < 1e-5, "cos 0 = 1");
        assert!(fi.roll_sin.abs() < 1e-5, "sin 0 = 0");
    }

    #[test]
    fn tilted_eyes_give_the_eye_line_angle() {
        // Right eye 40px right and 40px down of the left eye → roll = atan2(40,40) = 45°.
        let det = detection(Bbox { x1: 0.0, y1: 0.0, x2: 100.0, y2: 100.0 }, (30.0, 30.0), (70.0, 70.0));
        let fi = face_instance(&det, 1.0, ATLAS, RANGE, 0.0, 1.0);
        let inv_sqrt2 = 1.0 / 2.0_f32.sqrt();
        assert!((fi.roll_cos - inv_sqrt2).abs() < 1e-5);
        assert!((fi.roll_sin - inv_sqrt2).abs() < 1e-5);
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

    fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
        (0..3).map(|c| (a[c] - b[c]).abs()).sum()
    }

    #[test]
    fn no_faces_is_passthrough() {
        // Feature off: with no faces the pixel is the untouched video color.
        let out = composite_pixel(VIDEO, &[], (50.0, 50.0), WHITE, BLUE, |_, _| [1.0; 3], |_, _| [1.0; 3]);
        assert_eq!(out, VIDEO);
    }

    #[test]
    fn zero_detail_still_paints_the_white_face() {
        // Inside the face disc but no MSDF detail (median 0): the white face background still shows —
        // the "white component" fix. A detail-only composite would leave the video here.
        let out = composite_pixel(
            VIDEO,
            &[centered_face(4.0)],
            (50.0, 50.0),
            WHITE,
            BLUE,
            |_, _| [0.0; 3],
            |_, _| [0.0; 3],
        );
        assert!(dist(out, WHITE) < dist(out, VIDEO), "center is the white face, not video: {out:?}");
    }

    #[test]
    fn full_detail_paints_blue_over_the_white_face() {
        // MSDF fully inside (median 1) → the blue logo detail covers the white face at this pixel.
        let out = composite_pixel(
            VIDEO,
            &[centered_face(4.0)],
            (50.0, 50.0),
            WHITE,
            BLUE,
            |_, _| [1.0; 3],
            |_, _| [1.0; 3],
        );
        for c in 0..3 {
            assert!((out[c] - BLUE[c]).abs() < 1e-4, "channel {c} is blue detail");
        }
    }

    #[test]
    fn logo_is_blue_detail_on_a_white_face() {
        // The all-blue synthetic logo composited over video: the open middle of the face shows the
        // WHITE background (the fix — not video, not blue), while a linework texel reads blue.
        const N: u32 = 64;
        let static_buf = crate::logo::synthetic_static(N, 8.0);
        let text_buf = crate::logo::synthetic_text(N, 8.0, 8);
        let sample = |buf: &[u8], u: f32, v: f32| -> [f32; 3] {
            let x = (u.clamp(0.0, 1.0) * (N as f32 - 1.0)) as u32;
            let y = (v.clamp(0.0, 1.0) * (N as f32 - 1.0)) as u32;
            let i = ((y * N + x) * 4) as usize;
            let c = f32::from(buf[i]) / 255.0;
            [c, c, c]
        };
        let face = centered_face(8.0);
        let out = composite_pixel(VIDEO, &[face], (50.0, 50.0), WHITE, BLUE, |u, v| sample(&static_buf, u, v), |u, v| sample(&text_buf, u, v));
        // Center is the open face middle → the white background, not video.
        assert!(dist(out, WHITE) < dist(out, VIDEO), "face center is white, not video: {out:?}");
    }

    #[test]
    fn roll_rotates_which_mask_texel_a_pixel_samples() {
        // A static layer "inside" only on its bottom half (v > 0.5). The pixel is in the mask corner
        // (outside the white face disc, so only the blue detail is in play). With roll 0 it maps to
        // the mask's bottom half → blue paints; roll 90° maps the same pixel to the top half → no
        // paint. Same pixel, different coverage: the overlay is genuinely rotated.
        let bottom_inside = |_u: f32, v: f32| if v > 0.5 { [1.0; 3] } else { [0.0; 3] };
        let px = (95.0, 70.0); // corner-ward of center (50,50): mask radius ≈ 0.49 > the disc's 0.46

        let no_roll = composite_pixel(VIDEO, &[centered_face(8.0)], px, WHITE, BLUE, bottom_inside, |_, _| [0.0; 3]);
        assert!(no_roll != VIDEO, "roll 0: maps to the mask's bottom half → paints");

        let mut rolled = centered_face(8.0);
        rolled.roll_cos = 0.0;
        rolled.roll_sin = 1.0; // roll = 90°
        let rolled_out =
            composite_pixel(VIDEO, &[rolled], px, WHITE, BLUE, bottom_inside, |_, _| [0.0; 3]);
        assert_eq!(rolled_out, VIDEO, "roll 90°: the same pixel maps to the top half → no paint");
    }
}
