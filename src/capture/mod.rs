//! Frame sources. Phase 0 provides the webcam; the video-file source (ffmpeg) arrives in Phase 6.
//! Every source yields tightly-packed RGB8 frames behind [`FrameSource`], so downstream stages
//! (detection, GPU upload) never depend on where a frame came from.

pub mod file;
pub mod pattern;
pub mod webcam;

pub use crate::capture::file::VideoFile;
pub use crate::capture::pattern::TestPattern;
pub use crate::capture::webcam::Webcam;

/// One captured frame: tightly-packed RGB8, exactly `width * height * 3` bytes.
#[derive(Clone)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

/// A source of RGB frames — the webcam now, a video file later.
pub trait FrameSource {
    /// Blocks for and returns the next frame.
    fn next_frame(&mut self) -> anyhow::Result<Frame>;

    /// The source's reported `(width, height)`.
    fn dimensions(&self) -> (u32, u32);
}
