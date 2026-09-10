pub mod file;
pub mod pattern;
pub mod webcam;

pub use crate::capture::file::VideoFile;
pub use crate::capture::pattern::TestPattern;
pub use crate::capture::webcam::Webcam;

#[derive(Clone)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

pub trait FrameSource {
    fn next_frame(&mut self) -> anyhow::Result<Frame>;

    fn dimensions(&self) -> (u32, u32);
}
