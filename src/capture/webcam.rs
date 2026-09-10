//! Webcam capture via nokhwa (Media Foundation on Windows) → RGB8; format-selection split for tests.

use anyhow::Context;
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{
    ApiBackend, CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType,
    Resolution,
};
use nokhwa::{Camera, query};

use crate::capture::{Frame, FrameSource};

/// A live webcam opened on one device index.
pub struct Webcam {
    camera: Camera,
}

/// The requested capture format: closest match to `size`, else the device's highest. Always RGB8.
fn requested_format(size: Option<(u32, u32)>) -> RequestedFormat<'static> {
    match size {
        Some((w, h)) => RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(
            CameraFormat::new(Resolution::new(w, h), FrameFormat::YUYV, 30),
        )),
        None => RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestResolution),
    }
}

/// Formats one enumerated device as `[index] name — description`.
fn format_camera_line(index: &CameraIndex, name: &str, description: &str) -> String {
    format!("[{index}] {name} — {description}")
}

impl Webcam {
    /// Opens device `index`, requesting `size` (closest supported) or the highest resolution when
    /// `size` is `None`, then starts the capture stream.
    pub fn open(index: u32, size: Option<(u32, u32)>) -> anyhow::Result<Self> {
        let mut camera = Camera::new(CameraIndex::Index(index), requested_format(size))
            .with_context(|| format!("opening camera index {index}"))?;
        camera.open_stream().context("starting camera stream")?;
        Ok(Self { camera })
    }

    /// The camera's reported capture rate (frames per second).
    pub fn frame_rate(&self) -> u32 {
        self.camera.frame_rate()
    }
}

/// Enumerates available capture devices via the platform's native backend, formatted for display.
pub fn list_cameras() -> anyhow::Result<Vec<String>> {
    let infos = query(ApiBackend::Auto).context("querying capture devices")?;
    Ok(infos
        .iter()
        .map(|info| format_camera_line(info.index(), &info.human_name(), info.description()))
        .collect())
}

impl FrameSource for Webcam {
    fn next_frame(&mut self) -> anyhow::Result<Frame> {
        let buffer = self.camera.frame().context("capturing frame")?;
        let image = buffer
            .decode_image::<RgbFormat>()
            .context("decoding frame to RGB")?;
        Ok(Frame {
            width: image.width(),
            height: image.height(),
            rgb: image.into_raw(),
        })
    }

    fn dimensions(&self) -> (u32, u32) {
        let res = self.camera.resolution();
        (res.width(), res.height())
    }
}

#[cfg(test)]
mod tests {
    use super::{Webcam, format_camera_line, list_cameras, requested_format};
    use nokhwa::utils::CameraIndex;

    #[test]
    fn format_camera_line_layout() {
        let line = format_camera_line(&CameraIndex::Index(2), "HD Webcam", "USB Video");
        assert_eq!(line, "[2] HD Webcam — USB Video");
    }

    #[test]
    fn requested_format_builds_both_branches() {
        let _sized = requested_format(Some((640, 480)));
        let _highest = requested_format(None);
    }

    #[test]
    fn list_cameras_never_errors() {
        assert!(list_cameras().is_ok());
    }

    #[test]
    fn open_absent_device_errors() {
        assert!(Webcam::open(9999, None).is_err());
    }
}
