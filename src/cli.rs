//! Command-line arguments: the webcam source, an optional requested resolution, device enumeration,
//! the headless probe, the test-pattern demo, and (with the `detect` feature) a SCRFD model path.

use std::path::PathBuf;

use clap::Parser;

/// Overlay the Laughing Man logo on faces in a webcam or video stream.
#[derive(Parser, Debug)]
#[command(name = "laughing-man", version, about)]
pub struct Cli {
    /// Webcam device index to capture from.
    #[arg(short, long, default_value_t = 0)]
    pub source: u32,

    /// Requested capture resolution, e.g. `1280x720`. Defaults to the camera's highest.
    #[arg(long, value_parser = parse_size)]
    pub size: Option<(u32, u32)>,

    /// List available capture devices and exit.
    #[arg(long)]
    pub list: bool,

    /// Print capture dimensions and the measured rate to stdout instead of opening a window.
    #[arg(long)]
    pub probe: bool,

    /// Decode this video file (mp4/mkv/…) as the frame source instead of the webcam. ffmpeg is
    /// auto-downloaded on first use if not installed.
    #[arg(long)]
    pub video: Option<PathBuf>,

    /// Present a synthetic scrolling test pattern instead of the webcam (no camera needed).
    #[arg(long)]
    pub test_pattern: bool,

    /// Exit after presenting this many frames (for smoke tests); default runs until the window closes.
    #[arg(long)]
    pub present_frames: Option<u32>,

    /// Path to a SCRFD `*_kps` ONNX model (requires the `detect` feature). Without it, a synthetic
    /// demo face drives the overlay.
    #[arg(long)]
    pub model: Option<PathBuf>,

    /// Baked static-layer MTSDF PNG (from `laughing-bake`). Given with `--logo-text`, the real logo
    /// is composited; otherwise a procedural placeholder is used.
    #[arg(long)]
    pub logo_static: Option<PathBuf>,

    /// Baked text-ring MTSDF PNG (from `laughing-bake`).
    #[arg(long)]
    pub logo_text: Option<PathBuf>,

    /// Radius (0..1 of the mask) of the white face disc painted behind the blue logo — the "white
    /// background" an all-blue logo SVG can't supply. 0 disables it (blue-only).
    #[arg(long, default_value_t = 0.46)]
    pub face_fill: f32,

    /// Number of frames to sample when measuring the capture rate.
    #[arg(long, default_value_t = 120)]
    pub frames: u32,
}

/// Parses a `WIDTHxHEIGHT` string (e.g. `640x480`) into a `(width, height)` pair.
fn parse_size(raw: &str) -> Result<(u32, u32), String> {
    let (w, h) = raw
        .split_once(['x', 'X'])
        .ok_or_else(|| format!("expected WIDTHxHEIGHT, got `{raw}`"))?;
    let width = w.trim().parse::<u32>().map_err(|e| e.to_string())?;
    let height = h.trim().parse::<u32>().map_err(|e| e.to_string())?;
    if width == 0 || height == 0 {
        return Err("width and height must both be non-zero".to_owned());
    }
    Ok((width, height))
}

#[cfg(test)]
mod tests {
    use super::parse_size;

    #[test]
    fn parses_lowercase_x() {
        assert_eq!(parse_size("640x480"), Ok((640, 480)));
    }

    #[test]
    fn parses_uppercase_x_and_whitespace() {
        assert_eq!(parse_size(" 1280 X 720 "), Ok((1280, 720)));
    }

    #[test]
    fn rejects_missing_separator() {
        assert!(parse_size("640-480").is_err());
    }

    #[test]
    fn rejects_zero_dimension() {
        assert!(parse_size("0x480").is_err());
        assert!(parse_size("640x0").is_err());
    }

    #[test]
    fn rejects_non_numeric() {
        assert!(parse_size("wide x tall").is_err());
    }
}
