use std::path::PathBuf;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "laughing-man",
    version,
    about = "Overlay the Laughing Man logo on faces in a webcam or video stream."
)]
pub struct Cli {
    #[arg(short, long, default_value_t = 0, help = "Webcam device index to capture from.")]
    pub source: u32,

    #[arg(
        long,
        value_parser = parse_size,
        help = "Requested capture resolution, e.g. `1280x720`. Defaults to the camera's highest."
    )]
    pub size: Option<(u32, u32)>,

    #[arg(long, help = "List available capture devices and exit.")]
    pub list: bool,

    #[arg(long, help = "Print capture dimensions and the measured rate to stdout instead of opening a window.")]
    pub probe: bool,

    #[arg(
        long,
        help = "Decode this video file (mp4/mkv/…) as the frame source instead of the webcam. ffmpeg is auto-downloaded on first use if not installed."
    )]
    pub video: Option<PathBuf>,

    #[arg(long, help = "Present a synthetic scrolling test pattern instead of the webcam (no camera needed).")]
    pub test_pattern: bool,

    #[arg(long, help = "Exit after presenting this many frames (for smoke tests); default runs until the window closes.")]
    pub present_frames: Option<u32>,

    #[arg(
        long,
        help = "Path to a SCRFD `*_kps` ONNX model (requires the `detect` feature). Without it, a synthetic demo face drives the overlay."
    )]
    pub model: Option<PathBuf>,

    #[arg(
        long,
        help = "Baked static-layer MTSDF PNG (from `laughing-bake`). Given with `--logo-text`, the real logo is composited; otherwise a procedural placeholder is used."
    )]
    pub logo_static: Option<PathBuf>,

    #[arg(long, help = "Baked text-ring MTSDF PNG (from `laughing-bake`).")]
    pub logo_text: Option<PathBuf>,

    #[arg(
        long,
        help = "Baked front-layer MTSDF PNG (features + cap). Its silhouette occludes the rotating text ring so the hat reads as in front of it. Only its alpha is used, at load; not uploaded."
    )]
    pub logo_front: Option<PathBuf>,

    #[arg(long, default_value_t = 120, help = "Number of frames to sample when measuring the capture rate.")]
    pub frames: u32,
}

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
