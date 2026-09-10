//! LaughingMan library: capture, face detection, the MSDF logo overlay, and GPU presentation.
//! The `laughing-man` binary is a thin shim over [`run`]; orchestration lives here so
//! it is unit-testable. Coverage note: `gpu/**` and the device/ffmpeg I/O sources are excluded from
//! the floor (bulwark `ignore_regex`), verified by the offscreen render test + running the app.

pub mod capture;
pub mod cli;
pub mod detect;
mod gpu;
pub mod logo;
#[cfg(feature = "bake")]
pub mod mask;
pub mod num;
pub mod overlay;
pub mod shaders;

use std::time::Instant;

use anyhow::Context;

use std::path::PathBuf;

use crate::capture::{FrameSource, TestPattern, VideoFile, Webcam};
use crate::cli::Cli;
use crate::detect::{DemoFaceProvider, FaceProvider};

/// Builds the face provider: the SCRFD detector when a model is supplied (and the `detect` feature
/// is on), otherwise a synthetic demo face.
#[cfg(feature = "detect")]
fn build_face_provider(model: Option<PathBuf>) -> anyhow::Result<Box<dyn FaceProvider>> {
    match model {
        Some(path) => Ok(Box::new(crate::detect::session::ScrfdDetector::from_file(&path)?)),
        None => Ok(Box::new(DemoFaceProvider)),
    }
}

#[cfg(not(feature = "detect"))]
fn build_face_provider(model: Option<PathBuf>) -> anyhow::Result<Box<dyn FaceProvider>> {
    if model.is_some() {
        eprintln!("--model ignored: rebuild with `--features detect` to run the SCRFD detector");
    }
    Ok(Box::new(DemoFaceProvider))
}

/// Runs the application for the given parsed arguments.
pub fn run(args: Cli) -> anyhow::Result<()> {
    if args.list {
        let cameras = capture::webcam::list_cameras()?;
        print!("{}", list_report(&cameras));
        return Ok(());
    }

    if let Some(video) = &args.video {
        // Video-file source: decode with ffmpeg and composite over each tracked face.
        let source = VideoFile::open(video)
            .with_context(|| format!("opening video {}", video.display()))?;
        return present_source(Box::new(source), &args);
    }

    if args.test_pattern {
        // No-camera path: a synthetic scrolling pattern with the demo/real overlay (smoke test).
        let (width, height) = args.size.unwrap_or((640, 480));
        return present_source(Box::new(TestPattern::new(width, height)), &args);
    }

    let webcam = Webcam::open(args.source, args.size)
        .with_context(|| format!("opening webcam source {}", args.source))?;

    if args.probe {
        // Headless path: report dimensions + measured rate without opening a window.
        let mut webcam = webcam;
        let reported_fps = webcam.frame_rate();
        return capture_loop(&mut webcam, args.source, reported_fps, args.frames);
    }

    present_source(Box::new(webcam), &args)
}

/// Composites the Laughing Man mask over `source` in a window — the single presentation entry
/// point every frame source dispatches to.
fn present_source(source: Box<dyn FrameSource>, args: &Cli) -> anyhow::Result<()> {
    let faces = build_face_provider(args.model.clone())?;
    gpu::present_window(
        source,
        faces,
        args.source,
        gpu::DEFAULT_RING_OMEGA,
        args.logo_static.clone(),
        args.logo_text.clone(),
        args.logo_front.clone(),
        args.present_frames,
    )
}

/// Reports the first frame, then measures the achieved capture rate over `frames` frames. Generic
/// over [`FrameSource`] so the orchestration is testable with a fake source (no camera required);
/// the concrete `Webcam` supplies frames in `run`.
fn capture_loop(
    source: &mut impl FrameSource,
    index: u32,
    reported_fps: u32,
    frames: u32,
) -> anyhow::Result<()> {
    // Prime the stream and report what the source actually gave us (which may differ from the
    // requested size).
    let first = source.next_frame().context("capturing first frame")?;
    let (rw, rh) = source.dimensions();
    println!(
        "{}",
        report_line(
            index,
            rw,
            rh,
            reported_fps,
            first.width,
            first.height,
            first.rgb.len(),
        )
    );

    let n = frames.max(1);
    let start = Instant::now();
    for _ in 0..n {
        source
            .next_frame()
            .context("capturing frame during rate probe")?;
    }
    let fps = measured_fps(n, start.elapsed().as_secs_f64());
    println!("measured {fps:.1} fps over {n} frames");
    Ok(())
}

/// Formats the `--list` report: a device count followed by one indented line per device.
fn list_report(cameras: &[String]) -> String {
    let mut out = format!("{} capture device(s) found:\n", cameras.len());
    for line in cameras {
        out.push_str("  ");
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// Formats the post-open camera report line (reported resolution/rate and the decoded frame).
fn report_line(
    source: u32,
    reported_w: u32,
    reported_h: u32,
    fps: u32,
    frame_w: u32,
    frame_h: u32,
    bytes: usize,
) -> String {
    format!(
        "camera {source}: reported {reported_w}x{reported_h} @ {fps} fps, \
         decoded frame {frame_w}x{frame_h} ({bytes} bytes RGB)"
    )
}

/// Frames captured per second, given a frame count and the elapsed seconds. `frames` is clamped to
/// at least 1 so a zero-count probe can't divide the count to a meaningless zero.
fn measured_fps(frames: u32, elapsed_secs: f64) -> f64 {
    f64::from(frames.max(1)) / elapsed_secs
}

#[cfg(test)]
mod tests {
    use super::{capture_loop, list_report, measured_fps, report_line, run};
    use crate::capture::{Frame, FrameSource};
    use crate::cli::Cli;

    /// A source that yields the same tiny frame forever, counting how many times it was polled.
    struct FakeSource {
        polls: usize,
    }

    impl FrameSource for FakeSource {
        fn next_frame(&mut self) -> anyhow::Result<Frame> {
            self.polls += 1;
            // 2x2 RGB8 = 12 bytes.
            Ok(Frame {
                width: 2,
                height: 2,
                rgb: vec![0u8; 12],
            })
        }
        fn dimensions(&self) -> (u32, u32) {
            (2, 2)
        }
    }

    /// A source that always fails, to exercise error propagation.
    struct FailingSource;

    impl FrameSource for FailingSource {
        fn next_frame(&mut self) -> anyhow::Result<Frame> {
            anyhow::bail!("simulated capture failure")
        }
        fn dimensions(&self) -> (u32, u32) {
            (0, 0)
        }
    }

    #[test]
    fn capture_loop_polls_first_plus_frames() {
        let mut src = FakeSource { polls: 0 };
        assert!(capture_loop(&mut src, 0, 30, 3).is_ok());
        // One prime read plus three probe reads.
        assert_eq!(src.polls, 4);
    }

    #[test]
    fn capture_loop_clamps_zero_frames_to_one_probe() {
        let mut src = FakeSource { polls: 0 };
        assert!(capture_loop(&mut src, 0, 30, 0).is_ok());
        // Prime read plus one clamped probe read.
        assert_eq!(src.polls, 2);
    }

    #[test]
    fn capture_loop_propagates_source_error() {
        assert!(capture_loop(&mut FailingSource, 0, 30, 5).is_err());
    }

    #[test]
    fn list_report_zero_devices() {
        assert_eq!(list_report(&[]), "0 capture device(s) found:\n");
    }

    #[test]
    fn list_report_counts_and_indents() {
        let report = list_report(&["[0] Cam — usb".to_owned(), "[1] Other — usb".to_owned()]);
        assert!(report.starts_with("2 capture device(s) found:\n"));
        assert!(report.contains("\n  [0] Cam — usb\n"));
        assert!(report.ends_with("  [1] Other — usb\n"));
    }

    #[test]
    fn report_line_formats_all_fields() {
        let line = report_line(0, 1280, 720, 30, 1280, 720, 2_764_800);
        assert_eq!(
            line,
            "camera 0: reported 1280x720 @ 30 fps, decoded frame 1280x720 (2764800 bytes RGB)"
        );
    }

    #[test]
    fn measured_fps_divides_count_by_time() {
        assert!((measured_fps(60, 2.0) - 30.0).abs() < 1e-9);
    }

    #[test]
    fn measured_fps_clamps_zero_frames() {
        assert!((measured_fps(0, 1.0) - 1.0).abs() < 1e-9);  // frames clamped to >= 1, so the result stays finite rather than 0/elapsed nonsense.
    }

    #[test]
    fn run_list_branch_succeeds_headless() {
        // Enumerating devices must succeed on any machine (0 or more devices), never error/panic.
        let cli = Cli {
            source: 0,
            size: None,
            list: true,
            probe: false,
            test_pattern: false,
            present_frames: None,
            model: None,
            logo_static: None,
            logo_text: None,
            logo_front: None,
            video: None,
            frames: 1,
        };
        assert!(run(cli).is_ok());
    }

    #[test]
    fn run_capture_errors_on_absent_device() {
        let cli = Cli {
            source: 9999,
            size: None,
            list: false,
            probe: true,
            test_pattern: false,
            present_frames: None,
            model: None,
            logo_static: None,
            logo_text: None,
            logo_front: None,
            video: None,
            frames: 1,
        }; // A wildly out-of-range index cannot open; run must return the contextual error, not panic.
        assert!(run(cli).is_err());
    }
}
