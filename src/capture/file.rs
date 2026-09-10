//! Video-file frames via ffmpeg (`ffmpeg-sidecar`, auto-downloaded). A worker thread decodes to
//! rgb24 into a bounded channel; the last frame is held on EOF (a freeze) rather than erroring.

use std::path::Path;
use std::sync::mpsc::{Receiver, SyncSender, TryRecvError, sync_channel};
use std::thread::JoinHandle;
use std::time::Instant;

use anyhow::{Context as _, anyhow};
use ffmpeg_sidecar::command::FfmpegCommand;
use rodio::{Decoder, DeviceSinkBuilder, MixerDeviceSink, Player};

use crate::capture::{Frame, FrameSource};

// The audio output — the device sink and the player must stay alive for playback to continue.
struct Audio {
    _sink: MixerDeviceSink,
    _player: Player,
}

// Plays the file's own audio track (symphonia). `None` (logged, silent) when no device/stream/codec.
fn start_audio(path: &Path) -> Option<Audio> {
    let mut sink = DeviceSinkBuilder::open_default_sink().ok()?;
    // Silence rodio's "audio will stop" warning when we drop the sink on exit.
    sink.log_on_drop(false);
    let player = Player::connect_new(sink.mixer());
    let file = std::fs::File::open(path).ok()?;
    let decoder = Decoder::try_from(file).ok()?;
    player.append(decoder);
    Some(Audio {
        _sink: sink,
        _player: player,
    })
}

// Bounded so ffmpeg blocks (back-pressure) once a few frames are buffered ahead of the renderer.
const FRAME_BUFFER: usize = 8;

// A decoded frame plus its presentation timestamp (seconds), used to pace playback to real time.
type TimedFrame = (Frame, f32);

/// A video file decoded to RGB frames by a background ffmpeg process, presented in real time.
pub struct VideoFile {
    frames: Receiver<TimedFrame>,
    current: Option<TimedFrame>,
    pending: Option<TimedFrame>,
    start: Option<Instant>,
    width: u32,
    height: u32,
    _worker: JoinHandle<()>,
    _audio: Option<Audio>,
}

impl VideoFile {
    /// Opens `path`, ensuring an ffmpeg binary is available, and blocks until the first frame so the
    /// dimensions are known.
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        ffmpeg_sidecar::download::auto_download().context("obtaining an ffmpeg binary")?;

        // Start audio playback of the file's own track (best-effort; silent video if unavailable).
        let audio = start_audio(path);
        if audio.is_none() {
            eprintln!("audio: no output device or no decodable audio track; playing video silently");
        }

        let path = path
            .to_str()
            .ok_or_else(|| anyhow!("video path is not valid UTF-8"))?
            .to_owned();

        let (meta_tx, meta_rx) = std::sync::mpsc::channel::<anyhow::Result<(u32, u32)>>();
        let (frame_tx, frames) = sync_channel::<TimedFrame>(FRAME_BUFFER);
        let worker = std::thread::spawn(move || decode_loop(&path, &meta_tx, &frame_tx));

        let (width, height) = meta_rx.recv().context("ffmpeg produced no output")??;
        Ok(Self {
            frames,
            current: None,
            pending: None,
            start: None,
            width,
            height,
            _worker: worker,
            _audio: audio,
        })
    }
}

// Runs ffmpeg to completion: announce the first frame's dims on `meta_tx`, stream frames to `frame_tx`.
fn decode_loop(
    path: &str,
    meta_tx: &std::sync::mpsc::Sender<anyhow::Result<(u32, u32)>>,
    frame_tx: &SyncSender<TimedFrame>,
) {
    let mut child = match FfmpegCommand::new().input(path).rawvideo().spawn() {
        Ok(child) => child,
        Err(e) => {
            let _ = meta_tx.send(Err(anyhow!("spawning ffmpeg: {e}")));
            return;
        }
    };
    let iter = match child.iter() {
        Ok(iter) => iter,
        Err(e) => {
            let _ = meta_tx.send(Err(anyhow!("reading ffmpeg output: {e}")));
            return;
        }
    };

    let mut announced = false;
    for frame in iter.filter_frames() {
        if !announced {
            let _ = meta_tx.send(Ok((frame.width, frame.height)));
            announced = true;
        }
        let timed = (
            Frame {
                width: frame.width,
                height: frame.height,
                rgb: frame.data,
            },
            frame.timestamp,
        );
        if frame_tx.send(timed).is_err() {
            break; // The app dropped the receiver (exiting).
        }
    }
    if !announced {
        let _ = meta_tx.send(Err(anyhow!("no frames decoded from the video")));
    }
}

impl FrameSource for VideoFile {
    fn next_frame(&mut self) -> anyhow::Result<Frame> {
        let now = Instant::now();
        let start = *self.start.get_or_insert(now);
        let elapsed = (now - start).as_secs_f32();

        // Advance to the newest due frame; hold otherwise — pacing to real time, catching up if behind.
        loop {
            if self.pending.is_none() {
                match self.frames.try_recv() {
                    Ok(frame) => self.pending = Some(frame),
                    Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
                }
            }
            match &self.pending {
                Some((_, ts)) if *ts <= elapsed => self.current = self.pending.take(),
                _ => break,
            }
        }

        if let Some((frame, _)) = &self.current {
            return Ok(frame.clone());
        }
        // No frame shown yet — block for the first one so the window isn't empty.
        let (frame, ts) = self.frames.recv().map_err(|_| anyhow!("video ended before any frame"))?;
        self.current = Some((frame.clone(), ts));
        Ok(frame)
    }

    fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}
