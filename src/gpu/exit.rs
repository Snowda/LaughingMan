//! Why the presenter's event loop ended, reported once it returns.

use std::fmt;
use std::time::Duration;

use winit::keyboard::KeyCode;

/// What ended the event loop. The first recorded reason wins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExitReason {
    CloseButton,
    Key(KeyCode),
    FrameLimit(u32),
    InitFailed,
}

/// The one-line exit summary: the reason (or that none was recorded), frames presented, and run time.
pub fn exit_report(reason: Option<ExitReason>, presented: u32, ran: Duration) -> impl fmt::Display {
    fmt::from_fn(move |f| {
        f.write_str("exit: ")?;
        match reason {
            Some(ExitReason::CloseButton) => f.write_str("window close button")?,
            Some(ExitReason::Key(code)) => write!(f, "{code:?} key")?,
            Some(ExitReason::FrameLimit(limit)) => write!(f, "--present-frames limit ({limit})")?,
            Some(ExitReason::InitFailed) => f.write_str("presenter init failed")?,
            None => f.write_str("no recorded reason (the OS or winit ended the loop)")?,
        }
        write!(f, " after {presented} frame(s) in {:.1} s", ran.as_secs_f64())
    })
}

#[cfg(test)]
mod tests {
    use super::{ExitReason, exit_report};
    use std::time::Duration;
    use winit::keyboard::KeyCode;

    #[test]
    fn each_reason_names_itself_with_frames_and_run_time() {
        let ran: Duration = Duration::from_millis(5_520);
        let cases: [(Option<ExitReason>, &str); 6] = [
            (Some(ExitReason::CloseButton), "exit: window close button after 37 frame(s) in 5.5 s"),
            (Some(ExitReason::Key(KeyCode::Space)), "exit: Space key after 37 frame(s) in 5.5 s"),
            (Some(ExitReason::Key(KeyCode::Escape)), "exit: Escape key after 37 frame(s) in 5.5 s"),
            (Some(ExitReason::FrameLimit(37)), "exit: --present-frames limit (37) after 37 frame(s) in 5.5 s"),
            (Some(ExitReason::InitFailed), "exit: presenter init failed after 37 frame(s) in 5.5 s"),
            (None, "exit: no recorded reason (the OS or winit ended the loop) after 37 frame(s) in 5.5 s"),
        ];
        for (reason, expected) in cases {
            assert_eq!(exit_report(reason, 37, ran).to_string(), expected, "report for {reason:?}");
        }
    }
}
