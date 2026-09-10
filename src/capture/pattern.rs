//! A synthetic animated frame source: scrolling SMPTE-style vertical color bars. Lets the presenter
//! run (and be verified) without a camera, and gives the user a zero-hardware demo/smoke test.

use crate::capture::{Frame, FrameSource};
use crate::num::Cast as _;

// Eight classic color bars (RGB), left to right.
const BARS: [[u8; 3]; 8] = [
    [255, 255, 255],
    [255, 255, 0],
    [0, 255, 255],
    [0, 255, 0],
    [255, 0, 255],
    [255, 0, 0],
    [0, 0, 255],
    [0, 0, 0],
];

/// Generates scrolling color bars at a fixed resolution. The bars shift left each frame so motion
/// is visible — a static-image bug (or a frozen upload) shows up immediately.
pub struct TestPattern {
    width: u32,
    height: u32,
    frame: u32,
}

impl TestPattern {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width: width.max(1),
            height: height.max(1),
            frame: 0,
        }
    }

    // Builds one frame's RGB bytes for the current scroll offset. Pure, so it is unit-testable.
    fn render(&self) -> Vec<u8> {
        let bar_width = (self.width / 8).max(1);
        let mut rgb = Vec::with_capacity((self.width * self.height * 3).to_usize());
        for _y in 0..self.height {
            for x in 0..self.width {
                let shifted = (x + self.frame * 2) % self.width;
                let index = ((shifted / bar_width) % 8).to_usize();
                rgb.extend_from_slice(&BARS[index]);
            }
        }
        rgb
    }
}

impl FrameSource for TestPattern {
    fn next_frame(&mut self) -> anyhow::Result<Frame> {
        let rgb = self.render();
        self.frame = self.frame.wrapping_add(1);
        Ok(Frame {
            width: self.width,
            height: self.height,
            rgb,
        })
    }

    fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

#[cfg(test)]
mod tests {
    use super::TestPattern;
    use crate::capture::FrameSource;
    use std::error::Error;

    #[test]
    fn frame_is_correctly_sized_rgb() -> Result<(), Box<dyn Error>> {
        let mut pattern = TestPattern::new(64, 32);
        let frame = pattern.next_frame()?;
        assert_eq!(frame.width, 64);
        assert_eq!(frame.height, 32);
        assert_eq!(frame.rgb.len(), 64 * 32 * 3);
        Ok(())
    }

    #[test]
    fn scrolls_between_frames() -> Result<(), Box<dyn Error>> {
        let mut pattern = TestPattern::new(64, 8);
        let first = pattern.next_frame()?.rgb;
        let second = pattern.next_frame()?.rgb;
        // The scroll offset advances, so consecutive frames differ.
        assert_ne!(first, second);
        Ok(())
    }

    #[test]
    fn dimensions_are_clamped_nonzero() {
        assert_eq!(TestPattern::new(0, 0).dimensions(), (1, 1));
    }
}
