//! The SCRFD ONNX session (feature `detect`). Loads the model once, then per frame: preprocess →
//! run → feed the nine stride outputs to the pure [`scrfd::assemble`]. The InsightFace SCRFD_*_KPS
//! weights are user-provided (non-commercial license — fetched at runtime, never committed); YuNet
//! (MIT) is the license-clean swap behind the same [`Detector`] trait.
#![allow(clippy::as_conversions, clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use std::path::Path;

use anyhow::Context as _;
use ort::session::Session;
use ort::value::Tensor;

use crate::detect::scrfd::{
    self, INPUT_SIZE, NMS_THRESHOLD, SCORE_THRESHOLD, STRIDES, StrideOutputs,
};
use crate::detect::{Detection, Detector, FaceProvider};

// SCRFD emits nine output tensors: scores[0..3], bbox_preds[3..6], kps_preds[6..9], one per stride.
const NUM_OUTPUTS: usize = 9;

/// A SCRFD face detector backed by an ONNX Runtime session.
pub struct ScrfdDetector {
    session: Session,
    score_threshold: f32,
    nms_threshold: f32,
}

impl ScrfdDetector {
    /// Loads a SCRFD `*_kps` ONNX model from `model_path`.
    pub fn from_file(model_path: &Path) -> anyhow::Result<Self> {
        let session = Session::builder()
            .context("creating the ONNX session builder")?
            .commit_from_file(model_path)
            .with_context(|| format!("loading SCRFD model {}", model_path.display()))?;
        Ok(Self {
            session,
            score_threshold: SCORE_THRESHOLD,
            nms_threshold: NMS_THRESHOLD,
        })
    }

    /// Overrides the detection confidence threshold (default [`SCORE_THRESHOLD`]).
    #[must_use]
    pub fn with_score_threshold(mut self, threshold: f32) -> Self {
        self.score_threshold = threshold;
        self
    }
}

impl Detector for ScrfdDetector {
    fn detect(&mut self, rgb: &[u8], width: u32, height: u32) -> anyhow::Result<Vec<Detection>> {
        let scale = scrfd::letterbox_scale(width, height, INPUT_SIZE);
        let input = scrfd::preprocess(rgb, width, height, INPUT_SIZE);
        let dim = INPUT_SIZE as i64;
        let tensor = Tensor::from_array((vec![1_i64, 3, dim, dim], input))
            .context("building the input tensor")?;

        let outputs = self
            .session
            .run(ort::inputs![tensor])
            .context("running SCRFD inference")?;

        // Copy each output to an owned buffer so the borrowed session outputs can be released before
        // decoding (which borrows the owned buffers instead).
        let mut raw: Vec<Vec<f32>> = Vec::with_capacity(NUM_OUTPUTS);
        for i in 0..NUM_OUTPUTS {
            let (_shape, data) = outputs[i]
                .try_extract_tensor::<f32>()
                .with_context(|| format!("extracting output tensor {i}"))?;
            raw.push(data.to_vec());
        }
        drop(outputs);

        // For a square 640 input, each stride's feature map is 640/stride square, 2 anchors/cell.
        let per_stride: Vec<StrideOutputs> = (0..STRIDES.len())
            .map(|i| {
                let feat = INPUT_SIZE / STRIDES[i];
                (
                    raw[i].as_slice(),
                    raw[i + STRIDES.len()].as_slice(),
                    raw[i + 2 * STRIDES.len()].as_slice(),
                    feat,
                    feat,
                )
            })
            .collect();

        Ok(scrfd::assemble(
            &per_stride,
            scale,
            self.score_threshold,
            self.nms_threshold,
        ))
    }
}

impl FaceProvider for ScrfdDetector {
    fn detect_frame(
        &mut self,
        rgb: &[u8],
        width: u32,
        height: u32,
        _elapsed_secs: f32,
    ) -> anyhow::Result<Vec<Detection>> {
        self.detect(rgb, width, height)
    }
}
