//! Microphone capture, resampling and voice-activity detection for Reach.
//!
//! Audio is processed in memory only. Nothing in this crate writes samples to
//! disk or to a log. Capture produces mono 16 kHz `f32` audio, which the
//! endpointing state machine turns into complete utterances for speech-to-text.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod capture;
pub mod convert;
pub mod endpoint;

pub use capture::{MicCapture, MicCaptureError};
pub use convert::{
    ResampleTo16kError, downmix_to_mono, f32_to_f32, i16_to_f32, resample_to_16k, u16_to_f32,
};
pub use endpoint::{EndpointConfig, EndpointConfigError, Endpointer, Utterance};

/// Sample rate used throughout the speech pipeline.
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Number of samples in one endpointing frame: 20 ms at 16 kHz.
pub const FRAME_SAMPLES: usize = 320;
