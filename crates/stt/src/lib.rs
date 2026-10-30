//! Speech-to-text for Reach, backed by whisper-rs.
//!
//! The model is loaded once by [`Transcriber::load`] and reused for every
//! utterance. Audio samples and recognised text are never written to disk and
//! never logged.
//!
//! [`clean_transcript`] removes Whisper's noise annotations and rejects the
//! short hallucinations it produces on silence.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod clean;
pub mod transcriber;

pub use clean::clean_transcript;
pub use transcriber::{
    DEFAULT_INITIAL_PROMPT, MAX_SAMPLES, MIN_SAMPLES, SAMPLE_RATE, SttConfig, SttError, Transcriber,
};
