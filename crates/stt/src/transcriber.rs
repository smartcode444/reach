//! The whisper-rs transcriber.
//!
//! [`Transcriber::load`] loads the model once. Every later call to
//! [`Transcriber::transcribe`] reuses the same model and state. Audio samples
//! and recognised text are never written to disk and never logged.

use std::path::Path;

use reach_core::Heard;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

use crate::clean_transcript;

/// Sample rate expected by Whisper and by [`Transcriber::transcribe`].
pub const SAMPLE_RATE: u32 = 16_000;

/// Shortest accepted utterance: 300 ms at 16 kHz.
pub const MIN_SAMPLES: usize = (SAMPLE_RATE as usize * 300) / 1_000;

/// Longest accepted utterance: 15 s at 16 kHz.
pub const MAX_SAMPLES: usize = (SAMPLE_RATE as usize * 15_000) / 1_000;

/// Prompt that biases Whisper towards Reach's command vocabulary.
pub const DEFAULT_INITIAL_PROMPT: &str = "Reach, scroll down, scroll up, click submit, zoom in, zoom out, \
press enter, focus name field, type, open Chrome, confirm, cancel, stop.";

/// Settings for a [`Transcriber`].
///
/// Sampling is always greedy, timestamps and translation are always off, and
/// non-speech tokens are always suppressed. Those behaviours are part of the
/// Reach contract rather than user settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SttConfig {
    /// Number of decoding threads. Defaults to the available parallelism minus
    /// one, with a minimum of one, so the rest of the pipeline keeps a core.
    pub threads: usize,
    /// Language passed to Whisper. Defaults to `"en"`.
    pub language: String,
    /// Prompt used to bias decoding towards Reach's command words.
    pub initial_prompt: String,
}

impl Default for SttConfig {
    /// Returns the documented default settings.
    fn default() -> Self {
        Self {
            threads: default_threads(),
            language: "en".to_owned(),
            initial_prompt: DEFAULT_INITIAL_PROMPT.to_owned(),
        }
    }
}

/// Returns the default decoding thread count.
///
/// # Returns
/// `available_parallelism - 1`, with a minimum of one. If the platform cannot
/// report its parallelism, one thread is used.
fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|parallelism| parallelism.get().saturating_sub(1).max(1))
        .unwrap_or(1)
}

/// Reasons why loading or running the transcriber can fail.
#[derive(Debug, thiserror::Error)]
pub enum SttError {
    /// The model file could not be loaded.
    #[error("the speech model could not be loaded: {0}")]
    ModelLoad(#[source] whisper_rs::WhisperError),
    /// Whisper could not create its decoding state.
    #[error("the speech recogniser state could not be created: {0}")]
    StateInit(#[source] whisper_rs::WhisperError),
    /// A configuration string cannot be passed to whisper-rs safely.
    #[error("invalid speech-to-text configuration field: {field}")]
    InvalidConfig {
        /// Name of the offending field.
        field: &'static str,
    },
    /// The requested thread count cannot be passed to whisper-rs.
    #[error("invalid speech-to-text thread count: {threads}")]
    InvalidThreads {
        /// The rejected thread count.
        threads: usize,
    },
    /// The utterance is shorter than [`MIN_SAMPLES`].
    #[error("audio is too short: {samples} samples, minimum is {minimum}")]
    TooShort {
        /// Number of samples supplied.
        samples: usize,
        /// Smallest accepted number of samples.
        minimum: usize,
    },
    /// The utterance is longer than [`MAX_SAMPLES`].
    #[error("audio is too long: {samples} samples, maximum is {maximum}")]
    TooLong {
        /// Number of samples supplied.
        samples: usize,
        /// Largest accepted number of samples.
        maximum: usize,
    },
    /// Whisper failed while decoding.
    #[error("the speech recogniser failed: {0}")]
    Transcription(#[source] whisper_rs::WhisperError),
    /// Whisper returned text that is not valid UTF-8.
    #[error("the speech recogniser returned unreadable text: {0}")]
    Transcript(#[source] whisper_rs::WhisperError),
    /// The audio contained no usable speech after cleaning.
    #[error("no speech was recognised in the audio")]
    NoSpeech,
}

/// A loaded Whisper model and its reusable decoding state.
pub struct Transcriber {
    /// Decoding state. It keeps the loaded model alive through an internal
    /// reference-counted context.
    state: WhisperState,
    /// Decoding settings fixed when the model was loaded.
    config: SttConfig,
}

impl Transcriber {
    /// Loads the model at `model_path` and prepares it for transcription.
    ///
    /// The model is loaded exactly once. Keep the returned value alive and call
    /// [`transcribe`](Self::transcribe) for each utterance.
    ///
    /// # Arguments
    /// * `model_path` - Path to a whisper.cpp GGML model such as
    ///   `models/ggml-base.en.bin`.
    /// * `config` - Decoding settings.
    ///
    /// # Errors
    /// Returns [`SttError::InvalidThreads`] or [`SttError::InvalidConfig`] for
    /// unusable settings, and [`SttError::ModelLoad`] or
    /// [`SttError::StateInit`] when Whisper cannot initialise.
    pub fn load(model_path: impl AsRef<Path>, config: SttConfig) -> Result<Self, SttError> {
        // whisper.cpp writes decoded text and timing details to stderr by
        // default. Redirect its logs into whisper-rs, which discards them
        // because this crate enables neither the `log` nor the `tracing`
        // backend. This keeps transcripts out of logs, as AGENTS.md requires.
        whisper_rs::install_logging_hooks();
        validate_config(&config)?;

        let context = WhisperContext::new_with_params(
            model_path.as_ref(),
            WhisperContextParameters::default(),
        )
        .map_err(SttError::ModelLoad)?;
        let state = context.create_state().map_err(SttError::StateInit)?;

        Ok(Self { state, config })
    }

    /// Transcribes one mono 16 kHz `f32` utterance.
    ///
    /// # Arguments
    /// * `samples` - Mono samples at [`SAMPLE_RATE`]. The length must be
    ///   between [`MIN_SAMPLES`] and [`MAX_SAMPLES`].
    ///
    /// # Returns
    /// The cleaned text and, when token probabilities are available, their
    /// average as a confidence score.
    ///
    /// # Errors
    /// Returns [`SttError::TooShort`] or [`SttError::TooLong`] for audio
    /// outside the accepted range, [`SttError::Transcription`] or
    /// [`SttError::Transcript`] when Whisper fails, and [`SttError::NoSpeech`]
    /// when cleaning removes the whole result.
    pub fn transcribe(&mut self, samples: &[f32]) -> Result<Heard, SttError> {
        validate_sample_count(samples.len())?;

        let config = &self.config;
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 5 });
        params.set_n_threads(config.threads as i32);
        params.set_language(Some(config.language.as_str()));
        params.set_initial_prompt(config.initial_prompt.as_str());
        params.set_translate(false);
        params.set_no_timestamps(true);
        params.set_suppress_blank(true);
        params.set_suppress_nst(true);
        // Every utterance is independent. Without this, Whisper would use the
        // previous transcript as context and could repeat it.
        params.set_no_context(true);
        // Keep whisper.cpp from writing transcripts or timing noise to stderr.
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        self.state
            .full(params, samples)
            .map_err(SttError::Transcription)?;

        let mut raw = String::new();
        let mut probability_sum = 0.0_f64;
        let mut probability_count = 0_usize;

        for segment in self.state.as_iter() {
            raw.push(' ');
            raw.push_str(segment.to_str().map_err(SttError::Transcript)?);

            for index in 0..segment.n_tokens() {
                let Some(token) = segment.get_token(index) else {
                    continue;
                };
                let token_text = token.to_str().map_err(SttError::Transcript)?;
                let trimmed = token_text.trim();
                // Special tokens such as "[_SOT_]" are not words and would
                // skew the confidence average.
                if trimmed.is_empty() || trimmed.starts_with("[_") {
                    continue;
                }
                let probability = token.token_probability();
                if probability.is_finite() {
                    probability_sum += f64::from(probability.clamp(0.0, 1.0));
                    probability_count += 1;
                }
            }
        }

        let confidence = if probability_count == 0 {
            None
        } else {
            Some((probability_sum / probability_count as f64) as f32)
        };
        let text = clean_transcript(&raw).ok_or(SttError::NoSpeech)?;

        Ok(Heard { text, confidence })
    }
}

/// Checks that the settings can be passed to whisper-rs without panicking.
///
/// # Arguments
/// * `config` - Settings to validate.
///
/// # Errors
/// Returns [`SttError::InvalidThreads`] for a thread count outside the
/// `i32` range, and [`SttError::InvalidConfig`] for an empty language or a
/// string containing a null byte.
fn validate_config(config: &SttConfig) -> Result<(), SttError> {
    if config.threads == 0 || config.threads > i32::MAX as usize {
        return Err(SttError::InvalidThreads {
            threads: config.threads,
        });
    }
    if config.language.trim().is_empty() || config.language.contains('\0') {
        return Err(SttError::InvalidConfig { field: "language" });
    }
    if config.initial_prompt.contains('\0') {
        return Err(SttError::InvalidConfig {
            field: "initial_prompt",
        });
    }
    Ok(())
}

/// Checks that `samples` is a length the transcriber accepts.
///
/// # Arguments
/// * `samples` - Number of mono 16 kHz samples.
///
/// # Errors
/// Returns [`SttError::TooShort`] below [`MIN_SAMPLES`] and
/// [`SttError::TooLong`] above [`MAX_SAMPLES`].
fn validate_sample_count(samples: usize) -> Result<(), SttError> {
    if samples < MIN_SAMPLES {
        return Err(SttError::TooShort {
            samples,
            minimum: MIN_SAMPLES,
        });
    }
    if samples > MAX_SAMPLES {
        return Err(SttError::TooLong {
            samples,
            maximum: MAX_SAMPLES,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// The default configuration uses English and the command prompt.
    fn default_config_uses_english_and_the_command_prompt() {
        let config = SttConfig::default();
        assert!(config.threads >= 1);
        assert_eq!(config.language, "en");
        assert_eq!(config.initial_prompt, DEFAULT_INITIAL_PROMPT);
    }

    #[test]
    /// The default leaves one core free, with a floor of one thread.
    fn default_threads_leave_one_core_free() {
        let expected = std::thread::available_parallelism()
            .map(|parallelism| parallelism.get().saturating_sub(1).max(1))
            .unwrap_or(1);
        assert_eq!(SttConfig::default().threads, expected);
    }

    #[test]
    /// The transcriber can move to the dedicated speech thread.
    fn transcriber_is_send() {
        /// Compile-time proof that `T` implements `Send`.
        fn assert_send<T: Send>() {}

        assert_send::<Transcriber>();
    }

    #[test]
    /// Sample-count boundaries are inclusive at 300 ms and 15 s.
    fn sample_length_boundaries_are_enforced() {
        assert!(matches!(
            validate_sample_count(MIN_SAMPLES - 1),
            Err(SttError::TooShort { .. })
        ));
        assert!(validate_sample_count(MIN_SAMPLES).is_ok());
        assert!(validate_sample_count(MAX_SAMPLES).is_ok());
        assert!(matches!(
            validate_sample_count(MAX_SAMPLES + 1),
            Err(SttError::TooLong { .. })
        ));
    }

    #[test]
    /// Zero threads are rejected before the model is touched.
    fn zero_threads_are_rejected_before_loading() {
        let config = SttConfig {
            threads: 0,
            ..SttConfig::default()
        };
        assert!(matches!(
            Transcriber::load("missing-model.bin", config),
            Err(SttError::InvalidThreads { threads: 0 })
        ));
    }

    #[test]
    /// Null bytes are rejected before whisper-rs can panic on them.
    fn null_bytes_are_rejected_before_loading() {
        let config = SttConfig {
            language: "en\0".to_owned(),
            ..SttConfig::default()
        };
        assert!(matches!(
            Transcriber::load("missing-model.bin", config),
            Err(SttError::InvalidConfig { field: "language" })
        ));

        let config = SttConfig {
            initial_prompt: "prompt\0".to_owned(),
            ..SttConfig::default()
        };
        assert!(matches!(
            Transcriber::load("missing-model.bin", config),
            Err(SttError::InvalidConfig {
                field: "initial_prompt"
            })
        ));
    }

    #[test]
    /// An empty language is rejected instead of being passed to Whisper.
    fn empty_language_is_rejected() {
        let config = SttConfig {
            language: "   ".to_owned(),
            ..SttConfig::default()
        };
        assert!(matches!(
            Transcriber::load("missing-model.bin", config),
            Err(SttError::InvalidConfig { field: "language" })
        ));
    }

    #[test]
    /// A missing model file becomes a typed load error rather than a panic.
    fn missing_model_returns_a_typed_error() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("does-not-exist.bin");
        assert!(matches!(
            Transcriber::load(path, SttConfig::default()),
            Err(SttError::ModelLoad(_))
        ));
    }

    #[test]
    #[ignore = "requires models/ggml-base.en.bin; run with `cargo test -p stt -- --ignored --nocapture`"]
    /// Loads the real base English model, transcribes synthetic silence and
    /// prints timings. The audio and any recognised text are never printed.
    fn transcribes_synthetic_silence_with_the_real_model() {
        use std::time::Instant;

        let model_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/ggml-base.en.bin");
        let load_start = Instant::now();
        let mut transcriber =
            Transcriber::load(&model_path, SttConfig::default()).expect("the model must load");
        let load_time = load_start.elapsed();

        let silence = vec![0.0_f32; SAMPLE_RATE as usize * 3];
        let transcribe_start = Instant::now();
        let result = transcriber.transcribe(&silence);
        let transcribe_time = transcribe_start.elapsed();

        let outcome = match &result {
            Ok(_) => "text",
            Err(SttError::NoSpeech) => "no speech",
            Err(_) => "error",
        };
        println!(
            "model load: {load_time:?}, transcription: {transcribe_time:?}, outcome: {outcome}"
        );
        assert!(
            matches!(result, Err(SttError::NoSpeech)),
            "synthetic silence must clean to None"
        );
    }
}
