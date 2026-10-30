//! Energy-based utterance endpointing.
//!
//! The endpointer consumes mono 16 kHz `f32` audio and emits complete
//! [`Utterance`] values. It never reads a clock: all timing is derived from
//! sample counts, so behaviour is deterministic and easy to test.

use std::collections::VecDeque;

use crate::{FRAME_SAMPLES, TARGET_SAMPLE_RATE};

/// Configuration for [`Endpointer`].
///
/// The defaults are tuned for a quiet room and normal speech at 16 kHz. The
/// energy thresholds are RMS amplitudes in the range `0.0..=1.0`; speech is
/// usually around `0.02` to `0.2`, while room noise is usually below `0.005`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EndpointConfig {
    /// Number of samples in one analysis frame. Defaults to [`FRAME_SAMPLES`],
    /// which is 20 ms at 16 kHz.
    pub frame_samples: usize,
    /// RMS level that starts an utterance from silence. Defaults to `0.015`.
    pub start_threshold_rms: f32,
    /// RMS level below which a frame counts as silence during speech.
    /// Defaults to `0.008`. It must be lower than
    /// [`start_threshold_rms`](Self::start_threshold_rms) so quiet speech does
    /// not immediately end the utterance.
    pub end_threshold_rms: f32,
    /// Minimum amount of voiced audio needed to keep an utterance, in
    /// milliseconds. Defaults to `250`.
    pub min_speech_ms: u32,
    /// Amount of continuous silence that ends an utterance, in milliseconds.
    /// Defaults to `700`.
    pub end_silence_ms: u32,
    /// Audio kept from before speech starts, in milliseconds. Defaults to
    /// `200`.
    pub pre_roll_ms: u32,
    /// Maximum length of one utterance, in milliseconds. Defaults to `10_000`.
    pub max_utterance_ms: u32,
}

impl Default for EndpointConfig {
    /// Returns the documented default endpointing settings.
    fn default() -> Self {
        Self {
            frame_samples: FRAME_SAMPLES,
            start_threshold_rms: 0.015,
            end_threshold_rms: 0.008,
            min_speech_ms: 250,
            end_silence_ms: 700,
            pre_roll_ms: 200,
            max_utterance_ms: 10_000,
        }
    }
}

/// Reasons why an [`EndpointConfig`] cannot be used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointConfigError {
    /// The frame size is zero.
    ZeroFrameSamples,
    /// A threshold is not finite or not greater than zero.
    InvalidThreshold,
    /// The start threshold is not above the end threshold.
    ThresholdOrder,
    /// A duration is zero.
    ZeroDuration {
        /// Name of the configuration field that was zero.
        field: &'static str,
    },
    /// Minimum speech is longer than the maximum utterance.
    MinSpeechExceedsMax,
    /// Pre-roll is longer than the maximum utterance.
    PreRollExceedsMax,
}

impl std::fmt::Display for EndpointConfigError {
    /// Writes a human-readable description of the configuration problem.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroFrameSamples => {
                formatter.write_str("frame_samples must be greater than zero")
            }
            Self::InvalidThreshold => {
                formatter.write_str("energy thresholds must be finite and greater than zero")
            }
            Self::ThresholdOrder => {
                formatter.write_str("start_threshold_rms must be greater than end_threshold_rms")
            }
            Self::ZeroDuration { field } => {
                write!(formatter, "{field} must be greater than zero")
            }
            Self::MinSpeechExceedsMax => {
                formatter.write_str("min_speech_ms must not exceed max_utterance_ms")
            }
            Self::PreRollExceedsMax => {
                formatter.write_str("pre_roll_ms must not exceed max_utterance_ms")
            }
        }
    }
}

impl std::error::Error for EndpointConfigError {}

/// A completed piece of speech.
#[derive(Debug, Clone, PartialEq)]
pub struct Utterance {
    /// Mono 16 kHz `f32` samples, including the configured pre-roll and the
    /// trailing silence that ended the utterance.
    pub samples: Vec<f32>,
}

/// Turns a stream of audio frames into complete utterances.
///
/// Feed audio with [`push_frame`](Self::push_frame). Frames may be any length:
/// samples are buffered internally until a full `frame_samples` frame is
/// available, so callers do not have to align their buffers to the analysis
/// frame size.
#[derive(Debug)]
pub struct Endpointer {
    /// Settings used by the state machine.
    config: EndpointConfig,
    /// Whether the endpointer is waiting for speech or already inside it.
    state: State,
    /// Recent audio kept so an utterance can start before its first loud frame.
    pre_roll: Vec<f32>,
    /// Samples of the utterance currently being built.
    utterance: Vec<f32>,
    /// Number of voiced samples in the current utterance.
    voiced_samples: usize,
    /// Number of consecutive silent samples at the end of the utterance.
    trailing_silence: usize,
    /// Set after a maximum-length cut-off until the input goes quiet again.
    suppress_until_silence: bool,
    /// Partial frame carried between calls.
    pending: Vec<f32>,
    /// Utterances completed during a call but not returned yet.
    ready: VecDeque<Utterance>,
    /// Pre-roll length in samples.
    pre_roll_samples: usize,
    /// Minimum voiced length in samples.
    min_speech_samples: usize,
    /// Silence needed to end an utterance, in samples.
    end_silence_samples: usize,
    /// Maximum utterance length in samples.
    max_utterance_samples: usize,
}

/// The endpointer's position relative to speech.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// No speech is active; frames are collected into the pre-roll.
    Idle,
    /// Speech is active and samples are being appended to an utterance.
    Speech,
}

impl Endpointer {
    /// Creates an endpointer from `config`.
    ///
    /// # Errors
    /// Returns [`EndpointConfigError`] if the configuration is not usable.
    pub fn new(config: EndpointConfig) -> Result<Self, EndpointConfigError> {
        validate_config(&config)?;
        Ok(Self {
            config,
            state: State::Idle,
            pre_roll: Vec::new(),
            utterance: Vec::new(),
            voiced_samples: 0,
            trailing_silence: 0,
            suppress_until_silence: false,
            pending: Vec::with_capacity(config.frame_samples),
            ready: VecDeque::new(),
            pre_roll_samples: milliseconds_to_samples(config.pre_roll_ms),
            min_speech_samples: milliseconds_to_samples(config.min_speech_ms),
            end_silence_samples: milliseconds_to_samples(config.end_silence_ms),
            max_utterance_samples: milliseconds_to_samples(config.max_utterance_ms),
        })
    }

    /// Feeds audio samples and returns the utterance that ended, if any.
    ///
    /// Samples are buffered into fixed-size frames. At most one utterance is
    /// returned per call; if a single call completes more than one utterance,
    /// the remaining ones are returned by later calls.
    ///
    /// # Returns
    /// `Some(Utterance)` when the end of an utterance is detected on this
    /// call, otherwise `None`.
    pub fn push_frame(&mut self, frame: &[f32]) -> Option<Utterance> {
        let frame_samples = self.config.frame_samples;
        for &sample in frame {
            self.pending.push(sample);
            if self.pending.len() == frame_samples {
                // Move the buffer out so `process_frame` can borrow `self`
                // mutably without conflicting with the frame slice.
                let full_frame = std::mem::take(&mut self.pending);
                if let Some(utterance) = self.process_frame(&full_frame) {
                    self.ready.push_back(utterance);
                }
                self.pending = full_frame;
                self.pending.clear();
            }
        }
        self.ready.pop_front()
    }

    /// Processes one complete analysis frame and returns an utterance if this
    /// frame ended one.
    ///
    /// # Arguments
    /// * `frame` - Exactly `config.frame_samples` mono samples.
    ///
    /// # Returns
    /// The completed utterance, or `None` while speech is still in progress.
    fn process_frame(&mut self, frame: &[f32]) -> Option<Utterance> {
        let rms = frame_rms(frame);
        match self.state {
            State::Idle => {
                if self.suppress_until_silence {
                    // Discard the tail of a cut-off utterance until the input
                    // is quiet, so the same speech does not start again.
                    if rms < self.config.end_threshold_rms {
                        self.suppress_until_silence = false;
                        self.push_pre_roll(frame);
                    }
                    return None;
                }
                if rms >= self.config.start_threshold_rms {
                    self.state = State::Speech;
                    self.utterance.clear();
                    self.utterance.extend_from_slice(&self.pre_roll);
                    self.utterance.extend_from_slice(frame);
                    // Quiet speech that did not cross the start threshold is
                    // still speech and counts toward the minimum.
                    self.voiced_samples = self.count_voiced_samples(&self.pre_roll) + frame.len();
                    self.pre_roll.clear();
                    self.trailing_silence = 0;
                } else {
                    self.push_pre_roll(frame);
                }
                None
            }
            State::Speech => {
                self.utterance.extend_from_slice(frame);
                if rms >= self.config.end_threshold_rms {
                    self.voiced_samples += frame.len();
                    self.trailing_silence = 0;
                } else {
                    self.trailing_silence += frame.len();
                }

                if self.utterance.len() >= self.max_utterance_samples {
                    self.utterance.truncate(self.max_utterance_samples);
                    if self.voiced_samples >= self.min_speech_samples {
                        let utterance = self.take_utterance();
                        self.suppress_until_silence = true;
                        return Some(utterance);
                    }
                    self.discard_false_start();
                    self.suppress_until_silence = true;
                    return None;
                }

                if self.trailing_silence >= self.end_silence_samples {
                    if self.voiced_samples >= self.min_speech_samples {
                        return Some(self.take_utterance());
                    }
                    self.discard_false_start();
                }
                None
            }
        }
    }

    /// Adds a frame to the pre-roll and trims the oldest samples.
    ///
    /// # Arguments
    /// * `frame` - Frame to remember in case speech starts next.
    fn push_pre_roll(&mut self, frame: &[f32]) {
        self.pre_roll.extend_from_slice(frame);
        if self.pre_roll.len() > self.pre_roll_samples {
            let excess = self.pre_roll.len() - self.pre_roll_samples;
            self.pre_roll.drain(..excess);
        }
    }

    /// Finishes the current utterance and returns it to the caller.
    ///
    /// # Returns
    /// The accumulated samples, including pre-roll and trailing silence.
    fn take_utterance(&mut self) -> Utterance {
        let samples = std::mem::take(&mut self.utterance);
        self.state = State::Idle;
        self.voiced_samples = 0;
        self.trailing_silence = 0;
        self.pre_roll.clear();
        Utterance { samples }
    }

    /// Drops a too-short burst and keeps its tail as pre-roll for the next
    /// attempt.
    fn discard_false_start(&mut self) {
        let keep = self.pre_roll_samples.min(self.utterance.len());
        let start = self.utterance.len() - keep;
        self.pre_roll.clear();
        self.pre_roll.extend_from_slice(&self.utterance[start..]);
        self.utterance.clear();
        self.state = State::Idle;
        self.voiced_samples = 0;
        self.trailing_silence = 0;
    }

    /// Counts samples in `audio` that belong to frames at or above the end
    /// threshold.
    ///
    /// # Arguments
    /// * `audio` - Complete analysis frames, normally the pre-roll.
    ///
    /// # Returns
    /// The number of voiced samples.
    fn count_voiced_samples(&self, audio: &[f32]) -> usize {
        audio
            .chunks(self.config.frame_samples)
            .filter(|frame| frame_rms(frame) >= self.config.end_threshold_rms)
            .map(|frame| frame.len())
            .sum()
    }
}

/// Checks that every setting has a value the state machine can use.
///
/// # Arguments
/// * `config` - Configuration to validate.
///
/// # Errors
/// Returns the specific [`EndpointConfigError`] for the first invalid value.
fn validate_config(config: &EndpointConfig) -> Result<(), EndpointConfigError> {
    if config.frame_samples == 0 {
        return Err(EndpointConfigError::ZeroFrameSamples);
    }
    if !config.start_threshold_rms.is_finite()
        || config.start_threshold_rms <= 0.0
        || !config.end_threshold_rms.is_finite()
        || config.end_threshold_rms <= 0.0
    {
        return Err(EndpointConfigError::InvalidThreshold);
    }
    if config.end_threshold_rms >= config.start_threshold_rms {
        return Err(EndpointConfigError::ThresholdOrder);
    }

    for (field, value) in [
        ("min_speech_ms", config.min_speech_ms),
        ("end_silence_ms", config.end_silence_ms),
        ("pre_roll_ms", config.pre_roll_ms),
        ("max_utterance_ms", config.max_utterance_ms),
    ] {
        if value == 0 {
            return Err(EndpointConfigError::ZeroDuration { field });
        }
    }

    if config.min_speech_ms > config.max_utterance_ms {
        return Err(EndpointConfigError::MinSpeechExceedsMax);
    }
    if config.pre_roll_ms > config.max_utterance_ms {
        return Err(EndpointConfigError::PreRollExceedsMax);
    }
    Ok(())
}

/// Computes the root-mean-square amplitude of a frame.
///
/// # Arguments
/// * `frame` - Mono samples to measure.
///
/// # Returns
/// The RMS level, or `0.0` for an empty frame or a non-finite result.
fn frame_rms(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    let sum: f32 = frame.iter().map(|sample| sample * sample).sum();
    let rms = (sum / frame.len() as f32).sqrt();
    if rms.is_finite() { rms } else { 0.0 }
}

/// Converts a duration in milliseconds to a number of 16 kHz samples.
///
/// # Arguments
/// * `milliseconds` - Duration to convert.
///
/// # Returns
/// The equivalent sample count, saturating instead of overflowing.
fn milliseconds_to_samples(milliseconds: u32) -> usize {
    let samples = (milliseconds as u64).saturating_mul(TARGET_SAMPLE_RATE as u64) / 1_000;
    samples.min(usize::MAX as u64) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds one analysis frame filled with `amplitude`.
    fn frame(amplitude: f32) -> Vec<f32> {
        vec![amplitude; FRAME_SAMPLES]
    }

    /// Feeds `milliseconds` of constant-amplitude audio and collects emitted
    /// utterances. Durations are rounded down to whole 20 ms frames.
    fn feed(endpointer: &mut Endpointer, milliseconds: u32, amplitude: f32) -> Vec<Utterance> {
        let mut utterances = Vec::new();
        for _ in 0..(milliseconds / 20) {
            if let Some(utterance) = endpointer.push_frame(&frame(amplitude)) {
                utterances.push(utterance);
            }
        }
        utterances
    }

    /// Returns a fresh endpointer with default settings.
    fn make_endpointer() -> Endpointer {
        Endpointer::new(EndpointConfig::default()).expect("default config must be valid")
    }

    #[test]
    /// Silence alone must never produce an utterance.
    fn silence_produces_no_utterance() {
        let mut endpointer = make_endpointer();
        assert!(feed(&mut endpointer, 2_000, 0.0).is_empty());
    }

    #[test]
    /// One second of speech followed by 800 ms of silence gives one utterance.
    fn speech_then_silence_gives_one_utterance() {
        let mut endpointer = make_endpointer();
        assert!(feed(&mut endpointer, 500, 0.0).is_empty());

        let mut utterances = feed(&mut endpointer, 1_000, 0.1);
        assert!(utterances.is_empty());
        utterances.extend(feed(&mut endpointer, 800, 0.0));

        assert_eq!(utterances.len(), 1);
        // 200 ms pre-roll + 1000 ms speech + 700 ms trailing silence.
        assert_eq!(utterances[0].samples.len(), 1_900 * 16);
    }

    #[test]
    /// A short click is below the minimum speech duration and must be dropped.
    fn short_click_is_ignored() {
        let mut endpointer = make_endpointer();
        let mut utterances = feed(&mut endpointer, 100, 0.1);
        utterances.extend(feed(&mut endpointer, 1_000, 0.0));
        assert!(utterances.is_empty());
    }

    #[test]
    /// Two speech bursts separated by enough silence give two utterances.
    fn two_utterances_are_returned_separately() {
        let mut endpointer = make_endpointer();
        let mut utterances = feed(&mut endpointer, 500, 0.1);
        utterances.extend(feed(&mut endpointer, 1_000, 0.0));
        utterances.extend(feed(&mut endpointer, 500, 0.1));
        utterances.extend(feed(&mut endpointer, 1_000, 0.0));
        assert_eq!(utterances.len(), 2);
    }

    #[test]
    /// The utterance starts with the configured pre-roll.
    fn pre_roll_is_included_before_speech() {
        let mut endpointer = make_endpointer();
        assert!(feed(&mut endpointer, 500, 0.0).is_empty());

        let mut utterances = feed(&mut endpointer, 500, 0.1);
        utterances.extend(feed(&mut endpointer, 800, 0.0));

        assert_eq!(utterances.len(), 1);
        let samples = &utterances[0].samples;
        // 200 ms pre-roll + 500 ms speech + 700 ms trailing silence.
        assert_eq!(samples.len(), 1_400 * 16);
        assert!(samples[..200 * 16].iter().all(|sample| *sample == 0.0));
        assert!(samples[200 * 16..].iter().any(|sample| *sample > 0.0));
    }

    #[test]
    /// Continuous speech is cut off at the maximum utterance length.
    fn maximum_length_cuts_off_long_speech() {
        let mut endpointer = make_endpointer();
        let utterances = feed(&mut endpointer, 12_000, 0.1);
        assert_eq!(utterances.len(), 1);
        assert_eq!(utterances[0].samples.len(), 10_000 * 16);
    }

    #[test]
    /// Audio between the two thresholds must not start an utterance.
    fn levels_between_thresholds_do_not_start_speech() {
        let config = EndpointConfig::default();
        let middle = (config.start_threshold_rms + config.end_threshold_rms) / 2.0;
        let mut endpointer = Endpointer::new(config).expect("default config must be valid");
        assert!(feed(&mut endpointer, 2_000, middle).is_empty());
    }

    #[test]
    /// Audio between the two thresholds must not end an utterance either.
    fn levels_between_thresholds_do_not_end_speech() {
        let config = EndpointConfig::default();
        let middle = (config.start_threshold_rms + config.end_threshold_rms) / 2.0;
        let mut endpointer = Endpointer::new(config).expect("default config must be valid");

        assert!(feed(&mut endpointer, 500, 0.0).is_empty());
        let mut utterances = feed(&mut endpointer, 500, 0.1);
        utterances.extend(feed(&mut endpointer, 1_000, middle));
        assert!(
            utterances.is_empty(),
            "mid-level audio must not end the utterance"
        );
        utterances.extend(feed(&mut endpointer, 800, 0.0));

        assert_eq!(utterances.len(), 1);
        // 200 ms pre-roll + 500 ms speech + 1000 ms mid-level + 700 ms silence.
        assert_eq!(utterances[0].samples.len(), 2_400 * 16);
    }

    #[test]
    /// Speech shorter than the minimum is dropped; speech above it is kept.
    fn minimum_speech_is_required() {
        let mut endpointer = make_endpointer();
        assert!(feed(&mut endpointer, 500, 0.0).is_empty());
        let mut utterances = feed(&mut endpointer, 240, 0.1);
        utterances.extend(feed(&mut endpointer, 1_000, 0.0));
        assert!(utterances.is_empty());

        let mut endpointer = make_endpointer();
        assert!(feed(&mut endpointer, 500, 0.0).is_empty());
        let mut utterances = feed(&mut endpointer, 260, 0.1);
        utterances.extend(feed(&mut endpointer, 1_000, 0.0));
        assert_eq!(utterances.len(), 1);
    }

    #[test]
    /// Quiet speech that stays below the start threshold still counts toward
    /// the minimum once a louder frame starts the utterance.
    fn quiet_speech_in_pre_roll_counts_toward_minimum() {
        let config = EndpointConfig::default();
        let middle = (config.start_threshold_rms + config.end_threshold_rms) / 2.0;
        let mut endpointer = Endpointer::new(config).expect("default config must be valid");

        // 500 ms of quiet speech: below start, above end, so it fills the
        // pre-roll but does not start an utterance by itself.
        assert!(feed(&mut endpointer, 500, middle).is_empty());
        // 100 ms of loud speech starts the utterance. The 200 ms of quiet
        // speech still in the pre-roll pushes the voiced total over 250 ms.
        let mut utterances = feed(&mut endpointer, 100, 0.1);
        utterances.extend(feed(&mut endpointer, 800, 0.0));
        assert_eq!(utterances.len(), 1);
    }

    #[test]
    /// A maximum-length cut-off before enough voiced audio is a false start.
    fn maximum_length_before_minimum_speech_is_discarded() {
        let config = EndpointConfig {
            min_speech_ms: 950,
            max_utterance_ms: 1_000,
            ..EndpointConfig::default()
        };
        let mut endpointer = Endpointer::new(config).expect("config must be valid");
        assert!(feed(&mut endpointer, 500, 0.0).is_empty());

        // 200 ms pre-roll + 800 ms speech reaches the 1 s maximum, but only
        // 800 ms is voiced, which is below the 950 ms minimum.
        let utterances = feed(&mut endpointer, 2_000, 0.1);
        assert!(utterances.is_empty());
    }

    #[test]
    /// Frames shorter than the analysis frame are buffered until they add up.
    fn short_frames_are_buffered() {
        let mut endpointer = make_endpointer();
        let mut utterances = Vec::new();
        let half_frame = vec![0.1; FRAME_SAMPLES / 2];
        for _ in 0..50 {
            if let Some(utterance) = endpointer.push_frame(&half_frame) {
                utterances.push(utterance);
            }
        }
        for _ in 0..80 {
            if let Some(utterance) = endpointer.push_frame(&vec![0.0; FRAME_SAMPLES / 2]) {
                utterances.push(utterance);
            }
        }
        assert_eq!(utterances.len(), 1);
    }

    #[test]
    /// Invalid settings are rejected instead of causing a panic later.
    fn invalid_configuration_is_rejected() {
        let config = EndpointConfig {
            frame_samples: 0,
            ..EndpointConfig::default()
        };
        assert!(matches!(
            Endpointer::new(config),
            Err(EndpointConfigError::ZeroFrameSamples)
        ));

        let config = EndpointConfig {
            start_threshold_rms: 0.01,
            end_threshold_rms: 0.02,
            ..EndpointConfig::default()
        };
        assert!(matches!(
            Endpointer::new(config),
            Err(EndpointConfigError::ThresholdOrder)
        ));

        let config = EndpointConfig {
            end_silence_ms: 0,
            ..EndpointConfig::default()
        };
        assert!(matches!(
            Endpointer::new(config),
            Err(EndpointConfigError::ZeroDuration { .. })
        ));

        let config = EndpointConfig {
            pre_roll_ms: 2_000,
            max_utterance_ms: 1_000,
            ..EndpointConfig::default()
        };
        assert!(matches!(
            Endpointer::new(config),
            Err(EndpointConfigError::PreRollExceedsMax)
        ));
    }

    #[test]
    /// Millisecond durations convert to the expected 16 kHz sample counts.
    fn milliseconds_convert_to_samples() {
        assert_eq!(milliseconds_to_samples(1), 16);
        assert_eq!(milliseconds_to_samples(250), 4_000);
        assert_eq!(milliseconds_to_samples(10_000), 160_000);
    }
}
