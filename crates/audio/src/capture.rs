//! Microphone capture through cpal.
//!
//! [`MicCapture`] opens the default input device, converts every callback to
//! mono, resamples it to 16 kHz and sends fixed-size frames over a standard
//! channel. Audio exists only in memory: this module never writes samples to
//! disk and never logs their contents.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Data, SampleFormat, Stream};
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};

use crate::convert::{downmix_to_mono, i16_to_f32, u16_to_f32};
use crate::{FRAME_SAMPLES, TARGET_SAMPLE_RATE};

/// A running microphone stream that delivers mono 16 kHz frames.
///
/// Frames are sent on the [`Receiver`] returned by [`MicCapture::start`] in
/// the order they were captured. Each frame contains exactly [`FRAME_SAMPLES`]
/// samples (20 ms). Dropping the capture stops the stream and disconnects the
/// receiver.
pub struct MicCapture {
    /// The live cpal input stream. Dropping it stops capture.
    _stream: Stream,
    /// Shared flag that makes the callback discard frames while paused.
    paused: Arc<AtomicBool>,
    /// Set by the stream error callback if the device fails at runtime.
    stream_failed: Arc<AtomicBool>,
    /// Sample rate negotiated with the device.
    sample_rate: u32,
    /// Channel count negotiated with the device.
    channels: u16,
}

impl MicCapture {
    /// Opens the default input device and starts delivering 16 kHz frames.
    ///
    /// # Returns
    /// The running capture and the receiver for its frames. The capture must
    /// be kept alive for as long as frames are wanted.
    ///
    /// # Errors
    /// Returns [`MicCaptureError`] when there is no default input device, the
    /// device reports an unusable configuration, its sample format is not
    /// supported, or the stream cannot be built or started.
    pub fn start() -> Result<(Self, Receiver<Vec<f32>>), MicCaptureError> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or(MicCaptureError::NoInputDevice)?;
        let supported = device
            .default_input_config()
            .map_err(MicCaptureError::DefaultConfig)?;

        let sample_format = supported.sample_format();
        let stream_config = supported.config();
        let sample_rate = stream_config.sample_rate;
        let channels = stream_config.channels;

        match sample_format {
            SampleFormat::F32 | SampleFormat::I16 | SampleFormat::U16 => {}
            other => return Err(MicCaptureError::UnsupportedSampleFormat(other)),
        }

        let mut converter = MonoConverter::new(sample_rate)?;
        let (sender, receiver) = mpsc::channel();
        let paused = Arc::new(AtomicBool::new(false));
        let stream_failed = Arc::new(AtomicBool::new(false));

        let callback_paused = Arc::clone(&paused);
        let callback_failed = Arc::clone(&stream_failed);
        let data_callback = move |data: &Data, _info: &cpal::InputCallbackInfo| {
            if callback_paused.load(Ordering::Relaxed) {
                // Discard anything buffered before the pause so it cannot be
                // delivered after resuming.
                converter.clear();
                return;
            }

            let mono = match data.sample_format() {
                SampleFormat::F32 => data
                    .as_slice::<f32>()
                    .map(|samples| downmix_to_mono(samples, channels as usize)),
                SampleFormat::I16 => data
                    .as_slice::<i16>()
                    .map(|samples| downmix_to_mono(&i16_to_f32(samples), channels as usize)),
                SampleFormat::U16 => data
                    .as_slice::<u16>()
                    .map(|samples| downmix_to_mono(&u16_to_f32(samples), channels as usize)),
                // The format was checked before the stream was built, so this
                // arm is unreachable unless the backend changes format.
                _ => None,
            };

            if let Some(mono) = mono {
                converter.push(&mono, |frame| {
                    // A closed receiver means the application no longer wants
                    // audio; there is nothing useful to do in a realtime
                    // callback except drop the frame.
                    let _ = sender.send(frame);
                });
            }
        };
        let error_callback = move |_error: cpal::Error| {
            callback_failed.store(true, Ordering::Relaxed);
        };

        let stream = device
            .build_input_stream_raw(
                stream_config,
                sample_format,
                data_callback,
                error_callback,
                None,
            )
            .map_err(MicCaptureError::BuildStream)?;
        stream.play().map_err(MicCaptureError::Play)?;

        Ok((
            Self {
                _stream: stream,
                paused,
                stream_failed,
                sample_rate,
                channels,
            },
            receiver,
        ))
    }

    /// Stops delivering frames without closing the device.
    ///
    /// Frames captured while paused are discarded in the audio callback. An
    /// atomic flag is used instead of [`StreamTrait::pause`] because some
    /// backends do not support pausing and the flag cannot block the realtime
    /// callback. Audio buffered just before the pause is discarded too, so it
    /// cannot be delivered after [`resume`](Self::resume).
    pub fn pause(&self) {
        self.paused.store(true, Ordering::Relaxed);
    }

    /// Resumes delivery after [`pause`](Self::pause).
    pub fn resume(&self) {
        self.paused.store(false, Ordering::Relaxed);
    }

    /// Returns whether frames are currently being discarded.
    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Relaxed)
    }

    /// Returns whether the audio backend reported an error after startup.
    ///
    /// The caller can rebuild the capture when this becomes true.
    pub fn stream_failed(&self) -> bool {
        self.stream_failed.load(Ordering::Relaxed)
    }

    /// Returns the device sample rate negotiated at startup.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Returns the device channel count negotiated at startup.
    pub fn channels(&self) -> u16 {
        self.channels
    }
}

/// Converts a device-rate mono stream into fixed 16 kHz frames.
///
/// The resampler keeps its state between callbacks, so chunk boundaries do not
/// introduce discontinuities. Its startup delay appears as a few milliseconds
/// of leading silence, which is harmless for voice activity detection.
struct MonoConverter {
    /// Resampler, or `None` when the device already runs at 16 kHz.
    resampler: Option<Fft<f32>>,
    /// Device-rate samples that do not fill a complete resampler chunk yet.
    input_pending: Vec<f32>,
    /// Scratch buffer for one resampler output chunk.
    output_scratch: Vec<f32>,
    /// 16 kHz samples that do not fill an output frame yet.
    output_pending: Vec<f32>,
    /// Number of samples in each emitted frame.
    frame_samples: usize,
}

impl MonoConverter {
    /// Creates a converter for `input_rate`.
    ///
    /// # Errors
    /// Returns [`MicCaptureError::InvalidSampleRate`] for a zero rate and
    /// [`MicCaptureError::Resampler`] if rubato rejects the conversion.
    fn new(input_rate: u32) -> Result<Self, MicCaptureError> {
        if input_rate == 0 {
            return Err(MicCaptureError::InvalidSampleRate(input_rate));
        }

        let resampler = if input_rate == TARGET_SAMPLE_RATE {
            None
        } else {
            Some(
                Fft::<f32>::new(
                    input_rate as usize,
                    TARGET_SAMPLE_RATE as usize,
                    1_024,
                    1,
                    FixedSync::Input,
                )
                .map_err(MicCaptureError::Resampler)?,
            )
        };
        let output_scratch = match &resampler {
            Some(resampler) => vec![0.0; resampler.output_frames_max()],
            None => Vec::new(),
        };

        Ok(Self {
            resampler,
            input_pending: Vec::new(),
            output_scratch,
            output_pending: Vec::new(),
            frame_samples: FRAME_SAMPLES,
        })
    }

    /// Adds mono device-rate samples and emits every complete 16 kHz frame.
    ///
    /// # Arguments
    /// * `mono` - Mono samples at the device rate.
    /// * `emit` - Called once for each complete frame, in capture order.
    fn push(&mut self, mono: &[f32], mut emit: impl FnMut(Vec<f32>)) {
        if self.resampler.is_some() {
            self.input_pending.extend_from_slice(mono);
            self.resample_pending();
        } else {
            self.output_pending.extend_from_slice(mono);
        }
        self.emit_frames(&mut emit);
    }

    /// Drops buffered audio and resets the resampler.
    ///
    /// Called while capture is paused so samples from before the pause cannot
    /// be delivered after it resumes.
    fn clear(&mut self) {
        self.input_pending.clear();
        self.output_pending.clear();
        if let Some(resampler) = self.resampler.as_mut() {
            resampler.reset();
        }
    }

    /// Resamples every complete input chunk currently buffered.
    fn resample_pending(&mut self) {
        loop {
            let Some(resampler) = self.resampler.as_mut() else {
                return;
            };
            let needed = resampler.input_frames_next();
            if self.input_pending.len() < needed {
                return;
            }

            let output_frames = self.output_scratch.len();
            let mut produced = None;
            let mut failed = false;
            {
                let input = InterleavedSlice::new(&self.input_pending[..needed], 1, needed);
                let output = InterleavedSlice::new_mut(&mut self.output_scratch, 1, output_frames);
                match (input, output) {
                    (Ok(input), Ok(mut output)) => {
                        match resampler.process_into_buffer(&input, &mut output, None) {
                            Ok((_consumed, frames)) => produced = Some(frames),
                            Err(_) => failed = true,
                        }
                    }
                    _ => failed = true,
                }
            }

            if failed {
                // Drop the chunk so a persistent resampler error cannot wedge
                // the stream; the next chunk gets a fresh attempt.
                self.input_pending.drain(..needed);
                return;
            }
            let Some(frames) = produced else {
                return;
            };
            self.output_pending
                .extend_from_slice(&self.output_scratch[..frames]);
            self.input_pending.drain(..needed);
        }
    }

    /// Emits all complete frames from the 16 kHz output buffer.
    ///
    /// # Arguments
    /// * `emit` - Called once for each complete frame.
    fn emit_frames(&mut self, emit: &mut impl FnMut(Vec<f32>)) {
        while self.output_pending.len() >= self.frame_samples {
            let frame: Vec<f32> = self.output_pending.drain(..self.frame_samples).collect();
            emit(frame);
        }
    }
}

/// Reasons why microphone capture cannot start or keep running.
#[derive(Debug)]
pub enum MicCaptureError {
    /// No default input device is available.
    NoInputDevice,
    /// The device did not report a usable default input configuration.
    DefaultConfig(cpal::Error),
    /// The default configuration uses a sample format this crate cannot
    /// convert.
    UnsupportedSampleFormat(SampleFormat),
    /// The input stream could not be built.
    BuildStream(cpal::Error),
    /// The input stream could not be started.
    Play(cpal::Error),
    /// The device reported a zero sample rate.
    InvalidSampleRate(u32),
    /// The streaming resampler could not be created.
    Resampler(rubato::ResamplerConstructionError),
}

impl std::fmt::Display for MicCaptureError {
    /// Writes a human-readable description of the capture failure.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoInputDevice => formatter.write_str("no default microphone is available"),
            Self::DefaultConfig(error) => {
                write!(
                    formatter,
                    "could not read the default microphone configuration: {error}"
                )
            }
            Self::UnsupportedSampleFormat(format) => {
                write!(
                    formatter,
                    "the microphone uses an unsupported sample format: {format:?}"
                )
            }
            Self::BuildStream(error) => {
                write!(formatter, "could not open the microphone stream: {error}")
            }
            Self::Play(error) => {
                write!(formatter, "could not start the microphone stream: {error}")
            }
            Self::InvalidSampleRate(rate) => {
                write!(
                    formatter,
                    "the microphone reported an invalid sample rate of {rate} Hz"
                )
            }
            Self::Resampler(error) => {
                write!(
                    formatter,
                    "could not create the microphone resampler: {error}"
                )
            }
        }
    }
}

impl std::error::Error for MicCaptureError {
    /// Returns the underlying cpal or rubato error, when there is one.
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::DefaultConfig(error) | Self::BuildStream(error) | Self::Play(error) => {
                Some(error)
            }
            Self::Resampler(error) => Some(error),
            Self::NoInputDevice | Self::UnsupportedSampleFormat(_) | Self::InvalidSampleRate(_) => {
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// A 16 kHz device bypasses resampling and emits exact frames.
    fn converter_at_16k_emits_exact_frames() {
        let mut converter = MonoConverter::new(16_000).expect("16 kHz is valid");
        let mut frames = Vec::new();
        converter.push(&vec![0.1; FRAME_SAMPLES * 5], |frame| {
            frames.push(frame);
        });
        assert_eq!(frames.len(), 5);
        assert!(frames.iter().all(|frame| frame.len() == FRAME_SAMPLES));
    }

    #[test]
    /// A 48 kHz device is resampled and still emits whole 16 kHz frames.
    fn converter_at_48k_emits_16k_frames() {
        let mut converter = MonoConverter::new(48_000).expect("48 kHz is valid");
        let mut frames = Vec::new();
        converter.push(&vec![0.1; 4_800], |frame| frames.push(frame));
        assert!(!frames.is_empty());
        assert!(frames.iter().all(|frame| frame.len() == FRAME_SAMPLES));
        let total: usize = frames.iter().map(Vec::len).sum();
        // 100 ms at 16 kHz is 1600 samples; the resampler holds back a small
        // tail that has not filled its next input chunk yet.
        assert!((1_280..=1_600).contains(&total));
    }

    #[test]
    /// A zero device rate is rejected with a typed error.
    fn converter_rejects_zero_rate() {
        assert!(matches!(
            MonoConverter::new(0),
            Err(MicCaptureError::InvalidSampleRate(0))
        ));
    }

    #[test]
    /// Clearing while paused drops buffered pre-pause audio.
    fn clearing_discards_buffered_audio() {
        let mut converter = MonoConverter::new(48_000).expect("48 kHz is valid");
        converter.push(&vec![1.0; 4_800], |_frame| {});
        converter.clear();

        let mut frames = Vec::new();
        converter.push(&vec![0.0; 9_600], |frame| frames.push(frame));
        assert!(!frames.is_empty());
        assert!(frames.iter().flatten().all(|sample| *sample == 0.0));
    }

    #[test]
    #[ignore = "requires a real microphone; run with `cargo test -p audio -- --ignored`"]
    /// Records three seconds from the default microphone and prints its level
    /// and whether the endpointer detected an utterance. No audio is saved.
    fn captures_three_seconds_from_the_default_microphone() {
        use std::time::{Duration, Instant};

        let (capture, receiver) =
            MicCapture::start().expect("a default microphone must be available");
        capture.pause();
        assert!(capture.is_paused());
        capture.resume();
        assert!(!capture.is_paused());
        let mut endpointer =
            crate::Endpointer::new(crate::EndpointConfig::default()).expect("valid config");
        let mut sample_count = 0_usize;
        let mut peak = 0.0_f32;
        let mut sum_of_squares = 0.0_f64;
        let mut utterance_count = 0_usize;
        let deadline = Instant::now() + Duration::from_secs(3);

        while Instant::now() < deadline {
            match receiver.recv_timeout(Duration::from_millis(250)) {
                Ok(frame) => {
                    sample_count += frame.len();
                    for &sample in &frame {
                        peak = peak.max(sample.abs());
                        sum_of_squares += f64::from(sample) * f64::from(sample);
                    }
                    if endpointer.push_frame(&frame).is_some() {
                        utterance_count += 1;
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    panic!("microphone stream stopped unexpectedly");
                }
            }
        }

        let rms = if sample_count == 0 {
            0.0
        } else {
            (sum_of_squares / sample_count as f64).sqrt()
        };
        println!(
            "captured {sample_count} samples, RMS {rms:.4}, peak amplitude {peak:.4}, \
             utterances {utterance_count}"
        );
        assert!(sample_count > 0, "no audio frames arrived");
    }
}
