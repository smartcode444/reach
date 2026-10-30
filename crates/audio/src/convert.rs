//! Conversion helpers for microphone audio.
//!
//! Microphones deliver interleaved samples in several formats and sample
//! rates. These functions turn them into the mono 16 kHz `f32` samples used by
//! the rest of the speech pipeline. Everything happens in memory; no sample is
//! ever written to disk or logged.

use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};

use crate::TARGET_SAMPLE_RATE;

/// Downmixes interleaved audio to mono by averaging each frame's channels.
///
/// # Arguments
/// * `interleaved` - Samples in frame-major order (`L, R, L, R, ...`).
/// * `channels` - Number of channels per frame. A value of zero is treated as
///   one so the input is returned unchanged.
///
/// # Returns
/// One sample per frame. A trailing partial frame is ignored because it cannot
/// be averaged reliably.
pub fn downmix_to_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Converts signed 16-bit samples to `f32` in the range `-1.0..=1.0`.
///
/// # Arguments
/// * `samples` - Signed 16-bit PCM samples.
///
/// # Returns
/// Normalised samples. `i16::MIN` maps to `-1.0` and `i16::MAX` maps to just
/// below `1.0`, which matches the usual PCM convention.
pub fn i16_to_f32(samples: &[i16]) -> Vec<f32> {
    samples
        .iter()
        .map(|sample| *sample as f32 / 32_768.0)
        .collect()
}

/// Converts unsigned 16-bit samples to `f32` in the range `-1.0..=1.0`.
///
/// # Arguments
/// * `samples` - Unsigned 16-bit PCM samples with `32768` as the zero point.
///
/// # Returns
/// Normalised samples. `0` maps to `-1.0`, `32768` maps to `0.0` and `65535`
/// maps to just below `1.0`.
pub fn u16_to_f32(samples: &[u16]) -> Vec<f32> {
    samples
        .iter()
        .map(|sample| (*sample as f32 - 32_768.0) / 32_768.0)
        .collect()
}

/// Copies already-normalised `f32` samples.
///
/// This exists so the capture dispatch can treat all supported formats the
/// same way, and so the conversion helpers have a matching `f32` entry point.
///
/// # Arguments
/// * `samples` - Normalised `f32` samples.
///
/// # Returns
/// A copy of the input.
pub fn f32_to_f32(samples: &[f32]) -> Vec<f32> {
    samples.to_vec()
}

/// Resamples mono audio to 16 kHz.
///
/// The whole clip is resampled in one call with rubato's FFT resampler, which
/// trims the resampler's startup delay. Use this for a complete utterance;
/// streaming capture uses its own stateful resampler.
///
/// # Arguments
/// * `samples` - Mono samples at `input_rate`.
/// * `input_rate` - Sample rate of `samples` in Hz.
///
/// # Returns
/// Mono samples at [`TARGET_SAMPLE_RATE`]. The length is
/// `ceil(samples.len() * 16_000 / input_rate)`.
///
/// # Errors
/// Returns [`ResampleTo16kError`] if the rate is zero, the resampler cannot be
/// created, or rubato rejects the buffers.
pub fn resample_to_16k(samples: &[f32], input_rate: u32) -> Result<Vec<f32>, ResampleTo16kError> {
    if input_rate == 0 {
        return Err(ResampleTo16kError::InvalidSampleRate(input_rate));
    }
    if input_rate == TARGET_SAMPLE_RATE || samples.is_empty() {
        return Ok(samples.to_vec());
    }

    let mut resampler = Fft::<f32>::new(
        input_rate as usize,
        TARGET_SAMPLE_RATE as usize,
        1_024,
        1,
        FixedSync::Input,
    )
    .map_err(ResampleTo16kError::Construction)?;

    let input =
        InterleavedSlice::new(samples, 1, samples.len()).map_err(ResampleTo16kError::Buffer)?;
    let output = resampler
        .process_all(&input, samples.len(), None)
        .map_err(ResampleTo16kError::Process)?;

    Ok(output.take_data())
}

/// Reasons why [`resample_to_16k`] can fail.
#[derive(Debug)]
pub enum ResampleTo16kError {
    /// The input sample rate was zero.
    InvalidSampleRate(u32),
    /// Rubato could not build the resampler.
    Construction(rubato::ResamplerConstructionError),
    /// The sample slice could not be wrapped as an audio buffer.
    Buffer(rubato::audioadapter_buffers::SizeError),
    /// Rubato failed while processing the clip.
    Process(rubato::ResampleError),
}

impl std::fmt::Display for ResampleTo16kError {
    /// Writes a human-readable description of the resampling failure.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSampleRate(rate) => {
                write!(
                    formatter,
                    "cannot resample audio with a sample rate of {rate} Hz"
                )
            }
            Self::Construction(error) => {
                write!(formatter, "could not create the 16 kHz resampler: {error}")
            }
            Self::Buffer(error) => {
                write!(
                    formatter,
                    "could not wrap the audio buffer for resampling: {error}"
                )
            }
            Self::Process(error) => {
                write!(formatter, "resampling to 16 kHz failed: {error}")
            }
        }
    }
}

impl std::error::Error for ResampleTo16kError {
    /// Returns the underlying rubato or buffer error, when there is one.
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Construction(error) => Some(error),
            Self::Buffer(error) => Some(error),
            Self::Process(error) => Some(error),
            Self::InvalidSampleRate(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generates a mono sine wave with the given amplitude.
    fn sine(frequency: f32, sample_rate: u32, seconds: f32) -> Vec<f32> {
        let length = (sample_rate as f32 * seconds) as usize;
        (0..length)
            .map(|index| {
                let phase =
                    2.0 * std::f32::consts::PI * frequency * index as f32 / sample_rate as f32;
                phase.sin() * 0.5
            })
            .collect()
    }

    /// Returns the root-mean-square level of `samples`.
    fn rms(samples: &[f32]) -> f32 {
        let sum: f32 = samples.iter().map(|sample| sample * sample).sum();
        (sum / samples.len() as f32).sqrt()
    }

    #[test]
    /// Stereo frames are averaged into one sample each.
    fn downmix_stereo_to_mono() {
        let input = [1.0, 0.0, 0.5, 0.5, -1.0, 0.0];
        assert_eq!(downmix_to_mono(&input, 2), vec![0.5, 0.5, -0.5]);
    }

    #[test]
    /// A mono input is passed through unchanged.
    fn downmix_mono_passes_through() {
        let input = [0.1, -0.2, 0.3];
        assert_eq!(downmix_to_mono(&input, 1), input);
    }

    #[test]
    /// A zero channel count is treated as mono rather than dividing by zero.
    fn downmix_zero_channels_is_identity() {
        let input = [0.1, -0.2, 0.3];
        assert_eq!(downmix_to_mono(&input, 0), input);
    }

    #[test]
    /// Every channel contributes equally for surround layouts.
    fn downmix_surround_averages_all_channels() {
        let input = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(downmix_to_mono(&input, 6), vec![1.0 / 6.0]);
    }

    #[test]
    /// A trailing incomplete frame is ignored instead of producing a wrong
    /// average.
    fn downmix_ignores_incomplete_trailing_frame() {
        let input = [1.0, 0.0, 0.5];
        assert_eq!(downmix_to_mono(&input, 2), vec![0.5]);
    }

    #[test]
    /// Signed 16-bit full scale maps to the normalised float range.
    fn i16_full_scale_maps_to_f32_range() {
        let converted = i16_to_f32(&[0, i16::MAX, i16::MIN]);
        assert_eq!(converted[0], 0.0);
        assert!((converted[1] - 0.999_969_5).abs() < 0.000_1);
        assert_eq!(converted[2], -1.0);
    }

    #[test]
    /// Unsigned 16-bit silence maps to zero and the rails map to +/-1.
    fn u16_full_scale_maps_to_f32_range() {
        let converted = u16_to_f32(&[0, 32_768, u16::MAX]);
        assert_eq!(converted[0], -1.0);
        assert_eq!(converted[1], 0.0);
        assert!((converted[2] - 0.999_969_5).abs() < 0.000_1);
    }

    #[test]
    /// The f32 entry point copies its input.
    fn f32_conversion_is_identity() {
        let input = [0.0, 0.25, -0.5];
        assert_eq!(f32_to_f32(&input), input);
    }

    #[test]
    /// One second of 48 kHz audio becomes one second at 16 kHz.
    fn resampling_48k_sine_to_16k_preserves_length() {
        let input = sine(440.0, 48_000, 1.0);
        let output = resample_to_16k(&input, 48_000).expect("resampling must succeed");
        assert_eq!(output.len(), 16_000);
        assert!(
            (rms(&output) - rms(&input)).abs() < 0.05,
            "resampling should preserve the signal level"
        );
    }

    #[test]
    /// Audio that is already at 16 kHz is copied without filtering.
    fn resampling_16k_is_a_copy() {
        let input = sine(440.0, 16_000, 0.5);
        let output = resample_to_16k(&input, 16_000).expect("resampling must succeed");
        assert_eq!(output, input);
    }

    #[test]
    /// Empty input stays empty.
    fn resampling_empty_audio_is_empty() {
        let output = resample_to_16k(&[], 48_000).expect("resampling must succeed");
        assert!(output.is_empty());
    }

    #[test]
    /// A zero input rate is rejected instead of reaching rubato.
    fn resampling_zero_rate_is_rejected() {
        assert!(matches!(
            resample_to_16k(&[0.0], 0),
            Err(ResampleTo16kError::InvalidSampleRate(0))
        ));
    }
}
