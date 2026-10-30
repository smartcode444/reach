//! Audio capture and speech-to-text pipeline for voice control.
//!
//! Owns the audio frame ingestion, endpointing, and speech recognition pipeline.
//! Supplies utterances to [`Pipeline`](crate::pipeline::Pipeline) through
//! [`VoiceInput`], which implements [`BufRead`](std::io::BufRead) so spoken
//! confirmation answers can be consumed seamlessly by the pipeline.

use std::collections::VecDeque;
use std::io::{self, BufRead};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use audio::{EndpointConfig, Endpointer, MicCapture};
use reach_core::Heard;
use stt::{SAMPLE_RATE, SttConfig, Transcriber};

/// The default Whisper model container path.
pub const DEFAULT_STT_MODEL_PATH: &str = "models/ggml-base.en.bin";

/// Maximum duration the pipeline waits for a spoken confirmation answer: 6 seconds.
pub const CONFIRMATION_TIMEOUT: Duration = Duration::from_secs(6);

/// Maximum duration of one command utterance before forcing an endpoint cut: 6 seconds.
pub const COMMAND_MAX_UTTERANCE_MS: u32 = 6_000;

/// How long one frame poll waits before checking deadlines or shutdown flags.
pub const FRAME_WAIT: Duration = Duration::from_millis(250);

/// The outcome of polling an audio frame source.
#[derive(Debug, PartialEq)]
pub enum FramePoll {
    /// A single audio frame was delivered.
    Frame(Vec<f32>),
    /// The poll timed out before a frame arrived.
    Timeout,
    /// The audio source was closed or failed.
    Closed,
}

/// Supplies audio frames to the voice pipeline.
pub trait FrameSource: Send {
    /// Waits up to `timeout` for the next frame.
    ///
    /// # Arguments
    /// * `timeout` - Maximum time to wait.
    ///
    /// # Returns
    /// [`FramePoll`] indicating the frame, a timeout, or a closed source.
    fn next_frame(&mut self, timeout: Duration) -> FramePoll;
}

/// Pauses and resumes microphone frame delivery.
pub trait CaptureControl: Send + Sync {
    /// Stops delivering frames.
    fn pause(&self);

    /// Resumes delivering frames after a pause.
    fn resume(&self);
}

/// Resumes capture when it goes out of scope.
pub struct PauseGuard<'a> {
    /// Controller handle resumed on drop.
    control: &'a dyn CaptureControl,
}

impl<'a> PauseGuard<'a> {
    /// Pauses `control` and returns a guard that will resume it on drop.
    ///
    /// # Arguments
    /// * `control` - The capture controller.
    ///
    /// # Returns
    /// An instance of [`PauseGuard`].
    pub fn new(control: &'a dyn CaptureControl) -> Self {
        control.pause();
        Self { control }
    }
}

impl Drop for PauseGuard<'_> {
    fn drop(&mut self) {
        self.control.resume();
    }
}

/// Commands sent to the microphone capture worker thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CaptureCommand {
    /// Pause frame delivery.
    Pause,
    /// Resume frame delivery.
    Resume,
    /// Terminate capture thread.
    Stop,
}

/// Cloneable handle for controlling microphone frame delivery.
#[derive(Clone)]
pub struct CaptureHandle {
    /// Channel for delivering commands to the capture worker.
    commands: Sender<CaptureCommand>,
}

impl CaptureControl for CaptureHandle {
    fn pause(&self) {
        let _ = self.commands.send(CaptureCommand::Pause);
    }

    fn resume(&self) {
        let _ = self.commands.send(CaptureCommand::Resume);
    }
}

/// Audio capture source running on a dedicated background thread.
pub struct CaptureSource {
    /// Receives audio frames from the capture worker thread.
    frames: Receiver<Vec<f32>>,
    /// Sends commands to pause, resume, or stop the capture thread.
    commands: Sender<CaptureCommand>,
    /// Background thread handle joined on drop.
    handle: Option<JoinHandle<()>>,
}

impl CaptureSource {
    /// Starts capturing audio from the default system microphone.
    ///
    /// # Returns
    /// An initialized [`CaptureSource`].
    ///
    /// # Errors
    /// Returns error message if microphone cannot be opened.
    pub fn start() -> Result<Self, String> {
        let (command_sender, command_receiver) = mpsc::channel();
        let (frame_sender, frame_receiver) = mpsc::channel();
        let (ready_sender, ready_receiver) = mpsc::channel();

        let handle = thread::Builder::new()
            .name("reach-capture".to_owned())
            .spawn(move || {
                let (capture, frames) = match MicCapture::start() {
                    Ok(pair) => pair,
                    Err(error) => {
                        let _ = ready_sender.send(Err(error.to_string()));
                        return;
                    }
                };
                if ready_sender.send(Ok(())).is_err() {
                    return;
                }

                'capture: loop {
                    loop {
                        match command_receiver.try_recv() {
                            Ok(CaptureCommand::Pause) => capture.pause(),
                            Ok(CaptureCommand::Resume) => capture.resume(),
                            Ok(CaptureCommand::Stop) | Err(TryRecvError::Disconnected) => {
                                break 'capture;
                            }
                            Err(TryRecvError::Empty) => break,
                        }
                    }
                    match frames.recv_timeout(Duration::from_millis(20)) {
                        Ok(frame) => {
                            if frame_sender.send(frame).is_err() {
                                break;
                            }
                        }
                        Err(RecvTimeoutError::Timeout) => {}
                        Err(RecvTimeoutError::Disconnected) => break,
                    }
                }
            })
            .map_err(|error| format!("could not start the microphone worker: {error}"))?;

        match ready_receiver.recv_timeout(Duration::from_secs(10)) {
            Ok(Ok(())) => Ok(Self {
                frames: frame_receiver,
                commands: command_sender,
                handle: Some(handle),
            }),
            Ok(Err(error)) => {
                let _ = handle.join();
                Err(format!("could not start the microphone: {error}"))
            }
            Err(_) => {
                let _ = command_sender.send(CaptureCommand::Stop);
                Err("the microphone did not start in time".to_owned())
            }
        }
    }

    /// Returns a controller handle to pause/resume this capture source.
    ///
    /// # Returns
    /// An instance of [`CaptureHandle`].
    pub fn handle(&self) -> CaptureHandle {
        CaptureHandle {
            commands: self.commands.clone(),
        }
    }
}

impl FrameSource for CaptureSource {
    fn next_frame(&mut self, timeout: Duration) -> FramePoll {
        match self.frames.recv_timeout(timeout) {
            Ok(frame) => FramePoll::Frame(frame),
            Err(RecvTimeoutError::Timeout) => FramePoll::Timeout,
            Err(RecvTimeoutError::Disconnected) => FramePoll::Closed,
        }
    }
}

impl Drop for CaptureSource {
    fn drop(&mut self) {
        let _ = self.commands.send(CaptureCommand::Stop);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Transcribes speech samples into recognised text.
pub trait SpeechToText: Send {
    /// Transcribes audio samples into recognised [`Heard`] text.
    ///
    /// # Arguments
    /// * `samples` - Monophonic 16 kHz floating point audio samples.
    ///
    /// # Returns
    /// The recognised text, or `None` if audio contains no speech.
    ///
    /// # Errors
    /// Returns error message if transcription fails.
    fn transcribe(&mut self, samples: &[f32]) -> Result<Option<Heard>, String>;
}

impl SpeechToText for Transcriber {
    fn transcribe(&mut self, samples: &[f32]) -> Result<Option<Heard>, String> {
        match Transcriber::transcribe(self, samples) {
            Ok(heard) => {
                let Some(text) = stt::clean_transcript(&heard.text) else {
                    return Ok(None);
                };
                Ok(Some(Heard {
                    text,
                    confidence: heard.confidence,
                }))
            }
            Err(stt::SttError::NoSpeech) => Ok(None),
            Err(error) => Err(format!("speech recognition failed: {error}")),
        }
    }
}

/// An executor wrapper that pauses microphone delivery during intent execution.
pub struct PausingExecutor<E> {
    /// Underlying wrapped executor.
    inner: E,
    /// Controller used to pause frame delivery while executing.
    control: Box<dyn CaptureControl>,
}

impl<E> PausingExecutor<E> {
    /// Wraps `inner` to pause microphone input while an action executes.
    ///
    /// # Arguments
    /// * `inner` - Underlying executor.
    /// * `control` - Capture controller handle.
    ///
    /// # Returns
    /// An instance of [`PausingExecutor`].
    pub fn new(inner: E, control: Box<dyn CaptureControl>) -> Self {
        Self { inner, control }
    }

    /// Returns a reference to the inner executor.
    ///
    /// # Returns
    /// Reference to `E`.
    pub fn inner(&self) -> &E {
        &self.inner
    }

    /// Returns a mutable reference to the inner executor.
    ///
    /// # Returns
    /// Mutable reference to `E`.
    pub fn inner_mut(&mut self) -> &mut E {
        &mut self.inner
    }
}

impl<E: reach_core::Executor> reach_core::Executor for PausingExecutor<E> {
    fn run(&mut self, intent: &reach_core::Intent) -> Result<String, String> {
        let _guard = PauseGuard::new(self.control.as_ref());
        let res = self.inner.run(intent);
        // Wait 200 ms after action completes before resuming mic to avoid capturing residual noise
        thread::sleep(Duration::from_millis(200));
        res
    }
}

/// A recognised speech utterance along with timing metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct VoiceUtterance {
    /// The transcribed text and model confidence.
    pub heard: Heard,
    /// Estimated time the speaker stopped talking.
    pub speech_ended: Instant,
    /// Duration between speech end and start of transcription.
    pub queue_wait: Duration,
    /// Duration taken by speech recognition.
    pub transcription_time: Duration,
}

/// Ingests audio frames, endpoints speech, and presents utterances.
pub struct VoiceInput {
    /// Audio frame source delivering raw chunks.
    frames: Box<dyn FrameSource>,
    /// Speech recognition engine.
    stt: Box<dyn SpeechToText>,
    /// Voice activity and endpoint detector.
    endpointer: Endpointer,
    /// Trailing silence duration for utterance end estimation.
    trailing_silence: Duration,
    /// Maximum duration allowed for confirmation replies.
    confirmation_timeout: Duration,
    /// Buffered bytes for the [`std::io::BufRead`] implementation.
    pending: Vec<u8>,
    /// Number of bytes consumed from `pending`.
    consumed: usize,
    /// Set to true once the frame source closes.
    closed: bool,
    /// Minimum RMS required to start speech.
    start_threshold_rms: f32,
    /// Whether an utterance is currently being processed by the pipeline.
    is_processing: bool,
    /// Timestamp when processing of the last utterance finished.
    last_processing_ended: Option<Instant>,
    /// Buffer of utterances ready to return.
    queue: VecDeque<VoiceUtterance>,
}

impl VoiceInput {
    /// Creates a new [`VoiceInput`] from a frame source, recognizer, and config.
    ///
    /// # Arguments
    /// * `frames` - Audio frame source.
    /// * `stt` - Speech to text recognizer.
    /// * `endpoint_config` - Endpointing parameters.
    /// * `confirmation_timeout` - Timeout for confirmation answers.
    ///
    /// # Returns
    /// An initialized [`VoiceInput`].
    ///
    /// # Errors
    /// Returns error string if endpoint configuration is invalid.
    pub fn new(
        frames: Box<dyn FrameSource>,
        stt: Box<dyn SpeechToText>,
        endpoint_config: EndpointConfig,
        confirmation_timeout: Duration,
    ) -> Result<Self, String> {
        let endpointer = Endpointer::new(endpoint_config)
            .map_err(|e| format!("invalid endpoint configuration: {e}"))?;
        Ok(Self {
            frames,
            stt,
            endpointer,
            trailing_silence: Duration::from_millis(u64::from(endpoint_config.end_silence_ms)),
            start_threshold_rms: endpoint_config.start_threshold_rms,
            confirmation_timeout,
            pending: Vec::new(),
            consumed: 0,
            closed: false,
            is_processing: false,
            last_processing_ended: None,
            queue: VecDeque::new(),
        })
    }

    /// Creates the production microphone-driven voice input using system capture and Whisper.
    ///
    /// # Arguments
    /// * `stt_model_path` - Path to the Whisper model file.
    /// * `whisper_threads` - Number of decoding threads (0 = auto).
    /// * `endpoint_config` - Voice activity and endpointing configuration.
    ///
    /// # Returns
    /// A pair containing the initialized [`VoiceInput`] and the cloneable [`CaptureHandle`].
    ///
    /// # Errors
    /// Returns error message if microphone cannot be started or Whisper model cannot be loaded.
    pub fn create_default(
        stt_model_path: &str,
        whisper_threads: usize,
        endpoint_config: EndpointConfig,
    ) -> Result<(Self, CaptureHandle), String> {
        let capture = CaptureSource::start()?;
        let control = capture.handle();
        let mut stt_config = SttConfig::default();
        if whisper_threads > 0 {
            stt_config.threads = whisper_threads;
        }
        let transcriber = Transcriber::load(stt_model_path, stt_config)
            .map_err(|e| format!("failed to load STT model '{stt_model_path}': {e}"))?;

        let voice = Self::new(
            Box::new(capture),
            Box::new(transcriber),
            endpoint_config,
            CONFIRMATION_TIMEOUT,
        )?;

        Ok((voice, control))
    }

    /// Enqueues an utterance into the internal buffer for immediate delivery.
    ///
    /// # Arguments
    /// * `utterance` - Utterance to return on the next call.
    pub fn enqueue_utterance(&mut self, utterance: VoiceUtterance) {
        self.queue.push_back(utterance);
    }

    /// Returns true if the underlying audio frame source is closed.
    ///
    /// # Returns
    /// `true` if closed, `false` otherwise.
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// Marks that the pipeline has started processing an utterance.
    pub fn mark_processing_started(&mut self) {
        self.is_processing = true;
    }

    /// Marks that the pipeline has finished processing an utterance.
    pub fn mark_processing_ended(&mut self) {
        self.is_processing = false;
        self.last_processing_ended = Some(Instant::now());
    }

    /// Sets whether the pipeline is currently processing.
    ///
    /// # Arguments
    /// * `processing` - True if processing, false if idle.
    pub fn set_processing(&mut self, processing: bool) {
        self.is_processing = processing;
        if !processing {
            self.last_processing_ended = Some(Instant::now());
        }
    }

    /// Waits for, endpoints, and transcribes the next utterance.
    ///
    /// # Arguments
    /// * `timeout` - Optional maximum duration to wait.
    ///
    /// # Returns
    /// Next [`VoiceUtterance`], or `None` on timeout or closed source.
    ///
    /// # Errors
    /// Returns error message if transcription fails.
    pub fn next_utterance(
        &mut self,
        timeout: Option<Duration>,
    ) -> Result<Option<VoiceUtterance>, String> {
        self.next_utterance_with_filter(timeout, true)
    }

    /// Internal utterance poller with busy-dropping filter control.
    ///
    /// # Arguments
    /// * `timeout` - Optional deadline for waiting.
    /// * `check_busy_drop` - True to apply busy-dropping logic.
    ///
    /// # Returns
    /// Next [`VoiceUtterance`], or `None`.
    ///
    /// # Errors
    /// Returns error string on speech recognition errors.
    pub fn next_utterance_with_filter(
        &mut self,
        timeout: Option<Duration>,
        check_busy_drop: bool,
    ) -> Result<Option<VoiceUtterance>, String> {
        if let Some(queued) = self.queue.pop_front() {
            return Ok(Some(queued));
        }

        if self.closed {
            return Ok(None);
        }

        let deadline = timeout.map(|d| Instant::now() + d);

        loop {
            let wait = match deadline {
                Some(dl) => {
                    let remaining = dl.saturating_duration_since(Instant::now());
                    if remaining.is_zero() {
                        return Ok(None);
                    }
                    remaining.min(FRAME_WAIT)
                }
                None => FRAME_WAIT,
            };

            match self.frames.next_frame(wait) {
                FramePoll::Frame(frame) => {
                    let Some(utterance) = self.endpointer.push_frame(&frame) else {
                        continue;
                    };
                    if let Some(item) = self.process_raw_utterance(utterance, check_busy_drop)? {
                        return Ok(Some(item));
                    }
                }
                FramePoll::Timeout => {}
                FramePoll::Closed => {
                    self.closed = true;
                    return Ok(None);
                }
            }
        }
    }

    /// Processes audio samples from the endpointer into a validated utterance.
    ///
    /// # Arguments
    /// * `utterance` - Raw audio frame utterance.
    /// * `check_busy_drop` - True if busy-dropping checks should be applied.
    ///
    /// # Returns
    /// An optional [`VoiceUtterance`].
    ///
    /// # Errors
    /// Returns error message if transcription fails.
    fn process_raw_utterance(
        &mut self,
        utterance: audio::Utterance,
        check_busy_drop: bool,
    ) -> Result<Option<VoiceUtterance>, String> {
        let now = Instant::now();
        let speech_ended = now.checked_sub(self.trailing_silence).unwrap_or(now);
        let queue_wait = now.saturating_duration_since(speech_ended);

        let arrived_while_busy = check_busy_drop
            && (self.is_processing
                || self
                    .last_processing_ended
                    .is_some_and(|ended| speech_ended <= ended));

        let duration_ms = clip_duration_ms(&utterance.samples);
        let rms = clip_rms(&utterance.samples);
        let is_short_or_quiet = duration_ms < 200 || rms < self.start_threshold_rms;

        let transcribe_start = Instant::now();
        let Some(heard) = self.stt.transcribe(&utterance.samples)? else {
            return Ok(None);
        };
        let transcription_time = transcribe_start.elapsed();

        if arrived_while_busy && is_short_or_quiet && !contains_wake_word(&heard.text) {
            return Ok(None);
        }

        if !self.is_processing {
            self.last_processing_ended = None;
        }

        Ok(Some(VoiceUtterance {
            heard,
            speech_ended,
            queue_wait,
            transcription_time,
        }))
    }
}

impl io::Read for VoiceInput {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let available = self.fill_buf()?;
        let amount = available.len().min(buffer.len());
        buffer[..amount].copy_from_slice(&available[..amount]);
        self.consume(amount);
        Ok(amount)
    }
}

impl BufRead for VoiceInput {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        if self.consumed < self.pending.len() {
            return Ok(&self.pending[self.consumed..]);
        }
        self.pending.clear();
        self.consumed = 0;

        let timeout = self.confirmation_timeout;
        match self.next_utterance_with_filter(Some(timeout), false) {
            Ok(Some(utterance)) => {
                self.pending
                    .extend_from_slice(utterance.heard.text.as_bytes());
                self.pending.push(b'\n');
                Ok(&self.pending)
            }
            Ok(None) => Ok(&[]),
            Err(error) => Err(io::Error::other(error)),
        }
    }

    fn consume(&mut self, amount: usize) {
        self.consumed = self.consumed.saturating_add(amount).min(self.pending.len());
    }
}

/// Computes the duration in milliseconds of audio samples at 16 kHz.
///
/// # Arguments
/// * `samples` - Slice of floating point audio samples.
///
/// # Returns
/// Milliseconds represented by the sample slice.
pub fn clip_duration_ms(samples: &[f32]) -> u64 {
    (samples.len() as u64 * 1000) / SAMPLE_RATE as u64
}

/// Computes the root-mean-square energy of an audio slice.
///
/// # Arguments
/// * `samples` - Slice of floating point audio samples.
///
/// # Returns
/// RMS amplitude value.
pub fn clip_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// Returns whether `text` contains the primary wake word "reach" or alias "rich".
///
/// # Arguments
/// * `text` - The recognised transcript to check.
///
/// # Returns
/// `true` if the wake word is found as a whole token, `false` otherwise.
pub fn contains_wake_word(text: &str) -> bool {
    let lower = text.to_lowercase();
    lower
        .split(|c: char| !c.is_alphanumeric())
        .any(|w| w == "reach" || w == "rich" || w == "ridge")
}

#[cfg(test)]
mod tests {
    use super::*;
    use reach_core::{Direction, Executor, Intent};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    /// Mock capture control tracking pause and resume invocations.
    #[derive(Default, Clone)]
    struct MockControl {
        is_paused: Arc<AtomicBool>,
        pause_count: Arc<AtomicUsize>,
        resume_count: Arc<AtomicUsize>,
    }

    impl CaptureControl for MockControl {
        fn pause(&self) {
            self.is_paused.store(true, Ordering::SeqCst);
            self.pause_count.fetch_add(1, Ordering::SeqCst);
        }

        fn resume(&self) {
            self.is_paused.store(false, Ordering::SeqCst);
            self.resume_count.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// Mock executor recording whether capture was paused while it executed.
    struct CheckingExecutor {
        control: MockControl,
        was_paused_during_run: bool,
    }

    impl reach_core::Executor for CheckingExecutor {
        fn run(&mut self, _intent: &Intent) -> Result<String, String> {
            self.was_paused_during_run = self.control.is_paused.load(Ordering::SeqCst);
            Ok("done".to_owned())
        }
    }

    /// Verifies that PausingExecutor pauses the microphone during execution and resumes afterwards.
    #[test]
    fn pausing_executor_pauses_during_run_and_resumes_after() {
        let control = MockControl::default();
        let checking = CheckingExecutor {
            control: control.clone(),
            was_paused_during_run: false,
        };
        let mut executor = PausingExecutor::new(checking, Box::new(control.clone()));
        let start = Instant::now();
        let intent = Intent::Scroll {
            direction: Direction::Down,
            amount: None,
        };
        let res = executor.run(&intent);
        let elapsed = start.elapsed();

        assert_eq!(res, Ok("done".to_owned()));
        assert!(executor.inner().was_paused_during_run);
        assert!(!control.is_paused.load(Ordering::SeqCst));
        assert_eq!(control.pause_count.load(Ordering::SeqCst), 1);
        assert_eq!(control.resume_count.load(Ordering::SeqCst), 1);
        // Verify 200 ms sleep was applied
        assert!(elapsed >= Duration::from_millis(190));
    }

    /// Verifies wake word detection helper.
    #[test]
    fn wake_word_detection() {
        assert!(contains_wake_word("reach scroll down"));
        assert!(contains_wake_word("rich click 5"));
        assert!(contains_wake_word("ridge open chrome"));
        assert!(!contains_wake_word("hello world"));
    }
}
