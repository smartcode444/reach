//! Voice mode for the CLI harness.
//!
//! The production flow is microphone capture on a worker thread, endpointing
//! and speech-to-text on the main thread, then the same [`Pipeline`] the typed
//! mode uses. A [`VoiceInput`] implements [`BufRead`], so the pipeline reads a
//! spoken confirmation answer exactly where it would read a typed one. The
//! pipeline remains the only routing path.
//!
//! No audio or transcript is written to disk or logged. With `--captions` the
//! cleaned transcript and the estimated end-of-speech-to-result latency are
//! printed to stdout; without it, nothing about the transcript is printed.

use std::io::{self, BufRead, Write};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use audio::{EndpointConfig, Endpointer, MicCapture};
use reach_core::Heard;
use stt::{SttConfig, SttError, Transcriber, clean_transcript};

use crate::executor::CompositeExecutor;
use crate::pipeline::{Pipeline, PipelineIo};
use crate::speaker::{NullSpeaker, Phrase, Speaker, TtsSpeaker};

/// The Whisper model loaded when `--stt-model` is not supplied.
pub const DEFAULT_STT_MODEL_PATH: &str = "models/ggml-base.en.bin";

/// How long the pipeline waits for a spoken yes or no after announcing an
/// action that needs confirmation.
pub const CONFIRMATION_TIMEOUT: Duration = Duration::from_secs(6);

/// How long one frame read waits before the main loop checks its deadline.
const FRAME_WAIT: Duration = Duration::from_millis(250);

/// The options accepted by `cli listen`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ListenArgs {
    /// Whether to load and use the Needle model.
    pub model: bool,
    /// Path to the Needle model container, if explicitly provided.
    pub model_path: Option<String>,
    /// Whether to print the transcript and latency to stdout.
    pub captions: bool,
    /// Whether to print route decisions instead of running them.
    pub dry_run: bool,
    /// Path to the Whisper GGML model, if explicitly provided.
    pub stt_model: Option<String>,
    /// Whether to disable voice output.
    pub no_voice: bool,
}

/// Parses the arguments following the `listen` subcommand.
///
/// # Arguments
/// * `args` - Everything after the `listen` keyword.
///
/// # Returns
/// The parsed voice-mode settings.
///
/// # Errors
/// Returns a message for an unknown option, a missing `--stt-model` path, or a
/// positional word. Positional words are rejected because voice mode has no
/// typed utterance and a mistyped flag must not become one.
pub fn parse_listen_args(args: &[String]) -> Result<ListenArgs, String> {
    let mut model = false;
    let mut model_path = None;
    let mut captions = false;
    let mut dry_run = false;
    let mut stt_model = None;
    let mut no_voice = false;

    let mut index = 0;
    while index < args.len() {
        let argument = &args[index];
        if argument == "--model" {
            model = true;
            if let Some(next) = args.get(index + 1).filter(|next| {
                !next.starts_with("--")
                    && (next.ends_with(".cact")
                        || next.ends_with(".bin")
                        || next.contains('/')
                        || next.contains('\\'))
            }) {
                model_path = Some(next.clone());
                index += 1;
            }
        } else if let Some(path) = argument.strip_prefix("--model=") {
            model = true;
            if !path.is_empty() {
                model_path = Some(path.to_owned());
            }
        } else if argument == "--no-voice" {
            no_voice = true;
        } else if argument == "--captions" {
            captions = true;
        } else if argument == "--dry-run" {
            dry_run = true;
        } else if argument == "--stt-model" {
            let Some(path) = args.get(index + 1).filter(|next| !next.starts_with("--")) else {
                return Err("--stt-model needs a path".to_owned());
            };
            stt_model = Some(path.clone());
            index += 1;
        } else if let Some(path) = argument.strip_prefix("--stt-model=") {
            if path.is_empty() {
                return Err("--stt-model needs a path".to_owned());
            }
            stt_model = Some(path.to_owned());
        } else if argument == "listen" {
            // Tolerated so callers may pass the full subcommand list.
        } else if argument.starts_with("--") {
            return Err(format!("unknown listen option: {argument}"));
        } else {
            return Err("listen does not take an utterance".to_owned());
        }
        index += 1;
    }

    Ok(ListenArgs {
        model,
        model_path,
        captions,
        dry_run,
        stt_model,
        no_voice,
    })
}

/// One frame read from the audio source.
#[derive(Debug)]
pub enum FramePoll {
    /// A mono 16 kHz frame arrived.
    Frame(Vec<f32>),
    /// No frame arrived before the wait elapsed.
    Timeout,
    /// The source has stopped and no more frames will arrive.
    Closed,
}

/// Supplies audio frames to the main loop.
pub trait FrameSource {
    /// Waits up to `timeout` for the next frame.
    ///
    /// # Arguments
    /// * `timeout` - Maximum time to wait. A short wait keeps the main loop
    ///   responsive to deadlines and shutdown.
    ///
    /// # Returns
    /// A frame, a timeout, or a closed source.
    fn next_frame(&mut self, timeout: Duration) -> FramePoll;
}

/// Pauses and resumes microphone delivery.
///
/// The executor and the feedback writer both hold a handle so audio is not
/// captured while an action runs or feedback is being produced.
pub trait CaptureControl {
    /// Stops delivering frames. Best effort: a stopped worker has nothing left
    /// to pause.
    fn pause(&self);

    /// Resumes delivering frames after [`pause`](Self::pause).
    fn resume(&self);
}

/// Resumes capture when it goes out of scope, including during an unwind.
struct PauseGuard<'a> {
    /// The control that will be resumed on drop.
    control: &'a dyn CaptureControl,
}

impl<'a> PauseGuard<'a> {
    /// Pauses `control` and returns a guard that resumes it.
    ///
    /// # Arguments
    /// * `control` - The capture handle to pause.
    ///
    /// # Returns
    /// A guard whose drop resumes the capture.
    fn new(control: &'a dyn CaptureControl) -> Self {
        control.pause();
        Self { control }
    }
}

impl Drop for PauseGuard<'_> {
    /// Resumes capture when the guard leaves scope.
    fn drop(&mut self) {
        self.control.resume();
    }
}

/// Writes feedback while pausing the microphone and speaking feedback phrases.
///
/// Output lines written to this writer are displayed on `inner` and, if they
/// match a known feedback phrase, spoken aloud through `speaker` while pausing
/// microphone capture. Microphone capture resumes 200 ms after speech concludes
/// so Reach never captures its own voice.
pub struct FeedbackWriter<W: Write> {
    /// The stream that receives the feedback.
    inner: W,
    /// The capture handle paused around each write and speech output.
    control: Box<dyn CaptureControl>,
    /// The speaker that announces feedback.
    speaker: Box<dyn Speaker>,
    /// Internal line buffer to detect complete lines for speech output.
    buffer: String,
}

impl<W: Write> FeedbackWriter<W> {
    /// Creates a feedback writer around `inner`, `control`, and `speaker`.
    ///
    /// # Arguments
    /// * `inner` - The stream that receives user-facing feedback.
    /// * `control` - The capture handle paused while feedback is written or spoken.
    /// * `speaker` - The speaker engine used for spoken feedback.
    ///
    /// # Returns
    /// An initialized [`FeedbackWriter`].
    pub fn new(inner: W, control: Box<dyn CaptureControl>, speaker: Box<dyn Speaker>) -> Self {
        Self {
            inner,
            control,
            speaker,
            buffer: String::new(),
        }
    }

    /// Speaks `phrase`, pausing capture during speech and waiting 200 ms after speech ends.
    ///
    /// # Arguments
    /// * `phrase` - The closed phrase to speak.
    pub fn speak(&mut self, phrase: Phrase) {
        let _guard = PauseGuard::new(self.control.as_ref());
        if let Err(_error) = self.speaker.speak(phrase) {
            // Degrade to silent output; do not crash the loop.
        }
        std::thread::sleep(Duration::from_millis(200));
    }

    /// Accesses the underlying speaker for testing.
    ///
    /// # Returns
    /// A reference to the [`Speaker`] trait object.
    pub fn speaker(&self) -> &dyn Speaker {
        self.speaker.as_ref()
    }
}

impl<W: Write> Write for FeedbackWriter<W> {
    /// Writes feedback with the microphone paused and speaks detected phrases.
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let _guard = PauseGuard::new(self.control.as_ref());
        let written = self.inner.write(buffer)?;

        if let Ok(text) = std::str::from_utf8(&buffer[..written]) {
            self.buffer.push_str(text);
            while let Some(pos) = self.buffer.find('\n') {
                let line = self.buffer[..pos].trim().to_owned();
                self.buffer.drain(..=pos);
                if let Some(phrase) = crate::speaker::phrase_from_line(&line) {
                    if let Err(_error) = self.speaker.speak(phrase) {
                        // Degrade to silent output; do not crash the loop.
                    }
                    std::thread::sleep(Duration::from_millis(200));
                }
            }
        }

        Ok(written)
    }

    /// Flushes the underlying stream without pausing capture.
    fn flush(&mut self) -> io::Result<()> {
        let _guard = PauseGuard::new(self.control.as_ref());
        if !self.buffer.is_empty() {
            let line = self.buffer.trim().to_owned();
            self.buffer.clear();
            if let Some(phrase) = crate::speaker::phrase_from_line(&line) {
                if let Err(_error) = self.speaker.speak(phrase) {
                    // Degrade to silent output; do not crash the loop.
                }
                std::thread::sleep(Duration::from_millis(200));
            }
        }
        self.inner.flush()
    }
}

/// An executor that pauses the microphone for the duration of each action.
pub struct PausingExecutor<E> {
    /// The executor that performs the action.
    inner: E,
    /// The capture handle paused while the action runs.
    control: Box<dyn CaptureControl>,
}

impl<E> PausingExecutor<E> {
    /// Wraps `inner` so every action runs with the microphone paused.
    ///
    /// # Arguments
    /// * `inner` - The real executor.
    /// * `control` - The capture handle paused around execution.
    ///
    /// # Returns
    /// An initialized [`PausingExecutor`].
    pub fn new(inner: E, control: Box<dyn CaptureControl>) -> Self {
        Self { inner, control }
    }
}

impl<E: reach_core::Executor> reach_core::Executor for PausingExecutor<E> {
    /// Runs the action with capture paused, resuming even on failure.
    fn run(&mut self, intent: &reach_core::Intent) -> Result<String, String> {
        let _guard = PauseGuard::new(self.control.as_ref());
        self.inner.run(intent)
    }
}

/// A command sent to the capture worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CaptureCommand {
    /// Stop delivering frames.
    Pause,
    /// Resume delivering frames.
    Resume,
    /// Stop the worker and release the microphone.
    Stop,
}

/// A cloneable handle for controlling the capture worker.
#[derive(Clone)]
pub struct CaptureHandle {
    /// The channel shared with the worker.
    commands: Sender<CaptureCommand>,
}

impl CaptureControl for CaptureHandle {
    /// Asks the worker to stop delivering frames.
    fn pause(&self) {
        let _ = self.commands.send(CaptureCommand::Pause);
    }

    /// Asks the worker to resume delivering frames.
    fn resume(&self) {
        let _ = self.commands.send(CaptureCommand::Resume);
    }
}

/// A microphone capture running on its own thread.
pub struct CaptureSource {
    /// Frames delivered by the worker.
    frames: Receiver<Vec<f32>>,
    /// Commands sent to the worker.
    commands: Sender<CaptureCommand>,
    /// The worker thread, joined when this source is dropped.
    worker: Option<JoinHandle<()>>,
}

impl CaptureSource {
    /// Opens the default microphone on a dedicated worker thread.
    ///
    /// # Returns
    /// A running source and its frame receiver. Keep the source alive for as
    /// long as frames are wanted.
    ///
    /// # Errors
    /// Returns a message when the worker cannot be spawned or the microphone
    /// cannot be opened.
    pub fn start() -> Result<Self, String> {
        let (command_sender, command_receiver) = mpsc::channel();
        let (frame_sender, frame_receiver) = mpsc::channel();
        let (ready_sender, ready_receiver) = mpsc::channel();

        let worker = thread::Builder::new()
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
                    // Drain every queued command first. A confirmation writes
                    // several short feedback lines, and processing only one
                    // pause/resume per frame would leave the microphone paused
                    // when the answer starts.
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
                worker: Some(worker),
            }),
            Ok(Err(error)) => {
                let _ = worker.join();
                Err(format!("could not start the microphone: {error}"))
            }
            Err(_) => {
                // Do not join here: a microphone that is slow to open must not
                // block the error path. The worker still sees the stop command
                // once it finishes opening.
                let _ = command_sender.send(CaptureCommand::Stop);
                Err("the microphone did not start in time".to_owned())
            }
        }
    }

    /// Returns a handle that can pause and resume the worker.
    ///
    /// # Returns
    /// A cloneable [`CaptureHandle`] sharing this source's command channel.
    pub fn handle(&self) -> CaptureHandle {
        CaptureHandle {
            commands: self.commands.clone(),
        }
    }
}

impl FrameSource for CaptureSource {
    /// Receives the next microphone frame.
    fn next_frame(&mut self, timeout: Duration) -> FramePoll {
        match self.frames.recv_timeout(timeout) {
            Ok(frame) => FramePoll::Frame(frame),
            Err(RecvTimeoutError::Timeout) => FramePoll::Timeout,
            Err(RecvTimeoutError::Disconnected) => FramePoll::Closed,
        }
    }
}

impl Drop for CaptureSource {
    /// Stops the worker and joins it so the microphone is released.
    fn drop(&mut self) {
        let _ = self.commands.send(CaptureCommand::Stop);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Turns audio samples into text.
///
/// Production uses [`Transcriber`]; tests use a scripted implementation so no
/// model file or microphone is needed.
pub trait SpeechToText {
    /// Transcribes one mono 16 kHz utterance.
    ///
    /// # Arguments
    /// * `samples` - The endpointed audio.
    ///
    /// # Returns
    /// The cleaned text and its confidence, or `None` when the audio contained
    /// no usable speech.
    ///
    /// # Errors
    /// Returns a message when the recogniser fails. The message never contains
    /// the transcript.
    fn transcribe(&mut self, samples: &[f32]) -> Result<Option<Heard>, String>;
}

impl SpeechToText for Transcriber {
    /// Transcribes with Whisper and applies the shared transcript cleanup.
    fn transcribe(&mut self, samples: &[f32]) -> Result<Option<Heard>, String> {
        match Transcriber::transcribe(self, samples) {
            Ok(heard) => {
                let Some(text) = clean_transcript(&heard.text) else {
                    return Ok(None);
                };
                Ok(Some(Heard {
                    text,
                    confidence: heard.confidence,
                }))
            }
            Err(SttError::NoSpeech) => Ok(None),
            Err(error) => Err(format!("speech recognition failed: {error}")),
        }
    }
}

/// One recognised utterance and the estimated instant its speech ended.
#[derive(Debug, Clone, PartialEq)]
pub struct VoiceUtterance {
    /// The cleaned transcript.
    pub heard: Heard,
    /// Estimated end of speech, derived by subtracting the endpointer's
    /// trailing-silence window from the moment the utterance was emitted.
    pub speech_ended: Instant,
    /// Time spent waiting for trailing silence before emitting the utterance.
    pub endpoint_wait: Duration,
    /// Time taken to transcribe the audio into text.
    pub transcription_time: Duration,
}

/// The voice-backed input consumed by the pipeline.
///
/// The main loop calls [`next_utterance`](Self::next_utterance) for commands.
/// During confirmation the pipeline calls [`BufRead::read_line`] on the same
/// value, so the next spoken utterance becomes the yes/no answer. A read that
/// times out returns end-of-input, which the pipeline treats as a cancellation.
pub struct VoiceInput {
    /// The microphone or a scripted audio source.
    frames: Box<dyn FrameSource>,
    /// The speech recogniser or a scripted stand-in.
    stt: Box<dyn SpeechToText>,
    /// The endpointing state machine.
    endpointer: Endpointer,
    /// Trailing silence included in each emitted utterance, used to estimate
    /// when the user actually stopped speaking.
    trailing_silence: Duration,
    /// How long a confirmation read waits for the next utterance.
    confirmation_timeout: Duration,
    /// The line currently offered to `BufRead`.
    pending: Vec<u8>,
    /// How many bytes of `pending` have already been consumed.
    consumed: usize,
    /// Set once the audio source reports that it has closed.
    closed: bool,
}

impl VoiceInput {
    /// Creates a voice input from an audio source and a recogniser.
    ///
    /// # Arguments
    /// * `frames` - The audio frame source.
    /// * `stt` - The speech recogniser.
    /// * `endpoint_config` - Endpointing settings. The trailing-silence
    ///   setting is also used to estimate when speech ended.
    /// * `confirmation_timeout` - How long a confirmation read waits for an
    ///   answer before reporting end-of-input.
    ///
    /// # Returns
    /// An initialized [`VoiceInput`].
    ///
    /// # Errors
    /// Returns a message when the endpointing configuration is invalid.
    pub fn new(
        frames: Box<dyn FrameSource>,
        stt: Box<dyn SpeechToText>,
        endpoint_config: EndpointConfig,
        confirmation_timeout: Duration,
    ) -> Result<Self, String> {
        let endpointer = Endpointer::new(endpoint_config)
            .map_err(|error| format!("invalid endpoint configuration: {error}"))?;
        Ok(Self {
            frames,
            stt,
            endpointer,
            trailing_silence: Duration::from_millis(u64::from(endpoint_config.end_silence_ms)),
            confirmation_timeout,
            pending: Vec::new(),
            consumed: 0,
            closed: false,
        })
    }

    /// Waits for, endpoints and transcribes the next utterance.
    ///
    /// # Arguments
    /// * `timeout` - Overall deadline for an utterance, or `None` to wait until
    ///   speech arrives or the source closes.
    ///
    /// # Returns
    /// The recognised utterance, or `None` when the deadline passed or the
    /// source closed. Audio that cleans to no speech is skipped.
    ///
    /// # Errors
    /// Returns the recogniser's error message. It never contains the
    /// transcript.
    pub fn next_utterance(
        &mut self,
        timeout: Option<Duration>,
    ) -> Result<Option<VoiceUtterance>, String> {
        if self.closed {
            return Ok(None);
        }

        let deadline = timeout.map(|duration| Instant::now() + duration);
        loop {
            let wait = match deadline {
                Some(deadline) => {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    if remaining.is_zero() {
                        return Ok(None);
                    }
                    remaining.min(FRAME_WAIT)
                }
                None => FRAME_WAIT,
            };

            match self.frames.next_frame(wait) {
                FramePoll::Frame(frame) => {
                    if let Some(utterance) = self.endpointer.push_frame(&frame) {
                        let now = Instant::now();
                        let speech_ended = now
                            .checked_sub(self.trailing_silence)
                            .unwrap_or(now);
                        let endpoint_wait = now.saturating_duration_since(speech_ended);
                        let transcribe_start = Instant::now();
                        if let Some(heard) = self.stt.transcribe(&utterance.samples)? {
                            let transcription_time = transcribe_start.elapsed();
                            return Ok(Some(VoiceUtterance {
                                heard,
                                speech_ended,
                                endpoint_wait,
                                transcription_time,
                            }));
                        }
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
}

impl io::Read for VoiceInput {
    /// Reads bytes from the current spoken line into `buffer`.
    ///
    /// `BufRead` requires `Read`; the pipeline itself only uses the buffered
    /// line interface.
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let available = self.fill_buf()?;
        let amount = available.len().min(buffer.len());
        buffer[..amount].copy_from_slice(&available[..amount]);
        self.consume(amount);
        Ok(amount)
    }
}

impl BufRead for VoiceInput {
    /// Offers the next spoken utterance as one newline-terminated line.
    ///
    /// The pipeline uses this only while waiting for a confirmation answer. A
    /// timeout or a closed source returns an empty buffer, which `read_line`
    /// reports as end-of-input; the pipeline then cancels the action.
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        if self.consumed < self.pending.len() {
            return Ok(&self.pending[self.consumed..]);
        }
        self.pending.clear();
        self.consumed = 0;

        let timeout = self.confirmation_timeout;
        match self.next_utterance(Some(timeout)) {
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

    /// Marks `amount` bytes of the current line as consumed.
    fn consume(&mut self, amount: usize) {
        self.consumed = self.consumed.saturating_add(amount).min(self.pending.len());
    }
}

/// Runs the voice loop until the microphone closes or Ctrl+C exits.
///
/// # Arguments
/// * `pipeline` - The shared wake, routing and execution path.
/// * `voice` - The voice-backed input and confirmation answer source.
/// * `feedback` - The writer for captions, announcements and results.
/// * `captions` - Whether to print the transcript and latency.
///
/// # Returns
/// `Ok(())` when the audio source closes.
///
/// # Errors
/// Returns a message when recognition, a pipeline read, or feedback writing
/// fails. The message never contains the transcript.
pub fn run_loop<M, E>(
    pipeline: &mut Pipeline<M, E>,
    voice: &mut VoiceInput,
    feedback: &mut dyn Write,
    captions: bool,
) -> Result<(), String>
where
    M: nlu::IntentModel,
    E: reach_core::Executor,
{
    while let Some(input) = voice.next_utterance(None)? {

        if captions {
            writeln!(feedback, "heard: {}", input.heard.text)
                .map_err(|error| format!("could not write captions: {error}"))?;
        }

        {
            let mut io = PipelineIo::new(&mut *voice, &mut *feedback);
            // Passing the estimated speech-end time, rather than the time
            // transcription finished, keeps a slow model from consuming the
            // wake window before the next command is handled.
            pipeline
                .process(&input.heard.text, input.speech_ended, &mut io)
                .map_err(|error| format!("pipeline I/O failed: {error}"))?;
        }

        if captions {
            let latency = Instant::now().saturating_duration_since(input.speech_ended);
            writeln!(
                feedback,
                "latency: {} ms (endpoint wait: {} ms, transcription time: {} ms, parse/route time: {} ms, execute time: {} ms)",
                latency.as_millis(),
                input.endpoint_wait.as_millis(),
                input.transcription_time.as_millis(),
                pipeline.last_route_duration().as_millis(),
                pipeline.last_execute_duration().as_millis(),
            )
            .map_err(|error| format!("could not write latency: {error}"))?;
        }
    }

    Ok(())
}

/// Runs `cli listen`.
///
/// # Arguments
/// * `args` - Everything after the `listen` subcommand.
///
/// # Returns
/// `Ok(())` when the microphone closes and the loop exits.
///
/// # Errors
/// Returns a message for an invalid option, a model load failure, a microphone
/// failure, or a pipeline error. No message contains the transcript.
pub fn run(args: &[String]) -> Result<(), String> {
    let parsed = parse_listen_args(args)?;
    install_ctrl_c_handler();

    let model = crate::load_cli_model(parsed.model, parsed.model_path.as_deref())?;
    let executor = if parsed.dry_run {
        crate::CliExecutor::DryRun
    } else {
        crate::CliExecutor::Real(CompositeExecutor::new()?)
    };

    let stt_path = parsed
        .stt_model
        .as_deref()
        .unwrap_or(DEFAULT_STT_MODEL_PATH);
    let transcriber = Transcriber::load(stt_path, SttConfig::default())
        .map_err(|error| format!("failed to load the speech model: {error}"))?;

    let capture = CaptureSource::start()?;
    let control = capture.handle();

    let mut pipeline = Pipeline::new(
        model,
        PausingExecutor::new(executor, Box::new(control.clone())),
        parsed.dry_run,
    );
    let mut voice = VoiceInput::new(
        Box::new(capture),
        Box::new(transcriber),
        EndpointConfig::default(),
        CONFIRMATION_TIMEOUT,
    )?;

    let stdout = io::stdout();
    let speaker: Box<dyn Speaker> = if parsed.no_voice {
        Box::new(NullSpeaker::new())
    } else {
        match TtsSpeaker::new() {
            Ok(speaker) => Box::new(speaker),
            Err(_) => Box::new(NullSpeaker::new()),
        }
    };
    let mut feedback = FeedbackWriter::new(stdout.lock(), Box::new(control), speaker);
    let result = run_loop(&mut pipeline, &mut voice, &mut feedback, parsed.captions);
    exec::release_all_keys();
    result
}

/// Installs the Ctrl+C handler that releases simulated input before exiting.
///
/// Registration failure is ignored because there may be no console attached
/// (for example when stdout is piped); the normal exit path still releases the
/// keys.
pub fn install_ctrl_c_handler() {
    #[cfg(windows)]
    {
        // SAFETY: `ctrl_handler` has the `PHANDLER_ROUTINE` signature and is a
        // plain function that remains valid for the lifetime of the process.
        let _ = unsafe {
            windows::Win32::System::Console::SetConsoleCtrlHandler(Some(ctrl_handler), true)
        };
    }
}

/// Handles Ctrl+C and Ctrl+Break by releasing simulated input and exiting.
///
/// # Arguments
/// * `ctrl_type` - The console control event supplied by Windows.
///
/// # Returns
/// Zero (not handled) for every event other than Ctrl+C and Ctrl+Break. Those
/// two release input and terminate the process, so no return value is seen.
#[cfg(windows)]
unsafe extern "system" fn ctrl_handler(ctrl_type: u32) -> windows::core::BOOL {
    use windows::Win32::System::Console::{CTRL_BREAK_EVENT, CTRL_C_EVENT};

    if ctrl_type == CTRL_C_EVENT || ctrl_type == CTRL_BREAK_EVENT {
        // A panic must not unwind across the Windows callback boundary.
        let _ = std::panic::catch_unwind(exec::release_all_keys);
        std::process::exit(130);
    }
    windows::core::BOOL(0)
}

/// Unit tests for voice mode.
#[cfg(test)]
mod tests {
    use crate::speaker::ActionLabel;
    use std::cell::{Cell, RefCell};
    use std::collections::VecDeque;
    use std::rc::Rc;

    use audio::FRAME_SAMPLES;
    use nlu::CLARIFY_QUESTION;
    use reach_core::{Direction, Executor, Intent};

    use crate::pipeline::DisabledModel;

    use super::*;

    /// An audio source that returns a fixed script of events.
    struct ScriptedFrames {
        /// Events left to return.
        events: VecDeque<FramePoll>,
    }

    impl ScriptedFrames {
        /// Creates a source from the scripted events.
        fn new(events: Vec<FramePoll>) -> Self {
            Self {
                events: events.into(),
            }
        }
    }

    impl FrameSource for ScriptedFrames {
        /// Returns the next scripted event, or `Closed` when exhausted.
        fn next_frame(&mut self, _timeout: Duration) -> FramePoll {
            self.events.pop_front().unwrap_or(FramePoll::Closed)
        }
    }

    /// A recogniser that returns scripted text in order.
    struct ScriptedTranscriber {
        /// Replies left to return.
        replies: VecDeque<Option<String>>,
    }

    impl ScriptedTranscriber {
        /// Creates a recogniser from scripted replies.
        fn new(replies: Vec<Option<&str>>) -> Self {
            Self {
                replies: replies
                    .into_iter()
                    .map(|reply| reply.map(str::to_owned))
                    .collect(),
            }
        }
    }

    impl SpeechToText for ScriptedTranscriber {
        /// Returns the next scripted reply after applying the shared cleanup.
        fn transcribe(&mut self, _samples: &[f32]) -> Result<Option<Heard>, String> {
            let reply = self.replies.pop_front().flatten();
            Ok(reply
                .and_then(|text| clean_transcript(&text))
                .map(|text| Heard {
                    text,
                    confidence: Some(1.0),
                }))
        }
    }

    /// A capture control that records pause and resume calls.
    #[derive(Clone, Default)]
    struct ScriptedControl {
        /// Whether capture is currently paused.
        paused: Rc<Cell<bool>>,
        /// Number of pause calls.
        pauses: Rc<Cell<usize>>,
        /// Number of resume calls.
        resumes: Rc<Cell<usize>>,
    }

    impl ScriptedControl {
        /// Returns whether capture is paused.
        fn is_paused(&self) -> bool {
            self.paused.get()
        }

        /// Returns the number of pause calls.
        fn pause_count(&self) -> usize {
            self.pauses.get()
        }

        /// Returns the number of resume calls.
        fn resume_count(&self) -> usize {
            self.resumes.get()
        }
    }

    impl CaptureControl for ScriptedControl {
        /// Marks capture as paused.
        fn pause(&self) {
            self.paused.set(true);
            self.pauses.set(self.pauses.get() + 1);
        }

        /// Marks capture as running.
        fn resume(&self) {
            self.paused.set(false);
            self.resumes.set(self.resumes.get() + 1);
        }
    }

    /// A fake executor that records intents and the pause state during each run.
    struct RecordingExecutor {
        /// Intents passed to `run`, in order.
        intents: Rc<RefCell<Vec<Intent>>>,
        /// Whether capture was paused during each run.
        paused_during_run: Rc<RefCell<Vec<bool>>>,
        /// The control shared with the pausing executor.
        control: ScriptedControl,
    }

    impl Executor for RecordingExecutor {
        /// Records the intent and the pause state, then reports success.
        fn run(&mut self, intent: &Intent) -> Result<String, String> {
            self.intents.borrow_mut().push(intent.clone());
            self.paused_during_run
                .borrow_mut()
                .push(self.control.is_paused());
            Ok("executed".to_owned())
        }
    }

    /// The observable result of one scripted listen run.
    struct RunResult {
        /// Intents the executor received.
        intents: Vec<Intent>,
        /// Whether capture was paused during each execution.
        paused_during_run: Vec<bool>,
        /// Everything written to the feedback stream.
        output: String,
        /// Number of pause calls.
        pause_count: usize,
        /// Number of resume calls.
        resume_count: usize,
        /// Phrases spoken through the speaker.
        phrases: Vec<Phrase>,
    }

    /// Returns a fast endpointing configuration for the scripted frames.
    fn test_endpoint_config() -> EndpointConfig {
        EndpointConfig {
            min_speech_ms: 100,
            end_silence_ms: 100,
            pre_roll_ms: 20,
            max_utterance_ms: 5_000,
            ..EndpointConfig::default()
        }
    }

    /// Builds one 20 ms speech frame.
    fn speech_frame() -> FramePoll {
        FramePoll::Frame(vec![0.1; FRAME_SAMPLES])
    }

    /// Builds one 20 ms silence frame.
    fn silence_frame() -> FramePoll {
        FramePoll::Frame(vec![0.0; FRAME_SAMPLES])
    }

    /// Builds enough frames for one complete utterance: 600 ms of speech and
    /// 100 ms of trailing silence.
    fn utterance_frames() -> Vec<FramePoll> {
        let mut events = Vec::new();
        for _ in 0..30 {
            events.push(speech_frame());
        }
        for _ in 0..5 {
            events.push(silence_frame());
        }
        events
    }

    /// Runs one scripted conversation and returns what happened.
    fn run_script(
        frames: Vec<FramePoll>,
        transcripts: Vec<Option<&str>>,
        confirmation_timeout: Duration,
        captions: bool,
    ) -> RunResult {
        let frames = ScriptedFrames::new(frames);
        let transcriber = ScriptedTranscriber::new(transcripts);
        let mut voice = VoiceInput::new(
            Box::new(frames),
            Box::new(transcriber),
            test_endpoint_config(),
            confirmation_timeout,
        )
        .expect("the test endpoint configuration must be valid");

        let control = ScriptedControl::default();
        let intents = Rc::new(RefCell::new(Vec::new()));
        let paused_during_run = Rc::new(RefCell::new(Vec::new()));
        let executor = RecordingExecutor {
            intents: Rc::clone(&intents),
            paused_during_run: Rc::clone(&paused_during_run),
            control: control.clone(),
        };
        let mut pipeline = Pipeline::new(
            DisabledModel,
            PausingExecutor::new(executor, Box::new(control.clone())),
            false,
        );

        let mut output = Vec::new();
        let speaker = NullSpeaker::new();
        let recorded = speaker.clone();
        {
            let mut feedback = FeedbackWriter::new(&mut output, Box::new(control.clone()), Box::new(speaker));
            run_loop(&mut pipeline, &mut voice, &mut feedback, captions)
                .expect("the scripted pipeline must finish");
        }

        RunResult {
            intents: intents.borrow().clone(),
            paused_during_run: paused_during_run.borrow().clone(),
            output: String::from_utf8(output).expect("feedback must be valid UTF-8"),
            pause_count: control.pause_count(),
            resume_count: control.resume_count(),
            phrases: recorded.phrases(),
        }
    }

    /// A bare wake word opens the window for the next command.
    #[test]
    fn wake_only_opens_the_window_for_the_next_command() {
        let mut frames = utterance_frames();
        frames.extend(utterance_frames());

        let result = run_script(
            frames,
            vec![Some("Reach"), Some("scroll down")],
            Duration::ZERO,
            false,
        );

        assert_eq!(
            result.intents,
            vec![Intent::Scroll {
                direction: Direction::Down,
                amount: None,
            }]
        );
        assert!(result.output.contains("listening"));
    }

    /// "stop" closes the open window, so the following command is ignored.
    #[test]
    fn stop_closes_the_listening_window() {
        let mut frames = utterance_frames();
        frames.extend(utterance_frames());
        frames.extend(utterance_frames());

        let result = run_script(
            frames,
            vec![Some("Reach"), Some("stop"), Some("scroll down")],
            Duration::ZERO,
            false,
        );

        assert!(result.intents.contains(&Intent::Cancel));
        assert!(
            !result
                .intents
                .iter()
                .any(|intent| matches!(intent, Intent::Scroll { .. })),
            "a command after stop must not run"
        );
        assert!(result.output.contains("ignored"));
    }

    /// Speech without the wake word is ignored and never reaches the executor.
    #[test]
    fn command_without_the_wake_word_is_ignored() {
        let result = run_script(
            utterance_frames(),
            vec![Some("scroll down")],
            Duration::ZERO,
            false,
        );

        assert!(result.intents.is_empty());
        assert!(result.output.contains("ignored"));
    }

    /// A spoken "yes" confirms a risky action and runs it.
    #[test]
    fn confirm_yes_runs_the_action() {
        let mut frames = utterance_frames();
        frames.extend(utterance_frames());

        let result = run_script(
            frames,
            vec![Some("Reach click Submit"), Some("yes")],
            Duration::from_secs(1),
            false,
        );

        assert_eq!(
            result.intents,
            vec![Intent::ClickElement {
                target: "Submit".to_owned(),
            }]
        );
        assert!(result.output.contains("confirm:"));
        assert_eq!(result.paused_during_run, vec![true]);
        assert_eq!(result.pause_count, result.resume_count);
    }

    /// A spoken "no" cancels a risky action.
    #[test]
    fn confirm_no_cancels_the_action() {
        let mut frames = utterance_frames();
        frames.extend(utterance_frames());

        let result = run_script(
            frames,
            vec![Some("Reach click Submit"), Some("no")],
            Duration::from_secs(1),
            false,
        );

        assert!(result.intents.is_empty());
        assert!(result.output.contains("cancelled"));
    }

    /// The word "confirm" is a spoken yes.
    #[test]
    fn confirm_word_runs_the_action() {
        let mut frames = utterance_frames();
        frames.extend(utterance_frames());

        let result = run_script(
            frames,
            vec![Some("Reach click Submit"), Some("confirm")],
            Duration::from_secs(1),
            false,
        );

        assert_eq!(
            result.intents,
            vec![Intent::ClickElement {
                target: "Submit".to_owned(),
            }]
        );
    }

    /// The word "cancel" is a spoken no.
    #[test]
    fn cancel_word_cancels_the_action() {
        let mut frames = utterance_frames();
        frames.extend(utterance_frames());

        let result = run_script(
            frames,
            vec![Some("Reach click Submit"), Some("cancel")],
            Duration::from_secs(1),
            false,
        );

        assert!(result.intents.is_empty());
        assert!(result.output.contains("cancelled"));
    }

    /// A confirmation that times out cancels instead of running the action.
    #[test]
    fn confirm_timeout_cancels_the_action() {
        let result = run_script(
            utterance_frames(),
            vec![Some("Reach click Submit")],
            Duration::ZERO,
            false,
        );

        assert!(result.intents.is_empty());
        assert!(result.output.contains(CLARIFY_QUESTION));
    }

    /// An answer outside the rule grammar cancels instead of running the action.
    #[test]
    fn confirm_with_an_unrecognised_answer_cancels_the_action() {
        let mut frames = utterance_frames();
        frames.extend(utterance_frames());

        let result = run_script(
            frames,
            vec![Some("Reach click Submit"), Some("maybe")],
            Duration::from_secs(1),
            false,
        );

        assert!(result.intents.is_empty());
        assert!(result.output.contains(CLARIFY_QUESTION));
    }

    /// Captions print the transcript and the estimated latency.
    #[test]
    fn captions_print_the_transcript_and_latency() {
        let result = run_script(
            utterance_frames(),
            vec![Some("Reach scroll down")],
            Duration::ZERO,
            true,
        );

        assert!(result.output.contains("heard: Reach scroll down"));
        assert!(result.output.contains("latency:"));
        assert!(result.output.contains("endpoint wait:"));
        assert!(result.output.contains("transcription time:"));
        assert!(result.output.contains("parse/route time:"));
        assert!(result.output.contains("execute time:"));
        assert!(result.output.contains(" ms"));
    }

    /// Captions print latency broken down into endpoint wait, transcription, parse/route and execute time.
    #[test]
    fn captions_print_latency_breakdown_metrics() {
        let result = run_script(
            utterance_frames(),
            vec![Some("Reach scroll down")],
            Duration::ZERO,
            true,
        );

        let latency_line = result
            .output
            .lines()
            .find(|line| line.starts_with("latency:"))
            .expect("latency line must be present");

        assert!(latency_line.contains("endpoint wait:"));
        assert!(latency_line.contains("transcription time:"));
        assert!(latency_line.contains("parse/route time:"));
        assert!(latency_line.contains("execute time:"));
    }

    /// Without captions, the transcript is never printed.
    #[test]
    fn without_captions_the_transcript_is_not_printed() {
        let result = run_script(
            utterance_frames(),
            vec![Some("Reach scroll down")],
            Duration::ZERO,
            false,
        );

        assert!(!result.output.contains("Reach"));
        assert!(!result.output.contains("scroll down"));
    }

    /// The parser accepts every listen option and an explicit STT model.
    #[test]
    fn parses_listen_options() {
        let args = vec![
            "--model".to_owned(),
            "models/needle3.cact".to_owned(),
            "--captions".to_owned(),
            "--dry-run".to_owned(),
            "--stt-model".to_owned(),
            "models/ggml-base.en.bin".to_owned(),
        ];
        let parsed = parse_listen_args(&args).expect("the options are valid");

        assert!(parsed.model);
        assert_eq!(parsed.model_path.as_deref(), Some("models/needle3.cact"));
        assert!(parsed.captions);
        assert!(parsed.dry_run);
        assert_eq!(parsed.stt_model.as_deref(), Some("models/ggml-base.en.bin"));
    }

    /// The parser accepts the equals form of the STT model option.
    #[test]
    fn parses_stt_model_equals_form() {
        let args = vec!["--stt-model=models/tiny.en.bin".to_owned()];
        let parsed = parse_listen_args(&args).expect("the option is valid");

        assert_eq!(parsed.stt_model.as_deref(), Some("models/tiny.en.bin"));
    }

    /// A missing STT model path is rejected instead of using the default.
    #[test]
    fn rejects_a_missing_stt_model_path() {
        assert!(parse_listen_args(&["--stt-model".to_owned()]).is_err());
        assert!(parse_listen_args(&["--stt-model=".to_owned()]).is_err());
    }

    /// Unknown options and positional words are rejected.
    #[test]
    fn rejects_unknown_options_and_positional_words() {
        assert!(parse_listen_args(&["--microphone".to_owned()]).is_err());
        assert!(parse_listen_args(&["scroll".to_owned()]).is_err());
    }

    /// Verifies the --no-voice option is parsed.
    #[test]
    fn parses_no_voice_option() {
        let args = parse_listen_args(&["--no-voice".to_owned()]).unwrap();
        assert!(args.no_voice);

        let default_args = parse_listen_args(&[]).unwrap();
        assert!(!default_args.no_voice);
    }

    /// Tests that voice feedback records listening and done phrases during wake and execution.
    #[test]
    fn voice_feedback_records_phrases_during_wake_and_execution() {
        let mut frames = utterance_frames();
        frames.extend(utterance_frames());

        let result = run_script(
            frames,
            vec![Some("Reach"), Some("scroll down")],
            Duration::ZERO,
            false,
        );

        assert_eq!(result.phrases, vec![Phrase::Listening, Phrase::Done]);
    }

    /// Tests that voice feedback records confirmation prompt and cancellation.
    #[test]
    fn voice_feedback_records_confirming_and_cancelled() {
        let mut frames = utterance_frames();
        frames.extend(utterance_frames());

        let result = run_script(
            frames,
            vec![Some("Reach click submit"), Some("no")],
            Duration::from_secs(1),
            false,
        );

        assert_eq!(
            result.phrases,
            vec![
                Phrase::Confirming(ActionLabel::Submit),
                Phrase::Cancelled
            ]
        );
    }

    /// Tests that FeedbackWriter speak explicitly pauses, speaks, and resumes.
    #[test]
    fn feedback_writer_speak_pauses_and_resumes_mic() {
        let control = ScriptedControl::default();
        let speaker = NullSpeaker::new();
        let recorded = speaker.clone();
        let mut output = Vec::new();
        let mut feedback = FeedbackWriter::new(&mut output, Box::new(control.clone()), Box::new(speaker));

        assert_eq!(control.pause_count(), 0);
        assert_eq!(control.resume_count(), 0);

        feedback.speak(Phrase::Listening);

        assert_eq!(control.pause_count(), 1);
        assert_eq!(control.resume_count(), 1);
        assert_eq!(recorded.phrases(), vec![Phrase::Listening]);
    }
}
