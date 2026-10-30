//! Worker-thread loop that consumes transcripts and emits status events.
//!
//! Feeds the shared pipeline from either microphone-driven [`InputSource::Voice`]
//! or channel-driven [`InputSource::Text`]. The loop emits [`StatusEvent`] values
//! to keep the status HUD updated in real time.

use std::io::{self, BufRead, Read, Write};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::time::{Duration, Instant};

use nlu::IntentModel;
use reach_core::Executor;

use crate::pipeline::{Pipeline, StepOutcome};
use crate::status::{ErrorKind, StatusEvent};
use crate::voice::VoiceInput;

/// How often the worker re-checks the listening window while waiting for input.
const LISTEN_POLL: Duration = Duration::from_millis(100);

/// Why a worker loop ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerEnd {
    /// The input source closed, so there is nothing more to process.
    InputClosed,
    /// The user exited or triggered an emergency stop.
    Exited,
    /// The status receiver was dropped.
    StatusClosed,
}

/// The source of utterances for the shared runtime worker.
pub enum InputSource {
    /// Microphone-driven voice input.
    Voice(Box<VoiceInput>),
    /// Channel-driven text input (console stdin / tests).
    Text(Receiver<String>),
}

/// Runs the worker loop on an explicit [`InputSource`].
///
/// # Arguments
/// * `pipeline` - The configured pipeline.
/// * `input` - The voice or text input source.
/// * `status` - Channel to deliver status events to.
///
/// # Returns
/// Reason why the worker stopped.
pub fn run_worker_source<M: IntentModel + Send + 'static, E: Executor>(
    pipeline: &mut Pipeline<M, E>,
    input: &mut InputSource,
    status: &Sender<StatusEvent>,
) -> WorkerEnd {
    match input {
        InputSource::Text(lines) => run_text_worker(pipeline, lines, status),
        InputSource::Voice(voice) => run_voice_worker(pipeline, voice, status),
    }
}

/// Runs the worker on channel-delivered text transcripts (legacy / testing).
///
/// # Arguments
/// * `pipeline` - The configured pipeline.
/// * `lines` - Receiver channel carrying transcript strings.
/// * `status` - Channel to deliver status events to.
///
/// # Returns
/// Reason why the worker stopped.
pub fn run_worker<M: IntentModel + Send + 'static, E: Executor>(
    pipeline: &mut Pipeline<M, E>,
    lines: &Receiver<String>,
    status: &Sender<StatusEvent>,
) -> WorkerEnd {
    run_text_worker(pipeline, lines, status)
}

/// Runs the voice-driven worker loop using microphone input.
///
/// # Arguments
/// * `pipeline` - Configured pipeline.
/// * `voice` - Microphone voice input.
/// * `status` - Status event sink.
///
/// # Returns
/// Why the worker loop terminated.
pub fn run_voice_worker<M: IntentModel + Send + 'static, E: Executor>(
    pipeline: &mut Pipeline<M, E>,
    voice: &mut VoiceInput,
    status: &Sender<StatusEvent>,
) -> WorkerEnd {
    if status.send(StatusEvent::Asleep).is_err() {
        return WorkerEnd::StatusClosed;
    }

    loop {
        // Check emergency stop flag between steps
        if pipeline.is_stopped() {
            return finish(status, WorkerEnd::Exited);
        }

        let timeout = if pipeline.is_listening(Instant::now()) {
            Some(LISTEN_POLL)
        } else {
            None
        };

        let utterance_opt = match voice.next_utterance(timeout) {
            Ok(Some(u)) => Some(u),
            Ok(None) => {
                if voice.is_closed() {
                    return finish(status, WorkerEnd::InputClosed);
                }
                if !pipeline.is_listening(Instant::now())
                    && status.send(StatusEvent::Asleep).is_err()
                {
                    return WorkerEnd::StatusClosed;
                }
                continue;
            }
            Err(_) => {
                let _ = status.send(StatusEvent::Error {
                    kind: ErrorKind::Io,
                });
                return finish(status, WorkerEnd::InputClosed);
            }
        };

        let Some(input) = utterance_opt else {
            continue;
        };

        voice.mark_processing_started();
        let now = input.speech_ended;
        let mut status_closed = false;
        let mut writer = NullWriter;

        let result =
            pipeline.process_observed(&input.heard.text, now, voice, &mut writer, &mut |event| {
                if status.send(event).is_err() {
                    status_closed = true;
                }
            });

        voice.mark_processing_ended();

        if status_closed {
            return WorkerEnd::StatusClosed;
        }

        if result.is_err() {
            if status
                .send(StatusEvent::Error {
                    kind: ErrorKind::Io,
                })
                .is_err()
            {
                return WorkerEnd::StatusClosed;
            }
            return finish(status, WorkerEnd::InputClosed);
        }

        let executed_action = matches!(result, Ok(StepOutcome::Handled { executed: true, .. }));
        if !executed_action
            && !pipeline.is_listening(Instant::now())
            && status.send(StatusEvent::Asleep).is_err()
        {
            return WorkerEnd::StatusClosed;
        }
    }
}

/// Runs the text-driven worker loop using console/channel input.
///
/// # Arguments
/// * `pipeline` - Configured pipeline.
/// * `lines` - Channel delivering typed text lines.
/// * `status` - Status event sink.
///
/// # Returns
/// Why the worker loop terminated.
pub fn run_text_worker<M: IntentModel + Send + 'static, E: Executor>(
    pipeline: &mut Pipeline<M, E>,
    lines: &Receiver<String>,
    status: &Sender<StatusEvent>,
) -> WorkerEnd {
    if status.send(StatusEvent::Asleep).is_err() {
        return WorkerEnd::StatusClosed;
    }

    loop {
        if pipeline.is_stopped() {
            return finish(status, WorkerEnd::Exited);
        }

        let line = if pipeline.is_listening(Instant::now()) {
            match lines.recv_timeout(LISTEN_POLL) {
                Ok(line) => line,
                Err(RecvTimeoutError::Timeout) => {
                    if !pipeline.is_listening(Instant::now())
                        && status.send(StatusEvent::Asleep).is_err()
                    {
                        return WorkerEnd::StatusClosed;
                    }
                    continue;
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return finish(status, WorkerEnd::InputClosed);
                }
            }
        } else {
            match lines.recv() {
                Ok(line) => line,
                Err(_) => return finish(status, WorkerEnd::InputClosed),
            }
        };
        let utterance = line.trim();
        if utterance.is_empty() {
            continue;
        }
        if utterance.eq_ignore_ascii_case("exit") || utterance.eq_ignore_ascii_case("quit") {
            return finish(status, WorkerEnd::Exited);
        }

        let now = Instant::now();
        let mut reader = ChannelReader::new(lines);
        let mut writer = NullWriter;
        let mut status_closed = false;
        let result =
            pipeline.process_observed(utterance, now, &mut reader, &mut writer, &mut |event| {
                if status.send(event).is_err() {
                    status_closed = true;
                }
            });

        if status_closed {
            return WorkerEnd::StatusClosed;
        }
        if result.is_err() {
            if status
                .send(StatusEvent::Error {
                    kind: ErrorKind::Io,
                })
                .is_err()
            {
                return WorkerEnd::StatusClosed;
            }
            return finish(status, WorkerEnd::InputClosed);
        }

        let executed_action = matches!(result, Ok(StepOutcome::Handled { executed: true, .. }));
        if !executed_action
            && !pipeline.is_listening(now)
            && status.send(StatusEvent::Asleep).is_err()
        {
            return WorkerEnd::StatusClosed;
        }
    }
}

/// Sends the final `Stopped` event and converts a send failure into [`WorkerEnd::StatusClosed`].
fn finish(status: &Sender<StatusEvent>, end: WorkerEnd) -> WorkerEnd {
    if status.send(StatusEvent::Stopped).is_err() {
        WorkerEnd::StatusClosed
    } else {
        end
    }
}

/// A blocking line reader backed by the worker's input channel.
struct ChannelReader<'a> {
    /// Channel delivering typed text lines from the input loop.
    receiver: &'a Receiver<String>,
    /// In-memory buffer holding the current line being consumed.
    buffer: Vec<u8>,
    /// Number of bytes already consumed from `buffer`.
    consumed: usize,
    /// Indicates whether the underlying channel was disconnected.
    eof: bool,
}

impl<'a> ChannelReader<'a> {
    /// Wraps `receiver` in a blocking line reader.
    fn new(receiver: &'a Receiver<String>) -> Self {
        Self {
            receiver,
            buffer: Vec::new(),
            consumed: 0,
            eof: false,
        }
    }
}

impl Read for ChannelReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let amount = {
            let available = self.fill_buf()?;
            let amount = available.len().min(buf.len());
            buf[..amount].copy_from_slice(&available[..amount]);
            amount
        };
        self.consume(amount);
        Ok(amount)
    }
}

impl BufRead for ChannelReader<'_> {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        if self.consumed < self.buffer.len() {
            return Ok(&self.buffer[self.consumed..]);
        }
        if self.eof {
            return Ok(&[]);
        }
        self.buffer.clear();
        self.consumed = 0;
        match self.receiver.recv() {
            Ok(line) => {
                self.buffer.extend_from_slice(line.as_bytes());
                self.buffer.push(b'\n');
                Ok(&self.buffer)
            }
            Err(_) => {
                self.eof = true;
                Ok(&[])
            }
        }
    }

    fn consume(&mut self, amount: usize) {
        self.consumed = self.consumed.saturating_add(amount).min(self.buffer.len());
    }
}

/// A [`Write`] implementation that discards all bytes.
pub struct NullWriter;

impl Write for NullWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;
    use std::time::Duration;

    use audio::{EndpointConfig, FRAME_SAMPLES};
    use reach_core::{Executor, Heard, Intent};

    use super::*;
    use crate::pipeline::{DisabledModel, Pipeline};
    use crate::status::StatusEvent;
    use crate::voice::{FramePoll, FrameSource, SpeechToText, VoiceInput};

    /// Recording executor that tracks executed intents.
    #[derive(Default)]
    struct RecordingExecutor {
        intents: Vec<Intent>,
    }

    impl Executor for RecordingExecutor {
        fn run(&mut self, intent: &Intent) -> Result<String, String> {
            self.intents.push(intent.clone());
            Ok("ok".to_owned())
        }
    }

    /// Scripted frame source for testing voice input.
    struct ScriptedFrames {
        events: VecDeque<FramePoll>,
    }

    impl FrameSource for ScriptedFrames {
        fn next_frame(&mut self, _timeout: Duration) -> FramePoll {
            self.events.pop_front().unwrap_or(FramePoll::Closed)
        }
    }

    /// Scripted transcriber for testing voice input.
    struct ScriptedTranscriber {
        replies: VecDeque<Option<String>>,
    }

    impl SpeechToText for ScriptedTranscriber {
        fn transcribe(&mut self, _samples: &[f32]) -> Result<Option<Heard>, String> {
            let reply = self.replies.pop_front().flatten();
            Ok(reply.map(|text| Heard {
                text,
                confidence: Some(0.95),
            }))
        }
    }

    /// Builds a simulated voice utterance with voiced audio frames.
    fn make_voice_input(transcripts: Vec<&str>) -> VoiceInput {
        let mut events = Vec::new();
        for _ in 0..transcripts.len() {
            for _ in 0..30 {
                events.push(FramePoll::Frame(vec![0.1; FRAME_SAMPLES]));
            }
            for _ in 0..5 {
                events.push(FramePoll::Frame(vec![0.0; FRAME_SAMPLES]));
            }
        }
        events.push(FramePoll::Closed);

        let frames = Box::new(ScriptedFrames {
            events: events.into(),
        });
        let stt = Box::new(ScriptedTranscriber {
            replies: transcripts
                .into_iter()
                .map(|s| Some(s.to_owned()))
                .collect(),
        });

        VoiceInput::new(
            frames,
            stt,
            EndpointConfig {
                min_speech_ms: 100,
                end_silence_ms: 100,
                pre_roll_ms: 20,
                max_utterance_ms: 5_000,
                ..EndpointConfig::default()
            },
            Duration::from_millis(100),
        )
        .unwrap()
    }

    #[test]
    fn voice_wake_only_opens_window_and_emits_listening() {
        let voice = make_voice_input(vec!["Reach"]);
        let mut pipeline = Pipeline::new(DisabledModel, RecordingExecutor::default(), false);
        let (status_tx, status_rx) = mpsc::channel();

        let mut input = InputSource::Voice(Box::new(voice));
        let _ = run_worker_source(&mut pipeline, &mut input, &status_tx);

        let events: Vec<StatusEvent> = status_rx.try_iter().collect();
        assert!(events.contains(&StatusEvent::Asleep));
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StatusEvent::Heard { text } if text == "Reach"))
        );
        assert!(events.contains(&StatusEvent::Listening));
    }

    #[test]
    fn voice_command_without_wake_word_is_ignored() {
        let voice = make_voice_input(vec!["scroll down"]);
        let mut pipeline = Pipeline::new(DisabledModel, RecordingExecutor::default(), false);
        let (status_tx, status_rx) = mpsc::channel();

        let mut input = InputSource::Voice(Box::new(voice));
        let _ = run_worker_source(&mut pipeline, &mut input, &status_tx);

        let events: Vec<StatusEvent> = status_rx.try_iter().collect();
        assert!(events.contains(&StatusEvent::Asleep));
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StatusEvent::Heard { text } if text == "scroll down"))
        );
        // Action must not run when wake word is absent
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, StatusEvent::Action { .. }))
        );
    }

    #[test]
    fn voice_confirm_yes_runs_action() {
        let voice = make_voice_input(vec!["Reach click submit", "yes"]);
        let mut pipeline = Pipeline::new(DisabledModel, RecordingExecutor::default(), false);
        let (status_tx, status_rx) = mpsc::channel();

        let mut input = InputSource::Voice(Box::new(voice));
        let _ = run_worker_source(&mut pipeline, &mut input, &status_tx);

        let events: Vec<StatusEvent> = status_rx.try_iter().collect();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StatusEvent::Confirming { .. }))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StatusEvent::Action { .. }))
        );
    }

    #[test]
    /// Verifies that speaking "no" cancels the pending action.
    fn voice_confirm_no_cancels_action() {
        let voice = make_voice_input(vec!["Reach click submit", "no"]);
        let mut pipeline = Pipeline::new(DisabledModel, RecordingExecutor::default(), false);
        let (status_tx, status_rx) = mpsc::channel();

        let mut input = InputSource::Voice(Box::new(voice));
        let _ = run_worker_source(&mut pipeline, &mut input, &status_tx);

        let events: Vec<StatusEvent> = status_rx.try_iter().collect();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StatusEvent::Confirming { .. }))
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, StatusEvent::Action { .. }))
        );
        assert_eq!(pipeline.executor().intents.len(), 0);
    }

    #[test]
    /// Verifies that speaking "Reach, yes" confirms the pending action.
    fn voice_confirm_reach_yes_runs_action() {
        let voice = make_voice_input(vec!["Reach click submit", "Reach, yes"]);
        let mut pipeline = Pipeline::new(DisabledModel, RecordingExecutor::default(), false);
        let (status_tx, status_rx) = mpsc::channel();

        let mut input = InputSource::Voice(Box::new(voice));
        let _ = run_worker_source(&mut pipeline, &mut input, &status_tx);

        let events: Vec<StatusEvent> = status_rx.try_iter().collect();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StatusEvent::Confirming { .. }))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StatusEvent::Action { .. }))
        );
        assert_eq!(pipeline.executor().intents.len(), 1);
    }

    #[test]
    /// Verifies that speaking "reach yes" confirms the pending action.
    fn voice_confirm_reach_yes_without_comma_runs_action() {
        let voice = make_voice_input(vec!["Reach click submit", "reach yes"]);
        let mut pipeline = Pipeline::new(DisabledModel, RecordingExecutor::default(), false);
        let (status_tx, status_rx) = mpsc::channel();

        let mut input = InputSource::Voice(Box::new(voice));
        let _ = run_worker_source(&mut pipeline, &mut input, &status_tx);

        let events: Vec<StatusEvent> = status_rx.try_iter().collect();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StatusEvent::Action { .. }))
        );
        assert_eq!(pipeline.executor().intents.len(), 1);
    }

    #[test]
    /// Verifies that speaking "rich yes" with the acoustic wake alias confirms the pending action.
    fn voice_confirm_rich_yes_runs_action() {
        let voice = make_voice_input(vec!["Reach click submit", "rich yes"]);
        let mut pipeline = Pipeline::new(DisabledModel, RecordingExecutor::default(), false);
        let (status_tx, status_rx) = mpsc::channel();

        let mut input = InputSource::Voice(Box::new(voice));
        let _ = run_worker_source(&mut pipeline, &mut input, &status_tx);

        let events: Vec<StatusEvent> = status_rx.try_iter().collect();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StatusEvent::Action { .. }))
        );
        assert_eq!(pipeline.executor().intents.len(), 1);
    }

    #[test]
    /// Verifies that speaking "Reach, no" cancels the pending action.
    fn voice_confirm_reach_no_cancels_action() {
        let voice = make_voice_input(vec!["Reach click submit", "Reach, no"]);
        let mut pipeline = Pipeline::new(DisabledModel, RecordingExecutor::default(), false);
        let (status_tx, status_rx) = mpsc::channel();

        let mut input = InputSource::Voice(Box::new(voice));
        let _ = run_worker_source(&mut pipeline, &mut input, &status_tx);

        let events: Vec<StatusEvent> = status_rx.try_iter().collect();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StatusEvent::Confirming { .. }))
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, StatusEvent::Action { .. }))
        );
        assert_eq!(pipeline.executor().intents.len(), 0);
    }

    #[test]
    fn text_worker_emits_heard_and_runs_commands() {
        let (lines_tx, lines_rx) = mpsc::channel();
        let (status_tx, status_rx) = mpsc::channel();
        let mut pipeline = Pipeline::new(DisabledModel, RecordingExecutor::default(), false);

        lines_tx.send("Reach scroll down".to_owned()).unwrap();
        lines_tx.send("exit".to_owned()).unwrap();

        let mut input = InputSource::Text(lines_rx);
        let end = run_worker_source(&mut pipeline, &mut input, &status_tx);
        assert_eq!(end, WorkerEnd::Exited);

        let events: Vec<StatusEvent> = status_rx.try_iter().collect();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StatusEvent::Heard { text } if text == "Reach scroll down"))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StatusEvent::Action { label } if *label == "scroll"))
        );
        assert_eq!(pipeline.executor().intents.len(), 1);
    }

    #[test]
    fn unexecutable_command_emits_not_understood_and_recovers_immediately() {
        let (lines_tx, lines_rx) = mpsc::channel();
        let (status_tx, status_rx) = mpsc::channel();
        let mut pipeline = Pipeline::new(DisabledModel, RecordingExecutor::default(), false);

        lines_tx.send("reach what's my name".to_owned()).unwrap();
        lines_tx.send("reach scroll down".to_owned()).unwrap();
        lines_tx.send("exit".to_owned()).unwrap();

        let mut input = InputSource::Text(lines_rx);
        let end = run_worker_source(&mut pipeline, &mut input, &status_tx);
        assert_eq!(end, WorkerEnd::Exited);

        let events: Vec<StatusEvent> = status_rx.try_iter().collect();
        assert!(events.contains(&StatusEvent::NotUnderstood));
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StatusEvent::Action { label } if *label == "scroll"))
        );
        assert_eq!(pipeline.executor().intents.len(), 1);
    }

    #[test]
    fn emergency_stop_halts_loop_between_steps() {
        let (lines_tx, lines_rx) = mpsc::channel();
        let (status_tx, _status_rx) = mpsc::channel();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let mut pipeline = Pipeline::new(DisabledModel, RecordingExecutor::default(), false)
            .with_stop_flag(Arc::clone(&stop_flag));

        // Trigger stop flag
        stop_flag.store(true, Ordering::SeqCst);
        lines_tx.send("Reach scroll down".to_owned()).unwrap();

        let mut input = InputSource::Text(lines_rx);
        let end = run_worker_source(&mut pipeline, &mut input, &status_tx);
        assert_eq!(end, WorkerEnd::Exited);
        assert_eq!(pipeline.executor().intents.len(), 0);
    }
}
