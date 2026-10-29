//! Worker-thread loop that consumes transcripts and emits status events.
//!
//! The microphone path will eventually feed this loop. Until `audio` and `stt`
//! are implemented, the app feeds it typed lines from the console. Only the
//! input source changes; the status channel and the pipeline stay the same.
//!
//! The loop owns no desktop state and never logs a transcript, an element name
//! or an executor error. Everything the status window can display arrives as a
//! [`StatusEvent`].

use std::io::{self, BufRead, Read, Write};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::time::{Duration, Instant};

use nlu::IntentModel;
use reach_core::Executor;

use crate::pipeline::{Pipeline, StepOutcome};
use crate::status::{ErrorKind, StatusEvent};

/// How often the worker re-checks the listening window while waiting for input.
///
/// The wake window is five seconds by default. Polling a few times a second
/// lets the status window return to `Asleep` when the window expires, without
/// waking the worker in a tight loop.
const LISTEN_POLL: Duration = Duration::from_millis(100);

/// Why [`run_worker`] returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerEnd {
    /// The input channel closed, so there is nothing more to process.
    InputClosed,
    /// The user typed `exit` or `quit`.
    Exited,
    /// The status receiver was dropped, so there is nowhere to report to.
    StatusClosed,
}

/// Runs the shared voice loop until input ends, the user exits, or the status
/// receiver goes away.
///
/// The worker emits an initial [`StatusEvent::Asleep`], one
/// [`StatusEvent::Heard`] per non-empty line, and a final
/// [`StatusEvent::Stopped`] unless the status channel itself closed.
///
/// # Arguments
/// * `pipeline` - The configured wake, routing and execution pipeline.
/// * `lines` - Transcripts from the microphone or, for now, the console.
/// * `status` - Sink for the status window.
///
/// # Returns
/// Why the loop ended.
pub fn run_worker<M: IntentModel, E: Executor>(
    pipeline: &mut Pipeline<M, E>,
    lines: &Receiver<String>,
    status: &Sender<StatusEvent>,
) -> WorkerEnd {
    if status.send(StatusEvent::Asleep).is_err() {
        return WorkerEnd::StatusClosed;
    }

    loop {
        let line = if pipeline.is_listening(Instant::now()) {
            match lines.recv_timeout(LISTEN_POLL) {
                Ok(line) => line,
                Err(RecvTimeoutError::Timeout) => {
                    // The window may have expired while we waited. Reporting
                    // `Asleep` here keeps the indicator truthful even when the
                    // user says nothing after the wake word.
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

        // A bare wake word leaves the listening window open; the pipeline has
        // already reported `Listening`. When an action ran, it is held on screen
        // instead of immediately returning to asleep.
        let executed_action = matches!(result, Ok(StepOutcome::Handled { executed: true, .. }));
        if !executed_action
            && !pipeline.is_listening(now)
            && status.send(StatusEvent::Asleep).is_err()
        {
            return WorkerEnd::StatusClosed;
        }
    }
}

/// Sends the final `Stopped` event and converts a send failure into
/// [`WorkerEnd::StatusClosed`].
///
/// # Arguments
/// * `status` - Sink for the status window.
/// * `end` - The reason the loop ended before the final event.
///
/// # Returns
/// `end` when `Stopped` was delivered, otherwise [`WorkerEnd::StatusClosed`].
fn finish(status: &Sender<StatusEvent>, end: WorkerEnd) -> WorkerEnd {
    if status.send(StatusEvent::Stopped).is_err() {
        WorkerEnd::StatusClosed
    } else {
        end
    }
}

/// A blocking line reader backed by the worker's input channel.
///
/// The pipeline reads a confirmation answer immediately after reporting a
/// pending action. This adapter lets the same channel serve both the normal
/// transcripts and that answer, without a second input source.
struct ChannelReader<'a> {
    /// The channel carrying transcript lines.
    receiver: &'a Receiver<String>,
    /// The bytes of the current line plus a synthetic newline.
    buffer: Vec<u8>,
    /// How many bytes of `buffer` have already been consumed.
    consumed: usize,
    /// True once the channel has disconnected.
    eof: bool,
}

impl<'a> ChannelReader<'a> {
    /// Wraps `receiver` in a blocking line reader.
    ///
    /// # Arguments
    /// * `receiver` - The channel carrying transcript lines.
    ///
    /// # Returns
    /// An empty reader positioned before the next line.
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
    /// Copies bytes from the current line into `buf`.
    ///
    /// # Arguments
    /// * `buf` - Destination buffer.
    ///
    /// # Returns
    /// The number of bytes copied, or zero at end of input.
    ///
    /// # Errors
    /// Never returns an error; a disconnected channel is reported as end of
    /// input.
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
    /// Blocks until the next line is available or the channel disconnects.
    ///
    /// # Returns
    /// The unconsumed part of the current line, including its synthetic
    /// newline, or an empty slice at end of input.
    ///
    /// # Errors
    /// Never returns an error.
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        if self.consumed == self.buffer.len() && !self.eof {
            match self.receiver.recv() {
                Ok(line) => {
                    self.buffer.clear();
                    self.consumed = 0;
                    self.buffer.extend_from_slice(line.as_bytes());
                    self.buffer.push(b'\n');
                }
                Err(_) => self.eof = true,
            }
        }
        Ok(&self.buffer[self.consumed..])
    }

    /// Marks `amount` bytes of the current line as consumed.
    ///
    /// # Arguments
    /// * `amount` - Number of bytes to discard.
    fn consume(&mut self, amount: usize) {
        self.consumed = (self.consumed + amount).min(self.buffer.len());
    }
}

/// A writer that discards human-readable pipeline output.
///
/// The app reports through [`StatusEvent`], so the pipeline's text is not
/// needed and must not be printed to a console that may not exist in release.
struct NullWriter;

impl Write for NullWriter {
    /// Discards `buf` and reports it as written.
    ///
    /// # Arguments
    /// * `buf` - Bytes to discard.
    ///
    /// # Returns
    /// The length of `buf`.
    ///
    /// # Errors
    /// Never returns an error.
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        Ok(buf.len())
    }

    /// Does nothing; the writer has no buffer.
    ///
    /// # Errors
    /// Never returns an error.
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    use nlu::wake::WakeConfig;
    use reach_core::{Direction, Executor, Intent};

    use crate::pipeline::{DisabledModel, Pipeline, format_intent_action};
    use crate::status::{ErrorKind, StatusEvent};
    use crate::worker::{WorkerEnd, run_worker};

    /// A stand-in executor that records intents and can fail on demand.
    struct FakeExecutor {
        /// When set, every intent returns this error text.
        error: Option<String>,
        /// Intents passed to [`Executor::run`], in order.
        intents: Vec<Intent>,
    }

    impl FakeExecutor {
        /// Returns an executor that succeeds and records intents.
        fn ok() -> Self {
            Self {
                error: None,
                intents: Vec::new(),
            }
        }

        /// Returns an executor that always fails with `message`.
        fn failing(message: &str) -> Self {
            Self {
                error: Some(message.to_owned()),
                intents: Vec::new(),
            }
        }
    }

    impl Executor for FakeExecutor {
        /// Records `intent`, then succeeds or returns the configured error.
        fn run(&mut self, intent: &Intent) -> Result<String, String> {
            self.intents.push(intent.clone());
            match &self.error {
                Some(error) => Err(error.clone()),
                None => Ok("executed".to_owned()),
            }
        }
    }

    /// Runs a finite script through the worker and returns its status stream.
    ///
    /// The input channel is closed after the scripted lines, so the worker
    /// always finishes and the test can collect the events without blocking.
    fn run_script(
        lines: &[&str],
        executor: FakeExecutor,
    ) -> (WorkerEnd, Vec<StatusEvent>, Vec<Intent>) {
        let (line_sender, line_receiver) = mpsc::channel();
        for line in lines {
            line_sender.send((*line).to_owned()).unwrap();
        }
        drop(line_sender);

        let (status_sender, status_receiver) = mpsc::channel();
        let mut pipeline = Pipeline::new(DisabledModel, executor, false);
        let end = run_worker(&mut pipeline, &line_receiver, &status_sender);
        let events = status_receiver.try_iter().collect();
        let intents = pipeline.executor().intents.clone();
        (end, events, intents)
    }

    /// A bare wake word opens the listening window and reports it.
    #[test]
    fn wake_word_emits_listening() {
        let (end, events, intents) = run_script(&["Reach"], FakeExecutor::ok());
        assert_eq!(end, WorkerEnd::InputClosed);
        assert!(intents.is_empty());
        assert_eq!(
            events,
            vec![
                StatusEvent::Asleep,
                StatusEvent::Heard {
                    text: "Reach".to_owned(),
                },
                StatusEvent::Listening,
                StatusEvent::Stopped,
            ]
        );
    }

    /// A compound wake+command utterance emits Listening before Heard and Action.
    #[test]
    fn compound_wake_command_emits_listening_before_heard_and_action() {
        let (end, events, intents) = run_script(&["Reach scroll down"], FakeExecutor::ok());
        let intent = Intent::Scroll {
            direction: Direction::Down,
            amount: None,
        };
        assert_eq!(end, WorkerEnd::InputClosed);
        assert_eq!(intents, vec![intent.clone()]);
        assert_eq!(
            events,
            vec![
                StatusEvent::Asleep,
                StatusEvent::Listening,
                StatusEvent::Heard {
                    text: "Reach scroll down".to_owned(),
                },
                StatusEvent::Action {
                    label: format_intent_action(&intent),
                },
                StatusEvent::Stopped,
            ]
        );
    }

    /// After Action is emitted, Asleep is not emitted immediately in the same worker step.
    #[test]
    fn action_does_not_immediately_emit_asleep_in_same_step() {
        let (line_sender, line_receiver) = mpsc::channel();
        let (status_sender, status_receiver) = mpsc::channel();
        let mut pipeline = Pipeline::new(DisabledModel, FakeExecutor::ok(), false);

        // Send one compound command, but keep channel open to inspect step output
        line_sender.send("Reach scroll down".to_owned()).unwrap();

        let worker_thread =
            std::thread::spawn(move || run_worker(&mut pipeline, &line_receiver, &status_sender));

        std::thread::sleep(Duration::from_millis(150));
        let events: Vec<StatusEvent> = status_receiver.try_iter().collect();

        let action_idx = events
            .iter()
            .position(|e| matches!(e, StatusEvent::Action { .. }));
        assert!(
            action_idx.is_some(),
            "expected Action event, got: {events:?}"
        );
        let has_asleep_after_action = events[action_idx.unwrap()..]
            .iter()
            .any(|e| matches!(e, StatusEvent::Asleep));
        assert!(
            !has_asleep_after_action,
            "Asleep was emitted immediately after Action in the same step: {events:?}"
        );

        drop(line_sender);
        let _ = worker_thread.join();
    }

    /// With an execution delay, Listening and Heard are emitted before the executor runs.
    #[test]
    fn execution_delay_emits_status_before_running() {
        let (line_sender, line_receiver) = mpsc::channel();
        let (status_sender, status_receiver) = mpsc::channel();
        let mut pipeline = Pipeline::new(DisabledModel, FakeExecutor::ok(), false)
            .with_execution_delay(Duration::from_millis(100));

        line_sender.send("Reach scroll down".to_owned()).unwrap();

        let worker_thread =
            std::thread::spawn(move || run_worker(&mut pipeline, &line_receiver, &status_sender));

        // After 25ms (well before the 100ms delay elapses), Listening and Heard must already be emitted,
        // but Action must not yet be emitted.
        std::thread::sleep(Duration::from_millis(25));
        let early_events: Vec<StatusEvent> = status_receiver.try_iter().collect();
        assert!(early_events.contains(&StatusEvent::Listening));
        assert!(
            early_events
                .iter()
                .any(|e| matches!(e, StatusEvent::Heard { .. }))
        );
        assert!(
            !early_events
                .iter()
                .any(|e| matches!(e, StatusEvent::Action { .. }))
        );

        // After the remaining delay, Action arrives.
        std::thread::sleep(Duration::from_millis(125));
        let later_events: Vec<StatusEvent> = status_receiver.try_iter().collect();
        assert!(
            later_events
                .iter()
                .any(|e| matches!(e, StatusEvent::Action { .. }))
        );

        drop(line_sender);
        let _ = worker_thread.join();
    }

    /// Speech without the wake word returns the window to asleep.
    #[test]
    fn ignored_speech_emits_asleep() {
        let (_, events, intents) = run_script(&["scroll down"], FakeExecutor::ok());
        assert!(intents.is_empty());
        assert_eq!(
            events,
            vec![
                StatusEvent::Asleep,
                StatusEvent::Heard {
                    text: "scroll down".to_owned(),
                },
                StatusEvent::Asleep,
                StatusEvent::Stopped,
            ]
        );
    }

    /// An unmatched command asks the user to rephrase instead of guessing.
    #[test]
    fn unmatched_command_emits_not_understood() {
        let (_, events, intents) = run_script(&["Reach tell me a joke"], FakeExecutor::ok());
        assert!(intents.is_empty());
        assert_eq!(
            events,
            vec![
                StatusEvent::Asleep,
                StatusEvent::Listening,
                StatusEvent::Heard {
                    text: "Reach tell me a joke".to_owned(),
                },
                StatusEvent::NotUnderstood,
                StatusEvent::Asleep,
                StatusEvent::Stopped,
            ]
        );
    }

    /// A risky command announces confirmation, then the action on yes.
    #[test]
    fn confirmation_yes_emits_confirming_then_action() {
        let (_, events, intents) = run_script(&["Reach click Submit", "yes"], FakeExecutor::ok());
        let intent = Intent::ClickElement {
            target: "Submit".to_owned(),
        };
        let label = format_intent_action(&intent);
        assert_eq!(intents, vec![intent]);
        assert_eq!(
            events,
            vec![
                StatusEvent::Asleep,
                StatusEvent::Listening,
                StatusEvent::Heard {
                    text: "Reach click Submit".to_owned(),
                },
                StatusEvent::Confirming {
                    label: label.clone()
                },
                StatusEvent::Action { label },
                StatusEvent::Stopped,
            ]
        );
    }

    /// A denied confirmation never reaches the executor.
    #[test]
    fn confirmation_no_emits_asleep_without_running() {
        let (_, events, intents) = run_script(&["Reach click Submit", "no"], FakeExecutor::ok());
        let label = format_intent_action(&Intent::ClickElement {
            target: "Submit".to_owned(),
        });
        assert!(intents.is_empty());
        assert_eq!(
            events,
            vec![
                StatusEvent::Asleep,
                StatusEvent::Listening,
                StatusEvent::Heard {
                    text: "Reach click Submit".to_owned(),
                },
                StatusEvent::Confirming { label },
                StatusEvent::Asleep,
                StatusEvent::Stopped,
            ]
        );
    }

    /// Executor failures expose a kind only, never the executor's text.
    #[test]
    fn executor_error_emits_only_the_error_kind() {
        let (_, events, _) = run_script(
            &["Reach scroll down"],
            FakeExecutor::failing("private-button-name"),
        );
        assert!(events.contains(&StatusEvent::Error {
            kind: ErrorKind::Execution,
        }));
        let debug = format!("{events:?}");
        assert!(!debug.contains("private-button-name"));
    }

    /// An expired listening window returns the indicator to asleep.
    #[test]
    fn listening_window_timeout_emits_asleep() {
        let (line_sender, line_receiver) = mpsc::channel();
        line_sender.send("Reach".to_owned()).unwrap();
        // Keep the channel open long enough for the worker's poll to observe
        // the expired window, then close it so the test can finish.
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(1000));
            drop(line_sender);
        });

        let (status_sender, status_receiver) = mpsc::channel();
        let config = WakeConfig {
            window: Duration::from_millis(50),
            ..WakeConfig::default()
        };
        let mut pipeline =
            Pipeline::with_wake_config(DisabledModel, FakeExecutor::ok(), false, config);
        let end = run_worker(&mut pipeline, &line_receiver, &status_sender);
        assert_eq!(end, WorkerEnd::InputClosed);
        let events: Vec<StatusEvent> = status_receiver.try_iter().collect();
        assert_eq!(
            events,
            vec![
                StatusEvent::Asleep,
                StatusEvent::Heard {
                    text: "Reach".to_owned(),
                },
                StatusEvent::Listening,
                StatusEvent::Asleep,
                StatusEvent::Stopped,
            ]
        );
    }

    /// The exit command stops the loop without treating "exit" as speech.
    #[test]
    fn exit_emits_stopped_without_hearing() {
        let (end, events, intents) = run_script(&["exit"], FakeExecutor::ok());
        assert_eq!(end, WorkerEnd::Exited);
        assert!(intents.is_empty());
        assert_eq!(events, vec![StatusEvent::Asleep, StatusEvent::Stopped]);
    }

    /// A closed status channel stops the worker immediately.
    #[test]
    fn closed_status_channel_stops_the_worker() {
        let (line_sender, line_receiver) = mpsc::channel();
        line_sender.send("Reach".to_owned()).unwrap();
        drop(line_sender);
        let (status_sender, status_receiver) = mpsc::channel();
        drop(status_receiver);

        let mut pipeline = Pipeline::new(DisabledModel, FakeExecutor::ok(), false);
        assert_eq!(
            run_worker(&mut pipeline, &line_receiver, &status_sender),
            WorkerEnd::StatusClosed
        );
    }
}
