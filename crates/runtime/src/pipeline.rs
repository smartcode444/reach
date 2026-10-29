//! Shared wake, routing, confirmation and execution pipeline.
//!
//! Both the CLI and the desktop app call this pipeline. It owns the production
//! order: the wake session first, then [`nlu::route_utterance`], then one of the
//! [`Route`] outcomes. It never re-implements routing and never prints the
//! transcript to a log.
//!
//! The pipeline can report what it is doing through a status sink. The CLI
//! passes a no-op sink and keeps its text output; the app forwards the events
//! to its status window.

use std::io::{self, BufRead, Write};
use std::path::Path;
use std::time::Instant;

use nlu::tools::ToolSpec;
use nlu::wake::{GateResult, WakeConfig, WakeSession};
use nlu::{
    CLARIFY_QUESTION, IntentModel, ModelError, ModelReply, NeedleConfig, NeedleIntentModel, Route,
    route_utterance,
};
use reach_core::{Direction, Executor, Intent, Key, Mode, ZoomDir};

use crate::status::{ErrorKind, StatusEvent};

/// A disabled model that abstains on every utterance.
///
/// Used when no intent model is active; rule misses produce [`Route::Clarify`].
#[derive(Debug, Default, Clone, Copy)]
pub struct DisabledModel;

impl IntentModel for DisabledModel {
    /// Returns an empty call list, signifying abstention.
    ///
    /// # Arguments
    /// * `_utterance` - The recognised text (unused).
    /// * `_tools` - The declared tool specifications (unused).
    ///
    /// # Returns
    /// An empty [`ModelReply`].
    ///
    /// # Errors
    /// Never returns an error.
    fn infer(&mut self, _utterance: &str, _tools: &[ToolSpec]) -> Result<ModelReply, ModelError> {
        Ok(ModelReply {
            calls: Vec::new(),
            confidence: None,
            truncated: false,
        })
    }
}

/// The intent model selected for a run.
pub enum PipelineModel {
    /// No model was requested; every rules miss abstains.
    Disabled(DisabledModel),
    /// The real Needle model, loaded once at startup.
    Needle(Box<NeedleIntentModel>),
}

impl PipelineModel {
    /// Returns the offline model that abstains on every utterance.
    ///
    /// # Returns
    /// [`PipelineModel::Disabled`] with the shared [`DisabledModel`] value.
    pub fn disabled() -> Self {
        Self::Disabled(DisabledModel)
    }

    /// Loads Needle from `path` with the default configuration.
    ///
    /// # Arguments
    /// * `path` - The `.cact` container, normally `models/needle3.cact`.
    ///
    /// # Returns
    /// A model ready to answer [`IntentModel::infer`] calls.
    ///
    /// # Errors
    /// Returns [`ModelError`] when the container or its tokenizer cannot be
    /// loaded. The caller must not include the error text in a status event.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ModelError> {
        let model = NeedleIntentModel::load(path, NeedleConfig::default())?;
        Ok(Self::Needle(Box::new(model)))
    }
}

impl IntentModel for PipelineModel {
    /// Runs the selected model, or abstains when the model is off.
    ///
    /// # Arguments
    /// * `utterance` - The command text after the wake gate.
    /// * `tools` - The closed tool list the model may call.
    ///
    /// # Returns
    /// The model reply, or an empty call list when the model is off.
    ///
    /// # Errors
    /// Returns the real model's generation or payload error.
    fn infer(&mut self, utterance: &str, tools: &[ToolSpec]) -> Result<ModelReply, ModelError> {
        match self {
            Self::Disabled(model) => model.infer(utterance, tools),
            Self::Needle(model) => model.infer(utterance, tools),
        }
    }
}

/// The streams the pipeline reads answers from and writes results to.
pub struct PipelineIo<'a> {
    /// Reads confirmation answers.
    pub input: &'a mut dyn BufRead,
    /// Receives user-facing results.
    pub output: &'a mut dyn Write,
}

impl<'a> PipelineIo<'a> {
    /// Creates a pipeline I/O pair from borrowed streams.
    ///
    /// # Arguments
    /// * `input` - Stream to read confirmation answers from.
    /// * `output` - Stream to write feedback and execution results to.
    ///
    /// # Returns
    /// An initialized [`PipelineIo`].
    pub fn new(input: &'a mut dyn BufRead, output: &'a mut dyn Write) -> Self {
        Self { input, output }
    }
}

/// The outcome of processing a single utterance through the pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum StepOutcome {
    /// The utterance was ignored because the wake word was missing.
    Ignored,
    /// The utterance was only the wake word; the listening window opened.
    WakeOnly,
    /// The utterance was classified and routed as a command.
    Handled {
        /// The routing decision made.
        route: Route,
        /// The chosen intent, if one was selected.
        chosen_intent: Option<Intent>,
        /// Whether the executor was invoked.
        executed: bool,
    },
}

/// One typed-text pipeline with its wake session, model and executor.
pub struct Pipeline<M, E> {
    /// The wake gate and listening window shared by every line.
    session: WakeSession,
    /// The model used only when the rule grammar misses.
    model: M,
    /// The executor used only for `Run` and confirmed `Confirm` routes.
    executor: E,
    /// When true, routes are returned without executing.
    dry_run: bool,
}

impl<M, E> Pipeline<M, E> {
    /// Creates a pipeline with the default wake configuration.
    ///
    /// # Arguments
    /// * `model` - The intent model to consult on rule misses.
    /// * `executor` - The executor carrying out accepted intents.
    /// * `dry_run` - If true, routes are returned without executing.
    ///
    /// # Returns
    /// An initialized [`Pipeline`].
    pub fn new(model: M, executor: E, dry_run: bool) -> Self {
        Self::with_wake_config(model, executor, dry_run, WakeConfig::default())
    }

    /// Creates a pipeline with an explicit wake configuration.
    ///
    /// Tests use a short listening window so the worker's timeout path runs
    /// without waiting five seconds.
    ///
    /// # Arguments
    /// * `model` - The intent model to consult on rule misses.
    /// * `executor` - The executor carrying out accepted intents.
    /// * `dry_run` - If true, routes are returned without executing.
    /// * `config` - The wake word, aliases and listening-window length.
    ///
    /// # Returns
    /// An initialized [`Pipeline`].
    pub fn with_wake_config(model: M, executor: E, dry_run: bool, config: WakeConfig) -> Self {
        Self {
            session: WakeSession::new(config),
            model,
            executor,
            dry_run,
        }
    }

    /// Returns the executor so tests can inspect what ran.
    ///
    /// # Returns
    /// Reference to `E`.
    pub fn executor(&self) -> &E {
        &self.executor
    }

    /// Returns a mutable reference to the inner executor.
    ///
    /// # Returns
    /// Mutable reference to `E`.
    pub fn executor_mut(&mut self) -> &mut E {
        &mut self.executor
    }

    /// Reports whether routes are returned without executing.
    ///
    /// # Returns
    /// The `dry_run` flag passed to [`Pipeline::new`].
    pub fn dry_run(&self) -> bool {
        self.dry_run
    }
}

impl<M: IntentModel, E: Executor> Pipeline<M, E> {
    /// Processes one transcript and writes the user-facing result.
    ///
    /// # Arguments
    /// * `transcript` - One line of text, as if it came from the microphone.
    /// * `now` - The current time, injected so the listening window is
    ///   deterministic in tests.
    /// * `io` - The streams used for confirmation answers and results.
    ///
    /// # Returns
    /// The structured [`StepOutcome`] for the line.
    ///
    /// # Errors
    /// Returns an I/O error when reading an answer or writing a result fails.
    pub fn process(
        &mut self,
        transcript: &str,
        now: Instant,
        io: &mut PipelineIo<'_>,
    ) -> io::Result<StepOutcome> {
        self.process_with(transcript, now, io.input, io.output, &mut |_| {})
    }

    /// Processes an utterance and returns a structured [`StepOutcome`].
    ///
    /// # Arguments
    /// * `utterance` - Spoken or typed line of text.
    /// * `now` - Timestamp for listening window evaluation.
    /// * `reader` - Stream to read confirmation answers from.
    /// * `writer` - Stream to write output to.
    ///
    /// # Returns
    /// The [`StepOutcome`] indicating how the line was handled.
    ///
    /// # Errors
    /// Returns an I/O error if reading or writing fails.
    pub fn process_utterance<R: BufRead, W: Write>(
        &mut self,
        utterance: &str,
        now: Instant,
        reader: &mut R,
        writer: &mut W,
    ) -> io::Result<StepOutcome> {
        self.process_with(utterance, now, reader, writer, &mut |_| {})
    }

    /// Processes one utterance and reports each step through `status`.
    ///
    /// The status sink is called before a confirmation answer is read, so the
    /// window can show the pending action while the user is deciding.
    ///
    /// # Arguments
    /// * `utterance` - The recognised text, including any wake word.
    /// * `now` - Timestamp for listening window evaluation.
    /// * `reader` - Stream to read a confirmation answer from.
    /// * `writer` - Stream to write human-readable text to. The app passes a
    ///   silent writer; the CLI passes its stdout.
    /// * `status` - Sink for [`StatusEvent`] values.
    ///
    /// # Returns
    /// The [`StepOutcome`] indicating how the line was handled.
    ///
    /// # Errors
    /// Returns an I/O error if reading or writing fails. Status sinks cannot
    /// fail; a front end that loses its receiver records that itself.
    pub fn process_observed<R: BufRead, W: Write>(
        &mut self,
        utterance: &str,
        now: Instant,
        reader: &mut R,
        writer: &mut W,
        status: &mut dyn FnMut(StatusEvent),
    ) -> io::Result<StepOutcome> {
        self.process_with(utterance, now, reader, writer, status)
    }

    /// Reports whether the listening window is open at `now`.
    ///
    /// # Arguments
    /// * `now` - The current time, supplied by the caller.
    ///
    /// # Returns
    /// `true` while the next utterance can omit the wake word.
    pub fn is_listening(&self, now: Instant) -> bool {
        self.session.is_open(now)
    }

    /// Shared implementation for the text and status front ends.
    ///
    /// # Arguments
    /// * `transcript` - The recognised text.
    /// * `now` - Timestamp for listening window evaluation.
    /// * `input` - Stream to read a confirmation answer from.
    /// * `output` - Stream to write human-readable text to.
    /// * `status` - Sink for status events.
    ///
    /// # Returns
    /// The outcome of the line.
    ///
    /// # Errors
    /// Returns an I/O error if reading or writing fails.
    fn process_with(
        &mut self,
        transcript: &str,
        now: Instant,
        input: &mut dyn BufRead,
        output: &mut dyn Write,
        status: &mut dyn FnMut(StatusEvent),
    ) -> io::Result<StepOutcome> {
        match self.session.evaluate(transcript, now) {
            GateResult::Ignored => {
                writeln!(output, "ignored")?;
                Ok(StepOutcome::Ignored)
            }
            GateResult::WakeOnly => {
                writeln!(output, "listening")?;
                status(StatusEvent::Listening);
                Ok(StepOutcome::WakeOnly)
            }
            GateResult::Command(command) => {
                let route = route_utterance(&command, &mut self.model);
                let chosen_intent = match &route {
                    Route::Run(intent) | Route::Confirm(intent) => Some(intent.clone()),
                    _ => None,
                };

                if self.dry_run {
                    return Ok(StepOutcome::Handled {
                        route,
                        chosen_intent,
                        executed: false,
                    });
                }

                let executed = match &route {
                    Route::Run(intent) => self.execute_observed(intent, output, status)?,
                    Route::Confirm(intent) => {
                        self.confirm_observed(intent.clone(), input, output, status)?
                    }
                    Route::Clarify => {
                        writeln!(output, "{CLARIFY_QUESTION}")?;
                        status(StatusEvent::NotUnderstood);
                        false
                    }
                    Route::Reject(reason) => {
                        writeln!(output, "rejected: {reason:?}")?;
                        status(StatusEvent::NotUnderstood);
                        false
                    }
                };

                Ok(StepOutcome::Handled {
                    route,
                    chosen_intent,
                    executed,
                })
            }
        }
    }

    /// Runs one intent and reports success or a typed execution error.
    ///
    /// # Arguments
    /// * `intent` - The grounded, safe intent to run.
    /// * `output` - Stream that receives the executor's message.
    /// * `status` - Sink for the action or error event.
    ///
    /// # Returns
    /// `Ok(true)` because the executor was invoked, even if it returned an
    /// error. The status sink distinguishes success from failure.
    ///
    /// # Errors
    /// Returns an I/O error when writing the result fails.
    fn execute_observed(
        &mut self,
        intent: &Intent,
        output: &mut dyn Write,
        status: &mut dyn FnMut(StatusEvent),
    ) -> io::Result<bool> {
        match self.executor.run(intent) {
            Ok(message) => {
                writeln!(output, "{message}")?;
                status(StatusEvent::Action {
                    label: format_intent_action(intent),
                });
            }
            Err(error) => {
                writeln!(output, "execution failed: {error}")?;
                status(StatusEvent::Error {
                    kind: ErrorKind::Execution,
                });
            }
        }
        Ok(true)
    }

    /// Asks for a spoken-style yes/no answer and runs the intent only on yes.
    ///
    /// # Arguments
    /// * `intent` - The risky intent waiting for confirmation.
    /// * `input` - Stream to read the answer from.
    /// * `output` - Stream used to print the action and prompt.
    /// * `status` - Sink for the confirmation, action and error events.
    ///
    /// # Returns
    /// `Ok(true)` if the action was confirmed and the executor was invoked,
    /// `Ok(false)` otherwise.
    ///
    /// # Errors
    /// Returns an I/O error when reading the answer or writing a result fails.
    fn confirm_observed(
        &mut self,
        intent: Intent,
        input: &mut dyn BufRead,
        output: &mut dyn Write,
        status: &mut dyn FnMut(StatusEvent),
    ) -> io::Result<bool> {
        let label = format_intent_action(&intent);
        writeln!(output, "confirm: {label}")?;
        write!(output, "Run this action? (yes/no) ")?;
        output.flush()?;
        status(StatusEvent::Confirming { label });

        let Some(answer) = read_answer(input)? else {
            writeln!(output, "{CLARIFY_QUESTION}")?;
            status(StatusEvent::NotUnderstood);
            return Ok(false);
        };

        match nlu::parse_rule(answer.trim()) {
            Some(Intent::Confirm) => self.execute_observed(&intent, output, status),
            Some(Intent::Deny) => {
                writeln!(output, "cancelled")?;
                Ok(false)
            }
            _ => {
                writeln!(output, "{CLARIFY_QUESTION}")?;
                status(StatusEvent::NotUnderstood);
                Ok(false)
            }
        }
    }
}

/// Reads one line, returning `None` at end of input.
///
/// # Arguments
/// * `input` - The stream to read from.
///
/// # Returns
/// The line including its newline, or `None` when the stream is exhausted.
///
/// # Errors
/// Returns an I/O error when reading fails.
fn read_answer(input: &mut dyn BufRead) -> io::Result<Option<String>> {
    let mut line = String::new();
    let read = input.read_line(&mut line)?;
    if read == 0 { Ok(None) } else { Ok(Some(line)) }
}

/// Formats an [`Intent`] into a concise description for confirmation and status.
///
/// The label is deliberately human-readable: it is shown in the status window
/// and spoken by the feedback layer. It must never contain a raw model payload
/// or an executor error.
///
/// # Arguments
/// * `intent` - The intent to describe.
///
/// # Returns
/// A short description such as `click element 'Submit'`.
pub fn format_intent_action(intent: &Intent) -> String {
    match intent {
        Intent::ClickElement { target } => format!("click element '{target}'"),
        Intent::ClickNumber { n } => format!("click number {n}"),
        Intent::FocusField { target } => format!("focus field '{target}'"),
        Intent::TypeText { text, field } => match field {
            Some(f) => format!("type '{text}' into field '{f}'"),
            None => format!("type '{text}'"),
        },
        Intent::PressKey { key } => format!("press {}", format_key(*key)),
        Intent::Scroll { direction, amount } => match amount {
            Some(a) => format!("scroll {} by {a}", format_direction(*direction)),
            None => format!("scroll {}", format_direction(*direction)),
        },
        Intent::Zoom { direction } => format!("zoom {}", format_zoom(*direction)),
        Intent::OpenApp { name } => format!("open app '{name}'"),
        Intent::OpenUrl { url } => format!("open url '{url}'"),
        Intent::SwitchWindow { target } => match target {
            Some(t) => format!("switch window to '{t}'"),
            None => "switch to next window".to_owned(),
        },
        Intent::ShowNumbers => "show numbers overlay".to_owned(),
        Intent::HideNumbers => "hide numbers overlay".to_owned(),
        Intent::ShowGrid => "show mouse grid".to_owned(),
        Intent::GridSelect { cell } => format!("select grid cell {cell}"),
        Intent::SetMode { mode } => format!("set mode to {}", format_mode(*mode)),
        Intent::Confirm => "confirm".to_owned(),
        Intent::Deny => "deny".to_owned(),
        Intent::Undo => "undo".to_owned(),
        Intent::Cancel => "cancel".to_owned(),
        Intent::Clarify { question } => format!("ask: {question}"),
    }
}

/// Returns the lowercase word for a scroll direction.
///
/// # Arguments
/// * `direction` - The scroll direction from the intent.
///
/// # Returns
/// `"up"`, `"down"`, `"left"` or `"right"`.
fn format_direction(direction: Direction) -> &'static str {
    match direction {
        Direction::Up => "up",
        Direction::Down => "down",
        Direction::Left => "left",
        Direction::Right => "right",
    }
}

/// Returns the phrase for a zoom direction.
///
/// # Arguments
/// * `direction` - The zoom direction from the intent.
///
/// # Returns
/// `"in"`, `"out"` or `"reset"`.
fn format_zoom(direction: ZoomDir) -> &'static str {
    match direction {
        ZoomDir::In => "in",
        ZoomDir::Out => "out",
        ZoomDir::Reset => "reset",
    }
}

/// Returns the spoken name of a key.
///
/// # Arguments
/// * `key` - The key from the intent.
///
/// # Returns
/// A short display name such as `"Enter"` or `"Page Up"`.
fn format_key(key: Key) -> &'static str {
    match key {
        Key::Enter => "Enter",
        Key::Tab => "Tab",
        Key::Escape => "Escape",
        Key::Backspace => "Backspace",
        Key::Delete => "Delete",
        Key::Up => "Up arrow",
        Key::Down => "Down arrow",
        Key::Left => "Left arrow",
        Key::Right => "Right arrow",
        Key::PageUp => "Page Up",
        Key::PageDown => "Page Down",
        Key::Home => "Home",
        Key::End => "End",
    }
}

/// Returns the lowercase name of a mode.
///
/// # Arguments
/// * `mode` - The mode from the intent.
///
/// # Returns
/// `"command"`, `"dictation"` or `"sleep"`.
fn format_mode(mode: Mode) -> &'static str {
    match mode {
        Mode::Command => "command",
        Mode::Dictation => "dictation",
        Mode::Sleep => "sleep",
    }
}

#[cfg(test)]
mod tests {
    use reach_core::{Direction, Intent, Key, Mode, ZoomDir};

    use super::format_intent_action;

    /// Common actions get lowercase, speech-friendly labels.
    #[test]
    fn formats_common_actions_for_captions() {
        assert_eq!(
            format_intent_action(&Intent::Scroll {
                direction: Direction::Down,
                amount: None,
            }),
            "scroll down"
        );
        assert_eq!(
            format_intent_action(&Intent::Scroll {
                direction: Direction::Up,
                amount: Some(3),
            }),
            "scroll up by 3"
        );
        assert_eq!(
            format_intent_action(&Intent::Zoom {
                direction: ZoomDir::Out,
            }),
            "zoom out"
        );
        assert_eq!(
            format_intent_action(&Intent::PressKey { key: Key::PageDown }),
            "press Page Down"
        );
        assert_eq!(
            format_intent_action(&Intent::SetMode {
                mode: Mode::Command
            }),
            "set mode to command"
        );
    }
}
