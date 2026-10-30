//! Shared wake, routing, confirmation and execution pipeline.
//!
//! Both the CLI and the desktop app call this pipeline. It owns the production
//! order: the wake session first, then rule grammar and phrase rewrites, then
//! Needle model inference with a hard timeout, then one of the [`Route`] outcomes.
//! It never prints the transcript to a log.
//!
//! The pipeline reports what it is doing through a status sink. The CLI
//! passes a no-op sink and keeps its text output; the app forwards the events
//! to its status window.

use std::io::{self, BufRead, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use nlu::tools::ToolSpec;
use nlu::wake::{GateResult, WakeConfig, WakeSession};
use nlu::{
    CLARIFY_QUESTION, IntentModel, ModelError, ModelReply, NeedleConfig, NeedleIntentModel, Route,
};
use reach_core::{Executor, Intent};
use serde_json::{Map, Value, json};

use crate::config::PhrasesConfig;
use crate::status::{ErrorKind, StatusEvent};

/// Hard timeout for Needle model inference: 6 seconds.
pub const MODEL_TIMEOUT: Duration = Duration::from_secs(6);

/// The question printed or announced when Reach cannot parse or route an utterance.
pub const CLARIFY_TEXT: &str = CLARIFY_QUESTION;

/// A request sent to the background model worker thread.
struct ModelRequest {
    /// Unique correlation identifier for the request.
    id: u64,
    /// Spoken command string to classify.
    command: String,
    /// Declared tool specifications.
    tools: Vec<ToolSpec>,
}

/// A response delivered from the background model worker thread.
struct ModelResponse {
    /// Request identifier this response answers.
    id: u64,
    /// Classification reply or model error.
    result: Result<ModelReply, ModelError>,
}

/// Errors returned by the model worker.
#[derive(Debug)]
pub enum ModelWorkerError {
    /// Model call timed out before completing.
    Timeout,
    /// The model returned an error.
    Model(ModelError),
    /// The worker thread channel was closed.
    Disconnected,
}

/// Runs model inference on a dedicated background thread with a hard timeout.
pub struct ModelWorker {
    /// Sends classification requests to the worker thread.
    req_tx: Sender<ModelRequest>,
    /// Receives classification responses from the worker thread.
    resp_rx: Receiver<ModelResponse>,
    /// Next monotonic request identifier.
    next_id: u64,
    /// Background thread handle.
    _worker: Option<JoinHandle<()>>,
}

impl ModelWorker {
    /// Spawns a background worker thread that takes ownership of `model`.
    ///
    /// # Arguments
    /// * `model` - The intent model implementation to execute in the background.
    ///
    /// # Returns
    /// An initialized [`ModelWorker`].
    pub fn new<M: IntentModel + Send + 'static>(mut model: M) -> Self {
        let (req_tx, req_rx) = mpsc::channel::<ModelRequest>();
        let (resp_tx, resp_rx) = mpsc::channel::<ModelResponse>();

        let worker = thread::Builder::new()
            .name("reach-model-worker".to_owned())
            .spawn(move || {
                while let Ok(req) = req_rx.recv() {
                    let result = model.infer(&req.command, &req.tools);
                    if resp_tx.send(ModelResponse { id: req.id, result }).is_err() {
                        break;
                    }
                }
            })
            .ok();

        Self {
            req_tx,
            resp_rx,
            next_id: 1,
            _worker: worker,
        }
    }

    /// Submits `command` with `tools` to the background thread and waits up to `timeout`.
    ///
    /// # Arguments
    /// * `command` - The command text to infer.
    /// * `tools` - Declared tool specifications.
    /// * `timeout` - Hard execution deadline.
    ///
    /// # Returns
    /// The model reply, or an error if execution failed or timed out.
    pub fn infer(
        &mut self,
        command: &str,
        tools: &[ToolSpec],
        timeout: Duration,
    ) -> Result<ModelReply, ModelWorkerError> {
        while self.resp_rx.try_recv().is_ok() {}

        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);

        if self
            .req_tx
            .send(ModelRequest {
                id,
                command: command.to_owned(),
                tools: tools.to_vec(),
            })
            .is_err()
        {
            return Err(ModelWorkerError::Disconnected);
        }

        let start = Instant::now();
        loop {
            let elapsed = start.elapsed();
            if elapsed >= timeout {
                return Err(ModelWorkerError::Timeout);
            }
            let remaining = timeout.saturating_sub(elapsed);
            match self.resp_rx.recv_timeout(remaining) {
                Ok(resp) => {
                    if resp.id == id {
                        return resp.result.map_err(ModelWorkerError::Model);
                    }
                }
                Err(RecvTimeoutError::Timeout) => {
                    return Err(ModelWorkerError::Timeout);
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(ModelWorkerError::Disconnected);
                }
            }
        }
    }
}

/// A disabled model that abstains on every utterance.
#[derive(Debug, Default, Clone, Copy)]
pub struct DisabledModel;

impl IntentModel for DisabledModel {
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
    /// Returns [`ModelError`] when the container or its tokenizer cannot be loaded.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ModelError> {
        let model = NeedleIntentModel::load(path, NeedleConfig::default())?;
        Ok(Self::Needle(Box::new(model)))
    }
}

impl IntentModel for PipelineModel {
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
    /// Active wake gate and listening window state machine.
    session: WakeSession,
    /// Background model worker managing inference timeouts.
    model_worker: ModelWorker,
    /// Executor carrying out accepted intents.
    executor: E,
    /// Whether actions are reported without running.
    dry_run: bool,
    /// Optional pause before executing accepted actions.
    execution_delay: Duration,
    /// Maximum duration allowed for model inference.
    model_timeout: Duration,
    /// Phrase rewrite rules applied to commands.
    phrases: PhrasesConfig,
    /// Shared flag monitored for emergency aborts.
    stop_flag: Option<Arc<AtomicBool>>,
    /// Duration of the most recent routing pass.
    last_route_duration: Duration,
    /// Duration of the most recent rule grammar evaluation.
    last_rules_duration: Duration,
    /// Duration of the most recent model inference call.
    last_needle_duration: Duration,
    /// Duration of the most recent action execution.
    last_execute_duration: Duration,
    /// Type marker binding the model generic type parameter.
    _marker: std::marker::PhantomData<fn() -> M>,
}

impl<M, E> Pipeline<M, E> {
    /// Returns the duration of the most recent rule parse and routing decision.
    ///
    /// # Returns
    /// Total routing duration.
    pub fn last_route_duration(&self) -> Duration {
        self.last_route_duration
    }

    /// Returns the duration of the most recent rule grammar evaluation.
    ///
    /// # Returns
    /// Rule parse duration.
    pub fn last_rules_duration(&self) -> Duration {
        self.last_rules_duration
    }

    /// Returns the duration of the most recent Needle model inference.
    ///
    /// # Returns
    /// Model inference duration.
    pub fn last_needle_duration(&self) -> Duration {
        self.last_needle_duration
    }

    /// Returns the duration of the most recent intent execution.
    ///
    /// # Returns
    /// Action execution duration.
    pub fn last_execute_duration(&self) -> Duration {
        self.last_execute_duration
    }

    /// Configures phrase rewrite rules.
    ///
    /// # Arguments
    /// * `phrases` - Custom phrase mappings.
    ///
    /// # Returns
    /// Self with configured phrase rewrites.
    pub fn with_phrases(mut self, phrases: PhrasesConfig) -> Self {
        self.phrases = phrases;
        self
    }

    /// Sets the timeout applied to model inference calls.
    ///
    /// # Arguments
    /// * `timeout` - Maximum wait duration for model responses.
    pub fn set_model_timeout(&mut self, timeout: Duration) {
        self.model_timeout = timeout;
    }

    /// Attaches an emergency stop flag to monitor.
    ///
    /// # Arguments
    /// * `flag` - Atomic boolean flag set when emergency stop is triggered.
    ///
    /// # Returns
    /// Updated pipeline instance.
    pub fn with_stop_flag(mut self, flag: Arc<AtomicBool>) -> Self {
        self.stop_flag = Some(flag);
        self
    }

    /// Returns whether the emergency stop flag is currently active.
    ///
    /// # Returns
    /// True if emergency stop flag is set, false otherwise.
    pub fn is_stopped(&self) -> bool {
        self.stop_flag
            .as_ref()
            .is_some_and(|f| f.load(Ordering::SeqCst))
    }

    /// Sets an execution delay and returns the pipeline.
    ///
    /// # Arguments
    /// * `delay` - How long to pause before calling the executor.
    ///
    /// # Returns
    /// The updated [`Pipeline`].
    pub fn with_execution_delay(mut self, delay: Duration) -> Self {
        self.execution_delay = delay;
        self
    }

    /// Sets how long to wait before executing an accepted command.
    ///
    /// # Arguments
    /// * `delay` - How long to pause before calling the executor.
    pub fn set_execution_delay(&mut self, delay: Duration) {
        self.execution_delay = delay;
    }

    /// Returns the configured execution delay.
    ///
    /// # Returns
    /// The pause duration before command execution.
    pub fn execution_delay(&self) -> Duration {
        self.execution_delay
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

impl<M: IntentModel + Send + 'static, E: Executor> Pipeline<M, E> {
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
            model_worker: ModelWorker::new(model),
            executor,
            dry_run,
            execution_delay: Duration::ZERO,
            last_route_duration: Duration::ZERO,
            last_rules_duration: Duration::ZERO,
            last_needle_duration: Duration::ZERO,
            last_execute_duration: Duration::ZERO,
            model_timeout: MODEL_TIMEOUT,
            phrases: PhrasesConfig::default(),
            stop_flag: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// Processes one transcript and writes the user-facing result.
    ///
    /// # Arguments
    /// * `transcript` - One line of text.
    /// * `now` - The current time.
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
    /// # Arguments
    /// * `utterance` - The recognised text, including any wake word.
    /// * `now` - Timestamp for listening window evaluation.
    /// * `reader` - Stream to read a confirmation answer from.
    /// * `writer` - Stream to write human-readable text to.
    /// * `status` - Sink for [`StatusEvent`] values.
    ///
    /// # Returns
    /// The [`StepOutcome`] indicating how the line was handled.
    ///
    /// # Errors
    /// Returns an I/O error if reading or writing fails.
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
    /// * `now` - The current time.
    ///
    /// # Returns
    /// `true` while the next utterance can omit the wake word.
    pub fn is_listening(&self, now: Instant) -> bool {
        self.session.is_open(now)
    }

    /// Shared implementation for the text and status front ends.
    fn process_with(
        &mut self,
        transcript: &str,
        now: Instant,
        input: &mut dyn BufRead,
        output: &mut dyn Write,
        status: &mut dyn FnMut(StatusEvent),
    ) -> io::Result<StepOutcome> {
        self.last_route_duration = Duration::ZERO;
        self.last_rules_duration = Duration::ZERO;
        self.last_needle_duration = Duration::ZERO;
        self.last_execute_duration = Duration::ZERO;
        let route_start = Instant::now();

        let was_listening = self.session.is_open(now);
        match self.session.evaluate(transcript, now) {
            GateResult::Ignored => {
                self.last_route_duration = route_start.elapsed();
                writeln!(output, "ignored")?;
                status(StatusEvent::Heard {
                    text: transcript.to_owned(),
                });
                Ok(StepOutcome::Ignored)
            }
            GateResult::WakeOnly => {
                self.last_route_duration = route_start.elapsed();
                writeln!(output, "listening")?;
                status(StatusEvent::Heard {
                    text: transcript.to_owned(),
                });
                status(StatusEvent::Listening);
                Ok(StepOutcome::WakeOnly)
            }
            GateResult::Command(command) => {
                if !was_listening {
                    status(StatusEvent::Listening);
                }
                status(StatusEvent::Heard {
                    text: transcript.to_owned(),
                });
                let route = self.route_command(&command);
                self.last_route_duration = route_start.elapsed();

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

    /// Routes a command through phrase rewriting, rule grammar, and model inference.
    ///
    /// # Arguments
    /// * `command` - The command string after the wake gate.
    ///
    /// # Returns
    /// The resolved [`Route`].
    pub fn route_command(&mut self, command: &str) -> Route {
        let rewritten = self.phrases.rewrite(command);
        let rules_start = Instant::now();
        let rule_match = nlu::parse_rule(&rewritten);
        self.last_rules_duration = rules_start.elapsed();

        if let Some(intent) = rule_match {
            nlu::route_rule_intent(intent)
        } else {
            let command_to_model = rewritten.as_ref();
            let words = command_to_model.split_whitespace().count();
            if !(nlu::routing::MIN_MODEL_WORDS..=nlu::routing::MAX_MODEL_WORDS).contains(&words) {
                return Route::Clarify;
            }

            let needle_start = Instant::now();
            let tools = nlu::tool_specs();
            let timeout = self.model_timeout;
            let outcome = self.model_worker.infer(command_to_model, &tools, timeout);
            self.last_needle_duration = needle_start.elapsed();

            match outcome {
                Ok(reply) => nlu::route_model_reply(command_to_model, Ok(reply)),
                Err(ModelWorkerError::Timeout) => Route::Clarify,
                Err(ModelWorkerError::Disconnected) => Route::Clarify,
                Err(ModelWorkerError::Model(err)) => {
                    nlu::route_model_reply(command_to_model, Err(err))
                }
            }
        }
    }

    /// Runs one intent and reports success or a typed execution error.
    fn execute_observed(
        &mut self,
        intent: &Intent,
        output: &mut dyn Write,
        status: &mut dyn FnMut(StatusEvent),
    ) -> io::Result<bool> {
        let label = format_intent_action(intent);
        status(StatusEvent::Action {
            label: label.to_string(),
        });
        if !self.execution_delay.is_zero() {
            thread::sleep(self.execution_delay);
        }
        let exec_start = Instant::now();
        let result = self.executor.run(intent);
        self.last_execute_duration = exec_start.elapsed();
        match result {
            Ok(message) => {
                writeln!(output, "{message}")?;
                Ok(true)
            }
            Err(error) => {
                writeln!(output, "execution failed: {error}")?;
                status(StatusEvent::Error {
                    kind: ErrorKind::Execution,
                });
                Ok(false)
            }
        }
    }

    /// Asks for confirmation and executes only on an accepted answer.
    fn confirm_observed(
        &mut self,
        intent: Intent,
        input: &mut dyn BufRead,
        output: &mut dyn Write,
        status: &mut dyn FnMut(StatusEvent),
    ) -> io::Result<bool> {
        let label = format_intent_action(&intent);
        status(StatusEvent::Confirming {
            label: label.to_string(),
        });
        writeln!(output, "confirm: {label}? (yes/no)")?;
        output.flush()?;

        let mut answer = String::new();
        match input.read_line(&mut answer) {
            Ok(0) => {
                writeln!(output, "cancelled")?;
                status(StatusEvent::Asleep);
                Ok(false)
            }
            Ok(_) => {
                let trimmed = answer.trim();
                status(StatusEvent::Heard {
                    text: trimmed.to_string(),
                });
                let rule = parse_confirmation_answer(trimmed);
                match rule {
                    Some(Intent::Confirm) => self.execute_observed(&intent, output, status),
                    Some(Intent::Deny) | Some(Intent::Cancel) => {
                        writeln!(output, "cancelled")?;
                        status(StatusEvent::Asleep);
                        Ok(false)
                    }
                    _ => {
                        writeln!(output, "cancelled")?;
                        status(StatusEvent::NotUnderstood);
                        status(StatusEvent::Asleep);
                        Ok(false)
                    }
                }
            }
            Err(err) => {
                writeln!(output, "cancelled: {err}")?;
                status(StatusEvent::Error {
                    kind: ErrorKind::Io,
                });
                Ok(false)
            }
        }
    }
}

/// Parses an answer to a confirmation prompt.
///
/// Recognises direct affirmative/negative words, answers prefixed or suffixed with
/// the wake word, polite phrases, and repeats of the action command.
///
/// # Arguments
/// * `answer` - Spoken or typed answer string.
///
/// # Returns
/// An [`Intent::Confirm`], [`Intent::Deny`], or [`Intent::Cancel`] if recognized,
/// otherwise `None`.
pub fn parse_confirmation_answer(answer: &str) -> Option<Intent> {
    if let Some(intent) = nlu::parse_rule(answer) {
        match intent {
            Intent::Confirm | Intent::Deny | Intent::Cancel => return Some(intent),
            _ => {}
        }
    }
    let lower = answer.to_lowercase();
    let cleaned: String = lower
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect();
    let words: Vec<&str> = cleaned.split_whitespace().collect();
    if words.is_empty() {
        return None;
    }

    let filtered: Vec<&str> = words
        .into_iter()
        .filter(|&w| !matches!(w, "reach" | "rich" | "ridge" | "please" | "kindly"))
        .collect();

    match filtered.as_slice() {
        [word] => match *word {
            "yes" | "yeah" | "yep" | "sure" | "okay" | "ok" | "confirm" | "correct"
            | "affirmative" | "proceed" | "continue" | "accept" => Some(Intent::Confirm),
            "no" | "nope" | "nah" | "deny" | "negative" | "dont" => Some(Intent::Deny),
            "cancel" | "stop" | "abort" | "nevermind" => Some(Intent::Cancel),
            _ => None,
        },
        [first, second] => match (*first, *second) {
            ("do", "it") | ("go", "ahead") | ("thats", "right") => Some(Intent::Confirm),
            ("yes", "do") | ("yes", "please") | ("yes", "confirm") => Some(Intent::Confirm),
            ("no", "thanks") | ("no", "way") => Some(Intent::Deny),
            ("never", "mind") | ("forget", "it") | ("cancel", "that") => Some(Intent::Cancel),
            _ => None,
        },
        [first, second, third] => match (*first, *second, *third) {
            ("yes", "do", "it") | ("yes", "go", "ahead") => Some(Intent::Confirm),
            ("no", "thank", "you") => Some(Intent::Deny),
            _ => None,
        },
        _ => None,
    }
}

/// Returns the stable action label used in `StatusEvent::Action` and confirmations.
///
/// # Arguments
/// * `intent` - The intent whose action is being named.
///
/// # Returns
/// A static lower-case verb phrase describing the action.
pub fn format_intent_action(intent: &Intent) -> &'static str {
    match intent {
        Intent::ClickElement { .. } => "click element",
        Intent::ClickNumber { .. } => "click number",
        Intent::TypeText { .. } => "type text",
        Intent::FocusField { .. } => "focus field",
        Intent::PressKey { .. } => "press key",
        Intent::Scroll { .. } => "scroll",
        Intent::Zoom { .. } => "zoom",
        Intent::OpenApp { .. } => "open application",
        Intent::OpenUrl { .. } => "open url",
        Intent::SwitchWindow { .. } => "switch window",
        Intent::ShowNumbers => "show numbers",
        Intent::HideNumbers => "hide numbers",
        Intent::ShowGrid => "show grid",
        Intent::GridSelect { .. } => "grid select",
        Intent::SetMode { .. } => "set mode",
        Intent::Confirm => "confirm",
        Intent::Deny => "deny",
        Intent::Undo => "undo",
        Intent::Cancel => "cancel",
        Intent::Clarify { .. } => "clarify",
    }
}

/// Formats a [`Route`] as JSON for `--dry-run`.
///
/// # Arguments
/// * `route` - The routing decision made for an utterance.
///
/// # Returns
/// Single JSON line with no trailing newline.
pub fn route_json(route: &Route) -> String {
    match route {
        Route::Run(intent) => json!({
            "route": "run",
            "action": format_intent_action(intent),
            "intent": intent_json(intent),
        })
        .to_string(),
        Route::Confirm(intent) => json!({
            "route": "confirm",
            "action": format_intent_action(intent),
            "intent": intent_json(intent),
        })
        .to_string(),
        Route::Clarify => json!({
            "route": "clarify",
            "question": CLARIFY_QUESTION,
        })
        .to_string(),
        Route::Reject(reason) => json!({
            "route": "reject",
            "reason": format!("{reason:?}"),
        })
        .to_string(),
    }
}

/// Formats an [`Intent`] as a JSON object.
///
/// # Arguments
/// * `intent` - The structured intent to format.
///
/// # Returns
/// A serialized JSON string.
pub fn intent_json(intent: &Intent) -> String {
    let mut map = Map::new();
    match intent {
        Intent::ClickElement { target } => {
            map.insert("target".to_owned(), json!(target));
        }
        Intent::ClickNumber { n } => {
            map.insert("n".to_owned(), json!(n));
        }
        Intent::TypeText { text, field } => {
            map.insert("text".to_owned(), json!(text));
            if let Some(f) = field {
                map.insert("field".to_owned(), json!(f));
            }
        }
        Intent::FocusField { target } => {
            map.insert("target".to_owned(), json!(target));
        }
        Intent::PressKey { key } => {
            map.insert("key".to_owned(), json!(format!("{key:?}")));
        }
        Intent::Scroll { direction, amount } => {
            map.insert("direction".to_owned(), json!(format!("{direction:?}")));
            if let Some(amt) = amount {
                map.insert("amount".to_owned(), json!(amt));
            }
        }
        Intent::Zoom { direction } => {
            map.insert("direction".to_owned(), json!(format!("{direction:?}")));
        }
        Intent::OpenApp { name } => {
            map.insert("name".to_owned(), json!(name));
        }
        Intent::OpenUrl { url } => {
            map.insert("url".to_owned(), json!(url));
        }
        Intent::SwitchWindow { target: Some(t) } => {
            map.insert("target".to_owned(), json!(t));
        }
        Intent::GridSelect { cell } => {
            map.insert("cell".to_owned(), json!(cell));
        }
        Intent::SetMode { mode } => {
            map.insert("mode".to_owned(), json!(format!("{mode:?}")));
        }
        Intent::Clarify { question } => {
            map.insert("question".to_owned(), json!(question));
        }
        _ => {}
    }
    Value::Object(map).to_string()
}

/// Prints the JSON description of a routed command for `--dry-run`.
///
/// # Arguments
/// * `outcome` - The result returned by [`Pipeline::process`].
/// * `output` - Stream that receives the JSON line.
///
/// # Returns
/// `Ok(())` when the line was written, or when outcome was not a routed command.
///
/// # Errors
/// Returns an I/O error when writing fails.
pub fn print_dry_run(outcome: &StepOutcome, output: &mut dyn Write) -> io::Result<()> {
    if let StepOutcome::Handled { route, .. } = outcome {
        writeln!(output, "{}", route_json(route))?;
    }
    Ok(())
}
