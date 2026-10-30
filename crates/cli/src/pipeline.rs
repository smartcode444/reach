//! Typed-text pipeline glue for the CLI harness.
//!
//! The pipeline replaces the microphone with typed input while keeping the
//! production order: the wake session first, then [`nlu::route_utterance`],
//! then one of the [`Route`] outcomes. It never re-implements routing and never
//! prints the transcript.

use std::io::{self, BufRead, Write};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use nlu::tools::ToolSpec;
use nlu::wake::{GateResult, WakeConfig, WakeSession};
use nlu::{CLARIFY_QUESTION, IntentModel, ModelError, ModelReply, NeedleIntentModel, Route};
use reach_core::{Executor, Intent};
use serde_json::{Map, Value};

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

/// The model selected for this CLI run.
pub enum CliModel {
    /// The real Needle model, loaded once at startup.
    Needle(Box<NeedleIntentModel>),
    /// No model was requested; every rules miss abstains.
    Offline,
}

impl IntentModel for CliModel {
    /// Runs the selected model, or abstains when the model is off.
    ///
    /// # Arguments
    /// * `utterance` - The command text after the wake gate.
    /// * `tools` - The closed tool list the model may call.
    ///
    /// # Returns
    /// The model reply, or an empty call list when the model is off. The
    /// router treats an empty list as "no tool matches" and asks for a
    /// clarification.
    ///
    /// # Errors
    /// Returns the real model's load, generation or payload error.
    fn infer(&mut self, utterance: &str, tools: &[ToolSpec]) -> Result<ModelReply, ModelError> {
        match self {
            Self::Needle(model) => model.infer(utterance, tools),
            Self::Offline => Ok(ModelReply {
                calls: Vec::new(),
                confidence: None,
                truncated: false,
            }),
        }
    }
}

/// Dynamic wrapper selecting between disabled and active intent models.
pub enum PipelineModel {
    /// Model inference is disabled; rule misses will clarify.
    Disabled(DisabledModel),
    /// Real in-process Needle 3 engine.
    Needle(Box<nlu::NeedleIntentModel>),
    #[cfg(test)]
    /// Scripted fake model for unit tests.
    Scripted(nlu::ScriptedModel),
}

impl IntentModel for PipelineModel {
    /// Routes inference to the active model variant.
    ///
    /// # Arguments
    /// * `utterance` - The recognised text.
    /// * `tools` - The declared tool specifications.
    ///
    /// # Returns
    /// The model reply with selected calls and optional confidence.
    ///
    /// # Errors
    /// Returns an error if the underlying model fails.
    fn infer(&mut self, utterance: &str, tools: &[ToolSpec]) -> Result<ModelReply, ModelError> {
        match self {
            Self::Disabled(m) => m.infer(utterance, tools),
            Self::Needle(m) => m.infer(utterance, tools),
            #[cfg(test)]
            Self::Scripted(m) => m.infer(utterance, tools),
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

/// The outcome of processing a single utterance line through the pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum StepOutcome {
    /// Utterance was ignored because the wake word was missing and window was closed.
    Ignored,
    /// Utterance was solely the wake word; the listening window was opened.
    WakeOnly,
    /// Utterance was classified and routed as a command.
    Handled {
        /// The routing decision made.
        route: Route,
        /// The chosen intent, if one was selected.
        chosen_intent: Option<Intent>,
        /// Whether the executor was invoked.
        executed: bool,
    },
}

/// Hard timeout for Needle model inference: 6 seconds.
pub const MODEL_TIMEOUT: Duration = Duration::from_secs(6);

/// A request sent to the background model worker thread.
struct ModelRequest {
    /// Unique sequence identifier for request-response pairing.
    id: u64,
    /// Command text to infer.
    command: String,
    /// Declared tool specifications.
    tools: Vec<ToolSpec>,
}

/// A response delivered from the background model worker thread.
struct ModelResponse {
    /// Sequence identifier matching the originating request.
    id: u64,
    /// Outcome of model inference.
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
    /// Channel sender delivering requests to the worker thread.
    req_tx: Sender<ModelRequest>,
    /// Channel receiver collecting completed responses.
    resp_rx: Receiver<ModelResponse>,
    /// Monotonically increasing request identifier.
    next_id: u64,
    /// Handle to the worker thread.
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

    /// Runs model inference with a hard timeout.
    ///
    /// Drains any late responses from prior timed-out requests.
    ///
    /// # Arguments
    /// * `command` - Command text to infer.
    /// * `tools` - Declared tool specifications.
    /// * `timeout` - Maximum duration to wait before timing out.
    ///
    /// # Returns
    /// `Ok(ModelReply)` on completion, or [`ModelWorkerError::Timeout`] on expiration.
    pub fn infer(
        &mut self,
        command: &str,
        tools: &[ToolSpec],
        timeout: Duration,
    ) -> Result<ModelReply, ModelWorkerError> {
        // Drain any stale/late responses
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
/// One typed-text pipeline with its wake session, model and executor.
pub struct Pipeline<M, E> {
    /// The wake gate and listening window shared by every line.
    session: WakeSession,
    /// Background worker executing model calls with a hard timeout.
    model_worker: ModelWorker,
    /// The executor used only for `Run` and confirmed `Confirm` routes.
    executor: E,
    /// When true, routes are printed as JSON and nothing runs.
    dry_run: bool,
    /// Duration of the most recent rule parse and routing decision.
    last_route_duration: Duration,
    /// Duration of the most recent rule grammar evaluation.
    last_rules_duration: Duration,
    /// Duration of the most recent Needle model inference, if any.
    last_needle_duration: Duration,
    /// Duration of the most recent intent execution, if any.
    last_execute_duration: Duration,
    /// Timeout applied to model calls.
    model_timeout: Duration,
    /// Custom phrase rewrite configuration.
    phrases: crate::config::PhrasesConfig,
    /// Marker to retain type parameter M.
    _marker: std::marker::PhantomData<fn() -> M>,
}

impl<M, E> Pipeline<M, E> {
    /// Returns the duration of the most recent rule parse and routing decision.
    ///
    /// # Returns
    /// The [`Duration`] spent evaluating the wake gate, grammar rules, and model.
    pub fn last_route_duration(&self) -> Duration {
        self.last_route_duration
    }

    /// Returns the duration of the most recent rule grammar evaluation.
    ///
    /// # Returns
    /// The [`Duration`] spent matching the rule grammar.
    pub fn last_rules_duration(&self) -> Duration {
        self.last_rules_duration
    }

    /// Returns the duration of the most recent Needle model inference.
    ///
    /// # Returns
    /// The [`Duration`] spent in Needle inference, or zero if rules matched or no model was queried.
    pub fn last_needle_duration(&self) -> Duration {
        self.last_needle_duration
    }

    /// Returns the duration of the most recent intent execution.
    ///
    /// # Returns
    /// The [`Duration`] spent inside the executor's `run` method, or zero if no
    /// intent was executed.
    pub fn last_execute_duration(&self) -> Duration {
        self.last_execute_duration
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

    /// Sets the timeout for Needle model calls.
    ///
    /// # Arguments
    /// * `timeout` - Duration to wait for model inference before timing out.
    pub fn set_model_timeout(&mut self, timeout: Duration) {
        self.model_timeout = timeout;
    }

    /// Sets a custom wake word configuration.
    ///
    /// # Arguments
    /// * `config` - Wake word, aliases, and listening window settings.
    ///
    /// # Returns
    /// The updated pipeline instance.
    pub fn with_wake_config(mut self, config: WakeConfig) -> Self {
        self.session = WakeSession::new(config);
        self
    }

    /// Sets custom phrase rewrites mapping spoken phrases to canonical commands.
    ///
    /// # Arguments
    /// * `phrases` - The phrases configuration loaded from `config\phrases.toml`.
    ///
    /// # Returns
    /// The updated pipeline instance.
    pub fn with_phrases(mut self, phrases: crate::config::PhrasesConfig) -> Self {
        self.phrases = phrases;
        self
    }
}

impl<M: IntentModel + Send + 'static, E> Pipeline<M, E> {
    /// Creates a pipeline with the default wake configuration and 6 s model timeout.
    ///
    /// # Arguments
    /// * `model` - The intent model to consult on rule misses.
    /// * `executor` - The executor carrying out accepted intents.
    /// * `dry_run` - If true, outputs route decisions as JSON without executing.
    ///
    /// # Returns
    /// An initialized [`Pipeline`].
    pub fn new(model: M, executor: E, dry_run: bool) -> Self {
        Self {
            session: WakeSession::new(WakeConfig::default()),
            model_worker: ModelWorker::new(model),
            executor,
            dry_run,
            last_route_duration: Duration::ZERO,
            last_rules_duration: Duration::ZERO,
            last_needle_duration: Duration::ZERO,
            last_execute_duration: Duration::ZERO,
            model_timeout: MODEL_TIMEOUT,
            phrases: crate::config::PhrasesConfig::default(),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<M: IntentModel + Send + 'static, E: Executor> Pipeline<M, E> {
    /// Evaluates the rule grammar first, then Needle if rules miss, recording per-stage timings.
    ///
    /// # Arguments
    /// * `command` - The command string stripped of the wake word.
    ///
    /// # Returns
    /// The computed [`Route`].
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
                Err(ModelWorkerError::Model(err)) => {
                    nlu::route_model_reply(command_to_model, Err(err))
                }
                Err(ModelWorkerError::Disconnected) => Route::Clarify,
            }
        }
    }

    /// Processes one typed transcript and writes the user-facing result.
    ///
    /// # Arguments
    /// * `transcript` - One line of typed text, as if it came from the
    ///   microphone.
    /// * `now` - The current time, injected so the listening window is
    ///   deterministic in tests.
    /// * `io` - The streams used for confirmation answers and results.
    ///
    /// # Returns
    /// `Ok(())` after the result has been written, or an I/O error from the
    /// streams. The transcript itself is never written.
    ///
    /// # Errors
    /// Returns an error when reading an answer or writing a result fails.
    pub fn process(
        &mut self,
        transcript: &str,
        now: Instant,
        io: &mut PipelineIo<'_>,
    ) -> io::Result<()> {
        self.last_route_duration = Duration::ZERO;
        self.last_rules_duration = Duration::ZERO;
        self.last_needle_duration = Duration::ZERO;
        self.last_execute_duration = Duration::ZERO;
        let route_start = Instant::now();
        match self.session.evaluate(transcript, now) {
            GateResult::Ignored => {
                self.last_route_duration = route_start.elapsed();
                writeln!(io.output, "ignored")
            }
            GateResult::WakeOnly => {
                self.last_route_duration = route_start.elapsed();
                writeln!(io.output, "listening")
            }
            GateResult::Command(command) => {
                let route = self.route_command(&command);
                self.last_route_duration = route_start.elapsed();
                self.handle_route(route, io)
            }
        }
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
    /// Returns an error message if I/O fails.
    pub fn process_utterance<R: BufRead, W: Write>(
        &mut self,
        utterance: &str,
        now: Instant,
        reader: &mut R,
        writer: &mut W,
    ) -> Result<StepOutcome, String> {
        let gate_result = self.session.evaluate(utterance, now);
        let command = match gate_result {
            GateResult::Ignored => {
                writeln!(writer, "ignored").map_err(|e| e.to_string())?;
                return Ok(StepOutcome::Ignored);
            }
            GateResult::WakeOnly => {
                writeln!(writer, "listening").map_err(|e| e.to_string())?;
                return Ok(StepOutcome::WakeOnly);
            }
            GateResult::Command(command) => command,
        };

        let route = self.route_command(&command);
        let chosen_intent = match &route {
            Route::Run(intent) | Route::Confirm(intent) => Some(intent.clone()),
            _ => None,
        };

        if self.dry_run {
            writeln!(writer, "{}", route_json(&route)).map_err(|e| e.to_string())?;
            return Ok(StepOutcome::Handled {
                route,
                chosen_intent,
                executed: false,
            });
        }

        let mut io = PipelineIo::new(reader, writer);
        match &route {
            Route::Run(intent) => {
                self.execute(intent, &mut io).map_err(|e| e.to_string())?;
                Ok(StepOutcome::Handled {
                    route,
                    chosen_intent,
                    executed: true,
                })
            }
            Route::Confirm(intent) => {
                let executed = self
                    .confirm_and_report(intent.clone(), &mut io)
                    .map_err(|e| e.to_string())?;
                Ok(StepOutcome::Handled {
                    route,
                    chosen_intent,
                    executed,
                })
            }
            Route::Clarify => {
                writeln!(io.output, "{CLARIFY_QUESTION}").map_err(|e| e.to_string())?;
                Ok(StepOutcome::Handled {
                    route,
                    chosen_intent: None,
                    executed: false,
                })
            }
            Route::Reject(reason) => {
                writeln!(io.output, "rejected: {reason:?}").map_err(|e| e.to_string())?;
                Ok(StepOutcome::Handled {
                    route,
                    chosen_intent: None,
                    executed: false,
                })
            }
        }
    }

    /// Runs interactive line-by-line processing until EOF or exit command.
    ///
    /// # Arguments
    /// * `reader` - Stream to read input lines from.
    /// * `writer` - Stream to write output responses to.
    ///
    /// # Returns
    /// `Ok(())` on EOF or exit.
    ///
    /// # Errors
    /// Returns an error message if I/O fails.
    pub fn run_interactive<R: BufRead, W: Write>(
        &mut self,
        reader: &mut R,
        writer: &mut W,
    ) -> Result<(), String> {
        let mut line = String::new();
        while reader.read_line(&mut line).map_err(|e| e.to_string())? > 0 {
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
                    break;
                }
                let mut io = PipelineIo::new(reader, writer);
                self.process(trimmed, Instant::now(), &mut io)
                    .map_err(|e| e.to_string())?;
            }
            line.clear();
        }
        Ok(())
    }

    /// Writes or carries out one route.
    ///
    /// # Arguments
    /// * `route` - The decision returned by [`nlu::route_utterance`].
    /// * `io` - The streams used for confirmation answers and results.
    ///
    /// # Returns
    /// `Ok(())` once the route has been handled.
    ///
    /// # Errors
    /// Returns an I/O error from the streams.
    fn handle_route(&mut self, route: Route, io: &mut PipelineIo<'_>) -> io::Result<()> {
        if self.dry_run {
            return writeln!(io.output, "{}", route_json(&route));
        }

        match route {
            Route::Run(intent) => self.execute(&intent, io),
            Route::Confirm(intent) => {
                let _ = self.confirm_and_report(intent, io)?;
                Ok(())
            }
            Route::Clarify => writeln!(io.output, "{CLARIFY_QUESTION}"),
            Route::Reject(reason) => writeln!(io.output, "rejected: {reason:?}"),
        }
    }

    /// Runs one intent through the executor and writes its message.
    ///
    /// # Arguments
    /// * `intent` - The grounded, safe intent to run.
    /// * `io` - The stream that receives the executor's result.
    ///
    /// # Returns
    /// `Ok(())` whether the executor succeeded or failed.
    ///
    /// # Errors
    /// Returns an I/O error when writing the result fails.
    fn execute(&mut self, intent: &Intent, io: &mut PipelineIo<'_>) -> io::Result<()> {
        let exec_start = Instant::now();
        let result = self.executor.run(intent);
        self.last_execute_duration = exec_start.elapsed();
        match result {
            Ok(message) => writeln!(io.output, "{message}"),
            Err(error) => writeln!(io.output, "execution failed: {error}"),
        }
    }

    /// Asks for a spoken-style yes/no answer and runs the intent only on yes.
    ///
    /// # Arguments
    /// * `intent` - The risky intent waiting for confirmation.
    /// * `io` - The streams used to print the action, read the answer and
    ///   write the result.
    ///
    /// # Returns
    /// `Ok(true)` if the action was confirmed and executed, `Ok(false)` otherwise.
    ///
    /// # Errors
    /// Returns an I/O error when reading the answer or writing a result fails.
    fn confirm_and_report(&mut self, intent: Intent, io: &mut PipelineIo<'_>) -> io::Result<bool> {
        writeln!(io.output, "confirm: {}", intent_json(&intent))?;
        write!(io.output, "Run this action? (yes/no) ")?;
        io.output.flush()?;

        let Some(answer) = read_answer(io.input)? else {
            writeln!(io.output, "{CLARIFY_QUESTION}")?;
            return Ok(false);
        };

        match nlu::parse_rule(answer.trim()) {
            Some(Intent::Confirm) => {
                self.execute(&intent, io)?;
                Ok(true)
            }
            // "cancel", "stop" and "abort" are control intents rather than
            // `Deny`, but they mean the same thing at a confirmation prompt.
            Some(Intent::Deny) | Some(Intent::Cancel) => {
                writeln!(io.output, "cancelled")?;
                Ok(false)
            }
            _ => {
                writeln!(io.output, "{CLARIFY_QUESTION}")?;
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

/// Serialises one intent for the confirmation line.
///
/// # Arguments
/// * `intent` - The intent to show.
///
/// # Returns
/// The intent's JSON form, or `null` if it cannot be serialised.
pub fn intent_json(intent: &Intent) -> String {
    serde_json::to_string(intent).unwrap_or_else(|_| "null".to_owned())
}

/// Serialises a route and its intent for `--dry-run`.
///
/// # Arguments
/// * `route` - The decision to report.
///
/// # Returns
/// One JSON object with a `route` field and, where one exists, the chosen
/// intent. A rejection carries only the reason type, never the transcript or
/// the model's arguments.
pub fn route_json(route: &Route) -> String {
    let mut object = Map::new();
    let route_name = match route {
        Route::Run(_) => "run",
        Route::Confirm(_) => "confirm",
        Route::Clarify => "clarify",
        Route::Reject(_) => "reject",
    };
    object.insert("route".to_owned(), Value::String(route_name.to_owned()));

    match route {
        Route::Run(intent) | Route::Confirm(intent) => {
            object.insert(
                "intent".to_owned(),
                serde_json::to_value(intent).unwrap_or(Value::Null),
            );
        }
        Route::Clarify => {
            object.insert("intent".to_owned(), Value::Null);
        }
        Route::Reject(reason) => {
            object.insert("reason".to_owned(), Value::String(format!("{reason:?}")));
            object.insert("intent".to_owned(), Value::Null);
        }
    }

    Value::Object(object).to_string()
}

/// Converts a [`Route`] into a JSON value for dry-run inspection.
///
/// # Arguments
/// * `route` - The routing decision.
///
/// # Returns
/// A [`serde_json::Value`] representing the route.
pub fn route_to_json(route: &Route) -> serde_json::Value {
    match route {
        Route::Run(intent) => serde_json::json!({
            "type": "run",
            "intent": intent,
        }),
        Route::Confirm(intent) => serde_json::json!({
            "type": "confirm",
            "intent": intent,
        }),
        Route::Clarify => serde_json::json!({
            "type": "clarify",
            "question": CLARIFY_QUESTION,
        }),
        Route::Reject(reason) => serde_json::json!({
            "type": "reject",
            "reason": format!("{reason:?}"),
        }),
    }
}

/// Formats an [`Intent`] into a concise description for spoken/typed confirmation.
///
/// # Arguments
/// * `intent` - The intent awaiting confirmation.
///
/// # Returns
/// Human-readable action description.
pub fn format_intent_action(intent: &Intent) -> String {
    match intent {
        Intent::ClickElement { target } => format!("click element '{target}'"),
        Intent::ClickNumber { n } => format!("click number {n}"),
        Intent::FocusField { target } => format!("focus field '{target}'"),
        Intent::TypeText { text, field } => match field {
            Some(f) => format!("type '{text}' into field '{f}'"),
            None => format!("type '{text}'"),
        },
        Intent::PressKey { key } => format!("press key {key:?}"),
        Intent::Scroll { direction, amount } => match amount {
            Some(a) => format!("scroll {direction:?} by {a}"),
            None => format!("scroll {direction:?}"),
        },
        Intent::Zoom { direction } => format!("zoom {direction:?}"),
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
        Intent::SetMode { mode } => format!("set mode to {mode:?}"),
        Intent::Confirm => "confirm".to_owned(),
        Intent::Deny => "deny".to_owned(),
        Intent::Undo => "undo".to_owned(),
        Intent::Cancel => "cancel".to_owned(),
        Intent::Clarify { question } => format!("ask: {question}"),
    }
}

/// Options parsed from the `pipeline` subcommand arguments.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct PipelineArgs {
    /// Whether to load and use the Needle model.
    pub model: bool,
    /// Path to the model container, if explicitly provided.
    pub model_path: Option<String>,
    /// Whether to print routes as JSON instead of running them.
    pub dry_run: bool,
    /// The single transcript, or `None` for interactive stdin mode.
    pub utterance: Option<String>,
}

impl PipelineArgs {
    /// Parses CLI tokens following the `pipeline` subcommand keyword.
    ///
    /// # Arguments
    /// * `tokens` - Slice of argument strings following `pipeline`.
    ///
    /// # Returns
    /// An instance of [`PipelineArgs`].
    pub fn parse(tokens: &[String]) -> Self {
        parse_pipeline_args(tokens).unwrap_or(Self {
            model: false,
            model_path: None,
            dry_run: false,
            utterance: None,
        })
    }
}

/// Parses `pipeline` subcommand arguments.
///
/// # Arguments
/// * `args` - Everything after the `pipeline` subcommand.
///
/// # Returns
/// The parsed flags and the joined transcript, if one was supplied.
///
/// # Errors
/// Returns a message naming an unknown `--` option. The rest of the words are
/// never included in the message, so a mistyped flag cannot leak a transcript.
pub fn parse_pipeline_args(args: &[String]) -> Result<PipelineArgs, String> {
    let mut model = false;
    let mut model_path = None;
    let mut dry_run = false;
    let mut words = Vec::new();

    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if arg == "--model" {
            model = true;
            if let Some(next) = args.get(i + 1).filter(|next| {
                !next.starts_with("--")
                    && (next.ends_with(".cact")
                        || next.ends_with(".bin")
                        || next.contains('/')
                        || next.contains('\\'))
            }) {
                model_path = Some(next.clone());
                i += 1;
            }
        } else if let Some(path) = arg.strip_prefix("--model=") {
            model = true;
            if !path.is_empty() {
                model_path = Some(path.to_owned());
            }
        } else if arg == "--dry-run" {
            dry_run = true;
        } else if arg == "pipeline" {
            // Subcommand keyword tolerated in argument list
        } else if arg.starts_with("--") {
            return Err(format!("unknown pipeline option: {arg}"));
        } else {
            words.push(arg.clone());
        }
        i += 1;
    }

    let utterance = if words.is_empty() {
        None
    } else {
        Some(words.join(" "))
    };

    Ok(PipelineArgs {
        model,
        model_path,
        dry_run,
        utterance,
    })
}

/// Unit tests for the pipeline glue.
#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use nlu::{ModelCall, ModelReply, ScriptedModel};
    use reach_core::{Direction, Executor, Intent};
    use serde_json::{Value, json};

    use super::*;

    /// A fake executor that records every intent it is asked to run.
    #[derive(Debug, Default)]
    struct RecordingExecutor {
        /// Intents passed to `run`, in order.
        intents: Vec<Intent>,
    }

    impl Executor for RecordingExecutor {
        /// Records `intent` and reports success.
        fn run(&mut self, intent: &Intent) -> Result<String, String> {
            self.intents.push(intent.clone());
            Ok("executed".to_owned())
        }
    }

    /// Builds a single-call model reply with medium confidence.
    fn reply(name: &str, arguments: Value) -> ModelReply {
        ModelReply {
            calls: vec![ModelCall {
                name: name.to_owned(),
                arguments,
            }],
            confidence: Some(0.5),
            truncated: false,
        }
    }

    /// Runs one transcript through a fresh pipeline and returns its output and
    /// the intents the fake executor recorded.
    fn run_one<M: IntentModel + Send + 'static>(
        model: M,
        transcript: &str,
        dry_run: bool,
        answer: &str,
    ) -> (String, Vec<Intent>) {
        let mut input = Cursor::new(answer.as_bytes().to_vec());
        let mut output = Vec::new();
        let mut pipeline = Pipeline::new(model, RecordingExecutor::default(), dry_run);
        {
            let mut io = PipelineIo::new(&mut input, &mut output);
            pipeline
                .process(transcript, Instant::now(), &mut io)
                .unwrap();
        }
        (
            String::from_utf8(output).unwrap(),
            pipeline.executor().intents.clone(),
        )
    }

    /// A rule match runs through the executor without consulting the model.
    #[test]
    fn rule_match_runs_without_the_model() {
        let (output, intents) = run_one(
            ScriptedModel::new(Vec::new()),
            "Reach scroll down",
            false,
            "",
        );
        assert_eq!(
            intents,
            vec![Intent::Scroll {
                direction: Direction::Down,
                amount: None,
            }]
        );
        assert!(output.contains("executed"));
    }

    /// Speech without the wake word is ignored before routing.
    #[test]
    fn ignored_speech_does_not_reach_the_model() {
        let (output, intents) = run_one(ScriptedModel::new(Vec::new()), "scroll down", false, "");
        assert!(intents.is_empty());
        assert!(output.contains("ignored"));
    }

    /// A bare wake word opens the window for the next line.
    #[test]
    fn bare_wake_word_opens_the_window() {
        let mut input = Cursor::new(Vec::new());
        let mut output = Vec::new();
        let mut pipeline = Pipeline::new(
            ScriptedModel::new(Vec::new()),
            RecordingExecutor::default(),
            false,
        );
        {
            let mut io = PipelineIo::new(&mut input, &mut output);
            pipeline.process("Reach", Instant::now(), &mut io).unwrap();
            pipeline
                .process("scroll down", Instant::now(), &mut io)
                .unwrap();
        }
        assert_eq!(
            pipeline.executor().intents,
            vec![Intent::Scroll {
                direction: Direction::Down,
                amount: None,
            }]
        );
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains("listening"));
        assert!(text.contains("executed"));
    }

    /// Commands with fewer than 2 words are not sent to the model and ask for clarify.
    /// A slow model call times out after the configured timeout, returns Clarify, and ignores late results.
    #[test]
    fn model_timeout_returns_clarify_and_ignores_late_result() {
        use std::io::Cursor;

        struct SlowModel;
        impl IntentModel for SlowModel {
            fn infer(
                &mut self,
                _utterance: &str,
                _tools: &[ToolSpec],
            ) -> Result<ModelReply, ModelError> {
                thread::sleep(Duration::from_millis(150));
                Ok(reply("scroll", serde_json::json!({ "direction": "down" })))
            }
        }

        let mut pipeline = Pipeline::new(SlowModel, RecordingExecutor::default(), false);
        pipeline.set_model_timeout(Duration::from_millis(40));

        let mut input = Cursor::new(Vec::new());
        let mut output = Vec::new();
        let mut io = PipelineIo::new(&mut input, &mut output);

        // This command should time out and return clarify without executing
        pipeline
            .process("Reach tell me a joke", Instant::now(), &mut io)
            .unwrap();
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains(nlu::CLARIFY_QUESTION));
        assert!(pipeline.executor().intents.is_empty());

        // Wait for the slow model to finish in the background
        thread::sleep(Duration::from_millis(150));

        // Next command with a fast rule match or fast query should work cleanly
        let mut output2 = Vec::new();
        let mut io2 = PipelineIo::new(&mut input, &mut output2);
        pipeline
            .process("Reach scroll down", Instant::now(), &mut io2)
            .unwrap();
        let text2 = String::from_utf8(output2).unwrap();
        assert!(text2.contains("executed"));
    }

    #[test]
    fn command_with_fewer_than_two_words_is_not_sent_to_model() {
        // ScriptedModel with NO replies: if queried it would return ModelError::ScriptExhausted (ModelFailed).
        let model = ScriptedModel::new(Vec::new());
        let (output, intents) = run_one(model, "Reach whatever", false, "");
        assert!(output.contains(nlu::CLARIFY_QUESTION));
        assert!(!output.contains("ModelFailed"));
        assert!(intents.is_empty());
    }

    /// Commands with more than 15 words are not sent to the model and ask for clarify.
    #[test]
    fn command_with_more_than_fifteen_words_is_not_sent_to_model() {
        // 16 words in the command:
        let long_command = "Reach one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen";
        let model = ScriptedModel::new(Vec::new());
        let (output, intents) = run_one(model, long_command, false, "");
        assert!(output.contains(nlu::CLARIFY_QUESTION));
        assert!(!output.contains("ModelFailed"));
        assert!(intents.is_empty());
    }

    /// Without --model, a rules miss asks the user to rephrase.
    #[test]
    fn rules_miss_without_model_prints_clarify() {
        let (output, intents) = run_one(CliModel::Offline, "Reach tell me a joke", false, "");
        assert!(output.contains(nlu::CLARIFY_QUESTION));
        assert!(intents.is_empty());
    }

    /// With a model, a grounded free-form call runs.
    #[test]
    fn rules_miss_with_model_runs_a_grounded_call() {
        let model = ScriptedModel::new(vec![reply(
            "scroll",
            json!({ "direction": "down", "amount": 3 }),
        )]);
        let (output, intents) = run_one(
            model,
            "Reach scroll the page downward three times",
            false,
            "",
        );
        assert_eq!(
            intents,
            vec![Intent::Scroll {
                direction: Direction::Down,
                amount: Some(3),
            }]
        );
        assert!(output.contains("executed"));
    }

    /// An ungrounded call is rejected without printing the transcript or the
    /// model's invented arguments.
    #[test]
    fn ungrounded_call_reveals_only_the_reject_reason() {
        let model = ScriptedModel::new(vec![reply("click_element", json!({ "target": "submit" }))]);
        let (output, intents) = run_one(model, "Reach tell me a joke", false, "");
        assert!(intents.is_empty());
        assert!(output.contains("Ungrounded"));
        assert!(!output.contains("submit"));
        assert!(!output.contains("joke"));
    }

    /// A model error is reported as its reason type only.
    #[test]
    fn model_error_reveals_only_the_reject_reason() {
        let model = ScriptedModel::from_results(vec![Err(ModelError::NoTools)]);
        let (output, intents) = run_one(model, "Reach tell me a joke", false, "");
        assert!(intents.is_empty());
        assert!(output.contains("ModelFailed"));
        assert!(!output.contains("joke"));
    }

    /// A risky rule match waits for confirmation; "yes" runs it.
    #[test]
    fn confirm_yes_executes() {
        let (output, intents) = run_one(
            ScriptedModel::new(Vec::new()),
            "Reach click Submit",
            false,
            "yes\n",
        );
        assert_eq!(
            intents,
            vec![Intent::ClickElement {
                target: "Submit".to_owned(),
            }]
        );
        assert!(output.contains("click_element"));
        assert!(output.contains("executed"));
    }

    /// A risky rule match waits for confirmation; "no" does not run it.
    #[test]
    fn confirm_no_does_not_execute() {
        let (output, intents) = run_one(
            ScriptedModel::new(Vec::new()),
            "Reach click Submit",
            false,
            "no\n",
        );
        assert!(intents.is_empty());
        assert!(output.contains("cancelled"));
    }

    /// An answer outside the rule grammar's Confirm/Deny never runs the action.
    #[test]
    fn confirm_with_an_unrecognised_answer_does_not_execute() {
        let (output, intents) = run_one(
            ScriptedModel::new(Vec::new()),
            "Reach click Submit",
            false,
            "maybe\n",
        );
        assert!(intents.is_empty());
        assert!(output.contains(nlu::CLARIFY_QUESTION));
    }

    /// Dry run prints the route and intent as JSON and never executes.
    #[test]
    fn dry_run_prints_json_without_executing() {
        let (output, intents) = run_one(
            ScriptedModel::new(Vec::new()),
            "Reach click Submit",
            true,
            "yes\n",
        );
        assert!(intents.is_empty());
        let value: Value = serde_json::from_str(output.trim()).unwrap();
        assert_eq!(value["route"], "confirm");
        assert_eq!(value["intent"]["action"], "click_element");
        assert_eq!(value["intent"]["target"], "Submit");
        assert!(!output.contains("Run this action"));

        let (output, intents) = run_one(
            ScriptedModel::new(Vec::new()),
            "Reach scroll down",
            true,
            "",
        );
        assert!(intents.is_empty());
        let value: Value = serde_json::from_str(output.trim()).unwrap();
        assert_eq!(value["route"], "run");
        assert_eq!(value["intent"]["action"], "scroll");
    }

    /// A dry-run rejection reports only the reason type, never the transcript
    /// or the model's arguments.
    #[test]
    fn dry_run_rejection_reports_only_the_reason() {
        let model = ScriptedModel::new(vec![reply("click_element", json!({ "target": "submit" }))]);
        let (output, intents) = run_one(model, "Reach tell me a joke", true, "");
        assert!(intents.is_empty());
        let value: Value = serde_json::from_str(output.trim()).unwrap();
        assert_eq!(value["route"], "reject");
        assert_eq!(value["reason"], "Ungrounded");
        assert!(value["intent"].is_null());
        assert!(!output.contains("submit"));
        assert!(!output.contains("joke"));
    }

    /// The argument parser accepts both flags and joins the spoken words.
    #[test]
    fn parses_flags_and_joins_the_utterance() {
        let args = vec![
            "--model".to_owned(),
            "--dry-run".to_owned(),
            "Reach".to_owned(),
            "click".to_owned(),
            "Submit".to_owned(),
        ];
        let parsed = parse_pipeline_args(&args).unwrap();
        assert!(parsed.model);
        assert!(parsed.dry_run);
        assert_eq!(parsed.utterance.as_deref(), Some("Reach click Submit"));
    }

    /// The argument parser supports an explicit model path.
    #[test]
    fn parses_explicit_model_path_and_dry_run() {
        let args = vec![
            "--dry-run".to_owned(),
            "--model".to_owned(),
            "models/needle3.cact".to_owned(),
        ];
        let parsed = parse_pipeline_args(&args).unwrap();
        assert!(parsed.model);
        assert_eq!(parsed.model_path.as_deref(), Some("models/needle3.cact"));
        assert!(parsed.dry_run);
        assert_eq!(parsed.utterance, None);
    }

    /// No utterance means interactive mode with both flags off.
    #[test]
    fn empty_arguments_mean_interactive_mode() {
        let parsed = parse_pipeline_args(&[]).unwrap();
        assert!(!parsed.model);
        assert!(!parsed.dry_run);
        assert_eq!(parsed.utterance, None);
    }

    /// Unknown flags are rejected instead of becoming part of the transcript.
    #[test]
    fn unknown_flag_is_rejected() {
        assert!(parse_pipeline_args(&["--mic".to_owned()]).is_err());
    }

    /// Process utterance returns structured step outcome.
    #[test]
    fn process_utterance_step_outcome_behavior() {
        let mut input = Cursor::new(b"yes\n");
        let mut output = Vec::new();
        let mut pipeline = Pipeline::new(
            ScriptedModel::new(Vec::new()),
            RecordingExecutor::default(),
            false,
        );
        let outcome = pipeline
            .process_utterance(
                "Reach click Submit",
                Instant::now(),
                &mut input,
                &mut output,
            )
            .unwrap();
        assert_eq!(
            outcome,
            StepOutcome::Handled {
                route: Route::Confirm(Intent::ClickElement {
                    target: "Submit".to_owned()
                }),
                chosen_intent: Some(Intent::ClickElement {
                    target: "Submit".to_owned()
                }),
                executed: true,
            }
        );
        assert_eq!(
            pipeline.executor().intents,
            vec![Intent::ClickElement {
                target: "Submit".to_owned()
            }]
        );
    }
}
