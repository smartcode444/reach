//! Typed-text pipeline glue for the CLI harness.
//!
//! The pipeline replaces the microphone with typed input while keeping the
//! production order: the wake session first, then [`nlu::route_utterance`],
//! then one of the [`Route`] outcomes. It never re-implements routing and never
//! prints the transcript.

use std::io::{self, BufRead, Write};
use std::time::Instant;

use nlu::wake::{GateResult, WakeSession};
use nlu::{
    CLARIFY_QUESTION, IntentModel, ModelError, ModelReply, NeedleIntentModel, Route, ToolSpec,
};
use reach_core::{Executor, Intent};
use serde_json::{Map, Value};

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

/// The streams the pipeline reads answers from and writes results to.
pub struct PipelineIo<'a> {
    /// Reads confirmation answers.
    pub input: &'a mut dyn BufRead,
    /// Receives user-facing results.
    pub output: &'a mut dyn Write,
}

impl<'a> PipelineIo<'a> {
    /// Creates a pipeline I/O pair from borrowed streams.
    pub fn new(input: &'a mut dyn BufRead, output: &'a mut dyn Write) -> Self {
        Self { input, output }
    }
}

/// One typed-text pipeline with its wake session, model and executor.
pub struct Pipeline<M, E> {
    /// The wake gate and listening window shared by every line.
    session: WakeSession,
    /// The model used only when the rule grammar misses.
    model: M,
    /// The executor used only for `Run` and confirmed `Confirm` routes.
    executor: E,
    /// When true, routes are printed as JSON and nothing runs.
    dry_run: bool,
}

impl<M, E> Pipeline<M, E> {
    /// Creates a pipeline with the default wake configuration.
    pub fn new(model: M, executor: E, dry_run: bool) -> Self {
        Self {
            session: WakeSession::default(),
            model,
            executor,
            dry_run,
        }
    }

    /// Returns the executor so tests can inspect what ran.
    pub fn executor(&self) -> &E {
        &self.executor
    }
}

impl<M: IntentModel, E: Executor> Pipeline<M, E> {
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
        match self.session.evaluate(transcript, now) {
            GateResult::Ignored => writeln!(io.output, "ignored"),
            GateResult::WakeOnly => writeln!(io.output, "listening"),
            GateResult::Command(command) => {
                let route = nlu::route_utterance(&command, &mut self.model);
                self.handle_route(route, io)
            }
        }
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
            Route::Confirm(intent) => self.confirm(intent, io),
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
        match self.executor.run(intent) {
            Ok(message) => writeln!(io.output, "{message}"),
            Err(error) => writeln!(io.output, "execution failed: {error}"),
        }
    }

    /// Asks for a spoken-style yes/no answer and runs the intent only on yes.
    ///
    /// The answer is parsed by the rule grammar, never by the model, so the
    /// model can never confirm its own risky action.
    ///
    /// # Arguments
    /// * `intent` - The risky intent waiting for confirmation.
    /// * `io` - The streams used to print the action, read the answer and
    ///   write the result.
    ///
    /// # Returns
    /// `Ok(())` after the answer has been handled.
    ///
    /// # Errors
    /// Returns an I/O error when reading the answer or writing a result fails.
    fn confirm(&mut self, intent: Intent, io: &mut PipelineIo<'_>) -> io::Result<()> {
        writeln!(io.output, "confirm: {}", intent_json(&intent))?;
        write!(io.output, "Run this action? (yes/no) ")?;
        io.output.flush()?;

        let Some(answer) = read_answer(io.input)? else {
            return writeln!(io.output, "{CLARIFY_QUESTION}");
        };

        match nlu::parse_rule(answer.trim()) {
            Some(Intent::Confirm) => self.execute(&intent, io),
            Some(Intent::Deny) => writeln!(io.output, "cancelled"),
            _ => writeln!(io.output, "{CLARIFY_QUESTION}"),
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
fn intent_json(intent: &Intent) -> String {
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
fn route_json(route: &Route) -> String {
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

/// Options parsed from the `pipeline` subcommand arguments.
pub(crate) struct PipelineArgs {
    /// Whether to load and use the Needle model.
    pub(crate) model: bool,
    /// Whether to print routes as JSON instead of running them.
    pub(crate) dry_run: bool,
    /// The single transcript, or `None` for interactive stdin mode.
    pub(crate) utterance: Option<String>,
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
pub(crate) fn parse_pipeline_args(args: &[String]) -> Result<PipelineArgs, String> {
    let mut model = false;
    let mut dry_run = false;
    let mut words = Vec::new();

    for arg in args {
        match arg.as_str() {
            "--model" => model = true,
            "--dry-run" => dry_run = true,
            option if option.starts_with("--") => {
                return Err(format!("unknown pipeline option: {option}"));
            }
            word => words.push(word.to_owned()),
        }
    }

    let utterance = if words.is_empty() {
        None
    } else {
        Some(words.join(" "))
    };

    Ok(PipelineArgs {
        model,
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
    fn run_one<M: IntentModel>(
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
}
