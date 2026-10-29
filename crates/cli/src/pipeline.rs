//! Typed-text pipeline testing Reach without a microphone.
//!
//! Provides the complete orchestration:
//! `WakeSession` (gate and listening window) -> `route_utterance` (rules fast-path,
//! falling back to `IntentModel`) -> `Route` matching:
//! - `Run`: executes via [`reach_core::Executor`].
//! - `Confirm`: prints the action and prompts on stdin (parsed with rule grammar only).
//! - `Clarify`: prints the clarification question.
//! - `Reject`: prints only the rejection reason type.
//!
//! When `dry_run` is enabled, routes and chosen intents are printed as JSON without executing.

use std::io::{BufRead, Write};
use std::time::Instant;

use nlu::model::{IntentModel, ModelError, ModelReply};
use nlu::tools::ToolSpec;
use nlu::wake::{GateResult, WakeConfig, WakeSession};
use nlu::{CLARIFY_QUESTION, Route, route_utterance};
use reach_core::{Executor, Intent};

/// Parsed arguments for the `pipeline` subcommand.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct PipelineArgs {
    /// Whether the Needle model path is enabled.
    pub model_enabled: bool,
    /// Whether dry-run JSON serialization is enabled.
    pub dry_run: bool,
    /// Optional single utterance; if `None`, interactive mode is used.
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
        let mut model_enabled = false;
        let mut dry_run = false;
        let mut words = Vec::new();

        for token in tokens {
            match token.as_str() {
                "--model" => model_enabled = true,
                "--dry-run" => dry_run = true,
                other => words.push(other),
            }
        }

        let utterance = if words.is_empty() {
            None
        } else {
            Some(words.join(" "))
        };

        Self {
            model_enabled,
            dry_run,
            utterance,
        }
    }
}

/// A disabled model that abstains on every utterance.
///
/// Used when the `--model` flag is not specified. When passed to
/// [`nlu::route_utterance`], an abstention on rule misses produces
/// [`Route::Clarify`].
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

/// Orchestrates wake-word gating, grammar/model routing, confirmation, and execution.
pub struct Pipeline<M: IntentModel, E: Executor> {
    /// Wake-word gate and listening window state machine.
    session: WakeSession,
    /// Intent model consulted when the rule grammar has no match.
    model: M,
    /// Concrete executor for performing actions.
    executor: E,
    /// If true, routes and intents are serialized as JSON instead of executing.
    dry_run: bool,
}

impl<M: IntentModel, E: Executor> Pipeline<M, E> {
    /// Creates a new pipeline instance.
    ///
    /// # Arguments
    /// * `model` - The intent model to use on rule misses.
    /// * `executor` - The executor to run accepted intents.
    /// * `dry_run` - If true, actions are printed as JSON without executing.
    ///
    /// # Returns
    /// An initialized `Pipeline`.
    pub fn new(model: M, executor: E, dry_run: bool) -> Self {
        Self {
            session: WakeSession::new(WakeConfig::default()),
            model,
            executor,
            dry_run,
        }
    }

    /// Returns a reference to the inner executor.
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

    /// Processes a single utterance line at the given timestamp.
    ///
    /// # Arguments
    /// * `utterance` - Spoken or typed line of text.
    /// * `now` - Monotonic clock timestamp for listening window state.
    /// * `reader` - Input reader for reading confirmation (e.g. stdin).
    /// * `writer` - Output writer for captions and prompts (e.g. stdout).
    ///
    /// # Returns
    /// The [`StepOutcome`] indicating how the line was handled.
    ///
    /// # Errors
    /// Returns an error message if reading confirmation or writing output fails.
    pub fn process_utterance<R: BufRead, W: Write>(
        &mut self,
        utterance: &str,
        now: Instant,
        reader: &mut R,
        writer: &mut W,
    ) -> Result<StepOutcome, String> {
        let gate_result = self.session.evaluate(utterance, now);
        let command = match gate_result {
            GateResult::Ignored => return Ok(StepOutcome::Ignored),
            GateResult::WakeOnly => return Ok(StepOutcome::WakeOnly),
            GateResult::Command(command) => command,
        };

        let route = route_utterance(&command, &mut self.model);
        let chosen_intent = match &route {
            Route::Run(intent) | Route::Confirm(intent) => Some(intent.clone()),
            _ => None,
        };

        if self.dry_run {
            let route_val = route_to_json(&route);
            let intent_val = chosen_intent
                .as_ref()
                .map(|i| serde_json::to_value(i).unwrap_or(serde_json::Value::Null))
                .unwrap_or(serde_json::Value::Null);

            let route_json = serde_json::to_string_pretty(&route_val).map_err(|e| e.to_string())?;
            let intent_json =
                serde_json::to_string_pretty(&intent_val).map_err(|e| e.to_string())?;

            writeln!(writer, "{route_json}").map_err(|e| e.to_string())?;
            writeln!(writer, "{intent_json}").map_err(|e| e.to_string())?;

            return Ok(StepOutcome::Handled {
                route,
                chosen_intent,
                executed: false,
            });
        }

        match &route {
            Route::Run(intent) => match self.executor.run(intent) {
                Ok(msg) => {
                    writeln!(writer, "{msg}").map_err(|e| e.to_string())?;
                    Ok(StepOutcome::Handled {
                        route,
                        chosen_intent,
                        executed: true,
                    })
                }
                Err(err) => {
                    writeln!(writer, "execution failed: {err}").map_err(|e| e.to_string())?;
                    Ok(StepOutcome::Handled {
                        route,
                        chosen_intent,
                        executed: false,
                    })
                }
            },
            Route::Confirm(intent) => {
                let action_desc = format_intent_action(intent);
                writeln!(writer, "Confirm: {action_desc}? (yes/no)").map_err(|e| e.to_string())?;

                let mut answer = String::new();
                reader.read_line(&mut answer).map_err(|e| e.to_string())?;
                let answer_intent = nlu::parse_rule(&answer);

                if answer_intent == Some(Intent::Confirm) {
                    match self.executor.run(intent) {
                        Ok(msg) => {
                            writeln!(writer, "{msg}").map_err(|e| e.to_string())?;
                            Ok(StepOutcome::Handled {
                                route,
                                chosen_intent,
                                executed: true,
                            })
                        }
                        Err(err) => {
                            writeln!(writer, "execution failed: {err}")
                                .map_err(|e| e.to_string())?;
                            Ok(StepOutcome::Handled {
                                route,
                                chosen_intent,
                                executed: false,
                            })
                        }
                    }
                } else if answer_intent == Some(Intent::Deny) {
                    writeln!(writer, "Action cancelled.").map_err(|e| e.to_string())?;
                    Ok(StepOutcome::Handled {
                        route,
                        chosen_intent,
                        executed: false,
                    })
                } else {
                    writeln!(writer, "Action not confirmed.").map_err(|e| e.to_string())?;
                    Ok(StepOutcome::Handled {
                        route,
                        chosen_intent,
                        executed: false,
                    })
                }
            }
            Route::Clarify => {
                writeln!(writer, "{CLARIFY_QUESTION}").map_err(|e| e.to_string())?;
                Ok(StepOutcome::Handled {
                    route,
                    chosen_intent: None,
                    executed: false,
                })
            }
            Route::Reject(reason) => {
                writeln!(writer, "Rejected: {reason:?}").map_err(|e| e.to_string())?;
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
                self.process_utterance(trimmed, Instant::now(), reader, writer)?;
            }
            line.clear();
        }
        Ok(())
    }
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

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::time::Duration;

    use nlu::model::{ModelCall, ScriptedModel};
    use nlu::routing::RejectReason;
    use reach_core::Direction;
    use serde_json::json;

    use super::*;

    /// A fake executor that records all intents it was asked to execute.
    #[derive(Debug, Default)]
    struct RecordingExecutor {
        /// Recorded intents executed in chronological order.
        executed: Vec<Intent>,
    }

    impl Executor for RecordingExecutor {
        /// Records `intent` and returns a success message.
        fn run(&mut self, intent: &Intent) -> Result<String, String> {
            self.executed.push(intent.clone());
            Ok(format!("executed: {:?}", intent))
        }
    }

    /// Verifies that utterances without wake words are ignored when the window is closed.
    #[test]
    fn pipeline_ignores_utterance_without_wake_word() {
        let mut pipeline = Pipeline::new(DisabledModel, RecordingExecutor::default(), false);
        let mut input = Cursor::new(b"");
        let mut output = Vec::new();
        let outcome = pipeline
            .process_utterance("click submit", Instant::now(), &mut input, &mut output)
            .expect("process utterance failed");
        assert_eq!(outcome, StepOutcome::Ignored);
        assert!(pipeline.executor().executed.is_empty());
    }

    /// Verifies that speaking solely the wake word opens the listening window.
    #[test]
    fn pipeline_opens_window_on_wake_only() {
        let mut pipeline = Pipeline::new(DisabledModel, RecordingExecutor::default(), false);
        let mut input = Cursor::new(b"");
        let mut output = Vec::new();
        let outcome = pipeline
            .process_utterance("Reach", Instant::now(), &mut input, &mut output)
            .expect("process utterance failed");
        assert_eq!(outcome, StepOutcome::WakeOnly);
        assert!(pipeline.executor().executed.is_empty());
    }

    /// Verifies that commands spoken while the listening window is open are accepted.
    #[test]
    fn pipeline_handles_command_in_open_wake_window() {
        let mut pipeline = Pipeline::new(DisabledModel, RecordingExecutor::default(), false);
        let t0 = Instant::now();
        let mut input = Cursor::new(b"");
        let mut output = Vec::new();

        let outcome1 = pipeline
            .process_utterance("Reach", t0, &mut input, &mut output)
            .expect("wake utterance failed");
        assert_eq!(outcome1, StepOutcome::WakeOnly);

        let t1 = t0 + Duration::from_secs(1);
        let outcome2 = pipeline
            .process_utterance("scroll down", t1, &mut input, &mut output)
            .expect("command utterance failed");
        assert_eq!(
            outcome2,
            StepOutcome::Handled {
                route: Route::Run(Intent::Scroll {
                    direction: Direction::Down,
                    amount: None,
                }),
                chosen_intent: Some(Intent::Scroll {
                    direction: Direction::Down,
                    amount: None,
                }),
                executed: true,
            }
        );
        assert_eq!(
            pipeline.executor().executed,
            vec![Intent::Scroll {
                direction: Direction::Down,
                amount: None,
            }]
        );
    }

    /// Verifies that safe rule matches run immediately without calling model or prompting.
    #[test]
    fn pipeline_routes_rule_run_without_calling_model() {
        let mut pipeline = Pipeline::new(DisabledModel, RecordingExecutor::default(), false);
        let mut input = Cursor::new(b"");
        let mut output = Vec::new();
        let outcome = pipeline
            .process_utterance("Reach scroll down", Instant::now(), &mut input, &mut output)
            .expect("process utterance failed");
        assert_eq!(
            outcome,
            StepOutcome::Handled {
                route: Route::Run(Intent::Scroll {
                    direction: Direction::Down,
                    amount: None,
                }),
                chosen_intent: Some(Intent::Scroll {
                    direction: Direction::Down,
                    amount: None,
                }),
                executed: true,
            }
        );
        assert_eq!(
            pipeline.executor().executed,
            vec![Intent::Scroll {
                direction: Direction::Down,
                amount: None,
            }]
        );
    }

    /// Verifies that risky actions prompt for confirmation and execute on affirmative "yes".
    #[test]
    fn pipeline_confirms_risky_action_when_confirmed_yes() {
        let mut pipeline = Pipeline::new(DisabledModel, RecordingExecutor::default(), false);
        let mut input = Cursor::new(b"yes\n");
        let mut output = Vec::new();
        let outcome = pipeline
            .process_utterance(
                "Reach click submit",
                Instant::now(),
                &mut input,
                &mut output,
            )
            .expect("process utterance failed");

        let expected_intent = Intent::ClickElement {
            target: "submit".to_owned(),
        };
        assert_eq!(
            outcome,
            StepOutcome::Handled {
                route: Route::Confirm(expected_intent.clone()),
                chosen_intent: Some(expected_intent.clone()),
                executed: true,
            }
        );
        assert_eq!(pipeline.executor().executed, vec![expected_intent]);
        let output_str = String::from_utf8_lossy(&output);
        assert!(output_str.contains("Confirm: click element 'submit'?"));
    }

    /// Verifies that risky actions do not execute when denied with "no".
    #[test]
    fn pipeline_aborts_risky_action_when_denied_no() {
        let mut pipeline = Pipeline::new(DisabledModel, RecordingExecutor::default(), false);
        let mut input = Cursor::new(b"no\n");
        let mut output = Vec::new();
        let outcome = pipeline
            .process_utterance(
                "Reach click submit",
                Instant::now(),
                &mut input,
                &mut output,
            )
            .expect("process utterance failed");

        let expected_intent = Intent::ClickElement {
            target: "submit".to_owned(),
        };
        assert_eq!(
            outcome,
            StepOutcome::Handled {
                route: Route::Confirm(expected_intent),
                chosen_intent: Some(Intent::ClickElement {
                    target: "submit".to_owned(),
                }),
                executed: false,
            }
        );
        assert!(pipeline.executor().executed.is_empty());
        let output_str = String::from_utf8_lossy(&output);
        assert!(output_str.contains("Action cancelled."));
    }

    /// Verifies that when model inference is disabled, a rules miss prints Clarify question.
    #[test]
    fn pipeline_clarifies_rules_miss_when_model_disabled() {
        let mut pipeline = Pipeline::new(DisabledModel, RecordingExecutor::default(), false);
        let mut input = Cursor::new(b"");
        let mut output = Vec::new();
        let outcome = pipeline
            .process_utterance(
                "Reach banana pancake",
                Instant::now(),
                &mut input,
                &mut output,
            )
            .expect("process utterance failed");

        assert_eq!(
            outcome,
            StepOutcome::Handled {
                route: Route::Clarify,
                chosen_intent: None,
                executed: false,
            }
        );
        assert!(pipeline.executor().executed.is_empty());
        let output_str = String::from_utf8_lossy(&output);
        assert!(output_str.contains(CLARIFY_QUESTION));
    }

    /// Verifies that when model inference is enabled, a rules miss delegates to the model.
    #[test]
    fn pipeline_calls_model_on_rules_miss_when_enabled() {
        let model = ScriptedModel::new(vec![ModelReply {
            calls: vec![ModelCall {
                name: "scroll".to_owned(),
                arguments: json!({ "direction": "down" }),
            }],
            confidence: Some(0.8),
            truncated: false,
        }]);
        let mut pipeline = Pipeline::new(model, RecordingExecutor::default(), false);
        let mut input = Cursor::new(b"");
        let mut output = Vec::new();

        // "Reach please scroll" wait, "please" is a filler. Let's use an ungrammatical phrasing:
        // "Reach scroll on down"
        let outcome = pipeline
            .process_utterance(
                "Reach scroll on down",
                Instant::now(),
                &mut input,
                &mut output,
            )
            .expect("process utterance failed");

        let expected_intent = Intent::Scroll {
            direction: Direction::Down,
            amount: None,
        };
        assert_eq!(
            outcome,
            StepOutcome::Handled {
                route: Route::Run(expected_intent.clone()),
                chosen_intent: Some(expected_intent.clone()),
                executed: true,
            }
        );
        assert_eq!(pipeline.executor().executed, vec![expected_intent]);
    }

    /// Verifies that ungrounded model calls are rejected, printing only the error type.
    #[test]
    fn pipeline_rejects_ungrounded_model_call_printing_only_reason_type() {
        let model = ScriptedModel::new(vec![ModelReply {
            calls: vec![ModelCall {
                name: "click_element".to_owned(),
                arguments: json!({ "target": "delete" }),
            }],
            confidence: Some(0.8),
            truncated: false,
        }]);
        let mut pipeline = Pipeline::new(model, RecordingExecutor::default(), false);
        let mut input = Cursor::new(b"");
        let mut output = Vec::new();

        // Target "delete" is not grounded in "scroll on down"
        let outcome = pipeline
            .process_utterance(
                "Reach scroll on down",
                Instant::now(),
                &mut input,
                &mut output,
            )
            .expect("process utterance failed");

        assert_eq!(
            outcome,
            StepOutcome::Handled {
                route: Route::Reject(RejectReason::Ungrounded),
                chosen_intent: None,
                executed: false,
            }
        );
        assert!(pipeline.executor().executed.is_empty());
        let output_str = String::from_utf8_lossy(&output);
        assert!(output_str.contains("Rejected: Ungrounded"));
        // Transcripts and arguments must never be in rejection output
        assert!(!output_str.contains("delete"));
        assert!(!output_str.contains("scroll on down"));
    }

    /// Verifies that dry-run mode prints Route and Intent as JSON without executing anything.
    #[test]
    fn pipeline_dry_run_prints_json_without_executing() {
        let mut pipeline = Pipeline::new(DisabledModel, RecordingExecutor::default(), true);
        let mut input = Cursor::new(b"");
        let mut output = Vec::new();
        let outcome = pipeline
            .process_utterance("Reach scroll down", Instant::now(), &mut input, &mut output)
            .expect("process utterance failed");

        let expected_intent = Intent::Scroll {
            direction: Direction::Down,
            amount: None,
        };
        assert_eq!(
            outcome,
            StepOutcome::Handled {
                route: Route::Run(expected_intent.clone()),
                chosen_intent: Some(expected_intent),
                executed: false,
            }
        );
        assert!(pipeline.executor().executed.is_empty());
        let output_str = String::from_utf8_lossy(&output);
        assert!(output_str.contains(r#""type": "run""#));
        assert!(output_str.contains(r#""action": "scroll""#));
    }

    /// Verifies that confirmation answers are evaluated with the rule grammar only.
    #[test]
    fn pipeline_confirmation_uses_rule_grammar_not_model() {
        // Model would hallucinate a deny, but rule grammar parses "yeah" as Confirm
        let mut pipeline = Pipeline::new(DisabledModel, RecordingExecutor::default(), false);
        let mut input = Cursor::new(b"yeah\n");
        let mut output = Vec::new();
        let outcome = pipeline
            .process_utterance(
                "Reach click submit",
                Instant::now(),
                &mut input,
                &mut output,
            )
            .expect("process utterance failed");

        let expected_intent = Intent::ClickElement {
            target: "submit".to_owned(),
        };
        assert_eq!(
            outcome,
            StepOutcome::Handled {
                route: Route::Confirm(expected_intent.clone()),
                chosen_intent: Some(expected_intent.clone()),
                executed: true,
            }
        );
        assert_eq!(pipeline.executor().executed, vec![expected_intent]);
    }

    /// Verifies parsing of CLI arguments for the pipeline subcommand.
    #[test]
    fn parses_pipeline_cli_arguments() {
        let empty_args: Vec<String> = vec![];
        assert_eq!(
            PipelineArgs::parse(&empty_args),
            PipelineArgs {
                model_enabled: false,
                dry_run: false,
                utterance: None,
            }
        );

        let flag_args = vec!["--model".to_string(), "--dry-run".to_string()];
        assert_eq!(
            PipelineArgs::parse(&flag_args),
            PipelineArgs {
                model_enabled: true,
                dry_run: true,
                utterance: None,
            }
        );

        let utterance_args = vec![
            "--dry-run".to_string(),
            "Reach".to_string(),
            "click".to_string(),
            "submit".to_string(),
        ];
        assert_eq!(
            PipelineArgs::parse(&utterance_args),
            PipelineArgs {
                model_enabled: false,
                dry_run: true,
                utterance: Some("Reach click submit".to_string()),
            }
        );
    }

    /// Verifies interactive line reading until quit or exit.
    #[test]
    fn pipeline_interactive_mode_stops_on_exit() {
        let mut pipeline = Pipeline::new(DisabledModel, RecordingExecutor::default(), false);
        let mut input = Cursor::new(b"Reach scroll down\nexit\nReach scroll down\n");
        let mut output = Vec::new();

        pipeline
            .run_interactive(&mut input, &mut output)
            .expect("interactive run failed");

        assert_eq!(
            pipeline.executor().executed,
            vec![Intent::Scroll {
                direction: Direction::Down,
                amount: None,
            }]
        );
    }
}
