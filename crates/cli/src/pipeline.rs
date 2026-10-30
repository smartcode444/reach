//! Typed-text front end for the shared voice loop.
//!
//! The wake, routing, confirmation and execution path lives in `reach-runtime`
//! so the CLI and the desktop app cannot drift apart. This module keeps only
//! what is specific to the test harness: argument parsing and the JSON output
//! used by `--dry-run`.

use std::io::{self, Write};

use nlu::CLARIFY_QUESTION;
use reach_runtime::pipeline::StepOutcome;
use serde_json::{Map, Value};

pub use reach_runtime::pipeline::{
    DisabledModel, Pipeline, PipelineIo, PipelineModel, format_intent_action,
};

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

/// Serialises a route and its intent for `--dry-run`.
///
/// # Arguments
/// * `route` - The decision to report.
///
/// # Returns
/// One JSON object with a `route` field and, where one exists, the chosen
/// intent. A rejection carries only the reason type, never the transcript or
/// the model's arguments.
pub fn route_json(route: &nlu::Route) -> String {
    let mut object = Map::new();
    let route_name = match route {
        nlu::Route::Run(_) => "run",
        nlu::Route::Confirm(_) => "confirm",
        nlu::Route::Clarify => "clarify",
        nlu::Route::Reject(_) => "reject",
    };
    object.insert("route".to_owned(), Value::String(route_name.to_owned()));

    match route {
        nlu::Route::Run(intent) | nlu::Route::Confirm(intent) => {
            object.insert(
                "intent".to_owned(),
                serde_json::to_value(intent).unwrap_or(Value::Null),
            );
        }
        nlu::Route::Clarify => {
            object.insert("intent".to_owned(), Value::Null);
        }
        nlu::Route::Reject(reason) => {
            object.insert("reason".to_owned(), Value::String(format!("{reason:?}")));
            object.insert("intent".to_owned(), Value::Null);
        }
    }

    Value::Object(object).to_string()
}

/// Writes the dry-run JSON for one outcome.
///
/// The shared pipeline never prints this JSON itself, because `reach-runtime`
/// deliberately has no serde dependency. The CLI keeps the formatting here.
///
/// # Arguments
/// * `outcome` - The result returned by [`Pipeline::process`].
/// * `output` - Stream that receives the JSON line.
///
/// # Returns
/// `Ok(())` when the line was written, or when the outcome was not a routed
/// command.
///
/// # Errors
/// Returns an I/O error when writing fails.
pub fn print_dry_run(outcome: &StepOutcome, output: &mut dyn Write) -> io::Result<()> {
    if let StepOutcome::Handled { route, .. } = outcome {
        writeln!(output, "{}", route_json(route))?;
    }
    Ok(())
}

/// The question the CLI prints for a clarification.
///
/// Kept here so the CLI output stays identical even if the shared pipeline
/// changes how it reports the event.
pub const CLARIFY_TEXT: &str = CLARIFY_QUESTION;

/// Unit tests for the CLI-specific pipeline glue and shared pipeline behaviour.
#[cfg(test)]
mod tests {
    use std::io::{Cursor, Write};
    use std::time::Instant;

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
    fn run_one<M: nlu::IntentModel>(
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
            let outcome = pipeline
                .process(transcript, Instant::now(), &mut io)
                .unwrap();
            if dry_run {
                print_dry_run(&outcome, io.output).unwrap();
            }
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
        let (output, intents) = run_one(
            PipelineModel::Disabled(DisabledModel),
            "Reach tell me a joke",
            false,
            "",
        );
        assert!(output.contains(CLARIFY_TEXT));
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
        let model = ScriptedModel::from_results(vec![Err(nlu::ModelError::NoTools)]);
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
        assert!(output.contains("click element 'Submit'"));
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
        assert!(output.contains(CLARIFY_TEXT));
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
                route: nlu::Route::Confirm(Intent::ClickElement {
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

    /// The dry-run helper writes nothing for a line that was not routed.
    #[test]
    fn dry_run_helper_ignores_non_commands() {
        let mut output = Vec::new();
        print_dry_run(&StepOutcome::Ignored, &mut output).unwrap();
        assert!(output.is_empty());
        output.flush().unwrap();
    }
}
