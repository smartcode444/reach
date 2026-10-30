//! Command-line test harness for Reach.
//!
//! `cli pipeline` runs typed text through the same wake, routing and execution
//! path the microphone will use. The older direct-utterance mode remains for
//! quick rule-grammar checks.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod config;
pub mod executor;
pub mod listen;
pub mod pipeline;
pub mod speaker;

use std::io;
use std::time::Instant;

pub use executor::CompositeExecutor;
use nlu::{IntentModel, NeedleConfig, NeedleIntentModel};
use pipeline::{CliModel, Pipeline, PipelineIo};
use reach_core::{Executor, Intent};

/// The model file loaded when `--model` is passed.
const DEFAULT_MODEL_PATH: &str = "models/needle3.cact";

/// Entry point of the test harness.
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(error) = run(&args) {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

/// Runs the requested CLI mode.
///
/// # Arguments
/// * `args` - The process arguments after the binary name.
///
/// # Returns
/// `Ok(())` when the mode finished successfully.
///
/// # Errors
/// Returns a message for an unknown pipeline option, a model or executor load
/// failure, or an I/O failure.
fn run(args: &[String]) -> Result<(), String> {
    if args.first().map(String::as_str) == Some("listen") {
        return listen::run(&args[1..]);
    }
    if args.first().map(String::as_str) == Some("pipeline") {
        return run_pipeline(&args[1..]);
    }
    if args
        .iter()
        .any(|arg| arg == "--model" || arg == "--dry-run" || arg.starts_with("--model="))
    {
        return run_pipeline(args);
    }
    run_legacy(args)
}

/// Runs the original direct-utterance mode.
///
/// Supports rule-grammar utterances, `--say` commands, and JSON intents.
///
/// # Arguments
/// * `args` - Words to parse as one rule-grammar utterance or JSON intent.
///
/// # Returns
/// `Ok(())` after the intent runs or the help line is printed.
///
/// # Errors
/// Returns an executor initialization error, an execution error, or a parse
/// error that never includes the transcript.
fn run_legacy(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        println!("reach cli");
        return Ok(());
    }

    if args.first().map(String::as_str) == Some("--say") {
        let utterance = args[1..].join(" ");
        let mut executor = CompositeExecutor::new()?;
        let intent = nlu::parse_rule(&utterance)
            .ok_or_else(|| "could not parse utterance into intent".to_owned())?;
        let message = executor.run(&intent)?;
        println!("{message}");
        return Ok(());
    }

    let input = args.join(" ");
    let mut executor = CompositeExecutor::new()?;

    // Try deserializing as a raw JSON intent first:
    if let Ok(intent) = serde_json::from_str::<reach_core::Intent>(&input) {
        let message = executor.run(&intent)?;
        println!("{message}");
        return Ok(());
    }

    let intent = nlu::parse_rule(&input)
        .ok_or_else(|| "could not parse utterance into intent".to_owned())?;
    let message = executor.run(&intent)?;
    println!("{message}");
    Ok(())
}

/// Runs the typed-text pipeline once or interactively.
///
/// # Arguments
/// * `args` - Everything after the `pipeline` subcommand.
///
/// # Returns
/// `Ok(())` after the single transcript or every interactive line has been
/// processed.
///
/// # Errors
/// Returns a message for an unknown option, a model load failure, an executor
/// initialization failure, or an I/O failure.
fn run_pipeline(args: &[String]) -> Result<(), String> {
    let parsed = pipeline::parse_pipeline_args(args)?;

    // The model is loaded once, before the first line, so interactive runs do
    // not pay the load cost again.
    let model = load_cli_model(parsed.model, parsed.model_path.as_deref())?;

    let executor = if parsed.dry_run {
        CliExecutor::DryRun
    } else {
        CliExecutor::Real(CompositeExecutor::new()?)
    };

    let mut pipeline = Pipeline::new(model, executor, parsed.dry_run);

    match parsed.utterance {
        Some(utterance) => run_once(&mut pipeline, &utterance),
        None => run_interactive(&mut pipeline),
    }
}

/// Loads the selected intent model, or returns the offline stand-in.
///
/// # Arguments
/// * `model` - Whether the user asked for the Needle model.
/// * `model_path` - An optional explicit model path; the default is used when
///   it is absent.
///
/// # Returns
/// [`CliModel::Needle`] when `model` is true, otherwise [`CliModel::Offline`].
///
/// # Errors
/// Returns a load error when the model file cannot be opened or initialised.
pub(crate) fn load_cli_model(model: bool, model_path: Option<&str>) -> Result<CliModel, String> {
    load_cli_model_with_config(model, model_path, None)
}

/// Loads the selected intent model with layer configuration, or returns the offline stand-in.
///
/// # Arguments
/// * `model` - Whether the user asked for the Needle model.
/// * `model_path` - An optional explicit model path; the default is used when
///   it is absent.
/// * `layers` - Optional transformer layer count.
///
/// # Returns
/// [`CliModel::Needle`] when `model` is true, otherwise [`CliModel::Offline`].
///
/// # Errors
/// Returns a load error when the model file cannot be opened or initialised.
pub(crate) fn load_cli_model_with_config(
    model: bool,
    model_path: Option<&str>,
    layers: Option<usize>,
) -> Result<CliModel, String> {
    if !model {
        return Ok(CliModel::Offline);
    }
    let path = model_path.unwrap_or(DEFAULT_MODEL_PATH);
    let mut needle_config = NeedleConfig::default();
    if let Some(l) = layers {
        needle_config.layers = Some(l);
    }
    let model = NeedleIntentModel::load(path, needle_config)
        .map_err(|error| format!("failed to load Needle model: {error}"))?;
    Ok(CliModel::Needle(Box::new(model)))
}

/// The executor chosen for one CLI run.
pub(crate) enum CliExecutor {
    /// The real desktop executor.
    Real(CompositeExecutor),
    /// A placeholder used by dry runs; it refuses to execute.
    DryRun,
}

impl Executor for CliExecutor {
    /// Runs the real executor, or refuses in dry-run mode.
    ///
    /// # Arguments
    /// * `intent` - The intent to run.
    ///
    /// # Returns
    /// The real executor's message.
    ///
    /// # Errors
    /// Returns the real executor's error, or a refusal in dry-run mode. A dry
    /// run never reaches this method because the pipeline prints the route
    /// instead.
    fn run(&mut self, intent: &Intent) -> Result<String, String> {
        match self {
            Self::Real(executor) => executor.run(intent),
            Self::DryRun => Err("dry run must not execute".to_owned()),
        }
    }
}

/// Runs one transcript and writes its result to stdout.
///
/// # Arguments
/// * `pipeline` - The configured pipeline.
/// * `utterance` - The typed transcript, including the wake word.
///
/// # Returns
/// `Ok(())` after the result is written.
///
/// # Errors
/// Returns an I/O error from the pipeline streams.
fn run_once<M: IntentModel + Send + 'static, E: Executor>(
    pipeline: &mut Pipeline<M, E>,
    utterance: &str,
) -> Result<(), String> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut input = stdin.lock();
    let mut output = stdout.lock();
    let mut io = PipelineIo::new(&mut input, &mut output);
    pipeline
        .process(utterance, Instant::now(), &mut io)
        .map_err(|error| format!("pipeline I/O failed: {error}"))
}

/// Reads typed transcripts from stdin until end of input.
///
/// # Arguments
/// * `pipeline` - The configured pipeline, reused so the wake window and model
///   survive between lines.
///
/// # Returns
/// `Ok(())` at end of input.
///
/// # Errors
/// Returns an I/O error from the prompt, the stdin read or a pipeline result.
fn run_interactive<M: IntentModel + Send + 'static, E: Executor>(
    pipeline: &mut Pipeline<M, E>,
) -> Result<(), String> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut input = stdin.lock();
    let mut output = stdout.lock();
    let mut io = PipelineIo::new(&mut input, &mut output);

    loop {
        write!(io.output, "reach> ").map_err(|error| format!("pipeline I/O failed: {error}"))?;
        io.output
            .flush()
            .map_err(|error| format!("pipeline I/O failed: {error}"))?;

        let mut line = String::new();
        let read = io
            .input
            .read_line(&mut line)
            .map_err(|error| format!("pipeline I/O failed: {error}"))?;
        if read == 0 {
            break;
        }

        let transcript = line.trim();
        if transcript.is_empty() {
            continue;
        }

        pipeline
            .process(transcript, Instant::now(), &mut io)
            .map_err(|error| format!("pipeline I/O failed: {error}"))?;
    }

    Ok(())
}
