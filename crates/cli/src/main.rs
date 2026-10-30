//! Command-line test harness for Reach.
//!
//! `cli pipeline` runs typed text through the same wake, routing and execution
//! path the microphone will use. The older direct-utterance mode remains for
//! quick rule-grammar checks.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod executor;
pub mod pipeline;

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
    if args.first().map(String::as_str) == Some("pipeline") {
        return run_pipeline(&args[1..]);
    }
    run_legacy(args)
}

/// Runs the original direct-utterance mode.
///
/// # Arguments
/// * `args` - Words to parse as one rule-grammar utterance.
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

    let mut executor = CompositeExecutor::new()?;
    let input = args.join(" ");
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
    let model = if parsed.model {
        let model = NeedleIntentModel::load(DEFAULT_MODEL_PATH, NeedleConfig::default())
            .map_err(|error| format!("failed to load Needle model: {error}"))?;
        CliModel::Needle(Box::new(model))
    } else {
        CliModel::Offline
    };

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

/// The executor chosen for one CLI run.
enum CliExecutor {
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
fn run_once<M: IntentModel, E: Executor>(
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
fn run_interactive<M: IntentModel, E: Executor>(
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
