//! Command-line test harness for Reach.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod executor;
pub mod pipeline;

use std::time::Instant;

pub use executor::CompositeExecutor;
use pipeline::{DisabledModel, Pipeline, PipelineArgs, PipelineModel};
use reach_core::Executor;

/// Dispatches command-line arguments to the pipeline or single-intent execution.
///
/// # Arguments
/// * `args` - Command-line arguments excluding binary name.
///
/// # Returns
/// `Ok(())` on success.
///
/// # Errors
/// Returns an error message if model loading, execution, or pipeline I/O fails.
fn run_cli(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        println!("reach cli");
        return Ok(());
    }

    if let Some(first) = args.first() {
        if first == "pipeline" {
            let parsed = PipelineArgs::parse(&args[1..]);
            let model = if parsed.model_enabled {
                let needle = nlu::NeedleIntentModel::load(
                    "models/needle3.cact",
                    nlu::NeedleConfig::default(),
                )
                .map_err(|e| format!("failed to load needle model: {e}"))?;
                PipelineModel::Needle(Box::new(needle))
            } else {
                PipelineModel::Disabled(DisabledModel)
            };

            let exec = CompositeExecutor::new()?;
            let mut pipe = Pipeline::new(model, exec, parsed.dry_run);

            let stdin = std::io::stdin();
            let mut reader = stdin.lock();
            let stdout = std::io::stdout();
            let mut writer = stdout.lock();

            if let Some(ref utterance) = parsed.utterance {
                pipe.process_utterance(utterance, Instant::now(), &mut reader, &mut writer)?;
            } else {
                pipe.run_interactive(&mut reader, &mut writer)?;
            }
            return Ok(());
        }

        if first == "--say" {
            let utterance = args[1..].join(" ");
            let mut exec = CompositeExecutor::new()?;
            if let Some(intent) = nlu::parse_rule(&utterance) {
                match exec.run(&intent) {
                    Ok(msg) => println!("{msg}"),
                    Err(err) => eprintln!("execution failed: {err}"),
                }
            } else {
                eprintln!("could not parse utterance into intent: {utterance}");
            }
            return Ok(());
        }
    }

    let input = args.join(" ");
    let mut exec = CompositeExecutor::new()?;

    // Try deserializing as a JSON intent first:
    if let Ok(intent) = serde_json::from_str::<reach_core::Intent>(&input) {
        match exec.run(&intent) {
            Ok(msg) => println!("{msg}"),
            Err(err) => eprintln!("execution failed: {err}"),
        }
        return Ok(());
    }

    // Fall back to rule grammar:
    if let Some(intent) = nlu::parse_rule(&input) {
        match exec.run(&intent) {
            Ok(msg) => println!("{msg}"),
            Err(err) => eprintln!("execution failed: {err}"),
        }
    } else {
        eprintln!("could not parse utterance into intent: {input}");
    }

    Ok(())
}

/// Entry point of the test harness.
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(err) = run_cli(&args) {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
