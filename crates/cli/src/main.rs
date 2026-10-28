//! Command-line test harness for Reach.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod executor;

pub use executor::CompositeExecutor;
use reach_core::Executor;

/// Entry point of the test harness.
fn main() {
    let mut exec = match CompositeExecutor::new() {
        Ok(e) => e,
        Err(err) => {
            eprintln!("failed to initialize executor: {err}");
            std::process::exit(1);
        }
    };

    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        println!("reach cli");
        return;
    }

    let input = args.join(" ");
    if let Some(intent) = nlu::parse_rule(&input) {
        match exec.run(&intent) {
            Ok(msg) => println!("{msg}"),
            Err(err) => eprintln!("execution failed: {err}"),
        }
    } else {
        eprintln!("could not parse utterance into intent: {input}");
    }
}
