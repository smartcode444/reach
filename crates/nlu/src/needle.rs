//! The in-process Needle 3 engine behind [`IntentModel`].
//!
//! Loading is expensive, so [`NeedleIntentModel`] loads the container once and
//! reuses it. Generation always enables constrained decoding: the model may
//! only emit the tools in [`crate::tools::tool_specs`] and their declared
//! argument keys.
//!
//! ## What was measured on the real checkpoint
//!
//! These results come from `models/needle3.cact` and the ignored integration
//! test. None of them made ambient speech abstain reliably.
//!
//! | Change | Result |
//! | --- | --- |
//! | Full container depth | Best command accuracy. 8 layers repeated tokens; 6 layers hallucinated confidently. |
//! | System prompt ("only call a tool for a command") | No effect on ambient. `confidence_for` also builds its prompt with `system: None`, so a system prompt would desynchronise the score. |
//! | Short or specific tool descriptions | No effect; `hide_numbers` was still chosen for ambient. |
//! | Empty `required` arrays on every tool | No effect. |
//! | Original order (`no_action` last) | Malformed `click_number` for "scroll down"; moving `no_action` first fixed it. |
//! | `no_action` first | Fixed both "scroll down" and "click submit"; this is the shipped order. |
//! | Alphabetical order | Did not fix ambient hallucination. |
//! | Frequency order | Ambient still produced `scroll` with direction "sales". |
//! | Removing `no_action` | Did not stop hallucination. |
//!
//! The model boundary rejects malformed, truncated and incomplete output.
//! Ambient rejection is the job of the wake gate, grounding and routing, not
//! the model.

use std::path::Path;

use needle_infer::v3_engine::{StopReason, V3Engine, V3Options, extract_tool_call};

use crate::model::{IntentModel, ModelError, ModelReply, calls_from_payload, validate_calls};
use crate::tools::{ToolSpec, tools_json};

/// Reach's generation cap.
///
/// Needle's own default is 256, but that assumes a short tool list. With the
/// fourteen Reach tools the model can spend most of 256 tokens reasoning before
/// it finishes the call; the first real-model run stopped mid-payload. This is a
/// ceiling, not a target: generation still stops at the end-of-sequence token.
const REACH_MAX_NEW_TOKENS: usize = 512;

/// Configuration for loading and running the Needle model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NeedleConfig {
    /// The number of transformer layers, or `None` for the container's depth.
    ///
    /// Needle answers become unusable below about six layers, so callers should
    /// only pass a smaller rung when the corpus proves it works.
    pub layers: Option<usize>,
    /// The hard cap on generated tokens, including the ` thinking` block.
    pub max_new_tokens: usize,
}

impl Default for NeedleConfig {
    /// Returns the full-depth model with Reach's measured token budget.
    fn default() -> Self {
        Self {
            layers: None,
            max_new_tokens: REACH_MAX_NEW_TOKENS,
        }
    }
}

/// Detailed stage timings recorded during a Needle inference pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NeedleStageTimings {
    /// Time spent building the tool JSON specification and prompt.
    pub prompt_build: std::time::Duration,
    /// Time spent generating tokens through the transformer engine.
    pub generation: std::time::Duration,
    /// Time spent evaluating the completion through the confidence head.
    pub confidence_pass: std::time::Duration,
    /// Number of tools declared in the request.
    pub tool_count: usize,
    /// Total character length of the serialized tool list JSON.
    pub tool_chars: usize,
}

/// A Needle 3 engine loaded once and reused for every utterance.
pub struct NeedleIntentModel {
    /// The loaded engine; construction is the expensive step.
    engine: V3Engine,
    /// The generation cap taken from [`NeedleConfig`].
    max_new_tokens: usize,
    /// Stage timings of the most recent inference pass.
    last_stage_timings: NeedleStageTimings,
}

impl NeedleIntentModel {
    /// Loads the model from `path` exactly once.
    ///
    /// # Arguments
    /// * `path` - The `.cact` container, normally `models/needle3.cact`.
    /// * `config` - The layer depth and generation cap.
    ///
    /// # Returns
    /// A model ready to answer [`IntentModel::infer`] calls.
    ///
    /// # Errors
    /// Returns [`ModelError::Load`] when the container, its tokenizer or its
    /// confidence head cannot be loaded.
    pub fn load(path: impl AsRef<Path>, config: NeedleConfig) -> Result<Self, ModelError> {
        let engine = match config.layers {
            Some(layers) => V3Engine::load_with_depth(path, layers)?,
            None => V3Engine::load(path)?,
        };
        Ok(Self {
            engine,
            max_new_tokens: config.max_new_tokens,
            last_stage_timings: NeedleStageTimings::default(),
        })
    }

    /// Runs constrained generation and returns the raw result and tools JSON.
    ///
    /// Keeping this separate lets the ignored integration test print the raw
    /// completion when a reply is rejected, without duplicating the options.
    ///
    /// # Errors
    /// Returns [`ModelError::NoTools`] when `tools` is empty, because
    /// constrained decoding would otherwise be off.
    /// Returns the stage timings from the most recent inference pass.
    ///
    /// # Returns
    /// A [`NeedleStageTimings`] struct holding durations for prompt build,
    /// generation, and confidence pass, plus tool count and tool list characters.
    pub fn last_stage_timings(&self) -> NeedleStageTimings {
        self.last_stage_timings
    }

    /// Runs constrained generation and returns the raw result and tools JSON.
    ///
    /// Keeping this separate lets the ignored integration test print the raw
    /// completion when a reply is rejected, without duplicating the options.
    ///
    /// # Errors
    /// Returns [`ModelError::NoTools`] when `tools` is empty, because
    /// constrained decoding would otherwise be off.
    #[cfg(test)]
    fn generate(
        &self,
        utterance: &str,
        tools: &[ToolSpec],
    ) -> Result<(needle_infer::v3_engine::V3Result, String), ModelError> {
        if tools.is_empty() {
            return Err(ModelError::NoTools);
        }

        let tools_json = tools_json(tools);
        let options = V3Options {
            max_new_tokens: self.max_new_tokens,
            constrain: true,
            ..V3Options::default()
        };
        let result = self.engine.generate(utterance, &tools_json, &options);
        Ok((result, tools_json))
    }
}

impl IntentModel for NeedleIntentModel {
    /// Runs constrained generation and converts the tool-call payload to calls.
    ///
    /// Constrained decoding is always on. A completion with no `<tool_call>`
    /// block becomes an empty call list, which the router treats as "no tool
    /// matches" rather than an instruction to guess.
    ///
    /// # Returns
    /// The calls, the model's confidence (when the container exports a head)
    /// and the prompt-truncation flag. If `truncated` is true the reply is
    /// unusable and must be discarded.
    ///
    /// # Errors
    /// Returns [`ModelError::NoTools`] when `tools` is empty (which would leave
    /// generation unconstrained), [`ModelError::Incomplete`] when the token or
    /// context budget ran out before the reply finished, [`ModelError::InvalidJson`]
    /// or [`ModelError::InvalidPayloadShape`] when the payload is malformed, and
    /// [`ModelError::UndeclaredTool`] when a call names a tool outside `tools`.
    fn infer(&mut self, utterance: &str, tools: &[ToolSpec]) -> Result<ModelReply, ModelError> {
        if tools.is_empty() {
            return Err(ModelError::NoTools);
        }

        let prompt_start = std::time::Instant::now();
        let tools_json = tools_json(tools);
        let prompt_build = prompt_start.elapsed();
        let tool_count = tools.len();
        let tool_chars = tools_json.len();

        let options = V3Options {
            max_new_tokens: self.max_new_tokens,
            constrain: true,
            ..V3Options::default()
        };

        let gen_start = std::time::Instant::now();
        let result = self.engine.generate(utterance, &tools_json, &options);
        let generation = gen_start.elapsed();

        let calls = match extract_tool_call(&result.text) {
            Some(payload) => match calls_from_payload(&payload) {
                Ok(calls) => calls,
                Err(_) if ran_out_of_budget(result.stop) => {
                    return Err(ModelError::Incomplete(result.stop));
                }
                Err(error) => return Err(error),
            },
            None if ran_out_of_budget(result.stop) && !result.prompt_truncated => {
                return Err(ModelError::Incomplete(result.stop));
            }
            None => Vec::new(),
        };
        validate_calls(&calls, tools)?;

        // Score the full completion, not the query: `confidence_for` appends
        // this string to the formatted prompt, and `run_scored` passes
        // `result.text` for the same reason.
        let conf_start = std::time::Instant::now();
        let confidence = self
            .engine
            .confidence_for(utterance, &tools_json, &result.text);
        let confidence_pass = conf_start.elapsed();

        self.last_stage_timings = NeedleStageTimings {
            prompt_build,
            generation,
            confidence_pass,
            tool_count,
            tool_chars,
        };

        Ok(ModelReply {
            calls,
            confidence,
            truncated: result.prompt_truncated,
        })
    }
}

/// Reports whether generation stopped because it hit a token or context limit.
///
/// # Arguments
/// * `stop` - The reason the engine stopped.
///
/// # Returns
/// True for [`StopReason::MaxTokens`] and [`StopReason::MaxSeqLen`].
fn ran_out_of_budget(stop: StopReason) -> bool {
    matches!(stop, StopReason::MaxTokens | StopReason::MaxSeqLen)
}

/// Tests for the real model.
///
/// These are ignored by default: they load the 35 MB container and are meant to
/// be run by hand after the model file is in place.
#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::Instant;

    use reach_core::{Direction, Intent};

    use super::*;
    use crate::tools::{ToolSpec, call_to_intent, tool_specs};

    /// Returns the repository's model file from the crate directory.
    fn model_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("models")
            .join("needle3.cact")
    }

    /// Runs inference, printing the raw completion when the reply is rejected.
    ///
    /// The extra generation only happens on failure, where seeing the stop
    /// reason and the cut-off text is worth the cost.
    fn infer_or_report(
        model: &mut NeedleIntentModel,
        utterance: &str,
        tools: &[ToolSpec],
    ) -> Result<ModelReply, ModelError> {
        match model.infer(utterance, tools) {
            Ok(reply) => Ok(reply),
            Err(error) => {
                let (raw, _) = model.generate(utterance, tools)?;
                println!(
                    "{utterance:?} rejected: {error}; stop={:?} truncated={} positions={} text={:?}",
                    raw.stop, raw.prompt_truncated, raw.positions, raw.text
                );
                Err(error)
            }
        }
    }

    /// Prints one ambient utterance's raw completion and parsed outcome.
    ///
    /// This asserts nothing on purpose: ambient rejection is done by the wake
    /// gate, grounding and routing, not by the model. A different sentence can
    /// produce a syntactically valid call, so an abstention assertion here would
    /// pass only by luck.
    ///
    /// # Arguments
    /// * `model` - The loaded model.
    /// * `utterance` - A sentence that is not a computer command.
    /// * `tools` - The same tool list used for the command assertions.
    fn report_ambient(model: &NeedleIntentModel, utterance: &str, tools: &[ToolSpec]) {
        let started = Instant::now();
        let (raw, tools_json) = match model.generate(utterance, tools) {
            Ok(result) => result,
            Err(error) => {
                println!("{utterance:?}: generation failed: {error}");
                return;
            }
        };
        let elapsed = started.elapsed();
        let payload = extract_tool_call(&raw.text);
        let outcome = match payload.as_deref() {
            Some(payload) => match calls_from_payload(payload) {
                Ok(calls) => match validate_calls(&calls, tools) {
                    Ok(()) => format!("{calls:?}"),
                    Err(error) => format!("rejected: {error}"),
                },
                Err(error) => format!("rejected: {error}"),
            },
            None => "no <tool_call> block".to_owned(),
        };
        let confidence = model
            .engine
            .confidence_for(utterance, &tools_json, &raw.text);
        println!(
            "{utterance:?}: stop={:?} truncated={} positions={} confidence={confidence:?} latency={elapsed:?}\n  payload={payload:?}\n  outcome={outcome}\n  text={:?}",
            raw.stop, raw.prompt_truncated, raw.positions, raw.text
        );
    }

    /// The real model handles the two hard command cases and reports, without
    /// asserting, what it does with ambient speech.
    ///
    /// Ambient rejection is done by the wake gate, grounding and routing, not
    /// by the model: a different ambient sentence can produce a syntactically
    /// valid call, so an abstention assertion here would pass only by luck.
    ///
    /// Run with:
    /// `cargo test -p nlu --release -- --ignored --nocapture`
    #[test]
    #[ignore = "requires models/needle3.cact; run manually with --ignored --nocapture"]
    fn real_model_handles_commands_and_reports_ambient_speech() -> Result<(), ModelError> {
        let path = model_path();
        assert!(path.is_file(), "missing model file: {}", path.display());

        let started = Instant::now();
        let mut model = NeedleIntentModel::load(&path, NeedleConfig::default())?;
        println!("load: {:?}", started.elapsed());

        let tools = tool_specs();

        let started = Instant::now();
        let scroll = infer_or_report(&mut model, "scroll down", &tools)?;
        println!("scroll down: {scroll:?} in {:?}", started.elapsed());
        println!(
            "scroll down stage timings: {:?}",
            model.last_stage_timings()
        );
        assert!(!scroll.truncated, "the scroll prompt must not be truncated");
        assert_eq!(
            scroll.calls.len(),
            1,
            "scroll down must produce exactly one call"
        );
        let call = scroll
            .calls
            .first()
            .expect("scroll down must produce a tool call");
        assert_ne!(call.name, "no_action", "scroll down must not abstain");
        let intent = call_to_intent(&call.name, call.arguments.clone())
            .expect("scroll down must map to a typed intent");
        assert!(
            matches!(
                &intent,
                Intent::Scroll {
                    direction: Direction::Down,
                    ..
                }
            ),
            "expected scroll/down, got {intent:?}"
        );

        let started = Instant::now();
        let click = infer_or_report(&mut model, "click submit", &tools)?;
        println!("click submit: {click:?} in {:?}", started.elapsed());
        assert!(!click.truncated, "the click prompt must not be truncated");
        assert_eq!(
            click.calls.len(),
            1,
            "click submit must produce exactly one call"
        );
        let call = click
            .calls
            .first()
            .expect("click submit must produce a tool call");
        assert_eq!(call.name, "click_element", "expected click_element");
        let intent = call_to_intent(&call.name, call.arguments.clone())
            .expect("click submit must map to a typed intent");
        match &intent {
            Intent::ClickElement { target } => assert!(
                target.to_lowercase().contains("submit"),
                "expected a submit target, got {target:?}"
            ),
            other => panic!("expected click_element, got {other:?}"),
        }
        assert!(
            intent.needs_confirmation(),
            "a risky click must still need confirmation"
        );

        for utterance in [
            "the volume of sales went up",
            "hello how are you today",
            "tell me a joke",
            "I'll call you in five minutes",
        ] {
            report_ambient(&model, utterance, &tools);
        }

        Ok(())
    }
}
