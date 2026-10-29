//! The model boundary: typed replies, errors and the [`IntentModel`] trait.
//!
//! Everything outside this crate reaches Needle through [`IntentModel`] and
//! receives a [`ModelReply`]. A reply is still untrusted: the caller must
//! validate its calls and convert them with
//! [`crate::tools::call_to_intent`] before anything can run.

use std::collections::VecDeque;

use serde_json::{Map, Value};
use thiserror::Error;

use crate::tools::ToolSpec;

/// One function call the model asked for.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelCall {
    /// The tool name exactly as the model emitted it.
    pub name: String,
    /// The arguments object; a missing value is treated as an empty object.
    pub arguments: Value,
}

/// A complete model reply.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelReply {
    /// The calls in the order the model produced them.
    pub calls: Vec<ModelCall>,
    /// The model's own confidence, when the container exports a head.
    pub confidence: Option<f32>,
    /// True when the prompt was cut to fit the context.
    ///
    /// Such a reply is unusable: the model answered a partial prompt. Callers
    /// must discard it and ask the user to try again.
    pub truncated: bool,
}

/// Everything that can go wrong at the model boundary.
#[derive(Debug, Error)]
pub enum ModelError {
    /// The model container could not be loaded.
    #[error("the Needle model could not be loaded: {0}")]
    Load(#[from] needle_infer::v3_engine::V3EngineError),
    /// The tool-call payload was not valid JSON.
    #[error("the model tool-call payload is not valid JSON: {0}")]
    InvalidJson(#[from] serde_json::Error),
    /// The payload was valid JSON but not an array of calls.
    #[error("the model tool-call payload is not an array of calls")]
    InvalidPayloadShape,
    /// A call had no string `name`.
    #[error("a model call is missing a string `name`")]
    MissingCallName,
    /// The model named a tool that was not declared for this request.
    #[error("the model called an undeclared tool: {0}")]
    UndeclaredTool(String),
    /// The model hit the token or context limit before finishing its reply.
    #[error("the model ran out of budget before finishing its reply ({0:?})")]
    Incomplete(needle_infer::v3_engine::StopReason),
    /// No tools were supplied, so constrained generation would be off.
    #[error("the model was called without any declared tools")]
    NoTools,
    /// The scripted test model ran out of canned replies.
    #[error("the scripted model has no replies left")]
    ScriptExhausted,
}

/// Turns an utterance and the tool list into the model's reply.
pub trait IntentModel {
    /// Runs the model on `utterance` with the declared `tools`.
    ///
    /// # Arguments
    /// * `utterance` - The recognised text, already trimmed by the caller.
    /// * `tools` - The complete tool list the model may call.
    ///
    /// # Returns
    /// The function calls the model chose plus a confidence score. An empty
    /// call list means the model abstained; it is not permission to guess.
    ///
    /// # Errors
    /// Returns an error if the model cannot load, the payload is malformed, or
    /// the model names a tool outside `tools`.
    fn infer(&mut self, utterance: &str, tools: &[ToolSpec]) -> Result<ModelReply, ModelError>;
}

/// A model that returns canned replies or errors, used by tests.
///
/// The scripted model keeps `cargo test` from loading the 35 MB model file and
/// lets the router be tested against empty, malformed and risky replies.
#[derive(Debug, Default)]
pub struct ScriptedModel {
    /// Outcomes handed out in order; an error simulates a rejected completion.
    steps: VecDeque<Result<ModelReply, ModelError>>,
}

impl ScriptedModel {
    /// Creates a model that returns `replies` in order.
    ///
    /// # Arguments
    /// * `replies` - The canned replies; the first call takes the first one.
    pub fn new(replies: Vec<ModelReply>) -> Self {
        Self::from_results(replies.into_iter().map(Ok).collect())
    }

    /// Creates a model that returns replies and errors in order.
    ///
    /// Use this to script a malformed payload, an incomplete generation or an
    /// undeclared tool, so the caller's rejection path is tested without the
    /// real model.
    ///
    /// # Arguments
    /// * `results` - The canned outcomes; the first call takes the first one.
    pub fn from_results(results: Vec<Result<ModelReply, ModelError>>) -> Self {
        Self {
            steps: results.into(),
        }
    }
}

impl IntentModel for ScriptedModel {
    /// Returns the next canned reply or error.
    ///
    /// # Errors
    /// Returns the scripted error, or [`ModelError::ScriptExhausted`] once every
    /// outcome has been used, so a test that calls the model more often than
    /// expected fails loudly.
    fn infer(&mut self, _utterance: &str, _tools: &[ToolSpec]) -> Result<ModelReply, ModelError> {
        match self.steps.pop_front() {
            Some(result) => result,
            None => Err(ModelError::ScriptExhausted),
        }
    }
}

/// Parses the JSON array Needle puts between its tool-call markers.
///
/// # Arguments
/// * `payload` - The string returned by `extract_tool_call`, for example
///   `[{"name":"scroll","arguments":{"direction":"down"}}]`.
///
/// # Returns
/// The calls in order. A call without `arguments` gets an empty object.
///
/// # Errors
/// Returns [`ModelError::InvalidJson`] when the payload is not JSON,
/// [`ModelError::InvalidPayloadShape`] when it is not an array, and
/// [`ModelError::MissingCallName`] when a call has no string name.
pub(crate) fn calls_from_payload(payload: &str) -> Result<Vec<ModelCall>, ModelError> {
    let value: Value = serde_json::from_str(payload)?;
    let array = value.as_array().ok_or(ModelError::InvalidPayloadShape)?;
    let mut calls = Vec::with_capacity(array.len());

    for item in array {
        let name = item
            .get("name")
            .and_then(Value::as_str)
            .ok_or(ModelError::MissingCallName)?
            .to_owned();
        let arguments = item
            .get("arguments")
            .cloned()
            .unwrap_or_else(|| Value::Object(Map::new()));
        calls.push(ModelCall { name, arguments });
    }

    Ok(calls)
}

/// Checks that every call names a tool declared for this request.
///
/// Constrained decoding should make this impossible, but it is the second line
/// of defence: if the grammar is ever bypassed, an undeclared name is rejected
/// here instead of reaching the router.
///
/// # Arguments
/// * `calls` - The calls parsed from the model output.
/// * `tools` - The exact tool list supplied to the model.
///
/// # Errors
/// Returns [`ModelError::UndeclaredTool`] for the first unknown name.
pub(crate) fn validate_calls(calls: &[ModelCall], tools: &[ToolSpec]) -> Result<(), ModelError> {
    for call in calls {
        if !tools.iter().any(|tool| tool.name == call.name.as_str()) {
            return Err(ModelError::UndeclaredTool(call.name.clone()));
        }
    }
    Ok(())
}

/// Unit tests for the model boundary.
#[cfg(test)]
mod tests {
    use reach_core::Intent;
    use serde_json::json;

    use super::*;
    use crate::tools::{call_to_intent, tool_specs};

    /// Returns a simple reply for scripted-model tests.
    fn sample_reply() -> ModelReply {
        ModelReply {
            calls: vec![ModelCall {
                name: "scroll".to_owned(),
                arguments: json!({ "direction": "down" }),
            }],
            confidence: Some(0.9),
            truncated: false,
        }
    }

    /// The scripted model returns its replies in order.
    #[test]
    fn scripted_model_returns_replies_in_order() {
        let first = sample_reply();
        let second = ModelReply {
            calls: Vec::new(),
            confidence: None,
            truncated: false,
        };
        let mut model = ScriptedModel::new(vec![first.clone(), second.clone()]);

        assert_eq!(model.infer("scroll down", &tool_specs()).unwrap(), first);
        assert_eq!(model.infer("thank you", &tool_specs()).unwrap(), second);
    }

    /// Running out of canned replies is an explicit error, not an empty reply.
    #[test]
    fn scripted_model_reports_when_exhausted() {
        let mut model = ScriptedModel::default();
        assert!(matches!(
            model.infer("scroll down", &tool_specs()),
            Err(ModelError::ScriptExhausted)
        ));
    }

    /// The exact malformed payload the real model produced for ambient speech.
    ///
    /// Generation stopped at EOS in the middle of the JSON string, so the
    /// boundary must reject it. `Incomplete` is the other allowed rejection when
    /// generation hits the token cap; the ignored integration test prints the
    /// stop reason for each real completion.
    #[test]
    fn malformed_hide_numbers_payload_is_rejected() {
        // Copied verbatim from the real completion:
        // `<tool_call>[{"name":"hide_numbers","arguments":{"`
        let payload = r#"[{"name":"hide_numbers","arguments":{"#;
        let error = calls_from_payload(payload).expect_err("malformed JSON must not parse");
        assert!(matches!(&error, ModelError::InvalidJson(_)));

        let mut model = ScriptedModel::from_results(vec![Err(error)]);
        assert!(matches!(
            model.infer("the volume of sales went up", &tool_specs()),
            Err(ModelError::InvalidJson(_))
        ));
    }

    /// A well-formed payload becomes calls and missing arguments default to `{}`.
    #[test]
    fn parses_calls_and_defaults_arguments() {
        let calls = calls_from_payload(
            r#"[{"name":"scroll","arguments":{"direction":"down"}},{"name":"no_action"}]"#,
        )
        .unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "scroll");
        assert_eq!(calls[0].arguments, json!({ "direction": "down" }));
        assert_eq!(calls[1].name, "no_action");
        assert_eq!(calls[1].arguments, json!({}));
    }

    /// A single object is not the array Needle promises.
    #[test]
    fn rejects_a_non_array_payload() {
        assert!(matches!(
            calls_from_payload(r#"{"name":"scroll"}"#),
            Err(ModelError::InvalidPayloadShape)
        ));
    }

    /// A call without a string name is rejected.
    #[test]
    fn rejects_a_call_without_a_name() {
        assert!(matches!(
            calls_from_payload(r#"[{"arguments":{}}]"#),
            Err(ModelError::MissingCallName)
        ));
    }

    /// An undeclared tool name is rejected even if the JSON is valid.
    #[test]
    fn rejects_an_undeclared_tool() {
        let tools = vec![ToolSpec {
            name: "scroll",
            description: "Scroll the active window.",
            params: &[],
        }];
        let calls = vec![ModelCall {
            name: "open_url".to_owned(),
            arguments: json!({ "url": "https://example.com" }),
        }];
        assert!(matches!(
            validate_calls(&calls, &tools),
            Err(ModelError::UndeclaredTool(name)) if name == "open_url"
        ));
    }

    /// Declared tool names pass validation.
    #[test]
    fn accepts_declared_tools() {
        let tools = vec![ToolSpec {
            name: "scroll",
            description: "Scroll the active window.",
            params: &[],
        }];
        let calls = vec![ModelCall {
            name: "scroll".to_owned(),
            arguments: json!({ "direction": "down" }),
        }];
        assert!(validate_calls(&calls, &tools).is_ok());
    }

    /// A syntactically valid hallucination still parses to an Intent.
    ///
    /// This is the current boundary behaviour, not a safety claim: `hide_numbers`
    /// is a declared tool, so `call_to_intent` accepts it even though the
    /// utterance is ambient speech. The grounding and routing task must reject
    /// it; the model layer must not add a confidence threshold or a lexical gate
    /// to hide this.
    #[test]
    fn valid_looking_hallucination_becomes_an_intent() {
        // Exact payload from the real model for "hello how are you today" and
        // "I'll call you in five minutes" in the final integration run.
        let payload = r#"[{"name":"hide_numbers","arguments":{}}]"#;
        let calls = calls_from_payload(payload).expect("the fixture is valid JSON");
        let reply = ModelReply {
            calls: calls.clone(),
            confidence: Some(0.027_276_015),
            truncated: false,
        };
        let mut model = ScriptedModel::new(vec![reply]);
        let reply = model
            .infer("the volume of sales went up", &tool_specs())
            .expect("the scripted reply is available");

        assert_eq!(reply.calls.len(), 1);
        assert!(validate_calls(&reply.calls, &tool_specs()).is_ok());
        let intent = call_to_intent(&reply.calls[0].name, reply.calls[0].arguments.clone())
            .expect("hide_numbers is a declared tool");
        assert_eq!(intent, Intent::HideNumbers);
    }

    /// A truncated reply carries the flag the caller must act on.
    #[test]
    fn truncated_replies_are_marked() {
        let reply = ModelReply {
            calls: Vec::new(),
            confidence: None,
            truncated: true,
        };
        assert!(reply.truncated);
    }
}
