//! Turns rule matches and model replies into one safe decision.
//!
//! The router is the only place that decides whether an intent runs, waits for
//! confirmation, asks for a clarification, or is rejected. It never repairs a
//! model reply: a call must name a declared tool, deserialize into an
//! [`Intent`], and be grounded in the user's words before it can run.

use reach_core::Intent;

use crate::grammar::parse_rule;
use crate::grounding::is_intent_grounded;
use crate::model::{IntentModel, ModelError, ModelReply};
use crate::tools::{NO_ACTION, call_to_intent, tool_specs};

/// The question the feedback layer should speak for [`Route::Clarify`].
///
/// The wording matches the empty-reply behaviour in `AGENTS.md`. The router
/// returns the decision without text so the feedback layer stays free to add
/// captions or translate the question later.
pub const CLARIFY_QUESTION: &str = "I didn't catch that, could you say it another way?";

/// Why the router refused a model reply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectReason {
    /// The prompt was cut to fit the context, so the reply answered only part
    /// of the utterance.
    PromptTruncated,
    /// The model returned an error instead of a reply.
    ModelFailed,
    /// The model named a tool that is not in [`crate::tools::tool_specs`].
    UnknownTool,
    /// The call's name or arguments did not deserialize into an [`Intent`].
    InvalidIntent,
    /// The utterance did not contain a trigger for the named tool, or did not
    /// contain the string or number the model put in its arguments.
    Ungrounded,
}

/// What the app should do with one utterance.
#[derive(Debug, Clone, PartialEq)]
pub enum Route {
    /// Run the intent now.
    Run(Intent),
    /// Ask the user to confirm the intent before running it.
    Confirm(Intent),
    /// Ask the user to repeat or rephrase; nothing runs.
    Clarify,
    /// Reject the utterance; nothing runs.
    Reject(RejectReason),
}

/// Routes an intent that [`parse_rule`] already matched.
///
/// Rule matches are trusted to be grounded because the grammar matched the
/// user's words directly, so grounding is skipped. They still go through
/// [`Intent::needs_confirmation`] so a risky click cannot run unconfirmed.
///
/// # Arguments
/// * `intent` - The typed intent produced by the rule grammar.
///
/// # Returns
/// [`Route::Clarify`] when the grammar itself asked for clarification,
/// [`Route::Confirm`] when the intent is risky, otherwise [`Route::Run`].
pub fn route_rule_intent(intent: Intent) -> Route {
    if matches!(intent, Intent::Clarify { .. }) {
        return Route::Clarify;
    }
    if intent.needs_confirmation() {
        Route::Confirm(intent)
    } else {
        Route::Run(intent)
    }
}

/// Routes one model reply for `utterance`.
///
/// The checks run in this order, because each one makes the later checks
/// meaningless or unsafe:
///
/// 1. A model error is rejected; an undeclared tool keeps its own reason.
/// 2. A truncated prompt is rejected; the model answered a partial utterance.
/// 3. An empty call list asks for clarification; it is not permission to guess.
/// 4. More than one call asks for clarification; one utterance must mean one
///    action.
/// 5. `no_action` asks for clarification before conversion, because it is not
///    an [`Intent`] variant.
/// 6. A name outside [`crate::tools::tool_specs`] is rejected before
///    [`call_to_intent`], so rule-only intents such as `confirm` and `open_url`
///    can never be smuggled in by the model.
/// 7. A call that does not deserialize is rejected, not repaired.
/// 8. A call whose tool trigger or arguments the utterance does not contain is
///    rejected.
/// 9. Only then does [`Intent::needs_confirmation`] decide between
///    [`Route::Confirm`] and [`Route::Run`].
///
/// The reply's confidence is deliberately not used as a threshold here. The
/// measured confidences for correct commands (0.350 for "scroll down", 0.071
/// for "click submit") do not separate correct calls from hallucinations, so a
/// guessed threshold would either block valid commands or let hallucinations
/// through. Thresholds must come from the utterance corpus before they are
/// added.
///
/// # Arguments
/// * `utterance` - The recognised text that produced the reply.
/// * `result` - The model's reply, or the error that replaced it.
///
/// # Returns
/// The decision the app should carry out. A rejected reply never carries an
/// intent, so nothing can run by accident.
pub fn route_model_reply(utterance: &str, result: Result<ModelReply, ModelError>) -> Route {
    let reply = match result {
        Ok(reply) => reply,
        Err(ModelError::UndeclaredTool(_)) => {
            return Route::Reject(RejectReason::UnknownTool);
        }
        Err(_) => return Route::Reject(RejectReason::ModelFailed),
    };

    if reply.truncated {
        return Route::Reject(RejectReason::PromptTruncated);
    }
    if reply.calls.is_empty() {
        return Route::Clarify;
    }
    if reply.calls.len() > 1 {
        return Route::Clarify;
    }
    let Some(call) = reply.calls.first() else {
        return Route::Clarify;
    };

    if call.name == NO_ACTION {
        return Route::Clarify;
    }

    let tools = tool_specs();
    if !tools.iter().any(|tool| tool.name == call.name.as_str()) {
        return Route::Reject(RejectReason::UnknownTool);
    }

    let intent = match call_to_intent(&call.name, call.arguments.clone()) {
        Ok(intent) => intent,
        Err(_) => return Route::Reject(RejectReason::InvalidIntent),
    };

    if !is_intent_grounded(utterance, &call.name, &intent) {
        return Route::Reject(RejectReason::Ungrounded);
    }

    // TODO(routing): confidence thresholds are not applied yet. The utterance
    // corpus must measure the low/medium/high boundaries first; guessing them
    // would either block valid commands or let hallucinations through.
    if intent.needs_confirmation() {
        Route::Confirm(intent)
    } else {
        Route::Run(intent)
    }
}

/// Routes one utterance, trying the rule grammar before the model.
///
/// # Arguments
/// * `utterance` - The recognised text, already trimmed by the caller.
/// * `model` - The model used only when the rule grammar has no match.
///
/// # Returns
/// The same decisions as [`route_rule_intent`] and [`route_model_reply`].
pub fn route_utterance(utterance: &str, model: &mut impl IntentModel) -> Route {
    if let Some(intent) = parse_rule(utterance) {
        return route_rule_intent(intent);
    }

    let tools = tool_specs();
    route_model_reply(utterance, model.infer(utterance, &tools))
}

/// Unit tests for rule and model routing.
#[cfg(test)]
mod tests {
    use reach_core::Direction;
    use serde_json::{Value, json};

    use super::*;
    use crate::model::{ModelCall, ScriptedModel, calls_from_payload};

    /// Builds a single-call reply with no truncation.
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

    /// Builds an empty reply, which means the model abstained.
    fn empty_reply() -> ModelReply {
        ModelReply {
            calls: Vec::new(),
            confidence: None,
            truncated: false,
        }
    }

    /// A truncated prompt is rejected before the empty or call checks.
    #[test]
    fn truncated_reply_is_rejected_first() {
        let mut call_reply = reply("scroll", json!({ "direction": "down" }));
        call_reply.truncated = true;
        assert_eq!(
            route_model_reply("scroll down", Ok(call_reply)),
            Route::Reject(RejectReason::PromptTruncated)
        );

        let truncated_empty = ModelReply {
            calls: Vec::new(),
            confidence: None,
            truncated: true,
        };
        assert_eq!(
            route_model_reply("scroll down", Ok(truncated_empty)),
            Route::Reject(RejectReason::PromptTruncated)
        );
    }

    /// An empty call list is an abstention, not a command.
    #[test]
    fn empty_reply_asks_for_clarification() {
        assert_eq!(
            route_model_reply("scroll down", Ok(empty_reply())),
            Route::Clarify
        );
    }

    /// `no_action` is checked before conversion, because it is not an `Intent`.
    #[test]
    fn no_action_asks_for_clarification_before_conversion() {
        assert!(call_to_intent(NO_ACTION, json!({})).is_err());
        assert_eq!(
            route_model_reply("hello there", Ok(reply(NO_ACTION, json!({})))),
            Route::Clarify
        );
    }

    /// More than one call is ambiguous, so nothing runs.
    #[test]
    fn multiple_calls_ask_for_clarification() {
        let reply = ModelReply {
            calls: vec![
                ModelCall {
                    name: "scroll".to_owned(),
                    arguments: json!({ "direction": "down" }),
                },
                ModelCall {
                    name: "show_numbers".to_owned(),
                    arguments: json!({}),
                },
            ],
            confidence: Some(0.5),
            truncated: false,
        };
        assert_eq!(
            route_model_reply("scroll down and show numbers", Ok(reply)),
            Route::Clarify
        );
    }

    /// A scripted `confirm` call is rejected before `call_to_intent` can turn
    /// it into the rule-only `Intent::Confirm`.
    #[test]
    fn scripted_confirm_call_is_rejected_as_unknown_tool() {
        let scripted = reply("confirm", json!({}));
        assert_eq!(
            call_to_intent("confirm", json!({})).expect("confirm is a valid Intent variant"),
            Intent::Confirm
        );
        assert_eq!(
            route_model_reply("click submit", Ok(scripted.clone())),
            Route::Reject(RejectReason::UnknownTool)
        );

        let mut model = ScriptedModel::new(vec![scripted]);
        assert_eq!(
            route_utterance("do the thing", &mut model),
            Route::Reject(RejectReason::UnknownTool)
        );
    }

    /// Rule-only intents can never be smuggled in through the model.
    #[test]
    fn rule_only_intents_are_rejected_as_unknown_tools() {
        for name in [
            "open_url", "set_mode", "confirm", "deny", "undo", "cancel", "clarify",
        ] {
            assert_eq!(
                route_model_reply("click submit", Ok(reply(name, json!({})))),
                Route::Reject(RejectReason::UnknownTool),
                "{name} must not be a model tool"
            );
        }
    }

    /// An undeclared-tool error from the model boundary keeps the unknown-tool
    /// reason instead of becoming a generic model failure.
    #[test]
    fn undeclared_tool_error_keeps_unknown_tool_reason() {
        assert_eq!(
            route_model_reply(
                "click submit",
                Err(ModelError::UndeclaredTool("confirm".to_owned()))
            ),
            Route::Reject(RejectReason::UnknownTool)
        );
    }

    /// The captured malformed payload is a model failure, not an intent.
    #[test]
    fn malformed_hide_numbers_payload_is_rejected() {
        let payload = r#"[{"name":"hide_numbers","arguments":{"#;
        let error = calls_from_payload(payload).expect_err("malformed JSON must not parse");
        assert!(matches!(&error, ModelError::InvalidJson(_)));
        assert_eq!(
            route_model_reply("the volume of sales went up", Err(error)),
            Route::Reject(RejectReason::ModelFailed)
        );
    }

    /// The valid-looking hallucination is rejected because the utterance does
    /// not contain a hide-numbers trigger.
    #[test]
    fn valid_looking_hallucination_is_rejected_as_ungrounded() {
        let hallucination = reply("hide_numbers", json!({}));
        for utterance in [
            "the volume of sales went up",
            "hello how are you today",
            "I'll call you in five minutes",
        ] {
            assert_eq!(
                route_model_reply(utterance, Ok(hallucination.clone())),
                Route::Reject(RejectReason::Ungrounded),
                "{utterance:?} must not ground hide_numbers"
            );
        }

        assert_eq!(
            route_model_reply(
                "tell me a joke",
                Ok(reply("open_app", json!({ "name": "joke" })))
            ),
            Route::Reject(RejectReason::Ungrounded)
        );
    }

    /// A declared tool with bad arguments is rejected, not repaired.
    #[test]
    fn invalid_arguments_are_rejected() {
        assert_eq!(
            route_model_reply(
                "scroll down",
                Ok(reply("scroll", json!({ "direction": "sideways" })))
            ),
            Route::Reject(RejectReason::InvalidIntent)
        );
    }

    /// A numeric argument must be spoken, not inferred from context.
    #[test]
    fn numeric_arguments_are_checked_against_the_transcript() {
        assert_eq!(
            route_model_reply(
                "click the big one",
                Ok(reply("click_number", json!({ "n": 14 })))
            ),
            Route::Reject(RejectReason::Ungrounded)
        );
        assert_eq!(
            route_model_reply(
                "click fourteen",
                Ok(reply("click_number", json!({ "n": 14 })))
            ),
            Route::Run(Intent::ClickNumber { n: 14 })
        );
        assert_eq!(
            route_model_reply(
                "scroll down",
                Ok(reply(
                    "scroll",
                    json!({ "direction": "down", "amount": 40 })
                ))
            ),
            Route::Reject(RejectReason::Ungrounded)
        );
    }

    /// A string argument must appear in the transcript.
    #[test]
    fn string_arguments_are_checked_against_the_transcript() {
        assert_eq!(
            route_model_reply(
                "click the thing",
                Ok(reply("click_element", json!({ "target": "submit" })))
            ),
            Route::Reject(RejectReason::Ungrounded)
        );
        assert_eq!(
            route_model_reply(
                "click submit",
                Ok(reply("click_element", json!({ "target": "submit" })))
            ),
            Route::Confirm(Intent::ClickElement {
                target: "submit".to_owned(),
            })
        );
    }

    /// A grounded, safe model call runs.
    #[test]
    fn grounded_safe_model_call_runs() {
        assert_eq!(
            route_model_reply(
                "scroll down",
                Ok(reply("scroll", json!({ "direction": "down" })))
            ),
            Route::Run(Intent::Scroll {
                direction: Direction::Down,
                amount: None,
            })
        );
    }

    /// A grounded risky model call waits for confirmation.
    #[test]
    fn grounded_risky_model_call_asks_for_confirmation() {
        assert_eq!(
            route_model_reply(
                "click submit",
                Ok(reply("click_element", json!({ "target": "submit" })))
            ),
            Route::Confirm(Intent::ClickElement {
                target: "submit".to_owned(),
            })
        );
    }

    /// A grounded, safe click runs without confirmation.
    #[test]
    fn grounded_safe_click_runs() {
        assert_eq!(
            route_model_reply(
                "click help",
                Ok(reply("click_element", json!({ "target": "help" })))
            ),
            Route::Run(Intent::ClickElement {
                target: "help".to_owned(),
            })
        );
    }

    /// Rule matches skip grounding but still go through `needs_confirmation`.
    #[test]
    fn rule_path_skips_grounding_but_checks_confirmation() {
        let url = Intent::OpenUrl {
            url: "google.com".to_owned(),
        };
        assert_eq!(route_rule_intent(url.clone()), Route::Run(url));

        let submit = Intent::ClickElement {
            target: "submit".to_owned(),
        };
        assert_eq!(route_rule_intent(submit.clone()), Route::Confirm(submit));

        let help = Intent::ClickElement {
            target: "help".to_owned(),
        };
        assert_eq!(route_rule_intent(help.clone()), Route::Run(help));
    }

    /// The rule path does not call the model at all.
    #[test]
    fn rule_path_never_calls_the_model() {
        let mut exhausted = ScriptedModel::default();
        assert_eq!(
            route_utterance("open google dot com", &mut exhausted),
            Route::Run(Intent::OpenUrl {
                url: "google.com".to_owned(),
            })
        );

        let mut exhausted = ScriptedModel::default();
        assert_eq!(
            route_utterance("click submit", &mut exhausted),
            Route::Confirm(Intent::ClickElement {
                target: "submit".to_owned(),
            })
        );
    }

    /// A rule-produced clarification is routed as a clarification, not an action.
    #[test]
    fn rule_clarify_intent_is_a_clarification() {
        assert_eq!(
            route_rule_intent(Intent::Clarify {
                question: "Which one?".to_owned(),
            }),
            Route::Clarify
        );
    }
}
