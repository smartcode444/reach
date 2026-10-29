//! Rule-based command grammar and the Needle intent model for Reach.
//!
//! The wake gate decides whether speech should be handled at all. The rule
//! grammar is the fast path: it turns common spoken commands into a
//! [`reach_core::Intent`] without loading a model. When the grammar has no
//! answer, [`IntentModel`] runs Needle against the closed tool list in
//! [`tools::tool_specs`]. Every model reply is converted back into a typed
//! [`reach_core::Intent`] before the router sees it.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod grammar;
pub mod model;
pub mod needle;
pub mod numbers;
pub mod tools;
pub mod wake;

pub use grammar::parse_rule;
pub use model::{IntentModel, ModelCall, ModelError, ModelReply, ScriptedModel};
pub use needle::{NeedleConfig, NeedleIntentModel};
pub use tools::{ToolParam, ToolParamKind, ToolSpec, call_to_intent, tool_specs, tools_json};
