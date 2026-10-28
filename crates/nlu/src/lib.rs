//! Rule-based command grammar and the Needle intent model for Reach.
//!
//! The wake gate decides whether speech should be handled at all. The rule
//! grammar is the fast path: it turns common spoken commands into a
//! [`reach_core::Intent`] without loading a model. The model fallback lives
//! behind the `IntentModel` boundary and is added in a later step.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod grammar;
pub mod numbers;
pub mod wake;

pub use grammar::parse_rule;
