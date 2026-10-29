//! Shared voice-loop orchestration for Reach.
//!
//! The CLI and the desktop app must run exactly the same wake, routing,
//! confirmation and execution path. This crate owns that path so neither
//! front end can drift from the other, and so the app can observe it through
//! [`StatusEvent`] without parsing human-readable output.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod executor;
pub mod overlay;
pub mod pipeline;
pub mod status;
pub mod worker;

pub use executor::CompositeExecutor;
pub use overlay::{NumberedBadge, OverlayEvent, build_grid_overlay_event};
pub use pipeline::{
    DisabledModel, Pipeline, PipelineIo, PipelineModel, StepOutcome, format_intent_action,
};
pub use screen::Rect;
pub use status::{ErrorKind, StatusEvent};
pub use worker::{WorkerEnd, run_worker};
