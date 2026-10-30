//! Screen understanding: UI Automation, screenshots and OCR for Reach.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod element;
pub mod error;
pub mod grid;
pub mod label;
pub mod snapshot;

pub use element::{
    ActionMethod, ActionOutcome, CollectLimits, CollectResult, CollectionRoot, ForegroundFinder,
    LiveElement, SystemForegroundFinder, click_element, click_element_in, collect_foreground,
    collect_foreground_with, decide_collection_root, focus_field, focus_field_in, type_text,
    type_text_in,
};
pub use error::ScreenError;
pub use grid::{
    GridError, GridOutcome, GridState, MAX_GRID_LEVEL, cell_rect, get_foreground_window_rect,
    grid_cells,
};
pub use label::{
    ControlKind, ElementSummary, Match, MatchConfig, Rect, Resolution, normalise_label,
    rank_matches, resolve,
};
pub use snapshot::{
    NumberClickOutcome, NumberedElement, NumberedSnapshot, click_number, is_clickable,
    numbered_snapshot,
};
