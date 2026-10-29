//! Numbered snapshot collection and number-to-element resolution.
//!
//! Provides pure functions for extracting, ordering, and numbering clickable
//! on-screen elements, as well as resolving numbers from a snapshot to click
//! coordinates.
//!
//! No operating system calls or logging of element names take place here.

use crate::error::ScreenError;
use crate::label::{ControlKind, ElementSummary, Rect};

/// A single numbered element within a snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumberedElement {
    /// The 1-based index assigned to this element.
    pub number: u32,
    /// Bounding rectangle of the element on screen.
    pub rect: Rect,
    /// Metadata summary for the element.
    pub summary: ElementSummary,
}

/// A snapshot of numbered elements on the screen.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NumberedSnapshot {
    /// The numbered elements contained in this snapshot.
    pub elements: Vec<NumberedElement>,
}

impl NumberedSnapshot {
    /// Creates a new snapshot from a list of numbered elements.
    ///
    /// # Arguments
    /// * `elements` - The numbered elements in the snapshot.
    ///
    /// # Returns
    /// An initialized [`NumberedSnapshot`].
    pub fn new(elements: Vec<NumberedElement>) -> Self {
        Self { elements }
    }

    /// Finds an element in the snapshot by its 1-based number.
    ///
    /// # Arguments
    /// * `number` - The 1-based number to search for.
    ///
    /// # Returns
    /// A reference to the [`NumberedElement`] if found, or `None`.
    pub fn get(&self, number: u32) -> Option<&NumberedElement> {
        self.elements.iter().find(|e| e.number == number)
    }

    /// Returns the count of numbered elements in this snapshot.
    ///
    /// # Returns
    /// The number of elements.
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Returns `true` if this snapshot contains no elements.
    ///
    /// # Returns
    /// `true` if empty.
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }
}

/// Outcome of resolving a number click against a snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NumberClickOutcome {
    /// Click was resolved to absolute screen coordinates.
    NeedsClickAt {
        /// Absolute X coordinate.
        x: i32,
        /// Absolute Y coordinate.
        y: i32,
    },
    /// The requested number was not found, or no snapshot was taken.
    NotFound {
        /// Human-readable explanation.
        message: String,
    },
}

/// Returns whether an element summary represents a clickable, interactive control.
///
/// An element is clickable if:
/// - It is enabled and not off-screen.
/// - Its bounding rectangle has positive width and height.
/// - Its control kind is interactive (e.g. Button, Link, Edit, CheckBox, ComboBox,
///   MenuItem, TabItem, ListItem, RadioButton, or non-empty Other).
///
/// Document controls are structural and excluded.
///
/// # Arguments
/// * `summary` - The element summary to test.
///
/// # Returns
/// `true` if the element is clickable.
pub fn is_clickable(summary: &ElementSummary) -> bool {
    if !summary.enabled || summary.offscreen {
        return false;
    }
    if summary.rect.right <= summary.rect.left || summary.rect.bottom <= summary.rect.top {
        return false;
    }
    match summary.kind {
        ControlKind::Document => false,
        ControlKind::Button
        | ControlKind::Link
        | ControlKind::Edit
        | ControlKind::CheckBox
        | ControlKind::ComboBox
        | ControlKind::MenuItem
        | ControlKind::TabItem
        | ControlKind::ListItem
        | ControlKind::RadioButton => true,
        ControlKind::Other => !summary.name.trim().is_empty(),
    }
}

/// Returns a stable, numbered snapshot of clickable elements from the given summaries.
///
/// Filters elements to only include enabled, on-screen interactive controls with
/// positive bounding box area, and sorts them stably in reading order (top-to-bottom,
/// left-to-right).
///
/// # Arguments
/// * `elements` - Candidate element summaries.
///
/// # Returns
/// A [`NumberedSnapshot`] containing the ordered, 1-based numbered elements.
pub fn numbered_snapshot(elements: &[ElementSummary]) -> NumberedSnapshot {
    let mut clickable: Vec<ElementSummary> = elements
        .iter()
        .filter(|e| is_clickable(e))
        .cloned()
        .collect();

    // Sort stably in natural reading order: top-to-bottom, left-to-right.
    clickable.sort_by(|a, b| {
        a.rect
            .top
            .cmp(&b.rect.top)
            .then_with(|| a.rect.left.cmp(&b.rect.left))
            .then_with(|| a.rect.bottom.cmp(&b.rect.bottom))
            .then_with(|| a.rect.right.cmp(&b.rect.right))
            .then_with(|| a.name.cmp(&b.name))
    });

    let numbered = clickable
        .into_iter()
        .enumerate()
        .map(|(idx, summary)| NumberedElement {
            number: (idx + 1) as u32,
            rect: summary.rect,
            summary,
        })
        .collect();

    NumberedSnapshot::new(numbered)
}

/// Collects clickable elements from the active foreground window and produces a numbered snapshot.
///
/// # Returns
/// A [`NumberedSnapshot`] of on-screen elements.
///
/// # Errors
/// Returns [`ScreenError`] if the foreground window or elements cannot be queried.
pub fn collect_numbered_snapshot() -> Result<NumberedSnapshot, ScreenError> {
    let collect_res = crate::element::collect_foreground(crate::element::CollectLimits::default())?;
    let summaries: Vec<ElementSummary> = collect_res
        .elements
        .into_iter()
        .map(|e| e.summary)
        .collect();
    Ok(numbered_snapshot(&summaries))
}

/// Resolves a 1-based number against an optional snapshot.
///
/// If `snapshot` is `None`, returns [`NumberClickOutcome::NotFound`] instructing the user
/// to say "show numbers". If found in the snapshot, returns [`NumberClickOutcome::NeedsClickAt`]
/// with the geometric centre of the element.
///
/// # Arguments
/// * `snapshot` - The last snapshot taken by `ShowNumbers`, if any.
/// * `number` - The 1-based number to click.
///
/// # Returns
/// A [`NumberClickOutcome`] with click coordinates or a [`NumberClickOutcome::NotFound`]
/// directing the user to say "show numbers".
pub fn click_number(snapshot: Option<&NumberedSnapshot>, number: u32) -> NumberClickOutcome {
    let Some(snap) = snapshot else {
        return NumberClickOutcome::NotFound {
            message: "no snapshot available; please say 'show numbers' first".to_owned(),
        };
    };

    match snap.get(number) {
        Some(elem) => {
            let (x, y) = elem.rect.centre();
            NumberClickOutcome::NeedsClickAt { x, y }
        }
        None => NumberClickOutcome::NotFound {
            message: format!("number {number} not found; say 'show numbers' to refresh"),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::label::{ControlKind, ElementSummary, Rect};

    /// Helper to construct an element summary for testing.
    fn make_summary(name: &str, kind: ControlKind, rect: Rect) -> ElementSummary {
        ElementSummary {
            name: name.to_owned(),
            kind,
            rect,
            enabled: true,
            offscreen: false,
            is_password: false,
        }
    }

    /// Snapshot ordering is stable across two calls on the same input.
    #[test]
    fn snapshot_ordering_is_stable_across_two_calls_on_the_same_input() {
        let e1 = make_summary(
            "A",
            ControlKind::Button,
            Rect {
                left: 10,
                top: 50,
                right: 100,
                bottom: 80,
            },
        );
        let e2 = make_summary(
            "B",
            ControlKind::Link,
            Rect {
                left: 10,
                top: 10,
                right: 100,
                bottom: 40,
            },
        );
        let e3 = make_summary(
            "C",
            ControlKind::Button,
            Rect {
                left: 120,
                top: 10,
                right: 200,
                bottom: 40,
            },
        );

        let input = vec![e1.clone(), e2.clone(), e3.clone()];
        let snap1 = numbered_snapshot(&input);
        let snap2 = numbered_snapshot(&input);

        assert_eq!(snap1, snap2);
        assert_eq!(snap1.len(), 3);
        assert_eq!(snap1.elements[0].summary.name, "B");
        assert_eq!(snap1.elements[0].number, 1);
        assert_eq!(snap1.elements[1].summary.name, "C");
        assert_eq!(snap1.elements[1].number, 2);
        assert_eq!(snap1.elements[2].summary.name, "A");
        assert_eq!(snap1.elements[2].number, 3);
    }

    /// ClickNumber with no snapshot returns NotFound.
    #[test]
    fn click_number_with_no_snapshot_returns_not_found() {
        let outcome = click_number(None, 1);
        match outcome {
            NumberClickOutcome::NotFound { message } => {
                assert!(
                    message.contains("show numbers"),
                    "message must ask user to say 'show numbers': {message}"
                );
            }
            _ => panic!("expected NotFound outcome, got {outcome:?}"),
        }
    }

    /// ClickNumber with snapshot resolves to centre coordinates.
    #[test]
    fn click_number_with_snapshot_resolves_centre() {
        let e = make_summary(
            "Target",
            ControlKind::Button,
            Rect {
                left: 100,
                top: 200,
                right: 200,
                bottom: 300,
            },
        );
        let snap = numbered_snapshot(&[e]);
        let outcome = click_number(Some(&snap), 1);
        assert_eq!(outcome, NumberClickOutcome::NeedsClickAt { x: 150, y: 250 });

        let missing = click_number(Some(&snap), 99);
        match missing {
            NumberClickOutcome::NotFound { message } => {
                assert!(message.contains("99"));
            }
            _ => panic!("expected NotFound for missing number"),
        }
    }

    /// Disabled or off-screen elements are skipped in numbered snapshot.
    #[test]
    fn disabled_or_offscreen_elements_are_filtered() {
        let mut disabled = make_summary(
            "Disabled",
            ControlKind::Button,
            Rect {
                left: 0,
                top: 0,
                right: 50,
                bottom: 50,
            },
        );
        disabled.enabled = false;

        let mut offscreen = make_summary(
            "Offscreen",
            ControlKind::Button,
            Rect {
                left: 0,
                top: 0,
                right: 50,
                bottom: 50,
            },
        );
        offscreen.offscreen = true;

        let empty_rect = make_summary(
            "Empty",
            ControlKind::Button,
            Rect {
                left: 10,
                top: 10,
                right: 10,
                bottom: 10,
            },
        );
        let valid = make_summary(
            "Valid",
            ControlKind::Button,
            Rect {
                left: 0,
                top: 0,
                right: 100,
                bottom: 50,
            },
        );

        let snap = numbered_snapshot(&[disabled, offscreen, empty_rect, valid]);
        assert_eq!(snap.len(), 1);
        assert_eq!(snap.elements[0].summary.name, "Valid");
        assert_eq!(snap.elements[0].number, 1);
    }
}
