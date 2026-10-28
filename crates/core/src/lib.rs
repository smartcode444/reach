//! Shared contract for Reach: the [`Intent`] type every crate speaks.
//!
//! Both the rule parser and the Needle model produce an [`Intent`], and the
//! executor consumes one. Nothing else crosses those boundaries, so no shell
//! command or free-form string can ever reach the operating system.

use serde::{Deserialize, Serialize};

/// A direction for scrolling the active window.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    /// Toward the top of the content.
    Up,
    /// Toward the bottom of the content.
    Down,
    /// Toward the left edge of the content.
    Left,
    /// Toward the right edge of the content.
    Right,
}

/// How to change the zoom level of the active window.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ZoomDir {
    /// Make content larger.
    In,
    /// Make content smaller.
    Out,
    /// Return to the default zoom level.
    Reset,
}

/// A single key the user can ask the app to press.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Key {
    /// The Enter / Return key.
    Enter,
    /// The Tab key.
    Tab,
    /// The Escape key.
    Escape,
    /// The Backspace key.
    Backspace,
    /// The Delete key.
    Delete,
    /// The up arrow key.
    Up,
    /// The down arrow key.
    Down,
    /// The left arrow key.
    Left,
    /// The right arrow key.
    Right,
    /// The Page Up key.
    PageUp,
    /// The Page Down key.
    PageDown,
    /// The Home key.
    Home,
    /// The End key.
    End,
}

/// What the app does with the speech it hears.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    /// Speech is interpreted as commands.
    Command,
    /// Speech is typed out as text.
    Dictation,
    /// Speech is ignored until the wake phrase is heard.
    Sleep,
}

/// Everything Reach is allowed to do, as a closed set of typed actions.
///
/// The JSON form is internally tagged with `action`, for example
/// `{"action":"scroll","direction":"down"}`. A tool name that is not a
/// variant of this enum fails to deserialise and is rejected.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum Intent {
    /// Click the on-screen element whose accessible name best matches `target`.
    ClickElement {
        /// The spoken name of the element, for example "submit".
        target: String,
    },
    /// Click the element that the numbered overlay labelled with `n`.
    ClickNumber {
        /// The number shown on the overlay.
        n: u32,
    },
    /// Type `text`, optionally into the field named `field`.
    TypeText {
        /// The text to type.
        text: String,
        /// The spoken name of the field to focus first, if any.
        field: Option<String>,
    },
    /// Move keyboard focus to the field named `target`.
    FocusField {
        /// The spoken name of the field.
        target: String,
    },
    /// Press a single key.
    PressKey {
        /// The key to press.
        key: Key,
    },
    /// Scroll the active window.
    Scroll {
        /// Which way to scroll.
        direction: Direction,
        /// How many steps to scroll; the executor picks a default when absent.
        amount: Option<u32>,
    },
    /// Change the zoom level of the active window.
    Zoom {
        /// Zoom in, zoom out or reset.
        direction: ZoomDir,
    },
    /// Launch an application by its spoken name.
    OpenApp {
        /// The spoken application name, for example "chrome".
        name: String,
    },
    /// Open a web address in the browser.
    OpenUrl {
        /// The address to open.
        url: String,
    },
    /// Bring another window to the front.
    SwitchWindow {
        /// The spoken window title; `None` means the next window.
        target: Option<String>,
    },
    /// Draw numbers over every clickable element.
    ShowNumbers,
    /// Remove the numbered overlay.
    HideNumbers,
    /// Show the 3x3 mouse grid.
    ShowGrid,
    /// Pick a cell of the mouse grid.
    GridSelect {
        /// The cell number, 1 to 9.
        cell: u32,
    },
    /// Switch the listening mode.
    SetMode {
        /// The mode to switch to.
        mode: Mode,
    },
    /// Confirm the action the app is waiting on.
    Confirm,
    /// Refuse the action the app is waiting on.
    Deny,
    /// Undo the last action where that is possible.
    Undo,
    /// Cancel whatever the app is currently doing.
    Cancel,
    /// Ask the user a question instead of acting.
    Clarify {
        /// The question to show and speak.
        question: String,
    },
}

/// Words in an element name that make a click risky to run unconfirmed.
const RISKY_WORDS: [&str; 10] = [
    "submit", "send", "delete", "pay", "buy", "purchase", "confirm", "remove", "sign out",
    "log out",
];

/// Returns true if `label` contains any word from [`RISKY_WORDS`].
///
/// The comparison is case-insensitive so "Submit Form" and "submit" both match.
fn is_risky_label(label: &str) -> bool {
    let lowered = label.to_lowercase();
    RISKY_WORDS.iter().any(|word| lowered.contains(*word))
}

impl Intent {
    /// Reports whether this intent must be confirmed by the user before it runs.
    ///
    /// Clicking an element whose name suggests an irreversible action (submit,
    /// delete, pay and similar) needs confirmation. Everything else is treated
    /// as safe to run immediately.
    pub fn needs_confirmation(&self) -> bool {
        match self {
            Intent::ClickElement { target } => is_risky_label(target),
            _ => false,
        }
    }
}

/// A finished transcription from the speech-to-text stage.
#[derive(Debug, Clone, PartialEq)]
pub struct Heard {
    /// The recognised text, already trimmed by the caller.
    pub text: String,
    /// Average word confidence from 0.0 to 1.0, if the recogniser reports one.
    pub confidence: Option<f32>,
}

/// Something that can carry out an [`Intent`] on the computer.
pub trait Executor {
    /// Runs `intent` and returns a short message describing what happened.
    ///
    /// # Errors
    /// Returns a human-readable message if the action could not be completed.
    fn run(&mut self, intent: &Intent) -> Result<String, String>;
}

/// Unit tests for the shared contract.
#[cfg(test)]
mod tests {
    use super::*;

    /// A scroll command with no amount parses and leaves the amount empty.
    #[test]
    fn parses_scroll_without_amount() -> Result<(), serde_json::Error> {
        let intent: Intent = serde_json::from_str(r#"{"action":"scroll","direction":"down"}"#)?;
        assert_eq!(
            intent,
            Intent::Scroll {
                direction: Direction::Down,
                amount: None
            }
        );
        Ok(())
    }

    /// Multi-word key names use snake_case in JSON.
    #[test]
    fn parses_page_down_key() -> Result<(), serde_json::Error> {
        let intent: Intent = serde_json::from_str(r#"{"action":"press_key","key":"page_down"}"#)?;
        assert_eq!(intent, Intent::PressKey { key: Key::PageDown });
        Ok(())
    }

    /// A variant with no fields serialises to just its action tag.
    #[test]
    fn serialises_unit_variant() -> Result<(), serde_json::Error> {
        let json = serde_json::to_string(&Intent::ShowNumbers)?;
        assert_eq!(json, r#"{"action":"show_numbers"}"#);
        Ok(())
    }

    /// An action name that is not an `Intent` variant is rejected.
    #[test]
    fn rejects_unknown_action() {
        let result = serde_json::from_str::<Intent>(r#"{"action":"format_disk"}"#);
        assert!(result.is_err());
    }

    /// Risky labels need confirmation; ordinary clicks and scrolling do not.
    #[test]
    fn flags_risky_clicks() {
        let submit = Intent::ClickElement {
            target: "Submit form".to_owned(),
        };
        let help = Intent::ClickElement {
            target: "help".to_owned(),
        };
        let scroll = Intent::Scroll {
            direction: Direction::Down,
            amount: None,
        };
        assert!(submit.needs_confirmation());
        assert!(!help.needs_confirmation());
        assert!(!scroll.needs_confirmation());
    }
}
