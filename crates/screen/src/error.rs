//! Error types for the screen understanding and UI Automation crate.

use std::fmt;

/// Errors that can occur during screen inspection, element collection, or UI Automation actions.
#[derive(Debug)]
pub enum ScreenError {
    /// No active foreground window could be located.
    NoForegroundWindow,

    /// UI Automation returned an error during tree navigation or property inspection.
    Automation(String),

    /// Operation exceeded the specified time limit.
    Timeout,

    /// An element was requested that does not exist or has been destroyed.
    ElementNotFound(String),
}

impl fmt::Display for ScreenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoForegroundWindow => write!(f, "no foreground window found"),
            Self::Automation(msg) => write!(f, "automation error: {msg}"),
            Self::Timeout => write!(f, "operation timed out"),
            Self::ElementNotFound(target) => write!(f, "element not found: {target}"),
        }
    }
}

impl std::error::Error for ScreenError {}
