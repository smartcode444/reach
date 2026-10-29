//! Status messages emitted by the shared voice loop.
//!
//! The desktop app forwards these values to its status window. The CLI ignores
//! them and keeps writing text, so neither front end needs to parse the other's
//! output.
//!
//! Every variant is deliberately small and free of free-form error text. A
//! transcript may be shown to the user, but executor errors, UI element names
//! and model output must never travel through this channel.

/// One observable step of the voice loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StatusEvent {
    /// The listening window is open and the next utterance is a command.
    Listening,
    /// The loop is waiting for the wake word.
    Asleep,
    /// A transcript was accepted for processing.
    Heard {
        /// The text the user said, after trimming.
        text: String,
    },
    /// An action finished without an executor error.
    Action {
        /// A short, human-readable description of the action.
        label: String,
    },
    /// A risky action is waiting for a spoken yes or no.
    Confirming {
        /// A short, human-readable description of the action.
        label: String,
    },
    /// The utterance could not be routed to a safe action.
    NotUnderstood,
    /// The voice loop has stopped.
    Stopped,
    /// Something failed; only the failure kind is exposed.
    Error {
        /// The category of failure, suitable for a caption.
        kind: ErrorKind,
    },
}

/// The category of a failure reported through [`StatusEvent::Error`].
///
/// The variant carries no error text, path or element name. Callers may log the
/// original error privately, but the status window only receives the kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// The intent model could not load or answer.
    Model,
    /// The executor rejected or failed the action.
    Execution,
    /// Reading input or writing output failed.
    Io,
    /// A startup or internal invariant failed.
    Internal,
}

impl ErrorKind {
    /// Returns the stable lowercase name used by the status window.
    ///
    /// # Returns
    /// One of `"model"`, `"execution"`, `"io"` or `"internal"`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::Execution => "execution",
            Self::Io => "io",
            Self::Internal => "internal",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ErrorKind;

    /// The wire names stay lowercase and stable for the front end.
    #[test]
    fn error_kind_names_are_stable() {
        assert_eq!(ErrorKind::Model.as_str(), "model");
        assert_eq!(ErrorKind::Execution.as_str(), "execution");
        assert_eq!(ErrorKind::Io.as_str(), "io");
        assert_eq!(ErrorKind::Internal.as_str(), "internal");
    }
}
